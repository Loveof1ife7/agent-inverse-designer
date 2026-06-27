#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.AgentExplorer import AgentExplorer
from src.DatagenFEMEvaluator import DatagenFEMEvaluator
from src.InverseDesigner import InverseDesigner
from src.KnowledgeBase import KnowledgeBase
from src.Scheduler import StructureDiscoverySystem


DEFAULT_RAW_ROOT = ROOT / "train_datas" / "raw" / "P222_paired_dataset_0_99999_20260620"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _count_rows(path: Path) -> tuple[int, int]:
    counts = {"node_data": 0, "element_conn": 0}
    section = ""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("node_data"):
                section = "node_data"
                continue
            if line.startswith("element_conn"):
                section = "element_conn"
                continue
            if section and line.startswith("]"):
                section = ""
                continue
            if section and line.startswith("["):
                counts[section] += 1
    return counts["node_data"], counts["element_conn"]


def _load_geometry(path: Path) -> tuple[list[list[float]], list[list[int]]]:
    namespace: dict[str, Any] = {}
    exec(path.read_text(encoding="utf-8", errors="ignore"), {}, namespace)
    nodes_raw = namespace.get("node_data") or []
    elems_raw = namespace.get("element_conn") or []
    coordinates = [[float(row[1]), float(row[2]), float(row[3])] for row in nodes_raw]
    edges = []
    for row in elems_raw:
        if len(row) >= 3:
            edges.append([int(row[-2]), int(row[-1])])
        else:
            edges.append([int(row[0]), int(row[1])])
    return coordinates, edges


def _curve_summary(path: Path) -> dict[str, float]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        stress_key = "Stress" if "Stress" in (reader.fieldnames or []) else "Stress_MPa"
        for row in reader:
            try:
                rows.append((float(row["Strain"]), max(float(row[stress_key]), 0.0)))
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        return {"peak_stress": 0.0, "energy_0_30": 0.0}
    rows.sort(key=lambda item: item[0])
    peak = max(stress for _strain, stress in rows)
    energy = 0.0
    clipped = [(min(max(strain, 0.0), 0.30), stress) for strain, stress in rows if strain <= 0.30]
    for lhs, rhs in zip(clipped, clipped[1:]):
        energy += 0.5 * (lhs[1] + rhs[1]) * abs(rhs[0] - lhs[0])
    return {"peak_stress": peak, "energy_0_30": energy}


def select_simple_samples(raw_root: Path, total: int, scan_limit: int) -> list[dict[str, Any]]:
    geometry_dir = raw_root / "geometry"
    properties_dir = raw_root / "properties"
    files = sorted(
        geometry_dir.glob("*.txt"),
        key=lambda path: (path.stat().st_size, int(path.stem) if path.stem.isdigit() else 10**18),
    )[:scan_limit]
    samples = []
    for geometry_path in files:
        sample_id = geometry_path.stem
        property_path = properties_dir / f"{sample_id}.csv"
        if not property_path.exists():
            continue
        nodes, edges = _count_rows(geometry_path)
        if nodes <= 0 or edges <= 0:
            continue
        samples.append(
            {
                "sample_id": sample_id,
                "nodes": nodes,
                "edges": edges,
                "score": nodes + edges,
                "geometry_path": str(geometry_path.resolve()),
                "property_path": str(property_path.resolve()),
                "geometry_bytes": geometry_path.stat().st_size,
            }
        )
        if len(samples) >= total:
            break
    if len(samples) < total:
        raise RuntimeError(f"only selected {len(samples)} simple samples, need {total}")
    return samples


def materialize_batches(samples: list[dict[str, Any]], workspace: Path, batch_count: int, batch_size: int) -> list[list[dict[str, Any]]]:
    batches_root = workspace / "batches"
    batches_root.mkdir(parents=True, exist_ok=True)
    batches = []
    for batch_index in range(batch_count):
        batch = samples[batch_index * batch_size : (batch_index + 1) * batch_size]
        batch_dir = batches_root / f"batch_{batch_index + 1:02d}"
        (batch_dir / "geometry").mkdir(parents=True, exist_ok=True)
        (batch_dir / "properties").mkdir(parents=True, exist_ok=True)
        for item in batch:
            for key, subdir, ext in (("geometry_path", "geometry", ".txt"), ("property_path", "properties", ".csv")):
                src = Path(item[key])
                dst = batch_dir / subdir / f"{item['sample_id']}{ext}"
                if dst.exists():
                    continue
                try:
                    os.link(src, dst)
                    link_type = "hardlink"
                except OSError:
                    shutil.copy2(src, dst)
                    link_type = "copy"
                item[f"{subdir}_batch_path"] = str(dst)
                item["materialization"] = link_type
        batches.append(batch)
    return batches


def record_to_structure(item: dict[str, Any]) -> dict[str, Any]:
    coordinates, edges = _load_geometry(Path(item["geometry_path"]))
    curve = _curve_summary(Path(item["property_path"]))
    return {
        "structure_id": f"preprocessed:{item['sample_id']}",
        "sample_id": item["sample_id"],
        "coordinates": coordinates,
        "edges": edges,
        "node_count": len(coordinates),
        "edge_count": len(edges),
        "symmetry": "P222",
        "source": "active_learning_pool",
        "provenance": {
            "structure_path": item["geometry_path"],
            "property_path": item["property_path"],
            "curve_summary": curve,
        },
    }


def evaluate_pool_item(
    evaluator: DatagenFEMEvaluator,
    item: dict[str, Any],
    target_property: dict[str, float],
) -> dict[str, Any]:
    structure = record_to_structure(item)
    evaluation = evaluator.evaluate_explicit_structure(structure, target_property)
    return {
        "sample_id": item["sample_id"],
        "structure_id": f"preprocessed:{item['sample_id']}",
        "property": dict(evaluation.get("evaluated_property") or {}),
        "evaluated_property": dict(evaluation.get("evaluated_property") or {}),
        "property_error": dict(evaluation.get("property_error") or {}),
        "label": str(evaluation.get("label") or "failure"),
        "explicit_structure": structure,
        "validity": {
            "geometry_status": evaluation.get("geometry_status", "unknown"),
            "fem_status": evaluation.get("fem_status", "unknown"),
        },
        "fidelity": str((evaluation.get("raw_metrics") or {}).get("evaluator", evaluator.fem_backend)),
        "source": "active_learning_acquired_batch",
        "raw_metrics": dict(evaluation.get("raw_metrics") or {}),
    }


def acquire_batch(
    evaluator: DatagenFEMEvaluator,
    batch: list[dict[str, Any]],
    target_property: dict[str, float],
    output_path: Path,
) -> list[dict[str, Any]]:
    rows = [evaluate_pool_item(evaluator, item, target_property) for item in batch]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def _best_sample(result_dict: dict[str, Any]) -> dict[str, Any]:
    samples = list(result_dict.get("discovered_samples") or [])
    if not samples:
        return {}
    return min(
        samples,
        key=lambda sample: sum(_safe_float(value) for value in dict(sample.get("property_error") or {}).values()),
    )


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Active Learning Batch Demo",
        "",
        f"- fem_backend: `{summary['fem_backend']}`",
        f"- target_property: `{summary['target_property']}`",
        f"- target_source: `batch {summary['target_source']['batch_index']}, sample {summary['target_source']['sample_id']}`",
        f"- batch_count: `{summary['batch_count']}`",
        f"- batch_size: `{summary['batch_size']}`",
        f"- final_success: `{summary['final_success']}`",
        f"- final_round: `{summary['final_round']}`",
        "",
        "## Rounds",
        "",
    ]
    for item in summary["rounds"]:
        lines.extend(
            [
                f"### Round {item['round']}",
                "",
                f"- retrieval_space_size: `{item['retrieval_space_size']}`",
                f"- acquired_batches: `{item['acquired_batches']}`",
                f"- success: `{item['success']}`",
                f"- best_label: `{item.get('best_sample', {}).get('label', '')}`",
                f"- best_structure: `{item.get('best_sample', {}).get('structure_id', '')}`",
                f"- best_property: `{item.get('best_sample', {}).get('evaluated_property', {})}`",
                f"- best_error: `{item.get('best_sample', {}).get('property_error', {})}`",
                "",
            ]
        )
        if item.get("added_next_batch"):
            lines.append(f"- added_next_batch: `{item['added_next_batch']}`")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate batched active learning over simple P222 geometries.")
    parser.add_argument("--workspace", default="workspace/active_learning_batch_demo")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--batch-count", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--scan-limit", type=int, default=1000)
    parser.add_argument("--target-batch", type=int, default=7)
    parser.add_argument("--target-index", type=int, default=3)
    parser.add_argument("--target-property", default="")
    parser.add_argument("--max-iterations-per-round", type=int, default=1)
    parser.add_argument("--agent-batch-size", type=int, default=2)
    parser.add_argument("--experiment-budget", type=int, default=1)
    parser.add_argument("--fem-backend", choices=["proxy", "auto", "abaqus"], default="proxy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("AGENT_EXPLORER_ENABLE_LLM", "0")
    os.environ.setdefault("KNOWLEDGE_INTERPRETER_ENABLE_LLM", "0")
    AgentExplorer._global_llm_disable_reason = "active_learning_demo_uses_deterministic_agent"

    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = ROOT / workspace
    if args.fresh and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    total = args.batch_count * args.batch_size
    raw_root = Path(args.raw_root)
    if not raw_root.is_absolute():
        raw_root = ROOT / raw_root
    samples = select_simple_samples(raw_root, total=total, scan_limit=args.scan_limit)
    batches = materialize_batches(samples, workspace, args.batch_count, args.batch_size)
    batch_manifest = {
        "raw_root": str(raw_root),
        "batch_count": args.batch_count,
        "batch_size": args.batch_size,
        "selection_rule": "smallest geometry files with matching properties CSV",
        "batches": [
            {"batch_index": index, "samples": batch}
            for index, batch in enumerate(batches, start=1)
        ],
    }
    (workspace / "active_learning_batches.json").write_text(
        json.dumps(batch_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    kb = KnowledgeBase(workspace / "knowledge.sqlite")
    evaluator = DatagenFEMEvaluator(workspace_root=workspace, fem_backend=args.fem_backend)
    inverse = InverseDesigner(kb)
    try:
        target_batch = min(max(args.target_batch, 1), args.batch_count)
        target_index = min(max(args.target_index, 1), args.batch_size)
        target_item = batches[target_batch - 1][target_index - 1]
        if args.target_property:
            target_property = json.loads(args.target_property)
        else:
            target_eval = evaluate_pool_item(evaluator, target_item, {"stiffness_proxy": 1.0, "density_proxy": 1.0})
            target_property = {
                key: float(value)
                for key, value in target_eval["evaluated_property"].items()
                if key in {"stiffness_proxy", "density_proxy"}
            }

        acquired_rows = acquire_batch(
            evaluator,
            batches[0],
            target_property,
            workspace / "acquired_batches" / "batch_01_training_rows.json",
        )
        inverse.train(acquired_rows)
        retrieval_space_size = len(inverse.training_examples)
        rounds = []
        final_success = False
        final_round = 0

        for round_index in range(1, args.batch_count + 1):
            system = StructureDiscoverySystem(
                knowledge_base=kb,
                inverse_designer=inverse,
                agent_explorer=AgentExplorer(enable_llm=False),
                evaluator=evaluator,
                retrain_trigger=10**9,
                workspace_root=workspace / "closed_loop_rounds" / f"round_{round_index:02d}",
                log_path=workspace / "closed_loop_events.jsonl",
                agent_batch_size=args.agent_batch_size,
                experiment_budget=args.experiment_budget,
            )
            result = system.run(target_property=target_property, max_iterations=args.max_iterations_per_round)
            result_dict = result.to_dict()
            best = _best_sample(result_dict)
            round_summary = {
                "round": round_index,
                "retrieval_space_size": retrieval_space_size,
                "acquired_batches": list(range(1, round_index + 1)),
                "success": result.success,
                "iterations": result.iterations,
                "best_sample": best,
                "closed_loop_result": result_dict,
            }
            rounds.append(round_summary)
            print(
                f"[active-learning] round={round_index} "
                f"space={retrieval_space_size} success={result.success} "
                f"best={best.get('structure_id', '')} label={best.get('label', '')}"
            )
            if result.success:
                final_success = True
                final_round = round_index
                break
            if round_index < args.batch_count:
                next_batch_index = round_index + 1
                next_rows = acquire_batch(
                    evaluator,
                    batches[next_batch_index - 1],
                    target_property,
                    workspace / "acquired_batches" / f"batch_{next_batch_index:02d}_training_rows.json",
                )
                inverse.finetune(next_rows)
                retrieval_space_size = len(inverse.training_examples)
                round_summary["added_next_batch"] = next_batch_index

        if not final_success:
            final_round = len(rounds)

        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "workspace": str(workspace),
            "fem_backend": args.fem_backend,
            "target_property": target_property,
            "target_source": {
                "batch_index": target_batch,
                "target_index": target_index,
                "sample_id": target_item["sample_id"],
            },
            "batch_count": args.batch_count,
            "batch_size": args.batch_size,
            "final_success": final_success,
            "final_round": final_round,
            "rounds": rounds,
            "batch_manifest": str(workspace / "active_learning_batches.json"),
            "kb_path": str(workspace / "knowledge.sqlite"),
        }
        summary_json = workspace / "active_learning_summary.json"
        summary_md = workspace / "active_learning_summary.md"
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_summary(summary_md, summary)
        print(f"[active-learning] summary_json={summary_json}")
        print(f"[active-learning] summary_md={summary_md}")
        return 0
    finally:
        kb.close()


if __name__ == "__main__":
    raise SystemExit(main())
