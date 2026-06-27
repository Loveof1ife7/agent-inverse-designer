#!/usr/bin/env python
from __future__ import annotations

import argparse
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


DEFAULT_PREPROCESSED_JSONL = (
    ROOT
    / "train_datas"
    / "preprocessed"
    / "P222_paired_dataset_0_99999_20260620"
    / "inverse_truss_property_grid_v1_0_99.jsonl"
)
DEFAULT_HARD_TARGET = {
    "stiffness_proxy": 0.26,
    "density_proxy": 0.0188,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _property_error(target: dict[str, float], observed: dict[str, float]) -> dict[str, float]:
    errors = {}
    for key, target_value in target.items():
        observed_value = _safe_float(observed.get(key), 0.0)
        target_value = _safe_float(target_value, 0.0)
        scale = abs(target_value) if abs(target_value) > 1e-9 else 1.0
        errors[key] = abs(observed_value - target_value) / scale
    return errors


def _max_error(target: dict[str, float], observed: dict[str, float]) -> float:
    errors = _property_error(target, observed)
    return max(errors.values()) if errors else float("inf")


def _curve_summary(record: dict[str, Any]) -> dict[str, float]:
    y_values = [_safe_float(value) for value in record.get("y") or []]
    property_payload = dict(record.get("property") or {})
    summary = dict(property_payload.get("summary") or {})
    peak_stress = _safe_float(summary.get("peak_stress"), max(y_values) if y_values else 0.0)
    energy = _safe_float(summary.get("energy_0_30"), 0.0)
    if not energy and len(y_values) > 1:
        # The grid spacing is 0.01 in stress_grid_v1.
        energy = sum((y_values[index] + y_values[index - 1]) * 0.005 for index in range(1, len(y_values)))
    return {
        "peak_stress": peak_stress,
        "energy_0_30": energy,
    }


def load_preprocessed_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit > 0 and len(records) >= limit:
                break
    if not records:
        raise RuntimeError(f"no records loaded from {path}")
    return records


def build_inverse_training_rows(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries = [_curve_summary(record) for record in records]
    peak_scale = max((item["peak_stress"] for item in summaries), default=1.0) or 1.0
    energy_scale = max((item["energy_0_30"] for item in summaries), default=1.0) or 1.0
    rows = []
    for record, summary in zip(records, summaries):
        sample_id = str(record.get("sample_id") or len(rows))
        property_proxy = {
            "stiffness_proxy": summary["peak_stress"] / peak_scale,
            "density_proxy": summary["energy_0_30"] / energy_scale,
        }
        provenance = dict(record.get("provenance") or {})
        explicit_structure = {
            "structure_id": f"preprocessed:{sample_id}",
            "sample_id": sample_id,
            "coordinates": record.get("coordinates") or [],
            "edges": record.get("edges") or [],
            "node_count": int(record.get("n") or len(record.get("coordinates") or [])),
            "edge_count": len(record.get("edges") or []),
            "parent_sequence": record.get("parent_sequence") or [],
            "extra_edges": record.get("extra_edges") or [],
            "symmetry": provenance.get("symmetry") or "P222",
            "source": "preprocessed_inverse_dataset",
            "provenance": provenance,
        }
        rows.append(
            {
                "sample_id": sample_id,
                "structure_id": f"preprocessed:{sample_id}",
                "property": property_proxy,
                "curve_property": {
                    **summary,
                    "representation": dict(record.get("property") or {}).get("representation", "stress_grid_v1"),
                },
                "explicit_structure": explicit_structure,
                "validity": {
                    "geometry_status": "valid",
                    "fem_status": "success",
                },
                "fidelity": "preprocessed_curve_proxy",
                "source": "preprocessed_cold_start",
            }
        )
    return rows, {
        "property_bridge": "stress_grid_v1 summary -> normalized retrieval proxy",
        "peak_stress_scale": peak_scale,
        "energy_0_30_scale": energy_scale,
        "rows": len(rows),
    }


def best_sample_payload(result_dict: dict[str, Any]) -> dict[str, Any]:
    samples = list(result_dict.get("discovered_samples") or [])
    if not samples:
        return {}
    successes = [sample for sample in samples if sample.get("label") == "success"]
    if successes:
        return min(
            successes,
            key=lambda sample: sum(_safe_float(value) for value in dict(sample.get("property_error") or {}).values()),
        )
    return min(
        samples,
        key=lambda sample: sum(_safe_float(value) for value in dict(sample.get("property_error") or {}).values()),
    )


def probe_inverse_designer(
    inverse_designer: InverseDesigner,
    evaluator: DatagenFEMEvaluator,
    target_property: dict[str, float],
) -> dict[str, Any]:
    structure = inverse_designer.sample_structure(target_property)
    if structure is None:
        return {"candidate_found": False}
    evaluation = evaluator.evaluate_explicit_structure(structure, target_property)
    retrieved_property = dict(structure.get("retrieved_property") or {})
    return {
        "candidate_found": True,
        "candidate_id": structure.get("structure_id", ""),
        "retrieved_property": retrieved_property,
        "retrieval_distance": structure.get("retrieval_distance"),
        "explicit_structure_size": {
            "node_count": len(structure.get("coordinates") or structure.get("nodes") or []),
            "edge_count": len(structure.get("edges") or []),
        },
        "candidate_eval": evaluation,
        "max_relative_error": _max_error(target_property, evaluation.get("evaluated_property") or {}),
        "works": evaluation.get("label") == "success",
    }


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    result = summary["closed_loop_result"]
    inverse = summary["inverse_designer_probe"]
    best = summary.get("best_sample") or {}
    inverse_label = inverse.get("candidate_eval", {}).get("label", "none")
    final_label = best.get("label", "none")
    finetune_count = len(summary.get("finetune_events") or [])
    lines = [
        "# Closed-Loop Discovery Demo",
        "",
        "## Visual Overview",
        "",
        "```mermaid",
        "flowchart TD",
        f"    A[\"Cold-start dataset<br/>{summary['cold_start']['training_rows']} preprocessed structures\"] --> B[\"InverseDesigner<br/>retrieval-only\"]",
        f"    B --> C[\"Hard target<br/>{_format_property(summary['target_property'])}\"]",
        f"    C --> D[\"Retrieve explicit structure<br/>label: {inverse_label}\"]",
        "    D --> E{\"Target satisfied?\"}",
        "    E -- \"yes\" --> Z[\"Return retrieved structure\"]",
        "    E -- \"no\" --> F[\"AgentExplorer target schedule\"]",
        f"    F --> G[\"InverseDesigner samples scheduled targets<br/>{result['iterations']} iterations\"]",
        "    G --> J[\"Explicit structure FEM evaluation\"]",
        f"    J --> H[\"Dataset grows + finetune<br/>{finetune_count} updates\"]",
        f"    H --> I[\"Best sample<br/>label: {final_label}\"]",
        "```",
        "",
        "## Setup",
        "",
        f"- workspace: `{summary['workspace']}`",
        f"- kb_path: `{summary['kb_path']}`",
        f"- preprocessed_jsonl: `{summary['cold_start']['preprocessed_jsonl']}`",
        f"- training_rows: `{summary['cold_start']['training_rows']}`",
        f"- property_bridge: `{summary['cold_start']['property_bridge']['property_bridge']}`",
        f"- fem_backend: `{summary['fem_backend']}`",
        "",
        "## Hard Target",
        "",
        f"- target_property: `{summary['target_property']}`",
        f"- target_mode: `{summary['target_mode']}`",
        "",
        "## InverseDesigner Probe",
        "",
        f"- candidate_found: `{inverse['candidate_found']}`",
        f"- candidate_id: `{inverse.get('candidate_id', '')}`",
        f"- candidate_label_for_target: `{inverse.get('candidate_eval', {}).get('label', '')}`",
        f"- evaluated_property: `{inverse.get('candidate_eval', {}).get('evaluated_property', {})}`",
        f"- property_error: `{inverse.get('candidate_eval', {}).get('property_error', {})}`",
        "",
        "## Closed Loop",
        "",
        f"- success: `{result['success']}`",
        f"- task_status: `{result['task_status']}`",
        f"- iterations: `{result['iterations']}`",
        f"- discovered_samples: `{len(result['discovered_samples'])}`",
        f"- finetune_events: `{finetune_count}`",
        f"- experiment_root: `{result.get('experiment_paths', {}).get('root_dir', '')}`",
    ]
    if best:
        lines.extend(
            [
                "",
                "## Best Sample",
                "",
                f"- structure_id: `{best.get('structure_id', '')}`",
                f"- label: `{best.get('label', '')}`",
                f"- evaluated_property: `{best.get('evaluated_property', {})}`",
                f"- property_error: `{best.get('property_error', {})}`",
                f"- structure_path: `{best.get('structure_path', '')}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_property(payload: dict[str, Any]) -> str:
    return ", ".join(f"{key}={_safe_float(value):.4f}" for key, value in payload.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: cold-start from preprocessed truss-property pairs, retrieve with InverseDesigner, then run closed-loop discovery."
    )
    parser.add_argument("--workspace", default="workspace/closed_loop_discovery_demo")
    parser.add_argument("--fresh", action="store_true", help="delete the demo workspace before running")
    parser.add_argument("--preprocessed-jsonl", default=str(DEFAULT_PREPROCESSED_JSONL))
    parser.add_argument("--limit", type=int, default=100, help="number of preprocessed rows to load; <=0 means all")
    parser.add_argument("--target-property", default="", help="optional JSON override")
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--experiment-budget", type=int, default=8)
    parser.add_argument("--agent-batch-size", type=int, default=77)
    parser.add_argument("--retrain-trigger", type=int, default=4)
    parser.add_argument("--fem-backend", default="proxy", choices=["proxy", "auto", "abaqus"])
    parser.add_argument("--allow-failure", action="store_true", help="return 0 even if no success is found")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("AGENT_EXPLORER_ENABLE_LLM", "0")
    os.environ.setdefault("KNOWLEDGE_INTERPRETER_ENABLE_LLM", "0")
    AgentExplorer._global_llm_disable_reason = "demo_uses_deterministic_agent"

    workspace = Path(args.workspace)
    if not workspace.is_absolute():
        workspace = ROOT / workspace
    if args.fresh and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    kb_path = workspace / "knowledge.sqlite"
    preprocessed_jsonl = Path(args.preprocessed_jsonl)
    if not preprocessed_jsonl.is_absolute():
        preprocessed_jsonl = ROOT / preprocessed_jsonl

    print(f"[demo] workspace={workspace}")
    print(f"[demo] kb_path={kb_path}")
    print(f"[demo] preprocessed_jsonl={preprocessed_jsonl}")

    records = load_preprocessed_records(preprocessed_jsonl, args.limit)
    training_rows, bridge_summary = build_inverse_training_rows(records)
    training_path = workspace / "inverse_retrieval_training_dataset.json"
    training_path.write_text(json.dumps(training_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[demo] cold_start_training_rows={len(training_rows)}")

    target_property = json.loads(args.target_property) if args.target_property else dict(DEFAULT_HARD_TARGET)
    target_mode = "user_override" if args.target_property else "built_in_hard_low_density_target"
    print(f"[demo] target_property={target_property}")

    kb = KnowledgeBase(kb_path)
    try:
        evaluator = DatagenFEMEvaluator(workspace_root=workspace, fem_backend=args.fem_backend)
        inverse_designer = InverseDesigner(kb)
        inverse_designer.train(training_rows)
        inverse_probe = probe_inverse_designer(inverse_designer, evaluator, target_property)
        print(f"[demo] inverse_probe_label={inverse_probe.get('candidate_eval', {}).get('label', 'none')}")
        print(f"[demo] inverse_probe_property={inverse_probe.get('candidate_eval', {}).get('evaluated_property', {})}")

        system = StructureDiscoverySystem(
            knowledge_base=kb,
            inverse_designer=inverse_designer,
            agent_explorer=AgentExplorer(enable_llm=False),
            evaluator=evaluator,
            retrain_trigger=args.retrain_trigger,
            workspace_root=workspace,
            log_path=workspace / "closed_loop_events.jsonl",
            agent_batch_size=args.agent_batch_size,
            experiment_budget=args.experiment_budget,
        )
        result = system.run(target_property=target_property, max_iterations=args.max_iterations)
        result_dict = result.to_dict()
        finetune_events = [
            event.to_dict()
            for event in result.events
            if event.stage == "inverse_designer_finetune"
        ]
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "workspace": str(workspace),
            "kb_path": str(kb_path),
            "fem_backend": args.fem_backend,
            "target_mode": target_mode,
            "target_property": target_property,
            "cold_start": {
                "preprocessed_jsonl": str(preprocessed_jsonl),
                "training_dataset_path": str(training_path),
                "training_rows": len(training_rows),
                "property_bridge": bridge_summary,
                "inverse_training_examples": len(inverse_designer.training_examples),
            },
            "inverse_designer_probe": inverse_probe,
            "closed_loop_result": result_dict,
            "finetune_events": finetune_events,
            "best_sample": best_sample_payload(result_dict),
            "final_kb_statistics": kb.dataset_statistics(),
        }
        summary_json = workspace / "demo_summary.json"
        summary_md = workspace / "demo_summary.md"
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown_summary(summary_md, summary)

        print(f"[demo] success={result.success} iterations={result.iterations}")
        print(f"[demo] finetune_events={len(finetune_events)}")
        print(f"[demo] summary_json={summary_json}")
        print(f"[demo] summary_md={summary_md}")
        if not result.success and not args.allow_failure:
            return 2
        return 0
    finally:
        kb.close()


if __name__ == "__main__":
    raise SystemExit(main())
