from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.DatagenFEMEvaluator import DatagenFEMEvaluator
from src.DatasetManager import DatasetManager
from src.ForwardSurrogate import ForwardSurrogate
from src.HighPrecisionFEM import HighPrecisionFEM
from src.InverseDesigner import RemoteGraphMetaMatInverseDesigner
from src.Scheduler import DeterministicLoopConfig, DeterministicSurrogateClosedLoopSystem
from src.TargetCurvePlanner import TargetCurvePlanner
from src.curve_targets import normalize_target_property, stress_curve_error_metrics


DEFAULT_BASE_DIR = ROOT / "workspace" / "high_plateau_deterministic_surrogate_smoke_20260630_164159"
DEFAULT_TARGET_JSON = DEFAULT_BASE_DIR / "deterministic_surrogate_result.json"
DEFAULT_PANELS = DEFAULT_BASE_DIR / "workflow_panels.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal high-plateau remote C3D4 closed loop.")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--target-json", default=str(DEFAULT_TARGET_JSON))
    parser.add_argument("--workflow-panels", default=str(DEFAULT_PANELS))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--target-batch-size", type=int, default=10)
    parser.add_argument("--samples-per-target", type=int, default=1)
    parser.add_argument("--surrogate-top-k", type=int, default=6)
    parser.add_argument("--sim-batch-size", type=int, default=10)
    parser.add_argument("--min-forward-update-rows", type=int, default=10)
    parser.add_argument("--min-inverse-update-rows", type=int, default=20)
    parser.add_argument("--acceptance-curve-nmae", type=float, default=0.05)
    parser.add_argument("--surrogate-inverse-training-curve-nmae", type=float, default=0.05)
    parser.add_argument("--forward-batch-size", type=int, default=256)
    parser.add_argument("--forward-gpu-workers", type=int, default=5)
    parser.add_argument("--forward-compact", action="store_true")
    parser.add_argument("--remote-nodes", default="cnode1,cnode2")
    parser.add_argument("--remote-max-parallel", type=int, default=10)
    parser.add_argument("--remote-cpus-per-job", type=int, default=9)
    parser.add_argument("--remote-array", type=int, default=1)
    parser.add_argument("--remote-k-min", type=float, default=0.8)
    parser.add_argument("--remote-k-max", type=float, default=1.2)
    parser.add_argument("--remote-young", type=float, default=7.0)
    parser.add_argument("--remote-timeout-seconds", type=int, default=21600)
    parser.add_argument("--ssh-key", default=r"C:\Users\qbli1\.ssh\210.45.73.118_0702090547_rsa.txt")
    parser.add_argument("--dry-run-build", action="store_true", help="Build system and write manifest only; do not run iterations.")
    return parser.parse_args()


def configure_remote_env(args: argparse.Namespace) -> None:
    defaults = {
        "GID_C3D4_REMOTE_NODES": args.remote_nodes,
        "GID_C3D4_MAX_PARALLEL": str(args.remote_max_parallel),
        "GID_C3D4_CPUS_PER_JOB": str(args.remote_cpus_per_job),
        "GID_C3D4_ARRAY": str(args.remote_array),
        "GID_C3D4_K_MIN": str(args.remote_k_min),
        "GID_C3D4_K_MAX": str(args.remote_k_max),
        "GID_C3D4_YOUNG": str(args.remote_young),
        "GID_C3D4_REMOTE_TIMEOUT": str(args.remote_timeout_seconds),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    if args.ssh_key:
        os.environ.setdefault("GID_C3D4_REMOTE_SSH_KEY", args.ssh_key)


def load_target(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload.get("target_property"), dict):
        return normalize_target_property(payload["target_property"])
    results = payload.get("results")
    if isinstance(results, list) and results and isinstance(results[0].get("target_property"), dict):
        return normalize_target_property(results[0]["target_property"])
    plan = payload.get("target_plan")
    if isinstance(plan, dict) and isinstance(plan.get("final_target"), dict):
        return normalize_target_property(plan["final_target"])
    raise ValueError(f"Cannot find target curve in {path}")


def make_run_paths(base_dir: Path, run_name: str) -> dict[str, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name.strip() or f"formal_remote_{stamp}"
    run_root = base_dir / name
    log_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_root": run_root,
        "log_dir": log_dir,
        "run_log": log_dir / "run.log",
        "events": log_dir / "events.jsonl",
        "summary": log_dir / "live_summary.json",
        "result": run_root / "deterministic_surrogate_result.json",
    }


def configure_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("high_plateau_remote_closed_loop")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def build_config(args: argparse.Namespace) -> DeterministicLoopConfig:
    return DeterministicLoopConfig(
        target_batch_size=args.target_batch_size,
        samples_per_target=args.samples_per_target,
        max_iterations=args.max_iterations,
        finetune_policy="threshold",
        sim_batch_size=args.sim_batch_size,
        surrogate_top_k=args.surrogate_top_k,
        min_forward_update_rows=args.min_forward_update_rows,
        min_inverse_update_rows=args.min_inverse_update_rows,
        acceptance_curve_nmae=args.acceptance_curve_nmae,
        surrogate_inverse_training_curve_nmae=args.surrogate_inverse_training_curve_nmae,
        forward_batch_size=args.forward_batch_size,
        forward_gpu_workers=args.forward_gpu_workers,
        forward_compact=args.forward_compact,
        notes={
            "formal_run": True,
            "cpu_fem": "gid_c3d4_remote",
            "cpu_parallelism": {
                "sim_batch_size": args.sim_batch_size,
                "remote_max_parallel": args.remote_max_parallel,
                "remote_cpus_per_job": args.remote_cpus_per_job,
            },
        },
    )


def build_system(args: argparse.Namespace, paths: dict[str, Path], config: DeterministicLoopConfig) -> DeterministicSurrogateClosedLoopSystem:
    workspace_root = paths["run_root"]
    inverse = RemoteGraphMetaMatInverseDesigner.from_env(workspace_root=workspace_root)
    inverse.num_runs = config.inverse_num_runs
    inverse.top_k = config.inverse_top_k
    inverse.batch_size = config.inverse_batch_size
    forward = ForwardSurrogate(
        workspace_root=workspace_root,
        evaluator=DatagenFEMEvaluator(workspace_root=workspace_root, fem_backend="remote_forward"),
        batch_size=config.forward_batch_size,
        gpu_workers=config.forward_gpu_workers,
        compact=config.forward_compact,
    )
    high_precision = HighPrecisionFEM(
        workspace_root=workspace_root,
        backend="gid_c3d4_remote",
        align_remote_graphmetamat_to_p222=False,
    )
    return DeterministicSurrogateClosedLoopSystem(
        inverse_designer=inverse,
        target_planner=TargetCurvePlanner(),
        forward_surrogate=forward,
        high_precision_fem=high_precision,
        dataset_manager=DatasetManager(workspace_root / "deterministic_datasets"),
        config=config,
        workspace_root=workspace_root,
        task_id="formal_high_plateau_remote",
        log_path=paths["events"],
    )


def summarize_iteration(result: dict[str, Any]) -> dict[str, Any]:
    def best(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {}
        return min(items, key=lambda item: _safe_float(item.get("curve_nmae"), float("inf")))

    update = dict(result.get("dataset_update") or {})
    return {
        "iteration": result.get("iteration"),
        "surrogate_pairs": len(result.get("surrogate_pairs") or []),
        "surrogate_training_pairs": len(result.get("surrogate_training_pairs") or []),
        "simulation_pairs": len(result.get("simulation_pairs") or []),
        "backlog_size": result.get("simulation_backlog_size"),
        "best_surrogate_nmae": best(result.get("surrogate_acceptance") or []).get("curve_nmae"),
        "best_simulation_nmae": best(result.get("simulation_acceptance") or []).get("curve_nmae"),
        "accepted": bool(result.get("accepted")),
        "inverse_training_rows": update.get("inverse_training_rows"),
        "inverse_training_weight": update.get("inverse_training_weight"),
        "forward_training_rows": update.get("forward_training_rows"),
        "updated_inverse_designer": update.get("updated_inverse_designer"),
        "updated_forward_surrogate": update.get("updated_forward_surrogate"),
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_workflow_panels(path: Path, target: dict[str, Any], results: list[dict[str, Any]], config: DeterministicLoopConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = normalize_target_property(target)
    strain = list(target.get("strain_grid") or [])
    stress = list(target.get("stress") or [])
    summaries = [summarize_iteration(result) for result in results]
    if not strain or not stress:
        return

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax0.plot(strain, stress, color="black", linewidth=2.3, label="target")
    curve_items = _selected_best_curves(target, results)
    colors = ["#b4442f", "#d9922b", "#287c71", "#3157a4", "#7a5195"]
    for color, item in zip(colors, curve_items):
        curve = item["curve"]
        err = stress_curve_error_metrics(target, curve).get("curve_nmae")
        label = f"iter {item['iteration']} {item['source']} NMAE={_fmt(err)}"
        ax0.plot(curve.get("strain_grid") or [], curve.get("stress") or [], color=color, linewidth=1.7, label=label)
    ax0.set_title("A. best curves")
    ax0.set_xlabel("strain")
    ax0.set_ylabel("stress")
    ax0.grid(True, alpha=0.22)
    ax0.legend(loc="best")

    iterations = [int(item["iteration"]) for item in summaries]
    if iterations:
        ax1.plot(iterations, [_nan(item["best_surrogate_nmae"]) for item in summaries], marker="o", color="#8b4bb3", label="best surrogate")
        ax1.plot(iterations, [_nan(item["best_simulation_nmae"]) for item in summaries], marker="s", color="#287c71", label="best C3D4")
        ax1.axhline(config.acceptance_curve_nmae, color="black", linestyle="--", linewidth=1.1, label="accept threshold")
        for item in summaries:
            if item["updated_inverse_designer"]:
                ax1.axvline(int(item["iteration"]), color="#d9922b", alpha=0.42, linewidth=1.5)
            if item["updated_forward_surrogate"]:
                ax1.axvline(int(item["iteration"]), color="#3157a4", alpha=0.55, linewidth=2.0)
        ax1.set_xticks(iterations)
    ax1.set_title("B. best candidate error")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("curve NMAE")
    ax1.grid(True, alpha=0.22)
    ax1.legend(loc="upper right")

    if iterations:
        ax2.bar([x - 0.18 for x in iterations], [_nan(item["inverse_training_weight"]) for item in summaries], width=0.34, color="#d9922b", label="pending inverse weight")
        ax2.bar([x + 0.18 for x in iterations], [_nan(item["forward_training_rows"]) for item in summaries], width=0.34, color="#3157a4", label="pending forward rows")
        ax2.axhline(config.min_inverse_update_rows, color="#d9922b", linestyle="--", linewidth=1.1, label=f"inverse threshold={config.min_inverse_update_rows}")
        ax2.axhline(config.min_forward_update_rows, color="#3157a4", linestyle=":", linewidth=1.4, label=f"forward threshold={config.min_forward_update_rows}")
        ax2.set_xticks(iterations)
    ax2.set_title("C. threshold-triggered GPU updates")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("new rows / weighted rows")
    ax2.grid(True, axis="y", alpha=0.22)
    ax2.legend(loc="upper left")

    if iterations:
        ax3.plot(iterations, [item["surrogate_pairs"] for item in summaries], marker="o", color="#8b4bb3", label="surrogate predicted")
        ax3.plot(iterations, [item["surrogate_training_pairs"] for item in summaries], marker="^", color="#d9922b", label="surrogate train-gated")
        ax3.plot(iterations, [item["simulation_pairs"] for item in summaries], marker="s", color="#287c71", label="C3D4 simulation")
        ax3.plot(iterations, [item["backlog_size"] for item in summaries], marker=".", color="#6f6f6f", label="backlog remaining")
        ax3.axhline(config.sim_batch_size, color="black", linestyle="--", linewidth=1.0, label=f"sim batch={config.sim_batch_size}")
        ax3.set_xticks(iterations)
    ax3.set_title("D. throughput and backlog")
    ax3.set_xlabel("iteration")
    ax3.set_ylabel("count")
    ax3.grid(True, alpha=0.22)
    ax3.legend(loc="best")

    fig.suptitle("High plateau formal closed loop: remote C3D4 tracking", fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        (
            f"sim_batch_size={config.sim_batch_size} | surrogate_top_k={config.surrogate_top_k} | "
            f"min_forward={config.min_forward_update_rows} | min_inverse={config.min_inverse_update_rows} | "
            f"surrogate_inverse_gate={config.surrogate_inverse_training_curve_nmae}"
        ),
        ha="center",
        fontsize=9,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _selected_best_curves(target: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []
    indices = sorted({0, len(results) // 2, len(results) - 1})
    items: list[dict[str, Any]] = []
    for index in indices:
        result = results[index]
        source, curve = _best_curve(target, result.get("simulation_pairs") or [])
        if curve is None:
            source, curve = _best_curve(target, result.get("surrogate_pairs") or [])
        if curve is not None:
            items.append({"iteration": result.get("iteration"), "source": source, "curve": curve})
    return items


def _best_curve(target: dict[str, Any], pairs: list[dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
    best_pair = None
    best_error = float("inf")
    for pair in pairs:
        curve = dict(pair.get("stress_curve") or {})
        error = _safe_float(stress_curve_error_metrics(target, curve).get("curve_nmae"), float("inf"))
        if error < best_error:
            best_pair = pair
            best_error = error
    if best_pair is None:
        return "", None
    return str(best_pair.get("label_source") or "curve"), dict(best_pair.get("stress_curve") or {})


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _nan(value: Any) -> float:
    return _safe_float(value, float("nan"))


def _fmt(value: Any) -> str:
    number = _safe_float(value, float("nan"))
    return "nan" if math.isnan(number) else f"{number:.3f}"


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    target_path = Path(args.target_json).resolve()
    panels_path = Path(args.workflow_panels).resolve()
    configure_remote_env(args)
    paths = make_run_paths(base_dir, args.run_name)
    logger = configure_logger(paths["run_log"])
    config = build_config(args)
    target = load_target(target_path)
    system = build_system(args, paths, config)
    summary: dict[str, Any] = {
        "status": "built",
        "target_json": str(target_path),
        "workflow_panels": str(panels_path),
        "run_paths": {key: str(value) for key, value in paths.items()},
        "config": config.to_dict(),
        "remote_env": {
            key: os.environ.get(key, "")
            for key in (
                "GID_C3D4_REMOTE_NODES",
                "GID_C3D4_MAX_PARALLEL",
                "GID_C3D4_CPUS_PER_JOB",
                "GID_C3D4_ARRAY",
                "GID_C3D4_K_MIN",
                "GID_C3D4_K_MAX",
                "GID_C3D4_YOUNG",
                "GID_C3D4_REMOTE_TIMEOUT",
                "GID_C3D4_REMOTE_SSH_KEY",
            )
        },
        "iterations": [],
        "dataset_manager": system.dataset_manager.to_dict(),
        "experiment_paths": system.experiment_paths.to_dict(),
    }
    write_summary(paths["summary"], summary)
    logger.info("Run root: %s", paths["run_root"])
    logger.info("Target: %s", target_path)
    logger.info("Workflow panels: %s", panels_path)
    logger.info("Config: %s", json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True))
    logger.info("Remote C3D4: max_parallel=%s cpus_per_job=%s nodes=%s", args.remote_max_parallel, args.remote_cpus_per_job, args.remote_nodes)

    if args.dry_run_build:
        logger.info("Dry-run build complete; no iterations executed.")
        return 0

    results: list[dict[str, Any]] = []
    final_result: dict[str, Any] = {}
    try:
        for iteration in range(1, config.max_iterations + 1):
            logger.info("Iteration %s started", iteration)
            result = system.run_iteration(target, iteration=iteration)
            results.append(result)
            row = summarize_iteration(result)
            summary["status"] = "running"
            summary["iterations"].append(row)
            summary["dataset_manager"] = system.dataset_manager.to_dict()
            final_result = {
                "task_id": system.task_id,
                "workflow": "deterministic_surrogate",
                "accepted": bool(result.get("accepted")),
                "results": results,
                "dataset_manager": system.dataset_manager.to_dict(),
                "experiment_paths": system.experiment_paths.to_dict(),
            }
            write_summary(paths["summary"], summary)
            paths["result"].write_text(json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8")
            update_workflow_panels(panels_path, target, results, config)
            logger.info("Iteration %s summary: %s", iteration, json.dumps(row, ensure_ascii=False, sort_keys=True))
            if result.get("accepted"):
                logger.info("Accepted by HighPrecisionFEM at iteration %s", iteration)
                break
        summary["status"] = "complete"
        write_summary(paths["summary"], summary)
        logger.info("Run complete.")
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        write_summary(paths["summary"], summary)
        if final_result:
            paths["result"].write_text(json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.exception("Run failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
