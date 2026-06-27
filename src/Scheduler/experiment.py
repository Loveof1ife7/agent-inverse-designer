from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..closed_loop_contracts import ExperimentPaths


def make_experiment_paths(workspace_root: str | Path, task_id: str) -> ExperimentPaths:
    root = Path(workspace_root) / "experiments" / task_id
    runs = root / "runs"
    knowledge = root / "knowledge"
    for path in (root, runs, knowledge):
        path.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(
        root_dir=str(root.resolve()),
        runs_dir=str(runs.resolve()),
        events_dir=str(root.resolve()),
        artifacts_dir=str(root.resolve()),
        knowledge_dir=str(knowledge.resolve()),
    )


def make_task_id(prefix: str = "discovery") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{stamp}"


def dump_experiment_manifest(paths: ExperimentPaths, payload: dict):
    manifest = Path(paths.root_dir) / "manifest.json"
    data = {"paths": asdict(paths), **payload}
    manifest.write_text(__import__("json").dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
