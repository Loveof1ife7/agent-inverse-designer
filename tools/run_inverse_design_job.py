from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.InverseDesigner import DiffuMetaEquationBackend, GraphMetaMatTrussBackend, InverseDesigner
from src.KnowledgeBase import KnowledgeBase


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _target_property(request: dict[str, Any]) -> dict[str, Any]:
    target = dict(request.get("target_property") or request.get("target") or {})
    target_type = str(target.get("type") or "").strip()
    if target_type == "control_points_stress":
        return {"control_points_stress": list(target.get("values") or target.get("control_points_stress") or [])}
    if target_type == "stress_curve":
        if target.get("target_curve_path"):
            return {"target_curve_path": str(target["target_curve_path"])}
        return {"stress_curve": list(target.get("stress") or target.get("values") or [])}
    if "values" in target and len(target) <= 2:
        return {"target_vector": list(target.get("values") or [])}
    return target


def _make_backend(request: dict[str, Any]):
    family = str(request.get("structure_family") or request.get("family") or "").strip().lower()
    options = dict(request.get("options") or {})
    if family == "tpms":
        project_dir = Path(options.get("project_dir") or ROOT / "third-party" / "DiffusionMetamaterials")
        return DiffuMetaEquationBackend(
            name=str(options.get("name") or "tpms_diffumeta_remote"),
            structure_family="tpms",
            checkpoint_path=str(options.get("checkpoint_path") or project_dir / "model_checkpoints" / "model_checkpoint.pth"),
            project_dir=project_dir,
            num_samples=int(options.get("num_samples", request.get("num_samples", 200))),
            cfg_scale=float(options.get("cfg_scale", request.get("cfg_scale", 10.0))),
            timeout_seconds=int(options.get("timeout_seconds", 3600)),
        )
    if family == "truss":
        project_dir = Path(options.get("project_dir") or ROOT / "third-party" / "GraphMetaMat")
        return GraphMetaMatTrussBackend(
            name=str(options.get("name") or "graphmetamat_truss_remote"),
            project_dir=project_dir,
            num_runs=int(options.get("num_runs", request.get("num_runs", 16))),
            top_k=int(options.get("top_k", request.get("top_k", 1))),
            device=str(options.get("device", request.get("device", "cuda"))),
            timeout_seconds=int(options.get("timeout_seconds", 3600)),
        )
    raise ValueError(f"Unsupported structure_family={family!r}; expected tpms or truss")


def run_job(request: dict[str, Any]) -> dict[str, Any]:
    job_id = str(request.get("job_id") or "inverse_job")
    family = str(request.get("structure_family") or request.get("family") or "").strip().lower()
    workspace_root = Path(request.get("workspace_root") or ROOT / "workspace" / "remote_inverse_jobs" / job_id)
    workspace_root.mkdir(parents=True, exist_ok=True)
    kb = KnowledgeBase(workspace_root / "knowledge.sqlite")
    try:
        backend = _make_backend(request)
        inverse = InverseDesigner(
            kb,
            neural_backends=[backend],
            enable_neural=True,
            workspace_root=workspace_root,
            fallback_to_retrieval=False,
        )
        structure = inverse.sample_structure(_target_property(request), structure_family=family)
        return {
            "job_id": job_id,
            "status": "success" if structure is not None else "no_candidate",
            "structure_family": family,
            "candidate": structure,
            "backend_failures": inverse.backend_failures,
            "workspace_root": str(workspace_root.resolve()),
        }
    finally:
        kb.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one inverse-design job for Windows/remote orchestration.")
    parser.add_argument("--request", required=True, help="Path to request JSON.")
    parser.add_argument("--output", required=True, help="Path to write response JSON.")
    args = parser.parse_args()

    request = _read_json(args.request)
    try:
        response = run_job(request)
    except Exception as exc:
        response = {
            "job_id": str(request.get("job_id") or "inverse_job"),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(args.output, response)
        raise
    _write_json(args.output, response)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

