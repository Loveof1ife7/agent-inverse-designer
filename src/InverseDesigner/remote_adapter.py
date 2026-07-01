from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..closed_loop_contracts import TargetSchedule
from ..curve_targets import normalize_stress_curve_target
from .remote import (
    RemoteGraphMetaMatClient,
    RemoteInverseDesignerConfig,
    RemoteJobResult,
    default_truss_finetune_config,
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _safe_name(value: str, default: str = "remote_truss") -> str:
    text = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))
    return text[:140] or default


def _fixed_strain_grid(length: int = 256) -> list[float]:
    if length <= 1:
        return [0.0]
    return [0.3 * index / (length - 1) for index in range(length)]


def _as_float_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    values: list[float] = []
    for item in value:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            return []
    return values


def _is_pair_curve(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first = value[0]
    return isinstance(first, (list, tuple)) and len(first) >= 2


def _resample_curve(strain: list[float], stress: list[float], target_grid: list[float]) -> list[float]:
    if not strain or not stress:
        return []
    pairs = sorted((float(x), max(0.0, float(y))) for x, y in zip(strain, stress))
    merged: list[tuple[float, float]] = []
    index = 0
    while index < len(pairs):
        current_x = pairs[index][0]
        values = [pairs[index][1]]
        index += 1
        while index < len(pairs) and abs(pairs[index][0] - current_x) <= 1e-12:
            values.append(pairs[index][1])
            index += 1
        merged.append((current_x, sum(values) / len(values)))
    if len(merged) == 1:
        return [merged[0][1] for _ in target_grid]

    xs = [item[0] for item in merged]
    ys = [item[1] for item in merged]
    output: list[float] = []
    cursor = 0
    for target_x in target_grid:
        if target_x <= xs[0]:
            output.append(ys[0])
            continue
        if target_x >= xs[-1]:
            output.append(ys[-1])
            continue
        while cursor + 1 < len(xs) and xs[cursor + 1] < target_x:
            cursor += 1
        x0, x1 = xs[cursor], xs[cursor + 1]
        y0, y1 = ys[cursor], ys[cursor + 1]
        if abs(x1 - x0) <= 1e-12:
            output.append(y1)
        else:
            weight = (target_x - x0) / (x1 - x0)
            output.append(y0 * (1.0 - weight) + y1 * weight)
    return output


def _stress_curve_target(target_property: dict[str, Any]) -> dict[str, Any] | None:
    return normalize_stress_curve_target(target_property)


def _read_curve_csv(path: str | os.PathLike[str]) -> tuple[list[float], list[float]]:
    strain: list[float] = []
    stress: list[float] = []
    curve_path = Path(path)
    if not curve_path.exists():
        return strain, stress
    with curve_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                x = float(row[0])
                y = float(row[1])
            except ValueError:
                continue
            strain.append(x)
            stress.append(y)
    return strain, stress


@dataclass
class RemoteGraphMetaMatInverseDesigner:
    """Closed-loop adapter for the remote GraphMetaMat truss inverse designer."""

    client: RemoteGraphMetaMatClient = field(default_factory=RemoteGraphMetaMatClient)
    workspace_root: Path = field(default_factory=lambda: Path("workspace"))
    num_runs: int = field(default_factory=lambda: int(os.getenv("INVERSE_GRAPHMETAMAT_NUM_RUNS", os.getenv("INVERSE_TRUSS_NUM_RUNS", "16"))))
    top_k: int = field(default_factory=lambda: int(os.getenv("INVERSE_GRAPHMETAMAT_TOP_K", os.getenv("INVERSE_TRUSS_TOP_K", "1"))))
    batch_size: int = field(default_factory=lambda: int(os.getenv("INVERSE_GRAPHMETAMAT_BATCH_SIZE", "32")))
    device: str = field(default_factory=lambda: os.getenv("INVERSE_GRAPHMETAMAT_DEVICE", os.getenv("INVERSE_TRUSS_DEVICE", "cuda")))
    timeout_seconds: int = field(default_factory=lambda: int(os.getenv("INVERSE_GRAPHMETAMAT_TIMEOUT_SECONDS", "1800")))
    fallback_designer: Any | None = None
    download_artifacts: bool = True
    run_remote_finetune: bool = field(default_factory=lambda: _env_flag("REMOTE_INVERSE_RUN_FINETUNE", True))
    run_training: bool = field(default_factory=lambda: _env_flag("REMOTE_INVERSE_RUN_TRAINING", False))

    name: str = "remote_graphmetamat_truss"
    structure_family: str = "truss"
    representation: str = "graph_truss"

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root)
        self.sample_counter = 0
        self.backend_failures: list[dict[str, Any]] = []
        self.finetune_history: list[dict[str, Any]] = []

    @classmethod
    def from_env(
        cls,
        *,
        workspace_root: str | Path = "workspace",
        fallback_designer: Any | None = None,
    ) -> "RemoteGraphMetaMatInverseDesigner":
        return cls(
            client=RemoteGraphMetaMatClient(RemoteInverseDesignerConfig()),
            workspace_root=Path(workspace_root),
            fallback_designer=fallback_designer,
        )

    def sample_structure(
        self,
        target_property: dict[str, Any],
        structure_family: str | None = None,
        prefer_neural: bool | None = None,
    ) -> dict[str, Any] | None:
        del prefer_neural
        if structure_family and structure_family.lower() not in {"truss", "graph_truss"}:
            return self._fallback(target_property, structure_family=structure_family)

        target = _stress_curve_target(target_property)
        if target is None:
            self.backend_failures.append(
                {
                    "backend": self.name,
                    "structure_family": self.structure_family,
                    "reason": "remote_graphmetamat_requires_stress_curve_target",
                    "target_keys": sorted(str(key) for key in target_property),
                }
            )
            return self._fallback(target_property, structure_family=structure_family)

        self.sample_counter += 1
        job_id = _safe_name(
            "closed_loop_{stamp}_{index:06d}".format(
                stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
                index=self.sample_counter,
            )
        )
        request = {
            "job_id": job_id,
            "structure_family": "truss",
            "target": target,
            "options": {
                "project_dir": "third-party/GraphMetaMat",
                "num_runs": int(self.num_runs),
                "top_k": int(self.top_k),
                "device": self.device,
                "timeout_seconds": int(self.timeout_seconds),
            },
        }
        try:
            result = self.client.run_inverse_design(request, job_id=job_id, download_artifacts=self.download_artifacts)
        except Exception as exc:
            self.backend_failures.append(
                {
                    "backend": self.name,
                    "structure_family": self.structure_family,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "job_id": job_id,
                }
            )
            return self._fallback(target_property, structure_family=structure_family)

        candidate = dict(result.response.get("candidate") or {})
        if result.status != "success" or not candidate:
            self.backend_failures.append(
                {
                    "backend": self.name,
                    "structure_family": self.structure_family,
                    "reason": result.response.get("error") or "no_candidate",
                    "job_id": result.job_id,
                    "status": result.status,
                }
            )
            return self._fallback(target_property, structure_family=structure_family)
        return self._structure_from_candidate(candidate, result, target_property, target)

    def sample_schedule(self, schedule: TargetSchedule | dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(schedule, dict):
            schedule = TargetSchedule(**schedule)

        batch_records = self._sample_schedule_batch(schedule)
        if batch_records:
            return batch_records

        records: list[dict[str, Any]] = []
        for step_index, item in enumerate(schedule.scheduled_targets, start=1):
            requested_family = self._structure_family_from_schedule_item(item)
            for sample_index in range(1, item.samples + 1):
                structure = self.sample_structure(item.target_property, structure_family=requested_family or "truss")
                records.append(
                    {
                        "schedule_id": schedule.schedule_id,
                        "schedule_step": step_index,
                        "sample_index": sample_index,
                        "scheduled_target": dict(item.target_property),
                        "schedule_item": item.to_dict(),
                        "structure_family": requested_family or "truss",
                        "final_target": dict(schedule.final_target),
                        "structure": structure,
                        "status": "sampled" if structure is not None else "no_candidate",
                    }
                )
        return records

    def _sample_schedule_batch(self, schedule: TargetSchedule) -> list[dict[str, Any]]:
        requests: list[dict[str, Any]] = []
        curves: list[list[float]] = []
        for step_index, item in enumerate(schedule.scheduled_targets, start=1):
            requested_family = self._structure_family_from_schedule_item(item) or "truss"
            if requested_family.lower() not in {"truss", "graph_truss"}:
                return []
            target = _stress_curve_target(item.target_property)
            if target is None:
                return []
            stress = _as_float_list(target.get("stress"))
            if not stress:
                return []
            for sample_index in range(1, item.samples + 1):
                requests.append(
                    {
                        "schedule_id": schedule.schedule_id,
                        "schedule_step": step_index,
                        "sample_index": sample_index,
                        "scheduled_target": dict(item.target_property),
                        "normalized_target": target,
                        "schedule_item": item.to_dict(),
                        "structure_family": requested_family,
                        "final_target": dict(schedule.final_target),
                    }
                )
                curves.append(stress)

        if not requests:
            return []

        self.sample_counter += 1
        job_id = _safe_name(
            "closed_loop_batch_{stamp}_{index:06d}".format(
                stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
                index=self.sample_counter,
            )
        )
        try:
            result = self.client.run_inverse_design_batch(
                curves,
                job_id=job_id,
                num_runs=int(self.num_runs),
                top_k=max(int(self.top_k), len(requests)),
                batch_size=max(int(self.batch_size), len(requests)),
                device=self.device,
                timeout_seconds=int(self.timeout_seconds),
                download_artifacts=self.download_artifacts,
            )
        except Exception as exc:
            self.backend_failures.append(
                {
                    "backend": self.name,
                    "structure_family": self.structure_family,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "job_id": job_id,
                    "mode": "batch",
                }
            )
            return []

        candidates = list(result.response.get("candidates") or [])
        if result.status != "success" or not candidates:
            candidate = dict(result.response.get("candidate") or {})
            candidates = [candidate] if candidate else []
        records: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            request_index = self._candidate_target_index(candidate, candidate_index, len(requests))
            request = requests[request_index]
            structure = self._structure_from_candidate(
                dict(candidate),
                result,
                request["scheduled_target"],
                request["normalized_target"],
            )
            structure["remote_inverse_batch"] = {
                "job_id": result.job_id,
                "candidate_index": candidate_index,
                "target_index": request_index,
                "num_candidates": len(candidates),
            }
            records.append(
                {
                    **request,
                    "structure": structure,
                    "status": "sampled",
                }
            )
        return records

    def finetune(self, new_samples: list[dict[str, Any]]) -> None:
        rows = [row for row in (self._active_learning_row(item) for item in new_samples) if row]
        if not rows:
            self.finetune_history.append({"status": "skipped", "reason": "no_remote_truss_training_rows"})
            return

        round_id = _safe_name(
            os.getenv("REMOTE_INVERSE_FINETUNE_ROUND")
            or f"closed_loop_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        )
        local_round = self.client.config.local_workspace / round_id / "windows_eval"
        local_round.mkdir(parents=True, exist_ok=True)
        evaluated_path = local_round / "evaluated_samples.jsonl"
        with evaluated_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                import json

                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        candidates_dir = self._materialize_candidate_files(rows, local_round / "candidates")
        config = default_truss_finetune_config(
            round_id,
            remote_python=self.client.config.remote_python,
            min_accepted=1,
            enable_forward=_env_flag("REMOTE_INVERSE_FINETUNE_FORWARD", False),
            enable_inverse_rl=_env_flag("REMOTE_INVERSE_FINETUNE_INVERSE_RL", False),
            enable_inverse_il=_env_flag("REMOTE_INVERSE_FINETUNE_INVERSE_IL", False),
        )
        result = self.client.submit_truss_finetune(
            round_id=round_id,
            evaluated_samples_path=evaluated_path,
            candidates_dir=candidates_dir if candidates_dir.exists() else None,
            finetune_config=config,
            run_training=self.run_training,
            run_remote=self.run_remote_finetune,
        )
        self.finetune_history.append(result.to_dict())

    def _fallback(self, target_property: dict[str, Any], structure_family: str | None = None) -> dict[str, Any] | None:
        if self.fallback_designer is None or not hasattr(self.fallback_designer, "sample_structure"):
            return None
        return self.fallback_designer.sample_structure(target_property, structure_family=structure_family)

    def _structure_from_candidate(
        self,
        candidate: dict[str, Any],
        result: RemoteJobResult,
        target_property: dict[str, Any],
        target: dict[str, Any],
    ) -> dict[str, Any]:
        payload = dict(candidate)
        payload.setdefault("structure_id", f"{result.job_id}_rank01")
        payload.setdefault("structure_family", "truss")
        payload.setdefault("representation", "graph_truss")
        payload.setdefault("source", f"inverse_designer_remote:{self.name}")
        payload.setdefault("neural_backend", self.name)
        payload["target_property"] = dict(target_property)
        payload["requested_property"] = dict(target_property)
        payload["remote_requested_target"] = dict(target)
        payload["remote_inverse_job"] = {
            "job_id": result.job_id,
            "status": result.status,
            "local_dir": result.local_dir,
            "remote_dir": result.remote_dir,
            "response_path": result.response_path,
        }
        payload["artifacts"] = self._localize_artifacts(dict(payload.get("artifacts") or {}), result)
        return payload

    @staticmethod
    def _localize_artifacts(artifacts: dict[str, Any], result: RemoteJobResult) -> dict[str, Any]:
        localized = dict(artifacts)
        remote_prefix = result.remote_dir.rstrip("/") + "/"
        local_dir = Path(result.local_dir)
        local_artifacts: dict[str, str] = {}
        for key, value in artifacts.items():
            if not isinstance(value, str) or not value.startswith(remote_prefix):
                continue
            relative = value[len(remote_prefix) :].replace("/", os.sep)
            local_path = local_dir / relative
            local_artifacts[key] = str(local_path)
            if key in {"gpkl", "vtk", "graph_png", "curve_png", "top_designs_json", "summary_csv", "results_pkl", "output_dir"}:
                localized[key] = str(local_path)
        if local_artifacts:
            localized["remote"] = artifacts
            localized["local"] = local_artifacts
        return localized

    @staticmethod
    def _structure_family_from_schedule_item(item: Any) -> str:
        expected_effect = dict(getattr(item, "expected_effect", {}) or {})
        for key in ("structure_family", "family", "representation_family"):
            value = expected_effect.get(key)
            if value:
                return str(value)
        strategy = str(getattr(item, "strategy", "") or "")
        return "truss" if "truss" in strategy.lower() else ""

    @staticmethod
    def _candidate_target_index(candidate: dict[str, Any], fallback_index: int, target_count: int) -> int:
        for key in ("target_index", "target_idx", "index"):
            if candidate.get(key) is None:
                continue
            try:
                value = int(candidate[key])
            except (TypeError, ValueError):
                continue
            if 0 <= value < target_count:
                return value
        return min(max(0, int(fallback_index)), max(0, target_count - 1))

    def _active_learning_row(self, item: dict[str, Any]) -> dict[str, Any] | None:
        structure = dict(item.get("explicit_structure") or item.get("output_structure") or item.get("structure") or {})
        if not structure:
            return None
        validity = dict(item.get("validity") or item.get("status") or {})
        if validity.get("geometry_status") not in {"valid", "success"} or validity.get("fem_status") != "success":
            return None

        strain_grid = _fixed_strain_grid(256)
        stress = self._truth_stress(item, strain_grid)
        if not stress:
            return None

        sample_id = str(item.get("sample_id") or item.get("structure_id") or structure.get("structure_id") or "sample")
        scheduled_target = dict(item.get("structure_code", {}).get("scheduled_target") or structure.get("scheduled_target") or {})
        target = _stress_curve_target(scheduled_target) or dict(structure.get("remote_requested_target") or {})
        artifacts = dict(structure.get("artifacts") or {})
        gpkl_path = str(artifacts.get("gpkl") or dict(artifacts.get("local") or {}).get("gpkl") or structure.get("gpkl_path") or "")
        fem_raw = dict(item.get("raw_metrics") or {})
        fem_curve_path = str(fem_raw.get("fem_curve_path") or fem_raw.get("raw_curve_csv") or fem_raw.get("curve_path") or "")
        return {
            "candidate_id": _safe_name(sample_id, default="candidate"),
            "target_id": _safe_name(str(item.get("target_id") or sample_id), default="target"),
            "structure_family": "truss",
            "representation": "graph_truss",
            "inverse_designer": {
                "name": "GraphMetaMat",
                "requested_target": target or {"type": "stress_curve", "strain_grid": strain_grid, "stress": stress},
            },
            "structure": {
                "gpkl_path": gpkl_path,
                "coordinates": structure.get("coordinates", []),
                "edges": structure.get("edges", []),
                "edge_radii": structure.get("edge_radii", []),
                "rho": structure.get("rho"),
                "num_nodes": len(structure.get("coordinates", []) or []),
                "num_edges": len(structure.get("edges", []) or []),
            },
            "evaluation": {
                "eval_status": "success",
                "geometry_status": "valid",
                "fem_status": "success",
                "fidelity": item.get("fidelity") or validity.get("source") or "windows_fem",
                "artifacts": {
                    "raw_curve_csv": fem_curve_path,
                    "explicit_structure_path": structure.get("structure_path", ""),
                },
            },
            "response": {
                "task": "compression_stress_strain",
                "strain_grid": strain_grid,
                "stress": stress,
                "curve": [[x, y] for x, y in zip(strain_grid, stress)],
                "relative_density": structure.get("rho"),
            },
            "metrics": dict(item.get("structure_feature") or {}),
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "label_source": "windows_fem",
                "closed_loop_sample": dict(item),
            },
        }

    @staticmethod
    def _truth_stress(item: dict[str, Any], strain_grid: list[float]) -> list[float]:
        for source in (
            item.get("property"),
            item.get("evaluated_property"),
            item.get("response"),
            dict(item.get("output_structure") or {}).get("response"),
            dict(item.get("explicit_structure") or {}).get("response"),
        ):
            if not isinstance(source, dict):
                continue
            target = _stress_curve_target(source)
            if target:
                stress = _as_float_list(target.get("stress"))
                strain = _as_float_list(target.get("strain_grid")) or _fixed_strain_grid(len(stress))
                return _resample_curve(strain, stress, strain_grid)

        raw_metrics = dict(item.get("raw_metrics") or {})
        curve_path = raw_metrics.get("fem_curve_path") or raw_metrics.get("raw_curve_csv") or raw_metrics.get("curve_path")
        if curve_path:
            strain, stress = _read_curve_csv(curve_path)
            return _resample_curve(strain, stress, strain_grid)
        return []

    @staticmethod
    def _materialize_candidate_files(rows: list[dict[str, Any]], output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            gpkl_path = str(dict(row.get("structure") or {}).get("gpkl_path") or "")
            if not gpkl_path:
                continue
            source = Path(gpkl_path)
            if source.exists():
                shutil.copy2(source, output_dir / source.name)
        return output_dir


__all__ = ["RemoteGraphMetaMatInverseDesigner"]
