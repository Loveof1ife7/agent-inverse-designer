from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..DatagenFEMEvaluator import DatagenFEMEvaluator
from ..closed_loop_contracts import CurveLabelPair
from ..curve_targets import normalize_target_property


class ForwardSurrogate:
    """Fast approximate structure -> stress-curve predictor.

    This is a thin backend adapter around the current remote-forward/proxy
    evaluator path. Its labels are always marked as surrogate labels.
    """

    def __init__(
        self,
        *,
        workspace_root: str | Path = "workspace",
        evaluator: DatagenFEMEvaluator | None = None,
        backend: str = "remote_forward",
        label_weight: float = 0.25,
        batch_size: int = 256,
        gpu_workers: int = 5,
        compact: bool = False,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.evaluator = evaluator or DatagenFEMEvaluator(workspace_root=self.workspace_root, fem_backend=backend)
        self.label_weight = float(label_weight)
        self.batch_size = max(1, int(batch_size))
        self.gpu_workers = max(1, int(gpu_workers))
        self.compact = bool(compact)
        self.training_steps = 0
        self.training_examples: list[dict[str, Any]] = []

    def predict(
        self,
        structure: dict[str, Any],
        *,
        target_property: dict[str, Any] | None = None,
        pair_id: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> CurveLabelPair:
        target = normalize_target_property(target_property or structure.get("scheduled_target") or structure.get("target_property") or {})
        evaluation = self.evaluator.evaluate_explicit_structure(dict(structure), target)
        stress_curve = normalize_target_property(dict(evaluation.get("evaluated_property") or {}))
        resolved_pair_id = pair_id or self._pair_id("surr", structure)
        return CurveLabelPair(
            pair_id=resolved_pair_id,
            structure=dict(structure),
            stress_curve=stress_curve,
            label_source="surrogate",
            label_weight=self.label_weight,
            model_consumers=("InverseDesigner",),
            target_property=target,
            provenance={
                "created_at": datetime.now(timezone.utc).isoformat(),
                "backend": type(self.evaluator).__name__,
                "evaluation": evaluation,
                **dict(provenance or {}),
            },
        )

    def predict_many(
        self,
        structures: list[dict[str, Any]],
        *,
        target_properties: list[dict[str, Any]] | None = None,
        pair_prefix: str = "surr_batch",
        provenance: dict[str, Any] | None = None,
    ) -> list[CurveLabelPair]:
        if not structures:
            return []
        if self._can_remote_batch(structures):
            try:
                return self._predict_many_remote(
                    structures,
                    target_properties=target_properties,
                    pair_prefix=pair_prefix,
                    provenance=provenance,
                )
            except Exception:
                pass
        pairs: list[CurveLabelPair] = []
        target_properties = target_properties or [{} for _ in structures]
        for index, structure in enumerate(structures):
            pairs.append(
                self.predict(
                    structure,
                    target_property=target_properties[index] if index < len(target_properties) else {},
                    pair_id=f"{pair_prefix}_{index:04d}_{structure.get('structure_id', 'structure')}",
                    provenance={
                        "batch_index": index,
                        "batch_size": len(structures),
                        **dict(provenance or {}),
                    },
                )
            )
        return pairs

    def train(self, simulation_rows: list[dict[str, Any]]) -> None:
        rows = [dict(row) for row in simulation_rows if str(row.get("label_source") or "") == "simulation"]
        self.training_examples = rows
        self.training_steps += len(rows)

    def finetune(self, simulation_rows: list[dict[str, Any]]) -> None:
        rows = [dict(row) for row in simulation_rows if str(row.get("label_source") or "") == "simulation"]
        self.training_examples.extend(rows)
        self.training_steps += len(rows)

    @staticmethod
    def _pair_id(prefix: str, structure: dict[str, Any]) -> str:
        structure_id = str(structure.get("structure_id") or structure.get("sample_id") or "structure")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{prefix}_{structure_id}_{timestamp}"

    def _can_remote_batch(self, structures: list[dict[str, Any]]) -> bool:
        if getattr(self.evaluator, "fem_backend", "") != "remote_forward":
            return False
        if not hasattr(self.evaluator, "_remote_forward_client"):
            return False
        return all(self._graph_path_from_structure(structure) for structure in structures)

    def _predict_many_remote(
        self,
        structures: list[dict[str, Any]],
        *,
        target_properties: list[dict[str, Any]] | None,
        pair_prefix: str,
        provenance: dict[str, Any] | None,
    ) -> list[CurveLabelPair]:
        graph_paths = [self._graph_path_from_structure(structure) for structure in structures]
        job_id = self._pair_id(pair_prefix, {"structure_id": "remote_forward_batch"})
        result = self.evaluator._remote_forward_client().run_truss_forward_predict_batch(
            graph_paths,
            job_id=job_id,
            device=getattr(self.evaluator, "remote_forward_device", "cuda"),
            batch_size=self.batch_size,
            gpu_workers=self.gpu_workers,
            compact=self.compact,
            quiet=True,
        )
        predictions = list(result.response.get("predictions") or [])
        if not predictions and result.response.get("predicted_property"):
            predictions = [dict(result.response)]
        target_properties = target_properties or [{} for _ in structures]
        pairs: list[CurveLabelPair] = []
        for index, structure in enumerate(structures):
            prediction = dict(predictions[index]) if index < len(predictions) and isinstance(predictions[index], dict) else {}
            predicted_property = dict(prediction.get("predicted_property") or prediction.get("property") or {})
            stress_curve = normalize_target_property(predicted_property)
            pairs.append(
                CurveLabelPair(
                    pair_id=f"{pair_prefix}_{index:04d}_{structure.get('structure_id', 'structure')}",
                    structure=dict(structure),
                    stress_curve=stress_curve,
                    label_source="surrogate",
                    label_weight=self.label_weight,
                    model_consumers=("InverseDesigner",),
                    target_property=normalize_target_property(target_properties[index] if index < len(target_properties) else {}),
                    provenance={
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "backend": "remote_graphmetamat_forward_batch",
                        "remote_forward_job": result.to_dict(),
                        "prediction": prediction,
                        "batch_index": index,
                        "batch_size": len(structures),
                        **dict(provenance or {}),
                    },
                )
            )
        return pairs

    @staticmethod
    def _graph_path_from_structure(structure: dict[str, Any]) -> str:
        artifacts = dict(structure.get("artifacts") or {})
        remote = dict(artifacts.get("remote") or {})
        local = dict(artifacts.get("local") or {})
        for source in (remote, artifacts, local, structure):
            for key in ("gpkl", "gpkl_path", "graph_path"):
                value = source.get(key) if isinstance(source, dict) else None
                if value:
                    return str(value)
        return ""
