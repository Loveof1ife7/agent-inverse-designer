from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..DatagenFEMEvaluator import DatagenFEMEvaluator
from ..closed_loop_contracts import CurveLabelPair
from ..curve_targets import normalize_target_property
from .gid_c3d4_remote import RemoteGidC3D4Evaluator


P222_NORMALIZED_TO_PHYSICAL_SCALE = 27.494974
P222_PHYSICAL_OFFSETS = (27.494974, 13.747487, 13.747487)


class HighPrecisionFEM:
    """Slow high-trust structure -> stress-curve simulator."""

    def __init__(
        self,
        *,
        workspace_root: str | Path = "workspace",
        evaluator: Any | None = None,
        backend: str = "abaqus",
        inverse_label_weight: float = 1.0,
        max_workers: int | None = None,
        align_remote_graphmetamat_to_p222: bool = True,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        is_gid_c3d4_backend = backend in {"gid_c3d4_remote", "remote_gid_c3d4", "c3d4_remote"}
        if evaluator is not None:
            self.evaluator = evaluator
        elif is_gid_c3d4_backend:
            self.evaluator = RemoteGidC3D4Evaluator(workspace_root=self.workspace_root)
        else:
            self.evaluator = DatagenFEMEvaluator(workspace_root=self.workspace_root, fem_backend=backend)
        self.inverse_label_weight = float(inverse_label_weight)
        default_workers = max(1, int((os.cpu_count() or 1) * 0.3))
        self.max_workers = max(1, int(max_workers or default_workers))
        self.align_remote_graphmetamat_to_p222 = bool(align_remote_graphmetamat_to_p222 and not is_gid_c3d4_backend)

    def simulate(
        self,
        structure: dict[str, Any],
        *,
        target_property: dict[str, Any] | None = None,
        pair_id: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> CurveLabelPair:
        target = normalize_target_property(target_property or structure.get("scheduled_target") or structure.get("target_property") or {})
        fem_structure, alignment = self._prepare_fem_structure(dict(structure))
        evaluation = self.evaluator.evaluate_explicit_structure(fem_structure, target)
        stress_curve = normalize_target_property(dict(evaluation.get("evaluated_property") or {}))
        resolved_pair_id = pair_id or self._pair_id("sim", structure)
        return CurveLabelPair(
            pair_id=resolved_pair_id,
            structure=dict(structure),
            stress_curve=stress_curve,
            label_source="simulation",
            label_weight=self.inverse_label_weight,
            model_consumers=("InverseDesigner", "ForwardSurrogate"),
            target_property=target,
            provenance={
                "created_at": datetime.now(timezone.utc).isoformat(),
                "backend": type(self.evaluator).__name__,
                "evaluation": evaluation,
                "fem_coordinate_alignment": alignment,
                **dict(provenance or {}),
            },
        )

    def simulate_many(
        self,
        structures: list[dict[str, Any]],
        *,
        target_property: dict[str, Any] | None = None,
        max_workers: int | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> list[CurveLabelPair]:
        """Evaluate a batch of structures with local parallelism.

        This is intentionally an adapter-level parallel wrapper. It does not
        modify the truss FEM solver code under DatagenFEMEvaluator/core/truss.
        """

        if not structures:
            return []
        if hasattr(self.evaluator, "evaluate_many_explicit_structures"):
            return self._simulate_many_with_batch_evaluator(
                structures,
                target_property=target_property,
                max_workers=max_workers,
                provenance=provenance,
            )
        workers = max(1, int(max_workers or self.max_workers))
        if workers == 1 or len(structures) == 1:
            return [
                self.simulate(
                    structure,
                    target_property=target_property,
                    pair_id=self._batch_pair_id(index, structure),
                    provenance={
                        "batch_index": index,
                        "batch_size": len(structures),
                        "parallel_workers": 1,
                        **dict(provenance or {}),
                    },
                )
                for index, structure in enumerate(structures)
            ]

        results: list[CurveLabelPair | None] = [None] * len(structures)
        with ThreadPoolExecutor(max_workers=min(workers, len(structures))) as executor:
            future_to_index = {
                executor.submit(
                    self.simulate,
                    structure,
                    target_property=target_property,
                    pair_id=self._batch_pair_id(index, structure),
                    provenance={
                        "batch_index": index,
                        "batch_size": len(structures),
                        "parallel_workers": min(workers, len(structures)),
                        **dict(provenance or {}),
                    },
                ): index
                for index, structure in enumerate(structures)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
        return [result for result in results if result is not None]

    def _simulate_many_with_batch_evaluator(
        self,
        structures: list[dict[str, Any]],
        *,
        target_property: dict[str, Any] | None = None,
        max_workers: int | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> list[CurveLabelPair]:
        fem_structures: list[dict[str, Any]] = []
        alignments: list[dict[str, Any]] = []
        targets: list[dict[str, Any]] = []
        for structure in structures:
            target = normalize_target_property(
                target_property
                or structure.get("scheduled_target")
                or structure.get("target_property")
                or structure.get("final_target")
                or {}
            )
            fem_structure, alignment = self._prepare_fem_structure(dict(structure))
            fem_structures.append(fem_structure)
            alignments.append(alignment)
            targets.append(target)

        evaluations = self.evaluator.evaluate_many_explicit_structures(fem_structures, targets)
        pairs: list[CurveLabelPair] = []
        for index, (structure, target, alignment, evaluation) in enumerate(zip(structures, targets, alignments, evaluations)):
            stress_curve = normalize_target_property(dict(evaluation.get("evaluated_property") or {}))
            pairs.append(
                CurveLabelPair(
                    pair_id=self._batch_pair_id(index, structure),
                    structure=dict(structure),
                    stress_curve=stress_curve,
                    label_source="simulation",
                    label_weight=self.inverse_label_weight,
                    model_consumers=("InverseDesigner", "ForwardSurrogate"),
                    target_property=target,
                    provenance={
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "backend": type(self.evaluator).__name__,
                        "evaluation": evaluation,
                        "fem_coordinate_alignment": alignment,
                        "batch_index": index,
                        "batch_size": len(structures),
                        "parallel_workers": max(1, int(max_workers or self.max_workers)),
                        **dict(provenance or {}),
                    },
                )
            )
        return pairs

    @staticmethod
    def _pair_id(prefix: str, structure: dict[str, Any]) -> str:
        structure_id = str(structure.get("structure_id") or structure.get("sample_id") or "structure")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{prefix}_{structure_id}_{timestamp}"

    @staticmethod
    def _batch_pair_id(index: int, structure: dict[str, Any]) -> str:
        structure_id = str(structure.get("structure_id") or structure.get("sample_id") or "structure")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"sim_batch_{index:04d}_{structure_id}_{timestamp}"

    def _prepare_fem_structure(self, structure: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.align_remote_graphmetamat_to_p222:
            return structure, {"status": "disabled"}
        if not self._needs_p222_alignment(structure):
            return structure, {"status": "not_required"}

        coordinates = _as_coordinate_list(structure.get("coordinates"))
        scale = P222_NORMALIZED_TO_PHYSICAL_SCALE
        offsets = P222_PHYSICAL_OFFSETS
        aligned = dict(structure)
        aligned["coordinates"] = [
            [
                float(point[0]) * scale + offsets[0],
                float(point[1]) * scale + offsets[1],
                float(point[2]) * scale + offsets[2],
            ]
            for point in coordinates
        ]
        normalized_radius = _candidate_radius(structure)
        physical_radius = normalized_radius * scale if normalized_radius is not None else None
        if physical_radius is not None:
            aligned["fem_config_overrides"] = {
                **dict(structure.get("fem_config_overrides") or {}),
                "beam_radius": physical_radius,
            }
        aligned["fem_coordinate_alignment"] = {
            "status": "applied",
            "source_space": "graphmetamat_normalized_box",
            "target_space": "p222_physical_box",
            "scale": scale,
            "offsets": list(offsets),
            "rho": _candidate_rho(structure),
            "normalized_radius": normalized_radius,
            "physical_beam_radius": physical_radius,
        }
        return aligned, dict(aligned["fem_coordinate_alignment"])

    @staticmethod
    def _needs_p222_alignment(structure: dict[str, Any]) -> bool:
        if structure.get("crystal_txt_path") or structure.get("abaqus_txt_path") or structure.get("structure_path"):
            return False
        signature = " ".join(
            str(structure.get(key) or "")
            for key in ("source", "neural_backend", "representation", "structure_family")
        ).lower()
        if "graphmetamat" not in signature and "graph_truss" not in signature:
            return False
        coordinates = _as_coordinate_list(structure.get("coordinates"))
        if not coordinates:
            return False
        max_abs = max(abs(value) for point in coordinates for value in point)
        return max_abs <= 1.05


def _as_coordinate_list(payload: Any) -> list[list[float]]:
    if not isinstance(payload, list):
        return []
    coordinates: list[list[float]] = []
    for item in payload:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            return []
        try:
            point = [float(item[0]), float(item[1]), float(item[2])]
        except (TypeError, ValueError):
            return []
        coordinates.append(point)
    return coordinates


def _candidate_radius(structure: dict[str, Any]) -> float | None:
    radii = structure.get("edge_radii")
    if isinstance(radii, list) and radii:
        try:
            return float(radii[0])
        except (TypeError, ValueError):
            return None
    for key in ("radius", "beam_radius"):
        if key not in structure:
            continue
        try:
            return float(structure[key])
        except (TypeError, ValueError):
            return None
    return None


def _candidate_rho(structure: dict[str, Any]) -> float | None:
    for key in ("rho", "relative_density", "density_proxy"):
        if key not in structure:
            continue
        try:
            return float(structure[key])
        except (TypeError, ValueError):
            return None
    predicted = structure.get("predicted_property")
    if isinstance(predicted, dict):
        try:
            return float(predicted["rho"])
        except (KeyError, TypeError, ValueError):
            return None
    return None
