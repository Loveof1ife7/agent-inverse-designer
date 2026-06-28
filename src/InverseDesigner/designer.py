from __future__ import annotations

from pathlib import Path
from typing import Any

from ..KnowledgeBase import KnowledgeBase
from ..closed_loop_contracts import TargetSchedule
from .backends import NeuralInverseBackend, build_env_backends, neural_enabled_from_env


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _property_distance(target: dict[str, float], observed: dict[str, float]) -> float:
    keys = sorted(set(target) & set(observed))
    if not keys:
        return float("inf")
    total = 0.0
    for key in keys:
        target_value = _safe_float(target[key], 0.0)
        observed_value = _safe_float(observed[key], 0.0)
        scale = abs(target_value) if abs(target_value) > 1e-9 else 1.0
        total += abs(observed_value - target_value) / scale
    return total / len(keys)


class InverseDesigner:
    """
    Main-loop inverse designer.

    Contract:

        target property -> explicit truss structure

    The current implementation is deterministic nearest-neighbor retrieval over
    trainable examples that already contain explicit structures. AgentExplorer
    owns Meta search; InverseDesigner does not propose datagen configs.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        neural_backends: list[NeuralInverseBackend] | None = None,
        enable_neural: bool | None = None,
        workspace_root: str | Path | None = None,
        fallback_to_retrieval: bool = True,
    ):
        self.knowledge_base = knowledge_base
        self.workspace_root = Path(workspace_root or "workspace")
        self.neural_backends = list(neural_backends) if neural_backends is not None else build_env_backends(self.workspace_root)
        self.enable_neural = bool(enable_neural) if enable_neural is not None else neural_enabled_from_env(bool(self.neural_backends))
        self.fallback_to_retrieval = bool(fallback_to_retrieval)
        self.backend_failures: list[dict[str, Any]] = []
        self.training_steps = 0
        self.training_examples: list[dict[str, Any]] = []
        self.property_keys: tuple[str, ...] = ()

    def train(self, dataset: list[dict[str, Any]]) -> None:
        examples = self._prepare_examples(dataset)
        self.training_examples = examples
        self.training_steps += len(examples)

    def finetune(self, new_samples: list[dict[str, Any]]) -> None:
        examples = self._prepare_examples(new_samples)
        self.training_examples.extend(examples)
        self.training_steps += len(examples)

    def register_backend(self, backend: NeuralInverseBackend) -> None:
        self.neural_backends.append(backend)
        self.enable_neural = True

    def sample_structure(
        self,
        target_property: dict[str, float],
        structure_family: str | None = None,
        prefer_neural: bool | None = None,
    ) -> dict[str, Any] | None:
        """
        Return a generated or retrieved explicit structure for the target.

        Primary source when enabled: configured neural inverse-design backends
        for TPMS, truss, B-spline, voxel, or project-specific families.
        Fallback source: nearest-neighbor retrieval over trainable explicit
        structures and KnowledgeBase samples.
        """
        use_neural = self.enable_neural if prefer_neural is None else bool(prefer_neural)
        if use_neural:
            neural_structure = self._sample_neural_structure(target_property, structure_family=structure_family)
            if neural_structure is not None:
                return neural_structure
            if not self.fallback_to_retrieval:
                return None

        return self._sample_retrieval_structure(target_property)

    def _sample_neural_structure(
        self,
        target_property: dict[str, float],
        structure_family: str | None = None,
    ) -> dict[str, Any] | None:
        family = (structure_family or "").strip().lower()
        candidates = [
            backend
            for backend in self.neural_backends
            if not family or backend.structure_family.lower() == family
        ]
        for index, backend in enumerate(candidates, start=1):
            if not backend.available():
                self.backend_failures.append(
                    {
                        "backend": backend.name,
                        "structure_family": backend.structure_family,
                        "reason": "backend_unavailable",
                    }
                )
                continue
            output_dir = self.workspace_root / "inverse_designer" / backend.name / f"sample_{len(self.backend_failures) + index:06d}"
            try:
                structure = backend.sample(target_property=dict(target_property), output_dir=output_dir, sample_index=index)
            except Exception as exc:
                self.backend_failures.append(
                    {
                        "backend": backend.name,
                        "structure_family": backend.structure_family,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if not structure:
                self.backend_failures.append(
                    {
                        "backend": backend.name,
                        "structure_family": backend.structure_family,
                        "reason": "no_candidate",
                    }
                )
                continue
            payload = dict(structure)
            payload.setdefault("target_property", dict(target_property))
            payload.setdefault("requested_property", dict(target_property))
            payload.setdefault("source", f"inverse_designer_neural:{backend.name}")
            payload.setdefault("neural_backend", backend.name)
            payload.setdefault("structure_family", backend.structure_family)
            payload.setdefault("representation", backend.representation)
            return payload
        return None

    def _sample_retrieval_structure(self, target_property: dict[str, float]) -> dict[str, Any] | None:
        if self.training_examples:
            best = min(
                self.training_examples,
                key=lambda item: _property_distance(target_property, item["property"]),
            )
            return self._structure_payload(
                structure=best["explicit_structure"],
                structure_id=best["sample_id"],
                property_payload=best["property"],
                target_property=target_property,
                source="inverse_designer_retrieval",
            )

        explicit_samples = [
            sample
            for sample in self.knowledge_base.get_similar_property_samples(target_property, top_k=16)
            if sample.explicit_structure
        ]
        if not explicit_samples:
            return None

        best_sample = min(
            explicit_samples,
            key=lambda sample: _property_distance(target_property, sample.evaluated_property),
        )
        return self._structure_payload(
            structure=best_sample.explicit_structure,
            structure_id=best_sample.structure_id,
            property_payload=best_sample.evaluated_property,
            target_property=target_property,
            source="inverse_designer_kb_retrieval",
        )

    def sample_schedule(self, schedule: TargetSchedule | dict[str, Any]) -> list[dict[str, Any]]:
        """Sample explicit structures for every scheduled target.

        Returned records preserve target-schedule provenance so the scheduler can
        write raw observations and KnowledgeEvidence.
        """
        if isinstance(schedule, dict):
            schedule = TargetSchedule(**schedule)

        records: list[dict[str, Any]] = []
        for step_index, item in enumerate(schedule.scheduled_targets, start=1):
            requested_family = self._structure_family_from_schedule_item(item)
            for sample_index in range(1, item.samples + 1):
                structure_family = requested_family or self._round_robin_family(step_index, sample_index)
                structure = self.sample_structure(item.target_property, structure_family=structure_family)
                records.append(
                    {
                        "schedule_id": schedule.schedule_id,
                        "schedule_step": step_index,
                        "sample_index": sample_index,
                        "scheduled_target": dict(item.target_property),
                        "schedule_item": item.to_dict(),
                        "structure_family": structure_family,
                        "final_target": dict(schedule.final_target),
                        "structure": structure,
                        "status": "sampled" if structure is not None else "no_candidate",
                    }
                )
        return records

    def _round_robin_family(self, step_index: int, sample_index: int) -> str:
        if not (self.enable_neural and self.neural_backends):
            return ""
        families = list(dict.fromkeys(backend.structure_family for backend in self.neural_backends if backend.structure_family))
        if not families:
            return ""
        return families[(max(step_index, 1) + max(sample_index, 1) - 2) % len(families)]

    @staticmethod
    def _structure_family_from_schedule_item(item: Any) -> str:
        expected_effect = dict(getattr(item, "expected_effect", {}) or {})
        for key in ("structure_family", "family", "representation_family"):
            value = expected_effect.get(key)
            if value:
                return str(value)
        strategy = str(getattr(item, "strategy", "") or "")
        for family in ("tpms", "truss", "b_spline", "bspline", "voxel"):
            if family in strategy.lower():
                return "b_spline" if family == "bspline" else family
        return ""

    def _prepare_examples(self, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        examples = []
        for item in dataset:
            if not self._is_trainable(item):
                continue
            property_payload = dict(item.get("property") or item.get("evaluated_property") or {})
            explicit_structure = self._explicit_structure_from_record(item)
            if not property_payload or not explicit_structure:
                continue
            examples.append(
                {
                    "sample_id": str(item.get("sample_id") or item.get("structure_id") or ""),
                    "property": property_payload,
                    "explicit_structure": explicit_structure,
                    "weight": _safe_float(item.get("weight"), 1.0),
                }
            )
        if examples and not self.property_keys:
            self.property_keys = tuple(sorted(examples[0]["property"].keys()))
        return examples

    @staticmethod
    def _explicit_structure_from_record(item: dict[str, Any]) -> dict[str, Any]:
        explicit_structure = dict(item.get("explicit_structure") or item.get("structure_json") or {})
        if explicit_structure:
            explicit_structure.setdefault("structure_id", item.get("structure_id") or item.get("sample_id") or "")
            return explicit_structure

        structure = dict(item.get("structure") or {})
        if structure.get("coordinates") or structure.get("nodes") or structure.get("edges"):
            structure.setdefault("structure_id", item.get("structure_id") or item.get("sample_id") or "")
            return structure
        return {}

    @staticmethod
    def _is_trainable(item: dict[str, Any]) -> bool:
        validity = dict(item.get("validity") or item.get("status") or {})
        geometry_status = str(validity.get("geometry_status") or item.get("geometry_status") or "")
        fem_status = str(validity.get("fem_status") or item.get("fem_status") or "")
        property_payload = dict(item.get("property") or item.get("evaluated_property") or {})
        return geometry_status == "valid" and fem_status == "success" and bool(property_payload)

    @staticmethod
    def _structure_payload(
        structure: dict[str, Any],
        structure_id: str,
        property_payload: dict[str, float],
        target_property: dict[str, float],
        source: str,
    ) -> dict[str, Any]:
        payload = dict(structure)
        payload.setdefault("structure_id", structure_id)
        payload.setdefault("source", source)
        payload.setdefault("retrieved_property", dict(property_payload))
        payload.setdefault("retrieval_distance", _property_distance(target_property, property_payload))
        return payload
