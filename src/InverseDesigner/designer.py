from __future__ import annotations

from typing import Any

from ..KnowledgeBase import KnowledgeBase
from ..closed_loop_contracts import TargetSchedule


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

    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
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

    def sample_structure(self, target_property: dict[str, float]) -> dict[str, Any] | None:
        """
        Return the explicit structure whose recorded property is nearest to the target.

        Primary source: TrainingDataset examples.
        Fallback source: KnowledgeBase samples that carry explicit_structure.
        """
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
            for sample_index in range(1, item.samples + 1):
                structure = self.sample_structure(item.target_property)
                records.append(
                    {
                        "schedule_id": schedule.schedule_id,
                        "schedule_step": step_index,
                        "sample_index": sample_index,
                        "scheduled_target": dict(item.target_property),
                        "schedule_item": item.to_dict(),
                        "final_target": dict(schedule.final_target),
                        "structure": structure,
                        "status": "sampled" if structure is not None else "no_candidate",
                    }
                )
        return records

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
