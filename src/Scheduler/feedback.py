from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..closed_loop_contracts import KnowledgeSample


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sample_distance_to_target(sample: KnowledgeSample, target_property: dict[str, float] | None = None) -> float:
    if sample.property_error:
        return max(_safe_float(value, 0.0) for value in sample.property_error.values())

    target_property = dict(target_property or sample.target_property or {})
    if not target_property:
        return float("inf")

    errors = []
    for key, target in target_property.items():
        observed = _safe_float(sample.evaluated_property.get(key), 0.0)
        target_value = _safe_float(target, 0.0)
        scale = abs(target_value) if abs(target_value) > 1e-9 else 1.0
        errors.append(abs(observed - target_value) / scale)
    return max(errors) if errors else float("inf")


def _sample_diversity_key(sample: KnowledgeSample) -> tuple[Any, ...]:
    return (
        sample.symmetry,
        sample.basic_unit_type,
        sample.unit_cell_type,
        sample.topology_type,
        sample.connectivity_pattern,
        sample.parameter_config.get("max_bars"),
        round(_safe_float(sample.parameter_config.get("rho_target"), 0.0), 4),
    )


def _sample_id(sample: KnowledgeSample | None) -> str:
    return sample.structure_id if sample is not None else ""


@dataclass
class FeedbackSignal:
    """
    Low-intelligence, deterministic, target-aware control summary for one loop iteration.
    """

    target_property: dict[str, float]
    evaluated_count: int
    label_counts: dict[str, int]
    success_found: bool
    should_stop: bool
    best_sample: KnowledgeSample | None = None
    best_success: KnowledgeSample | None = None
    best_near_miss: KnowledgeSample | None = None
    representative_failures: list[KnowledgeSample] = field(default_factory=list)
    diversity_samples: list[KnowledgeSample] = field(default_factory=list)
    feedback_samples: list[KnowledgeSample] = field(default_factory=list)
    next_anchor_sample: KnowledgeSample | None = None
    distance_to_target: float = float("inf")
    main_error_direction: str = ""
    suggested_strategy: str = "explore"
    suggested_next_action: str = "explore"
    failure_modes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("best_sample", "best_success", "best_near_miss", "next_anchor_sample"):
            sample = getattr(self, key)
            payload[key] = sample.to_dict() if sample is not None else None
        for key in ("representative_failures", "diversity_samples", "feedback_samples"):
            payload[key] = [sample.to_dict() for sample in getattr(self, key)]
        return payload

    def brief(self) -> dict[str, Any]:
        return {
            "evaluated_count": self.evaluated_count,
            "label_counts": dict(self.label_counts),
            "success_found": self.success_found,
            "should_stop": self.should_stop,
            "best_sample_id": _sample_id(self.best_sample),
            "best_success_id": _sample_id(self.best_success),
            "best_near_miss_id": _sample_id(self.best_near_miss),
            "next_anchor_sample_id": _sample_id(self.next_anchor_sample),
            "representative_failure_ids": [_sample_id(sample) for sample in self.representative_failures],
            "diversity_sample_ids": [_sample_id(sample) for sample in self.diversity_samples],
            "feedback_sample_ids": [_sample_id(sample) for sample in self.feedback_samples],
            "distance_to_target": self.distance_to_target,
            "main_error_direction": self.main_error_direction,
            "suggested_strategy": self.suggested_strategy,
            "suggested_next_action": self.suggested_next_action,
            "failure_modes": list(self.failure_modes),
        }


class FeedbackSignalExtractor:
    """
    Controller-side extractor for short-term closed-loop feedback.

    It intentionally does not read KnowledgeBase, call LLMs, or interpret long-term
    knowledge. Historical learning remains the responsibility of KnowledgeRefiner
    and AgentKnowledgeInterpreter.
    """

    def __init__(
        self,
        max_representative_failures: int = 3,
        max_diversity_samples: int = 5,
        max_feedback_samples: int = 10,
    ):
        self.max_representative_failures = max(0, int(max_representative_failures))
        self.max_diversity_samples = max(0, int(max_diversity_samples))
        self.max_feedback_samples = max(1, int(max_feedback_samples))

    def extract(
        self,
        target_property: dict[str, float],
        evaluated_samples: list[KnowledgeSample],
    ) -> FeedbackSignal:
        label_counts = self._label_counts(evaluated_samples)
        ranked = sorted(
            evaluated_samples,
            key=lambda sample: (
                sample_distance_to_target(sample, target_property),
                sample.structure_id,
            ),
        )
        best_sample = ranked[0] if ranked else None
        best_success = self._best_by_label(ranked, "success")
        best_near_miss = self._best_by_label(ranked, "near_miss")
        representative_failures = self._representative_failures(ranked)
        feedback_samples = self._feedback_samples(
            best_success=best_success,
            best_near_miss=best_near_miss,
            best_sample=best_sample,
            representative_failures=representative_failures,
            ranked=ranked,
        )
        primary_ids = {
            sample.structure_id
            for sample in (best_success, best_near_miss, best_sample)
            if sample is not None
        }
        diversity_samples = [sample for sample in feedback_samples if sample.structure_id not in primary_ids]
        next_anchor = best_success if best_success is not None else (best_near_miss or best_sample)
        distance = sample_distance_to_target(best_sample, target_property) if best_sample is not None else float("inf")
        main_error_direction = self._main_error_direction(best_sample)
        strategy = self._suggested_strategy(best_success, best_near_miss, representative_failures, best_sample)
        failure_modes = self._failure_modes(ranked, main_error_direction)

        return FeedbackSignal(
            target_property=dict(target_property),
            evaluated_count=len(evaluated_samples),
            label_counts=label_counts,
            success_found=best_success is not None,
            should_stop=best_success is not None,
            best_sample=best_sample,
            best_success=best_success,
            best_near_miss=best_near_miss,
            representative_failures=representative_failures,
            diversity_samples=diversity_samples[: self.max_diversity_samples],
            feedback_samples=feedback_samples,
            next_anchor_sample=next_anchor,
            distance_to_target=distance,
            main_error_direction=main_error_direction,
            suggested_strategy=strategy,
            suggested_next_action=strategy,
            failure_modes=failure_modes,
        )

    @staticmethod
    def _label_counts(samples: list[KnowledgeSample]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample in samples:
            counts[sample.label] = counts.get(sample.label, 0) + 1
        return counts

    @staticmethod
    def _best_by_label(ranked: list[KnowledgeSample], label: str) -> KnowledgeSample | None:
        for sample in ranked:
            if sample.label == label:
                return sample
        return None

    def _representative_failures(self, ranked: list[KnowledgeSample]) -> list[KnowledgeSample]:
        failures = [sample for sample in ranked if sample.label == "failure"]
        selected: list[KnowledgeSample] = []
        seen = set()
        for sample in failures:
            key = _sample_diversity_key(sample)
            if key in seen:
                continue
            seen.add(key)
            selected.append(sample)
            if len(selected) >= self.max_representative_failures:
                break
        return selected

    def _feedback_samples(
        self,
        best_success: KnowledgeSample | None,
        best_near_miss: KnowledgeSample | None,
        best_sample: KnowledgeSample | None,
        representative_failures: list[KnowledgeSample],
        ranked: list[KnowledgeSample],
    ) -> list[KnowledgeSample]:
        selected: list[KnowledgeSample] = []
        seen_ids: set[str] = set()
        seen_diversity: set[tuple[Any, ...]] = set()

        def add(sample: KnowledgeSample | None, require_diversity: bool = False) -> None:
            if sample is None or len(selected) >= self.max_feedback_samples:
                return
            if sample.structure_id in seen_ids:
                return
            diversity_key = _sample_diversity_key(sample)
            if require_diversity and diversity_key in seen_diversity:
                return
            selected.append(sample)
            seen_ids.add(sample.structure_id)
            seen_diversity.add(diversity_key)

        add(best_success)
        add(best_near_miss)
        add(best_sample)
        for sample in representative_failures:
            add(sample)
        for sample in ranked:
            add(sample, require_diversity=True)
            if len(selected) >= self.max_feedback_samples:
                break
        return selected

    @staticmethod
    def _main_error_direction(sample: KnowledgeSample | None) -> str:
        if sample is None or not sample.property_error:
            return ""
        key, _value = max(sample.property_error.items(), key=lambda item: _safe_float(item[1], 0.0))
        return str(key)

    @staticmethod
    def _failure_modes(ranked: list[KnowledgeSample], main_error_direction: str) -> list[str]:
        modes: list[str] = []
        for sample in ranked:
            if sample.geometry_status and sample.geometry_status != "valid":
                modes.append("invalid_geometry")
            if sample.fem_status and sample.fem_status != "success":
                modes.append("fem_failed")
        if main_error_direction:
            modes.append(f"{main_error_direction}_mismatch")
        if any(sample.label == "failure" for sample in ranked):
            modes.append("target_not_met")
        return list(dict.fromkeys(modes))

    @staticmethod
    def _suggested_strategy(
        best_success: KnowledgeSample | None,
        best_near_miss: KnowledgeSample | None,
        representative_failures: list[KnowledgeSample],
        best_sample: KnowledgeSample | None,
    ) -> str:
        if best_success is not None:
            return "stop"
        if best_near_miss is not None:
            return "exploit_near_miss"
        if representative_failures:
            return "repair_failure"
        if best_sample is not None:
            return "explore_from_best"
        return "explore"


def extract_feedback_signal(
    target_property: dict[str, float],
    evaluated_samples: list[KnowledgeSample],
) -> FeedbackSignal:
    return FeedbackSignalExtractor().extract(target_property, evaluated_samples)


__all__ = [
    "FeedbackSignal",
    "FeedbackSignalExtractor",
    "extract_feedback_signal",
    "sample_distance_to_target",
]
