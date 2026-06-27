from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.closed_loop_contracts import KnowledgeSample
from src.Scheduler.feedback import FeedbackSignalExtractor, sample_distance_to_target


def _sample(
    structure_id: str,
    label: str,
    error: float,
    symmetry: str = "P222",
    connectivity_pattern: str = "default",
    max_bars: int = 10,
    rho_target: float = 0.1,
) -> KnowledgeSample:
    target = {"stiffness_proxy": 1.0, "density_proxy": 0.1}
    evaluated = {
        "stiffness_proxy": 1.0 - error,
        "density_proxy": 0.1,
    }
    return KnowledgeSample(
        structure_id=structure_id,
        structure_path="/tmp/fake.txt",
        unit_cell_type="symmetry_expanded_truss",
        basic_unit_type="edge_face_center_19node",
        topology_type="sparse_truss",
        symmetry=symmetry,
        connectivity_pattern=connectivity_pattern,
        parameter_config={"max_bars": max_bars, "rho_target": rho_target},
        target_property=target,
        evaluated_property=evaluated,
        property_error={"stiffness_proxy": error, "density_proxy": 0.0},
        fem_status="success",
        geometry_status="valid",
        label=label,
        source="test",
        timestamp="2026-01-01T00:00:00+00:00",
        metadata={},
    )


def test_feedback_signal_stops_on_best_success():
    samples = [
        _sample("near", "near_miss", 0.2),
        _sample("success_worse", "success", 0.08),
        _sample("success_best", "success", 0.03),
    ]

    feedback = FeedbackSignalExtractor().extract(samples[0].target_property, samples)

    assert feedback.success_found is True
    assert feedback.should_stop is True
    assert feedback.best_success is not None
    assert feedback.best_success.structure_id == "success_best"
    assert feedback.next_anchor_sample is feedback.best_success
    assert sample_distance_to_target(feedback.best_success) == 0.03


def test_feedback_signal_uses_near_miss_anchor_without_success():
    samples = [
        _sample("failure", "failure", 0.5),
        _sample("near_worse", "near_miss", 0.3),
        _sample("near_best", "near_miss", 0.12),
    ]

    feedback = FeedbackSignalExtractor().extract(samples[0].target_property, samples)

    assert feedback.success_found is False
    assert feedback.should_stop is False
    assert feedback.best_near_miss is not None
    assert feedback.best_near_miss.structure_id == "near_best"
    assert feedback.best_sample is feedback.best_near_miss
    assert feedback.next_anchor_sample is feedback.best_near_miss
    assert feedback.suggested_strategy == "exploit_near_miss"


def test_feedback_signal_keeps_representative_diverse_failures():
    samples = [
        _sample("same_family_worse", "failure", 0.7, connectivity_pattern="default", max_bars=10),
        _sample("same_family_best", "failure", 0.4, connectivity_pattern="default", max_bars=10),
        _sample("other_family", "failure", 0.6, connectivity_pattern="center_reinforced", max_bars=12),
    ]

    feedback = FeedbackSignalExtractor(max_representative_failures=3).extract(samples[0].target_property, samples)

    failure_ids = [sample.structure_id for sample in feedback.representative_failures]
    assert failure_ids == ["same_family_best", "other_family"]
    assert feedback.next_anchor_sample is feedback.best_sample
    assert feedback.suggested_strategy == "repair_failure"
