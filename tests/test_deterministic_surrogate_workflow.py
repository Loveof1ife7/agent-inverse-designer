from __future__ import annotations

from typing import Any

from src.DatasetManager import DatasetManager
from src.Scheduler import DeterministicLoopConfig, DeterministicSurrogateClosedLoopSystem
from src.TargetCurvePlanner import TargetCurvePlanner
from src.closed_loop_contracts import CurveLabelPair, TargetCurvePlan, TargetCurvePlanItem
from src.curve_targets import fixed_strain_grid, is_stress_curve_target, normalize_target_property


def _curve(scale: float = 1.0) -> dict[str, Any]:
    grid = fixed_strain_grid(8)
    return {
        "type": "stress_curve",
        "strain_grid": grid,
        "stress": [scale * index for index, _ in enumerate(grid)],
    }


class FakeInverseDesigner:
    def __init__(self) -> None:
        self.finetune_calls: list[list[dict[str, Any]]] = []
        self.sampled_targets: list[dict[str, Any]] = []

    def sample_structure(self, target_property: dict[str, Any], structure_family: str | None = None) -> dict[str, Any]:
        self.sampled_targets.append(dict(target_property))
        return {
            "structure_id": "fake_structure",
            "coordinates": [[0, 0, 0], [1, 0, 0]],
            "edges": [[0, 1]],
            "target_property": dict(target_property),
            "structure_family": structure_family or "truss",
        }

    def finetune(self, rows: list[dict[str, Any]]) -> None:
        self.finetune_calls.append(list(rows))


class FakeForwardSurrogate:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self.finetune_calls: list[list[dict[str, Any]]] = []

    def predict(
        self,
        structure: dict[str, Any],
        *,
        target_property: dict[str, Any] | None = None,
        pair_id: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> CurveLabelPair:
        return CurveLabelPair(
            pair_id=pair_id or "surrogate_pair",
            structure=dict(structure),
            stress_curve=_curve(self.scale),
            label_source="surrogate",
            label_weight=0.25,
            model_consumers=("InverseDesigner",),
            target_property=dict(target_property or {}),
            provenance=dict(provenance or {}),
        )

    def finetune(self, rows: list[dict[str, Any]]) -> None:
        self.finetune_calls.append(list(rows))


class FakeHighPrecisionFEM:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self.simulate_calls: list[dict[str, Any]] = []

    def simulate(
        self,
        structure: dict[str, Any],
        *,
        target_property: dict[str, Any] | None = None,
        pair_id: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> CurveLabelPair:
        self.simulate_calls.append(dict(structure))
        return CurveLabelPair(
            pair_id=pair_id or "simulation_pair",
            structure=dict(structure),
            stress_curve=_curve(self.scale),
            label_source="simulation",
            label_weight=1.0,
            model_consumers=("InverseDesigner", "ForwardSurrogate"),
            target_property=dict(target_property or {}),
            provenance=dict(provenance or {}),
        )


class MissingFinalTargetPlanner:
    def plan(
        self,
        final_target: dict[str, Any],
        *,
        iteration: int = 0,
        batch_size: int = 1,
        samples_per_target: int = 1,
    ) -> TargetCurvePlan:
        shifted = dict(final_target)
        shifted["stress"] = [value * 0.5 for value in final_target["stress"]]
        return TargetCurvePlan(
            plan_id=f"missing_final_{iteration}",
            final_target=final_target,
            target_curves=[
                TargetCurvePlanItem(
                    target_id="shifted_only",
                    target_property=shifted,
                    strategy="lower_boundary_sweep",
                    samples=samples_per_target,
                )
            ],
            planner=type(self).__name__,
            policy="test_missing_final_target",
            iteration=iteration,
        )


def test_deterministic_loop_config_defaults_use_cold_start_triggers():
    config = DeterministicLoopConfig()

    assert config.sim_batch_size == 24
    assert config.min_forward_update_rows == 128
    assert config.min_inverse_update_rows == 64
    assert config.finetune_policy == "threshold"
    assert config.auto_update_models is True


def test_deterministic_surrogate_iteration_records_both_label_sources_without_auto_finetune(tmp_path):
    inverse = FakeInverseDesigner()
    forward = FakeForwardSurrogate()
    dataset = DatasetManager(tmp_path / "datasets")
    system = DeterministicSurrogateClosedLoopSystem(
        inverse_designer=inverse,
        forward_surrogate=forward,
        high_precision_fem=FakeHighPrecisionFEM(),
        dataset_manager=dataset,
        config=DeterministicLoopConfig(
            target_batch_size=1,
            samples_per_target=1,
            sim_batch_size=1,
            finetune_policy="manual",
        ),
        workspace_root=tmp_path / "workspace",
    )

    result = system.run_iteration(_curve(1.0), iteration=1)

    assert len(result["surrogate_pairs"]) == 1
    assert len(result["simulation_pairs"]) == 1
    assert result["surrogate_pairs"][0]["label_source"] == "surrogate"
    assert result["simulation_pairs"][0]["label_source"] == "simulation"
    assert dataset.counts() == {
        "inverse_surrogate_pairs": 1,
        "inverse_simulation_pairs": 1,
        "forward_simulation_pairs": 1,
    }
    assert result["dataset_update"]["updated_inverse_designer"] is False
    assert result["dataset_update"]["updated_forward_surrogate"] is False
    assert inverse.finetune_calls == []
    assert forward.finetune_calls == []
    assert result["accepted"] is True
    assert result["accepted_structure"]["structure_id"]


def test_target_curve_planner_generates_mixed_batch_with_exact_target_first():
    target = normalize_target_property(_curve(1.0))
    plan = TargetCurvePlanner().plan(target, iteration=2, batch_size=32, samples_per_target=1)

    assert len(plan.target_curves) == 32
    assert plan.target_curves[0].strategy == "final_target"
    assert normalize_target_property(plan.target_curves[0].target_property) == target
    strategies = {item.strategy for item in plan.target_curves}
    assert "local_perturbation" in strategies
    assert "curriculum" in strategies
    assert {"lower_boundary_sweep", "upper_boundary_sweep"} & strategies
    assert all(is_stress_curve_target(item.target_property) for item in plan.target_curves)


def test_scheduler_forces_exact_final_target_as_first_inverse_input(tmp_path):
    inverse = FakeInverseDesigner()
    target = _curve(1.0)
    system = DeterministicSurrogateClosedLoopSystem(
        inverse_designer=inverse,
        target_planner=MissingFinalTargetPlanner(),
        forward_surrogate=FakeForwardSurrogate(),
        high_precision_fem=FakeHighPrecisionFEM(),
        dataset_manager=DatasetManager(tmp_path / "datasets"),
        config=DeterministicLoopConfig(
            target_batch_size=1,
            samples_per_target=1,
            finetune_policy="manual",
        ),
        workspace_root=tmp_path / "workspace",
    )

    result = system.run_iteration(target, iteration=1)

    assert result["target_plan"]["target_curves"][0]["strategy"] == "final_target"
    assert inverse.sampled_targets[0] == normalize_target_property(target)


def test_surrogate_reject_skips_high_precision_fem(tmp_path):
    high_precision = FakeHighPrecisionFEM(scale=1.0)
    system = DeterministicSurrogateClosedLoopSystem(
        inverse_designer=FakeInverseDesigner(),
        forward_surrogate=FakeForwardSurrogate(scale=0.5),
        high_precision_fem=high_precision,
        dataset_manager=DatasetManager(tmp_path / "datasets"),
        config=DeterministicLoopConfig(
            target_batch_size=1,
            samples_per_target=1,
            finetune_policy="manual",
        ),
        workspace_root=tmp_path / "workspace",
    )

    result = system.run_iteration(_curve(1.0), iteration=1)

    assert result["surrogate_acceptance"][0]["accepted"] is False
    assert result["simulation_pairs"] == []
    assert result["accepted"] is False
    assert high_precision.simulate_calls == []


def test_surrogate_accept_waits_for_full_high_precision_batch(tmp_path):
    high_precision = FakeHighPrecisionFEM(scale=1.0)
    system = DeterministicSurrogateClosedLoopSystem(
        inverse_designer=FakeInverseDesigner(),
        forward_surrogate=FakeForwardSurrogate(scale=1.0),
        high_precision_fem=high_precision,
        dataset_manager=DatasetManager(tmp_path / "datasets"),
        config=DeterministicLoopConfig(
            target_batch_size=1,
            samples_per_target=1,
            sim_batch_size=2,
            finetune_policy="manual",
        ),
        workspace_root=tmp_path / "workspace",
    )

    result = system.run_iteration(_curve(1.0), iteration=1)

    assert result["surrogate_acceptance"][0]["accepted"] is True
    assert result["simulation_backlog_size"] == 1
    assert result["simulation_pairs"] == []
    assert result["accepted"] is False
    assert high_precision.simulate_calls == []


def test_run_breaks_after_high_precision_accepts(tmp_path):
    system = DeterministicSurrogateClosedLoopSystem(
        inverse_designer=FakeInverseDesigner(),
        forward_surrogate=FakeForwardSurrogate(scale=1.0),
        high_precision_fem=FakeHighPrecisionFEM(scale=1.0),
        dataset_manager=DatasetManager(tmp_path / "datasets"),
        config=DeterministicLoopConfig(
            target_batch_size=1,
            samples_per_target=1,
            sim_batch_size=1,
            max_iterations=3,
            finetune_policy="manual",
        ),
        workspace_root=tmp_path / "workspace",
    )

    result = system.run(_curve(1.0))

    assert result["accepted"] is True
    assert result["iterations"] == 1
    assert result["accepted_structure"]["structure_id"]


def test_dataset_manager_only_feeds_simulation_rows_to_forward_surrogate(tmp_path):
    dataset = DatasetManager(tmp_path / "datasets")
    structure = {"structure_id": "s1"}
    dataset.append_surrogate_pair(
        CurveLabelPair(
            pair_id="surrogate_1",
            structure=structure,
            stress_curve=_curve(0.5),
            label_source="surrogate",
            model_consumers=("InverseDesigner",),
        )
    )
    dataset.append_simulation_pair(
        CurveLabelPair(
            pair_id="simulation_1",
            structure=structure,
            stress_curve=_curve(1.0),
            label_source="simulation",
            model_consumers=("InverseDesigner", "ForwardSurrogate"),
        )
    )

    inverse_rows = dataset.inverse_training_rows()
    forward_rows = dataset.forward_training_rows()

    assert [row["label_source"] for row in inverse_rows] == ["surrogate", "simulation"]
    assert [row["label_source"] for row in forward_rows] == ["simulation"]


def test_dataset_manager_uses_weighted_inverse_threshold_and_simulation_forward_threshold(tmp_path):
    dataset = DatasetManager(tmp_path / "datasets")
    inverse = FakeInverseDesigner()
    forward = FakeForwardSurrogate()

    for index in range(3):
        dataset.append_surrogate_pair(
            CurveLabelPair(
                pair_id=f"surrogate_{index}",
                structure={"structure_id": f"surr_{index}"},
                stress_curve=_curve(0.5),
                label_source="surrogate",
                model_consumers=("InverseDesigner",),
            )
        )

    summary = dataset.update_models(
        inverse_designer=inverse,
        forward_surrogate=forward,
        min_inverse_rows=1,
        min_forward_rows=1,
    )

    assert summary.inverse_training_rows == 3
    assert summary.inverse_training_weight == 0.75
    assert summary.forward_training_rows == 0
    assert summary.updated_inverse_designer is False
    assert summary.updated_forward_surrogate is False
    assert inverse.finetune_calls == []
    assert forward.finetune_calls == []

    dataset.append_surrogate_pair(
        CurveLabelPair(
            pair_id="surrogate_3",
            structure={"structure_id": "surr_3"},
            stress_curve=_curve(0.5),
            label_source="surrogate",
            model_consumers=("InverseDesigner",),
        )
    )

    summary = dataset.update_models(
        inverse_designer=inverse,
        forward_surrogate=forward,
        min_inverse_rows=1,
        min_forward_rows=1,
    )

    assert summary.inverse_training_rows == 4
    assert summary.inverse_training_weight == 1.0
    assert summary.forward_training_rows == 0
    assert summary.updated_inverse_designer is True
    assert summary.updated_forward_surrogate is False
    assert [row["label_source"] for row in inverse.finetune_calls[0]] == ["surrogate"] * 4
    assert forward.finetune_calls == []

    dataset.append_simulation_pair(
        CurveLabelPair(
            pair_id="simulation_1",
            structure={"structure_id": "sim_1"},
            stress_curve=_curve(1.0),
            label_source="simulation",
            model_consumers=("InverseDesigner", "ForwardSurrogate"),
        )
    )

    summary = dataset.update_models(
        inverse_designer=inverse,
        forward_surrogate=forward,
        min_inverse_rows=1,
        min_forward_rows=1,
    )

    assert summary.inverse_training_rows == 1
    assert summary.inverse_training_weight == 1.0
    assert summary.forward_training_rows == 1
    assert summary.updated_inverse_designer is True
    assert summary.updated_forward_surrogate is True
    assert inverse.finetune_calls[-1][0]["label_source"] == "simulation"
    assert forward.finetune_calls[-1][0]["label_source"] == "simulation"
