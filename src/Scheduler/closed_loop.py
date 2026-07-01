from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..DatasetManager import DatasetManager
from ..ForwardSurrogate import ForwardSurrogate
from ..HighPrecisionFEM import HighPrecisionFEM
from ..TargetCurvePlanner import TargetCurvePlanner
from ..curve_targets import normalize_target_property, stress_curve_error_metrics
from ..closed_loop_contracts import (
    CurveLabelPair,
    DatasetUpdateSummary,
    TargetCurvePlan,
    TargetCurvePlanItem,
)
from .events import EventStream
from .experiment import dump_experiment_manifest, make_experiment_paths, make_task_id

@dataclass(frozen=True)
class DeterministicLoopConfig:
    """Policy knobs for the simplified deterministic/surrogate workflow.

    Defaults use the cold-start closed-loop policy: full-batch CPU FEM,
    threshold-triggered GPU finetunes, and weighted inverse labels.
    """

    target_batch_size: int = 1
    samples_per_target: int = 1
    max_iterations: int = 1
    fast_queue_enabled: bool = True
    slow_queue_enabled: bool = True
    queue_policy: str = "serial_queues"
    finetune_policy: str = "threshold"
    min_inverse_update_rows: int = 64
    min_forward_update_rows: int = 128
    surrogate_top_k: int = 6
    sim_batch_size: int = 24
    finetune_min_new_rows: int = 100
    acceptance_curve_nmae: float = 0.05
    usable_training_curve_nmae: float = 0.20
    inverse_num_runs: int = 64
    inverse_batch_size: int = 32
    inverse_top_k: int = 32
    forward_batch_size: int = 256
    forward_gpu_workers: int = 5
    forward_compact: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_batch_size", max(1, int(self.target_batch_size)))
        object.__setattr__(self, "samples_per_target", max(1, int(self.samples_per_target)))
        object.__setattr__(self, "max_iterations", max(1, int(self.max_iterations)))
        object.__setattr__(self, "min_inverse_update_rows", max(1, int(self.min_inverse_update_rows)))
        object.__setattr__(self, "min_forward_update_rows", max(1, int(self.min_forward_update_rows)))
        object.__setattr__(self, "surrogate_top_k", max(1, int(self.surrogate_top_k)))
        object.__setattr__(self, "sim_batch_size", max(1, int(self.sim_batch_size)))
        object.__setattr__(self, "finetune_min_new_rows", max(1, int(self.finetune_min_new_rows)))
        object.__setattr__(self, "acceptance_curve_nmae", max(0.0, float(self.acceptance_curve_nmae)))
        object.__setattr__(self, "usable_training_curve_nmae", max(0.0, float(self.usable_training_curve_nmae)))
        object.__setattr__(self, "inverse_num_runs", max(1, int(self.inverse_num_runs)))
        object.__setattr__(self, "inverse_batch_size", max(1, int(self.inverse_batch_size)))
        object.__setattr__(self, "inverse_top_k", max(1, int(self.inverse_top_k)))
        object.__setattr__(self, "forward_batch_size", max(1, int(self.forward_batch_size)))
        object.__setattr__(self, "forward_gpu_workers", max(1, int(self.forward_gpu_workers)))

    @property
    def auto_update_models(self) -> bool:
        return self.finetune_policy.strip().lower() in {"auto", "threshold"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_batch_size": self.target_batch_size,
            "samples_per_target": self.samples_per_target,
            "max_iterations": self.max_iterations,
            "fast_queue_enabled": self.fast_queue_enabled,
            "slow_queue_enabled": self.slow_queue_enabled,
            "queue_policy": self.queue_policy,
            "finetune_policy": self.finetune_policy,
            "min_inverse_update_rows": self.min_inverse_update_rows,
            "min_forward_update_rows": self.min_forward_update_rows,
            "surrogate_top_k": self.surrogate_top_k,
            "sim_batch_size": self.sim_batch_size,
            "finetune_min_new_rows": self.finetune_min_new_rows,
            "acceptance_curve_nmae": self.acceptance_curve_nmae,
            "usable_training_curve_nmae": self.usable_training_curve_nmae,
            "inverse_num_runs": self.inverse_num_runs,
            "inverse_batch_size": self.inverse_batch_size,
            "inverse_top_k": self.inverse_top_k,
            "forward_batch_size": self.forward_batch_size,
            "forward_gpu_workers": self.forward_gpu_workers,
            "forward_compact": self.forward_compact,
            "notes": dict(self.notes),
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default



class DeterministicSurrogateClosedLoopSystem:
    """Scheduler for the simplified TargetCurvePlanner -> dual forward workflow.

    This class is the migration target for the new workflow. It deliberately
    avoids AgentExplorer, KnowledgeBase, KnowledgeRefiner, FeedbackSignal, and
    RawExperimentStore. Concrete finetune triggers and true async execution are
    policy hooks, not hard-coded defaults.
    """

    def __init__(
        self,
        *,
        inverse_designer: Any,
        target_planner: TargetCurvePlanner | None = None,
        forward_surrogate: ForwardSurrogate | None = None,
        high_precision_fem: HighPrecisionFEM | None = None,
        dataset_manager: DatasetManager | None = None,
        config: DeterministicLoopConfig | dict[str, Any] | None = None,
        workspace_root: str | Path | None = None,
        task_id: str | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root or "workspace").resolve()
        self.task_id = task_id or make_task_id()
        self.experiment_paths = make_experiment_paths(self.workspace_root, self.task_id)
        self.event_stream = EventStream(self.task_id, self.experiment_paths.events_dir, mirror_path=log_path)
        self.inverse_designer = inverse_designer
        self.target_planner = target_planner or TargetCurvePlanner()
        self.forward_surrogate = forward_surrogate or ForwardSurrogate(workspace_root=self.workspace_root)
        self.high_precision_fem = high_precision_fem or HighPrecisionFEM(workspace_root=self.workspace_root)
        self.dataset_manager = dataset_manager or DatasetManager(Path(self.experiment_paths.root_dir) / "datasets")
        self.config = config if isinstance(config, DeterministicLoopConfig) else DeterministicLoopConfig(**dict(config or {}))
        self.simulation_backlog: list[dict[str, Any]] = []
        self._simulation_backlog_ids: set[str] = set()
        dump_experiment_manifest(
            self.experiment_paths,
            {
                "layout_version": "deterministic_surrogate_v1",
                "task_id": self.task_id,
                "workflow": "TargetCurvePlanner -> InverseDesigner -> ForwardSurrogate/HighPrecisionFEM -> DatasetManager",
                "config": self.config.to_dict(),
                "dataset_manager": self.dataset_manager.to_dict(),
                "event_log": str(Path(self.event_stream.path).resolve()),
                "event_log_mirror": str(Path(log_path).resolve()) if log_path else None,
            },
        )

    def _emit(self, logs: list[dict[str, Any]], stage: str, status: str, payload: dict[str, Any]) -> None:
        event = self.event_stream.emit(stage=stage, status=status, payload=payload)
        logs.append(event.to_dict())

    def plan_targets(self, final_target: dict[str, Any], *, iteration: int) -> TargetCurvePlan:
        plan = self.target_planner.plan(
            normalize_target_property(final_target),
            iteration=iteration,
            batch_size=self.config.target_batch_size,
            samples_per_target=self.config.samples_per_target,
        )
        return self._ensure_final_target_in_plan(plan, final_target)

    def sample_structures(self, plan: TargetCurvePlan) -> list[dict[str, Any]]:
        schedule = plan.to_target_schedule()
        if hasattr(self.inverse_designer, "sample_schedule"):
            records = self.inverse_designer.sample_schedule(schedule)
        else:
            records = []
            for step_index, item in enumerate(schedule.scheduled_targets, start=1):
                for sample_index in range(1, item.samples + 1):
                    structure = self.inverse_designer.sample_structure(item.target_property)
                    records.append(
                        {
                            "schedule_id": schedule.schedule_id,
                            "schedule_step": step_index,
                            "sample_index": sample_index,
                            "scheduled_target": dict(item.target_property),
                            "schedule_item": item.to_dict(),
                            "final_target": dict(schedule.final_target),
                            "structure": structure,
                            "status": "sampled" if structure else "no_candidate",
                        }
                    )
        return [self._normalize_structure_record(plan, record) for record in records]

    def process_fast_queue(self, records: list[dict[str, Any]], *, iteration: int) -> list[CurveLabelPair]:
        if not self.config.fast_queue_enabled:
            return []
        candidate_records = [record for record in records if record.get("structure")]
        if hasattr(self.forward_surrogate, "predict_many"):
            raw_pairs = self.forward_surrogate.predict_many(
                [dict(record["structure"]) for record in candidate_records],
                target_properties=[dict(record.get("scheduled_target") or {}) for record in candidate_records],
                pair_prefix=f"surr_iter{iteration:03d}",
                provenance={
                    "iteration": iteration,
                    "queue": "fast_surrogate",
                },
            )
        else:
            raw_pairs = []
            for record in candidate_records:
                structure = dict(record.get("structure") or {})
                raw_pairs.append(
                    self.forward_surrogate.predict(
                        structure,
                        target_property=dict(record.get("scheduled_target") or {}),
                        pair_id=f"surr_iter{iteration:03d}_{structure.get('structure_id', 'structure')}",
                        provenance={
                            "iteration": iteration,
                            "queue": "fast_surrogate",
                            "target_plan": dict(record.get("target_plan") or {}),
                            "schedule_record": self._record_without_structure(record),
                        },
                    )
                )
        pairs: list[CurveLabelPair] = []
        for pair in raw_pairs:
            pairs.append(self.dataset_manager.append_surrogate_pair(pair))
        return pairs

    def process_slow_queue(
        self,
        records: list[dict[str, Any]],
        *,
        iteration: int,
        allowed_structure_ids: set[str] | None = None,
    ) -> list[CurveLabelPair]:
        if not self.config.slow_queue_enabled:
            return []
        selected_records = []
        for record in records:
            structure = dict(record.get("structure") or {})
            structure_id = str(structure.get("structure_id") or "")
            if structure and (allowed_structure_ids is None or structure_id in allowed_structure_ids):
                selected_records.append(record)
        if hasattr(self.high_precision_fem, "simulate_many"):
            raw_pairs = self.high_precision_fem.simulate_many(
                [dict(record["structure"]) for record in selected_records],
                target_property=None,
                provenance={
                    "iteration": iteration,
                    "queue": "slow_high_precision_fem",
                },
            )
        else:
            raw_pairs = []
            for record in selected_records:
                structure = dict(record.get("structure") or {})
                raw_pairs.append(
                    self.high_precision_fem.simulate(
                        structure,
                        target_property=dict(record.get("scheduled_target") or {}),
                        pair_id=f"sim_iter{iteration:03d}_{structure.get('structure_id', 'structure')}",
                        provenance={
                            "iteration": iteration,
                            "queue": "slow_high_precision_fem",
                            "target_plan": dict(record.get("target_plan") or {}),
                            "schedule_record": self._record_without_structure(record),
                        },
                    )
                )
        pairs: list[CurveLabelPair] = []
        for pair in raw_pairs:
            pairs.append(self.dataset_manager.append_simulation_pair(pair))
        return pairs

    def maybe_update_models(self) -> DatasetUpdateSummary:
        if not self.config.auto_update_models:
            counts = self.dataset_manager.counts()
            return DatasetUpdateSummary(
                inverse_surrogate_pairs=counts["inverse_surrogate_pairs"],
                inverse_simulation_pairs=counts["inverse_simulation_pairs"],
                forward_simulation_pairs=counts["forward_simulation_pairs"],
                inverse_training_rows=0,
                inverse_training_weight=0.0,
                forward_training_rows=0,
                updated_inverse_designer=False,
                updated_forward_surrogate=False,
            )
        return self.dataset_manager.update_models(
            inverse_designer=self.inverse_designer,
            forward_surrogate=self.forward_surrogate,
            min_inverse_rows=self.config.min_inverse_update_rows,
            min_forward_rows=self.config.min_forward_update_rows,
        )

    def run_iteration(self, final_target: dict[str, Any], *, iteration: int = 1) -> dict[str, Any]:
        """Run one deterministic workflow iteration.

        This method executes the two logical queues in a serial order today.
        True async scheduling should be introduced only after the queue policy is
        aligned.
        """

        if self.config.queue_policy != "serial_queues":
            raise NotImplementedError(
                "Only queue_policy='serial_queues' is implemented. Async policy needs explicit alignment."
            )

        logs: list[dict[str, Any]] = []
        target = normalize_target_property(final_target)
        self._emit(
            logs,
            stage="deterministic_loop",
            status="iteration_started",
            payload={
                "iteration": iteration,
                "target_property": target,
                "config": self.config.to_dict(),
            },
        )
        plan = self.plan_targets(target, iteration=iteration)
        self._emit(
            logs,
            stage="target_curve_planner",
            status="planned",
            payload=plan.to_dict(),
        )

        records = self.sample_structures(plan)
        self._emit(
            logs,
            stage="inverse_designer",
            status="sampled_structures",
            payload={
                "iteration": iteration,
                "plan_id": plan.plan_id,
                "requested_targets": len(plan.target_curves),
                "record_count": len(records),
                "sampled_count": sum(1 for record in records if record.get("structure")),
                "no_candidate_count": sum(1 for record in records if not record.get("structure")),
            },
        )

        surrogate_pairs = self.process_fast_queue(records, iteration=iteration)
        surrogate_acceptance = [
            self._pair_acceptance(pair, target)
            for pair in surrogate_pairs
        ]
        ranked_surrogate = sorted(surrogate_acceptance, key=lambda item: _safe_float(item.get("curve_nmae"), float("inf")))
        top_k_surrogate = ranked_surrogate[: self.config.surrogate_top_k]
        top_k_ids = {str(item.get("structure_id") or "") for item in top_k_surrogate}
        surrogate_accepted_count = sum(1 for item in ranked_surrogate if item["accepted"])
        self._append_simulation_backlog(records, top_k_ids=top_k_ids, iteration=iteration, surrogate_acceptance=surrogate_acceptance)
        self._emit(
            logs,
            stage="forward_surrogate",
            status="completed" if self.config.fast_queue_enabled else "disabled",
            payload={
                "iteration": iteration,
                "pair_count": len(surrogate_pairs),
                "label_source": "surrogate",
                "accepted_count": surrogate_accepted_count,
                "top_k_count": len(top_k_ids),
                "top_k_structure_ids": list(top_k_ids),
                "acceptance": surrogate_acceptance,
            },
        )

        if len(self.simulation_backlog) >= self.config.sim_batch_size:
            simulation_records = self._pop_simulation_batch(
                min_count=self.config.sim_batch_size,
                max_count=self.config.sim_batch_size,
            )
        else:
            simulation_records = []
        simulation_pairs = self.process_slow_queue(
            simulation_records,
            iteration=iteration,
            allowed_structure_ids=None,
        )
        simulation_acceptance = [
            self._pair_acceptance(pair, target)
            for pair in simulation_pairs
        ]
        accepted_pair = next(
            (
                pair
                for pair, acceptance in zip(simulation_pairs, simulation_acceptance)
                if acceptance["accepted"]
            ),
            None,
        )
        self._emit(
            logs,
            stage="high_precision_fem",
            status="completed" if self.config.slow_queue_enabled else "disabled",
            payload={
                "iteration": iteration,
                "pair_count": len(simulation_pairs),
                "label_source": "simulation",
                "candidate_count": len(simulation_records),
                "backlog_remaining": len(self.simulation_backlog),
                "accepted": accepted_pair is not None,
                "acceptance": simulation_acceptance,
            },
        )

        update_summary = self.maybe_update_models()
        self._emit(
            logs,
            stage="dataset_manager",
            status="model_update_skipped" if not self.config.auto_update_models else "model_update_checked",
            payload={
                "iteration": iteration,
                "finetune_policy": self.config.finetune_policy,
                "update_summary": update_summary.to_dict(),
                "dataset_manager": self.dataset_manager.to_dict(),
            },
        )
        return {
            "iteration": iteration,
            "target_property": target,
            "target_plan": plan.to_dict(),
            "structure_records": [self._record_without_structure(record) | {"has_structure": bool(record.get("structure"))} for record in records],
            "surrogate_pairs": [pair.to_dict() for pair in surrogate_pairs],
            "simulation_pairs": [pair.to_dict() for pair in simulation_pairs],
            "surrogate_acceptance": surrogate_acceptance,
            "surrogate_ranking": ranked_surrogate,
            "surrogate_top_k": top_k_surrogate,
            "simulation_acceptance": simulation_acceptance,
            "simulation_backlog_size": len(self.simulation_backlog),
            "accepted": accepted_pair is not None,
            "accepted_structure": dict(accepted_pair.structure) if accepted_pair is not None else {},
            "accepted_pair": accepted_pair.to_dict() if accepted_pair is not None else {},
            "dataset_update": update_summary.to_dict(),
            "events": logs,
        }

    def run(self, final_target: dict[str, Any], max_iterations: int | None = None) -> dict[str, Any]:
        iterations = max(1, int(max_iterations or self.config.max_iterations))
        results = []
        accepted_structure: dict[str, Any] = {}
        accepted_pair: dict[str, Any] = {}
        for iteration in range(1, iterations + 1):
            result = self.run_iteration(final_target, iteration=iteration)
            results.append(result)
            if result.get("accepted"):
                accepted_structure = dict(result.get("accepted_structure") or {})
                accepted_pair = dict(result.get("accepted_pair") or {})
                break
        return {
            "task_id": self.task_id,
            "workflow": "deterministic_surrogate",
            "iterations": len(results),
            "accepted": bool(accepted_structure),
            "accepted_structure": accepted_structure,
            "accepted_pair": accepted_pair,
            "results": results,
            "dataset_manager": self.dataset_manager.to_dict(),
            "experiment_paths": self.experiment_paths.to_dict(),
        }

    @staticmethod
    def _record_without_structure(record: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in record.items() if key != "structure"}

    def _append_simulation_backlog(
        self,
        records: list[dict[str, Any]],
        *,
        top_k_ids: set[str],
        iteration: int,
        surrogate_acceptance: list[dict[str, Any]],
    ) -> None:
        acceptance_by_id = {
            str(item.get("structure_id") or ""): dict(item)
            for item in surrogate_acceptance
        }
        for record in records:
            structure = dict(record.get("structure") or {})
            structure_id = str(structure.get("structure_id") or "")
            if not structure_id or structure_id not in top_k_ids or structure_id in self._simulation_backlog_ids:
                continue
            queued = dict(record)
            queued["simulation_queue"] = {
                "queued_at_iteration": int(iteration),
                "selection_reason": "surrogate_top_k",
                "surrogate_acceptance": acceptance_by_id.get(structure_id, {}),
            }
            self.simulation_backlog.append(queued)
            self._simulation_backlog_ids.add(structure_id)

    def _pop_simulation_batch(
        self,
        *,
        min_count: int,
        max_count: int,
        preferred_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if len(self.simulation_backlog) < max(1, int(min_count)):
            return []
        preferred_ids = set(preferred_ids or set())
        limit = max(1, int(max_count))
        selected: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []

        def record_id(item: dict[str, Any]) -> str:
            return str(dict(item.get("structure") or {}).get("structure_id") or "")

        for record in self.simulation_backlog:
            if len(selected) < limit and preferred_ids and record_id(record) in preferred_ids:
                selected.append(record)
            else:
                remaining.append(record)
        if len(selected) < limit:
            still_remaining: list[dict[str, Any]] = []
            for record in remaining:
                if len(selected) < limit:
                    selected.append(record)
                else:
                    still_remaining.append(record)
            remaining = still_remaining

        self.simulation_backlog = remaining
        self._simulation_backlog_ids = {record_id(record) for record in remaining if record_id(record)}
        return selected

    @staticmethod
    def _normalize_structure_record(plan: TargetCurvePlan, record: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(record)
        structure = dict(normalized.get("structure") or {})
        if structure:
            original_structure_id = str(structure.get("structure_id") or structure.get("sample_id") or "inverse_structure")
            unique_structure_id = (
                f"{original_structure_id}__{plan.plan_id}_"
                f"s{int(normalized.get('schedule_step') or 0):02d}_"
                f"n{int(normalized.get('sample_index') or 0):02d}"
            )
            structure["original_structure_id"] = original_structure_id
            structure["structure_id"] = unique_structure_id
            structure["target_plan"] = plan.to_dict()
            structure["scheduled_target"] = dict(normalized.get("scheduled_target") or {})
            structure["final_target"] = dict(plan.final_target)
            normalized["structure"] = structure
        normalized["target_plan"] = plan.to_dict()
        normalized.setdefault("final_target", dict(plan.final_target))
        return normalized

    @staticmethod
    def _ensure_final_target_in_plan(plan: TargetCurvePlan, final_target: dict[str, Any]) -> TargetCurvePlan:
        target = normalize_target_property(final_target)
        for item in plan.target_curves:
            if normalize_target_property(item.target_property) == target:
                ordered = [item] + [other for other in plan.target_curves if other is not item]
                return TargetCurvePlan(
                    plan_id=plan.plan_id,
                    final_target=target,
                    target_curves=ordered,
                    planner=plan.planner,
                    policy=plan.policy,
                    iteration=plan.iteration,
                )
        direct_item = TargetCurvePlanItem(
            target_id=f"{plan.plan_id}_final_target",
            target_property=target,
            strategy="final_target",
            samples=1,
            planner_meta={
                "forced_by_scheduler": True,
                "reason": "Every deterministic plan must include the exact final target.",
            },
        )
        return TargetCurvePlan(
            plan_id=plan.plan_id,
            final_target=target,
            target_curves=[direct_item, *plan.target_curves],
            planner=plan.planner,
            policy=plan.policy,
            iteration=plan.iteration,
        )

    def _pair_acceptance(self, pair: CurveLabelPair, target: dict[str, Any]) -> dict[str, Any]:
        error = stress_curve_error_metrics(target, pair.stress_curve)
        curve_nmae = _safe_float(error.get("curve_nmae"), float("inf"))
        return {
            "pair_id": pair.pair_id,
            "structure_id": str(pair.structure.get("structure_id") or ""),
            "label_source": pair.label_source,
            "curve_error": error,
            "curve_nmae": curve_nmae,
            "threshold": self.config.acceptance_curve_nmae,
            "accepted": bool(curve_nmae <= self.config.acceptance_curve_nmae),
        }


DeterministicSurrogateScheduler = DeterministicSurrogateClosedLoopSystem
