from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..AgentExplorer import AgentExplorer
from ..DatagenFEMEvaluator import DatagenFEMEvaluator, curve_aware_property_error
from ..DatasetManager import DatasetManager
from ..ForwardSurrogate import ForwardSurrogate
from ..HighPrecisionFEM import HighPrecisionFEM
from ..InverseDesigner import InverseDesigner
from ..KnowledgeBase import KnowledgeBase
from ..KnowledgeRefiner import KnowledgeRefiner
from ..ExperimentStore import RawExperimentStore
from ..TargetCurvePlanner import TargetCurvePlanner
from ..curve_targets import normalize_target_property, stress_curve_error_metrics
from ..closed_loop_contracts import (
    ClosedLoopResult,
    CurveLabelPair,
    DatagenConfig,
    DatasetUpdateSummary,
    KnowledgeSample,
    MetaCandidate,
    Observation,
    SchedulerContext,
    TargetCurvePlan,
    TargetCurvePlanItem,
    TargetSchedule,
    TargetScheduleItem,
    TargetScheduleProposal,
    TaskStatus,
)
from .events import EventStream
from .experiment import dump_experiment_manifest, make_experiment_paths, make_task_id
from .feedback import FeedbackSignalExtractor

CLOSED_LOOP_DEFAULT_RETRAIN_TRIGGER = 32


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


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    q = min(max(q, 0.0), 1.0)
    index = q * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _property_error(target_property: dict[str, float], evaluated_property: dict[str, float]) -> dict[str, float]:
    return curve_aware_property_error(target_property, evaluated_property)


def _aggregate_error_value(payload: Any) -> float | None:
    if isinstance(payload, (int, float)):
        return abs(float(payload))
    if isinstance(payload, dict):
        if "curve_nmae" in payload:
            try:
                return abs(float(payload["curve_nmae"]))
            except (TypeError, ValueError):
                pass
        values = []
        for key, value in payload.items():
            if str(key).lower() in {"label", "status"}:
                continue
            try:
                values.append(abs(float(value)))
            except (TypeError, ValueError):
                continue
        if values:
            return sum(values) / len(values)
    return None


class StructureDiscoverySystem:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        inverse_designer: InverseDesigner,
        agent_explorer: AgentExplorer,
        evaluator: DatagenFEMEvaluator,
        retrain_trigger: int = CLOSED_LOOP_DEFAULT_RETRAIN_TRIGGER,
        workspace_root: str | Path | None = None,
        task_id: str | None = None,
        log_path: str | Path | None = None,
        agent_batch_size: int = 77,
        experiment_budget: int = 1,
    ):
        self.knowledge_base = knowledge_base
        self.inverse_designer = inverse_designer
        self.agent_explorer = agent_explorer
        self.generator_evaluator = evaluator
        self.knowledge_refiner = KnowledgeRefiner()
        self.feedback_extractor = FeedbackSignalExtractor()
        self.retrain_trigger = int(retrain_trigger)
        self.agent_batch_size = max(1, int(agent_batch_size))
        self.experiment_budget = max(1, int(experiment_budget))
        self.min_iterations_before_success = max(
            0,
            int(os.getenv("CLOSED_LOOP_MIN_ITERATIONS_BEFORE_SUCCESS", "0") or 0),
        )
        self.workspace_root = Path(workspace_root or ".").resolve()
        self.task_id = task_id or make_task_id()
        self.experiment_paths = make_experiment_paths(self.workspace_root, self.task_id)
        self.raw_experiment_store = RawExperimentStore(self.experiment_paths.root_dir)
        self.event_stream = EventStream(self.task_id, self.experiment_paths.events_dir, mirror_path=log_path)
        self._last_training_count = 0
        dump_experiment_manifest(
            self.experiment_paths,
            {
                "layout_version": "flat_core_v1",
                "task_id": self.task_id,
                "retrain_trigger": self.retrain_trigger,
                "agent_batch_size": self.agent_batch_size,
                "experiment_budget": self.experiment_budget,
                "min_iterations_before_success": self.min_iterations_before_success,
                "raw_experiment_store": str(self.raw_experiment_store.path.resolve()),
                "event_log": str(Path(self.event_stream.path).resolve()),
                "event_log_mirror": str(Path(log_path).resolve()) if log_path else None,
            },
        )

    def _success_stop_allowed(self, iteration: int) -> bool:
        return int(iteration) >= self.min_iterations_before_success

    def _report_each_iteration_enabled(self) -> bool:
        return str(os.getenv("CLOSED_LOOP_REPORT_EACH_ITERATION", "")).strip().lower() in {"1", "true", "yes", "on"}

    def _maybe_write_surrogate_gt_report(self, iteration: int, logs: list[dict[str, Any]]) -> None:
        if not self._report_each_iteration_enabled():
            return
        try:
            from .surrogate_gt_report import generate_surrogate_gt_report

            report = generate_surrogate_gt_report(
                self.workspace_root,
                kb_path=Path(self.workspace_root) / "knowledge.sqlite",
                output_dir=Path(self.workspace_root) / "analysis",
            )
            self._emit(
                logs,
                stage="analysis",
                status="surrogate_gt_report_updated",
                payload={
                    "iteration": iteration,
                    "summary_path": report.summary_path,
                    "figure_dir": report.figure_dir,
                    "best_utility_error": report.best_utility_error,
                    "mean_realization_error": report.mean_realization_error,
                    "figures": report.figures,
                },
            )
        except Exception as exc:
            self._emit(
                logs,
                stage="analysis",
                status="surrogate_gt_report_failed",
                payload={"iteration": iteration, "error": f"{type(exc).__name__}: {exc}"},
            )

    def _emit(self, logs: list[dict[str, Any]], stage: str, status: str, payload: dict[str, Any]) -> None:
        event = self.event_stream.emit(stage=stage, status=status, payload=payload)
        logs.append(event.to_dict())

    def _dataset_distribution(self) -> dict[str, Any]:
        samples = self.knowledge_base.list_samples()
        stiffness = [_safe_float(sample.evaluated_property.get("stiffness_proxy")) for sample in samples if sample.evaluated_property]
        density = [_safe_float(sample.evaluated_property.get("density_proxy")) for sample in samples if sample.evaluated_property]
        label_counts: dict[str, int] = {}
        group_counts: dict[str, int] = {}
        for sample in samples:
            label_counts[sample.label] = label_counts.get(sample.label, 0) + 1
            group_counts[sample.symmetry] = group_counts.get(sample.symmetry, 0) + 1
        return {
            "total_samples": len(samples),
            "label_counts": label_counts,
            "group_counts": group_counts,
            "stiffness_proxy": self._distribution_summary(stiffness),
            "density_proxy": self._distribution_summary(density),
        }

    @staticmethod
    def _distribution_summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        return {
            "min": min(values),
            "p25": _percentile(values, 0.25),
            "p50": _percentile(values, 0.50),
            "p75": _percentile(values, 0.75),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    @staticmethod
    def _sample_brief(sample: KnowledgeSample | None) -> dict[str, Any]:
        if sample is None:
            return {}
        return {
            "structure_id": sample.structure_id,
            "group": sample.symmetry,
            "symmetry": sample.symmetry,
            "unit_cell_type": sample.unit_cell_type,
            "basic_unit_type": sample.basic_unit_type,
            "topology_type": sample.topology_type,
            "connectivity_pattern": sample.connectivity_pattern,
            "source": sample.source,
            "label": sample.label,
            "evaluated_property": dict(sample.evaluated_property),
            "property_error": dict(sample.property_error),
            "parameter_config": {
                "max_bars": sample.parameter_config.get("max_bars"),
                "rho_target": sample.parameter_config.get("rho_target"),
                "density_range": sample.parameter_config.get("density_range"),
                "connectivity_pattern": sample.connectivity_pattern,
            },
        }

    @staticmethod
    def _candidate_summary(candidate: KnowledgeSample | None, candidate_eval: dict[str, Any] | None, target_property: dict[str, float]) -> dict[str, Any]:
        if candidate is None:
            return {
                "target_property": dict(target_property),
                "candidate_found": False,
                "satisfies_target": False,
            }
        return {
            "target_property": dict(target_property),
            "candidate_found": True,
            "candidate": {
                "structure_id": candidate.structure_id,
                "group": candidate.symmetry,
                "evaluated_property": dict(candidate.evaluated_property),
                "parameter_config": {
                    "max_bars": candidate.parameter_config.get("max_bars"),
                    "rho_target": candidate.parameter_config.get("rho_target"),
                    "density_range": candidate.parameter_config.get("density_range"),
                    "connectivity_pattern": candidate.connectivity_pattern,
                },
            },
            "prediction_result": dict(candidate_eval or {}),
            "satisfies_target": bool(candidate_eval and candidate_eval.get("label") == "success"),
        }

    @staticmethod
    def _proposal_summary(
        iteration: int,
        proposal: DatagenConfig,
        target_property: dict[str, float],
        failed_candidate: KnowledgeSample | None,
    ) -> dict[str, Any]:
        return {
            "iteration": iteration,
            "target_property": dict(target_property),
            "based_on": failed_candidate.structure_id if failed_candidate else "",
            "suggestion_id": proposal.suggestion_id,
            "source": proposal.source,
            "group": proposal.group,
            "structure_parameters": {
                "symmetry": proposal.symmetry,
                "basic_unit_type": proposal.basic_unit_type,
                "unit_cell_type": proposal.unit_cell_type,
                "topology_type": proposal.topology_type,
                "connectivity_pattern": proposal.connectivity_pattern,
                "max_bars": proposal.max_bars,
                "rho_target": proposal.rho_target,
                "density_range": list(proposal.density_range),
                "num_samples": proposal.num_samples,
            },
            "hypothesis": proposal.hypothesis,
            "reason": proposal.reason,
            "expected_property": dict(proposal.expected_property),
            "confidence": proposal.confidence,
            "exploration_strategy": proposal.exploration_strategy,
        }

    @staticmethod
    def _evaluation_summary(
        iteration: int,
        target_property: dict[str, float],
        proposal: DatagenConfig,
        structures: list[dict[str, Any]],
        evaluated_samples: list[KnowledgeSample],
        selected: list[KnowledgeSample],
    ) -> dict[str, Any]:
        label_counts: dict[str, int] = {}
        for sample in selected:
            label_counts[sample.label] = label_counts.get(sample.label, 0) + 1

        best_sample = None
        best_score = None
        for sample in selected:
            score = sum(_safe_float(value) for value in sample.property_error.values())
            if best_score is None or score < best_score:
                best_score = score
                best_sample = sample

        satisfies_target = bool(best_sample and best_sample.label == "success")
        return {
            "iteration": iteration,
            "hypothesis": proposal.hypothesis,
            "target_property": dict(target_property),
            "generated_count": len(structures),
            "evaluated_count": len(evaluated_samples),
            "selected_count": len(selected),
            "label_counts": label_counts,
            "best_result": {
                "structure_id": best_sample.structure_id if best_sample else "",
                "label": best_sample.label if best_sample else "",
                "evaluated_property": dict(best_sample.evaluated_property) if best_sample else {},
                "property_error": dict(best_sample.property_error) if best_sample else {},
            },
            "satisfies_target": satisfies_target,
        }

    @staticmethod
    def _target_schedule_evaluation_summary(
        iteration: int,
        target_property: dict[str, float],
        schedule: TargetSchedule,
        structures: list[dict[str, Any]],
        evaluated_samples: list[KnowledgeSample],
        selected: list[KnowledgeSample],
    ) -> dict[str, Any]:
        label_counts: dict[str, int] = {}
        strategy_counts: dict[str, int] = {}
        scheduled_error_scores: list[float] = []
        for sample in selected:
            label_counts[sample.label] = label_counts.get(sample.label, 0) + 1
        for item in schedule.scheduled_targets:
            strategy_counts[item.strategy] = strategy_counts.get(item.strategy, 0) + 1
        for sample in evaluated_samples:
            scheduled_error = dict((sample.metadata or {}).get("error_to_scheduled_target") or {})
            if scheduled_error:
                scheduled_error_scores.append(max(_safe_float(value) for value in scheduled_error.values()))

        best_sample = None
        best_score = None
        for sample in selected:
            score = sum(_safe_float(value) for value in sample.property_error.values())
            if best_score is None or score < best_score:
                best_score = score
                best_sample = sample

        return {
            "iteration": iteration,
            "mode": "target_schedule",
            "schedule_id": schedule.schedule_id,
            "hypothesis": schedule.hypothesis,
            "selection_policy": schedule.selection_policy,
            "target_property": dict(target_property),
            "scheduled_target_count": len(schedule.scheduled_targets),
            "strategy_counts": strategy_counts,
            "generated_count": len(structures),
            "evaluated_count": len(evaluated_samples),
            "selected_count": len(selected),
            "label_counts": label_counts,
            "mean_error_to_scheduled_target": (
                sum(scheduled_error_scores) / len(scheduled_error_scores) if scheduled_error_scores else None
            ),
            "best_result": {
                "structure_id": best_sample.structure_id if best_sample else "",
                "label": best_sample.label if best_sample else "",
                "evaluated_property": dict(best_sample.evaluated_property) if best_sample else {},
                "property_error": dict(best_sample.property_error) if best_sample else {},
                "scheduled_target": dict((best_sample.metadata or {}).get("scheduled_target") or {}) if best_sample else {},
                "error_to_scheduled_target": dict((best_sample.metadata or {}).get("error_to_scheduled_target") or {}) if best_sample else {},
            },
            "satisfies_target": bool(best_sample and best_sample.label == "success"),
        }

    def _knowledge_update_summary(
        self,
        iteration: int,
        evaluated_samples: list[KnowledgeSample],
        selected: list[KnowledgeSample],
        added_evidence_count: int,
        knowledge_path: str,
    ) -> dict[str, Any]:
        return {
            "iteration": iteration,
            "evaluated_samples": len(evaluated_samples),
            "selected_samples": len(selected),
            "added_samples": len(evaluated_samples),
            "added_knowledge_evidence": added_evidence_count,
            "knowledge_snapshot_path": knowledge_path,
            "dataset_distribution": self._dataset_distribution(),
        }

    @staticmethod
    def _candidate_id(candidate: MetaCandidate, fallback_index: int) -> str:
        raw = candidate.candidate_id or f"candidate_{fallback_index:03d}"
        return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)[:120]

    @staticmethod
    def _rank_agent_candidates(agent_candidates: list[MetaCandidate]) -> list[MetaCandidate]:
        candidates = sorted(agent_candidates, key=lambda item: (-item.confidence, -item.score, item.strategy, item.diversity_key))
        seen = set()
        ranked = []
        for candidate in candidates:
            meta = candidate.meta
            key = (
                meta.group,
                meta.basic_unit_type,
                meta.unit_cell_type,
                meta.topology_type,
                meta.connectivity_pattern,
                meta.max_bars,
                round(float(meta.rho_target), 4),
                tuple(round(float(value), 4) for value in meta.density_range),
            )
            if key in seen:
                continue
            seen.add(key)
            ranked.append(candidate)
        return ranked

    def _select_executed_candidates(
        self,
        ranked_candidates: list[MetaCandidate],
        fallback_candidates: list[MetaCandidate],
    ) -> list[MetaCandidate]:
        if not ranked_candidates:
            return fallback_candidates[:1]
        if self.experiment_budget <= 1:
            return ranked_candidates[:1]

        selected: list[MetaCandidate] = [ranked_candidates[0]]
        selected_ids = {id(ranked_candidates[0])}
        strategy_order = (
            "exploitation",
            "counterfactual",
            "repair",
            "diversity",
            "high_risk",
        )

        def add(candidate: MetaCandidate) -> None:
            if id(candidate) in selected_ids or len(selected) >= self.experiment_budget:
                return
            selected.append(candidate)
            selected_ids.add(id(candidate))

        for strategy in strategy_order:
            for candidate in ranked_candidates:
                if candidate.strategy == strategy:
                    add(candidate)
                    break
            if len(selected) >= self.experiment_budget:
                return selected

        for candidate in ranked_candidates:
            add(candidate)
            if len(selected) >= self.experiment_budget:
                break
        return selected

    def _batch_design_summary(
        self,
        iteration: int,
        agent_candidates: list[MetaCandidate],
        ranked_candidates: list[MetaCandidate],
        executed_candidates: list[MetaCandidate],
    ) -> dict[str, Any]:
        strategy_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        for candidate in ranked_candidates:
            strategy_counts[candidate.strategy] = strategy_counts.get(candidate.strategy, 0) + 1
            source_counts[candidate.source_backend] = source_counts.get(candidate.source_backend, 0) + 1
        return {
            "iteration": iteration,
            "agent_candidate_count": len(agent_candidates),
            "ranked_candidate_count": len(ranked_candidates),
            "executed_candidate_count": len(executed_candidates),
            "experiment_budget": self.experiment_budget,
            "strategy_counts": strategy_counts,
            "source_counts": source_counts,
            "executed_candidates": [candidate.to_dict() for candidate in executed_candidates],
        }

    @staticmethod
    def _schedule_summary(iteration: int, schedule: TargetSchedule, proposal_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "iteration": iteration,
            "hypothesis": schedule.hypothesis,
            "reason": schedule.selection_policy,
            "structure_parameters": {
                "action_space": "target_schedule",
                "schedule_id": schedule.schedule_id,
                "scheduled_targets": [item.to_dict() for item in schedule.scheduled_targets],
            },
            "proposal": dict(proposal_payload),
            "schedule": schedule.to_dict(),
            "scheduled_target_count": len(schedule.scheduled_targets),
            "total_requested_samples": sum(item.samples for item in schedule.scheduled_targets),
            "strategy_counts": dict(proposal_payload.get("strategy_counts") or {}),
            "target_schedule_design": {
                "agent_candidate_count": len(schedule.scheduled_targets),
                "ranked_candidate_count": len(schedule.scheduled_targets),
                "executed_candidate_count": len(schedule.scheduled_targets),
                "experiment_budget": len(schedule.scheduled_targets),
                "strategy_counts": dict(proposal_payload.get("strategy_counts") or {}),
                "executed_candidates": [item.to_dict() for item in schedule.scheduled_targets],
            },
            "batch_design": {
                "agent_candidate_count": len(schedule.scheduled_targets),
                "ranked_candidate_count": len(schedule.scheduled_targets),
                "executed_candidate_count": len(schedule.scheduled_targets),
                "experiment_budget": len(schedule.scheduled_targets),
                "strategy_counts": dict(proposal_payload.get("strategy_counts") or {}),
                "executed_candidates": [item.to_dict() for item in schedule.scheduled_targets],
            },
        }

    @staticmethod
    def _schedule_item_brief(item: dict[str, Any]) -> dict[str, Any]:
        schedule_item = dict(item.get("schedule_item") or {})
        return {
            "schedule_id": item.get("schedule_id", ""),
            "schedule_step": item.get("schedule_step", 0),
            "sample_index": item.get("sample_index", 0),
            "scheduled_target": dict(item.get("scheduled_target") or {}),
            "strategy": schedule_item.get("strategy", ""),
            "target_id": schedule_item.get("target_id", ""),
            "reason": schedule_item.get("reason", ""),
        }

    def _apply_schedule_metadata(
        self,
        sample: KnowledgeSample,
        schedule: TargetSchedule,
        schedule_record: dict[str, Any],
        scheduled_error: dict[str, float],
        iteration: int | None = None,
    ) -> None:
        metadata = dict(sample.metadata or {})
        schedule_item = dict(schedule_record.get("schedule_item") or {})
        if iteration is not None:
            metadata["iteration"] = int(iteration)
        metadata["target_schedule"] = schedule.to_dict()
        metadata["schedule_item"] = schedule_item
        metadata["final_target"] = dict(schedule.final_target)
        metadata["scheduled_target"] = dict(schedule_record.get("scheduled_target") or {})
        metadata["error_to_scheduled_target"] = dict(scheduled_error)
        metadata["error_to_final_target"] = dict(sample.property_error)
        raw_metrics = dict(metadata.get("raw_metrics") or {})
        realization_error = _aggregate_error_value(scheduled_error)
        utility_error = _aggregate_error_value(sample.property_error)
        if realization_error is not None:
            raw_metrics.setdefault("realization_curve_mae", realization_error)
            raw_metrics.setdefault("scheduled_curve_mae", realization_error)
        if utility_error is not None:
            raw_metrics.setdefault("utility_curve_mae", utility_error)
            raw_metrics.setdefault("final_curve_mae", utility_error)
        if iteration is not None:
            raw_metrics["iteration"] = int(iteration)
        metadata["raw_metrics"] = raw_metrics
        metadata["fidelity"] = metadata.get("fidelity", "proxy")
        metadata["hypothesis"] = schedule.hypothesis or schedule_item.get("reason", "")
        metadata["reason"] = schedule_item.get("reason", schedule.selection_policy)
        metadata["agent_suggestion"] = {
            "schedule_id": schedule.schedule_id,
            "target_id": schedule_item.get("target_id", ""),
            "strategy": schedule_item.get("strategy", ""),
            "confidence": schedule.confidence,
            "hypothesis": schedule.hypothesis,
            "reason": schedule_item.get("reason", ""),
            "scheduled_target": dict(schedule_record.get("scheduled_target") or {}),
            "final_target": dict(schedule.final_target),
        }
        sample.metadata = metadata
        sample.parameter_config = {
            **dict(sample.parameter_config or {}),
            "source": "target_schedule",
            "schedule_id": schedule.schedule_id,
            "target_id": schedule_item.get("target_id", ""),
            "scheduled_target": dict(schedule_record.get("scheduled_target") or {}),
            "final_target": dict(schedule.final_target),
            "strategy": schedule_item.get("strategy", ""),
            "error_to_scheduled_target": dict(scheduled_error),
            "error_to_final_target": dict(sample.property_error),
            "training_target": "explicit_structure",
        }

    def _no_candidate_sample(
        self,
        iteration: int,
        schedule: TargetSchedule,
        schedule_record: dict[str, Any],
        target_property: dict[str, float],
    ) -> KnowledgeSample:
        schedule_item = dict(schedule_record.get("schedule_item") or {})
        scheduled_target = dict(schedule_record.get("scheduled_target") or {})
        structure_id = (
            f"no_candidate__{schedule.schedule_id}_"
            f"s{int(schedule_record.get('schedule_step') or 0):02d}_"
            f"n{int(schedule_record.get('sample_index') or 0):02d}"
        )
        sample = KnowledgeSample(
            structure_id=structure_id,
            structure_path="",
            unit_cell_type="",
            basic_unit_type="",
            topology_type="",
            symmetry="",
            connectivity_pattern="",
            parameter_config={},
            target_property=dict(target_property),
            evaluated_property={},
            property_error={key: 1.0 for key in target_property},
            fem_status="not_run",
            geometry_status="no_candidate",
            label="failure",
            source="inverse_designer_target_schedule",
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "sample_id": structure_id,
                "sample_type": "inverse_designer_no_candidate",
                "target_schedule": schedule.to_dict(),
                "schedule_item": schedule_item,
                "final_target": dict(schedule.final_target),
                "scheduled_target": scheduled_target,
                "error_to_scheduled_target": {key: 1.0 for key in scheduled_target},
                "error_to_final_target": {key: 1.0 for key in target_property},
                "hypothesis": schedule.hypothesis,
                "reason": schedule_item.get("reason", "InverseDesigner returned no explicit structure for this scheduled target."),
                "agent_suggestion": {
                    "schedule_id": schedule.schedule_id,
                    "target_id": schedule_item.get("target_id", ""),
                    "strategy": schedule_item.get("strategy", ""),
                    "confidence": schedule.confidence,
                    "scheduled_target": scheduled_target,
                    "final_target": dict(schedule.final_target),
                },
                "fidelity": "none",
                "raw_metrics": {"no_candidate": True, "iteration": iteration},
            },
            explicit_structure={},
        )
        self._apply_schedule_metadata(sample, schedule, schedule_record, {key: 1.0 for key in scheduled_target})
        return sample

    @staticmethod
    def _fallback_target_schedule_proposal(target_property: dict[str, float], schedule_size: int) -> TargetScheduleProposal:
        schedule_id = f"target_schedule_fallback_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        items = [
            TargetScheduleItem(
                target_id=f"{schedule_id}_t01",
                target_property=dict(target_property),
                strategy="exploitation",
                reason="Fallback schedule because AgentExplorer has no propose_target_schedule interface.",
                expected_effect={"goal": "direct_solution"},
                risk="medium",
                samples=1,
            )
        ]
        for index, scale in enumerate((0.95, 1.05, 0.9, 1.1), start=2):
            if len(items) >= max(1, int(schedule_size)):
                break
            items.append(
                TargetScheduleItem(
                    target_id=f"{schedule_id}_t{index:02d}",
                    target_property={key: float(value) * scale for key, value in target_property.items()},
                    strategy="explore",
                    reason="Fallback local target-space perturbation.",
                    expected_effect={"goal": "local_coverage"},
                    risk="medium",
                    samples=1,
                )
            )
        schedule = TargetSchedule(
            schedule_id=schedule_id,
            final_target=dict(target_property),
            scheduled_targets=items,
            hypothesis="Fallback target schedule.",
            selection_policy="Use final target plus small target-space perturbations.",
            confidence=0.25,
            source="scheduler_fallback",
        )
        return TargetScheduleProposal(
            proposal_id=f"{schedule_id}_proposal",
            source="scheduler_fallback",
            final_target=dict(target_property),
            schedules=[schedule],
            rationale=schedule.selection_policy,
            strategy_counts={item.strategy: sum(1 for other in items if other.strategy == item.strategy) for item in items},
        )

    def _sample_schedule_records(self, schedule: TargetSchedule) -> list[dict[str, Any]]:
        if hasattr(self.inverse_designer, "sample_schedule"):
            return self.inverse_designer.sample_schedule(schedule)
        records: list[dict[str, Any]] = []
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
                        "status": "sampled" if structure is not None else "no_candidate",
                    }
                )
        return records

    @staticmethod
    def _sample_to_observation(iteration: int, sample: KnowledgeSample) -> Observation:
        metadata = dict(sample.metadata or {})
        explicit_structure = dict(sample.explicit_structure or {})
        structure_payload = {
            "structure_id": sample.structure_id,
            "structure_path": sample.structure_path,
            "unit_cell_type": sample.unit_cell_type,
            "basic_unit_type": sample.basic_unit_type,
            "topology_type": sample.topology_type,
            "symmetry": sample.symmetry,
            "connectivity_pattern": sample.connectivity_pattern,
        }
        structure_payload.update({key: value for key, value in explicit_structure.items() if key not in structure_payload or not structure_payload[key]})
        structure_payload["structure_id"] = sample.structure_id
        structure_payload["structure_path"] = sample.structure_path
        meta_payload = dict(metadata.get("schedule_item") or metadata.get("datagen_config") or {})
        if metadata.get("target_schedule"):
            target_schedule = dict(metadata.get("target_schedule") or {})
            meta_payload.setdefault("schedule_id", target_schedule.get("schedule_id", ""))
            meta_payload.setdefault("source", target_schedule.get("source", "agent_explorer_target_schedule"))
            meta_payload.setdefault("confidence", target_schedule.get("confidence", 0.5))
            meta_payload.setdefault("hypothesis", target_schedule.get("hypothesis", ""))
        if metadata.get("scheduled_target"):
            meta_payload["scheduled_target"] = dict(metadata.get("scheduled_target") or {})
        if metadata.get("final_target"):
            meta_payload["final_target"] = dict(metadata.get("final_target") or {})
        if metadata.get("error_to_scheduled_target"):
            meta_payload["error_to_scheduled_target"] = dict(metadata.get("error_to_scheduled_target") or {})
        if metadata.get("error_to_final_target"):
            meta_payload["error_to_final_target"] = dict(metadata.get("error_to_final_target") or {})
        return Observation(
            observation_id=f"obs_{iteration:03d}_{sample.structure_id.replace(':', '_')}",
            iteration=iteration,
            meta=meta_payload,
            structure=structure_payload,
            property=dict(sample.evaluated_property),
            error=dict(sample.property_error),
            label=sample.label,
            fem_status=sample.fem_status,
            geometry_status=sample.geometry_status,
            artifacts=dict(metadata.get("artifacts") or {}),
            provenance={
                "source": sample.source,
                "target_property": dict(sample.target_property),
                "final_target": dict(metadata.get("final_target") or sample.target_property),
                "scheduled_target": dict(metadata.get("scheduled_target") or {}),
                "error_to_scheduled_target": dict(metadata.get("error_to_scheduled_target") or {}),
                "error_to_final_target": dict(metadata.get("error_to_final_target") or sample.property_error),
                "hypothesis": metadata.get("hypothesis", ""),
                "reason": metadata.get("reason", ""),
                "agent_suggestion": dict(metadata.get("agent_suggestion") or {}),
                "target_schedule": dict(metadata.get("target_schedule") or {}),
                "schedule_item": dict(metadata.get("schedule_item") or {}),
                "candidate": dict(metadata.get("candidate") or {}),
                "run": dict(metadata.get("run") or {}),
                "fidelity": metadata.get("fidelity", "proxy"),
            },
            raw_metrics=dict(metadata.get("raw_metrics") or {}),
        )

    @staticmethod
    def _inverse_candidate_observation(
        candidate: KnowledgeSample,
        candidate_eval: dict[str, Any],
        target_property: dict[str, float],
    ) -> Observation:
        metadata = dict(candidate.metadata or {})
        final_error = dict(candidate_eval.get("property_error") or candidate.property_error)
        meta_payload = dict(metadata.get("datagen_config") or candidate.parameter_config or {})
        meta_payload.setdefault("final_target", dict(target_property))
        meta_payload.setdefault("scheduled_target", dict(target_property))
        meta_payload.setdefault("error_to_scheduled_target", dict(final_error))
        meta_payload.setdefault("error_to_final_target", dict(final_error))
        raw_metrics = dict(metadata.get("raw_metrics") or {})
        aggregated_error = _aggregate_error_value(final_error)
        if aggregated_error is not None:
            raw_metrics.setdefault("realization_curve_mae", aggregated_error)
            raw_metrics.setdefault("scheduled_curve_mae", aggregated_error)
            raw_metrics.setdefault("utility_curve_mae", aggregated_error)
            raw_metrics.setdefault("final_curve_mae", aggregated_error)
        explicit_structure = dict(candidate.explicit_structure or {})
        structure_payload = {
            "structure_id": candidate.structure_id,
            "structure_path": candidate.structure_path,
            "unit_cell_type": candidate.unit_cell_type,
            "basic_unit_type": candidate.basic_unit_type,
            "topology_type": candidate.topology_type,
            "symmetry": candidate.symmetry,
            "connectivity_pattern": candidate.connectivity_pattern,
        }
        structure_payload.update({key: value for key, value in explicit_structure.items() if key not in structure_payload or not structure_payload[key]})
        structure_payload["structure_id"] = candidate.structure_id
        structure_payload["structure_path"] = candidate.structure_path
        return Observation(
            observation_id=f"obs_000_inverse_{candidate.structure_id.replace(':', '_')}",
            iteration=0,
            meta=meta_payload,
            structure=structure_payload,
            property=dict(candidate_eval.get("evaluated_property") or candidate.evaluated_property),
            error=final_error,
            label=str(candidate_eval.get("label") or candidate.label),
            fem_status=candidate.fem_status,
            geometry_status=candidate.geometry_status,
            artifacts=dict(metadata.get("artifacts") or {}),
            provenance={
                "source": "inverse_designer",
                "target_property": dict(target_property),
                "final_target": dict(target_property),
                "scheduled_target": dict(target_property),
                "error_to_scheduled_target": dict(candidate_eval.get("property_error") or candidate.property_error),
                "error_to_final_target": dict(candidate_eval.get("property_error") or candidate.property_error),
                "candidate_source": candidate.source,
                "agent_suggestion": dict(metadata.get("agent_suggestion") or {}),
                "run": dict(metadata.get("run") or {}),
                "fidelity": metadata.get("fidelity", "historical"),
            },
            raw_metrics=raw_metrics,
        )

    def _scheduler_context(
        self,
        target_property: dict[str, float],
        failed_candidate: KnowledgeSample | None,
        failed_eval: dict[str, Any] | None,
        knowledge_snapshot: dict[str, Any],
        knowledge_path: str,
        statistical_snapshot: dict[str, Any] | None = None,
        statistical_path: str = "",
    ) -> dict[str, Any]:
        statistical_snapshot = dict(statistical_snapshot or knowledge_snapshot)
        context = SchedulerContext(
            target_property=dict(target_property),
            failed_candidate={
                **self._sample_brief(failed_candidate),
                "evaluation": dict(failed_eval or {}),
            }
            if failed_candidate is not None
            else {},
            top_success=[self._sample_brief(sample) for sample in self.knowledge_base.query_top_success(target_property, 3)],
            near_miss=[self._sample_brief(sample) for sample in self.knowledge_base.query_top_near_miss(target_property, 3)],
            recent_failures=[self._sample_brief(sample) for sample in self.knowledge_base.query_recent_failures(target_property, 5)],
            diverse_failures=[self._sample_brief(sample) for sample in self.knowledge_base.query_diverse_failures(target_property, 5)],
            dataset_statistics=self.knowledge_base.dataset_statistics(),
            agent_knowledge_snapshot=knowledge_snapshot,
            budget_config={
                "agent_batch_size": self.agent_batch_size,
                "experiment_budget": self.experiment_budget,
                "retrain_trigger": self.retrain_trigger,
            },
            constraints=self.generator_evaluator.datagen_schema(),
        ).to_dict()

        # Contract key: KnowledgeRefiner -> AgentExplorer knowledge transfer.
        context["knowledge_decision_package"] = knowledge_snapshot
        context["reasoned_agent_knowledge_snapshot"] = knowledge_snapshot
        # Compatibility keys expected by older callers/proposers.
        context["agent_knowledge"] = knowledge_snapshot
        context["statistical_knowledge_snapshot"] = statistical_snapshot
        context["reasoned_knowledge_snapshot"] = knowledge_snapshot
        context["decision_ready_summary"] = statistical_snapshot.get("decision_ready_summary", {})
        context["reachable_targets"] = statistical_snapshot.get("reachable_targets", [])
        context["promising_anchors"] = statistical_snapshot.get("promising_anchors", [])
        context["risky_promising_regions"] = statistical_snapshot.get("risky_promising_regions", [])
        context["avoid_regions"] = statistical_snapshot.get("avoid_regions", [])
        context["inverse_bias_patterns"] = statistical_snapshot.get("inverse_bias_patterns", [])
        context["finetune_candidates"] = statistical_snapshot.get("finetune_candidates", [])
        context["agent_knowledge_path"] = knowledge_path
        context["statistical_knowledge_path"] = statistical_path
        context["top_success_brief"] = context["top_success"]
        context["near_miss_brief"] = context["near_miss"]
        context["recent_failures_brief"] = context["recent_failures"]
        context["diverse_failures_brief"] = context["diverse_failures"]
        return context

    @staticmethod
    def _fmt_float(value: Any) -> str:
        if value in ("", None):
            return "-"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    def _fmt_property(self, payload: dict[str, Any]) -> str:
        if not payload:
            return "-"
        return ", ".join(f"{key}={self._fmt_float(value)}" for key, value in payload.items())

    def _fmt_distribution(self, distribution: dict[str, Any]) -> list[str]:
        if not distribution:
            return ["- no samples"]
        lines = [
            f"- total_samples: `{distribution.get('total_samples', 0)}`",
            f"- label_counts: `{distribution.get('label_counts', {})}`",
            f"- group_counts: `{distribution.get('group_counts', {})}`",
        ]
        for key in ("stiffness_proxy", "density_proxy"):
            stats = distribution.get(key, {})
            if not stats:
                continue
            lines.append(
                f"- {key}: min={self._fmt_float(stats.get('min'))}, "
                f"p50={self._fmt_float(stats.get('p50'))}, "
                f"max={self._fmt_float(stats.get('max'))}, "
                f"mean={self._fmt_float(stats.get('mean'))}"
            )
        return lines

    @staticmethod
    def _event_by_stage(logs: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
        return [item for item in logs if item.get("stage") == stage]

    @staticmethod
    def _iteration_from_payload(item: dict[str, Any]) -> int | None:
        payload = item.get("payload", {})
        value = payload.get("iteration")
        return int(value) if isinstance(value, int) else None

    @staticmethod
    def _best_sample(discovered_samples: list[KnowledgeSample], success: bool) -> KnowledgeSample | None:
        if not discovered_samples:
            return None
        if success:
            for sample in discovered_samples:
                if sample.label == "success":
                    return sample
        best = None
        best_score = None
        for sample in discovered_samples:
            score = sum(_safe_float(value) for value in sample.property_error.values())
            if best_score is None or score < best_score:
                best_score = score
                best = sample
        return best

    def _write_run_summary(
        self,
        target_property: dict[str, float],
        task_status: TaskStatus,
        success: bool,
        iterations: int,
        logs: list[dict[str, Any]],
        discovered_samples: list[KnowledgeSample],
    ) -> str:
        path = Path(self.experiment_paths.root_dir) / "run_summary.md"
        lines = ["# Run Summary", ""]
        best_sample = self._best_sample(discovered_samples, success)
        lines.extend(
            [
                "## Overview",
                "",
                f"- task_id: `{self.task_id}`",
                f"- status: `{task_status.value}`",
                f"- success: `{success}`",
                f"- iterations: `{iterations}`",
                f"- target_property: `{self._fmt_property(target_property)}`",
                f"- experiment_root: `{self.experiment_paths.root_dir}`",
                f"- events_log: `{Path(self.event_stream.path).resolve()}`",
                "",
            ]
        )

        inverse_events = self._event_by_stage(logs, "inverse_designer")
        inverse_payload = inverse_events[0].get("payload", {}) if inverse_events else {}
        inverse_label = ""
        if inverse_payload.get("prediction_result"):
            inverse_label = str(inverse_payload["prediction_result"].get("label", ""))
        elif inverse_payload.get("satisfies_target"):
            inverse_label = "success"
        elif inverse_payload.get("candidate_found"):
            inverse_label = "not_success"
        else:
            inverse_label = "none"
        final_label = best_sample.label if best_sample is not None else "none"
        lines.extend(
            [
                "## Visual Overview",
                "",
                "```mermaid",
                "flowchart TD",
                f"    A[\"Target<br/>{self._fmt_property(target_property)}\"] --> B[\"InverseDesigner probe<br/>label: {inverse_label}\"]",
                "    B --> C{\"Probe satisfies target?\"}",
                "    C -- \"yes\" --> Z[\"Finished by inverse designer\"]",
                "    C -- \"no\" --> I1[\"Iteration 1<br/>datagen + evaluation\"]",
            ]
        )
        for iteration in range(1, iterations):
            lines.append(f"    I{iteration} --> I{iteration + 1}[\"Iteration {iteration + 1}<br/>datagen + evaluation\"]")
        if iterations > 0:
            lines.append(f"    I{iterations} --> O[\"Final outcome<br/>status: {task_status.value}<br/>best: {final_label}\"]")
        else:
            lines.append(f"    C --> O[\"Final outcome<br/>status: {task_status.value}<br/>best: {final_label}\"]")
        lines.extend(["```", ""])

        task_events = self._event_by_stage(logs, "task")
        if task_events:
            start_payload = task_events[0].get("payload", {})
            lines.extend(["## Dataset At Start", ""])
            lines.extend(self._fmt_distribution(start_payload.get("dataset_distribution", {})))
            lines.append("")

        if inverse_events:
            payload = inverse_events[0].get("payload", {})
            lines.extend(["## InverseDesigner", ""])
            lines.append(f"- target_property: `{self._fmt_property(payload.get('target_property', {}))}`")
            lines.append(f"- candidate_found: `{payload.get('candidate_found', False)}`")
            if payload.get("candidate"):
                candidate = payload["candidate"]
                lines.append(f"- candidate: `{candidate.get('structure_id', '')}` in group `{candidate.get('group', '')}`")
                lines.append(f"- predicted/evaluated property: `{self._fmt_property(candidate.get('evaluated_property', {}))}`")
                lines.append(f"- candidate_parameters: `{candidate.get('parameter_config', {})}`")
            lines.append(f"- satisfies_target: `{payload.get('satisfies_target', False)}`")
            lines.append("")

        dataset_events = {self._iteration_from_payload(item): item for item in self._event_by_stage(logs, "dataset_state")}
        agent_events = {self._iteration_from_payload(item): item for item in self._event_by_stage(logs, "agent_explorer")}
        eval_events = {self._iteration_from_payload(item): item for item in self._event_by_stage(logs, "evaluation")}
        update_events = {self._iteration_from_payload(item): item for item in self._event_by_stage(logs, "knowledge_update")}
        finetune_events = {self._iteration_from_payload(item): item for item in self._event_by_stage(logs, "inverse_designer_finetune")}

        for iteration in range(1, iterations + 1):
            lines.extend([f"## Iteration {iteration}", ""])
            dataset_payload = dataset_events.get(iteration, {}).get("payload", {})
            if dataset_payload:
                lines.append("Dataset snapshot:")
                lines.extend(self._fmt_distribution(dataset_payload.get("dataset_distribution", {})))
                if dataset_payload.get("knowledge_snapshot_path"):
                    lines.append(f"- knowledge_snapshot: `{dataset_payload['knowledge_snapshot_path']}`")
                lines.append("")

            agent_payload = agent_events.get(iteration, {}).get("payload", {})
            if agent_payload:
                lines.append("Agent reasoning:")
                lines.append(f"- based_on: `{agent_payload.get('based_on', '')}`")
                lines.append(f"- hypothesis: {agent_payload.get('hypothesis', '-')}")
                lines.append(f"- reason: {agent_payload.get('reason', '-')}")
                lines.append(f"- suggestion_source: `{agent_payload.get('source', '')}`")
                lines.append(f"- structure_parameters: `{agent_payload.get('structure_parameters', {})}`")
                lines.append(f"- expected_property: `{self._fmt_property(agent_payload.get('expected_property', {}))}`")
                lines.append(f"- confidence: `{self._fmt_float(agent_payload.get('confidence'))}`")
                lines.append("")

            eval_payload = eval_events.get(iteration, {}).get("payload", {})
            if eval_payload:
                lines.append("Datagen + evaluation:")
                lines.append(f"- generated_count: `{eval_payload.get('generated_count', 0)}`")
                lines.append(f"- evaluated_count: `{eval_payload.get('evaluated_count', 0)}`")
                lines.append(f"- selected_count: `{eval_payload.get('selected_count', 0)}`")
                lines.append(f"- label_counts: `{eval_payload.get('label_counts', {})}`")
                best_result = eval_payload.get("best_result", {})
                if best_result:
                    lines.append(
                        f"- best_result: `{best_result.get('structure_id', '')}` "
                        f"label=`{best_result.get('label', '')}` "
                        f"property=`{self._fmt_property(best_result.get('evaluated_property', {}))}`"
                    )
                lines.append(f"- satisfies_target: `{eval_payload.get('satisfies_target', False)}`")
                lines.append("")

            update_payload = update_events.get(iteration, {}).get("payload", {})
            if update_payload:
                lines.append("Knowledge update:")
                lines.append(f"- added_samples: `{update_payload.get('added_samples', 0)}`")
                if update_payload.get("knowledge_snapshot_path"):
                    lines.append(f"- knowledge_snapshot: `{update_payload['knowledge_snapshot_path']}`")
                lines.extend(self._fmt_distribution(update_payload.get("dataset_distribution", {})))
                lines.append("")

            finetune_payload = finetune_events.get(iteration, {}).get("payload", {})
            if finetune_payload:
                lines.append("Model update:")
                lines.append(f"- new_training_samples: `{finetune_payload.get('new_training_samples', 0)}`")
                lines.append("")

        lines.extend(["## Final Outcome", ""])
        if best_sample is not None:
            lines.append(f"- best_sample: `{best_sample.structure_id}`")
            lines.append(f"- best_sample_label: `{best_sample.label}`")
            lines.append(f"- best_sample_property: `{self._fmt_property(best_sample.evaluated_property)}`")
            lines.append(f"- best_sample_error: `{self._fmt_property(best_sample.property_error)}`")
            lines.append(f"- best_sample_path: `{best_sample.structure_path}`")
        else:
            lines.append("- no discovered sample")
        lines.append("")

        path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        return str(path.resolve())

    def _finalize_result(
        self,
        task_status: TaskStatus,
        target_property: dict[str, float],
        success: bool,
        iterations: int,
        discovered_samples: list[KnowledgeSample],
        logs: list[dict[str, Any]],
    ) -> ClosedLoopResult:
        self._write_run_summary(
            target_property=target_property,
            task_status=task_status,
            success=success,
            iterations=iterations,
            logs=logs,
            discovered_samples=discovered_samples,
        )
        return ClosedLoopResult(
            task_id=self.task_id,
            task_status=task_status.value,
            target_property=target_property,
            success=success,
            iterations=iterations,
            experiment_paths=self.experiment_paths,
            discovered_samples=discovered_samples,
            events=self.event_stream.events,
            logs=logs,
        )

    def _try_inverse_designer_structure(
        self,
        logs: list[dict[str, Any]],
        target_property: dict[str, float],
        phase: str,
        iteration: int = 0,
    ) -> tuple[KnowledgeSample | None, dict[str, Any] | None, bool]:
        if not hasattr(self.inverse_designer, "sample_structure"):
            self._emit(
                logs,
                stage="inverse_designer",
                status="no_structure_interface",
                payload={
                    **self._candidate_summary(None, None, target_property),
                    "phase": phase,
                    "iteration": iteration,
                    "training_target": "explicit_structure",
                },
            )
            return None, None, False

        structure_candidate = self.inverse_designer.sample_structure(target_property)
        if structure_candidate is None:
            self._emit(
                logs,
                stage="inverse_designer",
                status="no_structure_candidate",
                payload={
                    **self._candidate_summary(None, None, target_property),
                    "phase": phase,
                    "iteration": iteration,
                    "training_target": "explicit_structure",
                },
            )
            return None, None, False

        candidate_eval = self.generator_evaluator.evaluate_explicit_structure(structure_candidate, target_property)
        candidate = self.generator_evaluator.collect_explicit_structure_sample(
            structure=structure_candidate,
            evaluation=candidate_eval,
            target_property=target_property,
            source="inverse_designer",
        )
        candidate_observation = self._inverse_candidate_observation(candidate, candidate_eval, target_property)
        self.raw_experiment_store.append(candidate_observation)
        self.knowledge_base.add_knowledge_evidences(
            self.raw_experiment_store.project_knowledge_from([candidate_observation])
        )
        self.knowledge_base.add_sample(candidate)
        self._emit(
            logs,
            stage="inverse_designer",
            status="structure_retrieved",
            payload={
                **self._candidate_summary(candidate, candidate_eval, target_property),
                "phase": phase,
                "iteration": iteration,
                "training_target": "explicit_structure",
            },
        )
        return candidate, candidate_eval, candidate_eval.get("label") == "success"

    def run(self, target_property: dict[str, float], max_iterations: int = 3) -> ClosedLoopResult:
        target_property = normalize_target_property(target_property)
        logs = []
        discovered_samples = []
        task_status = TaskStatus.RUNNING

        self._emit(
            logs,
            stage="task",
            status=TaskStatus.RUNNING.value,
            payload={
                "target_property": dict(target_property),
                "max_iterations": max_iterations,
                "dataset_distribution": self._dataset_distribution(),
            },
        )

        failed_candidate = None
        failed_eval = None
        last_finetune_iteration = 0

        candidate, candidate_eval, inverse_success = self._try_inverse_designer_structure(
            logs=logs,
            target_property=target_property,
            phase="initial",
            iteration=0,
        )
        if inverse_success and candidate is not None and self._success_stop_allowed(0):
            self._emit(
                logs,
                stage="task",
                status=TaskStatus.SUCCEEDED.value,
                payload={
                    "reason": "inverse_designer_structure_success",
                    "best_result": self._sample_brief(candidate),
                },
            )
            return self._finalize_result(
                task_status=TaskStatus.SUCCEEDED,
                target_property=target_property,
                success=True,
                iterations=0,
                discovered_samples=[candidate],
                logs=logs,
            )
        if candidate is not None:
            failed_candidate = candidate
            failed_eval = candidate_eval

        for iteration in range(1, max_iterations + 1):
            failed_candidate_brief = (
                {**self._sample_brief(failed_candidate), "evaluation": dict(failed_eval or {})}
                if failed_candidate is not None
                else {}
            )
            runtime_context = {
                "iteration": iteration,
                "max_iterations": max_iterations,
                "dataset_statistics": self.knowledge_base.dataset_statistics(),
                "experiment_budget": self.experiment_budget,
                "agent_batch_size": self.agent_batch_size,
                "retrain_trigger": self.retrain_trigger,
                "after_finetune": last_finetune_iteration == iteration - 1 and last_finetune_iteration > 0,
                "last_finetune_iteration": last_finetune_iteration,
                "closed_loop_stalled": False,
            }
            knowledge_snapshot = self.knowledge_refiner.build_decision_package(
                kb=self.knowledge_base,
                final_target=target_property,
                failed_candidate=failed_candidate_brief,
                runtime_context=runtime_context,
            )
            statistical_snapshot = dict(knowledge_snapshot.get("deterministic_snapshot") or {})
            statistical_path = self.knowledge_refiner.write_snapshot(
                statistical_snapshot,
                Path(self.experiment_paths.knowledge_dir) / f"statistical_knowledge_iter_{iteration:03d}.json",
            )
            knowledge_path = self.knowledge_refiner.write_snapshot(
                knowledge_snapshot,
                Path(self.experiment_paths.knowledge_dir) / f"agent_knowledge_iter_{iteration:03d}.json",
            )
            context = self._scheduler_context(
                target_property=target_property,
                failed_candidate=failed_candidate,
                failed_eval=failed_eval,
                knowledge_snapshot=knowledge_snapshot,
                knowledge_path=knowledge_path,
                statistical_snapshot=statistical_snapshot,
                statistical_path=statistical_path,
            )
            self._emit(
                logs,
                stage="dataset_state",
                status="snapshot",
                payload={
                    "iteration": iteration,
                    "dataset_distribution": self._dataset_distribution(),
                    "knowledge_snapshot_path": knowledge_path,
                    "statistical_snapshot_path": statistical_path,
                },
            )

            previous_feedback = dict(failed_eval or {})
            if failed_candidate is not None:
                previous_feedback["next_anchor_sample"] = self._sample_brief(failed_candidate)
                previous_feedback.setdefault("best_sample", self._sample_brief(failed_candidate))
                previous_feedback.setdefault("main_error_direction", max(failed_candidate.property_error, key=failed_candidate.property_error.get) if failed_candidate.property_error else "")
                previous_feedback.setdefault("suggested_next_action", "exploit_near_miss" if failed_candidate.label == "near_miss" else "probe_reachability")

            if hasattr(self.agent_explorer, "propose_target_schedule"):
                schedule_proposal = self.agent_explorer.propose_target_schedule(
                    final_target=target_property,
                    feedback_signal=previous_feedback,
                    context=context,
                    schedule_size=self.agent_batch_size,
                )
            else:
                schedule_proposal = self._fallback_target_schedule_proposal(target_property, self.agent_batch_size)
            if not schedule_proposal.schedules:
                self._emit(
                    logs,
                    stage="agent_explorer",
                    status="no_target_schedule",
                    payload={
                        "iteration": iteration,
                        "target_property": dict(target_property),
                        "reason": "agent_explorer_returned_no_schedules",
                    },
                )
                continue
            schedule = schedule_proposal.schedules[0]
            self._emit(
                logs,
                stage="agent_explorer",
                status="target_schedule_proposed",
                payload=self._schedule_summary(iteration, schedule, schedule_proposal.to_dict()),
            )

            structures: list[dict[str, Any]] = []
            evaluated_samples: list[KnowledgeSample] = []
            no_candidate_records: list[dict[str, Any]] = []
            schedule_records = self._sample_schedule_records(schedule)
            for record in schedule_records:
                structure = record.get("structure")
                if not structure:
                    no_candidate_records.append(self._schedule_item_brief(record))
                    evaluated_samples.append(self._no_candidate_sample(iteration, schedule, record, target_property))
                    continue
                structure = dict(structure)
                original_structure_id = str(structure.get("structure_id") or structure.get("sample_id") or "inverse_structure")
                scheduled_target = dict(record.get("scheduled_target") or {})
                scheduled_item = dict(record.get("schedule_item") or {})
                unique_structure_id = (
                    f"{original_structure_id}__{schedule.schedule_id}_"
                    f"s{int(record.get('schedule_step') or 0):02d}_"
                    f"n{int(record.get('sample_index') or 0):02d}"
                )
                structure["original_structure_id"] = original_structure_id
                structure["structure_id"] = unique_structure_id
                structure["target_schedule"] = schedule.to_dict()
                structure["schedule_item"] = scheduled_item
                structure["scheduled_target"] = scheduled_target
                structure["final_target"] = dict(target_property)

                evaluation = self.generator_evaluator.evaluate_explicit_structure(structure, target_property)
                scheduled_error = curve_aware_property_error(
                    scheduled_target,
                    dict(evaluation.get("evaluated_property") or {}),
                    dict(evaluation.get("raw_metrics") or {}),
                )
                if scheduled_error.get("curve_nmae") is not None:
                    raw_metrics = dict(evaluation.get("raw_metrics") or {})
                    raw_metrics["scheduled_curve_metrics"] = dict(scheduled_error)
                    raw_metrics["scheduled_curve_mae"] = float(scheduled_error["curve_nmae"])
                    raw_metrics["realization_curve_mae"] = float(scheduled_error["curve_nmae"])
                    for key in (
                        "peak_error",
                        "energy_error",
                        "initial_modulus_error",
                        "peak_relative_delta",
                        "energy_relative_delta",
                        "initial_modulus_relative_delta",
                        "peak_target",
                        "peak_observed",
                        "energy_target",
                        "energy_observed",
                        "initial_modulus_target",
                        "initial_modulus_observed",
                    ):
                        if key in scheduled_error:
                            raw_metrics[key] = scheduled_error[key]
                    evaluation["raw_metrics"] = raw_metrics
                sample = self.generator_evaluator.collect_explicit_structure_sample(
                    structure=structure,
                    evaluation=evaluation,
                    target_property=target_property,
                    source="inverse_designer_target_schedule",
                )
                self._apply_schedule_metadata(sample, schedule, record, scheduled_error, iteration=iteration)
                structures.append(structure)
                evaluated_samples.append(sample)

            if no_candidate_records:
                self._emit(
                    logs,
                    stage="inverse_designer",
                    status="schedule_targets_missing_candidates",
                    payload={
                        "iteration": iteration,
                        "missing_count": len(no_candidate_records),
                        "records": no_candidate_records[:10],
                    },
                )

            observations = [self._sample_to_observation(iteration, sample) for sample in evaluated_samples]
            self.raw_experiment_store.append_many(observations)
            knowledge_evidence = self.raw_experiment_store.project_knowledge_from(observations)
            self.knowledge_base.add_knowledge_evidences(knowledge_evidence)
            self.knowledge_base.add_samples(evaluated_samples)
            feedback_signal = self.feedback_extractor.extract(target_property, evaluated_samples)
            selected = feedback_signal.feedback_samples
            discovered_samples.extend(selected)

            self._emit(
                logs,
                stage="evaluation",
                status="completed",
                payload={
                    **self._target_schedule_evaluation_summary(iteration, target_property, schedule, structures, evaluated_samples, selected),
                    "feedback_signal": feedback_signal.brief(),
                },
            )
            self._emit(
                logs,
                stage="knowledge_update",
                status="completed",
                payload=self._knowledge_update_summary(
                    iteration=iteration,
                    evaluated_samples=evaluated_samples,
                    selected=selected,
                    added_evidence_count=len(knowledge_evidence),
                    knowledge_path=knowledge_path,
                ),
            )
            self._maybe_write_surrogate_gt_report(iteration, logs)

            if (
                feedback_signal.should_stop
                and feedback_signal.best_success is not None
                and self._success_stop_allowed(iteration)
            ):
                task_status = TaskStatus.SUCCEEDED
                self._emit(
                    logs,
                    stage="task",
                    status=task_status.value,
                    payload={
                        "reason": "target_schedule_success",
                        "iteration": iteration,
                        "best_result": self._sample_brief(feedback_signal.best_success),
                        "feedback_signal": feedback_signal.brief(),
                    },
                )
                return self._finalize_result(
                    task_status=task_status,
                    target_property=target_property,
                    success=True,
                    iterations=iteration,
                    discovered_samples=discovered_samples,
                    logs=logs,
                )

            dataset_rows = self.raw_experiment_store.project_dataset()
            new_data = dataset_rows[self._last_training_count :]
            if len(new_data) >= self.retrain_trigger:
                self.inverse_designer.finetune(new_data)
                last_finetune_iteration = iteration
                self._last_training_count = len(dataset_rows)
                self.knowledge_base.mark_evidence_used_for_training()
                self._emit(
                    logs,
                    stage="inverse_designer_finetune",
                    status="completed",
                    payload={
                        "iteration": iteration,
                        "new_training_samples": len(new_data),
                        "dataset_distribution": self._dataset_distribution(),
                    },
                )
                retry_candidate, retry_eval, retry_success = self._try_inverse_designer_structure(
                    logs=logs,
                    target_property=target_property,
                    phase="post_finetune",
                    iteration=iteration,
                )
                if retry_success and retry_candidate is not None and self._success_stop_allowed(iteration):
                    task_status = TaskStatus.SUCCEEDED
                    final_discovered = list(discovered_samples)
                    if all(sample.structure_id != retry_candidate.structure_id for sample in final_discovered):
                        final_discovered.append(retry_candidate)
                    self._emit(
                        logs,
                        stage="task",
                        status=task_status.value,
                        payload={
                            "reason": "inverse_designer_structure_success_after_finetune",
                            "iteration": iteration,
                            "best_result": self._sample_brief(retry_candidate),
                        },
                    )
                    return self._finalize_result(
                        task_status=task_status,
                        target_property=target_property,
                        success=True,
                        iterations=iteration,
                        discovered_samples=final_discovered,
                        logs=logs,
                    )
                if retry_candidate is not None:
                    failed_candidate = retry_candidate
                    failed_eval = retry_eval

            if feedback_signal.next_anchor_sample is not None:
                failed_candidate = feedback_signal.next_anchor_sample
                failed_eval = {
                    "evaluated_property": failed_candidate.evaluated_property,
                    "property_error": failed_candidate.property_error,
                    "label": failed_candidate.label,
                }

        final_success = (
            self._best_sample(discovered_samples, success=True)
            if self.min_iterations_before_success > 0
            else None
        )
        task_status = TaskStatus.SUCCEEDED if final_success is not None else TaskStatus.FAILED
        self._emit(
            logs,
            stage="task",
            status=task_status.value,
            payload={
                "reason": "max_iterations_completed_with_success" if final_success is not None else "max_iterations_exceeded",
                "iterations": max_iterations,
                "dataset_distribution": self._dataset_distribution(),
                "best_result": self._sample_brief(final_success or failed_candidate),
            },
        )
        return self._finalize_result(
            task_status=task_status,
            target_property=target_property,
            success=final_success is not None,
            iterations=max_iterations,
            discovered_samples=discovered_samples,
            logs=logs,
        )


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


StructureDiscoveryScheduler = StructureDiscoverySystem
DeterministicSurrogateScheduler = DeterministicSurrogateClosedLoopSystem
