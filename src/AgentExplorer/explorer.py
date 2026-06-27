from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import requests

from ..closed_loop_contracts import (
    BatchProposal,
    DatagenConfig,
    DesignSearchParameters,
    KnowledgeSample,
    TargetSchedule,
    TargetScheduleItem,
    TargetScheduleProposal,
)
from ..MetaSpace import MetaSpace


DEFAULT_AGENT_EXPLORER_API_KEY = "e339901aa965318c253573770ae65bd0cfce8b93ca057ddc79725f80186382db"
DEFAULT_AGENT_EXPLORER_BASE_URL = "http://127.0.0.1:17777/v1"
DEFAULT_AGENT_EXPLORER_MODEL = "gpt-5.5"
MODEL_PREFERENCE = (
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("llm response does not contain a JSON object")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("llm response JSON must be an object")
    return payload


def _iter_sse_events(raw_text: str):
    event_name = ""
    data_lines: list[str] = []
    for line in raw_text.splitlines():
        if not line:
            if data_lines:
                yield event_name or "message", "\n".join(data_lines)
                event_name = ""
                data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if data_lines:
        yield event_name or "message", "\n".join(data_lines)


def _unique_tags(*groups: tuple[str, ...]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen = set()
    for group in groups:
        for item in group:
            if not item or item in seen:
                continue
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def _sample_value(sample: KnowledgeSample | dict | None, key: str, default: Any = "") -> Any:
    if sample is None:
        return default
    if isinstance(sample, dict):
        if key == "symmetry":
            return sample.get("symmetry") or sample.get("group") or default
        if key == "connectivity_pattern":
            return (
                sample.get("connectivity_pattern")
                or dict(sample.get("parameter_config") or {}).get("connectivity_pattern")
                or default
            )
        return sample.get(key, default)
    return getattr(sample, key, default)


class AgentExplorer:
    """
    Agent explorer with two modes:
    1. LLM suggestion through a Codex/OpenAI-compatible local endpoint.
    2. Deterministic heuristic fallback when the endpoint is unavailable.
    """

    _global_llm_disable_reason = ""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 20.0,
        enable_llm: bool | None = None,
        meta_space: MetaSpace | None = None,
    ):
        env_flag = os.getenv("AGENT_EXPLORER_ENABLE_LLM", "").strip().lower()
        if enable_llm is None:
            if env_flag:
                enable_llm = env_flag not in {"0", "false", "no", "off"}
            else:
                enable_llm = True

        self.enable_llm = bool(enable_llm)
        self.api_key = api_key or os.getenv("AGENT_EXPLORER_API_KEY") or DEFAULT_AGENT_EXPLORER_API_KEY
        self.base_url = (base_url or os.getenv("AGENT_EXPLORER_BASE_URL") or DEFAULT_AGENT_EXPLORER_BASE_URL).rstrip("/")
        self.model = model or os.getenv("AGENT_EXPLORER_MODEL", DEFAULT_AGENT_EXPLORER_MODEL)
        self.timeout = float(timeout)
        self.meta_space = meta_space or MetaSpace()
        self._session: requests.Session | None = None
        self._instance_llm_disable_reason = ""

    @staticmethod
    def _reasoned_snapshot(context: dict) -> dict[str, Any]:
        """KnowledgeInterpreter -> AgentExplorer contract.

        AgentExplorer may read current-round SchedulerContext fields directly,
        but historical knowledge interpretation enters through this snapshot.
        Older keys remain accepted so existing runs and tests stay readable.
        """
        for key in (
            "reasoned_agent_knowledge_snapshot",
            "reasoned_knowledge_snapshot",
            "agent_knowledge_snapshot",
            "agent_knowledge",
        ):
            value = context.get(key)
            if isinstance(value, dict):
                return value
        return {}

    @staticmethod
    def _statistical_snapshot(context: dict, reasoned_snapshot: dict[str, Any]) -> dict[str, Any]:
        statistical = reasoned_snapshot.get("statistical") or reasoned_snapshot.get("statistical_snapshot")
        if isinstance(statistical, dict):
            return statistical
        value = context.get("statistical_knowledge_snapshot")
        if isinstance(value, dict):
            return value
        return reasoned_snapshot

    def propose(
        self,
        target_property: dict[str, float],
        failed_candidate: KnowledgeSample | None,
        fem_result: dict | None,
        context: dict,
        datagen_schema: dict,
    ) -> DatagenConfig:
        heuristic = self._heuristic_propose(
            target_property=target_property,
            failed_candidate=failed_candidate,
            fem_result=fem_result,
            context=context,
            datagen_schema=datagen_schema,
        )

        llm_proposal, llm_error = self._try_llm_proposal(
            target_property=target_property,
            failed_candidate=failed_candidate,
            fem_result=fem_result,
            context=context,
            datagen_schema=datagen_schema,
            heuristic=heuristic,
        )
        if llm_proposal is not None:
            return llm_proposal
        if llm_error:
            failure_analysis = dict(heuristic.failure_analysis)
            failure_analysis["llm_fallback_reason"] = llm_error
            return replace(
                heuristic,
                failure_analysis=failure_analysis,
                tags=_unique_tags(heuristic.tags, ("llm_fallback",)),
            )
        return heuristic

    def propose_batch(
        self,
        target_property: dict[str, float],
        failed_candidate: KnowledgeSample | None,
        fem_result: dict | None,
        context: dict,
        datagen_schema: dict,
        batch_size: int = 77,
    ) -> BatchProposal:
        base = self.propose(
            target_property=target_property,
            failed_candidate=failed_candidate,
            fem_result=fem_result,
            context=context,
            datagen_schema=datagen_schema,
        )
        quotas = self._strategy_quotas(batch_size)
        candidates = self.meta_space.intervention_variants(
            base=base,
            target_property=target_property,
            source_backend="agent_designer",
            quotas=quotas,
        )
        if len(candidates) < batch_size:
            candidates.insert(
                0,
                self.meta_space.candidate(
                    meta=replace(base, num_samples=1),
                    source_backend="agent_designer",
                    strategy="exploitation",
                    confidence=base.confidence,
                    predicted_property=base.expected_property,
                    hypothesis_id=base.suggestion_id,
                    rationale=base.reason,
                    candidate_id=f"{base.suggestion_id}_base",
                ),
            )
        candidates = self.meta_space.deduplicate(candidates, limit=batch_size)
        strategy_counts: dict[str, int] = {}
        for candidate in candidates:
            strategy_counts[candidate.strategy] = strategy_counts.get(candidate.strategy, 0) + 1
        return BatchProposal(
            proposal_id=f"agent_batch_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
            source="agent_designer",
            target_property=dict(target_property),
            candidates=candidates,
            hypothesis=base.hypothesis,
            rationale=base.reason,
            strategy_counts=strategy_counts,
        )

    def propose_target_schedule(
        self,
        final_target: dict[str, float],
        feedback_signal: dict[str, Any] | None,
        context: dict[str, Any],
        schedule_size: int = 8,
    ) -> TargetScheduleProposal:
        """Design target-space actions for the online closed loop.

        This is the new AgentExplorer contract. It proposes targets for
        InverseDesigner to realize; it does not propose datagen parameters.
        """
        feedback_signal = dict(feedback_signal or {})
        schedule_size = max(1, int(schedule_size))
        schedule_id = f"target_schedule_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"

        final_target = {str(key): float(value) for key, value in final_target.items()}
        anchor_property = self._anchor_property_from_feedback(feedback_signal, context)
        main_error = str(feedback_signal.get("main_error_direction") or self._main_error_direction(final_target, anchor_property))
        suggested_action = str(feedback_signal.get("suggested_next_action") or feedback_signal.get("suggested_strategy") or "")
        if not suggested_action:
            suggested_action = "exploit_near_miss" if anchor_property else "probe_reachability"

        evidence_ids = self._evidence_ids_from_context(context)
        items: list[TargetScheduleItem] = []

        def add_item(
            strategy: str,
            target: dict[str, float],
            reason: str,
            risk: str = "medium",
            samples: int = 1,
            expected_effect: dict[str, Any] | None = None,
        ) -> None:
            if len(items) >= schedule_size:
                return
            target = self._sanitize_target(target, final_target)
            key = tuple(sorted((name, round(value, 6)) for name, value in target.items()))
            for existing in items:
                existing_key = tuple(sorted((name, round(value, 6)) for name, value in existing.target_property.items()))
                if existing_key == key and existing.strategy == strategy:
                    return
            items.append(
                TargetScheduleItem(
                    target_id=f"{schedule_id}_t{len(items) + 1:02d}",
                    target_property=target,
                    strategy=strategy,
                    reason=reason,
                    expected_effect=dict(expected_effect or {}),
                    risk=risk,
                    samples=max(1, int(samples)),
                    based_on_evidence=tuple(evidence_ids[:5]),
                )
            )

        add_item(
            "exploitation",
            final_target,
            "Always include the user's final target as the exploitation anchor.",
            risk="medium",
            samples=1,
            expected_effect={"goal": "direct_solution"},
        )

        if anchor_property:
            midpoint = {
                key: 0.5 * float(anchor_property.get(key, final_target[key])) + 0.5 * final_target[key]
                for key in final_target
            }
            add_item(
                "curriculum",
                midpoint,
                "Interpolate between the best current anchor and the final target.",
                risk="low",
                samples=1,
                expected_effect={"goal": "bridge_reachable_region_to_final_target"},
            )

        compensated = dict(final_target)
        if main_error:
            if main_error in compensated:
                direction = self._error_direction(final_target, anchor_property, main_error)
                if direction == "low":
                    compensated[main_error] = final_target[main_error] * 1.08
                elif direction == "high":
                    compensated[main_error] = final_target[main_error] * 0.92
            add_item(
                "repair",
                compensated,
                f"Compensate the dominant remaining error dimension: {main_error}.",
                risk="medium",
                samples=1,
                expected_effect={"main_error_direction": main_error, "suggested_action": suggested_action},
            )

        reachable_targets = self._reachable_targets_from_context(context)
        for target in reachable_targets:
            add_item(
                "reachability_probe",
                target,
                "Probe a historically reachable target region before moving back toward the final target.",
                risk="low",
                samples=1,
                expected_effect={"goal": "map_inverse_designer_reachability"},
            )

        for scale, risk in ((0.9, "low"), (1.1, "high")):
            offset = {key: value * scale for key, value in final_target.items()}
            add_item(
                "counterfactual",
                offset,
                f"Test a {scale:.1f}x target offset to estimate local target-space sensitivity.",
                risk=risk,
                samples=1,
                expected_effect={"goal": "information_gain"},
            )

        while len(items) < schedule_size:
            index = len(items)
            factor = 1.0 + (0.04 * ((index % 5) - 2))
            exploratory = {key: value * factor for key, value in final_target.items()}
            add_item(
                "explore",
                exploratory,
                "Fill remaining budget with small target-space perturbations around final target.",
                risk="medium",
                samples=1,
                expected_effect={"goal": "local_coverage"},
            )
            if len(items) == index:
                break

        strategy_counts: dict[str, int] = {}
        for item in items:
            strategy_counts[item.strategy] = strategy_counts.get(item.strategy, 0) + 1

        schedule = TargetSchedule(
            schedule_id=schedule_id,
            final_target=final_target,
            scheduled_targets=items,
            hypothesis=(
                "Use target-space scheduling to exploit the final target, bridge from reachable anchors, "
                "and test repair/counterfactual offsets before the next iteration."
            ),
            selection_policy=(
                "Prioritize final-target exploitation, feedback anchor interpolation, main-error repair, "
                "knowledge-backed reachability probes, then local counterfactual perturbations."
            ),
            confidence=0.65 if evidence_ids or anchor_property else 0.45,
            source="agent_explorer_target_schedule",
        )
        return TargetScheduleProposal(
            proposal_id=f"target_schedule_proposal_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
            source="agent_explorer",
            final_target=final_target,
            schedules=[schedule],
            rationale=schedule.selection_policy,
            strategy_counts=strategy_counts,
        )

    @staticmethod
    def _sanitize_target(target: dict[str, float], final_target: dict[str, float]) -> dict[str, float]:
        sanitized = {}
        for key, fallback in final_target.items():
            value = _as_float(target.get(key), fallback)
            sanitized[key] = max(0.0, value)
        return sanitized

    @staticmethod
    def _anchor_property_from_feedback(feedback_signal: dict[str, Any], context: dict[str, Any]) -> dict[str, float]:
        for key in ("next_anchor_sample", "best_near_miss", "best_sample", "best_success"):
            payload = feedback_signal.get(key)
            if isinstance(payload, dict):
                value = payload.get("evaluated_property") or payload.get("property")
                if isinstance(value, dict) and value:
                    return {str(name): float(val) for name, val in value.items()}
        failed = context.get("failed_candidate")
        if isinstance(failed, dict):
            value = failed.get("evaluated_property") or dict(failed.get("evaluation") or {}).get("evaluated_property")
            if isinstance(value, dict) and value:
                return {str(name): float(val) for name, val in value.items()}
        return {}

    @staticmethod
    def _main_error_direction(final_target: dict[str, float], anchor_property: dict[str, float]) -> str:
        if not anchor_property:
            return next(iter(final_target), "")
        best_key = ""
        best_error = -1.0
        for key, target_value in final_target.items():
            observed = _as_float(anchor_property.get(key), target_value)
            scale = abs(target_value) if abs(target_value) > 1e-9 else 1.0
            error = abs(observed - target_value) / scale
            if error > best_error:
                best_error = error
                best_key = key
        return best_key

    @staticmethod
    def _error_direction(final_target: dict[str, float], anchor_property: dict[str, float], key: str) -> str:
        if not anchor_property or key not in anchor_property:
            return ""
        observed = _as_float(anchor_property.get(key), final_target.get(key, 0.0))
        target = _as_float(final_target.get(key), observed)
        if observed < target:
            return "low"
        if observed > target:
            return "high"
        return ""

    @staticmethod
    def _evidence_ids_from_context(context: dict[str, Any]) -> list[str]:
        evidence_ids: list[str] = []
        snapshots = [
            context.get("agent_knowledge_snapshot"),
            context.get("reasoned_agent_knowledge_snapshot"),
            context.get("statistical_knowledge_snapshot"),
        ]
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            for key in ("evidence_citations", "useful_exemplars", "exemplar_samples"):
                value = snapshot.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            evidence_ids.append(item)
                        elif isinstance(item, dict):
                            evidence_ids.append(str(item.get("evidence_id") or item.get("observation_id") or ""))
                elif isinstance(value, dict):
                    for group in value.values():
                        if isinstance(group, list):
                            for item in group:
                                if isinstance(item, dict):
                                    evidence_ids.append(str(item.get("evidence_id") or item.get("observation_id") or ""))
        return [item for item in dict.fromkeys(evidence_ids) if item]

    @staticmethod
    def _reachable_targets_from_context(context: dict[str, Any]) -> list[dict[str, float]]:
        targets: list[dict[str, float]] = []
        for key in ("top_success", "near_miss"):
            for item in context.get(key, [])[:3]:
                if isinstance(item, dict):
                    value = item.get("evaluated_property") or item.get("property")
                    if isinstance(value, dict) and value:
                        targets.append({str(name): float(val) for name, val in value.items()})
        return targets

    @staticmethod
    def _strategy_quotas(batch_size: int) -> dict[str, int]:
        if batch_size >= 77:
            return {
                "exploitation": 20,
                "counterfactual": 12,
                "repair": 10,
                "diversity": 10,
                "high_risk": 4,
            }
        weights = {
            "exploitation": 0.35,
            "counterfactual": 0.22,
            "repair": 0.18,
            "diversity": 0.18,
            "high_risk": 0.07,
        }
        quotas = {key: max(0, int(batch_size * value)) for key, value in weights.items()}
        while sum(quotas.values()) < batch_size:
            quotas["exploitation"] += 1
        return quotas

    def _heuristic_propose(
        self,
        target_property: dict[str, float],
        failed_candidate: KnowledgeSample | None,
        fem_result: dict | None,
        context: dict,
        datagen_schema: dict,
    ) -> DatagenConfig:
        target_stiffness = float(target_property.get("stiffness_proxy", 0.0))
        target_density = float(target_property.get("density_proxy", 0.1))

        top_success = context.get("top_success", [])
        near_miss = context.get("near_miss", [])
        recent_failures = context.get("recent_failures", [])
        reasoned_snapshot = self._reasoned_snapshot(context)
        statistical_snapshot = self._statistical_snapshot(context, reasoned_snapshot)
        interpretation = dict(reasoned_snapshot.get("interpretation") or {})
        strategy_prior = dict(interpretation.get("target_specific_strategy_prior") or {})
        group_stats = statistical_snapshot.get("group_stats", [])
        config_pattern_stats = statistical_snapshot.get("config_pattern_stats", [])
        failure_patterns = statistical_snapshot.get("failure_patterns", [])

        group = failed_candidate.symmetry if failed_candidate else datagen_schema["default_group"]
        if top_success:
            group = str(_sample_value(top_success[0], "symmetry", group))
        elif group_stats:
            best_group = max(
                group_stats,
                key=lambda item: (
                    float(item.get("success_rate", 0.0)),
                    float(item.get("near_miss_rate", 0.0)),
                ),
            )
            if best_group.get("group"):
                group = str(best_group["group"])

        current_stiffness = 0.0
        if fem_result:
            current_stiffness = float(fem_result.get("evaluated_property", {}).get("stiffness_proxy", 0.0))
        elif failed_candidate:
            current_stiffness = float(failed_candidate.evaluated_property.get("stiffness_proxy", 0.0))

        max_bars = 10
        connectivity_pattern = "default"
        hypothesis = "Explore nearby symmetry-constrained sparse truss families."
        reason = "No prior failed candidate was available."
        topology_type = "sparse_truss"
        basic_unit_type = "edge_face_center_19node"
        unit_cell_type = "symmetry_expanded_truss"
        density_range = (max(0.01, target_density - 0.03), target_density + 0.03)

        matching_patterns = [item for item in config_pattern_stats if item.get("symmetry") == group]
        if matching_patterns:
            best_pattern = matching_patterns[0]
            max_bars = int(best_pattern.get("max_bars") or max_bars)
            connectivity_pattern = str(best_pattern.get("connectivity_pattern") or connectivity_pattern)
            topology_type = str(best_pattern.get("topology_type") or topology_type)
            basic_unit_type = str(best_pattern.get("basic_unit_type") or basic_unit_type)
            unit_cell_type = str(best_pattern.get("unit_cell_type") or unit_cell_type)
            hypothesis = "Reuse the best-performing structural pattern observed in this symmetry family."
            reason = "Refined knowledge indicates this configuration pattern has the strongest empirical prior."

        if failed_candidate:
            max_bars = int(failed_candidate.parameter_config.get("max_bars", 10))
            connectivity_pattern = failed_candidate.connectivity_pattern
            topology_type = failed_candidate.topology_type
            basic_unit_type = failed_candidate.basic_unit_type
            unit_cell_type = failed_candidate.unit_cell_type
            if current_stiffness < target_stiffness:
                max_bars = min(max_bars + 2, 14)
                connectivity_pattern = "higher_diagonal_density"
                hypothesis = "Increase bar richness to improve stiffness proxy."
                reason = "The previous candidate underperformed on stiffness_proxy."
            else:
                max_bars = max(max_bars - 1, 6)
                hypothesis = "Reduce over-connected topology while preserving property level."
                reason = "The previous candidate was dense enough; explore leaner topologies."

        if near_miss:
            group = str(_sample_value(near_miss[0], "symmetry", group))
            topology_type = str(_sample_value(near_miss[0], "topology_type", topology_type))
            basic_unit_type = str(_sample_value(near_miss[0], "basic_unit_type", basic_unit_type))
            unit_cell_type = str(_sample_value(near_miss[0], "unit_cell_type", unit_cell_type))
            hypothesis = "Explore around near-miss symmetry families."
            reason = "Near-miss samples indicate this symmetry family is close to the target."
        elif top_success:
            reason = "Known successful samples provide the best structural prior."
        elif failure_patterns:
            dominant_failure = failure_patterns[0].get("failure_reason", "")
            if dominant_failure == "stiffness_insufficient":
                hypothesis = "Prior failures suggest increasing stiffness while preserving density."
            elif dominant_failure == "density_too_high":
                hypothesis = "Prior failures suggest density control is the main bottleneck."
            elif dominant_failure:
                hypothesis = f"Prior failures are dominated by {dominant_failure}; adjust the configuration accordingly."

        preferred_direction = str(strategy_prior.get("preferred_direction", "")).strip()
        preferred_intervention = str(strategy_prior.get("preferred_intervention", "")).strip()
        if preferred_direction == "increase_connectivity_or_load_paths":
            connectivity_pattern = "higher_diagonal_density"
            max_bars = min(max_bars + 1, 14)
            hypothesis = "Reasoned knowledge favors stronger load paths for this target."
            reason = strategy_prior.get("rationale", "KnowledgeInterpreter suggests increasing connectivity or load paths.")
        elif preferred_direction == "tighten_density_control":
            density_center = target_density
            hypothesis = "Reasoned knowledge identifies density control as the primary bottleneck."
            reason = strategy_prior.get("rationale", "KnowledgeInterpreter suggests tighter density control.")
            density_range = (max(0.01, density_center - 0.015), density_center + 0.015)

        sample_count = 8 if not recent_failures else min(12, 4 + len(recent_failures))
        relative_gap = 0.0
        if target_stiffness:
            relative_gap = max(target_stiffness - current_stiffness, 0.0) / abs(target_stiffness)
        confidence = max(0.2, min(0.85, 0.55 + 0.25 * (1.0 - relative_gap)))
        expected_stiffness = current_stiffness
        if current_stiffness < target_stiffness:
            expected_stiffness = current_stiffness + 0.5 * (target_stiffness - current_stiffness)
        elif current_stiffness:
            expected_stiffness = min(current_stiffness, target_stiffness * 1.1)

        failure_analysis = {}
        if failed_candidate or fem_result:
            observed_property = fem_result.get("evaluated_property", {}) if fem_result else failed_candidate.evaluated_property
            previous_label = fem_result.get("label", failed_candidate.label if failed_candidate else "") if fem_result else failed_candidate.label
            failure_analysis = {
                "previous_sample_id": failed_candidate.structure_id if failed_candidate else "",
                "previous_label": previous_label,
                "observed_property": dict(observed_property),
                "target_property": dict(target_property),
                "underperformed_property": "stiffness_proxy" if current_stiffness < target_stiffness else "",
            }
        strategy = "increase_connectivity" if current_stiffness < target_stiffness else "local_refinement"
        if near_miss:
            strategy = "near_miss_refinement"
        if preferred_intervention:
            strategy = preferred_intervention
        suggestion_id = f"agent_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"

        return DatagenConfig(
            suggestion_id=suggestion_id,
            parent_sample_id=failed_candidate.structure_id if failed_candidate else "",
            target_property=dict(target_property),
            expected_property={
                "stiffness_proxy": float(expected_stiffness),
                "density_proxy": float(target_density),
            },
            objective="match_target_property",
            confidence=confidence,
            group=group,
            symmetry=group,
            basic_unit_type=basic_unit_type,
            unit_cell_type=unit_cell_type,
            topology_type=topology_type,
            num_samples=sample_count,
            max_bars=max_bars,
            rho_target=target_density,
            connectivity_pattern=connectivity_pattern,
            density_range=density_range,
            design_search_parameters=DesignSearchParameters(
                group=group,
                symmetry=group,
                max_bars=max_bars,
                rho_target=target_density,
            ),
            hypothesis=hypothesis,
            reason=reason,
            failure_analysis=failure_analysis,
            exploration_strategy=strategy,
            tags=("agent_suggestion", strategy, group),
        )

    def _try_llm_proposal(
        self,
        target_property: dict[str, float],
        failed_candidate: KnowledgeSample | None,
        fem_result: dict | None,
        context: dict,
        datagen_schema: dict,
        heuristic: DatagenConfig,
    ) -> tuple[DatagenConfig | None, str | None]:
        if not self.enable_llm:
            return None, None
        if self._instance_llm_disable_reason:
            return None, self._instance_llm_disable_reason
        if self.__class__._global_llm_disable_reason:
            return None, self.__class__._global_llm_disable_reason

        try:
            payload = self._request_llm_payload(
                target_property=target_property,
                failed_candidate=failed_candidate,
                fem_result=fem_result,
                context=context,
                datagen_schema=datagen_schema,
                heuristic=heuristic,
            )
            return self._config_from_llm_payload(payload, heuristic, failed_candidate), None
        except Exception as exc:  # pragma: no cover - exercised in integration fallback
            reason = f"llm_unavailable: {type(exc).__name__}: {exc}"
            self._instance_llm_disable_reason = reason
            self.__class__._global_llm_disable_reason = reason
            return None, reason

    def _http_session(self) -> requests.Session:
        if self._session is None:
            session = requests.Session()
            session.trust_env = False
            self._session = session
        return self._session

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _resolve_model(self) -> str:
        if self.model:
            return self.model
        response = self._http_session().get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        model_ids = [str(item.get("id", "")).strip() for item in payload.get("data", []) if item.get("id")]
        for preferred in MODEL_PREFERENCE:
            if preferred in model_ids:
                self.model = preferred
                return self.model
        if not model_ids:
            raise RuntimeError("no models available from agent explorer endpoint")
        self.model = model_ids[0]
        return self.model

    def _request_llm_payload(
        self,
        target_property: dict[str, float],
        failed_candidate: KnowledgeSample | None,
        fem_result: dict | None,
        context: dict,
        datagen_schema: dict,
        heuristic: DatagenConfig,
    ) -> dict[str, Any]:
        model = self._resolve_model()
        reasoned_snapshot = self._reasoned_snapshot(context)
        statistical_snapshot = self._statistical_snapshot(context, reasoned_snapshot)
        prompt_payload = {
            "target_property": dict(target_property),
            "failed_candidate": self._summarize_sample(failed_candidate),
            "fem_result": fem_result or {},
            "heuristic_baseline": heuristic.to_dict(),
            "datagen_schema": datagen_schema,
            "knowledge_context": {
                "top_success": [self._summarize_sample(item) for item in context.get("top_success", [])[:2]],
                "near_miss": [self._summarize_sample(item) for item in context.get("near_miss", [])[:2]],
                "recent_failures": [self._summarize_sample(item) for item in context.get("recent_failures", [])[:3]],
                "dataset_statistics": context.get("dataset_statistics", {}),
                "reasoned_agent_knowledge_snapshot": {
                    "interpretation": self._limit_payload(reasoned_snapshot.get("interpretation", {}), 5),
                    "statistical": {
                        "group_stats": self._limit_payload(statistical_snapshot.get("group_stats", []), 5),
                        "config_pattern_stats": self._limit_payload(statistical_snapshot.get("config_pattern_stats", []), 5),
                        "failure_patterns": self._limit_payload(statistical_snapshot.get("failure_patterns", []), 5),
                        "exemplar_samples": self._limit_payload(statistical_snapshot.get("exemplar_samples", {}), 5),
                    },
                },
            },
        }
        system_prompt = (
            "You are AgentExplorer in a truss-material closed loop. "
            "You must reason from failures, guess promising next structure parameters, imagine a plausible motif, "
            "and suggest the next DatagenConfig. Respond with one JSON object only."
        )
        user_prompt = (
            "Return a JSON object with these fields: "
            "group, symmetry, basic_unit_type, unit_cell_type, topology_type, connectivity_pattern, "
            "max_bars, rho_target, density_range, num_samples, expected_property, hypothesis, reason, "
            "imagination, failure_reason_guess, exploration_strategy, confidence. "
            "Prefer the heuristic_baseline unless the evidence clearly supports a better change. "
            "Do not output markdown.\n\n"
            f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
        )
        input_text = f"{system_prompt}\n\n{user_prompt}"
        response = self._http_session().post(
            f"{self.base_url}/responses",
            headers=self._headers(),
            json={
                "model": model,
                "input": input_text,
                "reasoning": {"effort": "medium"},
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            message = response.text.strip().replace("\n", " ")
            raise RuntimeError(f"responses_failed status={response.status_code} body={message[:300]}")
        content = self._extract_responses_output(response.text)
        return _extract_json_object(content)

    @staticmethod
    def _extract_responses_output(raw_text: str) -> str:
        output_chunks: list[str] = []
        done_texts: list[str] = []
        for event_name, data in _iter_sse_events(raw_text):
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event_name == "response.output_text.delta":
                delta = payload.get("delta", "")
                if isinstance(delta, str) and delta:
                    output_chunks.append(delta)
                continue
            if event_name == "response.failed":
                error = payload.get("response", {}).get("error") or payload.get("error") or payload
                raise RuntimeError(json.dumps(error, ensure_ascii=False))
            if event_name.endswith(".done"):
                text = payload.get("text")
                if isinstance(text, str) and text:
                    done_texts.append(text)
        content = "".join(output_chunks).strip()
        if content:
            return content
        fallback = "".join(done_texts).strip()
        if fallback:
            return fallback
        raw = raw_text.strip()
        if raw:
            return raw
        raise RuntimeError("responses returned empty output")

    def _config_from_llm_payload(
        self,
        payload: dict[str, Any],
        heuristic: DatagenConfig,
        failed_candidate: KnowledgeSample | None,
    ) -> DatagenConfig:
        density_range = payload.get("density_range", heuristic.density_range)
        if not isinstance(density_range, (list, tuple)) or len(density_range) != 2:
            density_range = heuristic.density_range
        density_min = _as_float(density_range[0], heuristic.density_range[0])
        density_max = _as_float(density_range[1], heuristic.density_range[1])
        if density_min > density_max:
            density_min, density_max = density_max, density_min

        expected_property = dict(heuristic.expected_property)
        llm_expected = payload.get("expected_property")
        if isinstance(llm_expected, dict):
            for key in ("stiffness_proxy", "density_proxy"):
                if key in llm_expected:
                    expected_property[key] = _as_float(llm_expected[key], expected_property.get(key, 0.0))

        failure_analysis = dict(heuristic.failure_analysis)
        imagination = str(payload.get("imagination", "")).strip()
        failure_reason_guess = str(payload.get("failure_reason_guess", "")).strip()
        if imagination:
            failure_analysis["llm_imagination"] = imagination
        if failure_reason_guess:
            failure_analysis["llm_failure_reason_guess"] = failure_reason_guess
        failure_analysis["llm_model"] = self._resolve_model()
        failure_analysis["llm_mode"] = "reason_guess_imagine_suggest"

        group = str(payload.get("group", heuristic.group)).strip() or heuristic.group
        symmetry = str(payload.get("symmetry", payload.get("group", heuristic.symmetry))).strip() or heuristic.symmetry
        max_bars = max(1, _as_int(payload.get("max_bars"), heuristic.max_bars))
        rho_target = _clamp(_as_float(payload.get("rho_target"), heuristic.rho_target), 0.001, 10.0)
        return DatagenConfig(
            suggestion_id=heuristic.suggestion_id,
            parent_sample_id=failed_candidate.structure_id if failed_candidate else heuristic.parent_sample_id,
            source="agent_exploration_llm",
            target_property=dict(heuristic.target_property),
            expected_property=expected_property,
            objective=str(payload.get("objective", heuristic.objective)).strip() or heuristic.objective,
            confidence=_clamp(_as_float(payload.get("confidence"), heuristic.confidence), 0.05, 0.99),
            group=group,
            basic_size=heuristic.basic_size,
            num_samples=max(1, _as_int(payload.get("num_samples"), heuristic.num_samples)),
            workers=heuristic.workers,
            batch=heuristic.batch,
            print_every=heuristic.print_every,
            run_dir=heuristic.run_dir,
            symmetry=symmetry,
            basic_unit_type=str(payload.get("basic_unit_type", heuristic.basic_unit_type)).strip() or heuristic.basic_unit_type,
            unit_cell_type=str(payload.get("unit_cell_type", heuristic.unit_cell_type)).strip() or heuristic.unit_cell_type,
            topology_type=str(payload.get("topology_type", heuristic.topology_type)).strip() or heuristic.topology_type,
            connectivity_pattern=str(payload.get("connectivity_pattern", heuristic.connectivity_pattern)).strip() or heuristic.connectivity_pattern,
            max_bars=max_bars,
            rho_target=rho_target,
            density_range=(density_min, density_max),
            parameter_ranges=dict(heuristic.parameter_ranges),
            sampling_strategy=heuristic.sampling_strategy,
            constraints=dict(heuristic.constraints),
            design_search_parameters=DesignSearchParameters(
                group=group,
                symmetry=symmetry,
                max_bars=max_bars,
                rho_target=rho_target,
                parameter_ranges=dict(heuristic.parameter_ranges),
                constraints=dict(heuristic.constraints),
            ),
            hypothesis=str(payload.get("hypothesis", heuristic.hypothesis)).strip() or heuristic.hypothesis,
            reason=str(payload.get("reason", heuristic.reason)).strip() or heuristic.reason,
            failure_analysis=failure_analysis,
            exploration_strategy=str(payload.get("exploration_strategy", heuristic.exploration_strategy)).strip() or heuristic.exploration_strategy,
            tags=_unique_tags(heuristic.tags, ("llm_suggestion",)),
        )

    @staticmethod
    def _summarize_sample(sample: KnowledgeSample | dict | None) -> dict[str, Any]:
        if sample is None:
            return {}
        if isinstance(sample, dict):
            parameter_config = dict(sample.get("parameter_config") or {})
            return {
                "structure_id": sample.get("structure_id", ""),
                "symmetry": sample.get("symmetry") or sample.get("group", ""),
                "unit_cell_type": sample.get("unit_cell_type", ""),
                "basic_unit_type": sample.get("basic_unit_type", ""),
                "topology_type": sample.get("topology_type", ""),
                "connectivity_pattern": sample.get("connectivity_pattern") or parameter_config.get("connectivity_pattern", ""),
                "parameter_config": parameter_config,
                "evaluated_property": dict(sample.get("evaluated_property") or sample.get("property") or {}),
                "property_error": dict(sample.get("property_error") or sample.get("error") or {}),
                "label": sample.get("label", ""),
                "source": sample.get("source", ""),
            }
        return {
            "structure_id": sample.structure_id,
            "symmetry": sample.symmetry,
            "unit_cell_type": sample.unit_cell_type,
            "basic_unit_type": sample.basic_unit_type,
            "topology_type": sample.topology_type,
            "connectivity_pattern": sample.connectivity_pattern,
            "parameter_config": dict(sample.parameter_config),
            "evaluated_property": dict(sample.evaluated_property),
            "property_error": dict(sample.property_error),
            "label": sample.label,
            "source": sample.source,
        }

    @staticmethod
    def _limit_payload(payload: Any, limit: int) -> Any:
        if isinstance(payload, list):
            return payload[:limit]
        if isinstance(payload, tuple):
            return list(payload[:limit])
        if isinstance(payload, dict):
            limited = {}
            for key, value in payload.items():
                if isinstance(value, list):
                    limited[key] = value[:limit]
                elif isinstance(value, tuple):
                    limited[key] = list(value[:limit])
                else:
                    limited[key] = value
            return limited
        return payload

    def select_samples(
        self,
        target_property: dict[str, float],
        evaluated_samples: list[KnowledgeSample],
    ) -> list[KnowledgeSample]:
        """
        Legacy compatibility helper.

        The closed-loop controller now extracts FeedbackSignal in
        src/Scheduler/feedback.py. AgentExplorer should propose candidates, not
        own target-aware feedback extraction.
        """
        del target_property
        successes = [sample for sample in evaluated_samples if sample.label == "success"]
        near_misses = [sample for sample in evaluated_samples if sample.label == "near_miss"]
        failures = [sample for sample in evaluated_samples if sample.label == "failure"]

        selected = []
        selected.extend(successes[:5])
        selected.extend(near_misses[:5])
        selected.extend(failures[:3])
        if not selected:
            return evaluated_samples[:5]
        return selected
