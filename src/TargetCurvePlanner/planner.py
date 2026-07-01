from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from ..closed_loop_contracts import TargetCurvePlan, TargetCurvePlanItem
from ..curve_targets import normalize_target_property


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _scale_sequence(values: Any, scale: float) -> list[float]:
    if not isinstance(values, (list, tuple)):
        return values
    return [_safe_float(value) * scale for value in values]


def scale_target_curve(target_property: dict[str, Any], scale: float) -> dict[str, Any]:
    """Scale stress-like targets while preserving strain grids."""

    scaled: dict[str, Any] = {}
    for key, value in target_property.items():
        key_text = str(key)
        key_lower = key_text.lower()
        if key_lower in {"strain", "strain_grid", "x"}:
            scaled[key_text] = list(value) if isinstance(value, (list, tuple)) else value
        elif key_lower in {"stress", "stress_curve", "y"}:
            scaled[key_text] = _scale_sequence(value, scale)
        elif isinstance(value, (int, float)):
            scaled[key_text] = float(value) * scale
        elif isinstance(value, (list, tuple)) and "stress" in key_lower:
            scaled[key_text] = _scale_sequence(value, scale)
        else:
            scaled[key_text] = value
    return normalize_target_property(scaled)


class TargetCurvePlanner:
    """Deterministic target stress-curve planner.

    The planner deliberately has no KnowledgeBase, FeedbackSignal, LLM, or
    hypothesis dependency. It produces reproducible target batches from the
    final target, batch size, and iteration index.
    """

    def __init__(
        self,
        *,
        policy: str = "mixed_deterministic",
        base_scales: tuple[float, ...] = (1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15),
    ) -> None:
        self.policy = str(policy or "scale_sweep")
        self.base_scales = tuple(float(value) for value in base_scales) or (1.0,)

    def plan(
        self,
        final_target: dict[str, Any],
        *,
        iteration: int = 0,
        batch_size: int = 1,
        samples_per_target: int = 1,
    ) -> TargetCurvePlan:
        normalized_target = normalize_target_property(final_target)
        batch_size = max(1, int(batch_size))
        samples_per_target = max(1, int(samples_per_target))
        iteration = max(0, int(iteration))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        plan_id = f"target_curve_plan_{iteration:03d}_{timestamp}"
        targets = self._mixed_targets(normalized_target, iteration=iteration, batch_size=batch_size)
        items = []
        for index, item in enumerate(targets, start=1):
            items.append(
                TargetCurvePlanItem(
                    target_id=f"{plan_id}_c{index:02d}",
                    target_property=item["target"],
                    strategy=item["strategy"],
                    samples=samples_per_target,
                    planner_meta={
                        **dict(item.get("meta") or {}),
                        "policy": self.policy,
                        "iteration": iteration,
                        "deterministic": True,
                    },
                )
            )
        return TargetCurvePlan(
            plan_id=plan_id,
            final_target=normalized_target,
            target_curves=items,
            planner=type(self).__name__,
            policy=self.policy,
            iteration=iteration,
        )

    def _mixed_targets(self, target: dict[str, Any], *, iteration: int, batch_size: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = [
            {
                "target": normalize_target_property(target),
                "strategy": "final_target",
                "meta": {"component": "exact_target", "scale": 1.0},
            }
        ]
        if batch_size <= 1:
            return items

        local_budget = max(0, int(round((batch_size - 1) * 0.48)))
        boundary_budget = max(0, int(round((batch_size - 1) * 0.26)))
        curriculum_budget = max(0, (batch_size - 1) - local_budget - boundary_budget)

        for index in range(local_budget):
            amplitude = 0.02 + 0.005 * ((index + iteration) % 4)
            sign = -1.0 if index % 2 else 1.0
            scale = 1.0 + sign * amplitude
            phase = (index + 1 + iteration) * 0.7
            items.append(
                {
                    "target": self._shape_perturb(target, scale=scale, amplitude=amplitude * 0.5, phase=phase),
                    "strategy": "local_perturbation",
                    "meta": {
                        "component": "local_perturbation",
                        "scale": scale,
                        "shape_amplitude": amplitude * 0.5,
                        "phase": phase,
                    },
                }
            )

        boundary_scales = self._scales_for_iteration(iteration=iteration, batch_size=max(1, boundary_budget + 1))[1:]
        while len(boundary_scales) < boundary_budget:
            cursor = len(boundary_scales) + 1
            boundary_scales.append(0.80 + 0.04 * cursor if cursor % 2 else 1.20 - 0.04 * cursor)
        for scale in boundary_scales[:boundary_budget]:
            items.append(
                {
                    "target": scale_target_curve(target, scale),
                    "strategy": self._strategy_for_scale(scale),
                    "meta": {"component": "boundary_sweep", "scale": scale},
                }
            )

        for index in range(curriculum_budget):
            alpha = min(1.0, 0.35 + 0.10 * iteration + 0.05 * index)
            easy_scale = 0.65 + 0.05 * (index % 4)
            items.append(
                {
                    "target": self._interpolate_curve(scale_target_curve(target, easy_scale), target, alpha),
                    "strategy": "curriculum",
                    "meta": {
                        "component": "curriculum",
                        "easy_scale": easy_scale,
                        "alpha_to_final": alpha,
                    },
                }
            )

        return items[:batch_size]

    @staticmethod
    def _shape_perturb(target: dict[str, Any], *, scale: float, amplitude: float, phase: float) -> dict[str, Any]:
        normalized = normalize_target_property(target)
        stress = normalized.get("stress")
        if not isinstance(stress, list) or not stress:
            return scale_target_curve(normalized, scale)
        n = max(1, len(stress) - 1)
        perturbed = []
        for index, value in enumerate(stress):
            envelope = math.sin(math.pi * index / n)
            wave = math.sin(2.0 * math.pi * index / max(1, n) + phase)
            factor = max(0.0, scale + amplitude * envelope * wave)
            perturbed.append(_safe_float(value) * factor)
        return normalize_target_property(
            {
                **normalized,
                "stress": perturbed,
            }
        )

    @staticmethod
    def _interpolate_curve(easy: dict[str, Any], target: dict[str, Any], alpha: float) -> dict[str, Any]:
        lhs = normalize_target_property(easy)
        rhs = normalize_target_property(target)
        lhs_stress = lhs.get("stress")
        rhs_stress = rhs.get("stress")
        if not isinstance(lhs_stress, list) or not isinstance(rhs_stress, list) or len(lhs_stress) != len(rhs_stress):
            return rhs
        weight = max(0.0, min(1.0, float(alpha)))
        stress = [
            _safe_float(l_value) * (1.0 - weight) + _safe_float(r_value) * weight
            for l_value, r_value in zip(lhs_stress, rhs_stress)
        ]
        return normalize_target_property({**rhs, "stress": stress})

    def _scales_for_iteration(self, *, iteration: int, batch_size: int) -> list[float]:
        scales = list(self.base_scales)
        if batch_size > len(scales):
            step = 0.05
            cursor = 1
            while len(scales) < batch_size:
                offset = step * (cursor + iteration)
                scales.extend([max(0.0, 1.0 - offset), 1.0 + offset])
                cursor += 1
        if iteration > 0 and len(scales) > 1:
            head, tail = scales[0], scales[1:]
            shift = iteration % len(tail)
            scales = [head] + tail[shift:] + tail[:shift]
        return scales[:batch_size]

    @staticmethod
    def _strategy_for_scale(scale: float) -> str:
        if abs(scale - 1.0) < 1e-9:
            return "final_target"
        if scale < 1.0:
            return "lower_boundary_sweep"
        return "upper_boundary_sweep"
