from __future__ import annotations

import math
from typing import Any


CURVE_STRAIN_MIN = 0.0
CURVE_STRAIN_MAX = 0.3
CURVE_GRID_LENGTH = 256
CURVE_GRID_VERSION = "strain_0_0.3_256_v1"
CURVE_TYPE_STRESS = "stress_curve"
EPS = 1.0e-12


def fixed_strain_grid(length: int = CURVE_GRID_LENGTH) -> list[float]:
    length = int(length)
    if length <= 1:
        return [CURVE_STRAIN_MIN]
    span = CURVE_STRAIN_MAX - CURVE_STRAIN_MIN
    return [CURVE_STRAIN_MIN + span * index / (length - 1) for index in range(length)]


def as_float_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    values: list[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(number):
            return []
        values.append(number)
    return values


def is_pair_curve(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first = value[0]
    return isinstance(first, (list, tuple)) and len(first) >= 2


def resample_stress_curve(
    strain: list[float],
    stress: list[float],
    target_grid: list[float] | None = None,
    *,
    clamp_negative: bool = True,
) -> list[float]:
    target_grid = list(target_grid or fixed_strain_grid())
    if not strain or not stress or not target_grid:
        return []

    pairs = []
    for x_value, y_value in zip(strain, stress):
        try:
            x = float(x_value)
            y = float(y_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        pairs.append((x, max(0.0, y) if clamp_negative else y))
    if not pairs:
        return []

    pairs.sort(key=lambda item: item[0])
    merged: list[tuple[float, float]] = []
    index = 0
    while index < len(pairs):
        current_x = pairs[index][0]
        values = [pairs[index][1]]
        index += 1
        while index < len(pairs) and abs(pairs[index][0] - current_x) <= EPS:
            values.append(pairs[index][1])
            index += 1
        merged.append((current_x, sum(values) / len(values)))

    if len(merged) == 1:
        return [merged[0][1] for _ in target_grid]

    xs = [item[0] for item in merged]
    ys = [item[1] for item in merged]
    output: list[float] = []
    cursor = 0
    for target_x in target_grid:
        if target_x <= xs[0]:
            output.append(ys[0])
            continue
        if target_x >= xs[-1]:
            output.append(ys[-1])
            continue
        while cursor + 1 < len(xs) and xs[cursor + 1] < target_x:
            cursor += 1
        x0, x1 = xs[cursor], xs[cursor + 1]
        y0, y1 = ys[cursor], ys[cursor + 1]
        if abs(x1 - x0) <= EPS:
            output.append(y1)
        else:
            weight = (target_x - x0) / (x1 - x0)
            output.append(y0 * (1.0 - weight) + y1 * weight)
    return output


def normalize_stress_curve_target(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    payload = dict(payload or {})
    target = payload.get("target")
    if isinstance(target, dict):
        nested = normalize_stress_curve_target(target)
        if nested is not None:
            return nested

    strain: list[float] = []
    stress: list[float] = []
    for key in ("stress", "stress_curve", "target_curve", "y", "curve"):
        value = payload.get(key)
        if _is_pair_curve_value(value):
            strain = []
            stress = []
            for item in value:
                try:
                    strain.append(float(item[0]))
                    stress.append(float(item[1]))
                except (TypeError, ValueError):
                    strain = []
                    stress = []
                    break
            if stress:
                break
        stress = as_float_list(value)
        if stress:
            strain = as_float_list(payload.get("strain_grid") or payload.get("strain") or payload.get("x"))
            if not strain:
                strain = fixed_strain_grid(len(stress))
            break

    if not stress:
        return None
    if not strain:
        strain = fixed_strain_grid(len(stress))

    grid = fixed_strain_grid()
    normalized_stress = resample_stress_curve(strain, stress, grid, clamp_negative=True)
    if len(normalized_stress) != len(grid):
        return None
    return {
        "type": CURVE_TYPE_STRESS,
        "strain_grid": grid,
        "stress": normalized_stress,
        "grid_version": CURVE_GRID_VERSION,
    }


def normalize_target_property(payload: dict[str, Any] | None) -> dict[str, Any]:
    curve = normalize_stress_curve_target(payload)
    if curve is not None:
        return curve
    return dict(payload or {})


def is_stress_curve_target(payload: dict[str, Any] | None) -> bool:
    return normalize_stress_curve_target(payload) is not None


def curve_summary_metrics(curve_payload: dict[str, Any] | None) -> dict[str, float]:
    curve = normalize_stress_curve_target(curve_payload)
    if curve is None:
        return {}
    strain = list(curve["strain_grid"])
    stress = list(curve["stress"])
    if not strain or not stress:
        return {}

    peak = max(stress) if stress else 0.0
    energy = 0.0
    for lhs_index in range(len(strain) - 1):
        dx = abs(strain[lhs_index + 1] - strain[lhs_index])
        energy += 0.5 * (stress[lhs_index] + stress[lhs_index + 1]) * dx

    initial_pairs = [(x, y) for x, y in zip(strain, stress) if 0.0 <= x <= 0.05]
    if len(initial_pairs) >= 2:
        x0, y0 = initial_pairs[0]
        x1, y1 = initial_pairs[-1]
        initial_modulus = (y1 - y0) / max(x1 - x0, EPS)
    else:
        x0, y0 = strain[0], stress[0]
        initial_modulus = y0 / max(x0, EPS) if x0 > 0 else 0.0

    return {
        "peak_stress": float(peak),
        "energy_absorption": float(energy),
        "initial_modulus": float(initial_modulus),
    }


def stress_curve_error_metrics(
    target_payload: dict[str, Any] | None,
    observed_payload: dict[str, Any] | None,
) -> dict[str, float]:
    target = normalize_stress_curve_target(target_payload)
    observed = normalize_stress_curve_target(observed_payload)
    if target is None or observed is None:
        return {}

    target_stress = list(target["stress"])
    observed_stress = list(observed["stress"])
    if not target_stress or len(target_stress) != len(observed_stress):
        return {}

    abs_errors = [abs(lhs - rhs) for lhs, rhs in zip(observed_stress, target_stress)]
    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(value * value for value in abs_errors) / len(abs_errors))
    scale = max(max(abs(value) for value in target_stress), EPS)
    target_summary = curve_summary_metrics(target)
    observed_summary = curve_summary_metrics(observed)

    def rel_error(key: str) -> float:
        target_value = float(target_summary.get(key, 0.0))
        observed_value = float(observed_summary.get(key, 0.0))
        return abs(observed_value - target_value) / max(abs(target_value), EPS)

    def rel_delta(key: str) -> float:
        target_value = float(target_summary.get(key, 0.0))
        observed_value = float(observed_summary.get(key, 0.0))
        return (observed_value - target_value) / max(abs(target_value), EPS)

    return {
        "curve_mae": float(mae),
        "curve_rmse": float(rmse),
        "curve_nmae": float(mae / scale),
        "curve_nrmse": float(rmse / scale),
        "curve_points": float(len(abs_errors)),
        "peak_error": rel_error("peak_stress"),
        "energy_error": rel_error("energy_absorption"),
        "initial_modulus_error": rel_error("initial_modulus"),
        "peak_relative_delta": rel_delta("peak_stress"),
        "energy_relative_delta": rel_delta("energy_absorption"),
        "initial_modulus_relative_delta": rel_delta("initial_modulus"),
        "peak_target": float(target_summary.get("peak_stress", 0.0)),
        "peak_observed": float(observed_summary.get("peak_stress", 0.0)),
        "energy_target": float(target_summary.get("energy_absorption", 0.0)),
        "energy_observed": float(observed_summary.get("energy_absorption", 0.0)),
        "initial_modulus_target": float(target_summary.get("initial_modulus", 0.0)),
        "initial_modulus_observed": float(observed_summary.get("initial_modulus", 0.0)),
    }


def blend_stress_curves(
    lhs_payload: dict[str, Any],
    rhs_payload: dict[str, Any],
    rhs_weight: float = 0.5,
) -> dict[str, Any] | None:
    lhs = normalize_stress_curve_target(lhs_payload)
    rhs = normalize_stress_curve_target(rhs_payload)
    if lhs is None or rhs is None:
        return None
    weight = max(0.0, min(1.0, float(rhs_weight)))
    stress = [
        lhs_value * (1.0 - weight) + rhs_value * weight
        for lhs_value, rhs_value in zip(lhs["stress"], rhs["stress"])
    ]
    return {
        "type": CURVE_TYPE_STRESS,
        "strain_grid": list(rhs["strain_grid"]),
        "stress": stress,
        "grid_version": CURVE_GRID_VERSION,
    }


def scale_stress_curve(payload: dict[str, Any], scale: float) -> dict[str, Any] | None:
    curve = normalize_stress_curve_target(payload)
    if curve is None:
        return None
    factor = max(0.0, float(scale))
    return {
        "type": CURVE_TYPE_STRESS,
        "strain_grid": list(curve["strain_grid"]),
        "stress": [max(0.0, factor * value) for value in curve["stress"]],
        "grid_version": CURVE_GRID_VERSION,
    }


def repair_stress_curve(payload: dict[str, Any], biases: list[str]) -> dict[str, Any] | None:
    curve = normalize_stress_curve_target(payload)
    if curve is None:
        return None
    bias_set = {str(item) for item in biases if item and item != "none"}
    strain = list(curve["strain_grid"])
    stress = list(curve["stress"])
    repaired: list[float] = []
    for x, y in zip(strain, stress):
        factor = 1.0
        if "under_realizes_peak" in bias_set and x >= 0.18:
            factor *= 1.08
        if "under_realizes_energy" in bias_set and 0.05 <= x <= 0.25:
            factor *= 1.06
        if "too_soft_initially" in bias_set and x <= 0.05:
            factor *= 1.08
        if not bias_set:
            factor *= 1.04
        repaired.append(max(0.0, y * factor))
    return {
        "type": CURVE_TYPE_STRESS,
        "strain_grid": strain,
        "stress": repaired,
        "grid_version": CURVE_GRID_VERSION,
    }


def _is_pair_curve_value(value: Any) -> bool:
    return is_pair_curve(value)
