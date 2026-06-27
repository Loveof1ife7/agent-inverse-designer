from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Iterable

from ..closed_loop_contracts import DatagenConfig, MetaCandidate


def _now_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


class MetaSpace:
    """Current executable Meta schema and simple intervention operators."""

    DEFAULT_GROUP = "P222"
    CONNECTIVITY_PATTERNS = (
        "default",
        "higher_diagonal_density",
        "reduced_connectivity",
        "center_reinforced",
    )
    STRATEGY_QUOTAS_77 = {
        "inverse": 21,
        "exploitation": 20,
        "counterfactual": 12,
        "repair": 10,
        "diversity": 10,
        "high_risk": 4,
    }

    def schema(self) -> dict:
        return {
            "fields": list(DatagenConfig.__dataclass_fields__),
            "groups": [self.DEFAULT_GROUP],
            "connectivity_patterns": list(self.CONNECTIVITY_PATTERNS),
            "max_bars": {"min": 1, "max": 20},
            "rho_target": {"min": 0.001, "max": 1.0},
            "density_range": {"min": 0.001, "max": 1.0},
            "strategy_quotas_77": dict(self.STRATEGY_QUOTAS_77),
        }

    def repair(self, meta: DatagenConfig) -> DatagenConfig:
        density_min, density_max = meta.density_range
        density_min = _clamp(density_min, 0.001, 1.0)
        density_max = _clamp(density_max, 0.001, 1.0)
        if density_min > density_max:
            density_min, density_max = density_max, density_min

        connectivity_pattern = meta.connectivity_pattern or "default"
        if connectivity_pattern not in self.CONNECTIVITY_PATTERNS:
            connectivity_pattern = "default"

        group = meta.group or meta.symmetry or self.DEFAULT_GROUP
        return replace(
            meta,
            group=group,
            symmetry=meta.symmetry or group,
            basic_size=max(1, int(meta.basic_size)),
            num_samples=max(1, int(meta.num_samples)),
            workers=max(1, int(meta.workers)),
            batch=max(1, int(meta.batch)),
            print_every=max(1, int(meta.print_every)),
            max_bars=max(1, min(int(meta.max_bars), 20)),
            rho_target=_clamp(meta.rho_target, 0.001, 1.0),
            density_range=(density_min, density_max),
            connectivity_pattern=connectivity_pattern,
        )

    def validate(self, meta: DatagenConfig) -> dict:
        repaired = self.repair(meta)
        return {
            "valid": repaired == meta,
            "repaired": repaired.to_dict(),
            "checks": {
                "max_bars_positive": meta.max_bars >= 1,
                "rho_target_range": 0.001 <= float(meta.rho_target) <= 1.0,
                "density_range_ordered": len(meta.density_range) == 2 and meta.density_range[0] <= meta.density_range[1],
                "connectivity_pattern_known": meta.connectivity_pattern in self.CONNECTIVITY_PATTERNS,
            },
        }

    def diversity_key(self, meta: DatagenConfig) -> str:
        return "|".join(
            [
                meta.group,
                meta.basic_unit_type,
                meta.topology_type,
                meta.connectivity_pattern,
                str(meta.max_bars),
            ]
        )

    def candidate(
        self,
        meta: DatagenConfig,
        source_backend: str,
        strategy: str,
        score: float = 0.0,
        confidence: float = 0.5,
        predicted_property: dict[str, float] | None = None,
        expected_error: dict[str, float] | None = None,
        hypothesis_id: str = "",
        rationale: str = "",
        candidate_id: str = "",
    ) -> MetaCandidate:
        repaired = self.repair(meta)
        return MetaCandidate(
            candidate_id=candidate_id or _now_id(f"{source_backend}_{strategy}"),
            meta=repaired,
            source_backend=source_backend,
            strategy=strategy,
            score=float(score),
            confidence=float(confidence),
            predicted_property=dict(predicted_property or repaired.expected_property),
            expected_error=dict(expected_error or {}),
            validity=self.validate(repaired),
            diversity_key=self.diversity_key(repaired),
            hypothesis_id=hypothesis_id,
            rationale=rationale or repaired.reason,
        )

    def deduplicate(self, candidates: Iterable[MetaCandidate], limit: int | None = None) -> list[MetaCandidate]:
        result: list[MetaCandidate] = []
        seen = set()
        for candidate in candidates:
            meta = self.repair(candidate.meta)
            key = (
                meta.group,
                meta.basic_unit_type,
                meta.unit_cell_type,
                meta.topology_type,
                meta.connectivity_pattern,
                meta.max_bars,
                round(float(meta.rho_target), 4),
                tuple(round(float(value), 4) for value in meta.density_range),
                candidate.strategy,
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(replace(candidate, meta=meta))
            if limit is not None and len(result) >= limit:
                break
        return result

    def intervention_variants(
        self,
        base: DatagenConfig,
        target_property: dict[str, float],
        source_backend: str,
        quotas: dict[str, int] | None = None,
    ) -> list[MetaCandidate]:
        quotas = quotas or {
            "exploitation": 3,
            "counterfactual": 2,
            "repair": 2,
            "diversity": 2,
            "high_risk": 1,
        }
        target_density = float(target_property.get("density_proxy", base.rho_target))
        variants: list[MetaCandidate] = []

        def add(strategy: str, meta: DatagenConfig, rationale: str, confidence: float) -> None:
            variants.append(
                self.candidate(
                    meta=meta,
                    source_backend=source_backend,
                    strategy=strategy,
                    confidence=confidence,
                    rationale=rationale,
                )
            )

        for index in range(quotas.get("exploitation", 0)):
            delta = (index - quotas.get("exploitation", 0) // 2) * 0.005
            add(
                "exploitation",
                replace(
                    base,
                    num_samples=1,
                    rho_target=_clamp(target_density + delta, 0.001, 1.0),
                    density_range=(
                        _clamp(target_density + delta - 0.02, 0.001, 1.0),
                        _clamp(target_density + delta + 0.02, 0.001, 1.0),
                    ),
                    reason="Local refinement around a promising or failed Meta.",
                    exploration_strategy="exploitation",
                ),
                "local density/connectivity refinement",
                0.65,
            )

        counterfactual_ops = [
            ("density_up", replace(base, num_samples=1, rho_target=_clamp(target_density + 0.02, 0.001, 1.0))),
            ("density_down", replace(base, num_samples=1, rho_target=_clamp(target_density - 0.02, 0.001, 1.0))),
            ("more_bars", replace(base, num_samples=1, max_bars=base.max_bars + 2)),
            ("fewer_bars", replace(base, num_samples=1, max_bars=max(1, base.max_bars - 2))),
        ]
        for name, meta in counterfactual_ops[: quotas.get("counterfactual", 0)]:
            add(
                "counterfactual",
                replace(meta, reason=f"Single-variable intervention: {name}.", exploration_strategy="counterfactual"),
                f"single-variable intervention {name}",
                0.55,
            )

        repair_ops = [
            replace(base, num_samples=1, connectivity_pattern="higher_diagonal_density", max_bars=base.max_bars + 2),
            replace(base, num_samples=1, connectivity_pattern="center_reinforced", max_bars=base.max_bars + 1),
            replace(base, num_samples=1, connectivity_pattern="reduced_connectivity", max_bars=max(1, base.max_bars - 1)),
        ]
        for meta in repair_ops[: quotas.get("repair", 0)]:
            add(
                "repair",
                replace(meta, reason="Repair candidate targeting known failure modes.", exploration_strategy="repair"),
                "failure-mode repair intervention",
                0.58,
            )

        patterns = list(self.CONNECTIVITY_PATTERNS)
        for index in range(quotas.get("diversity", 0)):
            add(
                "diversity",
                replace(
                    base,
                    num_samples=1,
                    connectivity_pattern=patterns[index % len(patterns)],
                    max_bars=max(1, base.max_bars + ((index % 5) - 2)),
                    reason="Diversity candidate to avoid one-family lock-in.",
                    exploration_strategy="diversity",
                ),
                "diverse structural family coverage",
                0.45,
            )

        for index in range(quotas.get("high_risk", 0)):
            add(
                "high_risk",
                replace(
                    base,
                    num_samples=1,
                    max_bars=14 + index,
                    rho_target=_clamp(target_density + 0.04, 0.001, 1.0),
                    density_range=(
                        _clamp(target_density + 0.01, 0.001, 1.0),
                        _clamp(target_density + 0.07, 0.001, 1.0),
                    ),
                    reason="High-risk candidate for sparse discovery beyond local evidence.",
                    exploration_strategy="high_risk",
                ),
                "high-uncertainty high-reward intervention",
                0.35,
            )

        return self.deduplicate(variants)
