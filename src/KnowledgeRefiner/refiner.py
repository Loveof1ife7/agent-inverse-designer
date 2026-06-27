from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..KnowledgeBase import KnowledgeBase
from ..closed_loop_contracts import KnowledgeEvidence, KnowledgeSample, StatisticalKnowledgeSnapshot


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
        t = _safe_float(target[key], 0.0)
        o = _safe_float(observed[key], 0.0)
        scale = abs(t) if abs(t) > 1e-9 else 1.0
        total += abs(o - t) / scale
    return total / len(keys)


class KnowledgeRefiner:
    """
    Deterministically convert sample-level evidence into agent-facing knowledge.
    """

    def build_snapshot(self, samples: list[KnowledgeSample], target_property: dict[str, float] | None = None) -> dict[str, Any]:
        target_property = dict(target_property or {})
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": len(samples),
            "group_stats": self._group_stats(samples),
            "config_pattern_stats": self._config_pattern_stats(samples),
            "failure_patterns": self._failure_patterns(samples),
            "exemplar_samples": self._exemplar_samples(samples, target_property),
        }

    def build_from_evidence(
        self,
        evidences: list[KnowledgeEvidence],
        target_property: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        target_property = dict(target_property or {})
        mechanism_patterns = self._evidence_mechanism_patterns(evidences)
        failure_patterns = self._evidence_failure_patterns(evidences)
        useful_exemplars = self._evidence_exemplars(evidences, target_property)
        snapshot = StatisticalKnowledgeSnapshot(
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_evidence=len(evidences),
            mechanism_patterns=mechanism_patterns,
            failure_patterns=failure_patterns,
            intervention_effects=self._evidence_intervention_effects(evidences),
            hypothesis_status=self._evidence_hypothesis_status(evidences),
            bad_regions=self._evidence_bad_regions(evidences),
            useful_exemplars=useful_exemplars,
        ).to_dict()

        # Backward-compatible aliases consumed by the current AgentExplorer.
        snapshot["total_samples"] = len(evidences)
        snapshot["group_stats"] = self._evidence_group_stats(evidences)
        snapshot["config_pattern_stats"] = mechanism_patterns
        snapshot["exemplar_samples"] = useful_exemplars
        return snapshot

    def build_from_knowledge_base(
        self,
        kb: KnowledgeBase,
        target_property: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        if hasattr(kb, "list_knowledge_evidence"):
            evidences = kb.list_knowledge_evidence()
            if evidences:
                return self.build_from_evidence(evidences, target_property=target_property)
        return self.build_snapshot(kb.list_samples(), target_property=target_property)

    def build_from_path(
        self,
        kb_path: str | Path,
        target_property: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        kb = KnowledgeBase(kb_path)
        try:
            return self.build_from_knowledge_base(kb, target_property=target_property)
        finally:
            kb.close()

    def write_snapshot(
        self,
        snapshot: dict[str, Any],
        output_path: str | Path,
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path.resolve())

    def _group_stats(self, samples: list[KnowledgeSample]) -> list[dict[str, Any]]:
        buckets: dict[str, list[KnowledgeSample]] = defaultdict(list)
        for sample in samples:
            buckets[sample.symmetry].append(sample)

        rows = []
        for group_name, group_samples in sorted(buckets.items()):
            stiffness_values = [_safe_float(sample.evaluated_property.get("stiffness_proxy")) for sample in group_samples]
            density_values = [_safe_float(sample.evaluated_property.get("density_proxy")) for sample in group_samples]
            successes = [sample for sample in group_samples if sample.label == "success"]
            near_misses = [sample for sample in group_samples if sample.label == "near_miss"]
            failures = [sample for sample in group_samples if sample.label == "failure"]
            common_failure = ""
            if failures:
                failure_counts: dict[str, int] = defaultdict(int)
                for sample in failures:
                    failure_counts[self._infer_failure_reason(sample)] += 1
                common_failure = max(failure_counts.items(), key=lambda item: item[1])[0]
            best_stiffness_sample = max(group_samples, key=lambda sample: _safe_float(sample.evaluated_property.get("stiffness_proxy")))
            best_density_sample = min(group_samples, key=lambda sample: _safe_float(sample.evaluated_property.get("density_proxy"), float("inf")))
            total = len(group_samples)
            rows.append(
                {
                    "group": group_name,
                    "num_samples": total,
                    "success_count": len(successes),
                    "near_miss_count": len(near_misses),
                    "failure_count": len(failures),
                    "success_rate": len(successes) / total if total else 0.0,
                    "near_miss_rate": len(near_misses) / total if total else 0.0,
                    "avg_stiffness": sum(stiffness_values) / total if total else 0.0,
                    "avg_density": sum(density_values) / total if total else 0.0,
                    "best_stiffness": _safe_float(best_stiffness_sample.evaluated_property.get("stiffness_proxy")),
                    "best_density": _safe_float(best_density_sample.evaluated_property.get("density_proxy")),
                    "best_sample_id": best_stiffness_sample.structure_id,
                    "common_failure": common_failure,
                }
            )
        return rows

    def _evidence_group_stats(self, evidences: list[KnowledgeEvidence]) -> list[dict[str, Any]]:
        buckets: dict[str, list[KnowledgeEvidence]] = defaultdict(list)
        for evidence in evidences:
            group_name = str(evidence.meta_summary.get("group") or evidence.meta_summary.get("symmetry") or "")
            buckets[group_name].append(evidence)

        rows = []
        for group_name, group_evidence in sorted(buckets.items()):
            total = len(group_evidence)
            success_count = sum(1 for item in group_evidence if item.label == "success")
            near_miss_count = sum(1 for item in group_evidence if item.label == "near_miss")
            failure_count = sum(1 for item in group_evidence if item.label not in {"success", "near_miss"})
            stiffness_values = [_safe_float(item.property_result.get("stiffness_proxy")) for item in group_evidence]
            density_values = [_safe_float(item.property_result.get("density_proxy")) for item in group_evidence]
            best = min(group_evidence, key=lambda item: sum(_safe_float(value) for value in item.error_to_target.values()))
            rows.append(
                {
                    "group": group_name,
                    "num_samples": total,
                    "success_count": success_count,
                    "near_miss_count": near_miss_count,
                    "failure_count": failure_count,
                    "success_rate": success_count / total if total else 0.0,
                    "near_miss_rate": near_miss_count / total if total else 0.0,
                    "avg_stiffness": sum(stiffness_values) / total if total else 0.0,
                    "avg_density": sum(density_values) / total if total else 0.0,
                    "best_sample_id": best.structure_id,
                    "common_failure": self._most_common_failure_reason(group_evidence),
                }
            )
        return rows

    def _evidence_mechanism_patterns(self, evidences: list[KnowledgeEvidence]) -> list[dict[str, Any]]:
        buckets: dict[tuple[Any, ...], list[KnowledgeEvidence]] = defaultdict(list)
        for evidence in evidences:
            meta = evidence.meta_summary
            key = (
                meta.get("basic_unit_type", ""),
                meta.get("topology_type", ""),
                meta.get("connectivity_pattern", ""),
                meta.get("symmetry") or meta.get("group") or "",
                meta.get("max_bars"),
                tuple(meta.get("density_range") or ()),
                meta.get("sampling_strategy", ""),
            )
            buckets[key].append(evidence)

        rows = []
        for key, group_evidence in buckets.items():
            basic_unit_type, topology_type, connectivity_pattern, symmetry, max_bars, density_range, sampling_strategy = key
            total = len(group_evidence)
            success_count = sum(1 for item in group_evidence if item.label == "success")
            near_miss_count = sum(1 for item in group_evidence if item.label == "near_miss")
            avg_error = sum(sum(_safe_float(value) for value in item.error_to_target.values()) for item in group_evidence) / total
            rows.append(
                {
                    "basic_unit_type": basic_unit_type,
                    "topology_type": topology_type,
                    "connectivity_pattern": connectivity_pattern,
                    "symmetry": symmetry,
                    "max_bars": max_bars,
                    "density_range": list(density_range),
                    "sampling_strategy": sampling_strategy,
                    "num_samples": total,
                    "success_count": success_count,
                    "near_miss_count": near_miss_count,
                    "success_rate": success_count / total if total else 0.0,
                    "useful_rate": (success_count + near_miss_count) / total if total else 0.0,
                    "avg_property_error": avg_error,
                    "example_evidence_ids": [item.evidence_id for item in group_evidence[:5]],
                }
            )
        rows.sort(key=lambda item: (-item["useful_rate"], item["avg_property_error"], -item["num_samples"]))
        return rows

    def _evidence_failure_patterns(self, evidences: list[KnowledgeEvidence]) -> list[dict[str, Any]]:
        buckets: dict[str, dict[str, Any]] = {}
        for evidence in evidences:
            if evidence.label == "success":
                continue
            reason = self._infer_evidence_failure_reason(evidence)
            bucket = buckets.setdefault(
                reason,
                {
                    "failure_reason": reason,
                    "count": 0,
                    "groups": defaultdict(int),
                    "example_evidence_ids": [],
                },
            )
            bucket["count"] += 1
            bucket["groups"][str(evidence.meta_summary.get("group") or evidence.meta_summary.get("symmetry") or "")] += 1
            if len(bucket["example_evidence_ids"]) < 5:
                bucket["example_evidence_ids"].append(evidence.evidence_id)

        rows = []
        for row in buckets.values():
            rows.append(
                {
                    "failure_reason": row["failure_reason"],
                    "count": row["count"],
                    "groups": dict(sorted(row["groups"].items())),
                    "example_evidence_ids": row["example_evidence_ids"],
                }
            )
        rows.sort(key=lambda item: (-item["count"], item["failure_reason"]))
        return rows

    def _evidence_intervention_effects(self, evidences: list[KnowledgeEvidence]) -> list[dict[str, Any]]:
        buckets: dict[str, list[KnowledgeEvidence]] = defaultdict(list)
        for evidence in evidences:
            buckets[evidence.intervention_type or "unspecified"].append(evidence)
        rows = []
        for intervention_type, group_evidence in sorted(buckets.items()):
            total = len(group_evidence)
            success_count = sum(1 for item in group_evidence if item.label == "success")
            near_miss_count = sum(1 for item in group_evidence if item.label == "near_miss")
            rows.append(
                {
                    "intervention_type": intervention_type,
                    "count": total,
                    "success_count": success_count,
                    "near_miss_count": near_miss_count,
                    "useful_rate": (success_count + near_miss_count) / total if total else 0.0,
                    "avg_error": sum(sum(_safe_float(value) for value in item.error_to_target.values()) for item in group_evidence) / total,
                    "example_evidence_ids": [item.evidence_id for item in group_evidence[:5]],
                }
            )
        rows.sort(key=lambda item: (-item["useful_rate"], item["avg_error"]))
        return rows

    def _evidence_hypothesis_status(self, evidences: list[KnowledgeEvidence]) -> list[dict[str, Any]]:
        buckets: dict[tuple[str, str], list[KnowledgeEvidence]] = defaultdict(list)
        for evidence in evidences:
            if not evidence.hypothesis and not evidence.hypothesis_id:
                continue
            buckets[(evidence.hypothesis_id, evidence.hypothesis)].append(evidence)
        rows = []
        for (hypothesis_id, hypothesis), group_evidence in sorted(buckets.items()):
            supported = [item for item in group_evidence if item.label in {"success", "near_miss"}]
            contradicted = [item for item in group_evidence if item.label not in {"success", "near_miss"}]
            total = len(group_evidence)
            rows.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "hypothesis": hypothesis,
                    "evidence_count": total,
                    "supported_count": len(supported),
                    "contradicted_count": len(contradicted),
                    "support_rate": len(supported) / total if total else 0.0,
                    "status": "supported" if len(supported) >= len(contradicted) else "contradicted",
                    "example_evidence_ids": [item.evidence_id for item in group_evidence[:5]],
                }
            )
        rows.sort(key=lambda item: (-item["support_rate"], -item["evidence_count"]))
        return rows

    def _evidence_bad_regions(self, evidences: list[KnowledgeEvidence]) -> list[dict[str, Any]]:
        buckets: dict[tuple[Any, ...], list[KnowledgeEvidence]] = defaultdict(list)
        for evidence in evidences:
            if evidence.label in {"success", "near_miss"}:
                continue
            meta = evidence.meta_summary
            key = (
                meta.get("symmetry") or meta.get("group") or "",
                meta.get("topology_type", ""),
                meta.get("connectivity_pattern", ""),
                meta.get("max_bars"),
            )
            buckets[key].append(evidence)
        rows = []
        for key, group_evidence in buckets.items():
            symmetry, topology_type, connectivity_pattern, max_bars = key
            rows.append(
                {
                    "symmetry": symmetry,
                    "topology_type": topology_type,
                    "connectivity_pattern": connectivity_pattern,
                    "max_bars": max_bars,
                    "failure_count": len(group_evidence),
                    "dominant_failure": self._most_common_failure_reason(group_evidence),
                    "example_evidence_ids": [item.evidence_id for item in group_evidence[:5]],
                }
            )
        rows.sort(key=lambda item: (-item["failure_count"], item["symmetry"]))
        return rows

    def _evidence_exemplars(
        self,
        evidences: list[KnowledgeEvidence],
        target_property: dict[str, float],
    ) -> dict[str, list[dict[str, Any]]]:
        def summarize(evidence: KnowledgeEvidence) -> dict[str, Any]:
            return {
                "evidence_id": evidence.evidence_id,
                "structure_id": evidence.structure_id,
                "group": evidence.meta_summary.get("group") or evidence.meta_summary.get("symmetry") or "",
                "label": evidence.label,
                "property": dict(evidence.property_result),
                "error": dict(evidence.error_to_target),
                "meta_summary": dict(evidence.meta_summary),
                "hypothesis": evidence.hypothesis,
                "intervention_type": evidence.intervention_type,
            }

        def rank(items: list[KnowledgeEvidence]) -> list[KnowledgeEvidence]:
            if target_property:
                return sorted(items, key=lambda item: _property_distance(target_property, item.property_result))
            return sorted(items, key=lambda item: sum(_safe_float(value) for value in item.error_to_target.values()))

        success = rank([item for item in evidences if item.label == "success"])
        near_miss = rank([item for item in evidences if item.label == "near_miss"])
        failure = rank([item for item in evidences if item.label not in {"success", "near_miss"}])
        return {
            "success": [summarize(item) for item in success[:5]],
            "near_miss": [summarize(item) for item in near_miss[:5]],
            "failure": [summarize(item) for item in failure[:5]],
        }

    def _most_common_failure_reason(self, evidences: list[KnowledgeEvidence]) -> str:
        if not evidences:
            return ""
        counts: dict[str, int] = defaultdict(int)
        for evidence in evidences:
            counts[self._infer_evidence_failure_reason(evidence)] += 1
        return max(counts.items(), key=lambda item: item[1])[0]

    def _infer_evidence_failure_reason(self, evidence: KnowledgeEvidence) -> str:
        if evidence.label == "success":
            return "success"
        failed_properties = list(evidence.effect_summary.get("failed_properties") or [])
        if "stiffness_proxy" in failed_properties:
            return "stiffness_insufficient"
        if "density_proxy" in failed_properties:
            return "density_mismatch"
        if evidence.label == "invalid":
            return "invalid_geometry"
        if evidence.label == "fem_failed":
            return "fem_failed"
        if evidence.error_to_target:
            worst = max(evidence.error_to_target.items(), key=lambda item: _safe_float(item[1]))[0]
            return f"{worst}_mismatch"
        return "property_mismatch"

    def _config_pattern_stats(self, samples: list[KnowledgeSample]) -> list[dict[str, Any]]:
        buckets: dict[tuple[Any, ...], list[KnowledgeSample]] = defaultdict(list)
        for sample in samples:
            density_range = tuple(sample.parameter_config.get("density_range", ()))
            key = (
                sample.basic_unit_type,
                sample.topology_type,
                sample.connectivity_pattern,
                sample.symmetry,
                sample.parameter_config.get("max_bars"),
                density_range,
                sample.parameter_config.get("sampling_strategy"),
            )
            buckets[key].append(sample)

        rows = []
        for key, group_samples in buckets.items():
            basic_unit_type, topology_type, connectivity_pattern, symmetry, max_bars, density_range, sampling_strategy = key
            total = len(group_samples)
            success_count = sum(1 for sample in group_samples if sample.label == "success")
            avg_error = sum(_property_distance(sample.target_property, sample.evaluated_property) for sample in group_samples) / total
            rows.append(
                {
                    "basic_unit_type": basic_unit_type,
                    "topology_type": topology_type,
                    "connectivity_pattern": connectivity_pattern,
                    "symmetry": symmetry,
                    "max_bars": max_bars,
                    "density_range": list(density_range),
                    "sampling_strategy": sampling_strategy,
                    "num_samples": total,
                    "success_count": success_count,
                    "success_rate": success_count / total if total else 0.0,
                    "avg_property_error": avg_error,
                }
            )
        rows.sort(key=lambda item: (-item["success_rate"], item["avg_property_error"], -item["num_samples"]))
        return rows

    def _failure_patterns(self, samples: list[KnowledgeSample]) -> list[dict[str, Any]]:
        buckets: dict[str, dict[str, Any]] = {}
        for sample in samples:
            if sample.label == "success":
                continue
            reason = self._infer_failure_reason(sample)
            bucket = buckets.setdefault(
                reason,
                {
                    "failure_reason": reason,
                    "count": 0,
                    "groups": defaultdict(int),
                    "example_sample_ids": [],
                },
            )
            bucket["count"] += 1
            bucket["groups"][sample.symmetry] += 1
            if len(bucket["example_sample_ids"]) < 5:
                bucket["example_sample_ids"].append(sample.structure_id)

        rows = []
        for row in buckets.values():
            rows.append(
                {
                    "failure_reason": row["failure_reason"],
                    "count": row["count"],
                    "groups": dict(sorted(row["groups"].items())),
                    "example_sample_ids": row["example_sample_ids"],
                }
            )
        rows.sort(key=lambda item: (-item["count"], item["failure_reason"]))
        return rows

    def _exemplar_samples(
        self,
        samples: list[KnowledgeSample],
        target_property: dict[str, float],
    ) -> dict[str, list[dict[str, Any]]]:
        def summarize(sample: KnowledgeSample) -> dict[str, Any]:
            return {
                "structure_id": sample.structure_id,
                "group": sample.symmetry,
                "label": sample.label,
                "structure_path": sample.structure_path,
                "evaluated_property": dict(sample.evaluated_property),
                "parameter_config": dict(sample.parameter_config),
            }

        success_samples = [sample for sample in samples if sample.label == "success"]
        near_miss_samples = [sample for sample in samples if sample.label == "near_miss"]
        failure_samples = [sample for sample in samples if sample.label == "failure"]

        if target_property:
            success_samples = sorted(success_samples, key=lambda sample: _property_distance(target_property, sample.evaluated_property))
            near_miss_samples = sorted(near_miss_samples, key=lambda sample: _property_distance(target_property, sample.evaluated_property))
            failure_samples = sorted(failure_samples, key=lambda sample: _property_distance(target_property, sample.evaluated_property))

        return {
            "success": [summarize(sample) for sample in success_samples[:5]],
            "near_miss": [summarize(sample) for sample in near_miss_samples[:5]],
            "failure": [summarize(sample) for sample in failure_samples[:5]],
        }

    def _infer_failure_reason(self, sample: KnowledgeSample) -> str:
        if sample.geometry_status != "valid":
            return "invalid_geometry"
        if sample.fem_status != "success":
            return "fem_failed"

        target_density = _safe_float(sample.target_property.get("density_proxy"))
        observed_density = _safe_float(sample.evaluated_property.get("density_proxy"))
        target_stiffness = _safe_float(sample.target_property.get("stiffness_proxy"))
        observed_stiffness = _safe_float(sample.evaluated_property.get("stiffness_proxy"))

        if sample.label == "success":
            return "success"
        if observed_stiffness + 1e-9 < target_stiffness:
            return "stiffness_insufficient"
        if observed_density - 1e-9 > target_density:
            return "density_too_high"
        if sample.connectivity_pattern == "default" and _safe_float(sample.parameter_config.get("max_bars"), 0.0) <= 8:
            return "low_connectivity"
        return "property_mismatch"
