from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..closed_loop_contracts import DatasetSample, Evidence, KnowledgeEvidence, Observation


class RawExperimentStore:
    """Append-only JSONL store for all evaluated experiments."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.root_dir / "raw_experiments.jsonl"

    def append(self, observation: Observation) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(observation.to_dict(), ensure_ascii=False) + "\n")

    def append_many(self, observations: Iterable[Observation]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            for observation in observations:
                handle.write(json.dumps(observation.to_dict(), ensure_ascii=False) + "\n")

    def list_observations(self) -> list[Observation]:
        if not self.path.exists():
            return []
        observations = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                observations.append(Observation(**json.loads(line)))
        return observations

    def project_dataset(self) -> list[dict]:
        return [sample.to_dict() for sample in self.project_dataset_samples()]

    def project_dataset_samples(self) -> list[DatasetSample]:
        samples = []
        for observation in self.list_observations():
            sample = self._dataset_sample_from_observation(observation)
            if sample is not None:
                samples.append(sample)
        return samples

    def _dataset_sample_from_observation(self, observation: Observation) -> DatasetSample | None:
        meta = dict(observation.meta)
        if observation.geometry_status != "valid" or observation.fem_status != "success":
            return DatasetSample(
                sample_id=observation.observation_id,
                input_property={},
                output_meta=self._meta_summary(meta),
                output_structure=dict(observation.structure),
                weight=0.2,
                validity_flag=observation.geometry_status if observation.geometry_status != "valid" else observation.fem_status,
                fidelity_flag=str(observation.provenance.get("fidelity") or "proxy"),
                structure_feature=self._structure_features(observation),
                source=str(observation.provenance.get("source") or ""),
            )
        weight_by_label = {"success": 1.0, "near_miss": 0.9, "failure": 0.5}
        return DatasetSample(
            sample_id=observation.observation_id,
            input_property=dict(observation.property),
            output_meta=self._meta_summary(meta),
            output_structure=dict(observation.structure),
            weight=weight_by_label.get(observation.label, 0.4),
            validity_flag="valid",
            fidelity_flag=str(observation.provenance.get("fidelity") or "proxy"),
            structure_feature=self._structure_features(observation),
            source=str(observation.provenance.get("source") or ""),
        )

    def project_evidence(self) -> list[Evidence]:
        evidence = []
        for observation in self.list_observations():
            hypothesis = str(observation.provenance.get("hypothesis") or "")
            reasoning = str(observation.provenance.get("reason") or observation.provenance.get("rationale") or "")
            evidence.append(
                Evidence(
                    evidence_id=f"ev_{observation.observation_id}",
                    observation_id=observation.observation_id,
                    meta=dict(observation.meta),
                    structure=dict(observation.structure),
                    property=dict(observation.property),
                    error=dict(observation.error),
                    label=observation.label,
                    hypothesis=hypothesis,
                    reasoning=reasoning,
                    provenance=dict(observation.provenance),
                    supports_hypothesis=observation.label in {"success", "near_miss"} and bool(hypothesis),
                    contradicts_hypothesis=observation.label == "failure" and bool(hypothesis),
                )
            )
        return evidence

    def project_knowledge(self) -> list[KnowledgeEvidence]:
        return [self._knowledge_from_observation(observation) for observation in self.list_observations()]

    def project_knowledge_from(self, observations: Iterable[Observation]) -> list[KnowledgeEvidence]:
        return [self._knowledge_from_observation(observation) for observation in observations]

    def _knowledge_from_observation(self, observation: Observation) -> KnowledgeEvidence:
        meta = dict(observation.meta)
        provenance = dict(observation.provenance)
        agent_suggestion = dict(provenance.get("agent_suggestion") or {})
        candidate = dict(provenance.get("candidate") or {})
        schedule_item = dict(provenance.get("schedule_item") or {})
        target_schedule = dict(provenance.get("target_schedule") or {})
        hypothesis = str(provenance.get("hypothesis") or agent_suggestion.get("hypothesis") or "")
        hypothesis_id = str(
            provenance.get("hypothesis_id")
            or candidate.get("hypothesis_id")
            or agent_suggestion.get("suggestion_id")
            or schedule_item.get("target_id")
            or target_schedule.get("schedule_id")
            or meta.get("suggestion_id")
            or ""
        )
        return KnowledgeEvidence(
            evidence_id=f"ke_{observation.observation_id}",
            observation_id=observation.observation_id,
            meta_id=str(
                meta.get("suggestion_id")
                or schedule_item.get("target_id")
                or target_schedule.get("schedule_id")
                or candidate.get("candidate_id")
                or observation.observation_id
            ),
            structure_id=str(observation.structure.get("structure_id") or observation.observation_id),
            meta_summary=self._meta_summary(meta),
            structure_features=self._structure_features(observation),
            property_result=dict(observation.property),
            error_to_target=dict(observation.error),
            label=observation.label,
            source=str(provenance.get("source") or meta.get("source") or ""),
            proposal_group=str(candidate.get("source_backend") or target_schedule.get("source") or meta.get("source") or ""),
            parent_id=str(meta.get("parent_sample_id") or schedule_item.get("target_id") or ""),
            hypothesis_id=hypothesis_id,
            hypothesis=hypothesis,
            intervention_type=str(candidate.get("strategy") or schedule_item.get("strategy") or meta.get("exploration_strategy") or ""),
            intervention_delta=dict(provenance.get("intervention_delta") or {}),
            effect_summary=self._effect_summary(observation),
            reasoning_tags=tuple(str(tag) for tag in meta.get("tags", ()) if tag),
            provenance=provenance,
            fidelity=str(provenance.get("fidelity") or "proxy"),
            confidence=float(candidate.get("confidence") or target_schedule.get("confidence") or meta.get("confidence") or 0.5),
        )

    @staticmethod
    def _meta_summary(meta: dict) -> dict:
        density_range = list(meta.get("density_range") or [])
        return {
            "schedule_id": meta.get("schedule_id", ""),
            "target_id": meta.get("target_id", ""),
            "strategy": meta.get("strategy", ""),
            "final_target": meta.get("final_target", {}),
            "scheduled_target": meta.get("scheduled_target", meta.get("target_property", {})),
            "error_to_scheduled_target": meta.get("error_to_scheduled_target", {}),
            "error_to_final_target": meta.get("error_to_final_target", {}),
            "group": meta.get("group") or meta.get("symmetry") or "",
            "symmetry": meta.get("symmetry") or meta.get("group") or "",
            "basic_unit_type": meta.get("basic_unit_type", ""),
            "unit_cell_type": meta.get("unit_cell_type", ""),
            "topology_type": meta.get("topology_type", ""),
            "connectivity_pattern": meta.get("connectivity_pattern", ""),
            "max_bars": meta.get("max_bars"),
            "rho_target": meta.get("rho_target"),
            "density_range": density_range,
            "density_min": density_range[0] if density_range else None,
            "density_max": density_range[1] if len(density_range) > 1 else None,
            "sampling_strategy": meta.get("sampling_strategy", ""),
        }

    @staticmethod
    def _structure_features(observation: Observation) -> dict:
        raw_metrics = dict(observation.raw_metrics or {})
        structure = dict(observation.structure or {})
        return {
            "structure_id": structure.get("structure_id", observation.observation_id),
            "unit_cell_type": structure.get("unit_cell_type", ""),
            "basic_unit_type": structure.get("basic_unit_type", ""),
            "topology_type": structure.get("topology_type", ""),
            "symmetry": structure.get("symmetry", ""),
            "connectivity_pattern": structure.get("connectivity_pattern", ""),
            "node_count": raw_metrics.get("node_count"),
            "edge_count": raw_metrics.get("edge_count"),
            "connectivity_ratio": raw_metrics.get("connectivity_ratio"),
            "connectivity_proxy": raw_metrics.get("connectivity_proxy"),
            "volume_proxy": raw_metrics.get("volume_proxy"),
        }

    @staticmethod
    def _effect_summary(observation: Observation) -> dict:
        return {
            "label": observation.label,
            "property": dict(observation.property),
            "error": dict(observation.error),
            "improved_properties": [
                key for key, value in observation.error.items() if isinstance(value, (int, float)) and float(value) <= 0.35
            ],
            "failed_properties": [
                key for key, value in observation.error.items() if isinstance(value, (int, float)) and float(value) > 0.35
            ],
        }
