from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class DesignSearchParameters:
    """Agent-facing design variables that may affect generated structure performance."""

    group: str = "P222"
    symmetry: str = "P222"
    max_bars: int = 10
    rho_target: float = 0.1
    r_physical: float = 1.0
    base_node_params: dict[str, float] = field(default_factory=dict)
    base_edges: tuple[tuple[int, int], ...] = ()
    parameter_ranges: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "group", self.group or self.symmetry or "P222")
        object.__setattr__(self, "symmetry", self.symmetry or self.group or "P222")
        object.__setattr__(self, "max_bars", int(self.max_bars))
        object.__setattr__(self, "rho_target", float(self.rho_target))
        object.__setattr__(self, "r_physical", float(self.r_physical))
        object.__setattr__(
            self,
            "base_edges",
            tuple(tuple(int(value) for value in edge) for edge in (self.base_edges or ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatagenConfig:
    # Suggestion identity and provenance. AgentExplorer owns these fields.
    suggestion_id: str = ""
    parent_sample_id: str = ""
    source: str = "agent_exploration"

    # Target and expectation for this proposed generation run.
    target_property: dict[str, float] = field(default_factory=dict)
    expected_property: dict[str, float] = field(default_factory=dict)
    objective: str = "match_target_property"
    confidence: float = 0.5

    # Runtime controls consumed by DatagenFEMEvaluator.
    group: str = "P222"
    basic_size: int = 4
    num_samples: int = 8
    workers: int = 1
    batch: int = 1
    print_every: int = 1
    run_dir: str = ""

    # Structure semantics persisted into KnowledgeBase.
    symmetry: str = "P222"
    basic_unit_type: str = "edge_face_center_19node"
    unit_cell_type: str = "symmetry_expanded_truss"
    topology_type: str = "sparse_truss"
    connectivity_pattern: str = "default"

    # Search-space controls and generated-structure parameters.
    max_bars: int = 10
    rho_target: float = 0.1
    density_range: tuple[float, float] = (0.08, 0.12)
    parameter_ranges: dict[str, Any] = field(default_factory=dict)
    sampling_strategy: str = "random_discrete"
    constraints: dict[str, Any] = field(default_factory=dict)
    design_search_parameters: DesignSearchParameters | dict[str, Any] = field(default_factory=DesignSearchParameters)

    # Agent reasoning. These explain why this generation run exists.
    hypothesis: str = ""
    reason: str = ""
    failure_analysis: dict[str, Any] = field(default_factory=dict)
    exploration_strategy: str = "local_refinement"
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        design_params = self.design_search_parameters
        if isinstance(design_params, dict):
            design_params = DesignSearchParameters(**design_params)
        design_params = replace(
            design_params,
            group=self.group,
            symmetry=self.symmetry or self.group,
            max_bars=self.max_bars,
            rho_target=self.rho_target,
            parameter_ranges=dict(self.parameter_ranges),
            constraints=dict(self.constraints),
        )
        object.__setattr__(self, "design_search_parameters", design_params)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetaCandidate:
    """A proposed executable Meta config plus ranking/provenance signals."""

    candidate_id: str
    meta: DatagenConfig
    source_backend: str = "unknown"
    strategy: str = "unspecified"
    score: float = 0.0
    confidence: float = 0.5
    predicted_property: dict[str, float] = field(default_factory=dict)
    expected_error: dict[str, float] = field(default_factory=dict)
    validity: dict[str, Any] = field(default_factory=dict)
    diversity_key: str = ""
    hypothesis_id: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["meta"] = self.meta.to_dict()
        return payload


@dataclass(frozen=True)
class BatchProposal:
    """A batch of Meta candidates generated from one design intent."""

    proposal_id: str
    source: str
    target_property: dict[str, float]
    candidates: list[MetaCandidate] = field(default_factory=list)
    hypothesis: str = ""
    rationale: str = ""
    strategy_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "source": self.source,
            "target_property": dict(self.target_property),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "strategy_counts": dict(self.strategy_counts),
        }


@dataclass(frozen=True)
class TargetScheduleItem:
    """One scheduled target that AgentExplorer asks InverseDesigner to realize."""

    target_id: str
    target_property: dict[str, float]
    strategy: str = "exploitation"
    reason: str = ""
    expected_effect: dict[str, Any] = field(default_factory=dict)
    risk: str = "medium"
    samples: int = 1
    based_on_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_property", {str(key): float(value) for key, value in self.target_property.items()})
        object.__setattr__(self, "samples", max(1, int(self.samples)))
        object.__setattr__(self, "based_on_evidence", tuple(str(item) for item in self.based_on_evidence if item))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["based_on_evidence"] = list(self.based_on_evidence)
        return payload


@dataclass(frozen=True)
class TargetSchedule:
    """AgentExplorer's online action: choose targets for InverseDesigner to sample."""

    schedule_id: str
    final_target: dict[str, float]
    scheduled_targets: list[TargetScheduleItem] = field(default_factory=list)
    hypothesis: str = ""
    selection_policy: str = ""
    confidence: float = 0.5
    source: str = "agent_explorer"

    def __post_init__(self) -> None:
        object.__setattr__(self, "final_target", {str(key): float(value) for key, value in self.final_target.items()})
        normalized = [
            item if isinstance(item, TargetScheduleItem) else TargetScheduleItem(**dict(item))
            for item in self.scheduled_targets
        ]
        object.__setattr__(self, "scheduled_targets", normalized)
        object.__setattr__(self, "confidence", float(self.confidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "final_target": dict(self.final_target),
            "scheduled_targets": [item.to_dict() for item in self.scheduled_targets],
            "hypothesis": self.hypothesis,
            "selection_policy": self.selection_policy,
            "confidence": float(self.confidence),
            "source": self.source,
        }


@dataclass(frozen=True)
class TargetScheduleProposal:
    """Wrapper for one or more target schedules and their strategy counts."""

    proposal_id: str
    source: str
    final_target: dict[str, float]
    schedules: list[TargetSchedule] = field(default_factory=list)
    rationale: str = ""
    strategy_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "source": self.source,
            "final_target": dict(self.final_target),
            "schedules": [schedule.to_dict() for schedule in self.schedules],
            "rationale": self.rationale,
            "strategy_counts": dict(self.strategy_counts),
        }


@dataclass
class FEMResult:
    structure_id: str
    evaluated_property: dict[str, float]
    fem_status: str
    geometry_status: str
    raw_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """One evaluated experiment before projecting into Dataset/Knowledge views."""

    observation_id: str
    iteration: int
    meta: dict[str, Any]
    structure: dict[str, Any]
    property: dict[str, float]
    error: dict[str, float]
    label: str
    fem_status: str
    geometry_status: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    raw_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    """Reasoning-oriented projection of an Observation."""

    evidence_id: str
    observation_id: str
    meta: dict[str, Any]
    structure: dict[str, Any]
    property: dict[str, float]
    error: dict[str, float]
    label: str
    hypothesis: str = ""
    reasoning: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    supports_hypothesis: bool = False
    contradicts_hypothesis: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetSample:
    """Training projection P_dataset(o_i)."""

    sample_id: str
    input_property: dict[str, float]
    output_meta: dict[str, Any]
    output_structure: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    validity_flag: str = "valid"
    fidelity_flag: str = "proxy"
    structure_feature: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        fem_status = "success" if self.validity_flag == "valid" else self.validity_flag
        geometry_status = "valid" if self.validity_flag in {"valid", "success"} else self.validity_flag
        return {
            "sample_id": self.sample_id,
            "structure_id": self.sample_id,
            "property": dict(self.input_property),
            "explicit_structure": dict(self.output_structure),
            "structure_code": dict(self.output_meta),
            "weight": float(self.weight),
            "validity": {
                "fem_status": fem_status,
                "geometry_status": geometry_status,
            },
            "status": {
                "fem_status": fem_status,
                "geometry_status": geometry_status,
                "source": self.source,
            },
            "structure_feature": dict(self.structure_feature),
            "fidelity": self.fidelity_flag,
        }


@dataclass(frozen=True)
class KnowledgeEvidence:
    """Reasoning projection P_knowledge(o_i)."""

    evidence_id: str
    observation_id: str
    meta_id: str
    structure_id: str
    meta_summary: dict[str, Any]
    structure_features: dict[str, Any]
    property_result: dict[str, float]
    error_to_target: dict[str, float]
    label: str
    source: str = ""
    proposal_group: str = ""
    parent_id: str = ""
    hypothesis_id: str = ""
    hypothesis: str = ""
    intervention_type: str = ""
    intervention_delta: dict[str, Any] = field(default_factory=dict)
    effect_summary: dict[str, Any] = field(default_factory=dict)
    reasoning_tags: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    fidelity: str = "proxy"
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasoning_tags"] = list(self.reasoning_tags)
        return payload


@dataclass(frozen=True)
class StatisticalKnowledgeSnapshot:
    """Deterministic statistical memory Z_stat_n = Refine(K_n)."""

    generated_at: str
    total_evidence: int
    mechanism_patterns: list[dict[str, Any]] = field(default_factory=list)
    failure_patterns: list[dict[str, Any]] = field(default_factory=list)
    intervention_effects: list[dict[str, Any]] = field(default_factory=list)
    hypothesis_status: list[dict[str, Any]] = field(default_factory=list)
    bad_regions: list[dict[str, Any]] = field(default_factory=list)
    useful_exemplars: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


AgentKnowledgeSnapshot = StatisticalKnowledgeSnapshot


@dataclass(frozen=True)
class AgentKnowledgeInterpretation:
    """Agent interpretation layer over a statistical snapshot."""

    interpreter: str
    generated_at: str
    target_property: dict[str, float] = field(default_factory=dict)
    mechanism_explanations: list[dict[str, Any]] = field(default_factory=list)
    dominant_failure_causes: list[dict[str, Any]] = field(default_factory=list)
    promising_intervention_families: list[dict[str, Any]] = field(default_factory=list)
    rejected_intervention_families: list[dict[str, Any]] = field(default_factory=list)
    target_specific_strategy_prior: dict[str, Any] = field(default_factory=dict)
    causal_tests_to_run: list[dict[str, Any]] = field(default_factory=list)
    evidence_citations: list[str] = field(default_factory=list)
    confidence: float = 0.5
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReasonedAgentKnowledgeSnapshot:
    """Reasoned memory R_n = {statistical snapshot, agent interpretation}."""

    generated_at: str
    statistical_snapshot: dict[str, Any]
    interpretation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Compatibility: expose statistical fields at top level for current AgentExplorer.
        for key, value in self.statistical_snapshot.items():
            payload.setdefault(key, value)
        payload["statistical"] = dict(self.statistical_snapshot)
        return payload


@dataclass(frozen=True)
class SchedulerContext:
    """Current target-conditioned decision state C_n."""

    target_property: dict[str, float]
    failed_candidate: dict[str, Any] = field(default_factory=dict)
    top_success: list[dict[str, Any]] = field(default_factory=list)
    near_miss: list[dict[str, Any]] = field(default_factory=list)
    recent_failures: list[dict[str, Any]] = field(default_factory=list)
    diverse_failures: list[dict[str, Any]] = field(default_factory=list)
    dataset_statistics: dict[str, Any] = field(default_factory=dict)
    agent_knowledge_snapshot: dict[str, Any] = field(default_factory=dict)
    budget_config: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeSample:
    structure_id: str
    structure_path: str
    unit_cell_type: str
    basic_unit_type: str
    topology_type: str
    symmetry: str
    connectivity_pattern: str
    parameter_config: dict[str, Any]
    target_property: dict[str, float]
    evaluated_property: dict[str, float]
    property_error: dict[str, float]
    fem_status: str
    geometry_status: str
    label: str
    source: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)
    explicit_structure: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class ExperimentPaths:
    root_dir: str
    runs_dir: str
    events_dir: str
    artifacts_dir: str
    knowledge_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SchedulerEvent:
    seq: int
    task_id: str
    timestamp: str
    stage: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClosedLoopResult:
    task_id: str
    task_status: str
    target_property: dict[str, float]
    success: bool
    iterations: int
    experiment_paths: ExperimentPaths | None = None
    discovered_samples: list[KnowledgeSample] = field(default_factory=list)
    events: list[SchedulerEvent] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_status": self.task_status,
            "target_property": dict(self.target_property),
            "success": bool(self.success),
            "iterations": int(self.iterations),
            "experiment_paths": self.experiment_paths.to_dict() if self.experiment_paths else None,
            "discovered_samples": [sample.to_dict() for sample in self.discovered_samples],
            "events": [event.to_dict() for event in self.events],
            "logs": list(self.logs),
        }
