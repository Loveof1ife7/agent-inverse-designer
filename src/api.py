import json

from .AgentExplorer import AgentExplorer
from .closed_loop_contracts import DatagenConfig
from .DatagenFEMEvaluator import DatagenFEMEvaluator
from .DatagenFEMEvaluator import (
    bootstrap_dataset_and_kb,
    csv_to_abaqus as convert_csv_to_abaqus,
    deduplicate_architecture_csv,
    expand_crystal as expand_crystal_structure,
    export_txt_to_vtk,
    generate_architecture_csv,
    preview_generation_batch,
    run_group_pipeline,
    solve_constraints as solve_group_constraints,
)
from .InverseDesigner import InverseDesigner
from .KnowledgeBase import KnowledgeBase
from .KnowledgeRefiner import KnowledgeRefiner
from .Scheduler import StructureDiscoverySystem
from .TrainingDataset import TrainingDatasetExporter


def _decode_json_if_possible(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _normalize_sql_record(record: dict) -> dict:
    json_like_fields = {
        "parameter_config",
        "target_property",
        "target_property_json",
        "evaluated_property",
        "evaluated_property_json",
        "property_error",
        "metadata",
        "meta_json",
        "config_json",
    }
    normalized = {}
    for key, value in record.items():
        normalized[key] = _decode_json_if_possible(value) if key in json_like_fields else value
    return normalized


def _normalize_kb_payload(payload):
    if isinstance(payload, list):
        normalized_items = []
        for item in payload:
            if hasattr(item, "to_dict"):
                normalized_items.append(item.to_dict())
            elif isinstance(item, dict):
                normalized_items.append(_normalize_kb_payload(item))
            else:
                normalized_items.append(_normalize_kb_payload(item))
        return normalized_items
    if isinstance(payload, dict):
        normalized = _normalize_sql_record(payload)
        return {key: _normalize_kb_payload(value) for key, value in normalized.items()}
    return payload


def bootstrap_seed_dataset(
    datagen_configs: list[DatagenConfig | dict],
    workspace_root: str = "workspace",
    kb_path: str = "workspace/knowledge.sqlite",
    output_dir: str | None = None,
):
    evaluator = DatagenFEMEvaluator(workspace_root=workspace_root)
    return evaluator.bootstrap_dataset_and_kb(
        datagen_configs=datagen_configs,
        kb_path=kb_path,
        output_dir=output_dir,
    )


def run_closed_loop_discovery(
    target_property: dict[str, float],
    workspace_root: str = "workspace",
    kb_path: str = "workspace/knowledge.sqlite",
    max_iterations: int = 3,
    retrain_trigger: int = 10,
    log_path: str | None = None,
):
    kb = KnowledgeBase(kb_path)
    try:
        system = StructureDiscoverySystem(
            knowledge_base=kb,
            inverse_designer=InverseDesigner(kb, workspace_root=workspace_root),
            agent_explorer=AgentExplorer(),
            evaluator=DatagenFEMEvaluator(workspace_root=workspace_root),
            retrain_trigger=retrain_trigger,
            workspace_root=workspace_root,
            log_path=log_path,
        )
        return system.run(target_property=target_property, max_iterations=max_iterations)
    finally:
        kb.close()


def kb_get_group_statistics(kb_path: str):
    kb = KnowledgeBase(kb_path)
    try:
        return kb.get_group_statistics()
    finally:
        kb.close()


def kb_get_sample_evidence(kb_path: str, sample_id: str):
    kb = KnowledgeBase(kb_path)
    try:
        payload = kb.get_sample_evidence(sample_id)
        return _normalize_kb_payload(payload) if payload is not None else None
    finally:
        kb.close()


def kb_get_run_provenance(kb_path: str, run_id: str):
    kb = KnowledgeBase(kb_path)
    try:
        payload = kb.get_run_provenance(run_id)
        return _normalize_kb_payload(payload) if payload is not None else None
    finally:
        kb.close()


def kb_query_samples(
    kb_path: str,
    query_type: str,
    top_k: int = 20,
    target_property: dict[str, float] | None = None,
    group: str | None = None,
    reason_type: str | None = None,
):
    kb = KnowledgeBase(kb_path)
    try:
        if query_type == "success":
            samples = kb.get_success_samples(target_property=target_property, group=group, top_k=top_k)
        elif query_type == "near_miss":
            samples = kb.get_near_miss_samples(target_property=target_property, group=group, top_k=top_k)
        elif query_type == "failure":
            samples = kb.get_failure_samples(group=group, reason_type=reason_type, top_k=top_k)
        elif query_type == "similar":
            if not target_property:
                raise ValueError("target_property is required for query_type='similar'")
            samples = kb.get_similar_property_samples(target_property=target_property, group=group, top_k=top_k)
        else:
            raise ValueError(f"unsupported query_type: {query_type}")
        return [sample.to_dict() for sample in samples]
    finally:
        kb.close()


def export_inverse_designer_dataset(
    kb_path: str,
    output_path: str | None = None,
    mark_used: bool = False,
):
    exporter = TrainingDatasetExporter()
    return exporter.export_from_path(kb_path=kb_path, output_path=output_path, mark_used=mark_used)


def refine_agent_knowledge(
    kb_path: str,
    target_property: dict[str, float] | None = None,
    output_path: str | None = None,
):
    refiner = KnowledgeRefiner()
    snapshot = refiner.build_from_path(kb_path=kb_path, target_property=target_property)
    if output_path:
        refiner.write_snapshot(snapshot, output_path)
    return snapshot


__all__ = [
    "bootstrap_seed_dataset",
    "convert_csv_to_abaqus",
    "DatagenConfig",
    "deduplicate_architecture_csv",
    "export_inverse_designer_dataset",
    "expand_crystal_structure",
    "export_txt_to_vtk",
    "generate_architecture_csv",
    "kb_get_group_statistics",
    "kb_get_run_provenance",
    "kb_get_sample_evidence",
    "kb_query_samples",
    "preview_generation_batch",
    "refine_agent_knowledge",
    "run_closed_loop_discovery",
    "run_group_pipeline",
    "solve_group_constraints",
]
