from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "AbaqusConversionResult": (".datagen_contracts", "AbaqusConversionResult"),
    "AgentDatagenAPI": (".agent_api", "AgentDatagenAPI"),
    "AgentExplorer": (".AgentExplorer", "AgentExplorer"),
    "ClosedLoopResult": (".closed_loop_contracts", "ClosedLoopResult"),
    "ConstraintSolveResult": (".datagen_contracts", "ConstraintSolveResult"),
    "CrystalExpansionResult": (".datagen_contracts", "CrystalExpansionResult"),
    "DatagenConfig": (".closed_loop_contracts", "DatagenConfig"),
    "DatagenFEMEvaluator": (".DatagenFEMEvaluator", "DatagenFEMEvaluator"),
    "FEMResult": (".closed_loop_contracts", "FEMResult"),
    "GenerationResult": (".datagen_contracts", "GenerationResult"),
    "GeneratorConfig": (".datagen_contracts", "GeneratorConfig"),
    "InverseDesigner": (".InverseDesigner", "InverseDesigner"),
    "KnowledgeBase": (".KnowledgeBase", "KnowledgeBase"),
    "KnowledgeSample": (".closed_loop_contracts", "KnowledgeSample"),
    "TargetSchedule": (".closed_loop_contracts", "TargetSchedule"),
    "TargetScheduleItem": (".closed_loop_contracts", "TargetScheduleItem"),
    "TargetScheduleProposal": (".closed_loop_contracts", "TargetScheduleProposal"),
    "PipelineConfig": (".datagen_contracts", "PipelineConfig"),
    "PipelineResult": (".datagen_contracts", "PipelineResult"),
    "StructureDiscoverySystem": (".Scheduler", "StructureDiscoverySystem"),
    "VtkExportResult": (".datagen_contracts", "VtkExportResult"),
    "convert_csv_to_abaqus": (".api", "convert_csv_to_abaqus"),
    "clean_dataset_and_reindex": (".DatagenFEMEvaluator", "clean_dataset_and_reindex"),
    "deduplicate_architecture_csv": (".api", "deduplicate_architecture_csv"),
    "export_inverse_designer_dataset": (".api", "export_inverse_designer_dataset"),
    "expand_crystal_structure": (".api", "expand_crystal_structure"),
    "export_txt_to_vtk": (".api", "export_txt_to_vtk"),
    "generate_architecture_csv": (".api", "generate_architecture_csv"),
    "kb_get_group_statistics": (".api", "kb_get_group_statistics"),
    "kb_get_run_provenance": (".api", "kb_get_run_provenance"),
    "kb_get_sample_evidence": (".api", "kb_get_sample_evidence"),
    "kb_query_samples": (".api", "kb_query_samples"),
    "KnowledgeRefiner": (".KnowledgeRefiner", "KnowledgeRefiner"),
    "TrainingDatasetExporter": (".TrainingDataset", "TrainingDatasetExporter"),
    "plot_truss": (".DatagenFEMEvaluator", "plot_truss"),
    "preview_generation_batch": (".api", "preview_generation_batch"),
    "refine_agent_knowledge": (".api", "refine_agent_knowledge"),
    "run_closed_loop_discovery": (".api", "run_closed_loop_discovery"),
    "run_group_pipeline": (".api", "run_group_pipeline"),
    "run_7zip_sharded": (".DatagenFEMEvaluator", "run_7zip_sharded"),
    "solve_group_constraints": (".api", "solve_group_constraints"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
