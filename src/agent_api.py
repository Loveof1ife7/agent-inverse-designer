from __future__ import annotations

from dataclasses import asdict

from .api import (
    bootstrap_seed_dataset,
    convert_csv_to_abaqus,
    deduplicate_architecture_csv,
    export_inverse_designer_dataset,
    export_txt_to_vtk,
    expand_crystal_structure,
    generate_architecture_csv,
    kb_get_group_statistics,
    kb_get_run_provenance,
    kb_get_sample_evidence,
    kb_query_samples,
    preview_generation_batch,
    refine_agent_knowledge,
    run_closed_loop_discovery,
    run_group_pipeline,
    solve_group_constraints,
)
from .closed_loop_contracts import DatagenConfig
from .datagen_contracts import GeneratorConfig, PipelineConfig


class AgentDatagenAPI:
    """Agent-facing facade that returns plain Python structures."""

    @staticmethod
    def solve_constraints(config: PipelineConfig | dict):
        cfg = _ensure_pipeline_config(config)
        return asdict(
            solve_group_constraints(
                group_name=cfg.group,
                db_path=cfg.group_db,
                export_path=None,
                show_plot=False,
            )
        )

    @staticmethod
    def preview_batch(config: GeneratorConfig | dict, batch_size: int, seed: int):
        cfg = _ensure_generator_config(config)
        return preview_generation_batch(cfg, batch_size=batch_size, seed=seed)

    @staticmethod
    def generate_csv(config: GeneratorConfig | dict, allow_single_process_fallback: bool = False):
        cfg = _ensure_generator_config(config)
        return asdict(
            generate_architecture_csv(cfg, allow_single_process_fallback=allow_single_process_fallback)
        )

    @staticmethod
    def run_pipeline(config: PipelineConfig | dict):
        cfg = _ensure_pipeline_config(config)
        return run_group_pipeline(cfg).to_dict()

    @staticmethod
    def convert_csv_to_abaqus(csv_path: str, out_dir: str, group_name: str, group_db_path: str):
        return asdict(convert_csv_to_abaqus(csv_path, out_dir, group_name, group_db_path))

    @staticmethod
    def expand_crystal(in_dir: str, out_dir: str, nx: int, ny: int, nz: int):
        return asdict(expand_crystal_structure(in_dir, out_dir, nx, ny, nz))

    @staticmethod
    def export_vtk(input_path: str, output_path: str = "", glob: str = "*.txt"):
        return asdict(export_txt_to_vtk(input_path, output_path or None, glob=glob))

    @staticmethod
    def deduplicate_csv(input_path: str, output_path: str):
        saved = deduplicate_architecture_csv(input_path, output_path)
        return {"saved_rows": saved, "output_path": output_path}

    @staticmethod
    def run_closed_loop(
        target_property: dict[str, float],
        workspace_root: str = "workspace",
        kb_path: str = "workspace/knowledge.sqlite",
        max_iterations: int = 3,
        retrain_trigger: int = 10,
        log_path: str = "",
    ):
        return run_closed_loop_discovery(
            target_property=target_property,
            workspace_root=workspace_root,
            kb_path=kb_path,
            max_iterations=max_iterations,
            retrain_trigger=retrain_trigger,
            log_path=log_path or None,
        ).to_dict()

    @staticmethod
    def bootstrap_seed_dataset(
        datagen_configs: list[DatagenConfig | dict],
        workspace_root: str = "workspace",
        kb_path: str = "workspace/knowledge.sqlite",
        output_dir: str = "",
    ):
        return bootstrap_seed_dataset(
            datagen_configs=datagen_configs,
            workspace_root=workspace_root,
            kb_path=kb_path,
            output_dir=output_dir or None,
        ).to_dict()

    @staticmethod
    def kb_group_statistics(kb_path: str):
        return kb_get_group_statistics(kb_path)

    @staticmethod
    def kb_sample_evidence(kb_path: str, sample_id: str):
        return kb_get_sample_evidence(kb_path, sample_id)

    @staticmethod
    def kb_run_provenance(kb_path: str, run_id: str):
        return kb_get_run_provenance(kb_path, run_id)

    @staticmethod
    def kb_query_samples(
        kb_path: str,
        query_type: str,
        top_k: int = 20,
        target_property: dict[str, float] | None = None,
        group: str = "",
        reason_type: str = "",
    ):
        return kb_query_samples(
            kb_path=kb_path,
            query_type=query_type,
            top_k=top_k,
            target_property=target_property,
            group=group or None,
            reason_type=reason_type or None,
        )

    @staticmethod
    def export_inverse_designer_dataset(
        kb_path: str,
        output_path: str = "",
        mark_used: bool = False,
    ):
        return export_inverse_designer_dataset(
            kb_path=kb_path,
            output_path=output_path or None,
            mark_used=mark_used,
        )

    @staticmethod
    def refine_agent_knowledge(
        kb_path: str,
        target_property: dict[str, float] | None = None,
        output_path: str = "",
    ):
        return refine_agent_knowledge(
            kb_path=kb_path,
            target_property=target_property,
            output_path=output_path or None,
        )

    @staticmethod
    def make_datagen_config(config: DatagenConfig | dict):
        if isinstance(config, DatagenConfig):
            return config.to_dict()
        return DatagenConfig(**config).to_dict()


def _ensure_generator_config(config: GeneratorConfig | dict) -> GeneratorConfig:
    if isinstance(config, GeneratorConfig):
        return config
    return GeneratorConfig(**config)


def _ensure_pipeline_config(config: PipelineConfig | dict) -> PipelineConfig:
    if isinstance(config, PipelineConfig):
        return config
    return PipelineConfig(**config)
