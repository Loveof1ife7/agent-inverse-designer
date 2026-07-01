from __future__ import annotations

import os

from .DatagenFEMEvaluator import (
    DatagenFEMEvaluator,
    csv_to_abaqus as convert_csv_to_abaqus,
    deduplicate_architecture_csv,
    expand_crystal as expand_crystal_structure,
    export_txt_to_vtk,
    generate_architecture_csv,
    get_structure_family_registry,
    get_supported_structure_families,
    preview_generation_batch,
    run_group_pipeline,
    solve_constraints as solve_group_constraints,
)
from .DatasetManager import DatasetManager
from .ForwardSurrogate import ForwardSurrogate
from .HighPrecisionFEM import HighPrecisionFEM
from .InverseDesigner import RemoteGraphMetaMatInverseDesigner
from .Scheduler import DeterministicLoopConfig, DeterministicSurrogateClosedLoopSystem
from .TargetCurvePlanner import TargetCurvePlanner
from .curve_targets import normalize_target_property


CLOSED_LOOP_DEFAULT_FEM_BACKEND = os.getenv(
    "CLOSED_LOOP_FEM_BACKEND",
    os.getenv("DATAGEN_FEM_BACKEND", "abaqus"),
).strip().lower()


def datagen_structure_family_registry():
    return get_structure_family_registry()


def datagen_supported_structure_families():
    return get_supported_structure_families()


def build_deterministic_surrogate_closed_loop(
    workspace_root: str = "workspace",
    *,
    inverse_designer_mode: str | None = "remote_graphmetamat",
    surrogate_backend: str = "remote_forward",
    high_precision_backend: str = CLOSED_LOOP_DEFAULT_FEM_BACKEND,
    config: DeterministicLoopConfig | dict | None = None,
    log_path: str | None = None,
):
    """Build the deterministic surrogate workflow without running it."""

    inverse_designer = _build_deterministic_inverse_designer(
        workspace_root=workspace_root,
        mode=inverse_designer_mode,
    )
    return DeterministicSurrogateClosedLoopSystem(
        inverse_designer=inverse_designer,
        target_planner=TargetCurvePlanner(),
        forward_surrogate=ForwardSurrogate(
            workspace_root=workspace_root,
            evaluator=DatagenFEMEvaluator(workspace_root=workspace_root, fem_backend=surrogate_backend),
        ),
        high_precision_fem=HighPrecisionFEM(
            workspace_root=workspace_root,
            evaluator=DatagenFEMEvaluator(workspace_root=workspace_root, fem_backend=high_precision_backend),
        ),
        dataset_manager=DatasetManager(f"{workspace_root}/deterministic_datasets"),
        config=config,
        workspace_root=workspace_root,
        log_path=log_path,
    )


def run_deterministic_surrogate_closed_loop(
    final_target: dict,
    workspace_root: str = "workspace",
    *,
    max_iterations: int | None = None,
    inverse_designer_mode: str | None = "remote_graphmetamat",
    surrogate_backend: str = "remote_forward",
    high_precision_backend: str = CLOSED_LOOP_DEFAULT_FEM_BACKEND,
    config: DeterministicLoopConfig | dict | None = None,
    log_path: str | None = None,
) -> dict:
    system = build_deterministic_surrogate_closed_loop(
        workspace_root=workspace_root,
        inverse_designer_mode=inverse_designer_mode,
        surrogate_backend=surrogate_backend,
        high_precision_backend=high_precision_backend,
        config=config,
        log_path=log_path,
    )
    return system.run(normalize_target_property(final_target), max_iterations=max_iterations)


def _build_deterministic_inverse_designer(
    *,
    workspace_root: str,
    mode: str | None = "remote_graphmetamat",
):
    resolved = (mode or "remote_graphmetamat").strip().lower()
    if resolved in {"remote_graphmetamat", "graphmetamat_remote", "remote_truss", "remote_graph_meta_mat"}:
        return RemoteGraphMetaMatInverseDesigner.from_env(workspace_root=workspace_root)
    raise ValueError(f"unsupported inverse_designer_mode for deterministic loop: {mode!r}")


__all__ = [
    "CLOSED_LOOP_DEFAULT_FEM_BACKEND",
    "build_deterministic_surrogate_closed_loop",
    "convert_csv_to_abaqus",
    "DatagenFEMEvaluator",
    "datagen_structure_family_registry",
    "datagen_supported_structure_families",
    "deduplicate_architecture_csv",
    "expand_crystal_structure",
    "export_txt_to_vtk",
    "generate_architecture_csv",
    "preview_generation_batch",
    "run_deterministic_surrogate_closed_loop",
    "run_group_pipeline",
    "solve_group_constraints",
]
