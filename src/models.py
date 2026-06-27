"""Backward-compatible import shim for datagen contracts.

Prefer importing from `src.datagen_contracts`.
"""

from .datagen_contracts import (
    AbaqusConversionResult,
    ConstraintSolveResult,
    CrystalExpansionResult,
    GenerationResult,
    GeneratorConfig,
    PipelineConfig,
    PipelineResult,
    VtkExportResult,
    normalize_path,
)

__all__ = [
    "AbaqusConversionResult",
    "ConstraintSolveResult",
    "CrystalExpansionResult",
    "GenerationResult",
    "GeneratorConfig",
    "PipelineConfig",
    "PipelineResult",
    "VtkExportResult",
    "normalize_path",
]
