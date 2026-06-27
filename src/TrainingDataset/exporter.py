from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..KnowledgeBase import KnowledgeBase


class TrainingDatasetExporter:
    """
    Build compact supervision datasets for InverseDesigner.
    This strips run logs, verbose provenance, and agent reasoning.
    """

    def export_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dataset = []
        for row in rows:
            parameter_config = dict(row.get("parameter_config") or {})
            density_range = list(parameter_config.get("density_range") or [])
            density_min = density_range[0] if len(density_range) >= 1 else None
            density_max = density_range[1] if len(density_range) >= 2 else None
            structure_code = {
                "symmetry": row.get("symmetry", ""),
                "basic_unit_type": row.get("basic_unit_type", ""),
                "unit_cell_type": row.get("unit_cell_type", ""),
                "topology_type": row.get("topology_type", ""),
                "connectivity_pattern": row.get("connectivity_pattern", ""),
                "max_bars": parameter_config.get("max_bars"),
                "rho_target": parameter_config.get("rho_target"),
                "density_min": density_min,
                "density_max": density_max,
                "density_range": density_range,
                "sampling_strategy": parameter_config.get("sampling_strategy"),
            }
            explicit_structure = dict(
                row.get("explicit_structure")
                or row.get("structure_json")
                or row.get("structure")
                or {}
            )
            dataset.append(
                {
                    "sample_id": row.get("structure_id", ""),
                    "structure_id": row.get("structure_id", ""),
                    "structure_path": row.get("structure_path", ""),
                    "explicit_structure": explicit_structure,
                    "training_target": "explicit_structure" if explicit_structure else "structure_code_legacy",
                    "structure_code": structure_code,
                    "structure": {
                        "unit_cell_type": row.get("unit_cell_type", ""),
                        "basic_unit_type": row.get("basic_unit_type", ""),
                        "topology_type": row.get("topology_type", ""),
                        "symmetry": row.get("symmetry", ""),
                        "connectivity_pattern": row.get("connectivity_pattern", ""),
                        "parameter_config": parameter_config,
                    },
                    "property": dict(row.get("evaluated_property") or {}),
                    "target_property": dict(row.get("target_property") or {}),
                    "label": row.get("label", ""),
                    "weight": 1.0,
                    "validity": {
                        "fem_status": row.get("fem_status", ""),
                        "geometry_status": row.get("geometry_status", ""),
                    },
                    "status": {
                        "fem_status": row.get("fem_status", ""),
                        "geometry_status": row.get("geometry_status", ""),
                        "label": row.get("label", ""),
                        "source": row.get("source", ""),
                    },
                    "generation_parameters": {
                        "max_bars": parameter_config.get("max_bars"),
                        "rho_target": parameter_config.get("rho_target"),
                        "density_range": density_range,
                        "density_min": density_min,
                        "density_max": density_max,
                        "sampling_strategy": parameter_config.get("sampling_strategy"),
                    },
                }
            )
        return dataset

    def export_from_knowledge_base(self, kb: KnowledgeBase, mark_used: bool = True) -> list[dict[str, Any]]:
        rows = kb.export_training_dataset(mark_used=mark_used)
        return self.export_rows(rows)

    def export_from_path(
        self,
        kb_path: str | Path,
        output_path: str | Path | None = None,
        mark_used: bool = True,
    ) -> list[dict[str, Any]]:
        kb = KnowledgeBase(kb_path)
        try:
            dataset = self.export_from_knowledge_base(kb, mark_used=mark_used)
        finally:
            kb.close()
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
        return dataset
