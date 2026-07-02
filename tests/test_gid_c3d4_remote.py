from __future__ import annotations

import csv
import json
from pathlib import Path

from src.HighPrecisionFEM import GidC3D4RemoteConfig, HighPrecisionFEM, RemoteGidC3D4Evaluator, export_structure_to_gid_dir


def _remote_inverse_structure() -> dict:
    return {
        "structure_id": "remote_graphmetamat_candidate",
        "source": "inverse_designer_remote:graphmetamat",
        "representation": "graph_truss",
        "coordinates": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        "edges": [[0, 1]],
        "edge_radii": [0.04],
        "rho": 0.1,
        "scheduled_target": {
            "type": "stress_curve",
            "strain_grid": [0.0, 0.3],
            "stress": [0.0, 0.001],
        },
    }


def test_export_remote_inverse_structure_to_gid_c3d4_inputs(tmp_path):
    out = tmp_path / "case"

    info = export_structure_to_gid_dir(
        _remote_inverse_structure(),
        out,
        target_property=_remote_inverse_structure()["scheduled_target"],
        gid="case_test",
    )

    assert info["normalized_input"] is True
    assert info["coordinate_scale"] == 5.0
    assert info["radius_mm"] == 0.2

    with (out / "nodes.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["x_mm"] == "-5.0"
    assert rows[1]["z_mm"] == "5.0"

    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["unit_cell_size_L_mm"] == 10.0
    assert meta["strut_radius_mm"] == 0.2
    assert meta["relative_density_rho"] == 0.1
    assert (out / "reference_curve.csv").exists()


def test_high_precision_fem_can_use_gid_c3d4_remote_batch_dry_run(tmp_path):
    evaluator = RemoteGidC3D4Evaluator(
        workspace_root=tmp_path,
        config=GidC3D4RemoteConfig(run_remote=False),
    )
    simulator = HighPrecisionFEM(
        workspace_root=tmp_path,
        evaluator=evaluator,
        align_remote_graphmetamat_to_p222=False,
    )

    pairs = simulator.simulate_many([_remote_inverse_structure()])

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.label_source == "simulation"
    assert pair.provenance["backend"] == "RemoteGidC3D4Evaluator"
    assert pair.provenance["evaluation"]["fem_status"] == "input_generated"
    local_batch = Path(pair.provenance["evaluation"]["raw_metrics"]["local_batch_dir"])
    assert (local_batch / "gid_c3d4_pipeline.py").exists()
    assert list((local_batch / "data").glob("*/nodes.csv"))


def test_gid_c3d4_backend_disables_p222_alignment(tmp_path):
    simulator = HighPrecisionFEM(workspace_root=tmp_path, backend="gid_c3d4_remote")

    assert simulator.align_remote_graphmetamat_to_p222 is False
