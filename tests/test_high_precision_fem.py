from __future__ import annotations

from pathlib import Path

from src.DatagenFEMEvaluator import AbaqusFEMConfig, DatagenFEMEvaluator
from src.HighPrecisionFEM import HighPrecisionFEM


class CaptureEvaluator:
    def __init__(self):
        self.structures = []

    def evaluate_explicit_structure(self, structure, target_property):
        self.structures.append(dict(structure))
        return {
            "structure_id": structure.get("structure_id", "captured"),
            "evaluated_property": {
                "type": "stress_curve",
                "strain_grid": [0.0, 0.3],
                "stress": [0.0, 0.1],
            },
            "raw_metrics": {},
            "fem_status": "success",
            "geometry_status": "valid",
        }


def _explicit_truss(index: int) -> dict:
    offset = float(index) * 0.01
    return {
        "structure_id": f"explicit_parallel_{index}",
        "coordinates": [
            [0.0 + offset, 0.0, 0.0],
            [1.0 + offset, 0.0, 0.0],
            [0.0 + offset, 1.0, 1.0],
            [1.0 + offset, 1.0, 1.0],
        ],
        "edges": [[0, 1], [1, 3], [0, 2], [2, 3]],
    }


def test_high_precision_fem_parallel_batch_generates_truss_inputs_without_solver(tmp_path):
    evaluator = DatagenFEMEvaluator(
        workspace_root=tmp_path,
        fem_backend="abaqus",
        fem_config=AbaqusFEMConfig(run_solver=False, cpus=1),
    )
    simulator = HighPrecisionFEM(
        workspace_root=tmp_path,
        evaluator=evaluator,
        max_workers=2,
    )

    pairs = simulator.simulate_many(
        [_explicit_truss(index) for index in range(4)],
        target_property={"density_proxy": 0.2},
        max_workers=2,
        provenance={"test": "parallel_input_generation"},
    )

    assert len(pairs) == 4
    for index, pair in enumerate(pairs):
        evaluation = pair.provenance["evaluation"]
        raw_metrics = evaluation["raw_metrics"]
        assert pair.label_source == "simulation"
        assert evaluation["fem_status"] == "input_generated"
        assert Path(raw_metrics["explicit_structure_path"]).exists()
        assert Path(raw_metrics["fem_inp_path"]).exists()
        assert pair.provenance["batch_index"] == index
        assert pair.provenance["parallel_workers"] == 2


def test_high_precision_fem_default_workers_use_30_percent_cpu(monkeypatch, tmp_path):
    monkeypatch.setattr("os.cpu_count", lambda: 20)

    simulator = HighPrecisionFEM(workspace_root=tmp_path)

    assert simulator.max_workers == 6


def test_high_precision_fem_aligns_remote_graphmetamat_normalized_coordinates(tmp_path):
    evaluator = CaptureEvaluator()
    simulator = HighPrecisionFEM(workspace_root=tmp_path, evaluator=evaluator)

    pair = simulator.simulate(
        {
            "structure_id": "remote_normalized",
            "representation": "graph_truss",
            "source": "inverse_designer_neural:graphmetamat_truss_remote",
            "coordinates": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            "edges": [[0, 1]],
            "rho": 0.1,
            "edge_radii": [0.04],
        },
        target_property={"stress": [0.0, 0.1], "strain_grid": [0.0, 0.3]},
    )

    sent = evaluator.structures[0]
    assert sent["coordinates"][0] == [0.0, -13.747487, -13.747487]
    assert sent["coordinates"][1] == [54.989948, 41.242461, 41.242461]
    assert sent["fem_config_overrides"]["beam_radius"] == 0.04 * 27.494974
    assert pair.provenance["fem_coordinate_alignment"]["status"] == "applied"
    assert pair.provenance["fem_coordinate_alignment"]["rho"] == 0.1
    assert pair.provenance["fem_coordinate_alignment"]["normalized_radius"] == 0.04


def test_high_precision_fem_does_not_align_plain_explicit_structure(tmp_path):
    evaluator = CaptureEvaluator()
    simulator = HighPrecisionFEM(workspace_root=tmp_path, evaluator=evaluator)

    pair = simulator.simulate(
        _explicit_truss(0),
        target_property={"stress": [0.0, 0.1], "strain_grid": [0.0, 0.3]},
    )

    assert evaluator.structures[0]["coordinates"][0] == [0.0, 0.0, 0.0]
    assert pair.provenance["fem_coordinate_alignment"]["status"] == "not_required"
