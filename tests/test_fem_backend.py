from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.DatagenFEMEvaluator import AbaqusFEMConfig, DatagenFEMEvaluator
from src.DatagenFEMEvaluator.core.truss import fem as core_fem
from src.InverseDesigner import RemoteJobResult


def _node_map_from_inp(inp_text: str) -> dict[int, tuple[float, float, float]]:
    nodes: dict[int, tuple[float, float, float]] = {}
    in_nodes = False
    for raw in inp_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("*"):
            in_nodes = line.upper() == "*NODE"
            continue
        if not in_nodes:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return nodes


def _r3d4_elements_from_inp(inp_text: str) -> list[list[int]]:
    elements: list[list[int]] = []
    in_r3d4 = False
    for raw in inp_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("*"):
            in_r3d4 = line.upper() == "*ELEMENT, TYPE=R3D4"
            continue
        if not in_r3d4:
            continue
        parts = [int(part.strip()) for part in line.split(",") if part.strip()]
        if len(parts) >= 5:
            elements.append(parts)
    return elements


def _quad_normal_z(nodes: dict[int, tuple[float, float, float]], element: list[int]) -> float:
    p0 = nodes[element[1]]
    p1 = nodes[element[2]]
    p2 = nodes[element[3]]
    ux, uy, _uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    vx, vy, _vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    return ux * vy - uy * vx


def _write_truss_txt(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Data ID: tiny_truss",
                "node_data = [",
                "    [1, 0.0, 0.0, 0.0],",
                "    [2, 1.0, 0.0, 0.0],",
                "    [3, 0.0, 1.0, 1.0],",
                "    [4, 1.0, 1.0, 1.0],",
                "]",
                "element_conn = [",
                "    [1, 2],",
                "    [2, 4],",
                "    [1, 3],",
                "    [3, 4],",
                "]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_core_fem_generates_abaqus_input_without_running_solver(tmp_path):
    truss_path = tmp_path / "tiny.txt"
    _write_truss_txt(truss_path)

    result = core_fem.evaluate_truss_file(
        structure_path=truss_path,
        structure_id="tiny:1",
        output_root=tmp_path / "fem_runs",
        config=AbaqusFEMConfig(run_solver=False),
    )

    assert result.status == "input_generated"
    assert Path(result.inp_path).exists()
    assert "abaqus_plate_compression" == result.raw_metrics["evaluator"]
    assert result.raw_metrics["node_count"] > 0
    assert result.raw_metrics["edge_count"] > 0
    inp_text = Path(result.inp_path).read_text(encoding="utf-8")
    assert "8.925, 0.48" in inp_text
    assert "3.207, 0.0" in inp_text
    assert "*NSET, NSET=RBM1" in inp_text
    extractor_text = (Path(result.run_dir) / "extract.py").read_text(encoding="utf-8")
    assert "Strain,Disp_mm,Force_N,Stress_MPa" in extractor_text


def test_core_fem_top_rigid_plate_faces_downward(tmp_path):
    truss_path = tmp_path / "tiny.txt"
    _write_truss_txt(truss_path)

    result = core_fem.evaluate_truss_file(
        structure_path=truss_path,
        structure_id="tiny:plate_normals",
        output_root=tmp_path / "fem_runs",
        config=AbaqusFEMConfig(run_solver=False),
    )

    inp_text = Path(result.inp_path).read_text(encoding="utf-8")
    nodes = _node_map_from_inp(inp_text)
    plates = _r3d4_elements_from_inp(inp_text)

    assert len(plates) == 2
    assert _quad_normal_z(nodes, plates[0]) > 0.0
    assert _quad_normal_z(nodes, plates[1]) < 0.0


def test_core_fem_node_z_mode_places_plates_near_node_extrema(tmp_path):
    truss_path = tmp_path / "tiny.txt"
    _write_truss_txt(truss_path)

    result = core_fem.evaluate_truss_file(
        structure_path=truss_path,
        structure_id="tiny:plate_gap",
        output_root=tmp_path / "fem_runs",
        config=AbaqusFEMConfig(run_solver=False, beam_radius=1.0, strain_ref_mode="NODE_Z"),
    )

    inp_text = Path(result.inp_path).read_text(encoding="utf-8")
    nodes = _node_map_from_inp(inp_text)
    plates = _r3d4_elements_from_inp(inp_text)
    bottom_z = [nodes[node_id][2] for node_id in plates[0][1:]]
    top_z = [nodes[node_id][2] for node_id in plates[1][1:]]

    assert max(bottom_z) < 0.0
    assert min(top_z) > 1.0
    assert min(top_z) < 1.01


def test_datagen_fem_auto_backend_falls_back_to_proxy_when_abaqus_missing(monkeypatch, tmp_path):
    truss_path = tmp_path / "tiny.txt"
    _write_truss_txt(truss_path)
    monkeypatch.setattr(core_fem, "find_abaqus_command", lambda _cmd="": "")

    evaluator = DatagenFEMEvaluator(workspace_root=tmp_path, fem_backend="auto")
    results = evaluator.fem_evaluate(
        [
            {
                "structure_id": "tiny:1",
                "crystal_txt_path": str(truss_path),
            }
        ]
    )

    assert len(results) == 1
    assert results[0].fem_status == "success"
    assert results[0].raw_metrics["evaluator"] == "simplified_proxy_v1"
    assert results[0].raw_metrics["fem_backend_fallback"] == "proxy"


def test_explicit_structure_abaqus_backend_generates_input_without_proxy(tmp_path):
    evaluator = DatagenFEMEvaluator(
        workspace_root=tmp_path,
        fem_backend="abaqus",
        fem_config=AbaqusFEMConfig(run_solver=False),
    )

    evaluation = evaluator.evaluate_explicit_structure(
        {
            "structure_id": "explicit:tiny",
            "coordinates": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            "edges": [[0, 1], [1, 3], [0, 2], [2, 3]],
        },
        target_property={"density_proxy": 0.2},
    )

    raw_metrics = evaluation["raw_metrics"]
    assert evaluation["fem_status"] == "input_generated"
    assert evaluation["geometry_status"] == "valid"
    assert raw_metrics["evaluator"] == "abaqus_plate_compression"
    assert raw_metrics["fem_run_status"] == "input_generated"
    assert Path(raw_metrics["explicit_structure_path"]).exists()
    assert Path(raw_metrics["fem_inp_path"]).exists()


def test_explicit_structure_abaqus_backend_accepts_beam_radius_override(tmp_path):
    evaluator = DatagenFEMEvaluator(
        workspace_root=tmp_path,
        fem_backend="abaqus",
        fem_config=AbaqusFEMConfig(run_solver=False, beam_radius=1.0),
    )

    evaluation = evaluator.evaluate_explicit_structure(
        {
            "structure_id": "explicit:radius_override",
            "coordinates": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            "edges": [[0, 1], [1, 3], [0, 2], [2, 3]],
            "fem_config_overrides": {"beam_radius": 2.5},
        },
        target_property={"density_proxy": 0.2},
    )

    inp_text = Path(evaluation["raw_metrics"]["fem_inp_path"]).read_text(encoding="utf-8")
    assert "*BEAM SECTION, SECTION=CIRC" in inp_text
    assert "\n2.5\n" in inp_text


def test_explicit_structure_auto_backend_falls_back_to_proxy_when_abaqus_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(core_fem, "find_abaqus_command", lambda _cmd="": "")
    evaluator = DatagenFEMEvaluator(workspace_root=tmp_path, fem_backend="auto")

    evaluation = evaluator.evaluate_explicit_structure(
        {
            "structure_id": "explicit:auto",
            "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "edges": [[0, 1]],
        },
        target_property={"density_proxy": 0.2},
    )

    assert evaluation["fem_status"] == "success"
    assert evaluation["raw_metrics"]["evaluator"] == "explicit_structure_proxy_v1"
    assert evaluation["raw_metrics"]["fem_backend_fallback"] == "proxy"
    assert evaluation["raw_metrics"]["fem_backend_fallback_reason"] == "abaqus_unavailable"


def test_explicit_structure_abaqus_backend_scores_stress_curve_target(monkeypatch, tmp_path):
    curve_path = tmp_path / "data.csv"
    curve_path.write_text(
        "Strain,Disp_mm,Force_N,Stress_MPa\n"
        "0.0,0.0,0.0,0.0\n"
        "0.3,1.0,1.0,0.6\n",
        encoding="utf-8",
    )

    def fake_evaluate_truss_file(*args, **kwargs):
        return core_fem.AbaqusFEMRunResult(
            structure_id="explicit:curve",
            status="success",
            run_dir=str(tmp_path),
            inp_path=str(tmp_path / "fake.inp"),
            curve_path=str(curve_path),
            evaluated_property={"density_proxy": 0.1},
            raw_metrics={"fem_curve_path": str(curve_path), "evaluator": "abaqus_plate_compression"},
        )

    monkeypatch.setattr(core_fem, "evaluate_truss_file", fake_evaluate_truss_file)
    evaluator = DatagenFEMEvaluator(workspace_root=tmp_path, fem_backend="abaqus")

    evaluation = evaluator.evaluate_explicit_structure(
        {
            "structure_id": "explicit:curve",
            "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "edges": [[0, 1]],
        },
        {"stress_curve": [0.0, 0.6], "strain_grid": [0.0, 0.3]},
    )

    assert evaluation["property_error"]["curve_nmae"] == 0.0
    assert evaluation["label"] == "success"
    assert evaluation["raw_metrics"]["utility_curve_mae"] == 0.0
    assert evaluation["raw_metrics"]["curve_metrics"]["curve_mae"] == 0.0


def test_explicit_structure_remote_forward_backend_scores_curve(monkeypatch, tmp_path):
    class FakeRemoteForwardClient:
        def run_truss_forward_predict(self, graph_path, *, job_id, device="cuda", download_response=True):
            assert graph_path == "remote/design.gpkl"
            assert device
            return RemoteJobResult(
                job_id=job_id,
                status="success",
                local_dir=str(tmp_path / "remote_forward"),
                remote_dir="/remote/forward",
                response_path=str(tmp_path / "remote_forward" / "forward_response.json"),
                response={
                    "status": "success",
                    "response": {
                        "strain_grid": [0.0, 0.3],
                        "stress": [0.0, 0.6],
                        "relative_density": 0.12,
                    },
                },
            )

    evaluator = DatagenFEMEvaluator(workspace_root=tmp_path, fem_backend="remote_forward")
    monkeypatch.setattr(evaluator, "_remote_forward_client", lambda: FakeRemoteForwardClient())

    evaluation = evaluator.evaluate_explicit_structure(
        {
            "structure_id": "remote_forward:curve",
            "artifacts": {"gpkl": "remote/design.gpkl"},
        },
        {"stress_curve": [0.0, 0.6], "strain_grid": [0.0, 0.3]},
    )

    assert evaluation["fem_status"] == "success"
    assert evaluation["property_error"]["curve_nmae"] == 0.0
    assert evaluation["evaluated_property"]["density_proxy"] == 0.12
    assert evaluation["raw_metrics"]["evaluator"] == "remote_graphmetamat_forward"
    assert evaluation["raw_metrics"]["fidelity"] == "remote_forward_surrogate"


def test_explicit_structure_remote_forward_backend_reads_predicted_property_schema(monkeypatch, tmp_path):
    class FakeRemoteForwardClient:
        def run_truss_forward_predict(self, graph_path, *, job_id, device="cuda", download_response=True):
            return RemoteJobResult(
                job_id=job_id,
                status="success",
                local_dir=str(tmp_path / "remote_forward"),
                remote_dir="/remote/forward",
                response_path=str(tmp_path / "remote_forward" / "forward_response.json"),
                response={
                    "status": "success",
                    "predicted_property": {
                        "task": "compression_stress_strain",
                        "strain_grid": [0.0, 0.15, 0.3],
                        "stress": [0.0, 0.4, 0.8],
                        "rho": 0.3,
                        "num_nodes": 38,
                        "num_edges": 72,
                    },
                },
            )

    evaluator = DatagenFEMEvaluator(workspace_root=tmp_path, fem_backend="remote_forward")
    monkeypatch.setattr(evaluator, "_remote_forward_client", lambda: FakeRemoteForwardClient())

    evaluation = evaluator.evaluate_explicit_structure(
        {
            "structure_id": "remote_forward:predicted_property",
            "artifacts": {"gpkl": "remote/design.gpkl"},
        },
        {"stress_curve": [0.0, 0.4, 0.8], "strain_grid": [0.0, 0.15, 0.3]},
    )

    assert evaluation["fem_status"] == "success"
    assert evaluation["property_error"]["curve_nmae"] == 0.0
    assert evaluation["evaluated_property"]["density_proxy"] == 0.3
    assert evaluation["raw_metrics"]["remote_forward_status"] == "success"
    assert len(evaluation["raw_metrics"]["stress_curve"]) == 256
    assert evaluation["raw_metrics"]["stress_curve"][0] == 0.0
    assert evaluation["raw_metrics"]["stress_curve"][-1] == 0.8
