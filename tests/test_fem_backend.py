from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.DatagenFEMEvaluator import AbaqusFEMConfig, DatagenFEMEvaluator
from src.DatagenFEMEvaluator.core import fem as core_fem


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
