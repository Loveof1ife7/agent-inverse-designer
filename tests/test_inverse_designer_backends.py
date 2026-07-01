from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pytest

from src.InverseDesigner import GraphMetaMatTrussBackend


class InverseDesignerBackendTests(unittest.TestCase):
    def test_graphmetamat_backend_parses_cli_outputs(self):
        pytest.importorskip("networkx")
        with tempfile.TemporaryDirectory(prefix="graphmetamat_backend_") as tmp_dir:
            project_dir = Path(tmp_dir) / "GraphMetaMat"
            project_dir.mkdir()
            script = project_dir / "run_inverse_designer.py"
            script.write_text(
                """
import argparse
import json
import pickle
from pathlib import Path

import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument("--target")
parser.add_argument("--out-dir")
parser.add_argument("--num-runs")
parser.add_argument("--top-k")
parser.add_argument("--device")
args = parser.parse_args()

out_dir = Path(args.out_dir)
export_dir = out_dir / "design_exports"
export_dir.mkdir(parents=True, exist_ok=True)
graph = nx.Graph()
graph.add_node(0, coord=[0.0, 0.0, 0.0])
graph.add_node(1, coord=[1.0, 0.0, 0.0])
graph.add_edge(0, 1, radius=0.04)
graph.graph["rho"] = 0.12
gpkl = export_dir / "rank_01_sample_000.gpkl"
with gpkl.open("wb") as handle:
    pickle.dump(graph, handle)
(out_dir / "results.pkl").write_bytes(b"placeholder")
(export_dir / "summary.csv").write_text("index,mae,mse,jaccard,rho,num_nodes,num_edges\\n0,0.1,0.01,0.9,0.12,2,1\\n", encoding="utf-8")
(export_dir / "rank_01_sample_000.vtk").write_text("vtk", encoding="utf-8")
(export_dir / "rank_01_sample_000_graph.png").write_bytes(b"png")
(export_dir / "rank_01_sample_000_curves.png").write_bytes(b"png")
(export_dir / "top_designs.json").write_text(json.dumps([{
    "index": 0,
    "mae": 0.1,
    "mse": 0.01,
    "jaccard": 0.9,
    "rho": 0.12,
    "num_nodes": 2,
    "num_edges": 1,
    "gpkl": str(gpkl),
    "vtk": str(export_dir / "rank_01_sample_000.vtk"),
    "graph_png": str(export_dir / "rank_01_sample_000_graph.png"),
    "curve_png": str(export_dir / "rank_01_sample_000_curves.png"),
}]), encoding="utf-8")
""".strip()
                + "\n",
                encoding="utf-8",
            )
            backend = GraphMetaMatTrussBackend(
                name="graphmetamat_fake",
                project_dir=project_dir,
                num_runs=1,
                top_k=1,
                device="cpu",
            )

            structure = backend.sample(
                {"stress_curve": [0.0, 0.1, 0.2]},
                output_dir=Path(tmp_dir) / "out",
                sample_index=1,
            )

            self.assertIsNotNone(structure)
            self.assertEqual(structure["structure_family"], "truss")
            self.assertEqual(structure["representation"], "graph_truss")
            self.assertEqual(len(structure["coordinates"]), 2)
            self.assertEqual(structure["edges"], [[0, 1]])
            self.assertEqual(structure["predicted_property"]["jaccard"], 0.9)
            self.assertTrue(Path(structure["artifacts"]["gpkl"]).exists())


if __name__ == "__main__":
    unittest.main()
