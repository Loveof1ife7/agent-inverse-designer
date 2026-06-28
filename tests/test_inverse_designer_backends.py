from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.InverseDesigner import GraphMetaMatTrussBackend, InverseDesigner
from src.KnowledgeBase import KnowledgeBase
from src.closed_loop_contracts import TargetSchedule, TargetScheduleItem


class FakeBackend:
    def __init__(self, name: str, structure_family: str, representation: str = "fake"):
        self.name = name
        self.structure_family = structure_family
        self.representation = representation
        self.calls: list[dict[str, Any]] = []

    def available(self) -> bool:
        return True

    def sample(self, target_property: dict[str, float], output_dir: Path, sample_index: int = 1) -> dict[str, Any] | None:
        self.calls.append(
            {
                "target_property": dict(target_property),
                "output_dir": str(output_dir),
                "sample_index": sample_index,
            }
        )
        return {
            "structure_id": f"{self.structure_family}_{sample_index}",
            "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "edges": [[0, 1]],
        }


class InverseDesignerBackendTests(unittest.TestCase):
    def test_sample_structure_routes_to_requested_neural_family(self):
        with tempfile.TemporaryDirectory(prefix="inverse_backend_") as tmp_dir:
            kb = KnowledgeBase(Path(tmp_dir) / "kb.sqlite")
            try:
                tpms = FakeBackend("tpms_fake", "tpms")
                voxel = FakeBackend("voxel_fake", "voxel")
                inverse = InverseDesigner(
                    kb,
                    neural_backends=[tpms, voxel],
                    enable_neural=True,
                    workspace_root=Path(tmp_dir) / "workspace",
                )

                structure = inverse.sample_structure(
                    {"stiffness_proxy": 0.5, "density_proxy": 0.2},
                    structure_family="voxel",
                )

                self.assertIsNotNone(structure)
                self.assertEqual(structure["structure_family"], "voxel")
                self.assertEqual(structure["neural_backend"], "voxel_fake")
                self.assertEqual(len(tpms.calls), 0)
                self.assertEqual(len(voxel.calls), 1)
            finally:
                kb.close()

    def test_sample_schedule_round_robins_neural_families(self):
        with tempfile.TemporaryDirectory(prefix="inverse_schedule_backend_") as tmp_dir:
            kb = KnowledgeBase(Path(tmp_dir) / "kb.sqlite")
            try:
                tpms = FakeBackend("tpms_fake", "tpms")
                truss = FakeBackend("truss_fake", "truss")
                voxel = FakeBackend("voxel_fake", "voxel")
                inverse = InverseDesigner(
                    kb,
                    neural_backends=[tpms, truss, voxel],
                    enable_neural=True,
                    workspace_root=Path(tmp_dir) / "workspace",
                )
                schedule = TargetSchedule(
                    schedule_id="sched",
                    final_target={"stiffness_proxy": 0.5},
                    scheduled_targets=[
                        TargetScheduleItem(target_id="t1", target_property={"stiffness_proxy": 0.5}),
                        TargetScheduleItem(target_id="t2", target_property={"stiffness_proxy": 0.6}),
                        TargetScheduleItem(target_id="t3", target_property={"stiffness_proxy": 0.7}),
                    ],
                )

                records = inverse.sample_schedule(schedule)

                self.assertEqual([record["structure_family"] for record in records], ["tpms", "truss", "voxel"])
                self.assertEqual([record["structure"]["neural_backend"] for record in records], ["tpms_fake", "truss_fake", "voxel_fake"])
            finally:
                kb.close()

    def test_retrieval_fallback_still_works_when_neural_disabled(self):
        with tempfile.TemporaryDirectory(prefix="inverse_retrieval_fallback_") as tmp_dir:
            kb = KnowledgeBase(Path(tmp_dir) / "kb.sqlite")
            try:
                inverse = InverseDesigner(kb, neural_backends=[FakeBackend("tpms_fake", "tpms")], enable_neural=False)
                inverse.train(
                    [
                        {
                            "sample_id": "retrieval_sample",
                            "property": {"stiffness_proxy": 1.0},
                            "explicit_structure": {
                                "structure_id": "retrieval_sample",
                                "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                                "edges": [[0, 1]],
                            },
                            "validity": {"geometry_status": "valid", "fem_status": "success"},
                        }
                    ]
                )

                structure = inverse.sample_structure({"stiffness_proxy": 1.0})

                self.assertIsNotNone(structure)
                self.assertEqual(structure["structure_id"], "retrieval_sample")
                self.assertEqual(structure["source"], "inverse_designer_retrieval")
            finally:
                kb.close()

    def test_graphmetamat_backend_parses_cli_outputs(self):
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
