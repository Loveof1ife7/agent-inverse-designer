from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.TrainingDataset.inverse_truss_preprocess import (
    PROPERTY_STRAIN_GRID,
    build_inverse_truss_record,
    export_inverse_truss_dataset,
    parse_property_csv,
)


GEOMETRY_TXT = """
node_data = [
    [1, 0.0, 0.0, 0.0],
    [2, 2.0, 0.0, 0.0],
    [3, 0.0, 2.0, 0.0],
    [4, 2.0, 2.0, 2.0],
]

element_conn = [
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 4],
]
"""

PROPERTY_CSV = """Strain,Stress
0.0,0.0
0.1,1.0
0.1,3.0
0.2,-5.0
0.25,5.0
"""

NON_AUTOREGRESSIVE_ORIGINAL_ORDER_GEOMETRY_TXT = """
node_data = [
    [1, 0.0, 0.0, 0.0],
    [2, 2.0, 0.0, 0.0],
    [3, 0.0, 2.0, 0.0],
    [4, 2.0, 2.0, 0.0],
]

element_conn = [
    [1, 4],
    [4, 2],
    [4, 3],
    [2, 3],
]
"""


class InverseTrussPreprocessTests(unittest.TestCase):
    def test_build_record_exports_property_and_geometry_supervision(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            geometry_dir = root / "structures"
            properties_dir = root / "properties"
            geometry_dir.mkdir()
            properties_dir.mkdir()
            geometry_path = geometry_dir / "7.txt"
            geometry_path.write_text(GEOMETRY_TXT, encoding="utf-8")
            (properties_dir / "7.csv").write_text(PROPERTY_CSV, encoding="utf-8")

            record = build_inverse_truss_record(geometry_path, root)

        self.assertEqual(record["sample_id"], "7")
        self.assertEqual(record["n"], 4)
        self.assertEqual(record["edges"], [[1, 2], [1, 3], [2, 4], [3, 4]])
        self.assertEqual(record["parent_sequence"], [1, 1, 2])
        self.assertEqual(record["k"], 1)
        self.assertEqual(record["extra_edges"], [[3, 4]])
        self.assertEqual(len(record["coordinate_tokens"]), 12)
        self.assertEqual(record["coordinates"][0], [0.0, 0.0, 0.0])
        self.assertEqual(record["coordinates"][3], [1.0, 1.0, 1.0])
        self.assertEqual(record["preprocessing"]["index_base"], 1)
        self.assertEqual(len(record["y"]), len(PROPERTY_STRAIN_GRID))
        self.assertEqual(record["property"]["representation"], "stress_grid_v1")
        self.assertEqual(record["property"]["duplicate_strain_count"], 1)
        self.assertEqual(record["property"]["negative_stress_count"], 1)
        self.assertEqual(record["property"]["extrapolated_point_count"], 5)

    def test_parent_sequence_is_reindexed_for_autoregressive_training(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            geometry_dir = root / "structures"
            properties_dir = root / "properties"
            geometry_dir.mkdir()
            properties_dir.mkdir()
            geometry_path = geometry_dir / "0.txt"
            geometry_path.write_text(NON_AUTOREGRESSIVE_ORIGINAL_ORDER_GEOMETRY_TXT, encoding="utf-8")
            (properties_dir / "0.csv").write_text(PROPERTY_CSV, encoding="utf-8")

            record = build_inverse_truss_record(geometry_path, root)

        self.assertEqual(record["preprocessing"]["node_order_original_ids"], [1, 4, 2, 3])
        self.assertEqual(record["parent_sequence"], [1, 2, 2])
        self.assertTrue(all(parent < child for child, parent in enumerate(record["parent_sequence"], start=2)))
        self.assertEqual(record["edges"], [[1, 2], [2, 3], [2, 4], [3, 4]])
        self.assertEqual(record["extra_edges"], [[3, 4]])
        self.assertEqual(record["coordinates"][1], [1.0, 1.0, 0.0])
        self.assertEqual(record["preprocessing"]["topology_tree_method"], "bfs_parent_pointer_reindexed_v2")

    def test_export_writes_jsonl_and_manifest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            geometry_dir = root / "structures"
            properties_dir = root / "properties"
            geometry_dir.mkdir(parents=True)
            properties_dir.mkdir()
            (geometry_dir / "0.txt").write_text(GEOMETRY_TXT, encoding="utf-8")
            (properties_dir / "0.csv").write_text(PROPERTY_CSV, encoding="utf-8")
            output_path = Path(tmp) / "out.jsonl"
            manifest_path = Path(tmp) / "manifest.json"

            manifest = export_inverse_truss_dataset(
                dataset_root=root,
                output_path=output_path,
                manifest_path=manifest_path,
            )

            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertTrue(manifest_path.exists())
            self.assertEqual(manifest["counts"], {"written": 1, "skipped": 0})
            self.assertEqual(manifest["property_status"], "stress_grid_v1")

    def test_parse_property_csv_samples_fixed_strain_grid(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "property.csv"
            path.write_text(PROPERTY_CSV, encoding="utf-8")

            parsed = parse_property_csv(path)

        self.assertEqual(len(parsed.y), 31)
        self.assertEqual(parsed.metadata["strain_grid"][0], 0.0)
        self.assertEqual(parsed.metadata["strain_grid"][-1], 0.3)
        self.assertAlmostEqual(parsed.y[10], 2.0)
        self.assertAlmostEqual(parsed.y[20], 0.0)
        self.assertAlmostEqual(parsed.y[30], 5.0)

    def test_geometry_only_mode_keeps_empty_y(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            geometry_dir = root / "structures"
            properties_dir = root / "properties"
            geometry_dir.mkdir()
            properties_dir.mkdir()
            geometry_path = geometry_dir / "7.txt"
            geometry_path.write_text(GEOMETRY_TXT, encoding="utf-8")

            record = build_inverse_truss_record(geometry_path, root, include_property=False)

        self.assertEqual(record["y"], [])
        self.assertNotIn("property", record)


if __name__ == "__main__":
    unittest.main()
