from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CORE = SRC / "DatagenFEMEvaluator" / "core" / "truss"
CORE_GROUP_DB = CORE / "symmetry_group_transforms.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api import (
    convert_csv_to_abaqus,
    deduplicate_architecture_csv,
    expand_crystal_structure,
    export_txt_to_vtk,
    run_group_pipeline,
)
from src.DatagenFEMEvaluator import (
    load_abaqus_module,
    load_crystal_module,
    load_generation_module,
    load_relation_module,
    load_truss_txt,
    preview_generation_batch,
    solve_constraints,
)
from src.datagen_contracts import GeneratorConfig, PipelineConfig


class RefactorCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy_generation = load_generation_module()
        cls.legacy_relation = load_relation_module()
        cls.legacy_abaqus = load_abaqus_module()
        cls.legacy_crystal = load_crystal_module()
        cls.node_names = cls.legacy_generation.GeometryGenerator(
            cls.legacy_generation.TrussConfig()
        ).node_names_ordered

    def _write_architecture_csv(self, csv_path: Path, rows):
        header = ["id", "name"]
        for name in self.node_names:
            header.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
        for i in range(len(self.node_names) ** 2):
            header.append(f"element_{i + 1}")

        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for row_id, payload in enumerate(rows):
                nodes_flat, adj_flat, _L, _bars = payload
                writer.writerow([row_id, f"sample_{row_id}"] + nodes_flat + adj_flat)

    def test_preview_batch_matches_legacy_without_constraints(self):
        config = GeneratorConfig(
            OUTPUT_DIR=str(ROOT / "tmp_test_out"),
            CSV_NAME="preview.csv",
            TARGET_SAMPLES=1,
            N_WORKERS=1,
            BATCH_PER_TASK=1,
        )
        seed = 12345
        batch_size = 2

        legacy_batch = self.legacy_generation.worker_generate_batch(config.to_kwargs(), batch_size, seed)
        refactored_batch = preview_generation_batch(config, batch_size=batch_size, seed=seed)

        self.assertEqual(legacy_batch, refactored_batch)
        self.assertGreater(len(refactored_batch), 0)

    def test_preview_batch_matches_legacy_with_constraints(self):
        with tempfile.TemporaryDirectory(prefix="truss_constraints_") as tmp_dir:
            constraints_path = Path(tmp_dir) / "constraints_P222.json"
            solve_constraints("P222", export_path=constraints_path, show_plot=False)

            config = GeneratorConfig(
                OUTPUT_DIR=tmp_dir,
                CSV_NAME="preview.csv",
                TARGET_SAMPLES=1,
                CONSTRAINTS_JSON=str(constraints_path),
                N_WORKERS=1,
                BATCH_PER_TASK=1,
            )
            seed = 24680
            batch_size = 1

            legacy_batch = self.legacy_generation.worker_generate_batch(config.to_kwargs(), batch_size, seed)
            refactored_batch = preview_generation_batch(config, batch_size=batch_size, seed=seed)

            self.assertEqual(legacy_batch, refactored_batch)
            self.assertGreater(len(refactored_batch), 0)

    def test_solve_constraints_matches_legacy_payload(self):
        with tempfile.TemporaryDirectory(prefix="truss_constraints_cmp_") as tmp_dir:
            export_path = Path(tmp_dir) / "constraints.json"
            refactored = solve_constraints("P222", export_path=export_path, show_plot=False)
            legacy = self.legacy_relation.solve_and_visualize_constraints(
                group_name="P222",
                db_path=str(CORE_GROUP_DB),
                export_path=None,
                show_plot=False,
            )

            self.assertEqual(refactored.payload, legacy)
            self.assertEqual(refactored.lattice_lengths, legacy.get("lattice_lengths"))
            self.assertTrue(export_path.exists())

            exported = json.loads(export_path.read_text(encoding="utf-8"))
            self.assertEqual(exported, legacy)

    def test_csv_to_abaqus_matches_legacy_output(self):
        config = GeneratorConfig(
            OUTPUT_DIR=str(ROOT / "tmp_test_out"),
            CSV_NAME="compat.csv",
            TARGET_SAMPLES=1,
        )
        sample_batch = preview_generation_batch(config, batch_size=1, seed=12345)
        self.assertEqual(len(sample_batch), 1)

        with tempfile.TemporaryDirectory(prefix="truss_abaqus_cmp_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "architecture.csv"
            ref_dir = tmp_path / "ref"
            legacy_dir = tmp_path / "legacy"
            self._write_architecture_csv(csv_path, sample_batch)

            result = convert_csv_to_abaqus(
                csv_path=str(csv_path),
                out_dir=str(ref_dir),
                group_name="P222",
                group_db_path=str(CORE_GROUP_DB),
            )
            self.assertEqual(result.txt_count, 1)

            generator = self.legacy_abaqus.TrussGenerator(
                str(csv_path),
                group_name="P222",
                group_db_path=str(CORE_GROUP_DB),
            )
            nodes, edges, name = generator.process_row(0)
            legacy_dir.mkdir(parents=True, exist_ok=True)
            generator.save_to_txt(str(legacy_dir / "0.txt"), nodes, edges, name)

            self.assertEqual(
                (ref_dir / "0.txt").read_text(encoding="utf-8"),
                (legacy_dir / "0.txt").read_text(encoding="utf-8"),
            )

    def test_crystal_expansion_matches_legacy_output(self):
        config = GeneratorConfig(
            OUTPUT_DIR=str(ROOT / "tmp_test_out"),
            CSV_NAME="compat.csv",
            TARGET_SAMPLES=1,
        )
        sample_batch = preview_generation_batch(config, batch_size=1, seed=12345)
        self.assertEqual(len(sample_batch), 1)

        with tempfile.TemporaryDirectory(prefix="truss_crystal_cmp_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "architecture.csv"
            abaqus_dir = tmp_path / "abaqus_txt"
            ref_crystal_dir = tmp_path / "ref_crystal"
            legacy_crystal_dir = tmp_path / "legacy_crystal"
            self._write_architecture_csv(csv_path, sample_batch)

            convert_csv_to_abaqus(
                csv_path=str(csv_path),
                out_dir=str(abaqus_dir),
                group_name="P222",
                group_db_path=str(CORE_GROUP_DB),
            )
            result = expand_crystal_structure(str(abaqus_dir), str(ref_crystal_dir), nx=2, ny=2, nz=4)
            self.assertEqual(result.failed, 0)
            self.assertEqual(result.processed, 1)

            legacy_crystal_dir.mkdir(parents=True, exist_ok=True)
            source_text = (abaqus_dir / "0.txt").read_text(encoding="utf-8")
            node_data, element_conn, places = self.legacy_crystal.parse_unitcell(source_text)
            global_nodes, global_elems = self.legacy_crystal.build_crystal(node_data, element_conn, places, 2, 2, 4)
            out_text = self.legacy_crystal.format_output(global_nodes, global_elems, "0")
            (legacy_crystal_dir / "0.txt").write_text(out_text, encoding="utf-8")

            self.assertEqual(
                (ref_crystal_dir / "0.txt").read_text(encoding="utf-8"),
                (legacy_crystal_dir / "0.txt").read_text(encoding="utf-8"),
            )

    def test_deduplicate_architecture_csv_reindexes_and_keeps_unique_topology(self):
        config = GeneratorConfig(
            OUTPUT_DIR=str(ROOT / "tmp_test_out"),
            CSV_NAME="dedup.csv",
            TARGET_SAMPLES=1,
        )
        sample_a = preview_generation_batch(config, batch_size=1, seed=12345)[0]
        sample_b = preview_generation_batch(config, batch_size=1, seed=24680)[0]

        with tempfile.TemporaryDirectory(prefix="truss_dedup_cmp_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "in.csv"
            out_path = tmp_path / "out.csv"
            self._write_architecture_csv(csv_path, [sample_a, sample_a, sample_b])

            saved = deduplicate_architecture_csv(csv_path, out_path)
            self.assertEqual(saved, 2)

            with out_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader)
                rows = list(reader)

            self.assertEqual(len(header), 420)
            self.assertEqual(len(rows), 2)
            self.assertEqual([row[0] for row in rows], ["0", "1"])

    def test_pipeline_smoke_produces_complete_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="truss_pipeline_smoke_") as tmp_dir:
            result = run_group_pipeline(
                PipelineConfig(
                    group="P222",
                    basic_size=4,
                    samples=1,
                    workers=1,
                    batch=1,
                    print_every=1,
                    run_dir=tmp_dir,
                )
            )

            self.assertTrue(Path(result.summary_path).exists())
            self.assertTrue(Path(result.generation.csv_path).exists())
            self.assertTrue(Path(result.constraints.constraints_path).exists())
            self.assertEqual(result.abaqus.txt_count, 1)
            self.assertEqual(result.crystal.failed, 0)
            self.assertEqual(result.replication, {"nx": 2, "ny": 2, "nz": 4})

    def test_auto_generate_script_matches_refactored_pipeline_contract(self):
        with tempfile.TemporaryDirectory(prefix="truss_auto_generate_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            legacy_run = tmp_path / "legacy_run"
            ref_run = tmp_path / "ref_run"

            legacy_cmd = [
                sys.executable,
                str(CORE / "auto_generate_4x4x4.py"),
                "P222",
                "--samples",
                "1",
                "--workers",
                "1",
                "--batch",
                "1",
                "--print-every",
                "1",
                "--run-dir",
                str(legacy_run),
                "--allow-single-process-fallback",
            ]
            legacy_proc = subprocess.run(
                legacy_cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(legacy_proc.returncode, 0)

            result = run_group_pipeline(
                PipelineConfig(
                    group="P222",
                    basic_size=4,
                    samples=1,
                    workers=1,
                    batch=1,
                    print_every=1,
                    run_dir=str(ref_run),
                    allow_single_process_fallback=True,
                )
            )

            legacy_summary = json.loads((legacy_run / "summary.json").read_text(encoding="utf-8"))
            ref_summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))

            for key in (
                "group",
                "basic_size",
                "lattice_lengths",
                "replication",
                "samples_target",
                "abaqus_txt_count",
                "crystal_processed",
                "crystal_failed",
            ):
                self.assertEqual(legacy_summary[key], ref_summary[key], key)

            self.assertTrue((legacy_run / "constraints_P222.json").exists())
            self.assertTrue((legacy_run / "P222-architecture.csv").exists())
            self.assertTrue((legacy_run / "abaqus_txt" / "0.txt").exists())
            self.assertTrue((legacy_run / "crystal_4x4x4" / "0.txt").exists())

    def test_export_txt_to_vtk_single_and_directory(self):
        config = GeneratorConfig(
            OUTPUT_DIR=str(ROOT / "tmp_test_out"),
            CSV_NAME="vtk.csv",
            TARGET_SAMPLES=1,
        )
        sample_batch = preview_generation_batch(config, batch_size=1, seed=12345)
        self.assertEqual(len(sample_batch), 1)

        with tempfile.TemporaryDirectory(prefix="truss_vtk_cmp_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "architecture.csv"
            abaqus_dir = tmp_path / "abaqus_txt"
            self._write_architecture_csv(csv_path, sample_batch)
            convert_csv_to_abaqus(
                csv_path=str(csv_path),
                out_dir=str(abaqus_dir),
                group_name="P222",
                group_db_path=str(CORE_GROUP_DB),
            )

            single_result = export_txt_to_vtk(abaqus_dir / "0.txt")
            single_vtk = Path(single_result.output_path)
            self.assertTrue(single_vtk.exists())
            text = single_vtk.read_text(encoding="utf-8")
            node_ids, _points, lines = load_truss_txt(abaqus_dir / "0.txt")
            self.assertIn("DATASET POLYDATA", text)
            self.assertIn(f"POINTS {len(node_ids)} float", text)
            self.assertIn(f"LINES {len(lines)} {len(lines) * 3}", text)
            self.assertIn("SCALARS node_id int 1", text)
            self.assertIn("SCALARS edge_id int 1", text)

            dir_result = export_txt_to_vtk(abaqus_dir, tmp_path / "abaqus_vtk")
            self.assertEqual(len(dir_result.exported_files), 1)
            self.assertTrue((tmp_path / "abaqus_vtk" / "0.vtk").exists())

    def test_cli_pipeline_and_vtk_smoke(self):
        with tempfile.TemporaryDirectory(prefix="truss_cli_smoke_") as tmp_dir:
            run_dir = Path(tmp_dir) / "cli_pipeline"
            cmd = [
                sys.executable,
                "-m",
                "src",
                "pipeline",
                "P222",
                "--samples",
                "1",
                "--workers",
                "1",
                "--batch",
                "1",
                "--print-every",
                "1",
                "--run-dir",
                str(run_dir),
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env={**os.environ, "PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(proc.returncode, 0)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["group"], "P222")
            self.assertTrue((run_dir / "abaqus_txt" / "0.txt").exists())

            vtk_cmd = [
                sys.executable,
                "-m",
                "src",
                "vtk",
                "--input",
                str(run_dir / "abaqus_txt"),
                "--output",
                str(run_dir / "abaqus_vtk"),
            ]
            vtk_proc = subprocess.run(
                vtk_cmd,
                cwd=str(ROOT),
                env={**os.environ, "PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
                check=True,
            )
            vtk_payload = json.loads(vtk_proc.stdout.strip())
            self.assertEqual(vtk_payload["count"], 1)
            self.assertTrue((run_dir / "abaqus_vtk" / "0.vtk").exists())


if __name__ == "__main__":
    unittest.main()
