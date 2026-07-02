from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..curve_targets import normalize_target_property, resample_stress_curve


DEFAULT_REMOTE_PYTHON = "/public/home/qingfang/.conda/envs/abaqus/bin/python"
DEFAULT_REMOTE_ABAQUS = "/public/home/qingfang/abaqus/Commands/abq2022"
DEFAULT_REMOTE_ROOT = "/public/home/qingfang/gid_c3d4_closed_loop"
DEFAULT_PIPELINE = Path(__file__).resolve().parent / "gid_c3d4_pipeline" / "gid_c3d4_pipeline.py"


@dataclass(frozen=True)
class GidC3D4RemoteConfig:
    """Remote CPU execution settings for GraphMetaMat C3D4 evaluations."""

    ssh_alias: str = field(default_factory=lambda: os.getenv("GID_C3D4_REMOTE_SSH_ALIAS", "qingfang@210.45.73.118"))
    ssh_key: str = field(default_factory=lambda: os.getenv("GID_C3D4_REMOTE_SSH_KEY", ""))
    ssh_options: tuple[str, ...] = ()
    remote_root: str = field(default_factory=lambda: os.getenv("GID_C3D4_REMOTE_ROOT", DEFAULT_REMOTE_ROOT))
    remote_python: str = field(default_factory=lambda: os.getenv("GID_C3D4_REMOTE_PYTHON", DEFAULT_REMOTE_PYTHON))
    remote_abaqus: str = field(default_factory=lambda: os.getenv("GID_C3D4_REMOTE_ABAQUS", DEFAULT_REMOTE_ABAQUS))
    nodes: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            item.strip()
            for item in os.getenv("GID_C3D4_REMOTE_NODES", "cnode1,cnode2").split(",")
            if item.strip()
        )
    )
    cpus_per_job: int = field(default_factory=lambda: int(os.getenv("GID_C3D4_CPUS_PER_JOB", "8")))
    max_parallel: int = field(default_factory=lambda: int(os.getenv("GID_C3D4_MAX_PARALLEL", "2")))
    array: int = field(default_factory=lambda: int(os.getenv("GID_C3D4_ARRAY", "1")))
    k_min: float = field(default_factory=lambda: float(os.getenv("GID_C3D4_K_MIN", "0.8")))
    k_max: float = field(default_factory=lambda: float(os.getenv("GID_C3D4_K_MAX", "1.2")))
    young: float = field(default_factory=lambda: float(os.getenv("GID_C3D4_YOUNG", "7.0")))
    normalized_cell_size_mm: float = 10.0
    pipeline_path: Path = DEFAULT_PIPELINE
    download_odb: bool = False
    run_remote: bool = True
    timeout_seconds: int = field(default_factory=lambda: int(os.getenv("GID_C3D4_REMOTE_TIMEOUT", "21600")))

    def ssh_base(self) -> list[str]:
        command = ["ssh"]
        if self.ssh_key:
            command.extend(["-i", self.ssh_key])
        command.extend(self.ssh_options)
        command.append(self.ssh_alias)
        return command

    def scp_base(self) -> list[str]:
        command = ["scp"]
        if self.ssh_key:
            command.extend(["-i", self.ssh_key])
        command.extend(self.ssh_options)
        return command


class RemoteGidC3D4Evaluator:
    """Evaluate inverse-designer GraphMetaMat trusses with remote C3D4 FEM.

    The evaluator accepts the structure dictionaries emitted by the remote
    inverse designer (`coordinates`, `edges`, `edge_radii`, `rho`) and stages
    them as `gid_c3d4_pipeline.py` input folders.  It then uploads a self
    contained batch to the CPU server, runs one foreground job per worker/node,
    downloads `data.csv`/plots/logs, and returns scheduler-compatible
    evaluation dictionaries.
    """

    def __init__(
        self,
        *,
        workspace_root: str | Path = "workspace",
        config: GidC3D4RemoteConfig | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.config = config or GidC3D4RemoteConfig()

    def evaluate_explicit_structure(self, structure: dict[str, Any], target_property: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.evaluate_many_explicit_structures([structure], [target_property or {}])[0]

    def evaluate_many_explicit_structures(
        self,
        structures: list[dict[str, Any]],
        target_properties: list[dict[str, Any] | None] | None = None,
    ) -> list[dict[str, Any]]:
        if not structures:
            return []
        targets = list(target_properties or [{} for _ in structures])
        if len(targets) != len(structures):
            raise ValueError("target_properties length must match structures length")

        batch = self._prepare_batch(structures, targets)
        if not self.config.run_remote:
            return [
                self._input_generated_evaluation(item, batch)
                for item in batch["manifest"]
            ]

        self._upload_and_run(batch)
        self._download_outputs(batch)
        return [
            self._evaluation_from_download(item, batch)
            for item in batch["manifest"]
        ]

    def _prepare_batch(self, structures: list[dict[str, Any]], targets: list[dict[str, Any] | None]) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        batch_id = f"gid_c3d4_{timestamp}_{_short_hash(json.dumps([_structure_id(s) for s in structures], sort_keys=True))}"
        local_root = self.workspace_root / "gid_c3d4_remote_batches" / batch_id
        data_root = local_root / "data"
        data_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.config.pipeline_path, local_root / "gid_c3d4_pipeline.py")
        (local_root / "run_gid_c3d4_batch.py").write_text(_REMOTE_BATCH_RUNNER, encoding="utf-8")

        manifest = []
        for index, (structure, target) in enumerate(zip(structures, targets)):
            target_norm = normalize_target_property(target or structure.get("scheduled_target") or structure.get("target_property") or {})
            case_name = f"c{index:04d}_{_short_hash(_structure_id(structure), length=12)}"
            case_dir = data_root / case_name
            export_info = export_structure_to_gid_dir(
                structure,
                case_dir,
                target_property=target_norm,
                cell_size_mm=self.config.normalized_cell_size_mm,
                gid=case_name,
            )
            manifest.append(
                {
                    "case": case_name,
                    "index": index,
                    "structure_id": _structure_id(structure),
                    "target_property": target_norm,
                    "export": export_info,
                }
            )

        manifest_path = local_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        input_tar = local_root / "input.tgz"
        with tarfile.open(input_tar, "w:gz") as tar:
            for name in ("gid_c3d4_pipeline.py", "run_gid_c3d4_batch.py", "manifest.json", "data"):
                tar.add(local_root / name, arcname=name)
        remote_dir = f"{self.config.remote_root.rstrip('/')}/{batch_id}"
        return {
            "batch_id": batch_id,
            "local_root": local_root,
            "input_tar": input_tar,
            "remote_dir": remote_dir,
            "manifest": manifest,
        }

    def _upload_and_run(self, batch: dict[str, Any]) -> None:
        remote_dir = str(batch["remote_dir"])
        self._run_ssh(f"rm -rf {_q(remote_dir)} && mkdir -p {_q(remote_dir)}")
        self._run([*self.config.scp_base(), str(batch["input_tar"]), f"{self.config.ssh_alias}:{remote_dir}/input.tgz"])
        self._run_ssh(f"cd {_q(remote_dir)} && tar -xzf input.tgz")
        nodes = ",".join(self.config.nodes)
        command = (
            f"cd {_q(remote_dir)} && {_q(self.config.remote_python)} run_gid_c3d4_batch.py "
            f"--nodes {_q(nodes)} "
            f"--python {_q(self.config.remote_python)} "
            f"--abaqus {_q(self.config.remote_abaqus)} "
            f"--cpus {int(self.config.cpus_per_job)} "
            f"--max-workers {max(1, int(self.config.max_parallel))} "
            f"--array {int(self.config.array)} "
            f"--k-min {float(self.config.k_min)} "
            f"--k-max {float(self.config.k_max)} "
            f"--young {float(self.config.young)}"
        )
        self._run_ssh(command, timeout=self.config.timeout_seconds)

    def _download_outputs(self, batch: dict[str, Any]) -> None:
        remote_dir = str(batch["remote_dir"])
        output_tar = Path(batch["local_root"]) / "outputs.tgz"
        patterns = [
            "-name data.csv",
            "-name '*_compare.png'",
            "-name '*.log'",
            "-name batch_results.json",
            "-name meta.json",
        ]
        if self.config.download_odb:
            patterns.append("-name '*.odb'")
        find_expr = " -o ".join(patterns)
        self._run_ssh(
            f"cd {_q(remote_dir)} && find . -type f \\( {find_expr} \\) -print0 | "
            "tar --null -czf outputs.tgz --files-from -"
        )
        self._run([*self.config.scp_base(), f"{self.config.ssh_alias}:{remote_dir}/outputs.tgz", str(output_tar)])
        with tarfile.open(output_tar, "r:gz") as tar:
            tar.extractall(Path(batch["local_root"]) / "download")

    def _evaluation_from_download(self, item: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        case = item["case"]
        download_root = Path(batch["local_root"]) / "download"
        run_root = download_root / "runs" / case
        data_candidates = list(run_root.glob("*/data.csv"))
        if not data_candidates:
            raise RuntimeError(f"C3D4 remote case {case} did not produce data.csv")
        data_csv = data_candidates[0]
        target = normalize_target_property(item.get("target_property") or {})
        evaluated = _curve_from_data_csv(data_csv, target)
        return {
            "structure_id": item["structure_id"],
            "evaluated_property": evaluated,
            "raw_metrics": {
                "backend": "RemoteGidC3D4Evaluator",
                "batch_id": batch["batch_id"],
                "case": case,
                "remote_dir": batch["remote_dir"],
                "local_batch_dir": str(batch["local_root"]),
                "raw_curve_csv": str(data_csv),
                "compare_png": _first_path(run_root.glob("*_compare.png")),
                "log_path": _first_path((download_root / "logs").glob(f"{case}.log")),
                "export": item.get("export", {}),
            },
            "fem_status": "success",
            "geometry_status": "valid",
        }

    @staticmethod
    def _input_generated_evaluation(item: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "structure_id": item["structure_id"],
            "evaluated_property": dict(item.get("target_property") or {}),
            "raw_metrics": {
                "backend": "RemoteGidC3D4Evaluator",
                "batch_id": batch["batch_id"],
                "local_batch_dir": str(batch["local_root"]),
                "case": item["case"],
                "export": item.get("export", {}),
            },
            "fem_status": "input_generated",
            "geometry_status": "valid",
        }

    def _run_ssh(self, remote_command: str, *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        return self._run([*self.config.ssh_base(), remote_command], timeout=timeout)

    @staticmethod
    def _run(command: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Command failed with exit code "
                f"{completed.returncode}: {' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        return completed


def export_structure_to_gid_dir(
    structure: dict[str, Any],
    out_dir: str | Path,
    *,
    target_property: dict[str, Any] | None = None,
    cell_size_mm: float = 10.0,
    gid: str | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    coordinates = _coordinates(structure)
    edges = _edges(structure)
    if not coordinates or not edges:
        raise ValueError("structure must contain coordinates and edges")
    max_abs = max(abs(value) for point in coordinates for value in point)
    normalized = max_abs <= 1.05
    scale = float(cell_size_mm) / 2.0 if normalized else 1.0
    radius = _radius(structure)
    if radius is None:
        raise ValueError("structure must contain edge_radii, radius, or beam_radius")
    radius_mm = radius * scale if normalized else radius
    rho = _rho(structure)

    with (out / "nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "x_mm", "y_mm", "z_mm"])
        for node_id, point in enumerate(coordinates):
            writer.writerow([node_id, point[0] * scale, point[1] * scale, point[2] * scale])

    with (out / "struts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_i", "node_j", "length_mm", "radius_mm"])
        for lhs, rhs in edges:
            p = coordinates[lhs]
            q = coordinates[rhs]
            length = math.dist([value * scale for value in p], [value * scale for value in q])
            writer.writerow([lhs, rhs, length, radius_mm])

    target_norm = normalize_target_property(target_property or {})
    if target_norm.get("strain_grid") and target_norm.get("stress"):
        with (out / "reference_curve.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["strain", "stress_normalized_by_Es"])
            for x, y in zip(target_norm["strain_grid"], target_norm["stress"]):
                writer.writerow([x, y])

    meta = {
        "gid": gid or _safe_name(_structure_id(structure), max_len=48),
        "source_structure_id": _structure_id(structure),
        "unit_cell_size_L_mm": float(cell_size_mm) if normalized else _cell_from_bounds(coordinates),
        "strut_radius_mm": radius_mm,
        "normalized_radius": radius if normalized else None,
        "relative_density_rho": rho,
        "nodes": len(coordinates),
        "struts": len(edges),
        "curve": {
            "stress_max_Es7scale": max([abs(float(value)) for value in target_norm.get("stress", [])], default=None),
        },
        "export_note": "normalized [-1,1] coordinates scaled to GraphMetaMat 10 mm cell" if normalized else "coordinates treated as millimeters",
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "struct_dir": str(out),
        "normalized_input": normalized,
        "coordinate_scale": scale,
        "radius_mm": radius_mm,
        "rho": rho,
        "nodes": len(coordinates),
        "edges": len(edges),
    }


def _curve_from_data_csv(data_csv: Path, target: dict[str, Any]) -> dict[str, Any]:
    strain: list[float] = []
    stress: list[float] = []
    with data_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                strain.append(abs(float(row["Strain"])))
                stress.append(abs(float(row["Stress_MPa"])))
            except (KeyError, TypeError, ValueError):
                continue
    grid = target.get("strain_grid") if isinstance(target.get("strain_grid"), list) else None
    if grid:
        return normalize_target_property(
            {
                "type": "stress_curve",
                "strain_grid": list(grid),
                "stress": resample_stress_curve(strain, stress, list(grid)),
            }
        )
    return normalize_target_property({"type": "stress_curve", "strain_grid": strain, "stress": stress})


def _coordinates(structure: dict[str, Any]) -> list[list[float]]:
    payload = structure.get("coordinates") or structure.get("nodes") or []
    result: list[list[float]] = []
    for item in payload:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            return []
        result.append([float(item[0]), float(item[1]), float(item[2])])
    return result


def _edges(structure: dict[str, Any]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for item in structure.get("edges") or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return []
        lhs, rhs = int(item[0]), int(item[1])
        if lhs != rhs:
            result.append((lhs, rhs))
    return result


def _radius(structure: dict[str, Any]) -> float | None:
    radii = structure.get("edge_radii")
    if isinstance(radii, list) and radii:
        return float(radii[0])
    for key in ("radius", "beam_radius", "strut_radius_mm"):
        if key in structure and structure[key] is not None:
            return float(structure[key])
    return None


def _rho(structure: dict[str, Any]) -> float | None:
    for key in ("rho", "relative_density", "density_proxy"):
        if key in structure and structure[key] is not None:
            return float(structure[key])
    predicted = structure.get("predicted_property")
    if isinstance(predicted, dict) and predicted.get("rho") is not None:
        return float(predicted["rho"])
    return None


def _cell_from_bounds(coordinates: list[list[float]]) -> float:
    spans = []
    for axis in range(3):
        values = [point[axis] for point in coordinates]
        spans.append(max(values) - min(values))
    return max(spans) if spans else 10.0


def _structure_id(structure: dict[str, Any]) -> str:
    return str(structure.get("structure_id") or structure.get("sample_id") or "structure")


def _safe_name(value: str, *, max_len: int = 80) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return (name or "structure")[:max_len]


def _short_hash(value: str, *, length: int = 10) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:length]


def _q(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def _first_path(paths: Any) -> str:
    for path in paths:
        return str(path)
    return ""


_REMOTE_BATCH_RUNNER = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
from pathlib import Path


def q(value):
    return shlex.quote(str(value))


def count_rows(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return 0


def run_case(item, node, args):
    case = item["case"]
    root = Path.cwd()
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    run_out = f"runs/{case}"
    cmd = (
        f"cd {q(str(root))} && "
        "export LM_LICENSE_FILE=27800@login01 && "
        f"rm -rf {q(run_out)} && "
        f"{q(args.python_cmd)} gid_c3d4_pipeline.py "
        f"--struct-dir {q('data/' + case)} "
        f"--out {q(run_out)} "
        f"--array {int(args.array)} "
        f"--cpus {int(args.cpus)} "
        f"--abaqus {q(args.abaqus)} "
        f"--k-min {float(args.k_min)} "
        f"--k-max {float(args.k_max)} "
        f"--young {float(args.young)}"
    )
    if node:
        full_cmd = f"ssh -o StrictHostKeyChecking=no {q(node)} {q(cmd)}"
    else:
        full_cmd = cmd
    log_path = log_dir / f"{case}.log"
    with open(log_path, "w", encoding="utf-8", errors="ignore") as handle:
        completed = subprocess.run(full_cmd, shell=True, stdout=handle, stderr=subprocess.STDOUT, check=False)
    data_paths = list((root / "runs" / case).glob("*/data.csv"))
    png_paths = list((root / "runs" / case).glob("*_compare.png"))
    return {
        "case": case,
        "node": node,
        "returncode": int(completed.returncode),
        "log": str(log_path),
        "data_csv": str(data_paths[0]) if data_paths else "",
        "compare_png": str(png_paths[0]) if png_paths else "",
        "curve_points": count_rows(data_paths[0]) if data_paths else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--nodes", default="")
    parser.add_argument("--python", dest="python_cmd", required=True)
    parser.add_argument("--abaqus", required=True)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--array", type=int, default=1)
    parser.add_argument("--k-min", type=float, default=0.8)
    parser.add_argument("--k-max", type=float, default=1.2)
    parser.add_argument("--young", type=float, default=7.0)
    args = parser.parse_args()

    manifest = json.load(open(args.manifest, "r", encoding="utf-8"))
    nodes = [item.strip() for item in args.nodes.split(",") if item.strip()]
    workers = max(1, int(args.max_workers))
    results = [None] * len(manifest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(manifest))) as executor:
        future_to_index = {}
        for index, item in enumerate(manifest):
            node = nodes[index % len(nodes)] if nodes else ""
            future_to_index[executor.submit(run_case, item, node, args)] = index
        for future in concurrent.futures.as_completed(future_to_index):
            results[future_to_index[future]] = future.result()
    with open("batch_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    failures = [item for item in results if item.get("returncode") != 0 or not item.get("data_csv")]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''
