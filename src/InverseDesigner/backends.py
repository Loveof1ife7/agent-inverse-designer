from __future__ import annotations

import csv
import importlib
import json
import os
import pickle
import shutil
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = PROJECT_ROOT / "third-party"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def _safe_name(value: str, default: str = "sample") -> str:
    text = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))
    return text[:120] or default


def _target_vector(target_property: dict[str, float], keys: tuple[str, ...]) -> list[float]:
    for preferred_key in ("control_points_stress", "stress_curve", "target_curve", "curve", "target", "target_vector", "condition"):
        value = target_property.get(preferred_key)
        if isinstance(value, (list, tuple)):
            return [float(item) for item in value]
    if keys:
        values: list[float] = []
        for key in keys:
            value = target_property.get(key, 0.0)
            if isinstance(value, (list, tuple)):
                values.extend(float(item) for item in value)
            else:
                values.append(float(value))
        return values
    if len(target_property) == 1:
        value = next(iter(target_property.values()))
        if isinstance(value, (list, tuple)):
            return [float(item) for item in value]
    return [float(target_property[key]) for key in sorted(target_property)]


def _write_curve_csv(curve: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(curve)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if count <= 1:
            writer.writerow([0.0, float(curve[0]) if curve else 0.0])
            return
        for index, stress in enumerate(curve):
            strain = 0.3 * index / (count - 1)
            writer.writerow([strain, float(stress)])


def _resolve_output_path(project_dir: Path, path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.is_absolute():
        path = project_dir / path
    return str(path.resolve())


def _csv_first_column(path: Path) -> list[str]:
    if not path.exists():
        return []
    rows: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if row and row[0]:
                rows.append(str(row[0]))
    return rows


class NeuralInverseBackend(Protocol):
    name: str
    structure_family: str
    representation: str

    def available(self) -> bool:
        ...

    def sample(self, target_property: dict[str, float], output_dir: Path, sample_index: int = 1) -> dict[str, Any] | None:
        ...


@dataclass
class CallableBackend:
    """Adapter for project-specific Python wrappers.

    The callable should accept keyword arguments:
    target_property, checkpoint_path, output_dir, num_samples, backend_config.
    It may return either one structure dict or a list of structure dicts.
    """

    name: str
    structure_family: str
    representation: str
    callable_path: str
    checkpoint_path: str = ""
    num_samples: int = 1
    config: dict[str, Any] = field(default_factory=dict)

    def available(self) -> bool:
        return bool(self.callable_path) and (not self.checkpoint_path or Path(self.checkpoint_path).exists())

    def sample(self, target_property: dict[str, float], output_dir: Path, sample_index: int = 1) -> dict[str, Any] | None:
        module_name, _, function_name = self.callable_path.partition(":")
        if not module_name or not function_name:
            raise ValueError(f"{self.name} callable_path must be 'module:function'")
        output_dir.mkdir(parents=True, exist_ok=True)
        module = importlib.import_module(module_name)
        fn = getattr(module, function_name)
        result = fn(
            target_property=dict(target_property),
            checkpoint_path=self.checkpoint_path,
            output_dir=str(output_dir),
            num_samples=self.num_samples,
            backend_config=dict(self.config),
        )
        if isinstance(result, list):
            result = result[0] if result else None
        if not isinstance(result, dict) or not result:
            return None
        return _annotate_structure(result, self, sample_index)


@dataclass
class CommandBackend:
    """Adapter for shell/CLI inference wrappers.

    The command is formatted with checkpoint_path, output_dir, target_json,
    target_csv, num_samples, and sample_index. The wrapper should write a JSON
    structure file to output_json, or a JSON list whose first element is used.
    """

    name: str
    structure_family: str
    representation: str
    command: str
    checkpoint_path: str = ""
    num_samples: int = 1
    timeout_seconds: int = 3600
    target_keys: tuple[str, ...] = ()

    def available(self) -> bool:
        return bool(self.command) and (not self.checkpoint_path or Path(self.checkpoint_path).exists())

    def sample(self, target_property: dict[str, float], output_dir: Path, sample_index: int = 1) -> dict[str, Any] | None:
        output_dir.mkdir(parents=True, exist_ok=True)
        target_json = output_dir / "target_property.json"
        target_csv = output_dir / "target_property.csv"
        output_json = output_dir / "generated_structure.json"
        target_json.write_text(json.dumps(target_property, ensure_ascii=False, indent=2), encoding="utf-8")
        target_csv.write_text(",".join(str(value) for value in _target_vector(target_property, self.target_keys)), encoding="utf-8")

        command = self.command.format(
            checkpoint_path=shlex.quote(str(self.checkpoint_path)),
            output_dir=shlex.quote(str(output_dir)),
            output_json=shlex.quote(str(output_json)),
            target_json=shlex.quote(str(target_json)),
            target_csv=shlex.quote(str(target_csv)),
            num_samples=int(self.num_samples),
            sample_index=int(sample_index),
        )
        subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            shell=True,
            check=True,
            timeout=max(1, int(self.timeout_seconds)),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        )
        if not output_json.exists():
            return None
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            payload = payload[0] if payload else None
        if not isinstance(payload, dict) or not payload:
            return None
        return _annotate_structure(payload, self, sample_index)


@dataclass
class DiffuMetaEquationBackend:
    """Adapter for third-party/DiffusionMetamaterials/sample.py."""

    name: str
    structure_family: str
    checkpoint_path: str
    project_dir: Path = THIRD_PARTY_ROOT / "DiffusionMetamaterials"
    representation: str = "implicit_equation"
    num_samples: int = 32
    cfg_scale: float = 4.0
    timeout_seconds: int = 3600
    target_keys: tuple[str, ...] = ()

    def available(self) -> bool:
        return bool(self.checkpoint_path) and Path(self.checkpoint_path).exists() and (self.project_dir / "sample.py").exists()

    def sample(self, target_property: dict[str, float], output_dir: Path, sample_index: int = 1) -> dict[str, Any] | None:
        output_dir.mkdir(parents=True, exist_ok=True)
        inv_target_dir = self.project_dir / "data" / "inv_design_target"
        inv_target_dir.mkdir(parents=True, exist_ok=True)
        inv_target_path = inv_target_dir / "inv_target.csv"
        inv_target_path.write_text(
            ",".join(str(value) for value in _target_vector(target_property, self.target_keys)),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "sample.py",
            "--cfg_scale",
            str(float(self.cfg_scale)),
            "--num_samples",
            str(int(self.num_samples)),
            "--model_checkpoint",
            str(Path(self.checkpoint_path).resolve()),
        ]
        subprocess.run(
            command,
            cwd=str(self.project_dir),
            check=True,
            timeout=max(1, int(self.timeout_seconds)),
            env={**os.environ, "PYTHONPATH": str(self.project_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        )
        result_dir = self.project_dir / "generation_results"
        equations = _csv_first_column(result_dir / "valid_sample_equations.csv")
        if not equations:
            equations = _csv_first_column(result_dir / "valid_samples.csv")
        if not equations:
            return None
        run_result_dir = output_dir / "generation_results"
        if run_result_dir.exists():
            shutil.rmtree(run_result_dir)
        if result_dir.exists():
            shutil.copytree(result_dir, run_result_dir)
        equation = equations[0]
        structure = {
            "structure_id": f"{self.name}_{_safe_name(self.structure_family)}_{sample_index:03d}",
            "structure_family": self.structure_family,
            "representation": self.representation,
            "implicit_equation": equation,
            "control_points_stress": _target_vector(target_property, self.target_keys),
            "target_property": dict(target_property),
            "artifacts": {
                "output_dir": str(run_result_dir.resolve()),
                "valid_equations_csv": str((run_result_dir / "valid_sample_equations.csv").resolve()),
                "invalid_equations_csv": str((run_result_dir / "invalid_sample_equations.csv").resolve()),
                "valid_target_csv": str((run_result_dir / "valid_target_c.csv").resolve()),
                "sample_output": str((run_result_dir / "sample_output.txt").resolve()),
            },
        }
        return _annotate_structure(structure, self, sample_index)


@dataclass
class GraphMetaMatTrussBackend:
    """Adapter for third-party/GraphMetaMat/run_inverse_designer.py."""

    name: str = "graphmetamat_truss"
    structure_family: str = "truss"
    project_dir: Path = THIRD_PARTY_ROOT / "GraphMetaMat"
    representation: str = "graph_truss"
    checkpoint_path: str = ""
    num_runs: int = 16
    top_k: int = 1
    device: str = "cuda"
    timeout_seconds: int = 3600

    def available(self) -> bool:
        if not (self.project_dir / "run_inverse_designer.py").exists():
            return False
        if self.checkpoint_path and not Path(self.checkpoint_path).exists():
            return False
        return True

    def sample(self, target_property: dict[str, float], output_dir: Path, sample_index: int = 1) -> dict[str, Any] | None:
        output_dir.mkdir(parents=True, exist_ok=True)
        target_path = self._target_path(target_property, output_dir)
        command = [
            sys.executable,
            "run_inverse_designer.py",
            "--target",
            str(target_path),
            "--out-dir",
            str(output_dir.resolve()),
            "--num-runs",
            str(int(self.num_runs)),
            "--top-k",
            str(int(self.top_k)),
            "--device",
            str(self.device),
        ]
        subprocess.run(
            command,
            cwd=str(self.project_dir),
            check=True,
            timeout=max(1, int(self.timeout_seconds)),
            env={**os.environ, "PYTHONPATH": str(self.project_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        )
        export_dir = output_dir / "design_exports"
        top_json = export_dir / "top_designs.json"
        if not top_json.exists():
            return None
        designs = json.loads(top_json.read_text(encoding="utf-8"))
        if not designs:
            return None
        design = dict(designs[0])
        gpkl_path = _resolve_output_path(self.project_dir, str(design.get("gpkl", "")))
        graph_payload = self._graph_payload(Path(gpkl_path)) if gpkl_path else {}
        structure_id = f"{self.name}_{_safe_name(self.structure_family)}_{sample_index:03d}"
        structure = {
            "structure_id": structure_id,
            "structure_family": self.structure_family,
            "representation": self.representation,
            "target_property": dict(target_property),
            "target_curve_path": str(target_path.resolve()),
            "coordinates": graph_payload.get("coordinates", []),
            "edges": graph_payload.get("edges", []),
            "edge_radii": graph_payload.get("edge_radii", []),
            "rho": design.get("rho", graph_payload.get("rho")),
            "predicted_property": {
                "mae": design.get("mae"),
                "mse": design.get("mse"),
                "jaccard": design.get("jaccard"),
                "rho": design.get("rho"),
                "num_nodes": design.get("num_nodes"),
                "num_edges": design.get("num_edges"),
            },
            "artifacts": {
                "output_dir": str(output_dir.resolve()),
                "results_pkl": str((output_dir / "results.pkl").resolve()),
                "summary_csv": str((export_dir / "summary.csv").resolve()),
                "top_designs_json": str(top_json.resolve()),
                "gpkl": gpkl_path,
                "vtk": _resolve_output_path(self.project_dir, str(design.get("vtk", ""))),
                "graph_png": _resolve_output_path(self.project_dir, str(design.get("graph_png", ""))),
                "curve_png": _resolve_output_path(self.project_dir, str(design.get("curve_png", ""))),
            },
        }
        return _annotate_structure(structure, self, sample_index)

    def _target_path(self, target_property: dict[str, Any], output_dir: Path) -> Path:
        raw_path = target_property.get("target_curve_path") or target_property.get("curve_path")
        if raw_path:
            return Path(str(raw_path)).expanduser().resolve()
        curve = _target_vector(target_property, ())
        if not curve:
            raise ValueError("GraphMetaMatTrussBackend requires stress_curve/target_curve/curve or target_curve_path")
        path = output_dir / "target_curve.csv"
        _write_curve_csv(curve, path)
        return path

    @staticmethod
    def _graph_payload(gpkl_path: Path) -> dict[str, Any]:
        if not gpkl_path.exists():
            return {}
        with gpkl_path.open("rb") as handle:
            graph = pickle.load(handle)
        nodes = sorted(graph.nodes())
        node_to_index = {node: index for index, node in enumerate(nodes)}
        coordinates = []
        for node in nodes:
            coord = graph.nodes[node].get("coord", [0.0, 0.0, 0.0])
            values = [float(value) for value in list(coord)[:3]]
            while len(values) < 3:
                values.append(0.0)
            coordinates.append(values)
        edges = []
        edge_radii = []
        for u, v, data in graph.edges(data=True):
            edges.append([node_to_index[u], node_to_index[v]])
            edge_radii.append(float(data.get("radius", 0.0)))
        return {
            "coordinates": coordinates,
            "edges": edges,
            "edge_radii": edge_radii,
            "rho": float(graph.graph.get("rho", 0.0)),
        }


@dataclass
class VoxelDiffusionBackend:
    """Adapter for third-party/microstructure_generation_3d checkpoints."""

    name: str
    structure_family: str
    checkpoint_path: str
    project_dir: Path = THIRD_PARTY_ROOT / "microstructure_generation_3d"
    representation: str = "density_voxel"
    num_samples: int = 1
    steps: int = 50
    tensor_w: float = 1.0
    timeout_seconds: int = 3600
    target_keys: tuple[str, ...] = ()

    def available(self) -> bool:
        return bool(self.checkpoint_path) and Path(self.checkpoint_path).exists() and (self.project_dir / "network" / "model_trainer.py").exists()

    def sample(self, target_property: dict[str, float], output_dir: Path, sample_index: int = 1) -> dict[str, Any] | None:
        output_dir.mkdir(parents=True, exist_ok=True)
        target_json = output_dir / "target_property.json"
        result_json = output_dir / "voxel_result.json"
        target_json.write_text(json.dumps(target_property, ensure_ascii=False, indent=2), encoding="utf-8")
        script = _voxel_inline_script()
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(Path(self.checkpoint_path).resolve()),
                str(target_json.resolve()),
                str(output_dir.resolve()),
                str(int(self.num_samples)),
                str(int(self.steps)),
                str(float(self.tensor_w)),
                ",".join(self.target_keys),
                str(result_json.resolve()),
            ],
            cwd=str(self.project_dir),
            check=True,
            timeout=max(1, int(self.timeout_seconds)),
            env={**os.environ, "PYTHONPATH": str(self.project_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        )
        if not result_json.exists():
            return None
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            payload = payload[0] if payload else None
        if not isinstance(payload, dict) or not payload:
            return None
        return _annotate_structure(payload, self, sample_index)


def _voxel_inline_script() -> str:
    return r'''
import json
import sys
from pathlib import Path

import numpy as np
import torch

from network.model_trainer import DiffusionModel


ckpt, target_path, output_dir, num_samples, steps, tensor_w, keys_csv, result_path = sys.argv[1:]
target = json.loads(Path(target_path).read_text(encoding="utf-8"))
keys = [item for item in keys_csv.split(",") if item]
if keys:
    tensor_c = np.array([float(target.get(key, 0.0)) for key in keys], dtype=np.float32)
else:
    tensor_c = np.array([float(target[key]) for key in sorted(target)], dtype=np.float32)
output = Path(output_dir)
output.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DiffusionModel.load_from_checkpoint(ckpt)
model = model.to(device)
generator = model.ema_model if hasattr(model, "ema_model") else model.model
res_tensor = generator.sample_with_tensor(
    tensor_c=tensor_c,
    batch_size=int(num_samples),
    steps=int(steps),
    truncated_index=0.0,
    tensor_w=float(tensor_w),
)
records = []
for index in range(int(num_samples)):
    voxel = res_tensor[index].squeeze().detach().cpu().numpy()
    voxel = (voxel > 0).astype(np.uint8)
    npy_path = output / f"voxel_{index:03d}.npy"
    np.save(npy_path, voxel)
    records.append({
        "structure_id": f"voxel_diffusion_{index:03d}",
        "structure_family": "voxel",
        "representation": "density_voxel",
        "voxel_path": str(npy_path),
        "voxel_shape": list(voxel.shape),
        "density_proxy": float(voxel.mean()),
        "target_property": target,
        "artifacts": {"voxel_npy": str(npy_path)},
    })
Path(result_path).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
'''


def _annotate_structure(structure: dict[str, Any], backend: NeuralInverseBackend, sample_index: int) -> dict[str, Any]:
    payload = dict(structure)
    payload.setdefault("structure_id", f"{backend.name}_{sample_index:03d}")
    payload.setdefault("structure_family", backend.structure_family)
    payload.setdefault("representation", backend.representation)
    payload.setdefault("source", f"inverse_designer_neural:{backend.name}")
    payload.setdefault("neural_backend", backend.name)
    payload.setdefault("sample_index", int(sample_index))
    return payload


def build_env_backends(workspace_root: str | Path | None = None) -> list[NeuralInverseBackend]:
    del workspace_root
    backends: list[NeuralInverseBackend] = []

    tpms_ckpt = _first_env("INVERSE_TPMS_CKPT", "TPMS_INVERSE_CKPT")
    if tpms_ckpt:
        backends.append(
            DiffuMetaEquationBackend(
                name="tpms_diffumeta",
                structure_family="tpms",
                checkpoint_path=tpms_ckpt,
                project_dir=Path(_first_env("INVERSE_TPMS_PROJECT_DIR")) if _first_env("INVERSE_TPMS_PROJECT_DIR") else THIRD_PARTY_ROOT / "DiffusionMetamaterials",
                num_samples=int(os.getenv("INVERSE_TPMS_NUM_SAMPLES", "32")),
                cfg_scale=float(os.getenv("INVERSE_TPMS_CFG_SCALE", "4.0")),
                target_keys=tuple(item for item in os.getenv("INVERSE_TPMS_TARGET_KEYS", "").split(",") if item),
            )
        )

    truss_callable = _first_env("INVERSE_TRUSS_CALLABLE")
    truss_command = _first_env("INVERSE_TRUSS_COMMAND")
    truss_ckpt = _first_env("INVERSE_TRUSS_CKPT", "TRUSS_INVERSE_CKPT")
    truss_backend = _first_env("INVERSE_TRUSS_BACKEND", "TRUSS_INVERSE_BACKEND").strip().lower()
    if truss_backend in {"graphmetamat", "graph_meta_mat"} or _env_flag("INVERSE_GRAPHMETAMAT_ENABLE", False):
        backends.append(
            GraphMetaMatTrussBackend(
                name="graphmetamat_truss",
                project_dir=Path(_first_env("INVERSE_GRAPHMETAMAT_PROJECT_DIR", "INVERSE_TRUSS_PROJECT_DIR")) if _first_env("INVERSE_GRAPHMETAMAT_PROJECT_DIR", "INVERSE_TRUSS_PROJECT_DIR") else THIRD_PARTY_ROOT / "GraphMetaMat",
                checkpoint_path=truss_ckpt,
                num_runs=int(os.getenv("INVERSE_GRAPHMETAMAT_NUM_RUNS", os.getenv("INVERSE_TRUSS_NUM_RUNS", "16"))),
                top_k=int(os.getenv("INVERSE_GRAPHMETAMAT_TOP_K", os.getenv("INVERSE_TRUSS_TOP_K", "1"))),
                device=os.getenv("INVERSE_GRAPHMETAMAT_DEVICE", os.getenv("INVERSE_TRUSS_DEVICE", "cuda")),
                timeout_seconds=int(os.getenv("INVERSE_GRAPHMETAMAT_TIMEOUT_SECONDS", "3600")),
            )
        )
    elif truss_callable:
        backends.append(
            CallableBackend(
                name="truss_callable",
                structure_family="truss",
                representation="graph_truss",
                callable_path=truss_callable,
                checkpoint_path=truss_ckpt,
                num_samples=int(os.getenv("INVERSE_TRUSS_NUM_SAMPLES", "1")),
            )
        )
    elif truss_command:
        backends.append(
            CommandBackend(
                name="truss_command",
                structure_family="truss",
                representation="graph_truss",
                command=truss_command,
                checkpoint_path=truss_ckpt,
                target_keys=tuple(item for item in os.getenv("INVERSE_TRUSS_TARGET_KEYS", "").split(",") if item),
            )
        )

    bspline_callable = _first_env("INVERSE_BSPLINE_CALLABLE", "INVERSE_B_SPLINE_CALLABLE")
    bspline_command = _first_env("INVERSE_BSPLINE_COMMAND", "INVERSE_B_SPLINE_COMMAND")
    bspline_ckpt = _first_env("INVERSE_BSPLINE_CKPT", "INVERSE_B_SPLINE_CKPT", "BSPLINE_INVERSE_CKPT")
    if bspline_callable:
        backends.append(
            CallableBackend(
                name="bspline_callable",
                structure_family="b_spline",
                representation="b_spline",
                callable_path=bspline_callable,
                checkpoint_path=bspline_ckpt,
                num_samples=int(os.getenv("INVERSE_BSPLINE_NUM_SAMPLES", "1")),
            )
        )
    elif bspline_command:
        backends.append(
            CommandBackend(
                name="bspline_command",
                structure_family="b_spline",
                representation="b_spline",
                command=bspline_command,
                checkpoint_path=bspline_ckpt,
                target_keys=tuple(item for item in os.getenv("INVERSE_BSPLINE_TARGET_KEYS", "").split(",") if item),
            )
        )

    voxel_ckpt = _first_env("INVERSE_VOXEL_CKPT", "VOXEL_INVERSE_CKPT")
    if voxel_ckpt:
        backends.append(
            VoxelDiffusionBackend(
                name="voxel_diffusion_3d",
                structure_family="voxel",
                checkpoint_path=voxel_ckpt,
                project_dir=Path(_first_env("INVERSE_VOXEL_PROJECT_DIR")) if _first_env("INVERSE_VOXEL_PROJECT_DIR") else THIRD_PARTY_ROOT / "microstructure_generation_3d",
                num_samples=int(os.getenv("INVERSE_VOXEL_NUM_SAMPLES", "1")),
                steps=int(os.getenv("INVERSE_VOXEL_STEPS", "50")),
                tensor_w=float(os.getenv("INVERSE_VOXEL_TENSOR_W", "1.0")),
                target_keys=tuple(item for item in os.getenv("INVERSE_VOXEL_TARGET_KEYS", "").split(",") if item),
            )
        )

    return backends


def neural_enabled_from_env(default: bool = False) -> bool:
    return _env_flag("INVERSE_DESIGNER_ENABLE_NEURAL", default)
