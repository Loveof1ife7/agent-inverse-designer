from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPHMETAMAT_ROOT = ROOT / "third-party" / "GraphMetaMat"
EXPECTED_CURVE_LEN = 256


@dataclass(frozen=True)
class AcceptedSample:
    record: dict[str, Any]
    gid: int
    cid: int
    split: str
    candidate_id: str
    graph_path: Path
    curve_path: Path
    polyhedron_path: Path | None


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            records.append(record)
    return records


def _resolve_existing_file(path_value: Any, roots: list[Path], label: str) -> Path:
    path = _resolve_path(path_value, roots)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path_value}")
    return path


def _status(record: dict[str, Any], key: str) -> str:
    return str((record.get("evaluation") or {}).get(key) or "").strip().lower()


def _stress(record: dict[str, Any]) -> np.ndarray | None:
    response = record.get("response") or {}
    stress = response.get("stress")
    if stress is None and response.get("curve") is not None:
        curve = np.asarray(response["curve"], dtype=float)
        if curve.ndim == 2 and curve.shape[1] >= 2:
            stress = curve[:, 1]
    if stress is None:
        return None
    stress_array = np.asarray(stress, dtype=float).reshape(-1)
    if len(stress_array) != EXPECTED_CURVE_LEN:
        return None
    if not np.all(np.isfinite(stress_array)):
        return None
    return stress_array


def _strain_grid(record: dict[str, Any]) -> np.ndarray:
    response = record.get("response") or {}
    strain = response.get("strain_grid")
    if strain is None and response.get("curve") is not None:
        curve = np.asarray(response["curve"], dtype=float)
        if curve.ndim == 2 and curve.shape[1] >= 2:
            strain = curve[:, 0]
    if strain is None:
        return np.linspace(0.0, 0.3, EXPECTED_CURVE_LEN)
    strain_array = np.asarray(strain, dtype=float).reshape(-1)
    if len(strain_array) != EXPECTED_CURVE_LEN or not np.all(np.isfinite(strain_array)):
        return np.linspace(0.0, 0.3, EXPECTED_CURVE_LEN)
    return strain_array


def _rejection_reason(record: dict[str, Any]) -> str | None:
    if str(record.get("structure_family") or "").strip().lower() != "truss":
        return "structure_family is not truss"
    if _status(record, "eval_status") != "success":
        return "evaluation.eval_status is not success"
    if _status(record, "geometry_status") != "valid":
        return "evaluation.geometry_status is not valid"
    if _status(record, "fem_status") != "success":
        return "evaluation.fem_status is not success"
    if _stress(record) is None:
        return f"response.stress is missing, non-finite, or not length {EXPECTED_CURVE_LEN}"
    structure = record.get("structure") or {}
    has_graph_path = bool(structure.get("gpkl_path") or structure.get("graph_path"))
    has_inline_graph = structure.get("coordinates") is not None and structure.get("edges") is not None
    if not has_graph_path and not has_inline_graph:
        return "structure has neither gpkl_path nor coordinates/edges"
    return None


def _resolve_path(path_value: Any, search_roots: list[Path]) -> Path | None:
    if path_value is None:
        return None
    raw = str(path_value).strip()
    if not raw:
        return None
    path_candidates = [Path(raw)]
    if "\\" in raw:
        path_candidates.append(Path(raw.replace("\\", "/")))
    for path in path_candidates:
        if path.is_absolute() and path.exists():
            return path
        if path.exists():
            return path.resolve()
    for root in search_roots:
        for path in path_candidates:
            candidate = root / path
            if candidate.exists():
                return candidate.resolve()
    names = {Path(raw).name, PureWindowsPath(raw).name}
    for name in sorted(name for name in names if name):
        for root in search_roots:
            for subdir in ("candidates", "graphs", "design_exports"):
                candidate = root / subdir / name
                if candidate.exists():
                    return candidate.resolve()
    return None


def _prepare_graphmetamat_imports(project_dir: Path) -> None:
    project_str = str(project_dir.resolve())
    if project_str in sys.path:
        sys.path.remove(project_str)
    sys.path.insert(0, project_str)


def _relabel_graph(graph: nx.Graph) -> nx.Graph:
    mapping = {node: index for index, node in enumerate(sorted(graph.nodes()))}
    if all(node == index for node, index in mapping.items()):
        return graph.copy()
    return nx.relabel_nodes(graph, mapping, copy=True)


def _graph_from_inline_structure(record: dict[str, Any]) -> nx.Graph:
    structure = record.get("structure") or {}
    coords = np.asarray(structure.get("coordinates"), dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("structure.coordinates must have shape [N, 3]")
    edges = structure.get("edges") or []
    radii = structure.get("edge_radii") or []
    graph = nx.Graph()
    for nid, coord in enumerate(coords):
        graph.add_node(int(nid), coord=np.asarray(coord, dtype=float))
    for edge_index, edge in enumerate(edges):
        if len(edge) != 2:
            raise ValueError(f"structure.edges[{edge_index}] must have two node ids")
        u, v = int(edge[0]), int(edge[1])
        radius = float(radii[edge_index]) if edge_index < len(radii) else 0.0
        graph.add_edge(u, v, radius=radius)
    rho = structure.get("rho") or (record.get("response") or {}).get("relative_density") or 0.0
    graph.graph["rho"] = float(rho)
    return graph


def _load_or_build_graph(record: dict[str, Any], search_roots: list[Path]) -> tuple[nx.Graph, Path | None]:
    structure = record.get("structure") or {}
    graph_path = _resolve_path(structure.get("gpkl_path") or structure.get("graph_path"), search_roots)
    if graph_path is not None:
        with graph_path.open("rb") as handle:
            graph = pickle.load(handle)
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"{graph_path} did not contain a networkx.Graph")
        return graph, graph_path
    return _graph_from_inline_structure(record), None


def _add_graph_features(graph: nx.Graph, gid: int, project_dir: Path) -> nx.Graph:
    _prepare_graphmetamat_imports(project_dir)
    from src.dataset_feats_edge import get_edge_feats, get_edge_index, get_edge_li
    from src.dataset_feats_node import get_node_feats

    graph = _relabel_graph(graph)
    for nid in graph.nodes():
        coord = np.asarray(graph.nodes[nid].get("coord"), dtype=float).reshape(-1)
        if coord.shape != (3,):
            raise ValueError(f"node {nid} is missing a 3D coord")
        graph.nodes[nid]["coord"] = coord
    for u, v in graph.edges():
        radius = graph.edges[u, v].get("radius", 0.0)
        graph.edges[u, v]["radius"] = float(radius)
    graph.graph["gid"] = int(gid)
    graph.graph["rho"] = float(graph.graph.get("rho", 0.0))
    edge_li = get_edge_li(graph)
    edge_index = get_edge_index(edge_li)
    graph.graph["edge_index"] = edge_index
    graph.graph["node_feats"] = get_node_feats(graph, edge_index)
    graph.graph["edge_feats"] = get_edge_feats(graph, edge_li)
    return graph


def _write_curve(record: dict[str, Any], cid: int, output_path: Path) -> None:
    stress = _stress(record)
    if stress is None:
        raise ValueError("record does not contain a valid 256-point stress response")
    strain = _strain_grid(record)
    payload = {
        "curve": np.stack([strain, stress], axis=-1),
        "cid": int(cid),
        "is_monotonic": bool((record.get("response") or {}).get("is_monotonic", True)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)


def _write_graph(
    record: dict[str, Any],
    gid: int,
    output_path: Path,
    project_dir: Path,
    search_roots: list[Path],
) -> tuple[Path | None, dict[str, Any]]:
    graph, source_path = _load_or_build_graph(record, search_roots)
    graph = _add_graph_features(graph, gid, project_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(graph, handle)
    return source_path, {
        "num_nodes": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
        "rho": float(graph.graph.get("rho", 0.0)),
    }


def _copy_polyhedron(record: dict[str, Any], output_graph_path: Path, search_roots: list[Path]) -> Path | None:
    structure = record.get("structure") or {}
    artifacts = ((record.get("inverse_designer") or {}).get("artifacts") or {})
    path_value = (
        structure.get("polyhedron_gpkl_path")
        or structure.get("polyhedron_path")
        or artifacts.get("polyhedron_gpkl")
        or artifacts.get("polyhedron_path")
    )
    source = _resolve_path(path_value, search_roots)
    if source is None:
        return None
    destination = output_graph_path.with_name(f"{output_graph_path.stem}_polyhedron.gpkl")
    shutil.copy2(source, destination)
    return destination


def _empty_dataset_dirs(dataset_output_dir: Path) -> None:
    if dataset_output_dir.exists():
        shutil.rmtree(dataset_output_dir)
    for split in ("train", "dev", "test"):
        (dataset_output_dir / split / "graphs").mkdir(parents=True, exist_ok=True)
        (dataset_output_dir / split / "curves").mkdir(parents=True, exist_ok=True)
        (dataset_output_dir / split / "mapping.tsv").write_text("", encoding="utf-8")


def _assign_split(index: int, total: int, dev_fraction: float) -> str:
    if total < 5 or dev_fraction <= 0:
        return "train"
    dev_count = max(1, int(round(total * dev_fraction)))
    return "dev" if index >= total - dev_count else "train"


def build_dataset(request: dict[str, Any]) -> dict[str, Any]:
    round_id = str(request.get("round_id") or request.get("job_id") or "round")
    workspace_root = Path(request.get("workspace_root") or ROOT / "workspace" / "truss_active_learning" / round_id)
    windows_eval_dir = Path(request.get("windows_eval_dir") or workspace_root / "windows_eval")
    dataset_output_dir = Path(
        request.get("dataset_output_dir") or workspace_root / f"dataset_active_{round_id}"
    )
    graphmetamat_project_dir = Path(request.get("graphmetamat_project_dir") or DEFAULT_GRAPHMETAMAT_ROOT)
    evaluated_samples_path = Path(
        request.get("evaluated_samples_path") or windows_eval_dir / "evaluated_samples.jsonl"
    )
    if not evaluated_samples_path.is_absolute():
        evaluated_samples_path = ROOT / evaluated_samples_path
    if not windows_eval_dir.is_absolute():
        windows_eval_dir = ROOT / windows_eval_dir
    if not workspace_root.is_absolute():
        workspace_root = ROOT / workspace_root
    if not dataset_output_dir.is_absolute():
        dataset_output_dir = ROOT / dataset_output_dir
    if not graphmetamat_project_dir.is_absolute():
        graphmetamat_project_dir = ROOT / graphmetamat_project_dir
    if not evaluated_samples_path.exists():
        raise FileNotFoundError(f"evaluated_samples.jsonl not found: {evaluated_samples_path}")
    if not graphmetamat_project_dir.exists():
        raise FileNotFoundError(f"GraphMetaMat project dir not found: {graphmetamat_project_dir}")

    records = _load_jsonl(evaluated_samples_path)
    accepted_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        reason = _rejection_reason(record)
        if reason is None:
            accepted_records.append(record)
        else:
            rejected.append(
                {
                    "line_index": index,
                    "candidate_id": record.get("candidate_id"),
                    "reason": reason,
                }
            )

    dev_fraction = float(request.get("dev_fraction", 0.1))
    start_gid = int(request.get("start_gid", 0))
    start_cid = int(request.get("start_cid", start_gid))
    search_roots = [
        windows_eval_dir,
        workspace_root,
        ROOT,
        dataset_output_dir,
    ]
    _empty_dataset_dirs(dataset_output_dir)

    mapping_rows: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    accepted: list[AcceptedSample] = []
    accepted_manifest: list[dict[str, Any]] = []
    polyhedron_count = 0
    for accepted_index, record in enumerate(accepted_records):
        gid = start_gid + accepted_index
        cid = start_cid + accepted_index
        split = _assign_split(accepted_index, len(accepted_records), dev_fraction)
        candidate_id = str(record.get("candidate_id") or f"{round_id}_sample_{accepted_index:06d}")
        graph_path = dataset_output_dir / split / "graphs" / f"{gid}.gpkl"
        curve_path = dataset_output_dir / split / "curves" / f"{cid}.pkl"
        source_graph_path, graph_stats = _write_graph(
            record,
            gid=gid,
            output_path=graph_path,
            project_dir=graphmetamat_project_dir,
            search_roots=search_roots,
        )
        _write_curve(record, cid=cid, output_path=curve_path)
        polyhedron_path = _copy_polyhedron(record, graph_path, search_roots)
        if polyhedron_path is not None:
            polyhedron_count += 1
        mapping_rows[split].append(f"{gid}\t{cid}\n")
        accepted.append(
            AcceptedSample(
                record=record,
                gid=gid,
                cid=cid,
                split=split,
                candidate_id=candidate_id,
                graph_path=graph_path,
                curve_path=curve_path,
                polyhedron_path=polyhedron_path,
            )
        )
        accepted_manifest.append(
            {
                "candidate_id": candidate_id,
                "target_id": record.get("target_id"),
                "split": split,
                "gid": gid,
                "cid": cid,
                "graph_path": str(graph_path),
                "curve_path": str(curve_path),
                "source_graph_path": str(source_graph_path) if source_graph_path else None,
                "polyhedron_path": str(polyhedron_path) if polyhedron_path else None,
                **graph_stats,
            }
        )

    for split, rows in mapping_rows.items():
        (dataset_output_dir / split / "mapping.tsv").write_text("".join(rows), encoding="utf-8")

    manifest = {
        "round_id": round_id,
        "created_by": "tools/run_truss_finetune_job.py",
        "evaluated_samples_path": str(evaluated_samples_path),
        "dataset_output_dir": str(dataset_output_dir),
        "graphmetamat_project_dir": str(graphmetamat_project_dir),
        "curve_contract": {
            "task": "compression_stress_strain",
            "length": EXPECTED_CURVE_LEN,
            "default_strain_grid": "np.linspace(0.0, 0.3, 256)",
        },
        "counts": {
            "records_total": len(records),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "train": len(mapping_rows["train"]),
            "dev": len(mapping_rows["dev"]),
            "test": len(mapping_rows["test"]),
            "polyhedron": polyhedron_count,
        },
        "can_run_forward_finetune": len(accepted) > 0,
        "can_run_inverse_il": len(accepted) > 0 and polyhedron_count == len(accepted),
        "accepted": accepted_manifest,
        "rejected": rejected,
    }
    manifest_path = dataset_output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    recommended_next_commands = [
        f"cd {graphmetamat_project_dir}",
        "copy logs/stressstrain_forward/config_*.yaml to src/config_*.yaml and set dataset.root_* to "
        f"{dataset_output_dir}",
        "python main_forward.py",
        "after forward validation, update inverse configs and run python main_inverse.py",
    ]
    return {
        "status": "success",
        "round_id": round_id,
        "workspace_root": str(workspace_root),
        "windows_eval_dir": str(windows_eval_dir),
        "evaluated_samples_path": str(evaluated_samples_path),
        "dataset_output_dir": str(dataset_output_dir),
        "manifest_path": str(manifest_path),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "split_counts": {
            "train": len(mapping_rows["train"]),
            "dev": len(mapping_rows["dev"]),
            "test": len(mapping_rows["test"]),
        },
        "can_run_forward_finetune": len(accepted) > 0,
        "can_run_inverse_il": len(accepted) > 0 and polyhedron_count == len(accepted),
        "polyhedron_count": polyhedron_count,
        "rejected": rejected,
        "recommended_next_commands": recommended_next_commands,
    }


def _load_finetune_config(request: dict[str, Any], response: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    inline_config = request.get("finetune_config")
    config_path_value = request.get("finetune_config_path")
    if inline_config is not None and not isinstance(inline_config, dict):
        raise TypeError("request.finetune_config must be an object when provided")
    if inline_config is not None:
        return inline_config, None
    if not config_path_value:
        default_path = Path(response["workspace_root"]) / "finetune_config.json"
        if not default_path.exists():
            return None, None
        config_path_value = default_path
    roots = [
        Path(response["workspace_root"]),
        Path(response["windows_eval_dir"]),
        ROOT,
    ]
    config_path = _resolve_existing_file(config_path_value, roots, "finetune_config")
    return _read_json(config_path), config_path


def _validate_stage(stage: Any, index: int, response: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not isinstance(stage, dict):
        return None, [f"stages[{index}] must be an object"]
    name = str(stage.get("name") or f"stage_{index}").strip()
    enabled = _as_bool(stage.get("enabled"), True)
    kind = str(stage.get("kind") or name).strip().lower()
    command = stage.get("command")
    cwd = str(stage.get("cwd") or "third-party/GraphMetaMat")
    timeout_seconds = int(stage.get("timeout_seconds", 24 * 3600))
    requires_polyhedron = _as_bool(stage.get("requires_polyhedron"), kind in {"inverse_il", "inverse_policy_il"})
    on_missing_polyhedron = str(stage.get("on_missing_polyhedron") or "skip").strip().lower()
    if on_missing_polyhedron not in {"skip", "fail"}:
        errors.append(f"stages[{index}].on_missing_polyhedron must be skip or fail")
    if enabled and command is None:
        errors.append(f"stages[{index}].command is required when enabled=true")
    if command is not None and not isinstance(command, (str, list)):
        errors.append(f"stages[{index}].command must be a string or list")
    if isinstance(command, list) and not all(isinstance(part, (str, int, float)) for part in command):
        errors.append(f"stages[{index}].command list must contain scalar command parts")
    if timeout_seconds <= 0:
        errors.append(f"stages[{index}].timeout_seconds must be positive")
    if requires_polyhedron and not response.get("can_run_inverse_il"):
        message = f"stages[{index}] requires polyhedron labels, but dataset cannot run inverse IL"
        if on_missing_polyhedron == "fail" and enabled:
            errors.append(message)
        else:
            enabled = False
    if errors:
        return None, errors
    return {
        "name": name,
        "kind": kind,
        "enabled": enabled,
        "command": command,
        "cwd": cwd,
        "timeout_seconds": timeout_seconds,
        "requires_polyhedron": requires_polyhedron,
        "on_missing_polyhedron": on_missing_polyhedron,
    }, []


def _validate_finetune_config(
    config: dict[str, Any] | None,
    config_path: Path | None,
    request: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    if config is None:
        return {
            "status": "missing",
            "config_path": None,
            "errors": [],
            "warnings": ["No finetune_config was provided; dataset build can run, but training is not configured."],
            "stages": [],
            "enabled_stage_count": 0,
        }
    errors: list[str] = []
    warnings: list[str] = []
    schema_version = str(config.get("schema_version") or "").strip()
    if schema_version != "truss_finetune_v1":
        errors.append("schema_version must be truss_finetune_v1")
    structure_family = str(config.get("structure_family") or "").strip().lower()
    if structure_family != "truss":
        errors.append("structure_family must be truss")
    round_id = str(config.get("round_id") or "").strip()
    if round_id and round_id != response["round_id"]:
        errors.append(f"round_id mismatch: config={round_id}, request={response['round_id']}")
    elif not round_id:
        warnings.append("round_id is missing in finetune_config")

    curve = config.get("curve") or {}
    if not isinstance(curve, dict):
        errors.append("curve must be an object")
        curve = {}
    if int(curve.get("length", EXPECTED_CURVE_LEN)) != EXPECTED_CURVE_LEN:
        errors.append(f"curve.length must be {EXPECTED_CURVE_LEN}")
    if str(curve.get("task") or "compression_stress_strain") != "compression_stress_strain":
        errors.append("curve.task must be compression_stress_strain")
    if abs(float(curve.get("strain_min", 0.0)) - 0.0) > 1e-12:
        errors.append("curve.strain_min must be 0.0")
    if abs(float(curve.get("strain_max", 0.3)) - 0.3) > 1e-12:
        errors.append("curve.strain_max must be 0.3")
    if str(curve.get("label_source") or "windows_fem") != "windows_fem":
        errors.append("curve.label_source must be windows_fem")

    data = config.get("data") or {}
    if not isinstance(data, dict):
        errors.append("data must be an object")
        data = {}
    min_accepted = int(data.get("min_accepted", 1))
    if min_accepted < 1:
        errors.append("data.min_accepted must be >= 1")
    if int(response.get("accepted_count", 0)) < min_accepted:
        errors.append(
            f"accepted_count {response.get('accepted_count', 0)} is less than data.min_accepted {min_accepted}"
        )
    require_no_rejected = _as_bool(data.get("require_no_rejected"), False)
    if require_no_rejected and int(response.get("rejected_count", 0)) > 0:
        errors.append("data.require_no_rejected=true, but rejected records exist")
    require_polyhedron_for_inverse_il = _as_bool(data.get("require_polyhedron_for_inverse_il"), False)
    if require_polyhedron_for_inverse_il and not response.get("can_run_inverse_il"):
        errors.append("data.require_polyhedron_for_inverse_il=true, but not every accepted sample has polyhedron")

    stages_raw = config.get("stages") or []
    if not isinstance(stages_raw, list):
        errors.append("stages must be a list")
        stages_raw = []
    stages: list[dict[str, Any]] = []
    for index, stage in enumerate(stages_raw):
        normalized_stage, stage_errors = _validate_stage(stage, index, response)
        errors.extend(stage_errors)
        if normalized_stage is not None:
            stages.append(normalized_stage)

    if not stages:
        warnings.append("finetune_config.stages is empty; training will be a config check only.")
    enabled_stage_count = sum(1 for stage in stages if stage["enabled"])
    return {
        "status": "valid" if not errors else "invalid",
        "config_path": str(config_path) if config_path else None,
        "schema_version": schema_version,
        "round_id": round_id,
        "errors": errors,
        "warnings": warnings,
        "stages": stages,
        "enabled_stage_count": enabled_stage_count,
    }


def _stage_workdir(stage: dict[str, Any], request: dict[str, Any]) -> Path:
    cwd = Path(stage.get("cwd") or request.get("graphmetamat_project_dir") or DEFAULT_GRAPHMETAMAT_ROOT)
    if not cwd.is_absolute():
        cwd = ROOT / cwd
    return cwd


def _configured_training_stages(request: dict[str, Any], config_check: dict[str, Any]) -> list[dict[str, Any]]:
    stages = [stage for stage in config_check.get("stages", []) if stage.get("enabled")]
    if stages:
        return stages
    commands = request.get("training_commands") or []
    if not commands:
        return []
    cwd = str(request.get("training_cwd") or request.get("graphmetamat_project_dir") or DEFAULT_GRAPHMETAMAT_ROOT)
    timeout_seconds = int(request.get("training_timeout_seconds", 24 * 3600))
    return [
        {
            "name": f"legacy_command_{index}",
            "kind": "legacy",
            "enabled": True,
            "command": command,
            "cwd": cwd,
            "timeout_seconds": timeout_seconds,
            "requires_polyhedron": False,
            "on_missing_polyhedron": "skip",
        }
        for index, command in enumerate(commands)
    ]


def _run_training_commands(request: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    run_training = _as_bool(request.get("run_training"), False)
    config_check = response.get("config_check") or {}
    stages = _configured_training_stages(request, config_check)
    if not run_training:
        return {
            "status": "skipped",
            "reason": "run_training is false",
            "configured_stage_count": len(stages),
        }
    if config_check.get("status") == "invalid":
        return {
            "status": "blocked_by_invalid_config",
            "reason": "finetune_config validation failed",
            "errors": config_check.get("errors", []),
        }
    if not stages:
        return {
            "status": "not_configured",
            "reason": "run_training is true, but no enabled finetune_config.stages or training_commands were provided",
            "recommended_next_commands": response.get("recommended_next_commands", []),
        }
    results = []
    for stage in stages:
        command = stage["command"]
        cwd = _stage_workdir(stage, request)
        timeout_seconds = int(stage.get("timeout_seconds", 24 * 3600))
        if isinstance(command, str):
            cmd = command
            shell = True
        else:
            cmd = [str(part) for part in command]
            shell = False
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            shell=shell,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        results.append(
            {
                "stage": stage["name"],
                "kind": stage["kind"],
                "command": command,
                "cwd": str(cwd),
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        )
        if completed.returncode != 0:
            return {"status": "failed", "results": results}
    return {"status": "success", "results": results}


def run_job(request: dict[str, Any]) -> dict[str, Any]:
    response = build_dataset(request)
    config, config_path = _load_finetune_config(request, response)
    response["config_check"] = _validate_finetune_config(config, config_path, request, response)
    training = _run_training_commands(request, response)
    response["training"] = training
    if training["status"] == "failed":
        response["status"] = "training_failed"
    elif training["status"] == "blocked_by_invalid_config":
        response["status"] = "config_invalid"
    elif training["status"] == "not_configured":
        response["status"] = "dataset_built_training_not_configured"
    return response


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a GraphMetaMat truss active-learning dataset from Windows FEM labels."
    )
    parser.add_argument("--request", required=True, help="Path to finetune request JSON.")
    parser.add_argument("--output", required=True, help="Path to write response JSON.")
    args = parser.parse_args()

    request = _read_json(args.request)
    try:
        response = run_job(request)
    except Exception as exc:
        response = {
            "status": "failed",
            "round_id": str(request.get("round_id") or request.get("job_id") or "round"),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(args.output, response)
        raise
    _write_json(args.output, response)
    print(json.dumps(response, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
