from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROPERTY_STRAIN_GRID = [round(index * 0.01, 2) for index in range(31)]


@dataclass(frozen=True)
class ParsedTruss:
    node_ids: list[int]
    coordinates: list[list[float]]
    edges: list[list[int]]


@dataclass(frozen=True)
class ParsedProperty:
    y: list[float]
    metadata: dict[str, Any]


def _extract_bracket_list(text: str, var_name: str) -> str:
    idx = text.find(var_name)
    if idx < 0:
        raise ValueError(f"missing variable: {var_name}")
    eq = text.find("=", idx)
    if eq < 0:
        raise ValueError(f"missing '=' for variable: {var_name}")
    start = text.find("[", eq)
    if start < 0:
        raise ValueError(f"missing '[' for variable: {var_name}")

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError(f"unterminated bracket list for variable: {var_name}")


def parse_geometry_txt(path: str | Path) -> ParsedTruss:
    text = Path(path).read_text(encoding="utf-8")
    node_rows = ast.literal_eval(_extract_bracket_list(text, "node_data"))
    edge_rows = ast.literal_eval(_extract_bracket_list(text, "element_conn"))

    node_ids: list[int] = []
    coordinates: list[list[float]] = []
    for row in node_rows:
        if len(row) < 4:
            raise ValueError(f"invalid node row in {path}: {row!r}")
        node_ids.append(int(row[0]))
        coordinates.append([float(row[1]), float(row[2]), float(row[3])])

    id_to_vertex = {node_id: index + 1 for index, node_id in enumerate(node_ids)}
    edge_set: set[tuple[int, int]] = set()
    for row in edge_rows:
        if len(row) < 2:
            continue
        left = id_to_vertex.get(int(row[0]))
        right = id_to_vertex.get(int(row[1]))
        if left is None or right is None or left == right:
            continue
        edge_set.add((min(left, right), max(left, right)))

    return ParsedTruss(
        node_ids=node_ids,
        coordinates=coordinates,
        edges=[list(edge) for edge in sorted(edge_set)],
    )


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _deduplicate_curve_points(points: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], int]:
    buckets: dict[float, list[float]] = {}
    for strain, stress in points:
        buckets.setdefault(strain, []).append(stress)
    deduplicated = [
        (strain, sum(stresses) / len(stresses))
        for strain, stresses in sorted(buckets.items())
    ]
    duplicate_count = sum(len(stresses) - 1 for stresses in buckets.values())
    return deduplicated, duplicate_count


def _interpolate_hold(points: list[tuple[float, float]], x_value: float) -> float:
    if not points:
        raise ValueError("cannot interpolate an empty property curve")
    if x_value <= points[0][0]:
        return points[0][1]
    if x_value >= points[-1][0]:
        return points[-1][1]
    for index in range(len(points) - 1):
        x0, y0 = points[index]
        x1, y1 = points[index + 1]
        if x0 <= x_value <= x1:
            if abs(x1 - x0) <= 1e-12:
                return y0
            ratio = (x_value - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return points[-1][1]


def _trapezoid_energy(strain_grid: list[float], stress_grid: list[float]) -> float:
    if len(strain_grid) != len(stress_grid) or len(strain_grid) < 2:
        return 0.0
    total = 0.0
    for index in range(len(strain_grid) - 1):
        width = strain_grid[index + 1] - strain_grid[index]
        total += 0.5 * width * (stress_grid[index] + stress_grid[index + 1])
    return total


def parse_property_csv(
    path: str | Path,
    strain_grid: list[float] | None = None,
) -> ParsedProperty:
    path = Path(path)
    strain_grid = list(strain_grid or PROPERTY_STRAIN_GRID)
    raw_points: list[tuple[float, float]] = []
    negative_stress_count = 0

    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            strain = _safe_float(row.get("Strain"))
            stress = _safe_float(row.get("Stress"))
            if strain is None or stress is None:
                continue
            if stress < 0.0:
                negative_stress_count += 1
                stress = 0.0
            raw_points.append((strain, stress))

    if not raw_points:
        raise ValueError(f"empty or invalid property curve: {path}")

    clean_points, duplicate_strain_count = _deduplicate_curve_points(raw_points)
    min_observed_strain = clean_points[0][0]
    max_observed_strain = clean_points[-1][0]
    stress_grid = [_interpolate_hold(clean_points, strain) for strain in strain_grid]
    extrapolated_point_count = sum(
        1 for strain in strain_grid
        if strain < min_observed_strain or strain > max_observed_strain
    )

    in_range_points = [
        (strain, stress)
        for strain, stress in clean_points
        if strain_grid[0] <= strain <= strain_grid[-1]
    ]
    if in_range_points:
        peak_strain, peak_stress = max(in_range_points, key=lambda item: item[1])
    else:
        peak_index = max(range(len(stress_grid)), key=lambda index: stress_grid[index])
        peak_strain = strain_grid[peak_index]
        peak_stress = stress_grid[peak_index]

    metadata = {
        "representation": "stress_grid_v1",
        "strain_grid": strain_grid,
        "stress_unit": "MPa",
        "strain_unit": "dimensionless",
        "interpolation": "linear",
        "duplicate_strain_policy": "mean_stress",
        "negative_stress_policy": "clamp_to_zero",
        "left_extrapolation": "hold_first",
        "right_extrapolation": "hold_last",
        "raw_point_count": len(raw_points),
        "clean_point_count": len(clean_points),
        "duplicate_strain_count": duplicate_strain_count,
        "negative_stress_count": negative_stress_count,
        "min_observed_strain": min_observed_strain,
        "max_observed_strain": max_observed_strain,
        "extrapolated_point_count": extrapolated_point_count,
        "summary": {
            "peak_stress": peak_stress,
            "peak_strain": peak_strain,
            "energy_0_30": _trapezoid_energy(strain_grid, stress_grid),
        },
    }
    return ParsedProperty(y=stress_grid, metadata=metadata)


def normalize_coordinates(coordinates: list[list[float]]) -> tuple[list[list[float]], dict[str, Any]]:
    if not coordinates:
        return [], {"type": "bbox_to_unit_cube", "origin": [0.0, 0.0, 0.0], "scale": 1.0}

    mins = [min(row[axis] for row in coordinates) for axis in range(3)]
    maxs = [max(row[axis] for row in coordinates) for axis in range(3)]
    scale = max(maxs[axis] - mins[axis] for axis in range(3))
    if scale <= 0.0:
        scale = 1.0

    normalized = [
        [round((row[axis] - mins[axis]) / scale, 8) for axis in range(3)]
        for row in coordinates
    ]
    return normalized, {
        "type": "bbox_to_unit_cube",
        "origin": [float(value) for value in mins],
        "scale": float(scale),
    }


def build_bfs_topology_supervision(n: int, edges: list[list[int]]) -> dict[str, Any]:
    adjacency: dict[int, list[int]] = {vertex: [] for vertex in range(1, n + 1)}
    edge_set = {tuple(edge) for edge in edges}
    for left, right in edge_set:
        adjacency[left].append(right)
        adjacency[right].append(left)
    for neighbors in adjacency.values():
        neighbors.sort()

    parent: dict[int, int] = {1: 0}
    node_order = [1]
    queue: deque[int] = deque([1])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor in parent:
                continue
            parent[neighbor] = current
            node_order.append(neighbor)
            queue.append(neighbor)

    if len(parent) != n:
        missing = [vertex for vertex in range(1, n + 1) if vertex not in parent]
        raise ValueError(f"graph is disconnected; missing vertices: {missing[:10]}")

    old_to_new = {old_vertex: new_vertex for new_vertex, old_vertex in enumerate(node_order, start=1)}
    parent_sequence = [old_to_new[parent[old_vertex]] for old_vertex in node_order[1:]]
    tree_edges = {
        tuple(sorted((old_to_new[old_vertex], old_to_new[parent[old_vertex]])))
        for old_vertex in node_order[1:]
    }
    remapped_edges = {
        tuple(sorted((old_to_new[left], old_to_new[right])))
        for left, right in edge_set
    }
    extra_edges = sorted(remapped_edges - tree_edges)
    return {
        "parent_sequence": parent_sequence,
        "node_order": node_order,
        "old_to_new": old_to_new,
        "edges": [list(edge) for edge in sorted(remapped_edges)],
        "tree_edges": [list(edge) for edge in sorted(tree_edges)],
        "extra_edges": [list(edge) for edge in extra_edges],
        "k": len(extra_edges),
    }


def quantize_coordinates(coordinates: list[list[float]], bits: int = 10) -> list[int]:
    max_token = (1 << bits) - 1
    tokens: list[int] = []
    for row in coordinates:
        for value in row:
            clipped = min(1.0, max(0.0, float(value)))
            tokens.append(int(round(clipped * max_token)))
    return tokens


def topology_prefix_tokens(n: int, parent_sequence: list[int], extra_edges: list[list[int]]) -> list[Any]:
    tokens: list[Any] = ["<N>", n, "<TREE>"]
    tokens.extend(parent_sequence)
    tokens.append("<EXTRA>")
    for left, right in extra_edges:
        tokens.extend([left, right])
    tokens.append("<COORD>")
    return tokens


def split_for_id(sample_index: int, train_ratio: float = 0.9, val_ratio: float = 0.05) -> str:
    bucket = sample_index % 100
    train_cutoff = int(round(train_ratio * 100))
    val_cutoff = train_cutoff + int(round(val_ratio * 100))
    if bucket < train_cutoff:
        return "train"
    if bucket < val_cutoff:
        return "val"
    return "test"


def build_inverse_truss_record(
    structure_path: str | Path,
    dataset_root: str | Path,
    quantization_bits: int = 10,
    include_property: bool = True,
) -> dict[str, Any]:
    structure_path = Path(structure_path)
    dataset_root = Path(dataset_root)
    sample_stem = structure_path.stem
    sample_index = int(sample_stem) if sample_stem.isdigit() else 0
    property_path = dataset_root / "properties" / f"{sample_stem}.csv"

    parsed = parse_geometry_txt(structure_path)
    topology = build_bfs_topology_supervision(len(parsed.coordinates), parsed.edges)
    reordered_coordinates = [
        parsed.coordinates[old_vertex - 1]
        for old_vertex in topology["node_order"]
    ]
    coordinates, normalization = normalize_coordinates(reordered_coordinates)
    coordinate_tokens = quantize_coordinates(coordinates, bits=quantization_bits)
    extra_edges = topology["extra_edges"]
    parsed_property = parse_property_csv(property_path) if include_property else None

    record: dict[str, Any] = {
        "sample_id": sample_stem,
        "split": split_for_id(sample_index),
        "version": "inverse_truss_property_grid_v1" if include_property else "inverse_truss_geometry_v1",
        "y": parsed_property.y if parsed_property is not None else [],
        "n": len(coordinates),
        "coordinates": coordinates,
        "edges": topology["edges"],
        "parent_sequence": topology["parent_sequence"],
        "k": topology["k"],
        "extra_edges": extra_edges,
        "topology_prefix_tokens": topology_prefix_tokens(len(coordinates), topology["parent_sequence"], extra_edges),
        "coordinate_tokens": coordinate_tokens,
        "preprocessing": {
            "coordinate_normalization": normalization,
            "coordinate_quantization_bits": quantization_bits,
            "topology_tree_method": "bfs_parent_pointer_reindexed_v2",
            "index_base": 1,
            "root_original_node_id": parsed.node_ids[topology["node_order"][0] - 1],
            "node_order_original_ids": [
                parsed.node_ids[old_vertex - 1]
                for old_vertex in topology["node_order"]
            ],
        },
        "provenance": {
            "source": "datagen_bootstrap_structure",
            "structure_path": str(structure_path),
            "property_path": str(property_path),
            "symmetry": "P222",
        },
    }
    if parsed_property is not None:
        record["property"] = parsed_property.metadata
    return record


def iter_structure_paths(dataset_root: str | Path, structure_dir_name: str | None = None) -> list[Path]:
    dataset_root = Path(dataset_root)
    if structure_dir_name is not None:
        structure_dir = dataset_root / structure_dir_name
    else:
        structure_dir = dataset_root / "structures"
        if not structure_dir.exists():
            structure_dir = dataset_root / "geometry"
    paths = [path for path in structure_dir.glob("*.txt") if path.stem.isdigit()]
    return sorted(paths, key=lambda path: int(path.stem))


def export_inverse_truss_dataset(
    dataset_root: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
    limit: int | None = None,
    quantization_bits: int = 10,
    include_property: bool = True,
    structure_dir_name: str | None = None,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paths = iter_structure_paths(dataset_root, structure_dir_name=structure_dir_name)
    if limit is not None:
        paths = paths[:limit]

    counts = {"written": 0, "skipped": 0}
    split_counts = {"train": 0, "val": 0, "test": 0}
    errors: list[dict[str, str]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for path in paths:
            try:
                record = build_inverse_truss_record(
                    path,
                    dataset_root=dataset_root,
                    quantization_bits=quantization_bits,
                    include_property=include_property,
                )
            except Exception as exc:  # noqa: BLE001 - export should continue and report bad samples.
                counts["skipped"] += 1
                errors.append({"path": str(path), "error": str(exc)})
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["written"] += 1
            split_counts[record["split"]] += 1

    manifest = {
        "dataset_root": str(dataset_root),
        "output_path": str(output_path),
        "schema_version": "inverse_truss_property_grid_v1" if include_property else "inverse_truss_geometry_v1",
        "index_base": 1,
        "coordinate_normalization": "bbox_to_unit_cube",
        "coordinate_quantization_bits": quantization_bits,
        "topology_tree_method": "bfs_parent_pointer_reindexed_v2",
        "structure_dir": str(paths[0].parent) if paths else str((dataset_root / (structure_dir_name or "structures"))),
        "property_status": "stress_grid_v1" if include_property else "not_exported_y_empty",
        "property_representation": {
            "name": "stress_grid_v1",
            "strain_grid": PROPERTY_STRAIN_GRID,
            "stress_unit": "MPa",
            "strain_unit": "dimensionless",
        } if include_property else None,
        "counts": counts,
        "split_counts": split_counts,
        "errors": errors[:100],
        "error_count": len(errors),
    }
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def export_inverse_truss_geometry_dataset(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return export_inverse_truss_dataset(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess paired truss structures for InverseDesigner training.")
    parser.add_argument("--dataset-root", required=True, help="Path containing structures/ and properties/ directories.")
    parser.add_argument("--structure-dir", default=None, help="Structure directory name under dataset root. Defaults to structures/, with geometry/ fallback.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--manifest", default=None, help="Optional manifest JSON path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples to export.")
    parser.add_argument("--quantization-bits", type=int, default=10)
    parser.add_argument("--structure-only", action="store_true", help="Do not read properties/<id>.csv; export y as [].")
    parser.add_argument("--geometry-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    manifest = export_inverse_truss_dataset(
        dataset_root=args.dataset_root,
        output_path=args.output,
        manifest_path=args.manifest,
        limit=args.limit,
        quantization_bits=args.quantization_bits,
        include_property=not (args.structure_only or args.geometry_only),
        structure_dir_name=args.structure_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
