from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _safe_name(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or fallback


def _write_vtk(record: dict[str, Any], vtk_path: Path) -> None:
    nodes = record.get("node_data")
    elements = record.get("element_conn")
    if not isinstance(nodes, list) or not isinstance(elements, list):
        raise ValueError("record must contain list fields 'node_data' and 'element_conn'")

    node_ids: list[int] = []
    points: list[tuple[float, float, float]] = []
    id_to_point_index: dict[int, int] = {}
    for index, row in enumerate(nodes):
        if not isinstance(row, list) or len(row) < 4:
            raise ValueError(f"invalid node_data row at index {index}: {row!r}")
        node_id = int(row[0])
        if node_id in id_to_point_index:
            raise ValueError(f"duplicate node id {node_id}")
        id_to_point_index[node_id] = len(points)
        node_ids.append(node_id)
        points.append((float(row[1]), float(row[2]), float(row[3])))

    lines: list[tuple[int, int]] = []
    edge_node_a: list[int] = []
    edge_node_b: list[int] = []
    for index, row in enumerate(elements):
        if not isinstance(row, list) or len(row) < 2:
            raise ValueError(f"invalid element_conn row at index {index}: {row!r}")
        node_a = int(row[0])
        node_b = int(row[1])
        try:
            point_a = id_to_point_index[node_a]
            point_b = id_to_point_index[node_b]
        except KeyError as exc:
            raise ValueError(f"element {index + 1} references missing node id {exc.args[0]}") from exc
        lines.append((point_a, point_b))
        edge_node_a.append(node_a)
        edge_node_b.append(node_b)

    title = _safe_name(record.get("sample_id"), f"data_{record.get('data_id', 'unknown')}")
    vtk_path.parent.mkdir(parents=True, exist_ok=True)
    with vtk_path.open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(f"{title}\n")
        handle.write("ASCII\n")
        handle.write("DATASET POLYDATA\n")
        handle.write(f"POINTS {len(points)} float\n")
        for x, y, z in points:
            handle.write(f"{x:.9g} {y:.9g} {z:.9g}\n")
        handle.write(f"LINES {len(lines)} {len(lines) * 3}\n")
        for point_a, point_b in lines:
            handle.write(f"2 {point_a} {point_b}\n")
        handle.write(f"POINT_DATA {len(points)}\n")
        handle.write("SCALARS node_id int 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for node_id in node_ids:
            handle.write(f"{node_id}\n")
        handle.write(f"CELL_DATA {len(lines)}\n")
        handle.write("SCALARS edge_id int 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for edge_id in range(1, len(lines) + 1):
            handle.write(f"{edge_id}\n")
        handle.write("SCALARS edge_node_a int 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for node_id in edge_node_a:
            handle.write(f"{node_id}\n")
        handle.write("SCALARS edge_node_b int 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for node_id in edge_node_b:
            handle.write(f"{node_id}\n")


def convert_jsonl(input_path: Path, output_dir: Path) -> list[Path]:
    exported: list[Path] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            data_id = record.get("data_id", line_index - 1)
            sample_name = _safe_name(record.get("sample_id"), "sample")
            stem = f"data_{int(data_id):05d}_{sample_name}" if isinstance(data_id, int) else f"line_{line_index:05d}_{sample_name}"
            vtk_path = output_dir / f"{stem}.vtk"
            _write_vtk(record, vtk_path)
            exported.append(vtk_path)
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert inverse-designer truss JSONL records to legacy VTK PolyData.")
    parser.add_argument("input", type=Path, help="JSONL file with node_data and element_conn fields")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated .vtk files")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input.parent / "vtk"
    exported = convert_jsonl(args.input, output_dir)
    print(json.dumps({"count": len(exported), "output_dir": str(output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
