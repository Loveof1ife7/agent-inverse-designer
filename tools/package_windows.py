#!/usr/bin/env python
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "train_datas" / "P222_paired_dataset_0_99999_20260620"


ALWAYS_EXCLUDED_PARTS = {
    ".git",
    ".agent",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    "archive",
    "workspace",
    "experiments",
    "dist",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
    ".zip",
    ".7z",
    ".tar",
    ".gz",
    ".rar",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Windows migration zip package.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "agent-material-windows-lite.zip",
        help="Output zip path.",
    )
    parser.add_argument(
        "--include-full-data",
        action="store_true",
        help="Include raw train_datas structures/ and properties/ directories.",
    )
    return parser.parse_args()


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def should_exclude(path: Path, include_full_data: bool) -> bool:
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)

    if parts & ALWAYS_EXCLUDED_PARTS:
        return True

    if "normalizing-flows" in parts:
        return True

    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True

    if not include_full_data and is_under(path, DEFAULT_DATASET):
        rel_to_dataset = path.relative_to(DEFAULT_DATASET)
        if rel_to_dataset.parts and rel_to_dataset.parts[0] in {"structures", "properties"}:
            return True

    return False


def iter_files(include_full_data: bool) -> list[Path]:
    include_roots = [
        ROOT / "src",
        ROOT / "demos",
        ROOT / "docs",
        ROOT / "tests",
        ROOT / "tools",
        ROOT / "train_datas",
    ]
    include_files = [
        ROOT / "requirements.txt",
        ROOT / "setup_and_workflow.md",
        ROOT / ".gitignore",
    ]

    files: list[Path] = []
    for file_path in include_files:
        if file_path.exists() and file_path.is_file() and not should_exclude(file_path, include_full_data):
            files.append(file_path)

    for root in include_roots:
        if not root.exists():
            continue
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if should_exclude(file_path, include_full_data):
                continue
            files.append(file_path)

    return sorted(set(files), key=lambda item: item.relative_to(ROOT).as_posix())


def write_zip(output: Path, files: list[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            archive.write(file_path, file_path.relative_to(ROOT).as_posix())


def main() -> None:
    args = parse_args()
    output = args.output
    if not output.is_absolute():
        output = ROOT / output

    files = iter_files(include_full_data=args.include_full_data)
    write_zip(output, files)

    size_mb = output.stat().st_size / (1024 * 1024)
    data_mode = "full-data" if args.include_full_data else "lite"
    print(f"package: {output}")
    print(f"mode: {data_mode}")
    print(f"files: {len(files)}")
    print(f"size_mb: {size_mb:.2f}")


if __name__ == "__main__":
    main()
