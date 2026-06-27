from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import MISSING, asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..closed_loop_contracts import DatagenConfig, FEMResult, KnowledgeSample
from ..datagen_contracts import (
    AbaqusConversionResult,
    ConstraintSolveResult,
    CrystalExpansionResult,
    GenerationResult,
    GeneratorConfig,
    PipelineConfig,
    PipelineResult,
    VtkExportResult,
)
from .core import abaqus_converter, constraints_solver, crystal_builder, dataset_generator, fem as core_fem
from .core.inspect_truss_txt import load_truss_from_txt


PACKAGE_DIR = Path(__file__).resolve().parent
CORE_DIR = PACKAGE_DIR / "core"
PROJECT_ROOT = PACKAGE_DIR.parent.parent
AbaqusFEMConfig = core_fem.AbaqusFEMConfig
AbaqusFEMRunResult = core_fem.AbaqusFEMRunResult


def _default_group_db() -> str:
    return str(CORE_DIR / "symmetry_group_transforms.json")


def _default_output_root() -> Path:
    return PROJECT_ROOT / "workspace"


def _as_path(path_like: str | os.PathLike[str]) -> Path:
    return Path(path_like).expanduser().resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_replication_counts(lattice_lengths: list[float] | tuple[float, float, float], basic_size: int) -> tuple[int, int, int]:
    if len(lattice_lengths) != 3:
        raise ValueError("lattice_lengths must contain exactly 3 values")

    counts = []
    for axis, value in zip(("x", "y", "z"), lattice_lengths):
        length = float(value)
        if length <= 0:
            raise ValueError(f"invalid lattice length on axis {axis}: {length}")
        ratio = float(basic_size) / length
        rounded = int(round(ratio))
        if abs(ratio - rounded) > 1e-9:
            raise ValueError(
                f"basic_size={basic_size} incompatible with lattice_lengths={list(lattice_lengths)} on axis={axis}"
            )
        counts.append(rounded)
    return tuple(counts)


def _make_group_command(config: "AutoGenerateConfig") -> list[str]:
    command = [
        sys.executable,
        str(CORE_DIR / "auto_generate_4x4x4.py"),
        config.group,
        "--basic-size",
        str(config.basic_size),
        "--samples",
        str(config.samples),
        "--workers",
        str(config.workers),
        "--batch",
        str(config.batch),
        "--print-every",
        str(config.print_every),
        "--group-db",
        config.group_db or _default_group_db(),
        "--max-bars",
        str(config.max_bars),
        "--rho-target",
        str(config.rho_target),
    ]
    if config.run_dir:
        command.extend(["--run-dir", config.run_dir])
    if config.resume:
        command.append("--resume")
    if config.allow_single_process_fallback:
        command.append("--allow-single-process-fallback")
    return command


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    root_str = str(PROJECT_ROOT)
    if current_pythonpath:
        paths = current_pythonpath.split(os.pathsep)
        if root_str not in paths:
            env["PYTHONPATH"] = os.pathsep.join([root_str, current_pythonpath])
    else:
        env["PYTHONPATH"] = root_str
    return env


def _summary_path_for_group(run_dir: Path) -> Path:
    return run_dir / "summary.json"


def _read_summary_if_exists(run_dir: Path) -> dict[str, Any] | None:
    summary_path = _summary_path_for_group(run_dir)
    if not summary_path.exists():
        return None
    return _load_json(summary_path)


def _not_exposed(*_args, **_kwargs):
    raise NotImplementedError(
        "This facade only exposes scheduler-level wrappers around core/auto_generate_4x4x4.py "
        "and core/run_all_groups_4x4x4.ps1 from DatagenFEMEvaluator/core."
    )


def _existing_group_db(path_like: str | os.PathLike[str] | None = None) -> str:
    if path_like:
        path = Path(path_like)
        if path.exists():
            return str(path)
    return _default_group_db()


@dataclass(frozen=True)
class AutoGenerateConfig:
    group: str = "P222"
    basic_size: int = 4
    samples: int = 25000
    workers: int = 18
    batch: int = 50
    print_every: int = 10
    group_db: str = field(default_factory=_default_group_db)
    run_dir: str = ""
    resume: bool = False
    allow_single_process_fallback: bool = False
    max_bars: int = 10
    rho_target: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AutoGenerateResult:
    group: str
    command: list[str]
    exit_code: int
    run_dir: str
    status: str
    summary_path: str | None = None
    summary: dict[str, Any] | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    generated_data_manifest_path: str | None = None
    knowledge_base_seed_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BatchGenerateConfig:
    workers: int = 18
    samples: int = 25000
    basic_size: int = 4
    max_bars: int = 10
    rho_target: float = 0.1
    poll_seconds: int = 10
    idle_timeout_minutes: int = 15
    group_timeout_minutes: int = 180
    stop_on_failure: bool = True
    include_groups: tuple[str, ...] = ()
    exclude_groups: tuple[str, ...] = ()
    group_db: str = field(default_factory=_default_group_db)
    output_root: str = ""
    batch_dir: str = ""
    resume: bool = True
    allow_single_process_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchGroupResult:
    group: str
    index: int
    total: int
    status: str
    exit_code: int | None
    run_dir: str
    stdout_path: str | None = None
    stderr_path: str | None = None
    summary_path: str | None = None
    summary: dict[str, Any] | None = None
    generated_data_manifest_path: str | None = None
    knowledge_base_seed_path: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchGenerateResult:
    output_root: str
    batch_dir: str
    progress_path: str
    groups_total: int
    groups_finished: int
    stop_triggered: bool
    results: list[BatchGroupResult]
    skipped: list[BatchGroupResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "batch_dir": self.batch_dir,
            "progress_path": self.progress_path,
            "groups_total": self.groups_total,
            "groups_finished": self.groups_finished,
            "stop_triggered": self.stop_triggered,
            "results": [item.to_dict() for item in self.results],
            "skipped": [item.to_dict() for item in self.skipped],
        }


@dataclass
class BootstrapDatagenResult:
    output_dir: str
    kb_path: str
    dataset_jsonl_path: str
    summary_path: str
    total_samples: int
    label_counts: dict[str, int]
    run_results: list[AutoGenerateResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "kb_path": self.kb_path,
            "dataset_jsonl_path": self.dataset_jsonl_path,
            "summary_path": self.summary_path,
            "total_samples": self.total_samples,
            "label_counts": dict(self.label_counts),
            "run_results": [item.to_dict() for item in self.run_results],
        }


def _dataclass_field_spec(cls) -> list[dict[str, Any]]:
    spec = []
    for item in fields(cls):
        default = None
        if item.default is not MISSING:
            default = item.default
        elif item.default_factory is not MISSING:  # type: ignore[attr-defined]
            try:
                default = item.default_factory()  # type: ignore[misc]
            except TypeError:
                default = "<factory>"
        spec.append(
            {
                "name": item.name,
                "type": str(item.type),
                "default": default,
            }
        )
    return spec


def get_interface_contract() -> dict[str, Any]:
    return {
        "package": "DatagenFEMEvaluator",
        "stable_public_interface": {
            "class": "DatagenFEMEvaluator",
            "functions": [
                "run_auto_generate_4x4x4",
                "run_all_groups_4x4x4",
            ],
            "methods": [
                "auto_generate_4x4x4",
                "run_group",
                "run_group_pipeline",
                "run_all_groups_4x4x4",
                "run_all_groups",
                "datagen_schema",
                "datagen",
                "fem_evaluate",
                "collect_samples",
                "evaluate_existing_candidate",
                "evaluate_explicit_structure",
                "collect_explicit_structure_sample",
            ],
        },
        "config_types": {
            "AutoGenerateConfig": _dataclass_field_spec(AutoGenerateConfig),
            "BatchGenerateConfig": _dataclass_field_spec(BatchGenerateConfig),
            "AbaqusFEMConfig": _dataclass_field_spec(core_fem.AbaqusFEMConfig),
        },
        "result_types": {
            "AutoGenerateResult": _dataclass_field_spec(AutoGenerateResult),
            "BatchGroupResult": _dataclass_field_spec(BatchGroupResult),
            "BatchGenerateResult": _dataclass_field_spec(BatchGenerateResult),
            "BootstrapDatagenResult": _dataclass_field_spec(BootstrapDatagenResult),
        },
        "artifacts": {
            "single_run": [
                "constraints_<group>.json",
                "<group>-architecture.csv",
                "abaqus_txt/*.txt",
                "crystal_4x4x4/*.txt",
                "summary.json",
                "generated_data_manifest.json",
                "knowledge_base_seed.jsonl",
                "auto_generate.stdout.log",
                "auto_generate.stderr.log",
                "bootstrap_dataset.jsonl",
                "bootstrap_summary.json",
            ],
            "batch_run": [
                "_batch/progress.tsv",
                "_batch/<index>_<group>.log",
                "_batch/<index>_<group>.err.log",
                "<output_root>/<group>/summary.json",
                "<output_root>/<group>/generated_data_manifest.json",
                "<output_root>/<group>/knowledge_base_seed.jsonl",
            ],
        },
        "knowledge_base_seed_record": {
            "structure_id": "string",
            "sample_index": "int|string",
            "csv_row_id": "string|null",
            "csv_name": "string",
            "group": "string",
            "basic_size": "int",
            "replication": {"nx": "int", "ny": "int", "nz": "int"},
            "csv_path": "string|null",
            "constraints_path": "string|null",
            "abaqus_txt_path": "string|null",
            "crystal_txt_path": "string|null",
            "run_dir": "string",
        },
        "notes": [
            "This facade schedules core/auto_generate_4x4x4.py and does not modify the core math pipeline.",
            "Paths are normalized for Linux/Windows-compatible launching through Python subprocess.",
            "The preferred KB ingestion artifact is knowledge_base_seed.jsonl.",
            "fem_evaluate supports proxy, abaqus, and auto backends. Proxy remains the default; Abaqus requires ABAQUS_CMD or abq2022/abaqus in PATH.",
            "bootstrap_dataset_and_kb builds a finite exploratory dataset and inserts the resulting knowledge samples into KnowledgeBase.",
        ],
    }


def run_auto_generate_4x4x4(config: AutoGenerateConfig | dict[str, Any]) -> AutoGenerateResult:
    if isinstance(config, dict):
        config = AutoGenerateConfig(**config)

    run_dir = _as_path(config.run_dir) if config.run_dir else _default_output_root() / config.group
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "auto_generate.stdout.log"
    stderr_path = run_dir / "auto_generate.stderr.log"
    command = _make_group_command(
        AutoGenerateConfig(
            group=config.group,
            basic_size=config.basic_size,
            samples=config.samples,
            workers=config.workers,
            batch=config.batch,
            print_every=config.print_every,
            group_db=config.group_db,
            run_dir=str(run_dir),
            resume=config.resume,
            allow_single_process_fallback=config.allow_single_process_fallback,
            max_bars=config.max_bars,
            rho_target=config.rho_target,
        )
    )

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=_subprocess_env(),
            check=False,
        )

    summary = _read_summary_if_exists(run_dir)
    summary_path = _summary_path_for_group(run_dir)
    generated_data_manifest_path = None
    knowledge_base_seed_path = None
    if summary is not None:
        generated_data_manifest_path, knowledge_base_seed_path = _write_generated_data_outputs(run_dir, summary)
    status = "DONE" if completed.returncode == 0 else "FAIL"
    return AutoGenerateResult(
        group=config.group,
        command=command,
        exit_code=int(completed.returncode),
        run_dir=str(run_dir),
        status=status,
        summary_path=str(summary_path) if summary_path.exists() else None,
        summary=summary,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        generated_data_manifest_path=generated_data_manifest_path,
        knowledge_base_seed_path=knowledge_base_seed_path,
    )


def run_group_pipeline(config: PipelineConfig | AutoGenerateConfig | dict[str, Any]) -> PipelineResult:
    if isinstance(config, PipelineConfig):
        pipeline_config = config
    elif isinstance(config, AutoGenerateConfig):
        pipeline_config = PipelineConfig(
            group=config.group,
            basic_size=config.basic_size,
            samples=config.samples,
            workers=config.workers,
            batch=config.batch,
            print_every=config.print_every,
            group_db=config.group_db,
            run_dir=config.run_dir,
            resume=config.resume,
            allow_single_process_fallback=config.allow_single_process_fallback,
            max_bars=config.max_bars,
            rho_target=config.rho_target,
        )
    else:
        pipeline_config = PipelineConfig(**config)

    auto_result = run_auto_generate_4x4x4(
        AutoGenerateConfig(
            group=pipeline_config.group,
            basic_size=pipeline_config.basic_size,
            samples=pipeline_config.samples,
            workers=pipeline_config.workers,
            batch=pipeline_config.batch,
            print_every=pipeline_config.print_every,
            group_db=_existing_group_db(pipeline_config.group_db),
            run_dir=pipeline_config.run_dir,
            resume=pipeline_config.resume,
            allow_single_process_fallback=pipeline_config.allow_single_process_fallback,
        )
    )
    summary = auto_result.summary or _read_summary_if_exists(Path(auto_result.run_dir)) or {}
    constraints_path = summary.get("constraints_json") or str(Path(auto_result.run_dir) / f"constraints_{pipeline_config.group}.json")
    csv_path = summary.get("csv_path") or str(Path(auto_result.run_dir) / f"{pipeline_config.group}-architecture.csv")
    abaqus_dir = summary.get("abaqus_txt_dir") or str(Path(auto_result.run_dir) / "abaqus_txt")
    crystal_dir = summary.get("crystal_dir") or str(Path(auto_result.run_dir) / "crystal_4x4x4")
    replication = dict(summary.get("replication") or {})
    return PipelineResult(
        group=pipeline_config.group,
        basic_size=pipeline_config.basic_size,
        run_dir=auto_result.run_dir,
        constraints=ConstraintSolveResult(
            group_name=pipeline_config.group,
            lattice_lengths=summary.get("lattice_lengths"),
            payload={"lattice_lengths": summary.get("lattice_lengths")},
            constraints_path=constraints_path,
        ),
        generation=GenerationResult(
            csv_path=csv_path,
            output_dir=auto_result.run_dir,
            sample_count=int(summary.get("samples_target") or pipeline_config.samples),
            config=GeneratorConfig(
                OUTPUT_DIR=auto_result.run_dir,
                CSV_NAME=Path(csv_path).name,
                TARGET_SAMPLES=int(summary.get("samples_target") or pipeline_config.samples),
                N_WORKERS=pipeline_config.workers,
                BATCH_PER_TASK=pipeline_config.batch,
                PRINT_EVERY=pipeline_config.print_every,
            ),
        ),
        abaqus=AbaqusConversionResult(
            output_dir=abaqus_dir,
            txt_count=int(summary.get("abaqus_txt_count") or 0),
            total_rows=int(summary.get("samples_target") or pipeline_config.samples),
        ),
        crystal=CrystalExpansionResult(
            output_dir=crystal_dir,
            processed=int(summary.get("crystal_processed") or 0),
            failed=int(summary.get("crystal_failed") or 0),
            error_log=str(summary.get("crystal_error_log") or Path(crystal_dir) / "errors.log"),
        ),
        replication={key: int(value) for key, value in replication.items()},
        summary_path=auto_result.summary_path or str(Path(auto_result.run_dir) / "summary.json"),
    )


def _load_group_names(group_db_path: Path, include_groups: tuple[str, ...], exclude_groups: tuple[str, ...]) -> list[str]:
    data = _load_json(group_db_path)
    groups = sorted(data.get("groups", {}).keys())
    if include_groups:
        include = {item.strip() for item in include_groups if item.strip()}
        groups = [item for item in groups if item in include]
    if exclude_groups:
        exclude = {item.strip() for item in exclude_groups if item.strip()}
        groups = [item for item in groups if item not in exclude]
    return groups


def _compatibility_detail(group_db_path: Path, group_name: str, basic_size: int) -> tuple[bool, str]:
    data = _load_json(group_db_path)
    group_payload = data.get("groups", {}).get(group_name, {})
    lattice_lengths = group_payload.get("lattice_lengths")
    if not lattice_lengths:
        return True, "lattice_lengths_missing_or_invalid"
    try:
        _compute_replication_counts(lattice_lengths, basic_size)
    except ValueError as exc:
        return False, str(exc)
    return True, "ok"


def _progress_state(run_dir: Path, stdout_path: Path, stderr_path: Path) -> tuple[int, int, int, int, int]:
    csv_size = 0
    log_size = stdout_path.stat().st_size if stdout_path.exists() else 0
    err_size = stderr_path.stat().st_size if stderr_path.exists() else 0
    abaqus_count = 0
    crystal_count = 0

    csv_candidates = sorted(run_dir.glob("*-architecture.csv"))
    if csv_candidates:
        csv_size = csv_candidates[0].stat().st_size

    abaqus_dir = run_dir / "abaqus_txt"
    if abaqus_dir.exists():
        abaqus_count = len(list(abaqus_dir.glob("*.txt")))

    crystal_dir = run_dir / "crystal_4x4x4"
    if crystal_dir.exists():
        crystal_count = len(list(crystal_dir.glob("*.txt")))

    return csv_size, log_size, err_size, abaqus_count, crystal_count


def _read_csv_name_map(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    mapping: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_id = str(row.get("id", "")).strip()
            if not row_id:
                continue
            mapping[row_id] = str(row.get("name", "")).strip()
    return mapping


def _sample_sort_key(stem: str):
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def _write_generated_data_outputs(run_dir: Path, summary: dict[str, Any]) -> tuple[str | None, str | None]:
    csv_path = Path(summary.get("csv_path", "")) if summary.get("csv_path") else None
    constraints_path = Path(summary.get("constraints_json", "")) if summary.get("constraints_json") else None
    abaqus_dir = Path(summary.get("abaqus_txt_dir", "")) if summary.get("abaqus_txt_dir") else None
    crystal_dir = Path(summary.get("crystal_dir", "")) if summary.get("crystal_dir") else None
    summary_path = run_dir / "summary.json"

    csv_name_map = _read_csv_name_map(csv_path) if csv_path else {}
    abaqus_files = {}
    crystal_files = {}
    if abaqus_dir and abaqus_dir.exists():
        abaqus_files = {path.stem: str(path.resolve()) for path in abaqus_dir.glob("*.txt")}
    if crystal_dir and crystal_dir.exists():
        crystal_files = {path.stem: str(path.resolve()) for path in crystal_dir.glob("*.txt")}

    sample_keys = sorted(set(abaqus_files) | set(crystal_files), key=_sample_sort_key)
    records = []
    group_name = str(summary.get("group", "unknown"))
    run_name = run_dir.name
    for stem in sample_keys:
        row_id = stem if stem.isdigit() else ""
        structure_id = f"{group_name}:{stem}" if run_name == group_name else f"{group_name}:{run_name}:{stem}"
        records.append(
            {
                "structure_id": structure_id,
                "sample_index": int(stem) if stem.isdigit() else stem,
                "csv_row_id": row_id or None,
                "csv_name": csv_name_map.get(row_id, ""),
                "group": group_name,
                "basic_size": summary.get("basic_size"),
                "replication": dict(summary.get("replication", {})),
                "csv_path": str(csv_path.resolve()) if csv_path and csv_path.exists() else None,
                "constraints_path": str(constraints_path.resolve()) if constraints_path and constraints_path.exists() else None,
                "abaqus_txt_path": abaqus_files.get(stem),
                "crystal_txt_path": crystal_files.get(stem),
                "run_dir": str(run_dir.resolve()),
            }
        )

    manifest = {
        "format_version": 1,
        "group": summary.get("group"),
        "run_dir": str(run_dir.resolve()),
        "summary_path": str(summary_path.resolve()) if summary_path.exists() else None,
        "artifacts": {
            "constraints_json": str(constraints_path.resolve()) if constraints_path and constraints_path.exists() else None,
            "csv_path": str(csv_path.resolve()) if csv_path and csv_path.exists() else None,
            "abaqus_txt_dir": str(abaqus_dir.resolve()) if abaqus_dir and abaqus_dir.exists() else None,
            "crystal_dir": str(crystal_dir.resolve()) if crystal_dir and crystal_dir.exists() else None,
            "crystal_error_log": summary.get("crystal_error_log"),
        },
        "counts": {
            "samples_target": summary.get("samples_target"),
            "abaqus_txt_count": summary.get("abaqus_txt_count"),
            "crystal_processed": summary.get("crystal_processed"),
            "crystal_failed": summary.get("crystal_failed"),
            "record_count": len(records),
        },
        "records": records,
    }

    manifest_path = run_dir / "generated_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    kb_seed_path = run_dir / "knowledge_base_seed.jsonl"
    with kb_seed_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return str(manifest_path), str(kb_seed_path)


def _read_jsonl_records(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    records = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def solve_constraints(
    group_name: str = "P222",
    db_path: str = "",
    export_path: str | os.PathLike[str] | None = None,
    show_plot: bool = False,
) -> ConstraintSolveResult:
    payload = constraints_solver.solve_and_visualize_constraints(
        group_name=group_name,
        db_path=_existing_group_db(db_path),
        export_path=str(export_path) if export_path else None,
        show_plot=show_plot,
    )
    return ConstraintSolveResult(
        group_name=group_name,
        lattice_lengths=payload.get("lattice_lengths"),
        payload=payload,
        constraints_path=str(Path(export_path).resolve()) if export_path else None,
    )


def preview_generation_batch(config: GeneratorConfig | dict[str, Any], batch_size: int, seed: int):
    if isinstance(config, GeneratorConfig):
        cfg = config.to_kwargs()
    else:
        cfg = dict(config)
    return dataset_generator.worker_generate_batch(cfg, batch_size, seed)


def _architecture_header(node_names: list[str]) -> list[str]:
    header = ["id", "name"]
    for name in node_names:
        header.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
    for index in range(len(node_names) ** 2):
        header.append(f"element_{index + 1}")
    return header


def generate_architecture_csv(
    config: GeneratorConfig | dict[str, Any],
    allow_single_process_fallback: bool = False,
) -> GenerationResult:
    del allow_single_process_fallback
    if isinstance(config, dict):
        config = GeneratorConfig(**config)
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / config.CSV_NAME
    generator = dataset_generator.GeometryGenerator(dataset_generator.TrussConfig(**config.to_kwargs()))
    rows = preview_generation_batch(config, batch_size=config.TARGET_SAMPLES, seed=0)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_architecture_header(generator.node_names_ordered))
        for row_id, payload in enumerate(rows):
            nodes_flat, adj_flat, _length, _bars = payload
            writer.writerow([row_id, f"sample_{row_id}"] + list(nodes_flat) + list(adj_flat))
    return GenerationResult(
        csv_path=str(csv_path.resolve()),
        output_dir=str(output_dir.resolve()),
        sample_count=len(rows),
        config=config,
    )


def csv_to_abaqus(
    csv_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    group_name: str,
    group_db_path: str = "",
) -> AbaqusConversionResult:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = abaqus_converter.TrussGenerator(
        str(csv_path),
        group_name=group_name,
        group_db_path=_existing_group_db(group_db_path),
    )
    txt_count = 0
    for index in range(len(generator.df)):
        nodes, edges, name = generator.process_row(index)
        if nodes is None:
            continue
        generator.save_to_txt(str(output_dir / f"{index}.txt"), nodes, edges, name)
        txt_count += 1
    return AbaqusConversionResult(
        output_dir=str(output_dir.resolve()),
        txt_count=txt_count,
        total_rows=len(generator.df),
    )


def expand_crystal(
    in_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    nx: int,
    ny: int,
    nz: int,
) -> CrystalExpansionResult:
    input_dir = Path(in_dir)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log = output_dir / "errors.log"
    processed = 0
    failed = 0
    with error_log.open("w", encoding="utf-8") as error_handle:
        for input_path in sorted(input_dir.glob("*.txt")):
            try:
                text = input_path.read_text(encoding="utf-8", errors="ignore")
                node_data, element_conn, places = crystal_builder.parse_unitcell(text)
                global_nodes, global_elems = crystal_builder.build_crystal(node_data, element_conn, places, nx, ny, nz)
                output_text = crystal_builder.format_output(global_nodes, global_elems, input_path.stem)
                (output_dir / input_path.name).write_text(output_text, encoding="utf-8")
                processed += 1
            except Exception as exc:
                failed += 1
                error_handle.write(f"[FAIL] {input_path} -> {exc}\n")
    return CrystalExpansionResult(
        output_dir=str(output_dir.resolve()),
        processed=processed,
        failed=failed,
        error_log=str(error_log.resolve()),
    )


def _write_vtk(txt_path: Path, vtk_path: Path) -> None:
    node_ids, points, lines = load_truss_from_txt(str(txt_path))
    vtk_path.parent.mkdir(parents=True, exist_ok=True)
    with vtk_path.open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write(f"{txt_path.stem}\n")
        handle.write("ASCII\n")
        handle.write("DATASET POLYDATA\n")
        handle.write(f"POINTS {len(node_ids)} float\n")
        for point in points:
            handle.write(f"{point[0]} {point[1]} {point[2]}\n")
        handle.write(f"LINES {len(lines)} {len(lines) * 3}\n")
        for line in lines:
            handle.write(f"2 {int(line[0]) - 1} {int(line[1]) - 1}\n")
        handle.write(f"POINT_DATA {len(node_ids)}\n")
        handle.write("SCALARS node_id int 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for node_id in node_ids:
            handle.write(f"{int(node_id)}\n")
        handle.write(f"CELL_DATA {len(lines)}\n")
        handle.write("SCALARS edge_id int 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for index in range(len(lines)):
            handle.write(f"{index + 1}\n")


def export_txt_to_vtk(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    glob: str = "*.txt",
) -> VtkExportResult:
    src = Path(input_path)
    exported = []
    if src.is_dir():
        out_dir = Path(output_path) if output_path else src / "vtk"
        for txt_path in sorted(src.glob(glob)):
            vtk_path = out_dir / f"{txt_path.stem}.vtk"
            _write_vtk(txt_path, vtk_path)
            exported.append(str(vtk_path.resolve()))
        return VtkExportResult(input_path=str(src.resolve()), output_path=str(out_dir.resolve()), exported_files=exported)
    vtk_path = Path(output_path) if output_path else src.with_suffix(".vtk")
    _write_vtk(src, vtk_path)
    return VtkExportResult(input_path=str(src.resolve()), output_path=str(vtk_path.resolve()), exported_files=[str(vtk_path.resolve())])


def deduplicate_architecture_csv(input_path: str | os.PathLike[str], output_path: str | os.PathLike[str]) -> int:
    input_csv = Path(input_path)
    output_csv = Path(output_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    saved_rows = []
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for row in reader:
            key = tuple(row[2:])
            if key in seen:
                continue
            seen.add(key)
            saved_rows.append(row)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, row in enumerate(saved_rows):
            row = list(row)
            row[0] = str(index)
            writer.writerow(row)
    return len(saved_rows)


def clean_dataset_and_reindex(*_args, **_kwargs):
    return deduplicate_architecture_csv(*_args, **_kwargs)


def plot_truss(*_args, **_kwargs):
    from .core.inspect_truss_txt import plot_truss as _plot_truss

    return _plot_truss(*_args, **_kwargs)


def run_7zip_sharded(*_args, **_kwargs):
    return _not_exposed(*_args, **_kwargs)


_NODE_RE = re.compile(r"\[\s*(\d+)\s*,\s*([^\],]+)\s*,\s*([^\],]+)\s*,\s*([^\],]+)\s*\]")
_EDGE_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")


def _parse_truss_txt(path: str | None) -> tuple[list[tuple[float, float, float]], list[tuple[int, int]]]:
    if not path:
        return [], []
    txt_path = Path(path)
    if not txt_path.exists():
        return [], []
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    nodes = []
    for _node_id, x, y, z in _NODE_RE.findall(text):
        try:
            nodes.append((float(x), float(y), float(z)))
        except ValueError:
            continue
    edges = [(int(a), int(b)) for a, b in _EDGE_RE.findall(text)]
    return nodes, edges


def _extent(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def _edge_length_sum(nodes: list[tuple[float, float, float]], edges: list[tuple[int, int]]) -> float:
    if not nodes:
        return 0.0
    total = 0.0
    for a, b in edges:
        ia = a - 1
        ib = b - 1
        if ia < 0 or ib < 0 or ia >= len(nodes) or ib >= len(nodes):
            continue
        ax, ay, az = nodes[ia]
        bx, by, bz = nodes[ib]
        total += ((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2) ** 0.5
    return total


def _coerce_points(payload: Any) -> list[tuple[float, float, float]]:
    points = []
    for item in payload or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        try:
            points.append((float(item[0]), float(item[1]), float(item[2])))
        except (TypeError, ValueError):
            continue
    return points


def _coerce_edges(payload: Any) -> list[tuple[int, int]]:
    edges = []
    for item in payload or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            edges.append((int(item[0]), int(item[1])))
        except (TypeError, ValueError):
            continue
    return edges


def _normalize_edge_indices(edges: list[tuple[int, int]], node_count: int) -> list[tuple[int, int]]:
    if not edges:
        return []
    min_index = min(min(a, b) for a, b in edges)
    max_index = max(max(a, b) for a, b in edges)
    if min_index == 0 and max_index < node_count:
        return [(a + 1, b + 1) for a, b in edges]
    return edges


def _nodes_edges_from_explicit_structure(structure: dict[str, Any]) -> tuple[list[tuple[float, float, float]], list[tuple[int, int]]]:
    nodes = _coerce_points(
        structure.get("coordinates")
        or structure.get("nodes")
        or structure.get("geometry", {}).get("coordinates")
        or structure.get("truss", {}).get("nodes")
    )
    edges = _coerce_edges(
        structure.get("edges")
        or structure.get("connectivity")
        or structure.get("topology", {}).get("edges")
        or structure.get("truss", {}).get("edges")
    )
    return nodes, _normalize_edge_indices(edges, len(nodes))


def _structure_identifier(structure: dict[str, Any], default: str = "inverse_structure") -> str:
    return str(structure.get("structure_id") or structure.get("sample_id") or structure.get("id") or default)


def _write_explicit_structure_truss_txt(
    structure_id: str,
    nodes: list[tuple[float, float, float]],
    edges: list[tuple[int, int]],
    output_dir: Path,
) -> Path:
    safe_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in structure_id)[:120] or "inverse_structure"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{safe_id}.txt"
    lines = [f"# Data ID: {structure_id}", "node_data = ["]
    for index, (x, y, z) in enumerate(nodes, start=1):
        lines.append(f"    [{index}, {x:.17g}, {y:.17g}, {z:.17g}],")
    lines.extend(["]", "element_conn = ["])
    for a, b in edges:
        lines.append(f"    [{int(a)}, {int(b)}],")
    lines.extend(["]", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _evaluate_nodes_edges_proxy(
    structure_id: str,
    nodes: list[tuple[float, float, float]],
    edges: list[tuple[int, int]],
    structure_path: str | None = None,
    evaluator_name: str = "simplified_proxy_v1",
) -> FEMResult:
    xs = [item[0] for item in nodes]
    ys = [item[1] for item in nodes]
    zs = [item[2] for item in nodes]
    volume = max(_extent(xs) * _extent(ys) * _extent(zs), 1.0)
    edge_length_sum = _edge_length_sum(nodes, edges)
    node_count = len(nodes)
    edge_count = len(edges)
    geometry_status = "valid" if node_count > 0 and edge_count > 0 else "invalid"
    fem_status = "success" if geometry_status == "valid" else "failed"
    if geometry_status == "valid":
        # Temporary FEM replacement: a stable, normalized structural proxy.
        density_proxy = min(edge_length_sum / volume, 1.0)
        connectivity_ratio = edge_count / max(node_count, 1)
        connectivity_proxy = min(connectivity_ratio / 3.0, 1.0)
        stiffness_proxy = min(0.7 * density_proxy + 0.3 * connectivity_proxy, 1.0)
    else:
        density_proxy = 0.0
        connectivity_ratio = 0.0
        connectivity_proxy = 0.0
        stiffness_proxy = 0.0
    return FEMResult(
        structure_id=structure_id,
        evaluated_property={
            "stiffness_proxy": float(stiffness_proxy),
            "density_proxy": float(density_proxy),
        },
        fem_status=fem_status,
        geometry_status=geometry_status,
        raw_metrics={
            "node_count": node_count,
            "edge_count": edge_count,
            "edge_length_sum": edge_length_sum,
            "volume_proxy": volume,
            "connectivity_ratio": connectivity_ratio,
            "connectivity_proxy": connectivity_proxy,
            "structure_path": structure_path,
            "evaluator": evaluator_name,
        },
    )


def _evaluate_structure_proxy(structure: dict[str, Any]) -> FEMResult:
    structure_path = structure.get("crystal_txt_path") or structure.get("abaqus_txt_path") or structure.get("structure_path")
    nodes, edges = _parse_truss_txt(structure_path)
    return _evaluate_nodes_edges_proxy(
        structure_id=str(structure.get("structure_id", "")),
        nodes=nodes,
        edges=edges,
        structure_path=structure_path,
        evaluator_name="simplified_proxy_v1",
    )


def _property_error(target_property: dict[str, float], evaluated_property: dict[str, float]) -> dict[str, float]:
    errors = {}
    for key, target_value in target_property.items():
        observed = float(evaluated_property.get(key, 0.0))
        target = float(target_value)
        scale = abs(target) if abs(target) > 1e-9 else 1.0
        errors[key] = abs(observed - target) / scale
    return errors


def _label_from_error(property_error: dict[str, float]) -> str:
    if not property_error:
        return "failure"
    max_error = max(float(value) for value in property_error.values())
    if max_error <= 0.1:
        return "success"
    if max_error <= 0.35:
        return "near_miss"
    return "failure"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    q = min(max(q, 0.0), 1.0)
    index = q * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _normalize_datagen_config(config: DatagenConfig | dict[str, Any]) -> DatagenConfig:
    if isinstance(config, DatagenConfig):
        return config
    return DatagenConfig(**config)


def _ensure_bootstrap_config_defaults(config: DatagenConfig, index: int, output_dir: Path) -> DatagenConfig:
    suggestion_id = config.suggestion_id or f"bootstrap_{index:03d}_{config.group}"
    run_dir = config.run_dir or str(output_dir / "runs" / suggestion_id)
    source = config.source or "bootstrap_seed"
    hypothesis = config.hypothesis or "Generate a finite exploratory seed dataset for downstream search."
    reason = config.reason or "Bootstrap the initial structured dataset and knowledge base."
    tags = config.tags or ("bootstrap_seed", config.group)
    return DatagenConfig(
        suggestion_id=suggestion_id,
        parent_sample_id=config.parent_sample_id,
        source=source,
        target_property=dict(config.target_property),
        expected_property=dict(config.expected_property),
        objective=config.objective,
        confidence=config.confidence,
        group=config.group,
        basic_size=config.basic_size,
        num_samples=config.num_samples,
        workers=config.workers,
        batch=config.batch,
        print_every=config.print_every,
        run_dir=run_dir,
        symmetry=config.symmetry,
        basic_unit_type=config.basic_unit_type,
        unit_cell_type=config.unit_cell_type,
        topology_type=config.topology_type,
        connectivity_pattern=config.connectivity_pattern,
        max_bars=config.max_bars,
        rho_target=config.rho_target,
        density_range=config.density_range,
        parameter_ranges=dict(config.parameter_ranges),
        sampling_strategy=config.sampling_strategy,
        constraints=dict(config.constraints),
        design_search_parameters=config.design_search_parameters,
        hypothesis=hypothesis,
        reason=reason,
        failure_analysis=dict(config.failure_analysis),
        exploration_strategy=config.exploration_strategy,
        tags=tuple(tags),
    )


def _derive_bootstrap_target_property(config: DatagenConfig, fem_results: list[FEMResult]) -> dict[str, float]:
    if config.target_property:
        return dict(config.target_property)
    if config.expected_property:
        return dict(config.expected_property)
    stiffness_values = [
        float(result.evaluated_property.get("stiffness_proxy", 0.0))
        for result in fem_results
        if result.fem_status == "success"
    ]
    density_values = [
        float(result.evaluated_property.get("density_proxy", config.rho_target))
        for result in fem_results
        if result.fem_status == "success"
    ]
    return {
        "stiffness_proxy": _percentile(stiffness_values, 0.85),
        "density_proxy": config.rho_target if not density_values else _percentile(density_values, 0.5),
    }


def _run_group_with_watchdog(
    group_name: str,
    index: int,
    total: int,
    config: BatchGenerateConfig,
    output_root: Path,
    batch_dir: Path,
    stop_flag: Path,
) -> BatchGroupResult:
    run_dir = output_root / group_name
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = batch_dir / f"{index:03d}_{group_name}.log"
    stderr_path = batch_dir / f"{index:03d}_{group_name}.err.log"
    auto_config = AutoGenerateConfig(
        group=group_name,
        basic_size=config.basic_size,
        samples=config.samples,
        workers=config.workers,
        batch=50,
        print_every=10,
        group_db=config.group_db,
        run_dir=str(run_dir),
        resume=config.resume,
        allow_single_process_fallback=config.allow_single_process_fallback,
        max_bars=config.max_bars,
        rho_target=config.rho_target,
    )
    command = _make_group_command(auto_config)

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=_subprocess_env(),
        )

    start_at = time.monotonic()
    last_progress_at = start_at
    last_state = _progress_state(run_dir, stdout_path, stderr_path)
    status = ""
    exit_code: int | None = None

    while True:
        time.sleep(max(int(config.poll_seconds), 1))

        if stop_flag.exists():
            if process.poll() is None:
                process.kill()
                process.wait()
            status = "STOPPED"
            exit_code = 130
            break

        current_state = _progress_state(run_dir, stdout_path, stderr_path)
        if current_state != last_state:
            last_state = current_state
            last_progress_at = time.monotonic()

        if process.poll() is not None:
            exit_code = int(process.returncode)
            status = "DONE" if exit_code == 0 else "FAIL"
            break

        elapsed_minutes = (time.monotonic() - start_at) / 60.0
        idle_minutes = (time.monotonic() - last_progress_at) / 60.0
        if elapsed_minutes > float(config.group_timeout_minutes):
            process.kill()
            process.wait()
            status = "TIMEOUT"
            exit_code = 124
            break
        if idle_minutes > float(config.idle_timeout_minutes):
            process.kill()
            process.wait()
            status = "IDLE_TIMEOUT"
            exit_code = 125
            break

    summary = _read_summary_if_exists(run_dir)
    summary_path = _summary_path_for_group(run_dir)
    generated_data_manifest_path = None
    knowledge_base_seed_path = None
    if summary is not None:
        generated_data_manifest_path, knowledge_base_seed_path = _write_generated_data_outputs(run_dir, summary)
    return BatchGroupResult(
        group=group_name,
        index=index,
        total=total,
        status=status,
        exit_code=exit_code,
        run_dir=str(run_dir),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        summary_path=str(summary_path) if summary_path.exists() else None,
        summary=summary,
        generated_data_manifest_path=generated_data_manifest_path,
        knowledge_base_seed_path=knowledge_base_seed_path,
        detail="",
    )


def run_all_groups_4x4x4(config: BatchGenerateConfig | dict[str, Any]) -> BatchGenerateResult:
    if isinstance(config, dict):
        config = BatchGenerateConfig(**config)

    group_db_path = _as_path(config.group_db or _default_group_db())
    output_root = _as_path(config.output_root) if config.output_root else _default_output_root()
    batch_dir = _as_path(config.batch_dir) if config.batch_dir else (output_root / "_batch")
    batch_dir.mkdir(parents=True, exist_ok=True)
    stop_flag = batch_dir / "STOP"
    if stop_flag.exists():
        stop_flag.unlink()

    groups = _load_group_names(group_db_path, config.include_groups, config.exclude_groups)
    progress_path = batch_dir / "progress.tsv"
    progress_path.write_text("timestamp\tindex\ttotal\tgroup\texit_code\tstatus\n", encoding="utf-8")

    skipped: list[BatchGroupResult] = []
    results: list[BatchGroupResult] = []
    stop_triggered = False

    for raw_index, group_name in enumerate(groups, start=1):
        run_dir = output_root / group_name
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if run_dir.exists():
            skipped_result = BatchGroupResult(
                group=group_name,
                index=raw_index,
                total=len(groups),
                status="SKIP_DIR_EXISTS",
                exit_code=0,
                run_dir=str(run_dir),
                detail=str(run_dir),
            )
            skipped.append(skipped_result)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{timestamp}\t{raw_index}\t{len(groups)}\t{group_name}\t0\tSKIP_DIR_EXISTS\n")
            continue

        compatible, detail = _compatibility_detail(group_db_path, group_name, config.basic_size)
        if not compatible:
            skipped_result = BatchGroupResult(
                group=group_name,
                index=raw_index,
                total=len(groups),
                status="SKIP_INCOMPATIBLE_BASIC_SIZE",
                exit_code=0,
                run_dir=str(run_dir),
                detail=detail,
            )
            skipped.append(skipped_result)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{timestamp}\t{raw_index}\t{len(groups)}\t{group_name}\t0\tSKIP_INCOMPATIBLE_BASIC_SIZE\n"
                )
            continue

        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp}\t{raw_index}\t{len(groups)}\t{group_name}\t\tSTART\n")

        result = _run_group_with_watchdog(
            group_name=group_name,
            index=raw_index,
            total=len(groups),
            config=config,
            output_root=output_root,
            batch_dir=batch_dir,
            stop_flag=stop_flag,
        )
        results.append(result)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with progress_path.open("a", encoding="utf-8") as handle:
            exit_code_str = "" if result.exit_code is None else str(result.exit_code)
            handle.write(f"{timestamp}\t{raw_index}\t{len(groups)}\t{group_name}\t{exit_code_str}\t{result.status}\n")

        if config.stop_on_failure and result.status != "DONE":
            stop_triggered = True
            break

    return BatchGenerateResult(
        output_root=str(output_root),
        batch_dir=str(batch_dir),
        progress_path=str(progress_path),
        groups_total=len(groups),
        groups_finished=len(results) + len(skipped),
        stop_triggered=stop_triggered,
        results=results,
        skipped=skipped,
    )


class DatagenFEMEvaluator:
    """
    Agent-facing scheduler facade.

    This class only wraps scheduler entrypoints under DatagenFEMEvaluator/core
    and does not change the algorithm scripts themselves.
    """

    def __init__(
        self,
        workspace_root: str | os.PathLike[str] | None = None,
        fem_backend: str | None = None,
        fem_config: core_fem.AbaqusFEMConfig | dict[str, Any] | None = None,
    ):
        self.workspace_root = _as_path(workspace_root) if workspace_root else _default_output_root()
        self.fem_backend = (fem_backend or os.getenv("DATAGEN_FEM_BACKEND", "proxy")).strip().lower()
        if isinstance(fem_config, core_fem.AbaqusFEMConfig):
            self.fem_config = fem_config
        elif isinstance(fem_config, dict):
            self.fem_config = core_fem.AbaqusFEMConfig(**fem_config)
        else:
            self.fem_config = core_fem.AbaqusFEMConfig(output_root=str(self.workspace_root / "fem_runs"))

    @staticmethod
    def interface_contract() -> dict[str, Any]:
        return get_interface_contract()

    def datagen_schema(self) -> dict[str, Any]:
        return {
            "default_group": "P222",
            "default_basic_size": 4,
            "default_num_samples": 8,
            "supported_outputs": [
                "summary.json",
                "generated_data_manifest.json",
                "knowledge_base_seed.jsonl",
                "bootstrap_dataset.jsonl",
                "bootstrap_summary.json",
                "abaqus_txt/*.txt",
                "crystal_4x4x4/*.txt",
            ],
            "config_fields": [item.name for item in fields(DatagenConfig)],
            "fem_backends": ["proxy", "abaqus", "auto"],
            "active_fem_backend": self.fem_backend,
        }

    def datagen(self, config: DatagenConfig | dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(config, dict):
            config = DatagenConfig(**config)

        run_dir = Path(config.run_dir) if config.run_dir else self.workspace_root / config.group
        result = self.auto_generate_4x4x4(
            AutoGenerateConfig(
                group=config.group,
                basic_size=config.basic_size,
                samples=config.num_samples,
                workers=config.workers,
                batch=config.batch,
                print_every=config.print_every,
                run_dir=str(run_dir),
                allow_single_process_fallback=True,
                max_bars=config.max_bars,
                rho_target=config.rho_target,
            )
        )

        records = _read_jsonl_records(result.knowledge_base_seed_path)
        for record in records:
            record["datagen_config"] = config.to_dict()
            record["datagen_result"] = result.to_dict()
            record["structure_path"] = (
                record.get("crystal_txt_path")
                or record.get("abaqus_txt_path")
                or record.get("csv_path")
                or ""
            )
        return records

    def fem_evaluate(self, structures: list[dict[str, Any]]) -> list[FEMResult]:
        if self.fem_backend == "proxy":
            return [_evaluate_structure_proxy(structure) for structure in structures]
        if self.fem_backend == "auto" and not core_fem.find_abaqus_command(self.fem_config.abaqus_cmd):
            results = []
            for structure in structures:
                result = _evaluate_structure_proxy(structure)
                result.raw_metrics["fem_backend_requested"] = "auto"
                result.raw_metrics["fem_backend_fallback"] = "proxy"
                result.raw_metrics["fem_backend_fallback_reason"] = "abaqus_unavailable"
                results.append(result)
            return results
        if self.fem_backend in {"abaqus", "auto"}:
            return [self._evaluate_structure_abaqus(structure) for structure in structures]
        raise ValueError(f"Unsupported fem_backend={self.fem_backend!r}; expected proxy, abaqus, or auto")

    def evaluate_explicit_structure(
        self,
        structure: dict[str, Any],
        target_property: dict[str, float],
    ) -> dict[str, Any]:
        structure_path = structure.get("crystal_txt_path") or structure.get("abaqus_txt_path") or structure.get("structure_path")
        structure_id = _structure_identifier(structure)
        if structure_path:
            fem_result = self.fem_evaluate([structure])[0]
        else:
            nodes, edges = _nodes_edges_from_explicit_structure(structure)
            if self.fem_backend == "proxy":
                fem_result = _evaluate_nodes_edges_proxy(
                    structure_id=structure_id,
                    nodes=nodes,
                    edges=edges,
                    structure_path=None,
                    evaluator_name="explicit_structure_proxy_v1",
                )
            elif self.fem_backend == "auto" and not core_fem.find_abaqus_command(self.fem_config.abaqus_cmd):
                fem_result = _evaluate_nodes_edges_proxy(
                    structure_id=structure_id,
                    nodes=nodes,
                    edges=edges,
                    structure_path=None,
                    evaluator_name="explicit_structure_proxy_v1",
                )
                fem_result.raw_metrics["fem_backend_requested"] = "auto"
                fem_result.raw_metrics["fem_backend_fallback"] = "proxy"
                fem_result.raw_metrics["fem_backend_fallback_reason"] = "abaqus_unavailable"
            elif self.fem_backend in {"abaqus", "auto"}:
                fem_run_id = str(structure.get("original_structure_id") or structure_id)
                explicit_path = _write_explicit_structure_truss_txt(
                    structure_id=fem_run_id,
                    nodes=nodes,
                    edges=edges,
                    output_dir=self.workspace_root / "explicit_structures",
                )
                fem_result = self._evaluate_structure_abaqus(
                    {
                        **structure,
                        "structure_id": structure_id,
                        "fem_run_id": fem_run_id,
                        "structure_path": str(explicit_path),
                    }
                )
                fem_result.raw_metrics["explicit_structure_path"] = str(explicit_path)
            else:
                raise ValueError(f"Unsupported fem_backend={self.fem_backend!r}; expected proxy, abaqus, or auto")
        property_error = _property_error(target_property, fem_result.evaluated_property)
        return {
            "structure_id": fem_result.structure_id,
            "evaluated_property": dict(fem_result.evaluated_property),
            "property_error": property_error,
            "label": _label_from_error(property_error),
            "fem_status": fem_result.fem_status,
            "geometry_status": fem_result.geometry_status,
            "raw_metrics": dict(fem_result.raw_metrics),
        }

    def collect_explicit_structure_sample(
        self,
        structure: dict[str, Any],
        evaluation: dict[str, Any],
        target_property: dict[str, float],
        source: str = "inverse_designer",
    ) -> KnowledgeSample:
        structure_id = str(
            structure.get("structure_id")
            or structure.get("sample_id")
            or evaluation.get("structure_id")
            or "inverse_structure"
        )
        evaluation_raw = dict(evaluation.get("raw_metrics") or {})
        artifact_structure_path = (
            structure.get("structure_path")
            or structure.get("crystal_txt_path")
            or structure.get("abaqus_txt_path")
            or evaluation_raw.get("explicit_structure_path")
            or ""
        )
        metadata = {
            "sample_id": structure_id,
            "sample_type": "inverse_designer_explicit_structure",
            "raw_metrics": evaluation_raw,
            "artifacts": {
                "structure_path": artifact_structure_path,
            },
            "explicit_structure": dict(structure),
            "retrieved_property": dict(structure.get("retrieved_property") or {}),
            "retrieval_distance": structure.get("retrieval_distance"),
            "fidelity": "abaqus" if "abaqus" in str(evaluation_raw.get("evaluator", "")) else "proxy",
        }
        return KnowledgeSample(
            structure_id=structure_id,
            structure_path=str(artifact_structure_path),
            unit_cell_type=str(structure.get("unit_cell_type") or "symmetry_expanded_truss"),
            basic_unit_type=str(structure.get("basic_unit_type") or "edge_face_center_19node"),
            topology_type=str(structure.get("topology_type") or "sparse_truss"),
            symmetry=str(structure.get("symmetry") or structure.get("group") or ""),
            connectivity_pattern=str(structure.get("connectivity_pattern") or "default"),
            parameter_config={
                "source": source,
                "retrieval_distance": structure.get("retrieval_distance"),
                "training_target": "explicit_structure",
            },
            target_property=dict(target_property),
            evaluated_property=dict(evaluation.get("evaluated_property") or {}),
            property_error=dict(evaluation.get("property_error") or {}),
            fem_status=str(evaluation.get("fem_status") or "unknown"),
            geometry_status=str(evaluation.get("geometry_status") or "unknown"),
            label=str(evaluation.get("label") or "failure"),
            source=source,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
            explicit_structure=dict(structure),
        )

    def _evaluate_structure_abaqus(self, structure: dict[str, Any]) -> FEMResult:
        structure_path = structure.get("crystal_txt_path") or structure.get("abaqus_txt_path") or structure.get("structure_path") or ""
        structure_id = _structure_identifier(structure, default=Path(structure_path).stem if structure_path else "inverse_structure")
        fem_run_id = str(structure.get("fem_run_id") or structure.get("original_structure_id") or structure_id)
        if not structure_path:
            return FEMResult(
                structure_id=structure_id,
                evaluated_property={"stiffness_proxy": 0.0, "density_proxy": 0.0},
                fem_status="failed",
                geometry_status="invalid",
                raw_metrics={
                    "evaluator": "abaqus_plate_compression",
                    "fem_backend": self.fem_backend,
                    "error": "structure_path missing",
                },
            )
        result = core_fem.evaluate_truss_file(
            structure_path=structure_path,
            structure_id=fem_run_id,
            output_root=self.workspace_root / "fem_runs",
            config=self.fem_config,
        )
        evaluated = dict(result.evaluated_property)
        if not evaluated:
            evaluated = {
                "stiffness_proxy": 0.0,
                "density_proxy": float(result.raw_metrics.get("density_proxy", 0.0)),
            }
        geometry_status = "valid" if result.status not in {"invalid_geometry", "setup_failed"} else "invalid"
        fem_status = "success" if result.status == "success" else result.status
        return FEMResult(
            structure_id=structure_id,
            evaluated_property=evaluated,
            fem_status=fem_status,
            geometry_status=geometry_status,
            raw_metrics={
                **dict(result.raw_metrics),
                "evaluator": "abaqus_plate_compression",
                "fem_backend": self.fem_backend,
                "fem_run_id": fem_run_id,
                "fem_run_status": result.status,
                "fem_run_dir": result.run_dir,
                "fem_inp_path": result.inp_path,
                "fem_curve_path": result.curve_path,
                "fem_error": result.error,
            },
        )

    def collect_samples(
        self,
        structures: list[dict[str, Any]],
        fem_results: list[FEMResult],
        target_property: dict[str, float],
        datagen_config: DatagenConfig | dict[str, Any],
    ) -> list[KnowledgeSample]:
        if isinstance(datagen_config, dict):
            datagen_config = DatagenConfig(**datagen_config)

        samples = []
        for structure, fem_result in zip(structures, fem_results):
            property_error = _property_error(target_property, fem_result.evaluated_property)
            label = _label_from_error(property_error)
            datagen_result = dict(structure.get("datagen_result") or {})
            run_dir = str(
                structure.get("run_dir")
                or structure.get("run_directory")
                or datagen_result.get("run_dir")
                or datagen_config.run_dir
                or ""
            )
            run_id = Path(run_dir).name if run_dir else (datagen_config.suggestion_id or datagen_config.group)
            datagen_summary = dict(datagen_result.get("summary") or {})
            samples.append(
                KnowledgeSample(
                    structure_id=str(structure.get("structure_id") or fem_result.structure_id),
                    structure_path=str(structure.get("structure_path") or structure.get("crystal_txt_path") or ""),
                    unit_cell_type=datagen_config.unit_cell_type,
                    basic_unit_type=datagen_config.basic_unit_type,
                    topology_type=datagen_config.topology_type,
                    symmetry=datagen_config.symmetry or datagen_config.group,
                    connectivity_pattern=datagen_config.connectivity_pattern,
                    parameter_config={
                        "suggestion_id": datagen_config.suggestion_id,
                        "parent_sample_id": datagen_config.parent_sample_id,
                        "max_bars": datagen_config.max_bars,
                        "rho_target": datagen_config.rho_target,
                        "density_range": datagen_config.density_range,
                        "parameter_ranges": datagen_config.parameter_ranges,
                        "sampling_strategy": datagen_config.sampling_strategy,
                        "constraints": datagen_config.constraints,
                        "design_search_parameters": datagen_config.design_search_parameters.to_dict(),
                        "exploration_strategy": datagen_config.exploration_strategy,
                    },
                    target_property=dict(target_property),
                    evaluated_property=dict(fem_result.evaluated_property),
                    property_error=property_error,
                    fem_status=fem_result.fem_status,
                    geometry_status=fem_result.geometry_status,
                    label=label,
                    source=datagen_config.source,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    metadata={
                        "datagen_config": datagen_config.to_dict(),
                        "datagen_result": datagen_result,
                        "agent_suggestion": {
                            "suggestion_id": datagen_config.suggestion_id,
                            "parent_sample_id": datagen_config.parent_sample_id,
                            "objective": datagen_config.objective,
                            "target_property": datagen_config.target_property,
                            "expected_property": datagen_config.expected_property,
                            "confidence": datagen_config.confidence,
                            "hypothesis": datagen_config.hypothesis,
                            "reason": datagen_config.reason,
                            "failure_analysis": datagen_config.failure_analysis,
                            "exploration_strategy": datagen_config.exploration_strategy,
                            "tags": datagen_config.tags,
                        },
                        "run": {
                            "run_id": run_id,
                            "group": datagen_config.group,
                            "run_dir": run_dir,
                            "status": datagen_result.get("status", "unknown"),
                            "target_sample_count": datagen_summary.get("samples_target", 0),
                            "generated_sample_count": datagen_summary.get("crystal_processed", 0),
                            "summary_path": datagen_result.get("summary_path") or "",
                            "generated_data_manifest_path": datagen_result.get("generated_data_manifest_path") or "",
                            "knowledge_base_seed_path": datagen_result.get("knowledge_base_seed_path") or "",
                        },
                        "sample_id": str(structure.get("structure_id") or fem_result.structure_id),
                        "sample_type": datagen_config.exploration_strategy,
                        "raw_metrics": fem_result.raw_metrics,
                        "artifacts": {
                            "csv_path": structure.get("csv_path"),
                            "constraints_path": structure.get("constraints_path"),
                            "abaqus_txt_path": structure.get("abaqus_txt_path"),
                            "crystal_txt_path": structure.get("crystal_txt_path"),
                            "run_dir": run_dir,
                        },
                        "hypothesis": datagen_config.hypothesis,
                        "reason": datagen_config.reason,
                    },
                )
            )
        return samples

    def evaluate_existing_candidate(
        self,
        candidate: KnowledgeSample,
        target_property: dict[str, float],
    ) -> dict[str, Any]:
        property_error = _property_error(target_property, candidate.evaluated_property)
        return {
            "structure_id": candidate.structure_id,
            "evaluated_property": dict(candidate.evaluated_property),
            "property_error": property_error,
            "label": _label_from_error(property_error),
        }

    def bootstrap_dataset_and_kb(
        self,
        datagen_configs: list[DatagenConfig | dict[str, Any]],
        kb_path: str | os.PathLike[str],
        output_dir: str | os.PathLike[str] | None = None,
    ) -> BootstrapDatagenResult:
        bootstrap_dir = _as_path(output_dir) if output_dir else (self.workspace_root / "bootstrap_seed")
        bootstrap_dir.mkdir(parents=True, exist_ok=True)

        from ..KnowledgeBase import KnowledgeBase

        normalized_configs = [
            _ensure_bootstrap_config_defaults(_normalize_datagen_config(config), index + 1, bootstrap_dir)
            for index, config in enumerate(datagen_configs)
        ]

        run_results: list[AutoGenerateResult] = []
        all_samples: list[KnowledgeSample] = []
        dataset_jsonl_path = bootstrap_dir / "bootstrap_dataset.jsonl"
        summary_path = bootstrap_dir / "bootstrap_summary.json"

        for config in normalized_configs:
            structures = self.datagen(config)
            datagen_result_dict = structures[0]["datagen_result"] if structures else {}
            if datagen_result_dict:
                run_results.append(AutoGenerateResult(**datagen_result_dict))
            fem_results = self.fem_evaluate(structures)
            bootstrap_target = _derive_bootstrap_target_property(config, fem_results)
            seeded_config = DatagenConfig(
                suggestion_id=config.suggestion_id,
                parent_sample_id=config.parent_sample_id,
                source=config.source,
                target_property=bootstrap_target,
                expected_property=dict(config.expected_property) if config.expected_property else dict(bootstrap_target),
                objective=config.objective,
                confidence=config.confidence,
                group=config.group,
                basic_size=config.basic_size,
                num_samples=config.num_samples,
                workers=config.workers,
                batch=config.batch,
                print_every=config.print_every,
                run_dir=config.run_dir,
                symmetry=config.symmetry,
                basic_unit_type=config.basic_unit_type,
                unit_cell_type=config.unit_cell_type,
                topology_type=config.topology_type,
                connectivity_pattern=config.connectivity_pattern,
                max_bars=config.max_bars,
                rho_target=config.rho_target,
                density_range=config.density_range,
                parameter_ranges=dict(config.parameter_ranges),
                sampling_strategy=config.sampling_strategy,
                constraints=dict(config.constraints),
                design_search_parameters=config.design_search_parameters,
                hypothesis=config.hypothesis,
                reason=config.reason,
                failure_analysis=dict(config.failure_analysis),
                exploration_strategy=config.exploration_strategy,
                tags=tuple(config.tags),
            )
            samples = self.collect_samples(
                structures=structures,
                fem_results=fem_results,
                target_property=bootstrap_target,
                datagen_config=seeded_config,
            )
            for sample in samples:
                sample.metadata.setdefault("bootstrap", {})
                sample.metadata["bootstrap"].update(
                    {
                        "mode": "finite_exploration_seed",
                        "seed_target_property": bootstrap_target,
                        "seed_objective": "build_regularized_base_dataset_and_knowledge_base",
                    }
                )
            all_samples.extend(samples)

        label_counts: dict[str, int] = {}
        with dataset_jsonl_path.open("w", encoding="utf-8") as handle:
            for sample in all_samples:
                label_counts[sample.label] = label_counts.get(sample.label, 0) + 1
                handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

        kb_realpath = _as_path(kb_path)
        kb = KnowledgeBase(kb_realpath)
        try:
            kb.add_samples(all_samples)
        finally:
            kb.close()

        summary = {
            "format_version": 1,
            "output_dir": str(bootstrap_dir),
            "kb_path": str(kb_realpath),
            "dataset_jsonl_path": str(dataset_jsonl_path),
            "total_samples": len(all_samples),
            "label_counts": label_counts,
            "groups": [config.group for config in normalized_configs],
            "suggestion_ids": [config.suggestion_id for config in normalized_configs],
            "run_dirs": [config.run_dir for config in normalized_configs],
            "runs": [result.to_dict() for result in run_results],
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return BootstrapDatagenResult(
            output_dir=str(bootstrap_dir),
            kb_path=str(kb_realpath),
            dataset_jsonl_path=str(dataset_jsonl_path),
            summary_path=str(summary_path),
            total_samples=len(all_samples),
            label_counts=label_counts,
            run_results=run_results,
        )

    def auto_generate_4x4x4(self, config: AutoGenerateConfig | dict[str, Any]) -> AutoGenerateResult:
        if isinstance(config, dict):
            config = AutoGenerateConfig(**config)
        if not config.run_dir:
            config = AutoGenerateConfig(
                group=config.group,
                basic_size=config.basic_size,
                samples=config.samples,
                workers=config.workers,
                batch=config.batch,
                print_every=config.print_every,
                group_db=config.group_db,
                run_dir=str(self.workspace_root / config.group),
                resume=config.resume,
                allow_single_process_fallback=config.allow_single_process_fallback,
                max_bars=config.max_bars,
                rho_target=config.rho_target,
            )
        return run_auto_generate_4x4x4(config)

    def run_group(self, config: AutoGenerateConfig | dict[str, Any]) -> AutoGenerateResult:
        return self.auto_generate_4x4x4(config)

    def run_group_pipeline(self, config: PipelineConfig | AutoGenerateConfig | dict[str, Any]) -> PipelineResult:
        return run_group_pipeline(config)

    def run_all_groups_4x4x4(self, config: BatchGenerateConfig | dict[str, Any]) -> BatchGenerateResult:
        if isinstance(config, dict):
            config = BatchGenerateConfig(**config)
        if not config.output_root:
            config = BatchGenerateConfig(
                workers=config.workers,
                samples=config.samples,
                basic_size=config.basic_size,
                max_bars=config.max_bars,
                rho_target=config.rho_target,
                poll_seconds=config.poll_seconds,
                idle_timeout_minutes=config.idle_timeout_minutes,
                group_timeout_minutes=config.group_timeout_minutes,
                stop_on_failure=config.stop_on_failure,
                include_groups=config.include_groups,
                exclude_groups=config.exclude_groups,
                group_db=config.group_db,
                output_root=str(self.workspace_root),
                batch_dir=config.batch_dir,
                resume=config.resume,
                allow_single_process_fallback=config.allow_single_process_fallback,
            )
        return run_all_groups_4x4x4(config)

    def run_all_groups(self, config: BatchGenerateConfig | dict[str, Any]) -> BatchGenerateResult:
        return self.run_all_groups_4x4x4(config)


auto_generate_4x4x4 = run_auto_generate_4x4x4
bootstrap_dataset_and_kb = DatagenFEMEvaluator.bootstrap_dataset_and_kb


__all__ = [
    "AbaqusFEMConfig",
    "AbaqusFEMRunResult",
    "AutoGenerateConfig",
    "AutoGenerateResult",
    "BatchGenerateConfig",
    "BatchGenerateResult",
    "BatchGroupResult",
    "BootstrapDatagenResult",
    "DatagenFEMEvaluator",
    "auto_generate_4x4x4",
    "bootstrap_dataset_and_kb",
    "clean_dataset_and_reindex",
    "csv_to_abaqus",
    "deduplicate_architecture_csv",
    "expand_crystal",
    "export_txt_to_vtk",
    "generate_architecture_csv",
    "get_interface_contract",
    "plot_truss",
    "preview_generation_batch",
    "run_all_groups_4x4x4",
    "run_7zip_sharded",
    "run_auto_generate_4x4x4",
    "run_group_pipeline",
    "solve_constraints",
]
