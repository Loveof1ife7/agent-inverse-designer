from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GeneratorConfig:
    OUTPUT_DIR: str = r"C:\Users\admin\Desktop\3Dtruss\Aba2"
    CSV_NAME: str = "Aba2-architecture.csv"
    TARGET_SAMPLES: int = 25000
    RESUME_GENERATION: bool = True
    MAX_BARS: int = 10
    RHO_TARGET: float = 0.1
    R_PHYSICAL: float = 1.0
    MIN_BAR_CLEARANCE_PHYS: float = 4.0
    K_OPTIONS: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    P_OPTIONS: tuple = (0.25, 0.5, 0.75)
    CONSTRAINTS_JSON: str = ""
    TOLERANCE: float = 1e-5
    GENERATION_RETRIES: int = 800
    NX: int = 2
    NY: int = 2
    NZ: int = 2
    REQUIRE_ALL_NODES_CONNECTED: bool = False
    REJECT_INTERNAL_DEGREE1_AFTER_PBC: bool = True
    REJECT_BOUNDARY_DEGREE1_AFTER_PBC: bool = True
    N_WORKERS: int = 15
    TASKS_IN_FLIGHT_PER_WORKER: int = 2
    BATCH_PER_TASK: int = 50
    CSV_FLUSH_EVERY: int = 200
    CSV_WRITE_RETRIES: int = 12
    CSV_WRITE_RETRY_DELAY: float = 0.5
    PRINT_EVERY: int = 10
    MAX_NO_PROGRESS_BATCHES: int = 500
    MAX_NO_PROGRESS_SECONDS: int = 900

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineConfig:
    group: str = "P222"
    basic_size: int = 4
    samples: int = 25000
    workers: int = 18
    batch: int = 50
    print_every: int = 10
    max_bars: int = 10
    rho_target: float = 0.1
    group_db: str = "symmetry_group_transforms.json"
    run_dir: str = ""
    resume: bool = False
    allow_single_process_fallback: bool = False


@dataclass
class ConstraintSolveResult:
    group_name: str
    lattice_lengths: list[float] | None
    payload: dict[str, Any]
    constraints_path: str | None = None


@dataclass
class GenerationResult:
    csv_path: str
    output_dir: str
    sample_count: int
    config: GeneratorConfig


@dataclass
class AbaqusConversionResult:
    output_dir: str
    txt_count: int
    total_rows: int


@dataclass
class CrystalExpansionResult:
    output_dir: str
    processed: int
    failed: int
    error_log: str


@dataclass
class VtkExportResult:
    input_path: str
    output_path: str
    exported_files: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    group: str
    basic_size: int
    run_dir: str
    constraints: ConstraintSolveResult
    generation: GenerationResult
    abaqus: AbaqusConversionResult
    crystal: CrystalExpansionResult
    replication: dict[str, int]
    summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "basic_size": self.basic_size,
            "run_dir": self.run_dir,
            "constraints": asdict(self.constraints),
            "generation": asdict(self.generation),
            "abaqus": asdict(self.abaqus),
            "crystal": asdict(self.crystal),
            "replication": dict(self.replication),
            "summary_path": self.summary_path,
        }


def normalize_path(path_like: str | Path) -> str:
    return str(Path(path_like).resolve())
