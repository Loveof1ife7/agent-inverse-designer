from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.api import (
        CLOSED_LOOP_DEFAULT_FEM_BACKEND,
        convert_csv_to_abaqus,
        export_txt_to_vtk,
        generate_architecture_csv,
        run_deterministic_surrogate_closed_loop,
        run_group_pipeline,
        solve_group_constraints,
    )
    from src.datagen_contracts import GeneratorConfig, PipelineConfig
else:
    from .api import (
        CLOSED_LOOP_DEFAULT_FEM_BACKEND,
        convert_csv_to_abaqus,
        export_txt_to_vtk,
        generate_architecture_csv,
        run_deterministic_surrogate_closed_loop,
        run_group_pipeline,
        solve_group_constraints,
    )
    from .datagen_contracts import GeneratorConfig, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic surrogate closed-loop CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_constraints = sub.add_parser("constraints", help="solve symmetry constraints")
    p_constraints.add_argument("--group", default="P222")
    p_constraints.add_argument("--group-db", default="symmetry_group_transforms.json")
    p_constraints.add_argument("--export", default="")

    p_generate = sub.add_parser("generate", help="generate architecture csv")
    p_generate.add_argument("--output-dir", required=True)
    p_generate.add_argument("--csv-name", required=True)
    p_generate.add_argument("--samples", type=int, default=100)
    p_generate.add_argument("--constraints-json", default="")
    p_generate.add_argument("--workers", type=int, default=1)
    p_generate.add_argument("--batch", type=int, default=1)
    p_generate.add_argument("--print-every", type=int, default=10)
    p_generate.add_argument("--resume", action="store_true")
    p_generate.add_argument("--allow-single-process-fallback", action="store_true")

    p_pipeline = sub.add_parser("pipeline", help="run full datagen pipeline")
    p_pipeline.add_argument("group", nargs="?", default="P222")
    p_pipeline.add_argument("--basic-size", type=int, default=4)
    p_pipeline.add_argument("--samples", type=int, default=100)
    p_pipeline.add_argument("--workers", type=int, default=1)
    p_pipeline.add_argument("--batch", type=int, default=1)
    p_pipeline.add_argument("--print-every", type=int, default=10)
    p_pipeline.add_argument("--rho-target", type=float, default=0.1)
    p_pipeline.add_argument("--max-bars", type=int, default=10)
    p_pipeline.add_argument("--group-db", default="symmetry_group_transforms.json")
    p_pipeline.add_argument("--run-dir", default="")
    p_pipeline.add_argument("--resume", action="store_true")
    p_pipeline.add_argument("--allow-single-process-fallback", action="store_true")

    p_vtk = sub.add_parser("vtk", help="convert txt geometry to vtk")
    p_vtk.add_argument("--input", required=True)
    p_vtk.add_argument("--output", default="")
    p_vtk.add_argument("--glob", default="*.txt")

    p_abaqus = sub.add_parser("abaqus", help="convert architecture csv to abaqus txt")
    p_abaqus.add_argument("--csv", required=True)
    p_abaqus.add_argument("--out", required=True)
    p_abaqus.add_argument("--group", required=True)
    p_abaqus.add_argument("--group-db", default="symmetry_group_transforms.json")

    p_loop = sub.add_parser("deterministic-run", help="run the deterministic surrogate closed loop")
    p_loop.add_argument("--target-property", required=True, help='JSON target, e.g. {"type":"stress_curve",...}')
    p_loop.add_argument("--workspace-root", default="workspace")
    p_loop.add_argument("--max-iterations", type=int, default=1)
    p_loop.add_argument("--inverse-designer-mode", default="remote_graphmetamat")
    p_loop.add_argument("--surrogate-backend", default="remote_forward")
    p_loop.add_argument("--high-precision-backend", default=CLOSED_LOOP_DEFAULT_FEM_BACKEND)
    p_loop.add_argument("--log-path", default="")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "constraints":
        result = solve_group_constraints(
            group_name=args.group,
            db_path=args.group_db,
            export_path=args.export or None,
            show_plot=False,
        )
        print(json.dumps(result.payload, ensure_ascii=False, indent=2))
        return

    if args.command == "generate":
        config = GeneratorConfig(
            OUTPUT_DIR=args.output_dir,
            CSV_NAME=args.csv_name,
            TARGET_SAMPLES=args.samples,
            RESUME_GENERATION=args.resume,
            CONSTRAINTS_JSON=args.constraints_json,
            N_WORKERS=args.workers,
            BATCH_PER_TASK=args.batch,
            PRINT_EVERY=args.print_every,
        )
        result = generate_architecture_csv(
            config,
            allow_single_process_fallback=args.allow_single_process_fallback,
        )
        print(json.dumps({"csv_path": result.csv_path, "sample_count": result.sample_count}, ensure_ascii=False))
        return

    if args.command == "pipeline":
        config = PipelineConfig(
            group=args.group,
            basic_size=args.basic_size,
            samples=args.samples,
            workers=args.workers,
            batch=args.batch,
            print_every=args.print_every,
            rho_target=args.rho_target,
            max_bars=args.max_bars,
            group_db=args.group_db,
            run_dir=args.run_dir,
            resume=args.resume,
            allow_single_process_fallback=args.allow_single_process_fallback,
        )
        result = run_group_pipeline(config)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "vtk":
        result = export_txt_to_vtk(args.input, args.output or None, glob=args.glob)
        print(json.dumps({"output_path": result.output_path, "count": len(result.exported_files)}, ensure_ascii=False))
        return

    if args.command == "abaqus":
        result = convert_csv_to_abaqus(args.csv, args.out, args.group, args.group_db)
        print(json.dumps({"output_dir": result.output_dir, "txt_count": result.txt_count}, ensure_ascii=False))
        return

    if args.command == "deterministic-run":
        target_property = json.loads(args.target_property)
        result = run_deterministic_surrogate_closed_loop(
            final_target=target_property,
            workspace_root=args.workspace_root,
            max_iterations=args.max_iterations,
            inverse_designer_mode=args.inverse_designer_mode,
            surrogate_backend=args.surrogate_backend,
            high_precision_backend=args.high_precision_backend,
            log_path=args.log_path or None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
