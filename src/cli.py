from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.DatagenFEMEvaluator import BatchGenerateConfig, run_all_groups_4x4x4, run_auto_generate_4x4x4
    from src.api import (
        bootstrap_seed_dataset,
        convert_csv_to_abaqus,
        export_inverse_designer_dataset,
        export_txt_to_vtk,
        generate_architecture_csv,
        kb_get_group_statistics,
        kb_get_run_provenance,
        kb_get_sample_evidence,
        kb_query_samples,
        refine_agent_knowledge,
        run_closed_loop_discovery,
        solve_group_constraints,
    )
    from src.closed_loop_contracts import DatagenConfig
    from src.datagen_contracts import GeneratorConfig, PipelineConfig
else:
    from .DatagenFEMEvaluator import BatchGenerateConfig, run_all_groups_4x4x4, run_auto_generate_4x4x4
    from .api import (
        bootstrap_seed_dataset,
        convert_csv_to_abaqus,
        export_inverse_designer_dataset,
        export_txt_to_vtk,
        generate_architecture_csv,
        kb_get_group_statistics,
        kb_get_run_provenance,
        kb_get_sample_evidence,
        kb_query_samples,
        refine_agent_knowledge,
        run_closed_loop_discovery,
        solve_group_constraints,
    )
    from .closed_loop_contracts import DatagenConfig
    from .datagen_contracts import GeneratorConfig, PipelineConfig


def build_parser():
    parser = argparse.ArgumentParser(description="Refactored Truss datagen CLI")
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

    p_batch = sub.add_parser("batch", help="run all compatible groups through the scheduler facade")
    p_batch.add_argument("--workers", type=int, default=1)
    p_batch.add_argument("--samples", type=int, default=1)
    p_batch.add_argument("--basic-size", type=int, default=4)
    p_batch.add_argument("--rho-target", type=float, default=0.1)
    p_batch.add_argument("--max-bars", type=int, default=10)
    p_batch.add_argument("--poll-seconds", type=int, default=10)
    p_batch.add_argument("--idle-timeout-minutes", type=int, default=15)
    p_batch.add_argument("--group-timeout-minutes", type=int, default=180)
    p_batch.add_argument("--group-db", default="")
    p_batch.add_argument("--output-root", default="")
    p_batch.add_argument("--batch-dir", default="")
    p_batch.add_argument("--include-group", action="append", default=[])
    p_batch.add_argument("--exclude-group", action="append", default=[])
    p_batch.add_argument("--stop-on-failure", action="store_true")
    p_batch.add_argument("--no-stop-on-failure", action="store_true")
    p_batch.add_argument("--resume", action="store_true")
    p_batch.add_argument("--allow-single-process-fallback", action="store_true")

    p_vtk = sub.add_parser("vtk", help="convert txt geometry to vtk")
    p_vtk.add_argument("--input", required=True)
    p_vtk.add_argument("--output", default="")
    p_vtk.add_argument("--glob", default="*.txt")

    p_abaqus = sub.add_parser("abaqus", help="convert architecture csv to abaqus txt")
    p_abaqus.add_argument("--csv", required=True)
    p_abaqus.add_argument("--out", required=True)
    p_abaqus.add_argument("--group", required=True)
    p_abaqus.add_argument("--group-db", default="symmetry_group_transforms.json")

    p_discover = sub.add_parser("discover", help="run minimal closed-loop structural discovery")
    p_discover.add_argument("--target-property", required=True, help='JSON string, e.g. {"stiffness_proxy":25,"density_proxy":0.1}')
    p_discover.add_argument("--workspace-root", default="workspace")
    p_discover.add_argument("--kb-path", default="workspace/knowledge.sqlite")
    p_discover.add_argument("--max-iterations", type=int, default=3)
    p_discover.add_argument("--retrain-trigger", type=int, default=10)
    p_discover.add_argument("--log-path", default="")

    p_bootstrap = sub.add_parser("bootstrap", help="build a finite exploratory seed dataset and seed knowledge base")
    p_bootstrap.add_argument("--workspace-root", default="workspace")
    p_bootstrap.add_argument("--kb-path", default="workspace/knowledge.sqlite")
    p_bootstrap.add_argument("--output-dir", default="")
    p_bootstrap.add_argument("--group", action="append", default=[])
    p_bootstrap.add_argument("--samples", type=int, default=4)
    p_bootstrap.add_argument("--workers", type=int, default=1)
    p_bootstrap.add_argument("--batch", type=int, default=1)
    p_bootstrap.add_argument("--print-every", type=int, default=1)
    p_bootstrap.add_argument("--basic-size", type=int, default=4)
    p_bootstrap.add_argument("--rho-target", type=float, default=0.1)
    p_bootstrap.add_argument("--max-bars", type=int, default=10)

    p_kb_stats = sub.add_parser("kb-stats", help="show knowledge-base group statistics")
    p_kb_stats.add_argument("--kb-path", required=True)

    p_kb_sample = sub.add_parser("kb-sample", help="show one sample and its evidence bundle")
    p_kb_sample.add_argument("--kb-path", required=True)
    p_kb_sample.add_argument("--sample-id", required=True)

    p_kb_run = sub.add_parser("kb-run", help="show one run and its provenance bundle")
    p_kb_run.add_argument("--kb-path", required=True)
    p_kb_run.add_argument("--run-id", required=True)

    p_kb_query = sub.add_parser("kb-query", help="query samples from the knowledge base")
    p_kb_query.add_argument("--kb-path", required=True)
    p_kb_query.add_argument("--type", required=True, choices=["success", "near_miss", "failure", "similar"])
    p_kb_query.add_argument("--top-k", type=int, default=20)
    p_kb_query.add_argument("--group", default="")
    p_kb_query.add_argument("--reason-type", default="")
    p_kb_query.add_argument("--target-property", default="", help='JSON string, e.g. {"stiffness_proxy":25,"density_proxy":0.1}')

    p_train_export = sub.add_parser("export-training-dataset", help="export compact InverseDesigner training data")
    p_train_export.add_argument("--kb-path", required=True)
    p_train_export.add_argument("--output", default="")
    p_train_export.add_argument("--mark-used", action="store_true")

    p_refine = sub.add_parser("refine-knowledge", help="build agent-facing knowledge summaries from the evidence DB")
    p_refine.add_argument("--kb-path", required=True)
    p_refine.add_argument("--output", default="")
    p_refine.add_argument("--target-property", default="", help='optional JSON string, e.g. {"stiffness_proxy":25,"density_proxy":0.1}')

    return parser


def main():
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
        result = run_auto_generate_4x4x4(config.__dict__)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "batch":
        stop_on_failure = True
        if args.no_stop_on_failure:
            stop_on_failure = False
        elif args.stop_on_failure:
            stop_on_failure = True

        config = BatchGenerateConfig(
            workers=args.workers,
            samples=args.samples,
            basic_size=args.basic_size,
            rho_target=args.rho_target,
            max_bars=args.max_bars,
            poll_seconds=args.poll_seconds,
            idle_timeout_minutes=args.idle_timeout_minutes,
            group_timeout_minutes=args.group_timeout_minutes,
            stop_on_failure=stop_on_failure,
            include_groups=tuple(args.include_group),
            exclude_groups=tuple(args.exclude_group),
            group_db=args.group_db,
            output_root=args.output_root,
            batch_dir=args.batch_dir,
            resume=args.resume,
            allow_single_process_fallback=args.allow_single_process_fallback,
        )
        result = run_all_groups_4x4x4(config)
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

    if args.command == "discover":
        target_property = json.loads(args.target_property)
        result = run_closed_loop_discovery(
            target_property=target_property,
            workspace_root=args.workspace_root,
            kb_path=args.kb_path,
            max_iterations=args.max_iterations,
            retrain_trigger=args.retrain_trigger,
            log_path=args.log_path or None,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "bootstrap":
        groups = args.group or ["P222"]
        datagen_configs = [
            DatagenConfig(
                suggestion_id=f"bootstrap_{index:03d}_{group}",
                source="bootstrap_seed",
                group=group,
                symmetry=group,
                basic_size=args.basic_size,
                num_samples=args.samples,
                workers=args.workers,
                batch=args.batch,
                print_every=args.print_every,
                rho_target=args.rho_target,
                max_bars=args.max_bars,
                hypothesis="Bootstrap a finite exploratory base dataset.",
                reason="Create a regularized seed dataset and seed knowledge base for downstream search.",
                exploration_strategy="bootstrap_seed",
                tags=("bootstrap_seed", group),
            )
            for index, group in enumerate(groups, start=1)
        ]
        result = bootstrap_seed_dataset(
            datagen_configs=datagen_configs,
            workspace_root=args.workspace_root,
            kb_path=args.kb_path,
            output_dir=args.output_dir or None,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "kb-stats":
        print(json.dumps(kb_get_group_statistics(args.kb_path), ensure_ascii=False, indent=2))
        return

    if args.command == "kb-sample":
        result = kb_get_sample_evidence(args.kb_path, args.sample_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "kb-run":
        result = kb_get_run_provenance(args.kb_path, args.run_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "kb-query":
        target_property = json.loads(args.target_property) if args.target_property else None
        result = kb_query_samples(
            kb_path=args.kb_path,
            query_type=args.type,
            top_k=args.top_k,
            target_property=target_property,
            group=args.group or None,
            reason_type=args.reason_type or None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "export-training-dataset":
        result = export_inverse_designer_dataset(
            kb_path=args.kb_path,
            output_path=args.output or None,
            mark_used=args.mark_used,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "refine-knowledge":
        target_property = json.loads(args.target_property) if args.target_property else None
        result = refine_agent_knowledge(
            kb_path=args.kb_path,
            target_property=target_property,
            output_path=args.output or None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
