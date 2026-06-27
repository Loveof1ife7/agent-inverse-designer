#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键流水线（按群名）：
1) 关系推导并导出约束 JSON
2) 用约束 JSON 生成 architecture CSV
3) CSV -> Abaqus txt (node_data / element_conn)
4) 扩胞到“等效 4x4x4 基本域”尺寸

示例：
  python auto_generate_4x4x4.py P222
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path


# =========================================================
# USER CONFIG（优先改这里）
# =========================================================
# 你也可以在命令行覆盖这些默认值。
DEFAULT_GROUP = "P222"                       # 默认群名（也可命令行传入）
DEFAULT_BASIC_SIZE = 4                       # 目标基本域尺寸：4 => 4x4x4
DEFAULT_SAMPLES = 25000                      # 生成样本数
DEFAULT_WORKERS = 18                         # 并行 worker
DEFAULT_BATCH = 50                           # 每任务成功样本数
DEFAULT_PRINT_EVERY = 10                     # 进度打印间隔
DEFAULT_MAX_BARS = 10                        # base cube 内最多杆数
DEFAULT_RHO_TARGET = 0.1                     # 目标相对密度
DEFAULT_GROUP_DB = "symmetry_group_transforms.json"      # 群矩阵数据库
DEFAULT_RUN_DIR = ""                         # 为空则 <project>/workspace/<group>
DEFAULT_RESUME = False                       # 是否续跑
DEFAULT_ALLOW_SINGLE_PROCESS_FALLBACK = False  # 并行失败时是否降级单进程（默认否，强制并行）

# 脚本文件名（通常不需要改）
RELATION_SCRIPT = "constraints_solver.py"
GENERATOR_SCRIPT = "dataset_generator.py"
ABAQUS_SCRIPT = "abaqus_converter.py"
CRYSTAL_SCRIPT = "crystal_builder.py"


def load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_generator_module(root: Path, script_name: str):
    """
    关键：数据生成模块必须以“可导入模块名”加载，Windows spawn 才能在子进程反序列化 worker。
    """
    mod_name = Path(script_name).stem
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(mod_name)


def ensure_paths(root: Path):
    files = {
        "relation": root / RELATION_SCRIPT,
        "generator": root / GENERATOR_SCRIPT,
        "abaqus": root / ABAQUS_SCRIPT,
        "crystal": root / CRYSTAL_SCRIPT,
        "group_db": root / DEFAULT_GROUP_DB,
    }
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"缺少文件: {missing}")
    return files


def compute_replication_counts(lattice_lengths, basic_size: int):
    if not lattice_lengths or len(lattice_lengths) != 3:
        raise ValueError("lattice_lengths 无效，无法计算 4x4x4 等效扩胞倍数")

    counts = []
    for axis, L in zip(("x", "y", "z"), lattice_lengths):
        L = float(L)
        if L <= 0:
            raise ValueError(f"轴 {axis} 的周期长度无效: {L}")
        c = basic_size / L
        c_round = int(round(c))
        if abs(c - c_round) > 1e-9:
            raise ValueError(
                f"basic_size={basic_size} 与 lattice_lengths={lattice_lengths} 不整除，"
                f"axis={axis} 得到 {c}"
            )
        counts.append(c_round)
    return tuple(counts)


def run_constraints(relation_mod, group_name: str, group_db: Path, export_path: Path):
    payload = relation_mod.solve_and_visualize_constraints(
        group_name=group_name,
        db_path=str(group_db),
        export_path=str(export_path),
        show_plot=False,
    )
    if payload is None:
        raise RuntimeError("关系推导未返回可用 payload")
    return payload


def run_abaqus_conversion(aba_mod, csv_path: Path, out_dir: Path, group_name: str, group_db_path: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = aba_mod.TrussGenerator(
        str(csv_path),
        group_name=group_name,
        group_db_path=str(group_db_path),
    )
    total = len(gen.df)
    print(f"[ABAQUS] converting rows={total} -> {out_dir}")

    ok_count = 0
    for k in range(total):
        nodes, edges, name = gen.process_row(k)
        if nodes is None:
            continue
        gen.save_to_txt(str(out_dir / f"{k}.txt"), nodes, edges, name)
        ok_count += 1
        if k % 200 == 0:
            print(f"[ABAQUS] {k}/{total}")

    return ok_count


def run_crystal_expansion(crystal_mod, in_dir: Path, out_dir: Path, nx: int, ny: int, nz: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    err_log = out_dir / "errors.log"

    files = [p for p in in_dir.glob("*.txt")]
    files.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

    processed = 0
    failed = 0
    with err_log.open("w", encoding="utf-8") as elog:
        for idx, p in enumerate(files, 1):
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                node_data, element_conn, places = crystal_mod.parse_unitcell(txt)
                global_nodes, global_elems = crystal_mod.build_crystal(
                    node_data, element_conn, places, nx, ny, nz
                )
                out_text = crystal_mod.format_output(global_nodes, global_elems, p.stem)
                (out_dir / p.name).write_text(out_text, encoding="utf-8")
                processed += 1
                if idx % 200 == 0:
                    print(f"[CRYSTAL] {idx}/{len(files)}")
            except Exception as e:
                failed += 1
                elog.write(f"[FAIL] {p} -> {e}\n")

    return processed, failed, err_log


def parse_args():
    parser = argparse.ArgumentParser(description="按群名一键生成等效 4x4x4 基本域结构")
    parser.add_argument("group", nargs="?", default=DEFAULT_GROUP, help="群名，例如 P222 / Aba2 / Ccce")
    parser.add_argument(
        "--basic-size", type=int, default=DEFAULT_BASIC_SIZE, help="目标基本域尺寸 N（默认 4 表示 4x4x4）"
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="CSV 目标样本数")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="数据生成并行 worker 数")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="每个 worker 任务返回的成功样本数")
    parser.add_argument("--print-every", type=int, default=DEFAULT_PRINT_EVERY, help="数据生成打印间隔")
    parser.add_argument("--max-bars", type=int, default=DEFAULT_MAX_BARS, help="base cube 内最多杆数")
    parser.add_argument("--rho-target", type=float, default=DEFAULT_RHO_TARGET, help="目标相对密度")
    parser.add_argument("--group-db", default=DEFAULT_GROUP_DB, help="群矩阵 JSON 路径")
    parser.add_argument(
        "--run-dir",
        default=DEFAULT_RUN_DIR,
        help="输出目录（默认 <project>/workspace/<group>）",
    )
    parser.add_argument("--resume", action="store_true", default=DEFAULT_RESUME, help="CSV 若已存在则续跑")
    parser.add_argument(
        "--allow-single-process-fallback",
        action="store_true",
        default=DEFAULT_ALLOW_SINGLE_PROCESS_FALLBACK,
        help="并行失败时允许自动降级到单进程（默认关闭，保持强制并行）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    paths = ensure_paths(root)

    print(
        "[CONFIG] "
        f"group={args.group}, basic_size={args.basic_size}, samples={args.samples}, "
        f"workers={args.workers}, batch={args.batch}, print_every={args.print_every}, "
        f"max_bars={args.max_bars}, rho_target={args.rho_target}, "
        f"group_db={args.group_db}, resume={args.resume}, "
        f"allow_single_process_fallback={args.allow_single_process_fallback}"
    )

    default_out_root = root.parent.parent / "workspace"
    run_dir = Path(args.run_dir) if args.run_dir else (default_out_root / args.group)
    run_dir.mkdir(parents=True, exist_ok=True)

    constraints_json = run_dir / f"constraints_{args.group}.json"
    csv_name = f"{args.group}-architecture.csv"
    csv_path = run_dir / csv_name
    abaqus_dir = run_dir / "abaqus_txt"
    crystal_dir = run_dir / "crystal_4x4x4"
    summary_path = run_dir / "summary.json"

    relation_mod = load_module_from_path(paths["relation"], "relation_mod")
    gen_mod = load_generator_module(root, GENERATOR_SCRIPT)
    aba_mod = load_module_from_path(paths["abaqus"], "abaqus_mod")
    crystal_mod = load_module_from_path(paths["crystal"], "crystal_mod")

    group_db = Path(args.group_db)
    if not group_db.is_absolute():
        group_db = root / group_db

    print(f"[STEP1] solving constraints for group={args.group}")
    payload = run_constraints(relation_mod, args.group, group_db, constraints_json)
    lattice = payload.get("lattice_lengths")
    nx, ny, nz = compute_replication_counts(lattice, args.basic_size)
    print(f"[STEP1] lattice_lengths={lattice} => crystal replication (nx,ny,nz)=({nx},{ny},{nz})")

    print(f"[STEP2] generating csv samples={args.samples}")
    cfg = gen_mod.TrussConfig(
        OUTPUT_DIR=str(run_dir),
        CSV_NAME=csv_name,
        TARGET_SAMPLES=int(args.samples),
        RESUME_GENERATION=bool(args.resume),
        CONSTRAINTS_JSON=str(constraints_json),
        N_WORKERS=int(args.workers),
        BATCH_PER_TASK=int(args.batch),
        PRINT_EVERY=int(args.print_every),
        MAX_BARS=int(args.max_bars),
        RHO_TARGET=float(args.rho_target),
    )
    out_csv = Path(
        gen_mod.run_with_config(
            cfg,
            allow_single_process_fallback=bool(args.allow_single_process_fallback),
        )
    )
    print(f"[STEP2] csv saved: {out_csv}")

    print("[STEP3] converting csv -> abaqus txt")
    txt_count = run_abaqus_conversion(aba_mod, csv_path, abaqus_dir, args.group, group_db)
    print(f"[STEP3] txt files generated: {txt_count}")

    print(f"[STEP4] expanding crystal to equivalent {args.basic_size}x{args.basic_size}x{args.basic_size}")
    processed, failed, err_log = run_crystal_expansion(crystal_mod, abaqus_dir, crystal_dir, nx, ny, nz)
    print(f"[STEP4] expanded files: processed={processed}, failed={failed}")

    summary = {
        "group": args.group,
        "basic_size": args.basic_size,
        "lattice_lengths": lattice,
        "replication": {"nx": nx, "ny": ny, "nz": nz},
        "samples_target": args.samples,
        "max_bars": args.max_bars,
        "rho_target": args.rho_target,
        "run_dir": str(run_dir),
        "constraints_json": str(constraints_json),
        "csv_path": str(out_csv),
        "abaqus_txt_dir": str(abaqus_dir),
        "crystal_dir": str(crystal_dir),
        "abaqus_txt_count": txt_count,
        "crystal_processed": processed,
        "crystal_failed": failed,
        "crystal_error_log": str(err_log),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] summary: {summary_path}")


if __name__ == "__main__":
    main()
