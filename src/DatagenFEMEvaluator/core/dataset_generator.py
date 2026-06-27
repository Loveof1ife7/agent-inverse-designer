# -*- coding: utf-8 -*-
"""
P222 Truss dataset generator (FAST)
- 约束：
  1) base / unitcell / array(2x2x2) 连通（active nodes）
  2) unitcell 相对密度锁定 rho=0.1（通过 L 由 unitcell 总杆长反推）
  3) “无限周期晶体意义下”任意两杆（非共享节点）最短距离 >= MIN_BAR_CLEARANCE_PHYS
  4) unitcell 周期识别后 degree==1 剔除（internal + boundary，可开关）

- 加速：
  1) 多进程 ProcessPoolExecutor（Windows 需要 if __name__ == "__main__"）
  2) 每个任务批量生成 BATCH_PER_TASK 个成功样本，减少调度/序列化开销
  3) clearance 检查：对“新杆A”，只平移“旧杆集合”到最多 27 个周期像（通常远小于 27），不做 27^2
  4) AABB 下界距离使用向量化批量粗筛，只有少量候选才做精确线段-线段距离
  5) array 连通：Union-Find 合并边界节点，不生成阵列坐标

输出：CSV（你要求的表头与格式）
"""

import os
import csv
import uuid
import time
import random
import math
import hashlib
import json
import multiprocessing as mp
from dataclasses import dataclass, asdict
from itertools import combinations
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np
import sympy as sp


K_VAR_NAMES = [f"k_A{i}" for i in range(1, 13)]
Q_VAR_NAMES = [
    "p_fx", "p_fz", "p_bx", "p_bz",
    "p_ly", "p_lz", "p_ry", "p_rz",
    "p_tx", "p_ty", "p_btx", "p_bty",
]


# =========================================================
# 1) 配置
# =========================================================
@dataclass(frozen=True)
class TrussConfig:
    # 输出
    OUTPUT_DIR: str = r"C:\Users\admin\Desktop\3Dtruss\Aba2"
    CSV_NAME: str = "Aba2-architecture.csv"

    TARGET_SAMPLES: int = 25000
    RESUME_GENERATION: bool = True

    # base cube 内最多杆数
    MAX_BARS: int = 10

    # 密度/几何约束（注意：clearance 是“无限周期晶体意义”）
    RHO_TARGET: float = 0.1
    R_PHYSICAL: float = 1.0
    MIN_BAR_CLEARANCE_PHYS: float = 4.0

    # 离散取值
    K_OPTIONS: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    P_OPTIONS: tuple = (0.25, 0.5, 0.75)
    # 关系接口（可选）：来自 关系最终版.py --export
    # 为空字符串时使用 _get_random_params 里的默认硬编码关系
    CONSTRAINTS_JSON: str = ""

    # 数值
    TOLERANCE: float = 1e-5
    GENERATION_RETRIES: int = 800

    # 阵列验收尺寸
    NX: int = 2
    NY: int = 2
    NZ: int = 2

    # 连通判据
    REQUIRE_ALL_NODES_CONNECTED: bool = False  # False=只要求 active nodes 连通

    # 周期识别后 degree==1 剔除（只判 ==1，不判 <=1）
    REJECT_INTERNAL_DEGREE1_AFTER_PBC: bool = True
    REJECT_BOUNDARY_DEGREE1_AFTER_PBC: bool = True

    # 多进程参数（你要 15）
    N_WORKERS: int = 15
    TASKS_IN_FLIGHT_PER_WORKER: int = 2
    BATCH_PER_TASK: int = 50            # 每个进程任务返回多少“成功样本”（20~100 通常都行）
    CSV_FLUSH_EVERY: int = 200          # main 进程累积多少行写一次（I/O 加速）
    CSV_WRITE_RETRIES: int = 12         # CSV 写入失败重试次数（处理临时文件占用）
    CSV_WRITE_RETRY_DELAY: float = 0.5  # 每次重试基础等待秒数（线性退避）

    PRINT_EVERY: int = 10               # 进度打印（按写入后的 id）
    MAX_NO_PROGRESS_BATCHES: int = 500  # 连续多少批无新增样本则判定卡住
    MAX_NO_PROGRESS_SECONDS: int = 900  # 无新增样本超时（秒）


# =========================================================
# 2) 距离工具：线段-线段最短距离平方 + AABB 下界距离平方（含向量化）
# =========================================================
def min_distance2_segment_segment(p1, p2, p3, p4) -> float:
    """3D 线段-线段最短距离平方（避免 sqrt）"""
    u = p2 - p1
    v = p4 - p3
    w = p1 - p3
    a = float(np.dot(u, u)); b = float(np.dot(u, v)); c = float(np.dot(v, v))
    d = float(np.dot(u, w)); e = float(np.dot(v, w))
    D = a * c - b * b

    sc, sN, sD = 0.0, D, D
    tc, tN, tD = 0.0, D, D

    if D < 1e-12:
        sN, sD = 0.0, 1.0
        tN, tD = e, c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if sN < 0.0:
            sN, tN, tD = 0.0, e, c
        elif sN > sD:
            sN, tN, tD = sD, e + b, c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN, sD = -d, a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN, sD = (-d + b), a

    sc = 0.0 if abs(sN) < 1e-12 else sN / sD
    tc = 0.0 if abs(tN) < 1e-12 else tN / tD
    dP = w + sc * u - tc * v
    return float(np.dot(dP, dP))


def seg_aabb(p1, p2):
    mn = np.minimum(p1, p2)
    mx = np.maximum(p1, p2)
    return mn, mx


def aabb_distance2_vec(mn1, mx1, mn2_arr, mx2_arr):
    """
    AABB 到 AABB 的最短距离平方（下界），向量化：
      mn1,mx1: (3,)
      mn2_arr,mx2_arr: (m,3)
      返回: (m,) 的距离平方下界
    """
    # d = max(0, max(mn2-mx1, mn1-mx2))
    d = np.maximum(0.0, np.maximum(mn2_arr - mx1, mn1 - mx2_arr))
    return np.sum(d * d, axis=1)


def _default_p222_mats_norm():
    # 与历史实现保持一致：x 周期 1，y/z 周期 2（坐标区间可含负值）
    return [
        np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float64),
        np.array([[-1, 0, 0, 1],
                  [0, -1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float64),
        np.array([[-1, 0, 0, 1],
                  [0, 1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]], dtype=np.float64),
        np.array([[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]], dtype=np.float64),
    ]


def _make_symmetry_spec(op_mats_norm, lattice_lengths, group_name="P222"):
    mats = []
    if op_mats_norm:
        mats = [np.array(m, dtype=np.float64) for m in op_mats_norm]
    if not mats:
        mats = _default_p222_mats_norm()

    ll = np.array(lattice_lengths if lattice_lengths is not None else [1.0, 2.0, 2.0], dtype=np.float64)
    if ll.shape != (3,) or np.any(ll <= 0):
        ll = np.array([1.0, 2.0, 2.0], dtype=np.float64)

    corners = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    all_pts = []
    for m in mats:
        A = m[:3, :3]
        t = m[:3, 3]
        all_pts.append(corners @ A.T + t[None, :])
    all_pts = np.vstack(all_pts)
    bounds_min = np.min(all_pts, axis=0)
    bounds_max = np.max(all_pts, axis=0)

    return {
        "group_name": str(group_name),
        "op_mats_norm": mats,
        "lattice_lengths": ll,
        "bounds_norm": [bounds_min, bounds_max],
        "volume_norm": float(np.prod(ll)),
    }


def _load_symmetry_spec_from_payload(payload: dict, constraints_path: Path, strict: bool = False):
    group_name = payload.get("group_name", "P222")
    lattice_lengths = payload.get("lattice_lengths")

    # 1) 若 payload 已带矩阵，优先使用
    mats = payload.get("symmetry_matrices")
    if mats:
        return _make_symmetry_spec(mats, lattice_lengths, group_name)

    # 2) 否则尝试从群数据库回读
    db_path_raw = payload.get("group_db_path", "")
    if db_path_raw:
        db_path = Path(db_path_raw)
        if not db_path.is_absolute():
            db_path = constraints_path.parent / db_path
        if not db_path.exists():
            db_path = Path(__file__).resolve().parent / db_path_raw
        if db_path.exists():
            with db_path.open("r", encoding="utf-8") as f:
                db = json.load(f)
            groups = db.get("groups", {})
            gd = groups.get(group_name, {})
            mats = gd.get("M_sym")
            if lattice_lengths is None:
                lattice_lengths = gd.get("lattice_lengths")
            if mats:
                return _make_symmetry_spec(mats, lattice_lengths, group_name)

    # 3) 兜底：保持历史 P222 行为
    if strict:
        raise RuntimeError(
            f"无法为 group='{group_name}' 解析群矩阵。"
            f"请检查 constraints/group_db_path/M_sym 是否完整。"
        )
    return _make_symmetry_spec(None, None, "P222")


# =========================================================
# 3) 几何生成器（base 19 点，归一化 0~1）
# =========================================================
class ConstraintSampler:
    def __init__(self, cfg: TrussConfig):
        self.cfg = cfg
        self.enabled = False
        self.payload = None
        self.free_vars = []
        self.all_vars = []
        self.symbols = {}
        self.expr_map = {}
        self.symmetry_spec = _make_symmetry_spec(None, None, "P222")

        if not cfg.CONSTRAINTS_JSON:
            return

        p = Path(cfg.CONSTRAINTS_JSON)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        if not p.exists():
            raise FileNotFoundError(f"找不到 CONSTRAINTS_JSON: {p}")

        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        solved = payload.get("solved", {})
        vars_all = payload.get("variables", {}).get("all", [])
        free_all = payload.get("free_vars", {}).get("all", [])

        if not isinstance(solved, dict):
            raise ValueError("约束文件格式错误：'solved' 必须是对象(dict)")

        # 兜底：如果没给完整变量列表，就从 solved + free 推断
        if not vars_all:
            vars_all = sorted(set(list(solved.keys()) + list(free_all)))
        if not free_all:
            free_all = sorted([v for v in vars_all if v not in solved])

        self.payload = payload
        self.all_vars = list(vars_all)
        self.free_vars = list(free_all)
        self.symbols = {name: sp.Symbol(name) for name in self.all_vars}

        # 解析 solved 表达式
        for lhs, rhs_str in solved.items():
            if lhs not in self.symbols:
                self.symbols[lhs] = sp.Symbol(lhs)
                self.all_vars.append(lhs)
            expr = sp.sympify(rhs_str, locals=self.symbols)
            self.expr_map[lhs] = expr
            for sym in expr.free_symbols:
                nm = str(sym)
                if nm not in self.symbols:
                    self.symbols[nm] = sp.Symbol(nm)
                if nm not in self.all_vars:
                    self.all_vars.append(nm)

        self.enabled = True
        group_name = payload.get("group_name", "<unknown>")
        self.symmetry_spec = _load_symmetry_spec_from_payload(payload, p, strict=True)
        print(f"[CONSTRAINT] loaded: group={group_name} file={p}")

    def _sample_domain_value(self, var_name: str) -> float:
        if var_name.startswith("k_"):
            return float(random.choice(self.cfg.K_OPTIONS))
        if var_name.startswith("p_"):
            return float(random.choice(self.cfg.P_OPTIONS))
        raise ValueError(f"不支持的变量名: {var_name}")

    def sample(self) -> dict:
        if not self.enabled:
            return {}

        vals = {}
        # 1) 先采自由变量
        for name in self.free_vars:
            vals[name] = self._sample_domain_value(name)

        # 2) 再根据 solved 递推求值
        pending = dict(self.expr_map)
        max_rounds = max(5, len(pending) + 5)
        for _ in range(max_rounds):
            if not pending:
                break
            progressed = False
            for lhs, expr in list(pending.items()):
                deps = [str(s) for s in expr.free_symbols]
                if all(dep in vals for dep in deps):
                    subs = {self.symbols[d]: vals[d] for d in deps}
                    v = float(sp.N(expr.subs(subs)))
                    if abs(v) < 1e-12:
                        v = 0.0
                    vals[lhs] = v
                    del pending[lhs]
                    progressed = True
            if not progressed:
                break

        if pending:
            left = ", ".join(sorted(pending.keys()))
            raise RuntimeError(f"约束表达式存在未解依赖: {left}")

        # 3) 若仍有未覆盖变量，按域随机（防止约束文件不完整）
        for name in self.all_vars:
            if name not in vals:
                vals[name] = self._sample_domain_value(name)

        return vals


class GeometryGenerator:
    def __init__(self, cfg: TrussConfig):
        self.cfg = cfg
        self.constraint_sampler = ConstraintSampler(cfg)
        self.symmetry_spec = self.constraint_sampler.symmetry_spec
        self.node_names_ordered = [
            'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
            'A9', 'A10', 'A11', 'A12',
            'q_front', 'q_back', 'q_left', 'q_right', 'q_top', 'q_bottom',
            'v_center'
        ]

    def _get_random_params(self):
        cfg = self.cfg
        p = {}
        if self.constraint_sampler.enabled:
            p.update(self.constraint_sampler.sample())
        else:
            # ===================== 【PATCH_001_BEGIN：默认硬编码关系】 =====================
            # k 约束（独立变量：A3,A2,A5,A6,A9,A10；其余由等式绑定）
            p['k_A3']  = random.choice(cfg.K_OPTIONS)
            p['k_A1']  = p['k_A3']                 # k_A1 = k_A3

            p['k_A2']  = random.choice(cfg.K_OPTIONS)
            p['k_A4']  = p['k_A2']                 # k_A2 = k_A4

            p['k_A5']  = random.choice(cfg.K_OPTIONS)
            p['k_A7']  = p['k_A5']                 # k_A5 = k_A7

            p['k_A6']  = random.choice(cfg.K_OPTIONS)
            p['k_A8']  = p['k_A6']                 # k_A6 = k_A8

            p['k_A9']  = random.choice(cfg.K_OPTIONS)
            p['k_A11'] = p['k_A9']                 # k_A9 = k_A11

            p['k_A10'] = random.choice(cfg.K_OPTIONS)
            p['k_A12'] = p['k_A10']                # k_A10 = k_A12

            # p 约束（独立变量：p_bx,p_bz,p_ry,p_rz；其余由等式绑定）
            p['p_bx']  = random.choice(cfg.P_OPTIONS)
            p['p_fx']  = 1.0 - p['p_bx']           # p_fx = 1 - p_bx

            p['p_bz']  = random.choice(cfg.P_OPTIONS)
            p['p_fz']  = p['p_bz']                 # p_fz = p_bz

            p['p_ry']  = random.choice(cfg.P_OPTIONS)
            p['p_ly']  = 1.0 - p['p_ry']           # p_ly = 1 - p_ry

            p['p_rz']  = random.choice(cfg.P_OPTIONS)
            p['p_lz']  = p['p_rz']                 # p_lz = p_rz

            # 其他未指定的面内参数：保持原先固定值（可按需再放开）
            p['p_btx'] = 0.5
            p['p_tx']  = 0.5
            p['p_ty']  = 0.5
            p['p_bty'] = 0.5
            # ===================== 【PATCH_001_END】 =====================

        # 关系文件如果没覆盖完整变量，这里补齐
        for name in K_VAR_NAMES:
            if name not in p:
                p[name] = float(random.choice(cfg.K_OPTIONS))
        for name in Q_VAR_NAMES:
            if name not in p:
                p[name] = float(random.choice(cfg.P_OPTIONS))

        p['v_x'] = random.choice(cfg.P_OPTIONS)
        p['v_y'] = random.choice(cfg.P_OPTIONS)
        p['v_z'] = random.choice(cfg.P_OPTIONS)
        return p

    def generate_valid_geometry(self):
        tol = self.cfg.TOLERANCE
        for _ in range(200):
            p = self._get_random_params()
            k = lambda n: p[f'k_{n}']

            raw = {
                'A1': np.array([1 - k('A1'), 0, 1]), 'A2': np.array([0, k('A2'), 1]),
                'A3': np.array([k('A3'), 1, 1]),     'A4': np.array([1, 1 - k('A4'), 1]),
                'A5': np.array([1 - k('A5'), 0, 0]), 'A6': np.array([0, k('A6'), 0]),
                'A7': np.array([k('A7'), 1, 0]),     'A8': np.array([1, 1 - k('A8'), 0]),
                'A9': np.array([1, 0, 1 - k('A9')]), 'A10': np.array([0, 0, 1 - k('A10')]),
                'A11': np.array([0, 1, 1 - k('A11')]), 'A12': np.array([1, 1, 1 - k('A12')]),

                'q_front': np.array([p['p_fx'], 0, p['p_fz']]),
                'q_back':  np.array([p['p_bx'], 1, p['p_bz']]),
                'q_left':  np.array([0, p['p_ly'], p['p_lz']]),
                'q_right': np.array([1, p['p_ry'], p['p_rz']]),
                'q_top':   np.array([p['p_tx'], p['p_ty'], 1]),
                'q_bottom':np.array([p['p_btx'], p['p_bty'], 0]),

                'v_center': np.array([p['v_x'], p['v_y'], p['v_z']]),
            }

            nodes = np.array([raw[n] for n in self.node_names_ordered], dtype=np.float64)

            # 点不重合
            ok = True
            for i in range(len(nodes)):
                for j in range(i):
                    if np.linalg.norm(nodes[i] - nodes[j]) < tol:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return nodes

        return None


# =========================================================
# 4) 节点合并与边重映射（物理坐标）
# =========================================================
def clean_and_merge(nodes, edges, tol):
    if len(nodes) == 0:
        return nodes, edges

    decimals = max(0, int(-np.log10(tol)))
    rounded = np.round(nodes, decimals=decimals)

    _, unique_idx, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    nodes_new = nodes[unique_idx]

    edges_new = inverse[np.asarray(edges, dtype=int)]
    edges_new = np.sort(edges_new, axis=1)
    edges_new = edges_new[edges_new[:, 0] != edges_new[:, 1]]
    edges_new = np.unique(edges_new, axis=0)
    return nodes_new, edges_new


# =========================================================
# 5) 通用对称（物理坐标）
# =========================================================
def apply_symmetry_ops(nodes, edges, sym_spec, L, tol):
    mats = sym_spec["op_mats_norm"]
    all_nodes = []
    all_edges = []
    offset = 0

    for m in mats:
        A = m[:3, :3]
        t = m[:3, 3] * float(L)
        nt = nodes @ A.T + t[None, :]
        all_nodes.append(nt)
        all_edges.append(edges + offset)
        offset += len(nodes)

    nodes_uc = np.vstack(all_nodes)
    edges_uc = np.vstack(all_edges)
    return clean_and_merge(nodes_uc, edges_uc, tol)


# =========================================================
# 6) 连通性（active nodes）
# =========================================================
def is_connected(num_nodes, edges, require_all_nodes=False) -> bool:
    if len(edges) == 0:
        return False

    deg = np.zeros(num_nodes, dtype=int)
    adj = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1

    if require_all_nodes:
        if np.any(deg == 0):
            return False
        active = np.arange(num_nodes, dtype=int)
    else:
        active = np.where(deg > 0)[0]
        if len(active) == 0:
            return False

    start = int(active[0])
    visited = set([start])
    dq = deque([start])
    while dq:
        x = dq.popleft()
        for y in adj[x]:
            if y not in visited:
                visited.add(y)
                dq.append(y)

    return all(int(n) in visited for n in active)


# =========================================================
# 7) 周期识别(PBC)后的 unit cell：节点合并 + 边去重 + 自环保留
# =========================================================
def periodic_identify_unitcell(nodes_uc, edges_uc, sym_spec, L, tol):
    periods = sym_spec["lattice_lengths"] * float(L)
    bmin = np.array(sym_spec["bounds_norm"][0], dtype=np.float64) * float(L)
    bmax = np.array(sym_spec["bounds_norm"][1], dtype=np.float64) * float(L)
    wrapped = np.empty_like(nodes_uc)

    x = nodes_uc[:, 0]
    y = nodes_uc[:, 1]
    z = nodes_uc[:, 2]
    is_boundary = (
        (np.abs(x - bmin[0]) <= tol) | (np.abs(x - bmax[0]) <= tol) |
        (np.abs(y - bmin[1]) <= tol) | (np.abs(y - bmax[1]) <= tol) |
        (np.abs(z - bmin[2]) <= tol) | (np.abs(z - bmax[2]) <= tol)
    )
    for d in range(3):
        P = float(periods[d])
        rem = np.mod(nodes_uc[:, d] - bmin[d], P)
        wrapped[:, d] = rem

    decimals = max(0, int(-np.log10(tol)))
    wrapped_r = np.round(wrapped, decimals=decimals)

    _, unique_idx, inverse = np.unique(wrapped_r, axis=0, return_index=True, return_inverse=True)
    nodes_pbc = wrapped[unique_idx]

    is_boundary_pbc = np.zeros(len(nodes_pbc), dtype=bool)
    for i_old, i_new in enumerate(inverse):
        if is_boundary[i_old]:
            is_boundary_pbc[i_new] = True

    e = inverse[np.asarray(edges_uc, dtype=int)]
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)  # 保留自环（degree 会 +2）

    return nodes_pbc, e, is_boundary_pbc


def reject_degree1_after_pbc(nodes_pbc, edges_pbc, is_boundary_pbc,
                            reject_internal=True, reject_boundary=False) -> bool:
    n = len(nodes_pbc)
    if n == 0 or len(edges_pbc) == 0:
        return True

    deg = np.bincount(edges_pbc.reshape(-1), minlength=n)

    internal = ~is_boundary_pbc
    boundary = is_boundary_pbc

    bad = np.zeros(n, dtype=bool)
    if reject_internal:
        bad |= internal & (deg == 1)
    if reject_boundary:
        bad |= boundary & (deg == 1)

    return bool(np.any(bad))


# =========================================================
# 8) unit cell 密度锁定：由总杆长反推 L
# =========================================================
def estimate_L_unitcell(S_uc_norm, r, rho, volume_norm_uc):
    # rho = pi r^2 * (L * S_uc_norm) / (L^3 * V) = pi r^2 S_uc_norm / (L^2 V)
    # => L = r * sqrt( pi*S_uc_norm / (rho*V_NORM_UC) )
    return float(r * math.sqrt((math.pi * S_uc_norm) / (rho * float(volume_norm_uc))))


# =========================================================
# 9) 对称缓存（norm）+ 稀疏选边（核心：无限周期 clearance 检查）
# =========================================================
class SymmetryCache:
    """
    norm 下按群矩阵构建 unit cell，并合并节点，得到：
      - uc_nodes_norm: 合并后的 unit cell 节点（norm）
      - maps[s][i]: 第 s 个对称副本(base->uc) 的节点索引映射
    """
    def __init__(self, base_nodes_norm, tol, op_mats_norm, lattice_lengths):
        self.n = len(base_nodes_norm)
        self.n_ops = len(op_mats_norm)

        transformed = []
        for m in op_mats_norm:
            A = m[:3, :3]
            t = m[:3, 3]
            nt = base_nodes_norm @ A.T + t[None, :]
            transformed.append(nt)
        stacked = np.vstack(transformed)

        decimals = max(0, int(-np.log10(tol)))
        rounded = np.round(stacked, decimals=decimals)

        _, unique_idx, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
        self.uc_nodes_norm = stacked[unique_idx]
        self.inverse = inverse

        self.maps = []
        for s in range(self.n_ops):
            self.maps.append(self.inverse[s*self.n:(s+1)*self.n])


def generate_sparse_edges_on_base(nodes_norm, cfg: TrussConfig, sym_spec):
    """
    返回：
      base_edges: (m,2) base 内边（节点索引 0..18）
      S_uc_norm : unit cell（合并后、去重后）总归一化杆长，用于锁定 L

    clearance 严格按“无限周期晶体”检查：
      - 对新杆A，计算它的 AABB 是否靠近周期边界
      - 仅对需要的方向生成 shifts（1~27 个，通常远小于 27）
      - 将“旧杆集合”平移这些 shifts，与 A 做比较（不做 27^2）
      - AABB 下界距离向量化粗筛，只有少量候选做精确线段距离
    """
    n = len(nodes_norm)
    max_bars = cfg.MAX_BARS
    if max_bars < 1:
        return None, None

    tol = cfg.TOLERANCE
    tol2 = tol * tol
    decimals = max(0, int(-np.log10(tol)))

    periods_norm = np.array(sym_spec["lattice_lengths"], dtype=np.float64)
    cache = SymmetryCache(nodes_norm, tol, sym_spec["op_mats_norm"], periods_norm)
    uc_nodes = cache.uc_nodes_norm

    eps = 1e-12
    x0 = [i for i in range(n) if abs(nodes_norm[i, 0] - 0.0) < eps]
    x1 = [i for i in range(n) if abs(nodes_norm[i, 0] - 1.0) < eps]
    y0 = [i for i in range(n) if abs(nodes_norm[i, 1] - 0.0) < eps]
    y1 = [i for i in range(n) if abs(nodes_norm[i, 1] - 1.0) < eps]
    z0 = [i for i in range(n) if abs(nodes_norm[i, 2] - 0.0) < eps]
    z1 = [i for i in range(n) if abs(nodes_norm[i, 2] - 1.0) < eps]
    if not x0 or not x1:
        return None, None

    # used 节点数 k：树边至少 k-1 <= max_bars
    k_max = min(n, max_bars + 1)
    k_min = 4
    if k_max < k_min:
        k_min = max(2, k_max)
    k = random.randint(k_min, k_max)

    # 轻微偏置：确保 x=0 与 x=1 各至少一个（提高周期连通成功率）
    used = set()
    used.add(random.choice(x0))
    used.add(random.choice(x1))

    # 适度抽一点边界（提高成功率；不“避免 degree==1”，只提高连通成功率）
    boundary_groups = [y0, y1, z0, z1]
    random.shuffle(boundary_groups)
    for grp in boundary_groups:
        if len(used) >= k:
            break
        if grp:
            used.add(random.choice(grp))

    idxs = list(range(n))
    random.shuffle(idxs)
    for idx in idxs:
        if len(used) >= k:
            break
        used.add(idx)

    used = sorted(list(used))

    # unit cell 边集合（去重）
    edge_uc_set = set()
    # 几何缓存（旧边）
    p1_old = np.empty((0, 3), dtype=np.float64)
    p2_old = np.empty((0, 3), dtype=np.float64)
    mn_old = np.empty((0, 3), dtype=np.float64)
    mx_old = np.empty((0, 3), dtype=np.float64)

    S_uc_norm = 0.0

    def endpoints_share(p1, p2, q1, q2):
        return (
            np.dot(p1-q1, p1-q1) <= tol2 or np.dot(p1-q2, p1-q2) <= tol2 or
            np.dot(p2-q1, p2-q1) <= tol2 or np.dot(p2-q2, p2-q2) <= tol2
        )

    def add_uc_edges_from_base_edge(u, v):
        new_edges = []
        for s in range(cache.n_ops):
            uu = int(cache.maps[s][u])
            vv = int(cache.maps[s][v])
            if uu == vv:
                continue
            e = (uu, vv) if uu < vv else (vv, uu)
            if e not in edge_uc_set:
                new_edges.append(e)
        return new_edges

    bounds_min = np.array(sym_spec["bounds_norm"][0], dtype=np.float64)
    bounds_max = np.array(sym_spec["bounds_norm"][1], dtype=np.float64)

    # 根据新杆 AABB 靠近边界情况，决定要检查哪些周期像（轴向周期）
    def shifts_for_aabb(mn, mx, limit_norm):
        s_lists = []
        for d in range(3):
            P = float(periods_norm[d])
            lo = float(bounds_min[d])
            hi = float(bounds_max[d])
            lst = [0.0]
            if (mn[d] - lo) < limit_norm:
                lst.append(-P)
            if (hi - mx[d]) < limit_norm:
                lst.append(+P)
            lst = list(dict.fromkeys(lst))
            s_lists.append(lst)

        shifts = []
        for sx in s_lists[0]:
            for sy in s_lists[1]:
                for sz in s_lists[2]:
                    shifts.append(np.array([sx, sy, sz], dtype=np.float64))
        return shifts

    def can_add_base_edge(u, v):
        nonlocal S_uc_norm, p1_old, p2_old, mn_old, mx_old

        new_uc_edges = add_uc_edges_from_base_edge(u, v)
        if not new_uc_edges:
            return False

        # 先构造新边几何（最多 4 条）
        new_edges_geom = []
        add_len = 0.0
        for (a, b) in new_uc_edges:
            p1 = uc_nodes[a]
            p2 = uc_nodes[b]
            el = float(np.linalg.norm(p1 - p2))
            if el < 1e-12:
                continue
            mn, mx = seg_aabb(p1, p2)
            add_len += el
            new_edges_geom.append((a, b, p1, p2, mn, mx))

        if add_len < 1e-12:
            return False

        # 用 “加边后的总杆长” 反推 L，从而得到 limit_norm
        S_new = S_uc_norm + add_len
        L_est = estimate_L_unitcell(S_new, cfg.R_PHYSICAL, cfg.RHO_TARGET, sym_spec["volume_norm"])
        if L_est < 1e-12:
            return False

        limit_norm = cfg.MIN_BAR_CLEARANCE_PHYS / L_est
        limit2 = limit_norm * limit_norm

        m_old = len(p1_old)

        # ---------- new vs old（仅平移 old，最多 27 个周期像；AABB 向量化粗筛） ----------
        if m_old > 0:
            for (_, _, p1, p2, mn1, mx1) in new_edges_geom:
                shifts = shifts_for_aabb(mn1, mx1, limit_norm)
                for sh in shifts:
                    # 平移 old 的 AABB
                    mn2s = mn_old + sh
                    mx2s = mx_old + sh

                    # AABB 下界向量化粗筛
                    d2_lb = aabb_distance2_vec(mn1, mx1, mn2s, mx2s)
                    cand = np.nonzero(d2_lb < limit2)[0]
                    if cand.size == 0:
                        continue

                    # 精确距离（只对少量 cand）
                    for idx in cand.tolist():
                        q1 = p1_old[idx] + sh
                        q2 = p2_old[idx] + sh
                        if endpoints_share(p1, p2, q1, q2):
                            continue
                        d2 = min_distance2_segment_segment(p1, p2, q1, q2)
                        if d2 < limit2:
                            return False

        # ---------- new within new（最多 4 条，量很小，仍按 shifts 做） ----------
        ln = len(new_edges_geom)
        if ln >= 2:
            for i in range(ln):
                _, _, p1, p2, mn1, mx1 = new_edges_geom[i]
                shifts = shifts_for_aabb(mn1, mx1, limit_norm)
                for j in range(i):
                    _, _, q1, q2, mn2, mx2 = new_edges_geom[j]
                    for sh in shifts:
                        q1s = q1 + sh
                        q2s = q2 + sh
                        if endpoints_share(p1, p2, q1s, q2s):
                            continue
                        mn2s = mn2 + sh
                        mx2s = mx2 + sh
                        # 先做 AABB 下界（标量）
                        d = np.maximum(0.0, np.maximum(mn2s - mx1, mn1 - mx2s))
                        if float(np.dot(d, d)) >= limit2:
                            continue
                        d2 = min_distance2_segment_segment(p1, p2, q1s, q2s)
                        if d2 < limit2:
                            return False

        # commit：把新边写入 old-cache
        for (a, b, p1, p2, mn, mx) in new_edges_geom:
            e = (a, b) if a < b else (b, a)
            if e in edge_uc_set:
                continue
            edge_uc_set.add(e)
            S_uc_norm += float(np.linalg.norm(p1 - p2))

            p1_old = np.vstack([p1_old, p1[None, :]])
            p2_old = np.vstack([p2_old, p2[None, :]])
            mn_old = np.vstack([mn_old, mn[None, :]])
            mx_old = np.vstack([mx_old, mx[None, :]])

        return True

    # 先造生成树：used 子图连通（active nodes）
    base_edges = []
    order = used[:]
    random.shuffle(order)
    connected = [order[0]]
    remaining = order[1:]

    while remaining:
        v = remaining.pop(0)
        cand = connected[:]
        random.shuffle(cand)
        linked = False
        for u in cand:
            if len(base_edges) >= max_bars:
                break
            if can_add_base_edge(u, v):
                base_edges.append([u, v])
                connected.append(v)
                linked = True
                break
        if not linked:
            return None, None

    # 再补边到 <= max_bars
    target = random.randint(len(base_edges), max_bars)
    base_set = set(tuple(sorted(e)) for e in base_edges)

    cand_edges = []
    for (i, j) in combinations(used, 2):
        key = (i, j) if i < j else (j, i)
        if key in base_set:
            continue
        cand_edges.append((i, j))
    random.shuffle(cand_edges)

    for (u, v) in cand_edges:
        if len(base_edges) >= target:
            break
        if can_add_base_edge(u, v):
            base_edges.append([u, v])
            base_set.add((u, v) if u < v else (v, u))

    return np.asarray(base_edges, dtype=int), float(S_uc_norm)


# =========================================================
# 10) CSV 管理（主进程写：批量写入 + flush）
# =========================================================
def get_last_id_fast(filepath: str) -> int:
    if not os.path.exists(filepath):
        return -1

    with open(filepath, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return -1

        chunk = 4096
        pos = size
        data = b""
        while pos > 0:
            step = min(chunk, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
            if data.count(b"\n") >= 2:
                break

        lines = data.splitlines()
        if len(lines) < 2:
            return -1

        last = lines[-1].decode("utf-8", errors="ignore").strip()
        if not last:
            return -1
        try:
            return int(last.split(",")[0])
        except Exception:
            last_id = -1
            with open(filepath, "r", encoding="utf-8") as fr:
                reader = csv.reader(fr)
                next(reader, None)
                for row in reader:
                    if row:
                        last_id = int(row[0])
            return last_id


class CSVManager:
    def __init__(self, cfg: TrussConfig, node_names):
        self.cfg = cfg
        self.node_names = node_names
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        self.filepath = os.path.join(cfg.OUTPUT_DIR, cfg.CSV_NAME)

        self.num_nodes = len(node_names)
        self.num_elements = self.num_nodes * self.num_nodes  # 19*19=361

        if (not cfg.RESUME_GENERATION) or (not os.path.exists(self.filepath)):
            self._write_header()
            self.start_index = 0
            print(f"[CSV] create: {self.filepath}")
        else:
            last_id = get_last_id_fast(self.filepath)
            self.start_index = last_id + 1
            print(f"[CSV] resume: start_id={self.start_index}  file={self.filepath}")

        # 追加模式长期打开，减少 open/close
        self._fh = open(self.filepath, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)

    def close(self):
        try:
            self._fh.flush()
        finally:
            self._fh.close()

    def _write_header(self):
        header = ["id", "name"]
        for nm in self.node_names:
            header.extend([f"{nm}_x", f"{nm}_y", f"{nm}_z"])
        for i in range(self.num_elements):
            header.append(f"element_{i+1}")
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    def append_rows(self, rows):
        retries = max(1, int(self.cfg.CSV_WRITE_RETRIES))
        delay = max(0.01, float(self.cfg.CSV_WRITE_RETRY_DELAY))

        for i in range(retries):
            try:
                self._writer.writerows(rows)
                self._fh.flush()
                return
            except PermissionError:
                # Windows 上文件被短暂占用时，重开句柄并重试。
                if i == retries - 1:
                    raise
                try:
                    self._fh.close()
                except Exception:
                    pass
                time.sleep(delay * (i + 1))
                self._fh = open(self.filepath, "a", newline="", encoding="utf-8")
                self._writer = csv.writer(self._fh)

def adj_digest_from_flat(adj_flat) -> int:
    """
    只基于邻接关系去重：
    - adj_flat: 361个0/1
    返回一个稳定的 128-bit 整数摘要（极低碰撞概率）。
    """
    b = bytes(adj_flat)  # adj_flat 里是 0/1，合法
    return int.from_bytes(hashlib.blake2b(b, digest_size=16).digest(), "little")


def load_seen_adj_digests(filepath: str, adj_start: int, adj_len: int) -> set:
    """
    如果 RESUME_GENERATION=True，为了避免和历史 CSV 重复，
    启动时扫描一次已有 CSV，把历史邻接关系摘要装进 seen。
    """
    seen = set()
    if not os.path.exists(filepath):
        return seen

    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        adj_end = adj_start + adj_len

        for row in reader:
            if len(row) < adj_end:
                continue
            # row 里是字符串 '0'/'1'，快速转成 bytes 再 hash
            b = bytes(1 if v == "1" else 0 for v in row[adj_start:adj_end])
            d = int.from_bytes(hashlib.blake2b(b, digest_size=16).digest(), "little")
            seen.add(d)

    return seen

# =========================================================
# 11) array 连通性：Union-Find 合并边界节点（不生成阵列坐标）
# =========================================================
class DSU:
    __slots__ = ("p", "sz")
    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int32)
        self.sz = np.ones(n, dtype=np.int32)
    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x
    def union(self, a, b):
        ra = self.find(a); rb = self.find(b)
        if ra == rb:
            return ra
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]
        return ra


def is_array_connected_fast(uc_nodes, uc_edges, sym_spec, L, nx, ny, nz, tol, require_all_nodes=False) -> bool:
    n_uc = len(uc_nodes)
    if n_uc == 0 or len(uc_edges) == 0:
        return False

    decimals = max(0, int(-np.log10(tol)))

    bmin = np.array(sym_spec["bounds_norm"][0], dtype=np.float64) * float(L)
    bmax = np.array(sym_spec["bounds_norm"][1], dtype=np.float64) * float(L)

    x = uc_nodes[:, 0]
    y = uc_nodes[:, 1]
    z = uc_nodes[:, 2]

    X0 = np.where(np.abs(x - bmin[0]) <= tol)[0]
    XL = np.where(np.abs(x - bmax[0]) <= tol)[0]
    Y0 = np.where(np.abs(y - bmin[1]) <= tol)[0]
    YL = np.where(np.abs(y - bmax[1]) <= tol)[0]
    Z0 = np.where(np.abs(z - bmin[2]) <= tol)[0]
    ZL = np.where(np.abs(z - bmax[2]) <= tol)[0]

    map_X0 = {(round(float(y[i]), decimals), round(float(z[i]), decimals)): int(i) for i in X0}
    map_XL = {(round(float(y[i]), decimals), round(float(z[i]), decimals)): int(i) for i in XL}
    map_Y0 = {(round(float(x[i]), decimals), round(float(z[i]), decimals)): int(i) for i in Y0}
    map_YL = {(round(float(x[i]), decimals), round(float(z[i]), decimals)): int(i) for i in YL}
    map_Z0 = {(round(float(x[i]), decimals), round(float(y[i]), decimals)): int(i) for i in Z0}
    map_ZL = {(round(float(x[i]), decimals), round(float(y[i]), decimals)): int(i) for i in ZL}

    keys_X = list(set(map_XL.keys()) & set(map_X0.keys()))
    keys_Y = list(set(map_YL.keys()) & set(map_Y0.keys()))
    keys_Z = list(set(map_ZL.keys()) & set(map_Z0.keys()))

    n_copy = nx * ny * nz
    total_nodes = n_copy * n_uc
    dsu = DSU(total_nodes)

    def cid(i, j, k):
        return (i * ny + j) * nz + k

    # 合并相邻拷贝的共享边界点
    for i in range(nx - 1):
        for j in range(ny):
            for k in range(nz):
                c1 = cid(i, j, k)
                c2 = cid(i + 1, j, k)
                base1 = c1 * n_uc
                base2 = c2 * n_uc
                for key in keys_X:
                    uL = map_XL[key]
                    u0 = map_X0[key]
                    dsu.union(base1 + uL, base2 + u0)

    for i in range(nx):
        for j in range(ny - 1):
            for k in range(nz):
                c1 = cid(i, j, k)
                c2 = cid(i, j + 1, k)
                base1 = c1 * n_uc
                base2 = c2 * n_uc
                for key in keys_Y:
                    uh = map_YL[key]
                    ul = map_Y0[key]
                    dsu.union(base1 + uh, base2 + ul)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1):
                c1 = cid(i, j, k)
                c2 = cid(i, j, k + 1)
                base1 = c1 * n_uc
                base2 = c2 * n_uc
                for key in keys_Z:
                    uh = map_ZL[key]
                    ul = map_Z0[key]
                    dsu.union(base1 + uh, base2 + ul)

    # 建 adjacency（只针对有边的 reps）
    adj = {}
    active = set()

    for c in range(n_copy):
        base = c * n_uc
        for u, v in uc_edges:
            ru = dsu.find(base + int(u))
            rv = dsu.find(base + int(v))
            active.add(ru)
            active.add(rv)
            if ru == rv:
                continue
            adj.setdefault(ru, []).append(rv)
            adj.setdefault(rv, []).append(ru)

    if not active:
        return False

    if require_all_nodes:
        reps = set(dsu.find(i) for i in range(total_nodes))
        if reps != active:
            return False

    start = next(iter(active))
    dq = deque([start])
    seen = {start}
    while dq:
        x = dq.popleft()
        for y in adj.get(x, []):
            if y not in seen:
                seen.add(y)
                dq.append(y)

    return seen == active


# =========================================================
# 12) Worker：批量生成成功样本（不写文件，只返回数据）
# =========================================================
def worker_generate_batch(cfg_dict, batch_size: int, seed: int):
    """
    返回 list[tuple(nodes_flat57, adj_flat361, L, bars)]
    """
    cfg = TrussConfig(**cfg_dict)

    # 重要：每个任务都重置随机种子，保证不同任务彼此独立
    random.seed(seed)

    geo = GeometryGenerator(cfg)
    out = []

    # 给 batch 一个总体尝试上限，避免极端情况下卡死
    max_total_attempts = max(batch_size * cfg.GENERATION_RETRIES, 2000)
    total_attempts = 0

    while len(out) < batch_size and total_attempts < max_total_attempts:
        total_attempts += 1

        nodes_norm = geo.generate_valid_geometry()
        if nodes_norm is None:
            continue

        base_edges, S_uc_norm = generate_sparse_edges_on_base(nodes_norm, cfg, geo.symmetry_spec)
        if base_edges is None or len(base_edges) == 0:
            continue

        # (A) base 连通
        if not is_connected(len(nodes_norm), base_edges, cfg.REQUIRE_ALL_NODES_CONNECTED):
            continue

        # 锁 L
        L = estimate_L_unitcell(S_uc_norm, cfg.R_PHYSICAL, cfg.RHO_TARGET, geo.symmetry_spec["volume_norm"])
        if L < 1e-12:
            continue

        base_nodes_phys = nodes_norm * L

        # (B) unitcell
        uc_nodes, uc_edges = apply_symmetry_ops(base_nodes_phys, base_edges, geo.symmetry_spec, L, cfg.TOLERANCE)
        if not is_connected(len(uc_nodes), uc_edges, cfg.REQUIRE_ALL_NODES_CONNECTED):
            continue

        # (B2) PBC degree==1
        if cfg.REJECT_INTERNAL_DEGREE1_AFTER_PBC or cfg.REJECT_BOUNDARY_DEGREE1_AFTER_PBC:
            pbc_nodes, pbc_edges, pbc_is_boundary = periodic_identify_unitcell(
                uc_nodes, uc_edges, geo.symmetry_spec, L, cfg.TOLERANCE
            )
            if not is_connected(len(pbc_nodes), pbc_edges, cfg.REQUIRE_ALL_NODES_CONNECTED):
                continue
            if reject_degree1_after_pbc(
                pbc_nodes, pbc_edges, pbc_is_boundary,
                reject_internal=cfg.REJECT_INTERNAL_DEGREE1_AFTER_PBC,
                reject_boundary=cfg.REJECT_BOUNDARY_DEGREE1_AFTER_PBC
            ):
                continue

        # (C) array 连通（fast）
        if not is_array_connected_fast(
            uc_nodes, uc_edges, geo.symmetry_spec, L,
            nx=cfg.NX, ny=cfg.NY, nz=cfg.NZ,
            tol=cfg.TOLERANCE,
            require_all_nodes=cfg.REQUIRE_ALL_NODES_CONNECTED
        ):
            continue

        # 组装输出：base_nodes(19x3) + base adjacency(19x19)
        n = len(base_nodes_phys)
        adjm = np.zeros((n, n), dtype=np.int8)
        for u, v in base_edges:
            adjm[u, v] = 1
            adjm[v, u] = 1

        nodes_flat = base_nodes_phys.reshape(-1).astype(np.float64).tolist()  # 57 floats
        adj_flat = adjm.reshape(-1).astype(int).tolist()                      # 361 ints

        out.append((nodes_flat, adj_flat, float(L), int(len(base_edges))))

    return out


# =========================================================
# 13) 主程序：多进程流水线 + 主进程写 CSV
# =========================================================
def run_with_config(cfg: TrussConfig, allow_single_process_fallback: bool = False):
    cfg_dict = asdict(cfg)

    geo = GeometryGenerator(cfg)
    node_names = geo.node_names_ordered
    name_prefix = str(geo.symmetry_spec.get("group_name", "sample"))

    csv_mgr = CSVManager(cfg, node_names)
    # ---------- 去重：只看邻接关系（361个element） ----------
    adj_start = 2 + 3 * len(node_names)      # id,name + 57 coords
    adj_len = csv_mgr.num_elements           # 361
    seen_adj = set()

# 如果要 RESUME 且希望“全局不重复”，启动时把历史CSV扫一遍建立 seen
    if cfg.RESUME_GENERATION and os.path.exists(csv_mgr.filepath):
        seen_adj = load_seen_adj_digests(csv_mgr.filepath, adj_start, adj_len)
# --------------------------------------------------------

    next_id = csv_mgr.start_index

    # 多进程：Windows 必须 spawn，且所有代码必须在 __main__ 保护下
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    in_flight = max(1, cfg.N_WORKERS * cfg.TASKS_IN_FLIGHT_PER_WORKER)
    buffer_rows = []
    last_print_t = time.time()

    def make_seed():
        # 避免不同任务 seed 碰撞（随机 + 时间）
        return (int(time.time() * 1e6) ^ random.getrandbits(32)) & 0x7fffffff

    def consume_batch(batch):
        nonlocal next_id, last_print_t
        if not batch:
            return 0

        added = 0

        for (nodes_flat, adj_flat, L, bars) in batch:
            if next_id >= cfg.TARGET_SAMPLES:
                break

            # --- 去重：只要邻接关系一样就丢弃 ---
            d = adj_digest_from_flat(adj_flat)
            if d in seen_adj:
                continue
            seen_adj.add(d)
            # -----------------------------------

            name = f"{name_prefix}_{uuid.uuid4().hex[:6]}"
            row = [next_id, name] + nodes_flat + adj_flat
            buffer_rows.append(row)
            next_id += 1
            added += 1

            if len(buffer_rows) >= cfg.CSV_FLUSH_EVERY:
                csv_mgr.append_rows(buffer_rows)
                buffer_rows.clear()

            # 进度打印（不要太频繁，打印本身也会慢）
            if (next_id - 1) % cfg.PRINT_EVERY == 0:
                now = time.time()
                if now - last_print_t > 0.2:
                    print(f"[OK] id={next_id-1} | base bars={bars} | L={L:.6f} | "
                          f"workers={cfg.N_WORKERS} | batch={cfg.BATCH_PER_TASK}")
                    last_print_t = now
        return added

    try:
        try:
            with ProcessPoolExecutor(max_workers=cfg.N_WORKERS) as ex:
                futures = set()
                for _ in range(in_flight):
                    futures.add(ex.submit(worker_generate_batch, cfg_dict, cfg.BATCH_PER_TASK, make_seed()))

                no_progress_batches = 0
                last_progress_t = time.time()

                while next_id < cfg.TARGET_SAMPLES:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)

                    for fut in done:
                        batch = fut.result()  # list of tuples
                        # 立刻补一个任务，保持队列满
                        if next_id < cfg.TARGET_SAMPLES:
                            futures.add(ex.submit(worker_generate_batch, cfg_dict, cfg.BATCH_PER_TASK, make_seed()))
                        added = consume_batch(batch)
                        if added > 0:
                            no_progress_batches = 0
                            last_progress_t = time.time()
                        else:
                            no_progress_batches += 1
                            idle_seconds = time.time() - last_progress_t
                            if (
                                no_progress_batches >= cfg.MAX_NO_PROGRESS_BATCHES
                                or idle_seconds >= cfg.MAX_NO_PROGRESS_SECONDS
                            ):
                                raise RuntimeError(
                                    "No valid samples generated for too long: "
                                    f"no_progress_batches={no_progress_batches}, "
                                    f"idle_seconds={idle_seconds:.1f}, "
                                    f"group={geo.symmetry_spec.get('group_name','<unknown>')}, "
                                    f"target={cfg.TARGET_SAMPLES}, reached={next_id}"
                                )
        except Exception as ex:
            if not allow_single_process_fallback:
                raise
            print(f"[WARN] ProcessPool failed ({type(ex).__name__}: {ex}); fallback to single-process mode")
            while next_id < cfg.TARGET_SAMPLES:
                batch = worker_generate_batch(cfg_dict, cfg.BATCH_PER_TASK, make_seed())
                consume_batch(batch)

        # flush 余量
        if buffer_rows:
            csv_mgr.append_rows(buffer_rows)
            buffer_rows.clear()

    finally:
        csv_mgr.close()

    print(f"Done. CSV saved to: {csv_mgr.filepath}")
    return csv_mgr.filepath


def main():
    cfg = TrussConfig()
    run_with_config(cfg)


if __name__ == "__main__":
    main()
