import argparse
import json
from pathlib import Path

import sympy as sp
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # 或 QtAgg
import matplotlib.pyplot as plt


DEFAULT_GROUP_DB = Path(__file__).with_name("symmetry_group_transforms.json")


def _resolve_local_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent / p


def _split_var_names(var_names):
    k_names = sorted([n for n in var_names if n.startswith("k_")])
    q_names = sorted([n for n in var_names if n.startswith("p_")])
    return {"k": k_names, "q": q_names, "all": sorted(var_names)}


def _build_constraint_payload(group_name, db_realpath, lattice_lengths, all_vars, free_vars, solved_items,
                              k_indep_norm, q_indep_norm):
    all_names = [str(v) for v in all_vars]
    free_names = [str(v) for v in free_vars]
    solved_map = {str(v): str(sp.simplify(rhs)) for v, rhs in solved_items}
    indep_k = [str(sp.simplify(e)) for e in sorted(k_indep_norm, key=str)]
    indep_q = [str(sp.simplify(e)) for e in sorted(q_indep_norm, key=str)]

    return {
        "format_version": 1,
        "group_name": group_name,
        "group_db_path": db_realpath,
        "lattice_lengths": lattice_lengths,
        "variables": _split_var_names(all_names),
        "free_vars": _split_var_names(free_names),
        "solved": solved_map,
        "independent_equations": {
            "k": indep_k,
            "q": indep_q,
            "all": indep_k + indep_q,
        },
    }


def export_constraints_payload(payload, out_path: str):
    p = _resolve_local_path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(p)


def load_group_transforms(group_name: str, db_path: str):
    p = _resolve_local_path(db_path)

    with p.open("r", encoding="utf-8") as f:
        db = json.load(f)

    groups = db.get("groups", {})
    if group_name not in groups:
        available = ", ".join(sorted(groups.keys()))
        raise KeyError(f"群名 '{group_name}' 不存在。可选群：{available}")

    group_data = groups[group_name]
    m_raw = group_data.get("M_sym")
    t_raw = group_data.get("T_vec_np")
    if not m_raw or not t_raw:
        raise ValueError(f"群 '{group_name}' 缺少 M_sym 或 T_vec_np")

    m_sym = [sp.Matrix(m) for m in m_raw]
    m_np = [np.array(m, dtype=float) for m in m_raw]
    t_vec_np = [np.array(t, dtype=float) for t in t_raw]
    lattice_lengths = group_data.get("lattice_lengths")
    return m_sym, m_np, t_vec_np, lattice_lengths, str(p)


def solve_and_visualize_constraints(group_name="P222", db_path=str(DEFAULT_GROUP_DB),
                                    export_path=None, show_plot=True):
    M_sym, M_np, T_vec_np, lattice_lengths, db_realpath = load_group_transforms(group_name, db_path)
    print(f">>> [Config] group={group_name} | ops={len(M_sym)} | db={db_realpath}")
    if lattice_lengths is not None:
        print(f">>> [Config] lattice_lengths={lattice_lengths}")
    print(">>> [Step 1] 初始化符号与几何定义...")

    # ==========================================
    # 1. 定义符号变量
    # ==========================================
    # 棱参数 k (0~1)
    k_vars = sp.symbols('k_A1 k_A2 k_A3 k_A4 k_A5 k_A6 k_A7 k_A8 k_A9 k_A10 k_A11 k_A12')
    # 创建字典方便调用
    k_dict = {f'A{i+1}': k for i, k in enumerate(k_vars)}

    # 面心参数 q (每个面两个自由度)
    p_fx, p_fz = sp.symbols('p_fx p_fz')      # Front (y=0)
    p_bx, p_bz = sp.symbols('p_bx p_bz')      # Back (y=1)
    p_ly, p_lz = sp.symbols('p_ly p_lz')      # Left (x=0)
    p_ry, p_rz = sp.symbols('p_ry p_rz')      # Right (x=1)
    p_tx, p_ty = sp.symbols('p_tx p_ty')      # Top (z=1)
    p_btx, p_bty = sp.symbols('p_btx p_bty')  # Bottom (z=0)

    # ==========================================
    # 2. 定义基本域拓扑 (Fundamental Domain)
    # ==========================================
    # 顶点坐标 (整数网格 0~1)
    base_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], # 0-3 (Bottom)
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]  # 4-7 (Top)
    ])

    # 棱定义：包含顶点索引(idx)和符号坐标(pt)
    edges_def = {
        'A1':  {'idx': [5, 4], 'pt': sp.Matrix([1 - k_dict['A1'], 0, 1])},
        'A2':  {'idx': [4, 7], 'pt': sp.Matrix([0, k_dict['A2'], 1])},
        'A3':  {'idx': [7, 6], 'pt': sp.Matrix([k_dict['A3'], 1, 1])},
        'A4':  {'idx': [6, 5], 'pt': sp.Matrix([1, 1 - k_dict['A4'], 1])},
        'A5':  {'idx': [1, 0], 'pt': sp.Matrix([1 - k_dict['A5'], 0, 0])},
        'A6':  {'idx': [0, 3], 'pt': sp.Matrix([0, k_dict['A6'], 0])},
        'A7':  {'idx': [3, 2], 'pt': sp.Matrix([k_dict['A7'], 1, 0])},
        'A8':  {'idx': [2, 1], 'pt': sp.Matrix([1, 1 - k_dict['A8'], 0])},
        'A9':  {'idx': [5, 1], 'pt': sp.Matrix([1, 0, 1 - k_dict['A9']])},
        'A10': {'idx': [4, 0], 'pt': sp.Matrix([0, 0, 1 - k_dict['A10']])},
        'A11': {'idx': [7, 3], 'pt': sp.Matrix([0, 1, 1 - k_dict['A11']])},
        'A12': {'idx': [6, 2], 'pt': sp.Matrix([1, 1, 1 - k_dict['A12']])},
    }

    # 面定义
    faces_def = {
        'q_front':  {'idx': [0, 1, 5, 4], 'pt': sp.Matrix([p_fx, 0, p_fz])},
        'q_back':   {'idx': [3, 2, 6, 7], 'pt': sp.Matrix([p_bx, 1, p_bz])},
        'q_left':   {'idx': [0, 3, 7, 4], 'pt': sp.Matrix([0, p_ly, p_lz])},
        'q_right':  {'idx': [1, 2, 6, 5], 'pt': sp.Matrix([1, p_ry, p_rz])},
        'q_top':    {'idx': [4, 5, 6, 7], 'pt': sp.Matrix([p_tx, p_ty, 1])},
        'q_bottom': {'idx': [0, 1, 2, 3], 'pt': sp.Matrix([p_btx, p_bty, 0])},
    }

    # ==========================================
    # 3. 变换矩阵准备（从 JSON 读取指定群）
    # ==========================================



    def transform_sym_point(pt, M):
        pt_h = pt.col_join(sp.Matrix([1])) 
        res = M @ pt_h
        return res[:3, :]

    def get_geo_id(indices, M_numpy):
        # 1. 获取原始顶点坐标
        original_coords = base_vertices[indices] # Shape (N, 3)
        # 2. 变换
        ones = np.ones((len(indices), 1))
        coords_h = np.hstack([original_coords, ones])
        trans_h = (M_numpy @ coords_h.T).T
        trans_coords = trans_h[:, :3].astype(int)
        # 3. 生成指纹 (Frozenset 忽略顶点的顺序，只看几何位置)
        return frozenset(tuple(x) for x in trans_coords)

    # ==========================================
    # 4. 生成全单胞 (变换后的集合)
    # ==========================================
    print(">>> [Step 2] 生成变换后的几何对象...")
    all_edges = []
    all_faces = []

    for m_idx in range(len(M_sym)): # 按群中实际对称操作数量遍历
        # 棱
        for label, data in edges_def.items():
            all_edges.append({
                'geo_id': get_geo_id(data['idx'], M_np[m_idx]),
                'sym_pt': transform_sym_point(data['pt'], M_sym[m_idx]),
                'label': f"{label}_M{m_idx}",
                'type': 'edge'
            })
        # 面
        for label, data in faces_def.items():
            all_faces.append({
                'geo_id': get_geo_id(data['idx'], M_np[m_idx]),
                'sym_pt': transform_sym_point(data['pt'], M_sym[m_idx]),
                'label': f"{label}_M{m_idx}",
                'type': 'face'
            })

    # ==========================================
    # 5. 碰撞检测与方程建立
    # ==========================================
    print(">>> [Step 3] 检测重合与周期性边界...")
    raw_equations = []
    
    # 将对象放入查找表，key是 geo_id (几何位置)
    # 注意：可能有多个对象变换后重叠到同一个位置 (这就是我们要找的约束)
    edge_map = {} 
    for e in all_edges:
        gid = e['geo_id']
        if gid not in edge_map: edge_map[gid] = []
        edge_map[gid].append(e)

    face_map = {}
    for f in all_faces:
        gid = f['geo_id']
        if gid not in face_map: face_map[gid] = []
        face_map[gid].append(f)

    processed_pairs = set()

    # --- 核心逻辑：遍历每个“几何位置”，并在所有可能的平移下寻找匹配 ---
    
    # 1. 提取所有存在的几何位置 ID
    all_edge_gids = list(edge_map.keys())
    
    for i in range(len(all_edge_gids)):
        gid_1 = all_edge_gids[i]
        
        for j in range(len(all_edge_gids)):
            gid_2 = all_edge_gids[j]
            
            # 检查是否可以通过平移向量 T 让 gid_1 与 gid_2 重合
            for T in T_vec_np:
                # 快速检查：如果顶点加上 T 等于目标顶点
                # 这里用集合判断最准确
                shifted_set = frozenset(tuple(np.array(pt) + T) for pt in gid_1)
                
                if shifted_set == gid_2:
                    # 发现重合！建立该位置上所有对象的方程
                    list_obj_1 = edge_map[gid_1]
                    list_obj_2 = edge_map[gid_2]
                    
                    for obj1 in list_obj_1:
                        for obj2 in list_obj_2:
                            # 避免自己和自己比 (且 T=0)
                            if obj1 == obj2 and np.all(T == 0): continue
                            
                            # 建立去重键 (排序 label)
                            pair_key = tuple(sorted([obj1['label'], obj2['label']])) + tuple(T)
                            if pair_key in processed_pairs: continue
                            processed_pairs.add(pair_key)
                            
                            # 建立方程： P1 + T = P2
                            # 这里的 T 是数值向量，需要转为 sympy
                            T_sp = sp.Matrix(T).reshape(3, 1)
                            eq = sp.Eq(obj1['sym_pt'] + T_sp, obj2['sym_pt'])
                            raw_equations.append(eq)

    # 对面做同样的处理
    all_face_gids = list(face_map.keys())
    for i in range(len(all_face_gids)):
        gid_1 = all_face_gids[i]
        for j in range(len(all_face_gids)):
            gid_2 = all_face_gids[j]
            for T in T_vec_np:
                shifted_set = frozenset(tuple(np.array(pt) + T) for pt in gid_1)
                if shifted_set == gid_2:
                    list_obj_1 = face_map[gid_1]
                    list_obj_2 = face_map[gid_2]
                    for obj1 in list_obj_1:
                        for obj2 in list_obj_2:
                            if obj1 == obj2 and np.all(T == 0): continue
                            pair_key = tuple(sorted([obj1['label'], obj2['label']])) + tuple(T)
                            if pair_key in processed_pairs: continue
                            processed_pairs.add(pair_key)
                            T_sp = sp.Matrix(T).reshape(3, 1)
                            eq = sp.Eq(obj1['sym_pt'] + T_sp, obj2['sym_pt'])
                            raw_equations.append(eq)

    # ==========================================
    # 6. 求解与格式化输出
    # ==========================================
    # ==========================================
    # 6. 化简 + 拆成标量方程（expr == 0）
    #    然后用“秩增量”挑选线性无关的独立约束
    # ==========================================
    print(f">>> [Step 4] 原始向量方程数: {len(raw_equations)}. 正在拆解与化简...")

    # 变量列表（顺序固定，保证输出稳定）
    k_list = list(k_vars)
    q_list = [p_fx, p_fz, p_bx, p_bz, p_ly, p_lz, p_ry, p_rz, p_tx, p_ty, p_btx, p_bty]
    k_set = set(k_list)

    k_exprs = []          # 每个元素是“简化后的标量表达式”，代表 expr == 0
    q_exprs = []
    contradictions = []   # 纯数字 != 0 的矛盾

    for eq_vec in raw_equations:
        if eq_vec is True or eq_vec is False:
            continue
        if not hasattr(eq_vec, "lhs"):
            continue

        for dim in range(3):
            expr = sp.simplify(eq_vec.lhs[dim] - eq_vec.rhs[dim])
            if expr == 0:
                continue

            free_syms = expr.free_symbols
            if not free_syms:
                # 纯数字约束：如果不是 0 就矛盾
                contradictions.append(expr)
                continue

            if free_syms & k_set:
                k_exprs.append(expr)
            else:
                q_exprs.append(expr)

    def independent_linear_exprs(exprs, vars_):
        """
        从一组线性方程 expr==0 中挑出线性无关（逻辑独立）的最小子集。
        判定方式：把每个方程转成增广行 [A | b]，只保留能增加 rank 的行。
        返回：(kept_exprs, inconsistent_exprs)
        """
        kept_exprs = []
        kept_rows = []     # 每一行是 1×(n+1) 的 Matrix
        inconsistent = []

        # 用于快速提升稳定性：把表达式展开成标准线性形式
        #（如果你非常确定都线性，也可以删掉 expand）
        for expr in exprs:
            expr_std = sp.expand(expr)

            # 先尝试用 sympy 线性化
            try:
                A, b = sp.linear_eq_to_matrix([sp.Eq(expr_std, 0)], vars_)
            except Exception:
                # 非线性（理论上不该出现），先当作“无法处理”
                # 你如果后续真的有非线性，我可以给你 Groebner 版本做“逻辑独立”。
                kept_exprs.append(expr_std)
                continue

            # A*x = b  <=>  A*x - b = 0  => 增广行 [A | -b]
            row = A.row_join(-b)  # 1×(n+1)

            # 检查矛盾：系数全 0 但常数不 0
            if all(sp.simplify(row[0, j]) == 0 for j in range(row.shape[1]-1)) and sp.simplify(row[0, -1]) != 0:
                inconsistent.append(expr_std)
                continue

            if not kept_rows:
                kept_rows.append(row)
                kept_exprs.append(expr_std)
            else:
                M = sp.Matrix.vstack(*kept_rows)
                r0 = M.rank()
                M2 = sp.Matrix.vstack(M, row)
                r1 = M2.rank()
                if r1 > r0:
                    kept_rows.append(row)
                    kept_exprs.append(expr_std)

        return kept_exprs, inconsistent

    # 挑独立子集
    k_indep, k_bad = independent_linear_exprs(k_exprs, k_list)
    q_indep, q_bad = independent_linear_exprs(q_exprs, q_list)

    # 再做一次“完全形式去重”（同一 expr 可能来自不同来源）
    # 这里用 simplify+factor 归一化后做 set
    def normalize_expr(e):
        e2 = sp.simplify(sp.factor(e))
        # 归一化符号：把前导系数变成正（避免 e 和 -e 都出现）
        # 找到一个非零系数做 sign
        try:
            poly = sp.Poly(e2, *sorted(list(e2.free_symbols), key=str))
            coeffs = [c for c in poly.coeffs() if c != 0]
            if coeffs and coeffs[0].could_extract_minus_sign():
                e2 = -e2
        except Exception:
            pass
        return e2

    k_indep_norm = []
    seen = set()
    for e in k_indep:
        ne = normalize_expr(e)
        s = str(ne)
        if s not in seen:
            seen.add(s)
            k_indep_norm.append(ne)

    q_indep_norm = []
    seen = set()
    for e in q_indep:
        ne = normalize_expr(e)
        s = str(ne)
        if s not in seen:
            seen.add(s)
            q_indep_norm.append(ne)

    # ==========================================
    # 7. 输出（独立约束）
    # ==========================================
    print("\n" + "="*60)
    print(f" {group_name} 对称性约束结果（去除线性冗余后的独立集合）")
    print("="*60)

    if contradictions or k_bad or q_bad:
        print("\n[警告] 发现可能的矛盾/不可线性化项：")
        for c in contradictions:
            print("  contradiction:", c, "= 0  (纯数字矛盾)")
        for c in k_bad:
            print("  edge-inconsistent:", c, "= 0")
        for c in q_bad:
            print("  face-inconsistent:", c, "= 0")

    print(f"\n--- Edge(k) 独立约束数: {len(k_indep_norm)}  (原始标量约束候选: {len(k_exprs)}) ---")
    for e in sorted(k_indep_norm, key=str):
        print(sp.pretty(sp.Eq(e, 0)))

    print(f"\n--- Face(q) 独立约束数: {len(q_indep_norm)} (原始标量约束候选: {len(q_exprs)}) ---")
    for e in sorted(q_indep_norm, key=str):
        print(sp.pretty(sp.Eq(e, 0)))

    # （可选）你如果仍想画图展示
    if show_plot and (len(k_indep_norm) > 0 or len(q_indep_norm) > 0):
        fig, ax = plt.subplots(figsize=(10, 0.35*(len(k_indep_norm)+len(q_indep_norm)) + 2.5))
        ax.axis('off')
        y = 0.95
        ax.text(0.05, y, f"{group_name} Independent Constraints", fontsize=16, weight='bold')
        y -= 0.06

        ax.text(0.05, y, "Edge (k):", fontsize=13)
        y -= 0.04
        for e in sorted(k_indep_norm, key=str):
            ax.text(0.08, y, f"${sp.latex(sp.Eq(e,0))}$", fontsize=11)
            y -= 0.035

        y -= 0.03
        ax.text(0.05, y, "Face (q):", fontsize=13)
        y -= 0.04
        for e in sorted(q_indep_norm, key=str):
            ax.text(0.08, y, f"${sp.latex(sp.Eq(e,0))}$", fontsize=11)
            y -= 0.035

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis('off')
        ax.text(0.05, 0.9, f"{group_name} Independent Constraints", fontsize=16, weight='bold')
        ax.text(0.05, 0.75, f"raw_equations = {len(raw_equations)}", fontsize=12)
        ax.text(0.05, 0.6,  f"k_indep = {len(k_indep_norm)}, q_indep = {len(q_indep_norm)}", fontsize=12)

        plt.savefig(f"{group_name}_constraints.png", dpi=200, bbox_inches="tight")
        plt.show()


# ========================= 【中间要替换/插入】 =========================
    # ---- 把所有约束合并，并化到最简（RREF 参数化：pivot 变量 = 常数/自由变量线性组合）----
    def rref_parameterize(exprs, vars_):
        """
        输入：线性标量方程 expr==0 列表
        输出：
          solved: dict{pivot_var: affine_expr_in_free_vars}
          free_vars: list[free_var]
          inconsistent: list[constant]  (发现 0 = c 的矛盾时返回 c)
        """
        if not exprs:
            return {}, list(vars_), []

        eqs = [sp.Eq(sp.expand(e), 0) for e in exprs]
        try:
            A, b = sp.linear_eq_to_matrix(eqs, vars_)   # A*vars = b
        except Exception:
            # 假设应为线性；若出现非线性则退化为“不做最简化”
            return {}, list(vars_), []

        Aug = A.row_join(b)
        R, pivots = Aug.rref()

        n = len(vars_)
        inconsistent = []
        for i in range(R.rows):
            if all(sp.simplify(R[i, j]) == 0 for j in range(n)) and sp.simplify(R[i, n]) != 0:
                inconsistent.append(sp.simplify(R[i, n]))
        if inconsistent:
            return {}, [], inconsistent

        pivot_set = set(pivots)
        free_cols = [j for j in range(n) if j not in pivot_set]
        free_vars = [vars_[j] for j in free_cols]

        solved = {}
        # RREF: x_p + Σ r_{p,f} x_f = rhs  => x_p = rhs - Σ r_{p,f} x_f
        for row_i, col_j in enumerate(pivots):
            rhs = R[row_i, n]
            for fcol in free_cols:
                rhs -= R[row_i, fcol] * vars_[fcol]
            solved[vars_[col_j]] = sp.simplify(rhs)

        return solved, free_vars, []

    # 合并变量与方程
    all_vars = k_list + q_list
    all_exprs = k_indep_norm + q_indep_norm

    solved, free_vars, incon = rref_parameterize(all_exprs, all_vars)

    print("\n" + "="*60)
    print(f" {group_name} 最简约束（全部合并输出）")
    print("="*60)

    payload = None

    if incon:
        print("[矛盾] 发现 0 = c：")
        for c in incon:
            print("  0 =", sp.pretty(c))
    else:
        free_vars_sorted = sorted(free_vars, key=str)
        solved_items = sorted(solved.items(), key=lambda t: str(t[0]))

        print(f"\nfree vars ({len(free_vars_sorted)}):", [str(v) for v in free_vars_sorted])
        print(f"\nconstraints ({len(solved_items)}):")
        for v, rhs in solved_items:
            # 以“关系式”形式输出：v = rhs
            print(sp.pretty(sp.Eq(v, rhs)))

        # 如果你更想要统一的“expr = 0”形式，取消下面注释：
        # print("\nconstraints as expr = 0:")
        # for v, rhs in solved_items:
        #     print(sp.pretty(sp.Eq(v - rhs, 0)))

        payload = _build_constraint_payload(
            group_name=group_name,
            db_realpath=db_realpath,
            lattice_lengths=lattice_lengths,
            all_vars=all_vars,
            free_vars=free_vars_sorted,
            solved_items=solved_items,
            k_indep_norm=k_indep_norm,
            q_indep_norm=q_indep_norm,
        )
        if export_path:
            out_file = export_constraints_payload(payload, export_path)
            print(f"\n[EXPORT] constraints json -> {out_file}")

    # 关闭绘图与冗余打印（只保留最简合并结果）
    return payload
# ======================= 【补丁结束】 =======================

def solve_and_visualize_p222_final():
    """
    兼容旧调用：不传参时仍按 P222 运行。
    """
    return solve_and_visualize_constraints(group_name="P222", db_path=str(DEFAULT_GROUP_DB))


def parse_args():
    parser = argparse.ArgumentParser(description="按群名读取变换矩阵并求解 k/q 约束")
    parser.add_argument("--group", default="P222", help="群名，例如 P222、Aba2、Ccce")
    parser.add_argument("--db", default=str(DEFAULT_GROUP_DB), help="群矩阵 JSON 文件路径")
    parser.add_argument("--export", default="", help="导出约束 JSON 路径（可选）")
    parser.add_argument("--no-plot", action="store_true", help="不显示/保存约束图，仅输出文本与JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    solve_and_visualize_constraints(
        group_name=args.group,
        db_path=args.db,
        export_path=args.export if args.export else None,
        show_plot=not args.no_plot,
    )
