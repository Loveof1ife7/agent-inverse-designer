 # -*- coding: utf-8 -*-
import os
import subprocess
import math
import shutil
import sys
import argparse

try:
    import numpy as np
except ImportError:
    print("[Error] This script requires numpy.")
    sys.exit(1)

# Abaqus command selection
ABAQUS_CMD = os.environ.get("ABAQUS_CMD")
if not ABAQUS_CMD:
    ABAQUS_CMD = (
        shutil.which("abq2025")
        or shutil.which("abq2022")
        or shutil.which("abaqus")
        or "abaqus"
    )

# ==============================================================================
# 1) 用户配置
# ==============================================================================

INPUT_FOLDER  = r"D:\codes\agent-material-windows-lite\train_datas\raw\P222_paired_dataset_0_99999_20260620\geometry"
OUTPUT_FOLDER = r"D:\codes\agent-material-windows-lite\train_datas\raw\P222_paired_dataset_0_99999_20260620\fem_curves"

TARGET_FILE_INDEX = "0"
CPUS = 8

MODE = "PLATE_Z"   # 板压 + XY周期
NZ_STACK = 1          # Z方向堆叠层数

TARGET_STRAIN = -0.30    # 负号=压缩
STEP_TIME = 1
N_FIELD_FRAMES = 50
DAMPING_BETA = 0 #阻尼
MESH_SEGMENTS = 3

# ---------------- 后处理/应变参考配置 ----------------
# STRAIN_REF_MODE:
#   "NODE_Z"    -> 以节点包络高度 (单位胞几何长度) 作为应变基准
#   "SURFACE_Z" -> 以梁外表面包络高度作为应变基准
STRAIN_REF_MODE = "NODE_Z"

# 接触起点判定（基于RP_TOP反力）
CONTACT_FORCE_TOL = 1e-8     # 绝对阈值（按你的单位系）
CONTACT_FORCE_FRAC = 1e-4    # 相对阈值（占最大反力的比例）
DROP_PRECONTACT = True       # 是否丢弃接触前的数据点
# 是否把“接触起点”的反力作为零点（使应力在应变=0时为0）
ZERO_STRESS_AT_CONTACT = True

# 【关键修正】：必须设置为 True，下面的接触代码才会生效！
ENABLE_SELF_CONTACT = True
FRICTION_COEFF = 0.3

BEAM_RADIUS      = 1.0
YOUNG_MODULUS    = 8.925
POISSON_RATIO    = 0.48
MERGE_TOL        = 1e-3

YIELD_STRESS     = 3.207   # Dataset TPU95A yield stress
HARDENING_MOD    = 0          # 不再用单点硬化模型，下面用 PLASTIC_TABLE
MATERIAL_DENSITY = 1.11e-9

# Custom942_v2 完整硬化-软化-保持表
PLASTIC_TABLE = [
    (3.2070, 0.000000),
    (3.3664, 0.003404),
    (3.4871, 0.004926),
    (3.5863, 0.006874),
    (3.6534, 0.008789),
    (3.6963, 0.010881),
    (3.7302, 0.013308),
    (3.7437, 0.015616),
    (3.7437, 0.018024),
    (3.7291, 0.020805),
    (3.7078, 0.023368),
    (3.6645, 0.026360),
    (3.6645, 0.100000),
    (3.6645, 1.000000),
]

ORIENT_PARALLEL_COS = 0.95

FACE_TOL = 2e-3
POST_SMOOTH_WINDOW = 1
PLATE_SCALE_XY      = 2.0
# ==============================================================================
# ODB 提取脚本（板压 + 顶面节点集求和）
# ==============================================================================
ODB_EXTRACTOR_TEMPLATE = r"""# -*- coding: utf-8 -*-
import sys, os
from odbAccess import openOdb

LZ = __LZ_TARGET__
AREA = __AREA__
FORCE_TOL = __FORCE_TOL__
FORCE_FRAC = __FORCE_FRAC__
DROP_PRECONTACT = __DROP_PRECONTACT__
ZERO_STRESS_AT_CONTACT = __ZERO_STRESS_AT_CONTACT__

def get_nodeset(odb, set_name):
    key = set_name.upper()

    a = odb.rootAssembly
    if key in a.nodeSets:
        return a.nodeSets[key]

    for inst in a.instances.values():
        if key in inst.nodeSets:
            return inst.nodeSets[key]

    # 关键补丁：很多 input 里定义的 set 会落在 odb.parts
    for p in odb.parts.values():
        if key in p.nodeSets:
            return p.nodeSets[key]

    return None

def find_whole_model_history_region(step):
    for k in step.historyRegions.keys():
        ku = k.upper()
        if "WHOLE MODEL" in ku:
            return step.historyRegions[k]
    for k in step.historyRegions.keys():
        ku = k.upper()
        if "ASSEMBLY" in ku:
            return step.historyRegions[k]
    return None

def build_time_value_map(hist_output):
    m = {}
    for tt, vv in hist_output.data:
        m[float(tt)] = float(vv)
    return m

def nearest_value(t, tmap):
    if not tmap:
        return None
    best_t = None
    best_dt = 1.0e100
    for tt in tmap.keys():
        dt = abs(tt - t)
        if dt < best_dt:
            best_dt = dt
            best_t = tt
    return tmap[best_t]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: abaqus python extract.py job.odb out.csv\n")
        sys.exit(2)

    odb_path = sys.argv[1]
    out_csv  = sys.argv[2]

    if not os.path.exists(odb_path):
        sys.stderr.write("ODB not found: %s\n" % odb_path)
        sys.exit(3)

    odb = openOdb(odb_path, readOnly=True)
    step = odb.steps[list(odb.steps.keys())[-1]]

    top_set = get_nodeset(odb, "RP_TOP")

    if top_set is None:
        sys.stderr.write("Cannot find ZMAX_FACE in odb.\n")
        odb.close()
        sys.exit(10)

    whole = find_whole_model_history_region(step)
    t2ALLKE = {}
    t2ALLIE = {}
    t2ALLWK = {}
    if whole is not None:
        h = whole.historyOutputs
        if "ALLKE" in h:
            t2ALLKE = build_time_value_map(h["ALLKE"])
        if "ALLIE" in h:
            t2ALLIE = build_time_value_map(h["ALLIE"])
        if "ALLWK" in h:
            t2ALLWK = build_time_value_map(h["ALLWK"])

    records = []
    for fr in step.frames:
        fo = fr.fieldOutputs
        if "U" not in fo:
            continue

        rf_key = None
       
        if "RF" in fo:
            rf_key = "RF"
        elif "CF" in fo:
            rf_key = "CF"
        else:
            continue

        Uv_all = fo["U"].getSubset(region=top_set).values
        if not Uv_all:
            continue

        # 椤堕潰骞冲潎浣嶇Щ
        u3_avg = sum(float(v.data[2]) for v in Uv_all) / float(len(Uv_all))

        RFv_all = fo[rf_key].getSubset(region=top_set).values
        if not RFv_all:
            continue

        # 椤堕潰鎬诲弽鍔?
        rf3_sum = sum(float(v.data[2]) for v in RFv_all)

        t = float(fr.frameValue)

        ALLKE = nearest_value(t, t2ALLKE)
        ALLIE = nearest_value(t, t2ALLIE)
        ALLWK = nearest_value(t, t2ALLWK)

        records.append((t, u3_avg, rf3_sum, ALLKE, ALLIE, ALLWK))

    if not records:
        sys.stderr.write("No data written. Check FIELD output requests (U, RF/CF).\n")
        odb.close()
        sys.exit(20)

    max_rf = max(abs(r[2]) for r in records)
    thr = max(FORCE_TOL, FORCE_FRAC * max_rf)

    contact_idx = 0
    for i, r in enumerate(records):
        if abs(r[2]) >= thr:
            contact_idx = i
            break

    u3_contact = records[contact_idx][1]
    rf3_contact = records[contact_idx][2]

    wrote = 0
    with open(out_csv, "w") as f:
        f.write("Strain,Disp_mm,Force_N,Stress_MPa,ALLKE,ALLIE,ALLWK,KE_over_IE,Time_s\n")

        for i, r in enumerate(records):
            t, u3_avg, rf3_sum, ALLKE, ALLIE, ALLWK = r

            if i < contact_idx:
                if DROP_PRECONTACT:
                    continue
                disp_eff = 0.0
                rf3_eff = 0.0
            else:
                disp_eff = abs(u3_avg - u3_contact)
                if ZERO_STRESS_AT_CONTACT:
                    rf3_eff = rf3_sum - rf3_contact
                else:
                    rf3_eff = rf3_sum

            if LZ != 0.0:
                strain = disp_eff / LZ
            else:
                strain = float("nan")
            disp_mm = disp_eff

            force_n = -rf3_eff
            stress  = -rf3_eff / AREA

            if ALLKE is None: ALLKE = float("nan")
            if ALLIE is None: ALLIE = float("nan")
            if ALLWK is None: ALLWK = float("nan")

            if (ALLIE == ALLIE) and (ALLIE != 0.0):
                KE_over_IE = ALLKE / ALLIE
            else:
                KE_over_IE = float("nan")

            f.write("%g,%g,%g,%g,%g,%g,%g,%g,%g\n" % (
                strain, disp_mm, force_n, stress,
                ALLKE, ALLIE, ALLWK, KE_over_IE, t
            ))

            wrote += 1

    odb.close()
    if wrote == 0:
        sys.stderr.write("No data written. Check FIELD output requests (U, RF/CF).\n")
        sys.exit(20)
"""

# ==============================================================================
# 0) 基础路径
# ==============================================================================
def get_script_dir():
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()

THIS_DIR = get_script_dir()
RUN_DIR  = os.path.join(THIS_DIR, "RUN_PLATE_Z")

# ==============================================================================
# 2) 几何处理
# ==============================================================================
def load_raw_data(index):
    path = os.path.join(INPUT_FOLDER, f"{index}.txt")
    if not os.path.exists(path):
        return None, None, None

    fg = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        for l in raw.split("\n"):
            if "# Data ID:" in l:
                name = l.split(":")[-1].strip()
                break
        else:
            name = str(index)

        exec(raw, {}, fg)
    except Exception as e:
        print(f"[Load Error] {e}")
        return None, None, None

    return fg.get("node_data"), fg.get("element_conn"), name

def merge_nodes_gridhash(nodes_raw, elems_raw, tol):
    def key3(x, y, z):
        return (int(round(x / tol)), int(round(y / tol)), int(round(z / tol)))

    def elem_nodes(e):
        """
        兼容两种常见格式：
        - [n1, n2]
        - [eid, n1, n2] 或者更长：[eid, n1, n2, ...]
        """
        if e is None:
            raise ValueError("Element record is None")

        if len(e) >= 3:
            # 99% 的 truss 数据是 [eid, n1, n2]
            return int(e[-2]), int(e[-1])
        if len(e) == 2:
            return int(e[0]), int(e[1])
        raise ValueError(f"Bad element record: {e}")

    buckets = {}
    node_map = {}
    new_nodes = []
    nid = 1
    neighbor_shifts = [(i, j, k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)]

    for n in nodes_raw:
        old_id = int(n[0])
        x, y, z = float(n[1]), float(n[2]), float(n[3])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue

        k0 = key3(x, y, z)
        found_id = None

        for dx, dy, dz in neighbor_shifts:
            kk = (k0[0]+dx, k0[1]+dy, k0[2]+dz)
            if kk in buckets:
                for (eid, ex, ey, ez) in buckets[kk]:
                    if (x-ex)**2 + (y-ey)**2 + (z-ez)**2 <= tol**2:
                        found_id = eid
                        break
            if found_id:
                break

        if found_id is None:
            new_nodes.append([nid, x, y, z])
            buckets.setdefault(k0, []).append((nid, x, y, z))
            node_map[old_id] = nid
            nid += 1
        else:
            node_map[old_id] = found_id

    new_elems = []
    for e in elems_raw:
        try:
            a, b = elem_nodes(e)
            n1, n2 = node_map[a], node_map[b]
            if n1 != n2:
                new_elems.append([n1, n2])
        except:
            pass

    return new_nodes, new_elems

def refine_mesh_pure(nodes, elems, n_seg):
    if n_seg <= 1:
        return nodes, elems

    print(f"Refining mesh: splitting each strut into {n_seg} elements...")
    max_nid = max(int(n[0]) for n in nodes)
    node_map = {int(n[0]): (n[1], n[2], n[3]) for n in nodes}

    new_nodes = list(nodes)
    new_elems = []
    for e in elems:
        n1_id, n2_id = int(e[0]), int(e[1])
        p1, p2 = node_map[n1_id], node_map[n2_id]
        vec = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])

        chain = [n1_id]
        for k in range(1, n_seg):
            r = k / float(n_seg)
            nx = p1[0] + vec[0]*r
            ny = p1[1] + vec[1]*r
            nz = p1[2] + vec[2]*r
            max_nid += 1
            new_nodes.append([max_nid, nx, ny, nz])
            chain.append(max_nid)
        chain.append(n2_id)

        for k in range(n_seg):
            new_elems.append([chain[k], chain[k+1]])

    return new_nodes, new_elems

def remove_isolated_nodes(nodes, elems):
    used = set()
    for e in elems:
        used.add(int(e[0]))
        used.add(int(e[1]))
    new_nodes = [n for n in nodes if int(n[0]) in used]
    if len(new_nodes) != len(nodes):
        print(f"[Info] Removed isolated nodes: {len(nodes) - len(new_nodes)}")
    return new_nodes, elems

def compute_bounds_dims(nodes):
    xs = [n[1] for n in nodes]
    ys = [n[2] for n in nodes]
    zs = [n[3] for n in nodes]
    bounds = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
    dims = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]
    return bounds, dims

def stack_z_layers(nodes, elems, Lz, n_layers):
    if n_layers <= 1:
        return nodes, elems

    max_nid = max(int(n[0]) for n in nodes)
    nid_offset = max_nid + 1000

    nodes_out = []
    elems_out = []
    for k in range(n_layers):
        dz = k * Lz
        id_map = {}
        for n in nodes:
            old = int(n[0])
            new = old + k * nid_offset
            id_map[old] = new
            nodes_out.append([new, float(n[1]), float(n[2]), float(n[3]) + dz])
        for e in elems:
            elems_out.append([id_map[int(e[0])], id_map[int(e[1])]])

    print(f"[Stack] Z layers = {n_layers}, nodes: {len(nodes)} -> {len(nodes_out)}, elems: {len(elems)} -> {len(elems_out)}")
    return nodes_out, elems_out

# ==============================================================================
# 3) B31：n1 选择
# ==============================================================================
def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _norm(v):
    return math.sqrt(_dot(v, v))

def choose_n1_for_element(p1, p2):
    d = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
    nd = _norm(d)
    if nd <= 0.0:
        return (1.0, 0.0, 0.0)

    du = (d[0]/nd, d[1]/nd, d[2]/nd)
    candidates = [(1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0)]
    for c in candidates:
        if abs(_dot(du, c)) < ORIENT_PARALLEL_COS:
            return c
    return (0.0, 0.0, 1.0)

# ==============================================================================
# 4) XY 周期 PBC（无Z周期）
# ==============================================================================

    
def pick_anchor_node_on_zmin(nodes, elems, bounds, tol):
    xm, xM, ym, yM, zm, zM = bounds
    active = set()
    for e in elems:
        active.add(int(e[0])); active.add(int(e[1]))

    best = None
    best_d2 = 1e100
    for n in nodes:
        nid = int(n[0])
        if nid not in active:
            continue
        x, y, z = float(n[1]), float(n[2]), float(n[3])
        if abs(z - zm) > tol:
            continue
        d2 = (x-xm)**2 + (y-ym)**2 + (z-zm)**2
        if d2 < best_d2:
            best_d2 = d2
            best = nid

    if best is None:
        for n in nodes:
            nid = int(n[0])
            if nid not in active:
                continue
            x, y, z = float(n[1]), float(n[2]), float(n[3])
            d2 = (x-xm)**2 + (y-ym)**2 + (z-zm)**2
            if d2 < best_d2:
                best_d2 = d2
                best = nid

    return best

# ==============================================================================
# 5) INP 生成：板压(U3) + XY PBC(equation)
# ==============================================================================
def write_id_list(f, ids, per_line=16):
    for i in range(0, len(ids), per_line):
        f.write(", ".join(str(int(x)) for x in ids[i:i+per_line]) + "\n")

def write_equation_terms(f, terms):
    f.write(f"{len(terms)}\n")
    for nid, dof, coef in terms:
        f.write(f"{int(nid)}, {int(dof)}, {float(coef):.16g}\n")



def compute_beam_surface_z_extents(nodes, elems, radius):
    """
    估计 B31 圆截面杆系在全局 Z 方向的外表面包络：
    对每根直杆段，轴向单位向量 u=(ux,uy,uz)，圆截面平面 ⟂u，
    该圆截面在全局 Z 方向的最大投影半径为 R*sqrt(1-uz^2)。
    于是该杆段的外表面 z_min/z_max 约为：
        z_min_seg = min(z1,z2) - Rz
        z_max_seg = max(z1,z2) + Rz
    """
    nd = {int(n[0]): (float(n[1]), float(n[2]), float(n[3])) for n in nodes}
    zmin_surf =  1.0e100
    zmax_surf = -1.0e100

    for e in elems:
        n1 = int(e[0]); n2 = int(e[1])
        x1,y1,z1 = nd[n1]
        x2,y2,z2 = nd[n2]
        dx = x2-x1; dy = y2-y1; dz = z2-z1
        L = math.sqrt(dx*dx + dy*dy + dz*dz)
        if L <= 0.0:
            continue
        uz = dz / L
        # Rz = R*sqrt(1-uz^2)，数值保护
        t = 1.0 - uz*uz
        if t < 0.0:
            t = 0.0
        Rz = radius * math.sqrt(t)

        zmin_seg = (z1 if z1 < z2 else z2) - Rz
        zmax_seg = (z1 if z1 > z2 else z2) + Rz

        if zmin_seg < zmin_surf:
            zmin_surf = zmin_seg
        if zmax_seg > zmax_surf:
            zmax_surf = zmax_seg

    # 兜底：极端情况下 elems 为空
    if not math.isfinite(zmin_surf) or not math.isfinite(zmax_surf):
        zs = [float(n[3]) for n in nodes]
        zmin_surf = min(zs)
        zmax_surf = max(zs)

    return zmin_surf, zmax_surf



def create_inp(run_dir, nodes, elems, job_name, bounds, dims):
    """
    Abaqus/Explicit 准静态板压（刚性板 + General Contact）
    - 两块刚性板(R3D4) + 参考点 RP
    - 底板 RP 全固定
    - 顶板 RP 仅允许 U3 受控下压（U1/U2=0，转动=0）
    - 梁-梁自接触 + 梁-板接触（摩擦）
    - 额外用两个底面点只锁 U1/U2 防止整体 XY 漂移（不锁 U3）
    """
    xm, xM, ym, yM, zm, zM = bounds
    Lx, Ly, Lz = dims
    area = Lx * Ly

    # 节点坐标字典
    ndict = {int(n[0]): (float(n[1]), float(n[2]), float(n[3])) for n in nodes}

    # 顶/底面节点集（用于检查/后处理）
    zmin_ids, zmax_ids = [], []
    for nid, (x, y, z) in ndict.items():
        if abs(z - zm) <= FACE_TOL:
            zmin_ids.append(nid)
        if abs(z - zM) <= FACE_TOL:
            zmax_ids.append(nid)
    if (not zmin_ids) or (not zmax_ids):
        raise RuntimeError("ZMIN/ZMAX set empty. Check FACE_TOL.")

    # 防漂移点：底面挑 2 个相距最远的点，只锁 U1/U2
    rbm1 = zmin_ids[0]
    ax, ay, _ = ndict[rbm1]
    rbm2 = None
    best_d2 = -1.0
    for nid in zmin_ids:
        if nid == rbm1:
            continue
        x, y, _ = ndict[nid]
        d2 = (x - ax) ** 2 + (y - ay) ** 2
        if d2 > best_d2:
            best_d2 = d2
            rbm2 = nid

    # 刚性板尺寸与位置
    max_nid = max(int(n[0]) for n in nodes)
    margin = max(2.0 * BEAM_RADIUS, 0.02 * max(Lx, Ly))

    # 这个 gap 很关键：太大可能永远碰不到；可先用 0.0~0.2*R 验证接触是否工作
# ★关键：板的位置要按“梁外表面”放，而不是中心线极值 zm/zM
    # 这样避免初始就压进梁半径造成冲击（KE/IE第一个点偏大）
    clear = 1e-4 * BEAM_RADIUS   # 极小间隙；如果还冲击可调到 1e-3~1e-2*R
    # 【替换补丁：把板的 XY 尺寸按 PLATE_SCALE_XY 放大（以试样中心为基准对称扩展）】
    cx = 0.5 * (xm + xM)
    cy = 0.5 * (ym + yM)

    half_x = 0.5 * Lx * PLATE_SCALE_XY + margin
    half_y = 0.5 * Ly * PLATE_SCALE_XY + margin

    x0 = cx - half_x
    x1 = cx + half_x
    y0 = cy - half_y
    y1 = cy + half_y


    # --- 新逻辑：按杆方向估计外表面包络，再放板 ---
    z_surf_min, z_surf_max = compute_beam_surface_z_extents(nodes, elems, BEAM_RADIUS)

    z_bot = z_surf_min - clear
    z_top = z_surf_max + clear

    # 应变参考高度：节点包络 or 外表面包络
    LZ_NODE = Lz
    LZ_SURF = z_surf_max - z_surf_min
    if str(STRAIN_REF_MODE).upper() == "SURFACE_Z":
        LZ_TARGET = LZ_SURF
    else:
        LZ_TARGET = LZ_NODE

    # 让“接触后”的有效压缩量 = TARGET_STRAIN * LZ_TARGET
    sign = -1.0 if TARGET_STRAIN < 0.0 else 1.0
    disp_top = TARGET_STRAIN * LZ_TARGET + sign * clear
    # 【替换补丁结束】

    # 新节点：8个角点 + 2个参考点
    n_bot1 = max_nid + 1
    n_bot2 = max_nid + 2
    n_bot3 = max_nid + 3
    n_bot4 = max_nid + 4
    n_top1 = max_nid + 5
    n_top2 = max_nid + 6
    n_top3 = max_nid + 7
    n_top4 = max_nid + 8
    RP_BOT = max_nid + 9
    RP_TOP = max_nid + 10


    inp_path = os.path.join(run_dir, job_name + ".inp")

    with open(inp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("*HEADING\n")
        f.write(f"PLATE_Z_RIGIDPLATES_{job_name}_EXPLICIT\n")
        f.write("*PREPRINT, ECHO=NO, MODEL=NO\n")

        # ---------------- Nodes ----------------
        f.write("*NODE\n")
        for n in nodes:
            f.write(f"{int(n[0])}, {float(n[1]):.15g}, {float(n[2]):.15g}, {float(n[3]):.15g}\n")

        # plate corner nodes（两块板都按从 +Z 看 CCW 排列 => 法向 +Z）
        f.write(f"{n_bot1}, {x0:.15g}, {y0:.15g}, {z_bot:.15g}\n")
        f.write(f"{n_bot2}, {x1:.15g}, {y0:.15g}, {z_bot:.15g}\n")
        f.write(f"{n_bot3}, {x1:.15g}, {y1:.15g}, {z_bot:.15g}\n")
        f.write(f"{n_bot4}, {x0:.15g}, {y1:.15g}, {z_bot:.15g}\n")

        f.write(f"{n_top1}, {x0:.15g}, {y0:.15g}, {z_top:.15g}\n")
        f.write(f"{n_top2}, {x1:.15g}, {y0:.15g}, {z_top:.15g}\n")
        f.write(f"{n_top3}, {x1:.15g}, {y1:.15g}, {z_top:.15g}\n")
        f.write(f"{n_top4}, {x0:.15g}, {y1:.15g}, {z_top:.15g}\n")

        # reference points（板中心）
        cx = 0.5 * (xm + xM)
        cy = 0.5 * (ym + yM)
        f.write(f"{RP_BOT}, {cx:.15g}, {cy:.15g}, {z_bot:.15g}\n")
        f.write(f"{RP_TOP}, {cx:.15g}, {cy:.15g}, {z_top:.15g}\n")

        # ---------------- Sets ----------------
        f.write("*NSET, NSET=ZMIN_FACE\n")
        write_id_list(f, zmin_ids)
        f.write("*NSET, NSET=ZMAX_FACE\n")
        write_id_list(f, zmax_ids)

        f.write("*NSET, NSET=RP_BOT\n")
        f.write(f"{RP_BOT}\n")
        f.write("*NSET, NSET=RP_TOP\n")
        f.write(f"{RP_TOP}\n")

        f.write("*NSET, NSET=RBM1\n")
        f.write(f"{rbm1}\n")
        if rbm2 is not None:
            f.write("*NSET, NSET=RBM2\n")
            f.write(f"{rbm2}\n")

        # ---------------- Elements ----------------
        # beams
        f.write("*ELEMENT, TYPE=B31\n")
        for eid, e in enumerate(elems, start=1):
            f.write(f"{eid}, {int(e[0])}, {int(e[1])}\n")
        f.write(f"*ELSET, ELSET=EALL, GENERATE\n1, {len(elems)}, 1\n")

        # rigid plates
        e_bot = len(elems) + 1
        e_top = len(elems) + 2
        f.write("*ELEMENT, TYPE=R3D4\n")
        f.write(f"{e_bot}, {n_bot1}, {n_bot2}, {n_bot3}, {n_bot4}\n")
        f.write(f"{e_top}, {n_top1}, {n_top2}, {n_top3}, {n_top4}\n")
        f.write("*ELSET, ELSET=BOT_PLATE\n")
        f.write(f"{e_bot}\n")
        f.write("*ELSET, ELSET=TOP_PLATE\n")
        f.write(f"{e_top}\n")

        # ---------------- Material ----------------
        f.write("*MATERIAL, NAME=MAT1\n")
        f.write("*ELASTIC\n")
        f.write(f"{YOUNG_MODULUS}, {POISSON_RATIO}\n")
        f.write("*PLASTIC\n")
        for sig, eps_p in PLASTIC_TABLE:
            f.write(f"{sig}, {eps_p}\n")
        f.write("*DENSITY\n")
        f.write(f"{MATERIAL_DENSITY}\n")
        f.write(f"*DAMPING, ALPHA=0.0, BETA={DAMPING_BETA}\n")

        # ---------------- Beam sections ----------------
        group_map = {(1.0, 0.0, 0.0): [], (0.0, 1.0, 0.0): [], (0.0, 0.0, 1.0): []}
        for eid, e in enumerate(elems, start=1):
            n1, n2 = int(e[0]), int(e[1])
            n1vec = choose_n1_for_element(ndict[n1], ndict[n2])
            group_map[n1vec].append(eid)

        def write_beam_section(elset_name, n1vec):
            f.write(f"*BEAM SECTION, SECTION=CIRC, MATERIAL=MAT1, ELSET={elset_name}\n")
            f.write(f"{BEAM_RADIUS}\n")
            f.write(f"{n1vec[0]}, {n1vec[1]}, {n1vec[2]}\n")

        if group_map[(1.0, 0.0, 0.0)]:
            f.write("*ELSET, ELSET=EORI_X\n")
            write_id_list(f, group_map[(1.0, 0.0, 0.0)])
            write_beam_section("EORI_X", (1.0, 0.0, 0.0))
        if group_map[(0.0, 1.0, 0.0)]:
            f.write("*ELSET, ELSET=EORI_Y\n")
            write_id_list(f, group_map[(0.0, 1.0, 0.0)])
            write_beam_section("EORI_Y", (0.0, 1.0, 0.0))
        if group_map[(0.0, 0.0, 1.0)]:
            f.write("*ELSET, ELSET=EORI_Z\n")
            write_id_list(f, group_map[(0.0, 0.0, 1.0)])
            write_beam_section("EORI_Z", (0.0, 0.0, 1.0))

        # ---------------- Rigid bodies ----------------
        f.write("*RIGID BODY, REF NODE=RP_BOT, ELSET=BOT_PLATE\n")
        f.write("*RIGID BODY, REF NODE=RP_TOP, ELSET=TOP_PLATE\n")

        # ---------------- Contact (General contact) ----------------
        # ★关键修复：不要用 TYPE=ELEMENT 给 B31 造 surface；改用 ALL EXTERIOR，
        # 让 Abaqus 自动把梁段/刚体面都纳入 general contact，否则会出现“只有贴板那几根动、力不传导”
        f.write("*SURFACE INTERACTION, NAME=GLOBAL_INT\n")
        f.write("*FRICTION\n")
        f.write(f"{FRICTION_COEFF}\n")

        f.write("*CONTACT\n")

        # 对 B31/刚体板最稳：让 Explicit 自动生成整个模型外表面域
        if ENABLE_SELF_CONTACT:
            f.write("*CONTACT INCLUSIONS, ALL EXTERIOR\n")
        else:
            # 不要自接触时，只保留板-梁
            f.write("*CONTACT INCLUSIONS\n")
            f.write("SURF_BEAMS, SURF_BOT\n")
            f.write("SURF_BEAMS, SURF_TOP\n")

        # 全域赋予摩擦属性（global assignment）
        f.write("*CONTACT PROPERTY ASSIGNMENT\n")
        f.write(", , GLOBAL_INT\n")


# ---------------- BC ----------------
        f.write("*BOUNDARY\n")
        # 底板不动
        f.write("RP_BOT, 1, 6, 0.0\n")
        


        # 顶板：锁横向 + 锁转动，只允许U3
        f.write("RP_TOP, 1, 2, 0.0\n")
        f.write("RP_TOP, 4, 6, 0.0\n")

        # ---------------- Step (Explicit) ----------------
        f.write("*AMPLITUDE, NAME=RAMP_LOAD, DEFINITION=SMOOTH STEP\n")
        f.write(f"0.0, 0.0, {STEP_TIME}, 1.0\n")

        f.write("*STEP, NAME=Explicit_QS, NLGEOM=YES\n")
        f.write("*DYNAMIC, EXPLICIT\n")
        f.write(f", {STEP_TIME}\n")

        # outputs：务必把 RP_TOP 的 CF 输出出来（显式接触力最靠谱）
        # ---------------- Outputs ----------------
        # 关键：输出“全模型节点位移 U”，否则 ODB 里板子/下层杆会看起来不动（其实是没写位移）
        f.write(f"*OUTPUT, FIELD, NUMBER INTERVAL={N_FIELD_FRAMES}\n")

        # 全节点位移（包含梁节点 + 刚性板角点节点 + RP）
        f.write("*NODE OUTPUT\n")
        f.write("U\n")

        # 参考点输出反力/接触力（显式里 CF 通常更靠谱）
        f.write("*NODE OUTPUT, NSET=RP_TOP\n")
        f.write("RF, CF\n")
        f.write("*NODE OUTPUT, NSET=RP_BOT\n")
        f.write("RF, CF\n")

        # 梁单元输出
        f.write("*ELEMENT OUTPUT, ELSET=EALL\n")
        f.write("S\n")
        # （建议）把 general contact 的接触力/接触应力写进 ODB，方便你直接看接触是否真的发生
        f.write("*CONTACT OUTPUT, GENERAL CONTACT\n")
        f.write("CSTRESS, CFORCE\n")


        f.write("*OUTPUT, HISTORY, FREQUENCY=1\n")
        f.write("*ENERGY OUTPUT, VARIABLE=PRESELECT\n")

        # loading: 顶板下压
        f.write("*BOUNDARY, AMPLITUDE=RAMP_LOAD\n")
        f.write(f"RP_TOP, 3, 3, {disp_top}\n")

        f.write("*END STEP\n")

    # extractor 里会用到 LZ / AREA / 接触判定参数
    extract_py = os.path.join(run_dir, "extract.py")
    with open(extract_py, "w", encoding="utf-8", newline="\n") as g:
        g.write(
            ODB_EXTRACTOR_TEMPLATE
            .replace("__LZ_TARGET__", f"{LZ_TARGET:.17g}")
            .replace("__AREA__", f"{area:.17g}")
            .replace("__FORCE_TOL__", f"{CONTACT_FORCE_TOL:.17g}")
            .replace("__FORCE_FRAC__", f"{CONTACT_FORCE_FRAC:.17g}")
            .replace("__DROP_PRECONTACT__", "True" if DROP_PRECONTACT else "False")
            .replace("__ZERO_STRESS_AT_CONTACT__", "True" if ZERO_STRESS_AT_CONTACT else "False")
        )

    return inp_path


# ==============================================================================
# 6) 执行与后处理
# ==============================================================================
def run_job(run_dir, job_name):
    for ext in [".lck",".odb",".dat",".msg",".sta",".sel",".prt",".com",".pac",".inp~",".log"]:
        p = os.path.join(run_dir, job_name + ext)
        if os.path.exists(p):
            try:
                os.remove(p)
            except:
                pass

# ... 前面的代码不变 ...
    cmd = f"{ABAQUS_CMD} job={job_name} input={job_name}.inp cpus={CPUS} interactive ask_delete=OFF"
    
    my_env = os.environ.copy()
    my_env["ABA_GCONT_POOL_SIZE"] = "1000"  # 甚至可以设得更大，比如 "50000"
    
    print(f"Running Abaqus (Standard QS): {job_name}")
    with open(os.path.join(run_dir, "run.log"), "w", encoding="utf-8", errors="ignore") as f:
        # 【关键修改】：这里增加了 env=my_env
        subprocess.call(cmd, shell=True, cwd=run_dir, stdout=f, stderr=subprocess.STDOUT, env=my_env)

def extract_curve(run_dir, job_name):
    out_csv = os.path.join(run_dir, "data.csv")
    log_file = os.path.join(run_dir, "extract.log")
    odb_path = os.path.join(run_dir, job_name + ".odb")

    if os.path.exists(out_csv):
        os.remove(out_csv)

    cmd = f'{ABAQUS_CMD} python extract.py "{odb_path}" "{out_csv}"'
    with open(log_file, "w", encoding="utf-8", errors="ignore") as lg:
        ret = subprocess.call(cmd, shell=True, cwd=run_dir, stdout=lg, stderr=subprocess.STDOUT)

    if (not os.path.exists(out_csv)) or (os.path.getsize(out_csv) == 0):
        print(f"Failed to extract data (ret={ret}). See: {log_file}")
        return []

    data = []
    with open(out_csv, "r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                try:
                    strain = float(parts[0])
                    disp = float(parts[1])
                    force = float(parts[2])
                    stress = float(parts[3])
                    data.append((strain, disp, force, stress))
                except:
                    pass
    return data

def smooth_curve(data, w, col=2):
    if w <= 1 or len(data) < w:
        return data
    half = w // 2
    out = []
    for i in range(len(data)):
        j0 = max(0, i-half)
        j1 = min(len(data), i+half+1)
        base = list(data[i])
        base[col] = sum(data[j][col] for j in range(j0, j1)) / float(j1-j0)
        out.append(tuple(base))
    return out

def configure_from_cli():
    global INPUT_FOLDER, OUTPUT_FOLDER, TARGET_FILE_INDEX, CPUS, ABAQUS_CMD, RUN_DIR

    parser = argparse.ArgumentParser(description="Run Abaqus FEM for one P222 geometry sample.")
    parser.add_argument("--input-folder", default=INPUT_FOLDER, help="Folder containing <index>.txt geometry files.")
    parser.add_argument("--output-folder", default=OUTPUT_FOLDER, help="Folder for extracted stress-strain curves.")
    parser.add_argument("--index", default=TARGET_FILE_INDEX, help="Geometry file stem, for example 0 for geometry/0.txt.")
    parser.add_argument("--cpus", type=int, default=CPUS, help="Abaqus CPU count.")
    parser.add_argument("--abaqus-cmd", default=ABAQUS_CMD, help="Abaqus command, for example abq2025 or abaqus.")
    args = parser.parse_args()

    INPUT_FOLDER = os.path.abspath(args.input_folder)
    OUTPUT_FOLDER = os.path.abspath(args.output_folder)
    TARGET_FILE_INDEX = str(args.index)
    CPUS = int(args.cpus)
    ABAQUS_CMD = args.abaqus_cmd

    RUN_DIR = os.path.join(OUTPUT_FOLDER, f"RUN_PLATE_Z_{TARGET_FILE_INDEX}")
    if os.path.exists(RUN_DIR):
        shutil.rmtree(RUN_DIR)
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def main():
    configure_from_cli()

    if shutil.which(ABAQUS_CMD) is None and not os.path.exists(ABAQUS_CMD):
        print(f"[Error] Cannot find Abaqus command: {ABAQUS_CMD}")
        return

    print(f"--- Processing {TARGET_FILE_INDEX} (MODE={MODE}) ---")
    print(f"[Input Folder]  {INPUT_FOLDER}")
    print(f"[Output Folder] {OUTPUT_FOLDER}")
    print(f"[Abaqus Cmd]    {ABAQUS_CMD}")
    nodes_raw, elems_raw, name = load_raw_data(TARGET_FILE_INDEX)
    if not nodes_raw:
        print("Load failed")
        return

    nodes, elems = merge_nodes_gridhash(nodes_raw, elems_raw, MERGE_TOL)

    bounds0, dims0 = compute_bounds_dims(nodes)
    print(f"[Base Bounds] {bounds0}")
    print(f"[Base Dims]   Lx={dims0[0]:.6g}, Ly={dims0[1]:.6g}, Lz={dims0[2]:.6g}")

    nodes, elems = stack_z_layers(nodes, elems, dims0[2], NZ_STACK)
    nodes, elems = merge_nodes_gridhash(nodes, elems, MERGE_TOL)

    nodes, elems = refine_mesh_pure(nodes, elems, n_seg=MESH_SEGMENTS)

    # ★关键：细分后再合并一次，避免结构断开（特别是节点“几乎重合但不完全重合”）
    nodes, elems = merge_nodes_gridhash(nodes, elems, MERGE_TOL)

    nodes, elems = remove_isolated_nodes(nodes, elems)

    bounds, dims = compute_bounds_dims(nodes)

    print(f"[Final Bounds] {bounds}")
    print(f"[Final Dims]   Lx={dims[0]:.6g}, Ly={dims[1]:.6g}, Lz={dims[2]:.6g}")


    job = f"Job_PLATE_Z_FREE_{TARGET_FILE_INDEX}"
    create_inp(RUN_DIR, nodes, elems, job, bounds, dims)


    run_job(RUN_DIR, job)

    curve_data = extract_curve(RUN_DIR, job)
    if curve_data:
        curve_data = smooth_curve(curve_data, POST_SMOOTH_WINDOW, col=2)

        out_csv = os.path.join(OUTPUT_FOLDER, f"{name}_PLATE_Z_XYPBC_StackZ{NZ_STACK}_curve.csv")
        with open(out_csv, "w", encoding="utf-8", newline="\n") as f:
            f.write("Strain,Disp_mm,Force_N,Stress_MPa\n")
            for strain, disp, force, stress in curve_data:
                f.write(f"{strain},{disp},{force},{stress}\n")
        print(f"Success. Saved: {out_csv}")
    else:
        print("No curve data extracted.")

if __name__ == "__main__":
    main()





