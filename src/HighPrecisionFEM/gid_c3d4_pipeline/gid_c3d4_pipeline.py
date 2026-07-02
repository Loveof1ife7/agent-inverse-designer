# -*- coding: utf-8 -*-
"""GraphMetaMat 结构 -> C3D4 实体有限元 一体化脚本（自包含，无外部依赖）

一次运行从"导入结构"到"出对比图"全程走完：
  导入结构(nodes.csv/struts.csv/meta.json)
    -> [可选] N×N×N 阵列(--array, 默认 1 即输入单结构直接算) + 合并周期节点
    -> gmsh OCC 圆柱杆+节点球布尔融合建 C3D4 四面体网格(网格尺寸按半径缩放到 ~86% 体积)
    -> 写 Abaqus/Explicit inp(线弹性 Es=7, nu=0.3, 底固定+顶刚性板压 30%,
       无摩擦自接触, NLGEOM, TIE 刚性板, 固定质量缩放)
    -> abaqus 直接调核心求解
    -> 从 odb 提取应力应变曲线(σ=F/A=(N*L)^2, ε=u/(N*L))
    -> 和 meta.json 里数据集参考曲线对比出图

默认 --array 1 输入单个结构直接仿真; 若要复现数据集 6 条参考(参考是 2×2×2 压出来的),
用 --array 2。归一化面积 A=(N*L)^2 与高度 Lz=N*L 自动跟随 --array。
线弹性位移控制下 σ 精确正比于 Es, 故用 Es=4 跑后 ×7/4 与 Es=7 直接跑完全等价。

用法见同目录 README_gid_c3d4_pipeline.md。
示例:
  python gid_c3d4_pipeline.py --struct-dir 导出结构_6条/gid2979              # 单结构直接算
  python gid_c3d4_pipeline.py --struct-dir 导出结构_6条/gid2979 --array 2    # 2×2×2 对齐数据集参考
  python gid_c3d4_pipeline.py --struct-dir 导出结构_gid2978 --mesh-only
"""
from __future__ import annotations
import os, sys, csv, json, math, argparse, subprocess
from dataclasses import dataclass
from pathlib import Path


# ======================= 严格对齐设定(数据集 FE) =======================
CELL_DEFAULT = 10.0          # 单胞宽度 mm(meta.unit_cell_size_L_mm)
N_ARRAY_DEFAULT = 1          # 默认单结构直接算(不堆叠); --array 可设 2 对齐数据集 2x2x2 参考
E_S = 7.0                    # 数据集存储模量(论文最终材料 4, 数据集用 7)
POISSON = 0.3
# 归一化 A=(N*L)^2, Lz=N*L 运行时按 --array 与单胞尺寸自动算(N=2 时即 A=400,Lz=20 对齐数据集)
# 网格细度: CL=k*radius, 复现 gid2978 那档 ~86% 体积填充(相对细度与半径无关)
K_MIN = 0.06 / 0.19345406878514765   # ≈0.3101
K_MAX = 0.15 / 0.19345406878514765   # ≈0.7754
MERGE_TOL = 1e-3


# ======================= odb 提取脚本模板(abaqus python) =======================
ODB_EXTRACTOR_TEMPLATE = r"""# -*- coding: utf-8 -*-
import os
import sys
from odbAccess import openOdb

LZ = __LZ_TARGET__
AREA = __AREA__
FORCE_TOL = __FORCE_TOL__
FORCE_FRAC = __FORCE_FRAC__
DROP_PRECONTACT = __DROP_PRECONTACT__
ZERO_STRESS_AT_CONTACT = __ZERO_STRESS_AT_CONTACT__

def get_nodeset(odb, set_name):
    key = set_name.upper()
    assembly = odb.rootAssembly
    if key in assembly.nodeSets:
        return assembly.nodeSets[key]
    for inst in assembly.instances.values():
        if key in inst.nodeSets:
            return inst.nodeSets[key]
    for part in odb.parts.values():
        if key in part.nodeSets:
            return part.nodeSets[key]
    return None

def find_whole_model_history_region(step):
    for key in step.historyRegions.keys():
        upper = key.upper()
        if "WHOLE MODEL" in upper or "ASSEMBLY" in upper:
            return step.historyRegions[key]
    return None

def build_time_value_map(hist_output):
    return {float(t): float(v) for t, v in hist_output.data}

def nearest_value(t, value_map):
    if not value_map:
        return None
    best_t = min(value_map.keys(), key=lambda item: abs(item - t))
    return value_map[best_t]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: abaqus python extract.py job.odb out.csv\n")
        sys.exit(2)
    odb_path = sys.argv[1]
    out_csv = sys.argv[2]
    if not os.path.exists(odb_path):
        sys.stderr.write("ODB not found: %s\n" % odb_path)
        sys.exit(3)
    odb = openOdb(odb_path, readOnly=True)
    step = odb.steps[list(odb.steps.keys())[-1]]
    top_set = get_nodeset(odb, "RP_TOP")
    if top_set is None:
        sys.stderr.write("Cannot find RP_TOP in odb.\n"); odb.close(); sys.exit(10)
    whole = find_whole_model_history_region(step)
    t2allke = {}; t2allie = {}; t2allwk = {}
    if whole is not None:
        hist = whole.historyOutputs
        if "ALLKE" in hist: t2allke = build_time_value_map(hist["ALLKE"])
        if "ALLIE" in hist: t2allie = build_time_value_map(hist["ALLIE"])
        if "ALLWK" in hist: t2allwk = build_time_value_map(hist["ALLWK"])
    records = []
    for frame in step.frames:
        outputs = frame.fieldOutputs
        if "U" not in outputs: continue
        rf_key = "RF" if "RF" in outputs else ("CF" if "CF" in outputs else "")
        if not rf_key: continue
        top_u = outputs["U"].getSubset(region=top_set).values
        top_rf = outputs[rf_key].getSubset(region=top_set).values
        if not top_u or not top_rf: continue
        u3_avg = sum(float(value.data[2]) for value in top_u) / float(len(top_u))
        rf3_sum = sum(float(value.data[2]) for value in top_rf)
        t = float(frame.frameValue)
        records.append((t, u3_avg, rf3_sum,
                        nearest_value(t, t2allke), nearest_value(t, t2allie), nearest_value(t, t2allwk)))
    if not records:
        sys.stderr.write("No data written. Check FIELD output requests (U, RF/CF).\n"); odb.close(); sys.exit(20)
    max_rf = max(abs(record[2]) for record in records)
    threshold = max(FORCE_TOL, FORCE_FRAC * max_rf)
    contact_idx = 0
    for idx, record in enumerate(records):
        if abs(record[2]) >= threshold: contact_idx = idx; break
    u3_contact = records[contact_idx][1]; rf3_contact = records[contact_idx][2]
    wrote = 0
    with open(out_csv, "w") as handle:
        handle.write("Strain,Disp_mm,Force_N,Stress_MPa,ALLKE,ALLIE,ALLWK,KE_over_IE,Time_s\n")
        for idx, record in enumerate(records):
            t, u3_avg, rf3_sum, allke, allie, allwk = record
            if idx < contact_idx:
                if DROP_PRECONTACT: continue
                disp_eff = 0.0; rf3_eff = 0.0
            else:
                disp_eff = abs(u3_avg - u3_contact)
                rf3_eff = rf3_sum - rf3_contact if ZERO_STRESS_AT_CONTACT else rf3_sum
            strain = disp_eff / LZ if LZ != 0.0 else float("nan")
            force = -rf3_eff
            stress = -rf3_eff / AREA if AREA != 0.0 else float("nan")
            if allke is None: allke = float("nan")
            if allie is None: allie = float("nan")
            if allwk is None: allwk = float("nan")
            ke_over_ie = allke / allie if allie == allie and allie != 0.0 else float("nan")
            handle.write("%g,%g,%g,%g,%g,%g,%g,%g,%g\n" % (strain, disp_eff, force, stress, allke, allie, allwk, ke_over_ie, t))
            wrote += 1
    odb.close()
    if wrote == 0:
        sys.stderr.write("No data written after contact filtering.\n"); sys.exit(20)
"""


# ======================= 配置 =======================
@dataclass
class Config:
    abaqus_cmd: str = ""             # 空则自动找 abq2025/abq2022/abaqus
    cpus: int = 8
    young_modulus: float = E_S
    poisson_ratio: float = POISSON
    material_density: float = 1.11e-9
    damping_beta: float = 0.0
    friction_coeff: float = 0.0
    target_strain: float = -0.30
    step_time: float = 1.0
    n_field_frames: int = 50
    plate_scale_xy: float = 2.0
    contact_force_tol: float = 1e-8
    contact_force_frac: float = 1e-4
    drop_precontact: bool = True
    zero_stress_at_contact: bool = True
    timeout_seconds: int | None = 21600


# ======================= 结构导入 / 建网格 =======================
def load_gid(nodes_csv, struts_csv):
    """读数据集导出格式: nodes.csv(node_id,x_mm,y_mm,z_mm) struts.csv(node_i,node_j,...)"""
    nid = {}
    for r in csv.DictReader(open(nodes_csv, encoding='utf-8-sig')):
        nid[int(r['node_id'])] = (float(r['x_mm']), float(r['y_mm']), float(r['z_mm']))
    struts = [(int(r['node_i']), int(r['node_j'])) for r in csv.DictReader(open(struts_csv, encoding='utf-8-sig'))]
    return nid, struts


def tessellate(nid, struts, cell, n=2):
    """单胞平移复制成 n×n×n 阵列(节点尚未合并)。"""
    nodes, elems, gid = [], [], 0
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                off = (ix * cell, iy * cell, iz * cell); idmap = {}
                for oid, (x, y, z) in nid.items():
                    gid += 1; idmap[oid] = gid
                    nodes.append([gid, x + off[0], y + off[1], z + off[2]])
                for a, b in struts:
                    elems.append([idmap[a], idmap[b]])
    return nodes, elems


def merge_nodes_gridhash(nodes_raw, elems_raw, tol):
    """网格哈希合并重合节点(阵列交界处周期共享节点)。"""
    def key3(x, y, z):
        return (int(round(x / tol)), int(round(y / tol)), int(round(z / tol)))
    buckets = {}; node_map = {}; new_nodes = []
    shifts = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    nid = 1
    for node in nodes_raw:
        old_id = int(node[0]); x, y, z = float(node[1]), float(node[2]), float(node[3])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        key = key3(x, y, z); found = None
        for dx, dy, dz in shifts:
            nk = (key[0] + dx, key[1] + dy, key[2] + dz)
            for eid, ex, ey, ez in buckets.get(nk, []):
                if (x - ex) ** 2 + (y - ey) ** 2 + (z - ez) ** 2 <= tol ** 2:
                    found = eid; break
            if found is not None: break
        if found is None:
            new_nodes.append([nid, x, y, z]); buckets.setdefault(key, []).append((nid, x, y, z))
            node_map[old_id] = nid; nid += 1
        else:
            node_map[old_id] = found
    new_elems = []
    for a, b in elems_raw:
        n1, n2 = node_map.get(int(a)), node_map.get(int(b))
        if n1 and n2 and n1 != n2:
            new_elems.append([n1, n2])
    return new_nodes, new_elems


def build_mesh(node_map, elems, r, cl_min, cl_max):
    """OCC 每杆圆柱 + 每节点球 -> 布尔融合 -> C3D4 tet(尺寸 cl_min..cl_max)。返回 nodes3d,tets,填充体积。"""
    import gmsh
    gmsh.initialize(); gmsh.option.setNumber('General.Terminal', 0); gmsh.model.add('t')
    occ = gmsh.model.occ; vols = []
    for a, b in elems:
        x1, y1, z1 = node_map[int(a)]; x2, y2, z2 = node_map[int(b)]
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        if math.sqrt(dx * dx + dy * dy + dz * dz) < 1e-6: continue
        vols.append(occ.addCylinder(x1, y1, z1, dx, dy, dz, r))
    for _nid, (x, y, z) in node_map.items():
        vols.append(occ.addSphere(x, y, z, r))
    occ.synchronize()
    if len(vols) > 1:
        occ.fuse([(3, vols[0])], [(3, v) for v in vols[1:]])
    occ.synchronize()
    gmsh.option.setNumber('Mesh.MeshSizeMin', cl_min); gmsh.option.setNumber('Mesh.MeshSizeMax', cl_max)
    gmsh.option.setNumber('Mesh.Optimize', 1); gmsh.option.setNumber('Mesh.OptimizeNetgen', 1)
    gmsh.model.mesh.generate(3); gmsh.model.mesh.optimize('Netgen')
    ntags, coords, _ = gmsh.model.mesh.getNodes(); coords = coords.reshape(-1, 3)
    idof = {int(t): i + 1 for i, t in enumerate(ntags)}
    nodes3d = [[i + 1, float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])] for i in range(len(ntags))]
    _, en = gmsh.model.mesh.getElementsByType(4); en = en.reshape(-1, 4)
    tets = [[idof[int(en[k, j])] for j in range(4)] for k in range(len(en))]
    vol = 0.0
    for t in tets:
        p = [coords[t[j] - 1] for j in range(4)]; v1 = p[1] - p[0]; v2 = p[2] - p[0]; v3 = p[3] - p[0]
        vol += abs(float(v1[0] * (v2[1] * v3[2] - v2[2] * v3[1]) - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0]) + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]))) / 6.0
    gmsh.finalize()
    return nodes3d, tets, vol


# ======================= 写 inp / 求解 / 提取 =======================
def _write_id_list(h, ids, per_line=16):
    for i in range(0, len(ids), per_line):
        h.write(", ".join(str(int(v)) for v in ids[i:i + per_line]) + "\n")


def _write_extractor(run_path, lz_target, area, cfg):
    (run_path / "extract.py").write_text(
        ODB_EXTRACTOR_TEMPLATE
        .replace("__LZ_TARGET__", f"{lz_target:.17g}")
        .replace("__AREA__", f"{area:.17g}")
        .replace("__FORCE_TOL__", f"{cfg.contact_force_tol:.17g}")
        .replace("__FORCE_FRAC__", f"{cfg.contact_force_frac:.17g}")
        .replace("__DROP_PRECONTACT__", "True" if cfg.drop_precontact else "False")
        .replace("__ZERO_STRESS_AT_CONTACT__", "True" if cfg.zero_stress_at_contact else "False"),
        encoding="utf-8")


def write_solid_inp(run_dir, nodes3d, tets, job_name, cfg, r, nom_area, nom_lz):
    """写 Abaqus/Explicit 平板压缩 inp: C3D4 实体 + 上下 R3D4 刚性板(TIE 端部) + 全局无摩擦自接触
    + 固定质量缩放 + 顶板 SMOOTH STEP 压 30%。"""
    run_path = Path(run_dir); run_path.mkdir(parents=True, exist_ok=True)
    xs = [n[1] for n in nodes3d]; ys = [n[2] for n in nodes3d]; zs = [n[3] for n in nodes3d]
    xm, xM, ym, yM, zm, zM = min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)
    lx, ly = xM - xm, yM - ym
    area = max(nom_area, 1e-12)
    band = r  # 端部一个半径厚的帽子绑到刚性板
    zmin_ids = [int(n[0]) for n in nodes3d if n[3] <= zm + band]
    zmax_ids = [int(n[0]) for n in nodes3d if n[3] >= zM - band]
    if not zmin_ids or not zmax_ids:
        raise RuntimeError('empty tie band')
    N = len(nodes3d)
    n_bot = [N + i for i in range(1, 5)]; n_top = [N + i for i in range(5, 9)]
    rp_bot = N + 9; rp_top = N + 10
    cx, cy = 0.5 * (xm + xM), 0.5 * (ym + yM)
    margin = max(2.0 * r, 0.02 * max(lx, ly))
    hx = 0.5 * lx * cfg.plate_scale_xy + margin; hy = 0.5 * ly * cfg.plate_scale_xy + margin
    x0, x1 = cx - hx, cx + hx; y0, y1 = cy - hy, cy + hy
    clear = 1e-4 * r; z_bot = zm - clear; z_top = zM + clear
    lz_target = nom_lz
    sign = -1.0 if cfg.target_strain < 0 else 1.0
    disp_top = cfg.target_strain * lz_target + sign * clear
    inp_path = run_path / f'{job_name}.inp'
    with inp_path.open('w', encoding='utf-8', newline='\n') as h:
        h.write('*HEADING\n'); h.write(f'PLATE_Z_C3D4_{job_name}_EXPLICIT\n')
        h.write('*PREPRINT, ECHO=NO, MODEL=NO\n')
        h.write('*NODE\n')
        for n in nodes3d:
            h.write(f'{int(n[0])}, {n[1]:.15g}, {n[2]:.15g}, {n[3]:.15g}\n')
        for nid, x, y, z in [
            (n_bot[0], x0, y0, z_bot), (n_bot[1], x1, y0, z_bot), (n_bot[2], x1, y1, z_bot), (n_bot[3], x0, y1, z_bot),
            (n_top[0], x0, y0, z_top), (n_top[1], x1, y0, z_top), (n_top[2], x1, y1, z_top), (n_top[3], x0, y1, z_top),
            (rp_bot, cx, cy, z_bot), (rp_top, cx, cy, z_top)]:
            h.write(f'{nid}, {x:.15g}, {y:.15g}, {z:.15g}\n')
        h.write('*NSET, NSET=ZMIN_BAND\n'); _write_id_list(h, zmin_ids)
        h.write('*NSET, NSET=ZMAX_BAND\n'); _write_id_list(h, zmax_ids)
        h.write('*NSET, NSET=RP_BOT\n'); h.write(f'{rp_bot}\n')
        h.write('*NSET, NSET=RP_TOP\n'); h.write(f'{rp_top}\n')
        h.write('*ELEMENT, TYPE=C3D4\n')
        for eid, t in enumerate(tets, start=1):
            h.write(f'{eid}, {t[0]}, {t[1]}, {t[2]}, {t[3]}\n')
        h.write('*ELSET, ELSET=EALL, GENERATE\n'); h.write(f'1, {len(tets)}, 1\n')
        e_bot = len(tets) + 1; e_top = len(tets) + 2
        h.write('*ELEMENT, TYPE=R3D4\n')
        h.write(f'{e_bot}, {n_bot[0]}, {n_bot[1]}, {n_bot[2]}, {n_bot[3]}\n')
        h.write(f'{e_top}, {n_top[3]}, {n_top[2]}, {n_top[1]}, {n_top[0]}\n')
        h.write('*ELSET, ELSET=BOT_PLATE\n'); h.write(f'{e_bot}\n')
        h.write('*ELSET, ELSET=TOP_PLATE\n'); h.write(f'{e_top}\n')
        h.write('*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1\n')
        h.write('*MATERIAL, NAME=MAT1\n')
        h.write('*ELASTIC\n'); h.write(f'{cfg.young_modulus}, {cfg.poisson_ratio}\n')
        h.write('*DENSITY\n'); h.write(f'{cfg.material_density}\n')
        h.write(f'*DAMPING, ALPHA=0.0, BETA={cfg.damping_beta}\n')
        h.write('*RIGID BODY, REF NODE=RP_BOT, ELSET=BOT_PLATE, TIE NSET=ZMIN_BAND\n')
        h.write('*RIGID BODY, REF NODE=RP_TOP, ELSET=TOP_PLATE, TIE NSET=ZMAX_BAND\n')
        h.write('*SURFACE INTERACTION, NAME=GLOBAL_INT\n')
        h.write('*FRICTION\n'); h.write(f'{cfg.friction_coeff}\n')
        h.write('*BOUNDARY\n')
        h.write('RP_BOT, 1, 6, 0.0\n'); h.write('RP_TOP, 1, 2, 0.0\n'); h.write('RP_TOP, 4, 6, 0.0\n')
        h.write('*AMPLITUDE, NAME=RAMP_LOAD, DEFINITION=SMOOTH STEP\n')
        h.write(f'0.0, 0.0, {cfg.step_time}, 1.0\n')
        h.write('*STEP, NAME=Explicit_QS, NLGEOM=YES\n')
        h.write('*DYNAMIC, EXPLICIT\n'); h.write(f', {cfg.step_time}\n')
        h.write('*FIXED MASS SCALING, DT=5.0E-6, TYPE=BELOW MIN\n')
        h.write('*CONTACT\n'); h.write('*CONTACT INCLUSIONS, ALL EXTERIOR\n')
        h.write('*CONTACT PROPERTY ASSIGNMENT\n'); h.write(', , GLOBAL_INT\n')
        h.write(f'*OUTPUT, FIELD, NUMBER INTERVAL={cfg.n_field_frames}\n')
        h.write('*NODE OUTPUT\nU\n')
        h.write('*NODE OUTPUT, NSET=RP_TOP\nRF, CF\n')
        h.write('*NODE OUTPUT, NSET=RP_BOT\nRF, CF\n')
        h.write('*ELEMENT OUTPUT, ELSET=EALL\nS\n')
        h.write('*OUTPUT, HISTORY, FREQUENCY=1\n')
        h.write('*ENERGY OUTPUT, VARIABLE=PRESELECT\n')
        h.write('*BOUNDARY, AMPLITUDE=RAMP_LOAD\n')
        h.write(f'RP_TOP, 3, 3, {disp_top}\n')
        h.write('*END STEP\n')
    _write_extractor(run_path, lz_target, area, cfg)
    return str(inp_path), area, lz_target


def find_abaqus_command(abaqus_cmd=""):
    import shutil
    if abaqus_cmd or os.getenv("ABAQUS_CMD", ""):
        return abaqus_cmd or os.getenv("ABAQUS_CMD", "")
    return shutil.which("abq2025") or shutil.which("abq2022") or shutil.which("abaqus") or ""


def run_job(run_dir, job_name, cfg):
    run_path = Path(run_dir)
    abaqus_cmd = find_abaqus_command(cfg.abaqus_cmd)
    if not abaqus_cmd:
        raise FileNotFoundError("找不到 abaqus 命令,请设 --abaqus 或 ABAQUS_CMD 环境变量。")
    for suf in (".lck", ".odb", ".dat", ".msg", ".sta", ".sel", ".prt", ".com", ".pac", ".log"):
        p = run_path / f"{job_name}{suf}"
        if p.exists(): p.unlink()
    command = f"{abaqus_cmd} job={job_name} input={job_name}.inp cpus={int(cfg.cpus)} interactive ask_delete=OFF"
    env = os.environ.copy(); env["ABA_GCONT_POOL_SIZE"] = "1000"
    with (run_path / "run.log").open("w", encoding="utf-8", errors="ignore") as handle:
        done = subprocess.run(command, shell=True, cwd=str(run_path), stdout=handle,
                              stderr=subprocess.STDOUT, env=env, check=False, timeout=cfg.timeout_seconds)
    return int(done.returncode)


def extract_curve(run_dir, job_name, cfg):
    run_path = Path(run_dir).resolve()
    abaqus_cmd = find_abaqus_command(cfg.abaqus_cmd)
    out_csv = run_path / "data.csv"
    if out_csv.exists(): out_csv.unlink()
    odb = run_path / f"{job_name}.odb"
    command = f'{abaqus_cmd} python extract.py "{odb.resolve()}" "{out_csv.resolve()}"'
    with (run_path / "extract.log").open("w", encoding="utf-8", errors="ignore") as handle:
        subprocess.run(command, shell=True, cwd=str(run_path), stdout=handle, stderr=subprocess.STDOUT, check=False)
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return []
    data = []
    for row in csv.DictReader(open(out_csv, encoding="utf-8", errors="ignore")):
        try:
            data.append((float(row["Strain"]), float(row["Disp_mm"]), float(row["Force_N"]), float(row["Stress_MPa"])))
        except (KeyError, TypeError, ValueError):
            continue
    return data


# ======================= 对比出图 =======================
def plot_compare(sdir, curve, meta, out_png, gid, arr=1):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ref_csv = os.path.join(sdir, 'reference_curve.csv')
    plt.figure(figsize=(8.5, 6))
    if os.path.exists(ref_csv):
        rs, rv = [], []
        for r in csv.DictReader(open(ref_csv, encoding='utf-8-sig')):
            rs.append(float(r['strain'])); rv.append(abs(float(r['stress_normalized_by_Es'])))
        plt.plot(rs, rv, 'k-', lw=2.6, label=f'Reference dataset (peak {max(rv):.3e})')
    fs = [abs(c[0]) for c in curve]; fv = [abs(c[3]) for c in curve]
    if fv:
        plt.plot(fs, fv, 'r-', lw=2.2, label=f'C3D4 solid FE (peak {max(fv):.3e})')
    plt.xlabel('Strain'); plt.ylabel('Stress (MPa, raw Es=7 scale)')
    plt.title(f'{gid}  {arr}x{arr}x{arr}  Es=7  frictionless -- C3D4 vs reference')
    plt.legend(loc='upper left'); plt.grid(True, alpha=0.3)
    plt.xlim(0, 0.30); plt.ylim(bottom=0)
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()
    return out_png


# ======================= 主流程 =======================
def main():
    ap = argparse.ArgumentParser(description='GraphMetaMat 结构 -> C3D4 实体 FE 一体化')
    ap.add_argument('--struct-dir', required=True, help='含 nodes.csv/struts.csv/meta.json 的结构目录')
    ap.add_argument('--out', default='', help='输出目录(默认 struct-dir/c3d4_run)')
    ap.add_argument('--cpus', type=int, default=8)
    ap.add_argument('--abaqus', default='', help='abaqus 命令路径(默认自动找 abq2025/abq2022/abaqus)')
    ap.add_argument('--k-min', type=float, default=K_MIN, help='网格 CL_min = k_min*radius')
    ap.add_argument('--k-max', type=float, default=K_MAX, help='网格 CL_max = k_max*radius')
    ap.add_argument('--young', type=float, default=E_S, help='材料模量(默认数据集 7)')
    ap.add_argument('--array', type=int, default=N_ARRAY_DEFAULT,
                    help='N×N×N 阵列堆叠数,默认1(输入单结构直接算);设2对齐数据集2x2x2参考。归一化 A/Lz 自动跟随')
    ap.add_argument('--mesh-only', action='store_true', help='只建网格写 inp, 不求解')
    ap.add_argument('--no-plot', action='store_true')
    a = ap.parse_args()

    sdir = a.struct_dir
    meta = json.load(open(os.path.join(sdir, 'meta.json'), encoding='utf-8'))
    gid = f"gid{meta.get('gid', Path(sdir).name)}"
    r = float(meta['strut_radius_mm'])
    cell = float(meta.get('unit_cell_size_L_mm', CELL_DEFAULT))
    cl_min, cl_max = a.k_min * r, a.k_max * r
    arr = a.array
    nom_lz = arr * cell                 # ε = u / Lz
    nom_area = (arr * cell) ** 2        # σ = F / A
    ref_peak = (meta.get('curve', {}) or {}).get('stress_max_Es7scale') \
        or meta.get('reference_curve_max_stress_normalized')
    print(f'[{gid}] r={r:.5f} cell={cell} array={arr}x{arr}x{arr} A={nom_area:.0f} Lz={nom_lz:.0f} '
          f'CL={cl_min:.4f}..{cl_max:.4f} ref_peak={ref_peak}', flush=True)
    if arr != 2 and ref_peak:
        print(f'[{gid}] 注意: 数据集参考曲线是 2x2x2 压出来的, 当前 array={arr} 与参考不同尺度, 对比图仅供参考', flush=True)

    nid, struts = load_gid(os.path.join(sdir, 'nodes.csv'), os.path.join(sdir, 'struts.csv'))
    tn, te = tessellate(nid, struts, cell, arr)
    nodes, elems = merge_nodes_gridhash(tn, te, MERGE_TOL)
    seen = set(); dedup = []
    for aa, bb in elems:
        k = tuple(sorted((int(aa), int(bb))))
        if k not in seen: seen.add(k); dedup.append([int(aa), int(bb)])
    node_map = {int(n[0]): (float(n[1]), float(n[2]), float(n[3])) for n in nodes}
    print(f'[{gid}] merged nodes={len(nodes)} struts={len(dedup)} (building mesh...)', flush=True)
    nodes3d, tets, vol = build_mesh(node_map, dedup, r, cl_min, cl_max)
    nomvol = float(meta.get('relative_density_rho', 0)) * (arr * cell) ** 3
    pct = (vol / nomvol * 100) if nomvol else 0.0
    print(f'[{gid}] mesh nodes={len(nodes3d)} C3D4={len(tets)} volume={vol:.1f} mm^3 ({pct:.0f}% of rho*V={nomvol:.1f})', flush=True)

    out_root = a.out or os.path.join(sdir, 'c3d4_run')
    cfg = Config(abaqus_cmd=a.abaqus, cpus=a.cpus, young_modulus=a.young)
    run_dir = os.path.join(out_root, f'{gid}_c3d4'); job = f'Job_{gid}_c3d4'
    write_solid_inp(run_dir, nodes3d, tets, job, cfg, r, nom_area, nom_lz)
    print(f'[{gid}] inp written -> {run_dir}', flush=True)
    if a.mesh_only:
        print(f'[{gid}] === mesh-only done ===', flush=True); return

    print(f'[{gid}] === solving C3D4 (cpus={a.cpus}, Es={a.young}) ===', flush=True)
    rcode = run_job(run_dir, job, cfg); print(f'[{gid}] abaqus exit {rcode}', flush=True)
    curve = extract_curve(run_dir, job, cfg) if rcode == 0 else []
    peak = max((abs(s[3]) for s in curve), default=None)
    ratio = (peak / ref_peak) if (peak and ref_peak) else None
    print(f'[{gid}] curve pts={len(curve)} peak={peak} ref={ref_peak} FE/ref={ratio}', flush=True)
    if curve and not a.no_plot:
        png = plot_compare(sdir, curve, meta, os.path.join(out_root, f'{gid}_compare.png'), gid, arr)
        print(f'[{gid}] plot -> {png}', flush=True)
    print(f'[{gid}] === done ===', flush=True)


if __name__ == '__main__':
    main()
