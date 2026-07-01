from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PLASTIC_TABLE = (
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
)


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
        sys.stderr.write("Cannot find RP_TOP in odb.\n")
        odb.close()
        sys.exit(10)

    whole = find_whole_model_history_region(step)
    t2allke = {}
    t2allie = {}
    t2allwk = {}
    if whole is not None:
        hist = whole.historyOutputs
        if "ALLKE" in hist:
            t2allke = build_time_value_map(hist["ALLKE"])
        if "ALLIE" in hist:
            t2allie = build_time_value_map(hist["ALLIE"])
        if "ALLWK" in hist:
            t2allwk = build_time_value_map(hist["ALLWK"])

    records = []
    for frame in step.frames:
        outputs = frame.fieldOutputs
        if "U" not in outputs:
            continue
        rf_key = "RF" if "RF" in outputs else ("CF" if "CF" in outputs else "")
        if not rf_key:
            continue
        top_u = outputs["U"].getSubset(region=top_set).values
        top_rf = outputs[rf_key].getSubset(region=top_set).values
        if not top_u or not top_rf:
            continue
        u3_avg = sum(float(value.data[2]) for value in top_u) / float(len(top_u))
        rf3_sum = sum(float(value.data[2]) for value in top_rf)
        t = float(frame.frameValue)
        records.append((
            t,
            u3_avg,
            rf3_sum,
            nearest_value(t, t2allke),
            nearest_value(t, t2allie),
            nearest_value(t, t2allwk),
        ))

    if not records:
        sys.stderr.write("No data written. Check FIELD output requests (U, RF/CF).\n")
        odb.close()
        sys.exit(20)

    max_rf = max(abs(record[2]) for record in records)
    threshold = max(FORCE_TOL, FORCE_FRAC * max_rf)
    contact_idx = 0
    for idx, record in enumerate(records):
        if abs(record[2]) >= threshold:
            contact_idx = idx
            break

    u3_contact = records[contact_idx][1]
    rf3_contact = records[contact_idx][2]
    wrote = 0
    with open(out_csv, "w") as handle:
        handle.write("Strain,Disp_mm,Force_N,Stress_MPa,ALLKE,ALLIE,ALLWK,KE_over_IE,Time_s\n")
        for idx, record in enumerate(records):
            t, u3_avg, rf3_sum, allke, allie, allwk = record
            if idx < contact_idx:
                if DROP_PRECONTACT:
                    continue
                disp_eff = 0.0
                rf3_eff = 0.0
            else:
                disp_eff = abs(u3_avg - u3_contact)
                rf3_eff = rf3_sum - rf3_contact if ZERO_STRESS_AT_CONTACT else rf3_sum
            strain = disp_eff / LZ if LZ != 0.0 else float("nan")
            force = -rf3_eff
            stress = -rf3_eff / AREA if AREA != 0.0 else float("nan")
            if allke is None:
                allke = float("nan")
            if allie is None:
                allie = float("nan")
            if allwk is None:
                allwk = float("nan")
            ke_over_ie = allke / allie if allie == allie and allie != 0.0 else float("nan")
            handle.write("%g,%g,%g,%g,%g,%g,%g,%g,%g\n" % (
                strain, disp_eff, force, stress, allke, allie, allwk, ke_over_ie, t
            ))
            wrote += 1

    odb.close()
    if wrote == 0:
        sys.stderr.write("No data written after contact filtering.\n")
        sys.exit(20)
"""


@dataclass(frozen=True)
class AbaqusFEMConfig:
    output_root: str = ""
    abaqus_cmd: str = ""
    cpus: int = 8
    target_strain: float = -0.30
    step_time: float = 1.0
    n_field_frames: int = 50
    mesh_segments: int = 3
    merge_tol: float = 1e-3
    face_tol: float = 2e-3
    beam_radius: float = 1.0
    young_modulus: float = 8.925
    poisson_ratio: float = 0.48
    material_density: float = 1.11e-9
    damping_beta: float = 0.0
    friction_coeff: float = 0.3
    plate_scale_xy: float = 2.0
    strain_ref_mode: str = "NODE_Z"
    contact_force_tol: float = 1e-8
    contact_force_frac: float = 1e-4
    drop_precontact: bool = True
    zero_stress_at_contact: bool = True
    enable_self_contact: bool = True
    run_solver: bool = True
    reuse_existing_curve: bool = True
    timeout_seconds: int | None = None
    plastic_table: tuple[tuple[float, float], ...] = field(default_factory=lambda: tuple(PLASTIC_TABLE))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AbaqusFEMRunResult:
    structure_id: str
    status: str
    run_dir: str
    inp_path: str = ""
    curve_path: str = ""
    evaluated_property: dict[str, float] = field(default_factory=dict)
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_abaqus_command(abaqus_cmd: str = "") -> str:
    configured = abaqus_cmd or os.getenv("ABAQUS_CMD", "")
    if configured:
        return configured
    return shutil.which("abq2025") or shutil.which("abq2022") or shutil.which("abaqus") or ""


def load_truss_txt(path: str | os.PathLike[str]) -> tuple[list[list[float]], list[list[int]], str]:
    txt_path = Path(path)
    namespace: dict[str, Any] = {}
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    name = txt_path.stem
    for line in raw.splitlines():
        if "# Data ID:" in line:
            name = line.split(":", 1)[-1].strip() or name
            break
    exec(raw, {}, namespace)
    nodes = namespace.get("node_data") or []
    elems = namespace.get("element_conn") or []
    return nodes, elems, name


def _elem_nodes(element: list[Any] | tuple[Any, ...]) -> tuple[int, int]:
    if len(element) >= 3:
        return int(element[-2]), int(element[-1])
    if len(element) == 2:
        return int(element[0]), int(element[1])
    raise ValueError(f"Bad element record: {element}")


def merge_nodes_gridhash(nodes_raw: list[list[Any]], elems_raw: list[list[Any]], tol: float) -> tuple[list[list[float]], list[list[int]]]:
    def key3(x: float, y: float, z: float) -> tuple[int, int, int]:
        return (int(round(x / tol)), int(round(y / tol)), int(round(z / tol)))

    buckets: dict[tuple[int, int, int], list[tuple[int, float, float, float]]] = {}
    node_map: dict[int, int] = {}
    new_nodes: list[list[float]] = []
    neighbor_shifts = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    nid = 1

    for node in nodes_raw:
        old_id = int(node[0])
        x, y, z = float(node[1]), float(node[2]), float(node[3])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        key = key3(x, y, z)
        found_id = None
        for dx, dy, dz in neighbor_shifts:
            neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
            for existing_id, ex, ey, ez in buckets.get(neighbor_key, []):
                if (x - ex) ** 2 + (y - ey) ** 2 + (z - ez) ** 2 <= tol**2:
                    found_id = existing_id
                    break
            if found_id is not None:
                break
        if found_id is None:
            new_nodes.append([nid, x, y, z])
            buckets.setdefault(key, []).append((nid, x, y, z))
            node_map[old_id] = nid
            nid += 1
        else:
            node_map[old_id] = found_id

    new_elems = []
    for element in elems_raw:
        try:
            a, b = _elem_nodes(element)
            n1, n2 = node_map[a], node_map[b]
            if n1 != n2:
                new_elems.append([n1, n2])
        except (KeyError, TypeError, ValueError):
            continue
    return new_nodes, new_elems


def refine_mesh(nodes: list[list[float]], elems: list[list[int]], n_segments: int) -> tuple[list[list[float]], list[list[int]]]:
    if n_segments <= 1:
        return nodes, elems
    max_nid = max(int(node[0]) for node in nodes)
    node_map = {int(node[0]): (float(node[1]), float(node[2]), float(node[3])) for node in nodes}
    new_nodes = list(nodes)
    new_elems: list[list[int]] = []
    for n1_id, n2_id in elems:
        p1, p2 = node_map[int(n1_id)], node_map[int(n2_id)]
        chain = [int(n1_id)]
        for index in range(1, n_segments):
            ratio = index / float(n_segments)
            max_nid += 1
            new_nodes.append([
                max_nid,
                p1[0] + (p2[0] - p1[0]) * ratio,
                p1[1] + (p2[1] - p1[1]) * ratio,
                p1[2] + (p2[2] - p1[2]) * ratio,
            ])
            chain.append(max_nid)
        chain.append(int(n2_id))
        for index in range(n_segments):
            new_elems.append([chain[index], chain[index + 1]])
    return new_nodes, new_elems


def remove_isolated_nodes(nodes: list[list[float]], elems: list[list[int]]) -> tuple[list[list[float]], list[list[int]]]:
    used = {int(node_id) for elem in elems for node_id in elem}
    return [node for node in nodes if int(node[0]) in used], elems


def compute_bounds_dims(nodes: list[list[float]]) -> tuple[tuple[float, float, float, float, float, float], tuple[float, float, float]]:
    xs = [float(node[1]) for node in nodes]
    ys = [float(node[2]) for node in nodes]
    zs = [float(node[3]) for node in nodes]
    bounds = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
    dims = (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    return bounds, dims


def _dot(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> float:
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]


def choose_n1_for_element(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> tuple[float, float, float]:
    direction = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    norm = math.sqrt(_dot(direction, direction))
    if norm <= 0.0:
        return (1.0, 0.0, 0.0)
    unit = (direction[0] / norm, direction[1] / norm, direction[2] / norm)
    for candidate in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        if abs(_dot(unit, candidate)) < 0.95:
            return candidate
    return (0.0, 0.0, 1.0)


def compute_beam_surface_z_extents(nodes: list[list[float]], elems: list[list[int]], radius: float) -> tuple[float, float]:
    node_map = {int(node[0]): (float(node[1]), float(node[2]), float(node[3])) for node in nodes}
    zmin = 1.0e100
    zmax = -1.0e100
    for n1, n2 in elems:
        x1, y1, z1 = node_map[int(n1)]
        x2, y2, z2 = node_map[int(n2)]
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length <= 0.0:
            continue
        uz = dz / length
        rz = radius * math.sqrt(max(0.0, 1.0 - uz * uz))
        zmin = min(zmin, min(z1, z2) - rz)
        zmax = max(zmax, max(z1, z2) + rz)
    if not math.isfinite(zmin) or not math.isfinite(zmax):
        zs = [float(node[3]) for node in nodes]
        return min(zs), max(zs)
    return zmin, zmax


def write_id_list(handle, ids: list[int], per_line: int = 16) -> None:
    for index in range(0, len(ids), per_line):
        handle.write(", ".join(str(int(value)) for value in ids[index : index + per_line]) + "\n")


def _write_extractor(run_dir: Path, lz_target: float, area: float, config: AbaqusFEMConfig) -> Path:
    path = run_dir / "extract.py"
    path.write_text(
        ODB_EXTRACTOR_TEMPLATE
        .replace("__LZ_TARGET__", f"{lz_target:.17g}")
        .replace("__AREA__", f"{area:.17g}")
        .replace("__FORCE_TOL__", f"{config.contact_force_tol:.17g}")
        .replace("__FORCE_FRAC__", f"{config.contact_force_frac:.17g}")
        .replace("__DROP_PRECONTACT__", "True" if config.drop_precontact else "False")
        .replace("__ZERO_STRESS_AT_CONTACT__", "True" if config.zero_stress_at_contact else "False"),
        encoding="utf-8",
    )
    return path


def create_plate_compression_inp(
    run_dir: str | os.PathLike[str],
    nodes: list[list[float]],
    elems: list[list[int]],
    job_name: str,
    config: AbaqusFEMConfig | None = None,
) -> str:
    config = config or AbaqusFEMConfig()
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    bounds, dims = compute_bounds_dims(nodes)
    xm, xM, ym, yM, zm, zM = bounds
    lx, ly, lz = dims
    area = max(lx * ly, 1.0e-12)
    node_map = {int(node[0]): (float(node[1]), float(node[2]), float(node[3])) for node in nodes}
    zmin_ids = [nid for nid, (_x, _y, z) in node_map.items() if abs(z - zm) <= config.face_tol]
    zmax_ids = [nid for nid, (_x, _y, z) in node_map.items() if abs(z - zM) <= config.face_tol]
    if not zmin_ids or not zmax_ids:
        raise RuntimeError("ZMIN/ZMAX node sets are empty. Check face_tol or geometry.")

    rbm1 = zmin_ids[0]
    ax, ay, _ = node_map[rbm1]
    rbm2 = None
    best_d2 = -1.0
    for nid in zmin_ids:
        if nid == rbm1:
            continue
        x, y, _ = node_map[nid]
        d2 = (x - ax) ** 2 + (y - ay) ** 2
        if d2 > best_d2:
            best_d2 = d2
            rbm2 = nid

    max_nid = max(int(node[0]) for node in nodes)
    margin = max(2.0 * config.beam_radius, 0.02 * max(lx, ly))
    cx, cy = 0.5 * (xm + xM), 0.5 * (ym + yM)
    half_x = 0.5 * lx * config.plate_scale_xy + margin
    half_y = 0.5 * ly * config.plate_scale_xy + margin
    x0, x1 = cx - half_x, cx + half_x
    y0, y1 = cy - half_y, cy + half_y
    surf_zmin, surf_zmax = compute_beam_surface_z_extents(nodes, elems, config.beam_radius)
    clear = 1e-4 * config.beam_radius
    lz_surf = surf_zmax - surf_zmin
    use_surface_z = config.strain_ref_mode.upper() == "SURFACE_Z"
    if use_surface_z:
        z_bot = surf_zmin - clear
        z_top = surf_zmax + clear
        lz_target = lz_surf
    else:
        # General contact for B31 beams may not engage at the nominal circular
        # section radius. In NODE_Z mode, keep plates near node extrema so the
        # requested strain actually reaches the truss centerline geometry.
        z_bot = zm - clear
        z_top = zM + clear
        lz_target = lz
    sign = -1.0 if config.target_strain < 0.0 else 1.0
    disp_top = config.target_strain * lz_target + sign * clear

    n_bot = [max_nid + i for i in range(1, 5)]
    n_top = [max_nid + i for i in range(5, 9)]
    rp_bot = max_nid + 9
    rp_top = max_nid + 10
    inp_path = run_path / f"{job_name}.inp"

    with inp_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("*HEADING\n")
        handle.write(f"PLATE_Z_RIGIDPLATES_{job_name}_EXPLICIT\n")
        handle.write("*PREPRINT, ECHO=NO, MODEL=NO\n")
        handle.write("*NODE\n")
        for node in nodes:
            handle.write(f"{int(node[0])}, {float(node[1]):.15g}, {float(node[2]):.15g}, {float(node[3]):.15g}\n")
        plate_nodes = [
            (n_bot[0], x0, y0, z_bot), (n_bot[1], x1, y0, z_bot), (n_bot[2], x1, y1, z_bot), (n_bot[3], x0, y1, z_bot),
            (n_top[0], x0, y0, z_top), (n_top[1], x1, y0, z_top), (n_top[2], x1, y1, z_top), (n_top[3], x0, y1, z_top),
            (rp_bot, cx, cy, z_bot), (rp_top, cx, cy, z_top),
        ]
        for nid, x, y, z in plate_nodes:
            handle.write(f"{nid}, {x:.15g}, {y:.15g}, {z:.15g}\n")

        handle.write("*NSET, NSET=ZMIN_FACE\n")
        write_id_list(handle, zmin_ids)
        handle.write("*NSET, NSET=ZMAX_FACE\n")
        write_id_list(handle, zmax_ids)
        handle.write("*NSET, NSET=RP_BOT\n")
        handle.write(f"{rp_bot}\n")
        handle.write("*NSET, NSET=RP_TOP\n")
        handle.write(f"{rp_top}\n")
        handle.write("*NSET, NSET=RBM1\n")
        handle.write(f"{rbm1}\n")
        if rbm2 is not None:
            handle.write("*NSET, NSET=RBM2\n")
            handle.write(f"{rbm2}\n")

        handle.write("*ELEMENT, TYPE=B31\n")
        for eid, elem in enumerate(elems, start=1):
            handle.write(f"{eid}, {int(elem[0])}, {int(elem[1])}\n")
        handle.write(f"*ELSET, ELSET=EALL, GENERATE\n1, {len(elems)}, 1\n")
        e_bot = len(elems) + 1
        e_top = len(elems) + 2
        handle.write("*ELEMENT, TYPE=R3D4\n")
        handle.write(f"{e_bot}, {n_bot[0]}, {n_bot[1]}, {n_bot[2]}, {n_bot[3]}\n")
        # R3D4 contact is normal-side sensitive. The bottom plate should face
        # upward into the truss, while the top plate should face downward.
        handle.write(f"{e_top}, {n_top[3]}, {n_top[2]}, {n_top[1]}, {n_top[0]}\n")
        handle.write("*ELSET, ELSET=BOT_PLATE\n")
        handle.write(f"{e_bot}\n")
        handle.write("*ELSET, ELSET=TOP_PLATE\n")
        handle.write(f"{e_top}\n")

        handle.write("*MATERIAL, NAME=MAT1\n")
        handle.write("*ELASTIC\n")
        handle.write(f"{config.young_modulus}, {config.poisson_ratio}\n")
        handle.write("*PLASTIC\n")
        for stress, plastic_strain in config.plastic_table:
            handle.write(f"{stress}, {plastic_strain}\n")
        handle.write("*DENSITY\n")
        handle.write(f"{config.material_density}\n")
        handle.write(f"*DAMPING, ALPHA=0.0, BETA={config.damping_beta}\n")

        orientation_groups = {(1.0, 0.0, 0.0): [], (0.0, 1.0, 0.0): [], (0.0, 0.0, 1.0): []}
        for eid, elem in enumerate(elems, start=1):
            n1, n2 = int(elem[0]), int(elem[1])
            orientation_groups[choose_n1_for_element(node_map[n1], node_map[n2])].append(eid)
        for n1vec, eids in orientation_groups.items():
            if not eids:
                continue
            elset_name = f"EORI_{'XYZ'[list(orientation_groups).index(n1vec)]}"
            handle.write(f"*ELSET, ELSET={elset_name}\n")
            write_id_list(handle, eids)
            handle.write(f"*BEAM SECTION, SECTION=CIRC, MATERIAL=MAT1, ELSET={elset_name}\n")
            handle.write(f"{config.beam_radius}\n")
            handle.write(f"{n1vec[0]}, {n1vec[1]}, {n1vec[2]}\n")

        handle.write("*RIGID BODY, REF NODE=RP_BOT, ELSET=BOT_PLATE\n")
        handle.write("*RIGID BODY, REF NODE=RP_TOP, ELSET=TOP_PLATE\n")
        handle.write("*SURFACE INTERACTION, NAME=GLOBAL_INT\n")
        handle.write("*FRICTION\n")
        handle.write(f"{config.friction_coeff}\n")

        handle.write("*BOUNDARY\n")
        handle.write("RP_BOT, 1, 6, 0.0\n")
        handle.write("RP_TOP, 1, 2, 0.0\n")
        handle.write("RP_TOP, 4, 6, 0.0\n")
        handle.write("*AMPLITUDE, NAME=RAMP_LOAD, DEFINITION=SMOOTH STEP\n")
        handle.write(f"0.0, 0.0, {config.step_time}, 1.0\n")
        handle.write("*STEP, NAME=Explicit_QS, NLGEOM=YES\n")
        handle.write("*DYNAMIC, EXPLICIT\n")
        handle.write(f", {config.step_time}\n")
        handle.write("*CONTACT\n")
        if config.enable_self_contact:
            handle.write("*CONTACT INCLUSIONS, ALL EXTERIOR\n")
        else:
            handle.write("*CONTACT INCLUSIONS\n")
            handle.write("SURF_BEAMS, SURF_BOT\n")
            handle.write("SURF_BEAMS, SURF_TOP\n")
        handle.write("*CONTACT PROPERTY ASSIGNMENT\n")
        handle.write(", , GLOBAL_INT\n")
        handle.write(f"*OUTPUT, FIELD, NUMBER INTERVAL={config.n_field_frames}\n")
        handle.write("*NODE OUTPUT\nU\n")
        handle.write("*NODE OUTPUT, NSET=RP_TOP\nRF, CF\n")
        handle.write("*NODE OUTPUT, NSET=RP_BOT\nRF, CF\n")
        handle.write("*ELEMENT OUTPUT, ELSET=EALL\nS\n")
        handle.write("*CONTACT OUTPUT, GENERAL CONTACT\nCSTRESS, CFORCE\n")
        handle.write("*OUTPUT, HISTORY, FREQUENCY=1\n")
        handle.write("*ENERGY OUTPUT, VARIABLE=PRESELECT\n")
        handle.write("*BOUNDARY, AMPLITUDE=RAMP_LOAD\n")
        handle.write(f"RP_TOP, 3, 3, {disp_top}\n")
        handle.write("*END STEP\n")

    _write_extractor(run_path, lz_target, area, config)
    return str(inp_path)


def run_job(run_dir: str | os.PathLike[str], job_name: str, config: AbaqusFEMConfig) -> int:
    run_path = Path(run_dir)
    abaqus_cmd = find_abaqus_command(config.abaqus_cmd)
    if not abaqus_cmd:
        raise FileNotFoundError("Cannot find Abaqus command. Set ABAQUS_CMD or add abq2025/abq2022/abaqus to PATH.")
    for suffix in (".lck", ".odb", ".dat", ".msg", ".sta", ".sel", ".prt", ".com", ".pac", ".inp~", ".log"):
        path = run_path / f"{job_name}{suffix}"
        if path.exists():
            path.unlink()
    command = f"{abaqus_cmd} job={job_name} input={job_name}.inp cpus={int(config.cpus)} interactive ask_delete=OFF"
    env = os.environ.copy()
    env["ABA_GCONT_POOL_SIZE"] = "1000"
    with (run_path / "run.log").open("w", encoding="utf-8", errors="ignore") as handle:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(run_path),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
            timeout=config.timeout_seconds,
        )
    return int(completed.returncode)


def extract_curve(run_dir: str | os.PathLike[str], job_name: str, config: AbaqusFEMConfig) -> list[tuple[float, float, float, float]]:
    run_path = Path(run_dir).resolve()
    abaqus_cmd = find_abaqus_command(config.abaqus_cmd)
    if not abaqus_cmd:
        raise FileNotFoundError("Cannot find Abaqus command for ODB extraction.")
    out_csv = run_path / "data.csv"
    if out_csv.exists():
        out_csv.unlink()
    odb_path = run_path / f"{job_name}.odb"
    command = f'{abaqus_cmd} python extract.py "{odb_path.resolve()}" "{out_csv.resolve()}"'
    with (run_path / "extract.log").open("w", encoding="utf-8", errors="ignore") as handle:
        subprocess.run(command, shell=True, cwd=str(run_path), stdout=handle, stderr=subprocess.STDOUT, check=False)
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return []
    return _read_curve_csv(out_csv)


def _read_curve_csv(path: str | os.PathLike[str]) -> list[tuple[float, float, float, float]]:
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    data = []
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                data.append((
                    float(row["Strain"]),
                    float(row["Disp_mm"]),
                    float(row["Force_N"]),
                    float(row["Stress_MPa"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return data


def _edge_length_sum(nodes: list[list[float]], elems: list[list[int]]) -> float:
    node_map = {int(node[0]): (float(node[1]), float(node[2]), float(node[3])) for node in nodes}
    total = 0.0
    for n1, n2 in elems:
        if int(n1) not in node_map or int(n2) not in node_map:
            continue
        p1, p2 = node_map[int(n1)], node_map[int(n2)]
        total += math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    return total


def _geometry_metrics(nodes: list[list[float]], elems: list[list[int]]) -> dict[str, float]:
    bounds, dims = compute_bounds_dims(nodes)
    volume = max(dims[0] * dims[1] * dims[2], 1.0)
    length_sum = _edge_length_sum(nodes, elems)
    return {
        "node_count": float(len(nodes)),
        "edge_count": float(len(elems)),
        "edge_length_sum": float(length_sum),
        "volume_proxy": float(volume),
        "density_proxy": float(min(length_sum / volume, 1.0)),
        "connectivity_ratio": float(len(elems) / max(len(nodes), 1)),
        "bounds_x_min": bounds[0],
        "bounds_x_max": bounds[1],
        "bounds_y_min": bounds[2],
        "bounds_y_max": bounds[3],
        "bounds_z_min": bounds[4],
        "bounds_z_max": bounds[5],
    }


def metrics_from_curve(
    curve: list[tuple[float, float, float, float]],
    geometry_metrics: dict[str, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    valid = [(abs(strain), disp, force, stress) for strain, disp, force, stress in curve if math.isfinite(strain) and math.isfinite(stress)]
    if not valid:
        evaluated = {
            "stiffness_proxy": 0.0,
            "density_proxy": geometry_metrics.get("density_proxy", 0.0),
            "initial_modulus": 0.0,
            "peak_stress": 0.0,
            "energy_absorption": 0.0,
        }
        return evaluated, {"curve_points": 0, **geometry_metrics}
    valid.sort(key=lambda row: row[0])
    peak_stress = max(abs(row[3]) for row in valid)
    final_strain = max(row[0] for row in valid)
    initial = [row for row in valid if 0.0 < row[0] <= min(0.05, max(final_strain, 0.0))]
    if len(initial) >= 2:
        s0, stress0 = initial[0][0], abs(initial[0][3])
        s1, stress1 = initial[-1][0], abs(initial[-1][3])
        initial_modulus = (stress1 - stress0) / max(s1 - s0, 1.0e-12)
    else:
        first = valid[0]
        initial_modulus = abs(first[3]) / max(first[0], 1.0e-12) if first[0] > 0 else 0.0
    energy = 0.0
    for lhs, rhs in zip(valid, valid[1:]):
        energy += 0.5 * (abs(lhs[3]) + abs(rhs[3])) * abs(rhs[0] - lhs[0])
    density_proxy = geometry_metrics.get("density_proxy", 0.0)
    evaluated = {
        "stiffness_proxy": float(initial_modulus),
        "density_proxy": float(density_proxy),
        "initial_modulus": float(initial_modulus),
        "peak_stress": float(peak_stress),
        "energy_absorption": float(energy),
        "final_strain": float(final_strain),
    }
    raw = {
        "curve_points": len(valid),
        "peak_stress": peak_stress,
        "initial_modulus": initial_modulus,
        "energy_absorption": energy,
        "final_strain": final_strain,
        **geometry_metrics,
    }
    return evaluated, raw


def evaluate_truss_file(
    structure_path: str | os.PathLike[str],
    structure_id: str = "",
    output_root: str | os.PathLike[str] | None = None,
    config: AbaqusFEMConfig | None = None,
) -> AbaqusFEMRunResult:
    config = config or AbaqusFEMConfig()
    path = Path(structure_path)
    run_root = Path(output_root or config.output_root or path.parent / "fem_runs")
    safe_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in (structure_id or path.stem))[:120]
    run_dir = run_root / safe_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        nodes_raw, elems_raw, _name = load_truss_txt(path)
        nodes, elems = merge_nodes_gridhash(nodes_raw, elems_raw, config.merge_tol)
        nodes, elems = refine_mesh(nodes, elems, config.mesh_segments)
        nodes, elems = merge_nodes_gridhash(nodes, elems, config.merge_tol)
        nodes, elems = remove_isolated_nodes(nodes, elems)
        if not nodes or not elems:
            return AbaqusFEMRunResult(str(structure_id), "invalid_geometry", str(run_dir), error="empty nodes or elements")
        geometry_metrics = _geometry_metrics(nodes, elems)
        job_name = f"Job_PLATE_Z_{safe_id}"
        inp_path = create_plate_compression_inp(run_dir, nodes, elems, job_name, config)
    except Exception as exc:
        return AbaqusFEMRunResult(str(structure_id), "setup_failed", str(run_dir), error=str(exc))

    if not config.run_solver:
        return AbaqusFEMRunResult(
            str(structure_id),
            "input_generated",
            str(run_dir),
            inp_path=inp_path,
            raw_metrics={"evaluator": "abaqus_plate_compression", "run_solver": False, **geometry_metrics},
        )

    cached_curve_path = run_dir / "data.csv"
    if config.reuse_existing_curve:
        cached_curve = _read_curve_csv(cached_curve_path)
        if cached_curve:
            evaluated_property, raw_metrics = metrics_from_curve(cached_curve, geometry_metrics)
            return AbaqusFEMRunResult(
                str(structure_id),
                "success",
                str(run_dir),
                inp_path=inp_path,
                curve_path=str(cached_curve_path),
                evaluated_property=evaluated_property,
                raw_metrics={
                    "evaluator": "abaqus_plate_compression",
                    "cached_curve": True,
                    **raw_metrics,
                },
            )

    if not find_abaqus_command(config.abaqus_cmd):
        return AbaqusFEMRunResult(
            str(structure_id),
            "abaqus_unavailable",
            str(run_dir),
            inp_path=inp_path,
            raw_metrics={"evaluator": "abaqus_plate_compression", **geometry_metrics},
            error="Cannot find Abaqus command. Set ABAQUS_CMD or add abq2025/abq2022/abaqus to PATH.",
        )

    try:
        exit_code = run_job(run_dir, job_name, config)
        curve = extract_curve(run_dir, job_name, config) if exit_code == 0 else []
        evaluated_property, raw_metrics = metrics_from_curve(curve, geometry_metrics)
        curve_path = str(run_dir / "data.csv") if (run_dir / "data.csv").exists() else ""
        status = "success" if curve else "extraction_failed"
        return AbaqusFEMRunResult(
            str(structure_id),
            status,
            str(run_dir),
            inp_path=inp_path,
            curve_path=curve_path,
            evaluated_property=evaluated_property,
            raw_metrics={
                "evaluator": "abaqus_plate_compression",
                "abaqus_exit_code": exit_code,
                **raw_metrics,
            },
            error="" if curve else "Abaqus finished but no curve data was extracted.",
        )
    except Exception as exc:
        return AbaqusFEMRunResult(
            str(structure_id),
            "run_failed",
            str(run_dir),
            inp_path=inp_path,
            raw_metrics={"evaluator": "abaqus_plate_compression", **geometry_metrics},
            error=str(exc),
        )


__all__ = [
    "AbaqusFEMConfig",
    "AbaqusFEMRunResult",
    "create_plate_compression_inp",
    "evaluate_truss_file",
    "find_abaqus_command",
    "metrics_from_curve",
]
