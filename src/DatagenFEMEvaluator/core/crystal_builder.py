# batch_build_same_name.py
# -*- coding: utf-8 -*-
import os
import re
from decimal import Decimal, getcontext, ROUND_HALF_UP
from pathlib import Path

getcontext().prec = 50

INPUT_FOLDER  = r"C:\Users\admin\Desktop\3Dtruss\P222_1\Batch_Output_Files"
OUTPUT_FOLDER = r"C:\Users\admin\Desktop\3Dtruss\P222_1\Batch_Output_Crystal_422"  # 你要保存到的新文件夹

NX, NY, NZ = 2, 4, 2  # 晶体尺寸（需要 3x3x3 就改成 3,3,3）

# 跳过已存在输出（断点续跑）。不想跳过就设 False（会覆盖）
SKIP_IF_EXISTS = True


RE_NODE_BLOCK = re.compile(r"node_data\s*=\s*\[(.*?)\]\s*element_conn\s*=", flags=re.S)
RE_ELEM_BLOCK = re.compile(r"element_conn\s*=\s*\[(.*?)\]\s*\Z", flags=re.S)
RE_NODE_LINE  = re.compile(r"\[\s*(\d+)\s*,\s*([^\],]+)\s*,\s*([^\],]+)\s*,\s*([^\],]+)\s*\]")
RE_ELEM_LINE  = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")


def _max_decimal_places(num_strs):
    mx = 0
    for s in num_strs:
        s = s.strip()
        if "e" in s.lower():
            continue
        if "." in s:
            mx = max(mx, len(s.split(".", 1)[1]))
    return mx


def _quant_from_places(p):
    if p <= 0:
        return Decimal("1")
    return Decimal("0." + "0" * (p - 1) + "1")


def parse_unitcell(txt: str):
    m1 = RE_NODE_BLOCK.search(txt)
    if not m1:
        raise ValueError("未找到 node_data 块")
    node_block = m1.group(1)

    m2 = RE_ELEM_BLOCK.search(txt)
    if not m2:
        raise ValueError("未找到 element_conn 块")
    elem_block = m2.group(1)

    node_matches = RE_NODE_LINE.findall(node_block)
    if not node_matches:
        raise ValueError("node_data 解析失败：没有节点行")

    node_data = []
    xs_s, ys_s, zs_s = [], [], []
    for nid, xs, ys, zs in node_matches:
        xs, ys, zs = xs.strip(), ys.strip(), zs.strip()
        node_data.append([int(nid), xs, ys, zs])
        xs_s.append(xs); ys_s.append(ys); zs_s.append(zs)

    elem_matches = RE_ELEM_LINE.findall(elem_block)
    if not elem_matches:
        raise ValueError("element_conn 解析失败：没有连接行")

    element_conn = [[int(a), int(b)] for a, b in elem_matches]

    px = _max_decimal_places(xs_s)
    py = _max_decimal_places(ys_s)
    pz = _max_decimal_places(zs_s)
    return node_data, element_conn, (px, py, pz)


def build_crystal(node_data, element_conn, places, nx, ny, nz):
    px, py, pz = places
    qx, qy, qz = _quant_from_places(px), _quant_from_places(py), _quant_from_places(pz)

    nodes_by_id = {}
    xs, ys, zs = [], [], []
    for nid, xs_s, ys_s, zs_s in node_data:
        xd, yd, zd = Decimal(xs_s), Decimal(ys_s), Decimal(zs_s)
        nodes_by_id[nid] = (xd, yd, zd)
        xs.append(xd); ys.append(yd); zs.append(zd)

    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs) - min(zs)

    coord_to_gid = {}
    global_nodes = []
    next_gid = 1

    elem_set = set()
    global_elems = []

    for i in range(nx):
        ox = dx * Decimal(i)
        for j in range(ny):
            oy = dy * Decimal(j)
            for k in range(nz):
                oz = dz * Decimal(k)

                local_map = {}
                for nid, (xd, yd, zd) in nodes_by_id.items():
                    x2 = (xd + ox).quantize(qx, rounding=ROUND_HALF_UP)
                    y2 = (yd + oy).quantize(qy, rounding=ROUND_HALF_UP)
                    z2 = (zd + oz).quantize(qz, rounding=ROUND_HALF_UP)

                    x2s = format(x2, "f") if px > 0 else str(x2)
                    y2s = format(y2, "f") if py > 0 else str(y2)
                    z2s = format(z2, "f") if pz > 0 else str(z2)

                    key = (x2s, y2s, z2s)
                    gid = coord_to_gid.get(key)
                    if gid is None:
                        gid = next_gid
                        next_gid += 1
                        coord_to_gid[key] = gid
                        global_nodes.append((gid, x2s, y2s, z2s))
                    local_map[nid] = gid

                for a, b in element_conn:
                    ga, gb = local_map[a], local_map[b]
                    ekey = (ga, gb) if ga < gb else (gb, ga)
                    if ekey not in elem_set:
                        elem_set.add(ekey)
                        global_elems.append([ga, gb])

    return global_nodes, global_elems


def format_output(global_nodes, global_elems, header_text):
    lines = []
    lines.append("# ==========================================")
    lines.append(f"# Data ID: {header_text}")
    lines.append("# ==========================================")
    lines.append("node_data = [")
    for gid, xs, ys, zs in global_nodes:
        lines.append(f"    [{gid}, {xs}, {ys}, {zs}],")
    lines.append("]\n")
    lines.append("element_conn = [")
    for a, b in global_elems:
        lines.append(f"    [{a}, {b}],")
    lines.append("]")
    return "\n".join(lines) + "\n"


def main():
    in_root = Path(INPUT_FOLDER)
    out_root = Path(OUTPUT_FOLDER)
    out_root.mkdir(parents=True, exist_ok=True)

    err_log = out_root / "errors.log"
    processed = skipped = failed = 0

    with err_log.open("w", encoding="utf-8") as elog:
        for entry in os.scandir(in_root):
            if not entry.is_file():
                continue
            if not entry.name.lower().endswith(".txt"):
                continue

            in_path = Path(entry.path)
            out_path = out_root / entry.name  # ✅ 文件名完全不变：还是 1.txt

            if SKIP_IF_EXISTS and out_path.exists():
                skipped += 1
                continue

            try:
                txt = in_path.read_text(encoding="utf-8", errors="ignore")
                node_data, element_conn, places = parse_unitcell(txt)
                global_nodes, global_elems = build_crystal(node_data, element_conn, places, NX, NY, NZ)

                header = in_path.stem  # ✅ header 也不乱改
                out_text = format_output(global_nodes, global_elems, header)
                out_path.write_text(out_text, encoding="utf-8")
                processed += 1

                if (processed + skipped + failed) % 2000 == 0:
                    total = processed + skipped + failed
                    print(f"[Progress] total={total} processed={processed} skipped={skipped} failed={failed}")

            except Exception as e:
                failed += 1
                elog.write(f"[FAIL] {in_path} -> {e}\n")

    print("完成！")
    print(f"- 输入: {in_root}")
    print(f"- 输出: {out_root}")
    print(f"- processed={processed}, skipped={skipped}, failed={failed}")
    print(f"- 错误日志: {err_log}")


if __name__ == "__main__":
    main()
