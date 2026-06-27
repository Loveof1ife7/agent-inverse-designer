import os
import ast
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 配置区域
# =========================
TXT_PATH = r"C:\Users\admin\Desktop\3Dtruss\P222_1\Batch_Output_Files_422\9944.txt"

# 【修改点 1】：改为 True 以显示编号
SHOW_NODE_LABELS = True  
LABEL_LIMIT = 500  # 稍微调大一点，确保你的16个节点能显示

def extract_bracket_list(text: str, var_name: str) -> str:
    """
    从文本中提取 var_name = [ ... ] 的完整方括号块
    """
    key = var_name
    idx = text.find(key)
    if idx < 0:
        raise ValueError(f"找不到变量名: {var_name}")

    eq = text.find("=", idx)
    if eq < 0:
        raise ValueError(f"找不到 '=': {var_name}")

    start = text.find("[", eq)
    if start < 0:
        raise ValueError(f"找不到 '[': {var_name}")

    depth = 0
    end = None
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        raise ValueError(f"方括号未闭合: {var_name}")

    return text[start:end]


def load_truss_from_txt(path: str):
    # 如果为了测试方便，你可以把下面的 try-except 块打开，
    # 当找不到文件时自动使用你提供的那段测试数据。
    if not os.path.exists(path):
         print(f"注意：找不到文件 {path}，正在使用默认测试数据...")
         # 这里放你提供的测试数据，以便在没有文件时也能运行
         node_block = """[
            [1, 0.000000, -13.469052, -6.734526],
            [2, 0.000000, -13.469052, 6.734526],
            [3, 0.000000, -13.469052, 13.469052],
            [4, 0.000000, 13.469052, -13.469052],
            [5, 0.000000, 13.469052, -6.734526],
            [6, 0.000000, 13.469052, 6.734526],
            [7, 6.734526, -13.469052, 0.000000],
            [8, 6.734526, 0.000000, -13.469052],
            [9, 6.734526, 0.000000, 13.469052],
            [10, 6.734526, 13.469052, 0.000000],
            [11, 13.469052, -13.469052, -13.469052],
            [12, 13.469052, -13.469052, -6.734526],
            [13, 13.469052, -13.469052, 6.734526],
            [14, 13.469052, 13.469052, -6.734526],
            [15, 13.469052, 13.469052, 6.734526],
            [16, 13.469052, 13.469052, 13.469052],
        ]"""
         edge_block = """[
            [1, 7], [1, 8], [1, 12], [2, 3], [2, 9], [2, 13], [3, 9],
            [4, 5], [4, 8], [5, 8], [5, 14], [6, 9], [6, 10], [6, 15],
            [7, 13], [8, 11], [8, 12], [8, 14], [9, 13], [9, 15], [9, 16],
            [10, 14], [11, 12], [15, 16]
        ]"""
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        node_block = extract_bracket_list(text, "node_data")
        edge_block = extract_bracket_list(text, "element_conn")

    node_data = ast.literal_eval(node_block)
    element_conn = ast.literal_eval(edge_block)

    node_arr = np.array(node_data, dtype=float)
    if node_arr.ndim != 2 or node_arr.shape[1] < 4:
        raise ValueError("node_data 格式不对")

    node_ids = node_arr[:, 0].astype(int)
    xyz = node_arr[:, 1:4].astype(float)

    edge_arr = np.array(element_conn, dtype=int)
    if edge_arr.size == 0:
        edge_arr = edge_arr.reshape(0, 2)
    
    return node_ids, xyz, edge_arr


def validate_and_report(node_ids, xyz, edges_1based):
    n = len(node_ids)
    print(f"Nodes: {n}, Edges: {len(edges_1based)}")
    # ... (为了简洁，省略部分打印，保留核心逻辑) ...


def plot_truss(xyz, edges_1based, node_ids, show_labels=False, title="Truss"):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import matplotlib.patheffects as pe

    fig = plt.figure(figsize=(11, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    n = xyz.shape[0]
    m = len(edges_1based)

    # 深色高对比调色板
    dark_palette = np.array([
        [0.12, 0.47, 0.71],  # 深蓝
        [0.84, 0.15, 0.16],  # 深红
        [0.17, 0.63, 0.17],  # 深绿
        [1.00, 0.50, 0.05],  # 深橙
        [0.58, 0.40, 0.74],  # 紫
        [0.55, 0.34, 0.29],  # 棕
        [0.89, 0.47, 0.76],  # 洋红
        [0.25, 0.25, 0.25],  # 深灰
        [0.09, 0.75, 0.81],  # 青
        [0.74, 0.74, 0.13],  # 橄榄黄
        [0.20, 0.20, 0.70],  # 靛蓝
        [0.70, 0.20, 0.20],  # 暗红
        [0.20, 0.60, 0.30],  # 暗绿
        [0.90, 0.35, 0.10],  # 暗橙
        [0.45, 0.25, 0.65],  # 暗紫
        [0.10, 0.10, 0.10],  # 更深灰
    ], dtype=float)

    node_colors = dark_palette[np.arange(n) % len(dark_palette)]

    # 1) 画边：黑色底线 + 彩色线（不改网格）
    if m > 0:
        e0 = edges_1based[:, 0] - 1
        e1 = edges_1based[:, 1] - 1
        segments = np.stack([xyz[e0], xyz[e1]], axis=1)  # (m, 2, 3)

        edge_colors = dark_palette[np.arange(m) % len(dark_palette)]

        # 黑色描边底线
        ax.add_collection3d(Line3DCollection(
            segments, colors="black", linewidths=4.0, alpha=1.0
        ))
        # 彩色主线
        ax.add_collection3d(Line3DCollection(
            segments, colors=edge_colors, linewidths=2.6, alpha=1.0
        ))

    # 2) 画点：黑色底点 + 彩色点（不改网格）
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c="black", s=120, depthshade=True, alpha=1.0)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c=node_colors, s=70, depthshade=True, alpha=1.0,
               edgecolors="white", linewidths=0.6)

    # 3) 标注编号：黑字 + 白描边（不改网格）
    if show_labels:
        # 小偏移，避免文字压在点中心
        mn, mx = xyz.min(axis=0), xyz.max(axis=0)
        offset = 0.015 * (mx - mn).max()

        for real_id, (x, y, z) in zip(node_ids, xyz):
            t = ax.text(x + offset, y + offset, z + offset, str(real_id),
                        color="black", fontsize=11, fontweight="bold", zorder=200)
            t.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 等比例视图（保留你原来的逻辑）
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    center = (mn + mx) / 2
    max_range = (mx - mn).max() / 2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.show()


if __name__ == "__main__":
    node_ids, xyz, edges_1based = load_truss_from_txt(TXT_PATH)
    validate_and_report(node_ids, xyz, edges_1based)

    do_labels = SHOW_NODE_LABELS and (len(node_ids) <= LABEL_LIMIT)
    
    # 传入 node_ids 以便正确标注
    plot_truss(
        xyz,
        edges_1based,
        node_ids,  # 传入ID数组
        show_labels=do_labels,
        title=f"Geometry view: {os.path.basename(TXT_PATH)}"
    )