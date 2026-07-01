import pandas as pd
import numpy as np
import os
import json
import argparse
from pathlib import Path

# ==========================================
# 1. 配置参数
# ==========================================
CSV_PATH = ""

# 仅用于 CLI 默认值；自动流水线会显式传入
OUTPUT_DIR = "Batch_Output_Files"
GROUP_NAME = ""  # 推荐显式传入 --group
GROUP_DB_PATH = "symmetry_group_transforms.json"

# 节点合并容差
TOLERANCE = 1e-5 

# ==========================================
# 2. 核心处理类
# ==========================================
class TrussGenerator:
    def __init__(self, csv_path, group_name="", group_db_path="", symmetry_matrices=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到文件: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.group_name = group_name.strip() if group_name else ""
        self.group_db_path = group_db_path.strip() if group_db_path else ""
        
        # 自动识别列名
        cols = self.df.columns
        coord_cols = [c for c in cols if c.endswith('_x')]
        self.node_names = [c[:-2] for c in coord_cols]

        self.matrices_norm = self._resolve_symmetry_matrices(symmetry_matrices)
        self._validate_matrices(self.matrices_norm)
        src = "inline" if symmetry_matrices else f"group={self.group_name}"
        print(f"群矩阵加载完成: source={src}, ops={len(self.matrices_norm)}")
        print(f"数据加载完成，共 {len(self.df)} 行")

    @staticmethod
    def _validate_matrices(mats):
        if not mats:
            raise ValueError("群矩阵为空")
        for i, m in enumerate(mats):
            if m.shape != (4, 4):
                raise ValueError(f"第 {i} 个群矩阵维度错误: {m.shape}, 期望 (4,4)")

    def _resolve_db_path(self):
        if self.group_db_path:
            p = Path(self.group_db_path)
            if p.is_absolute():
                return p
            return (Path(__file__).resolve().parent / p).resolve()
        return (Path(__file__).resolve().parent / GROUP_DB_PATH).resolve()

    def _load_group_matrices(self):
        if not self.group_name:
            return None
        db_path = self._resolve_db_path()
        if not db_path.exists():
            raise FileNotFoundError(f"群数据库不存在: {db_path}")

        with db_path.open("r", encoding="utf-8") as f:
            db = json.load(f)
        groups = db.get("groups", {})
        if self.group_name not in groups:
            raise KeyError(f"群名 '{self.group_name}' 不存在于 {db_path}")
        m_raw = groups[self.group_name].get("M_sym")
        if not m_raw:
            raise ValueError(f"群 '{self.group_name}' 缺少 M_sym")
        return [np.array(m, dtype=float) for m in m_raw]

    def _resolve_symmetry_matrices(self, symmetry_matrices):
        if symmetry_matrices:
            return [np.array(m, dtype=float) for m in symmetry_matrices]
        mats = self._load_group_matrices()
        if mats:
            return mats
        raise ValueError("未提供群矩阵。请通过 --group/--group-db 指定群。")

    def clean_and_merge(self, nodes, edges):
        """合并空间坐标重合的点，并更新连接关系"""
        if len(nodes) == 0: 
            return nodes, edges

        # 1. 量化坐标
        decimals = int(-np.log10(TOLERANCE))
        rounded_nodes = np.round(nodes, decimals=decimals)
        
        # 2. 寻找唯一节点
        _, unique_indices, inverse = np.unique(
            rounded_nodes, axis=0, return_index=True, return_inverse=True
        )
        cleaned_nodes = nodes[unique_indices]

        # 3. 重映射边
        new_edges = inverse[edges]

        # 4. 移除自环和重复边
        new_edges = np.sort(new_edges, axis=1)
        new_edges = new_edges[new_edges[:, 0] != new_edges[:, 1]]
        new_edges = np.unique(new_edges, axis=0)

        return cleaned_nodes, new_edges

    # ================================
    # 【新增】删除孤立点逻辑（只加这块）
    # ================================
    def remove_isolated_nodes(self, nodes, edges):
        """
        删除所有不属于任何边端点的孤立节点，并对节点重新编号，同时更新 edges。
        """
        if nodes is None or len(nodes) == 0:
            return nodes, edges

        if edges is None or len(edges) == 0:
            # 没有任何边 => 按需求：没有“线段顶点”的点，全部删掉
            return np.empty((0, 3), dtype=float), np.empty((0, 2), dtype=int)

        used = np.unique(edges.reshape(-1)).astype(int)  # 所有出现在边里的节点索引
        new_nodes = nodes[used]

        mapping = -np.ones(len(nodes), dtype=int)
        mapping[used] = np.arange(len(used), dtype=int)

        new_edges = mapping[edges]
        return new_nodes, new_edges
    # ================================
    # 【新增结束】
    # ================================

    def process_row(self, k):
        if k >= len(self.df):
            print(f"Error: Index {k} out of bounds")
            return None, None, None
        
        row = self.df.iloc[k]
        name = row['name']
        
        # 获取物理边长
        L = float(row['A4_x'])
        
        # 提取基础数据
        base_nodes = []
        for n in self.node_names:
            base_nodes.append([row[f"{n}_x"], row[f"{n}_y"], row[f"{n}_z"]])
        base_nodes = np.array(base_nodes)

        num_base = len(base_nodes)
        adj_cols = [f"element_{i+1}" for i in range(num_base**2)]
        adj_flat = row[adj_cols].values.astype(int).reshape(num_base, num_base)
        
        rows, cols = np.triu_indices(num_base, k=1)
        valid = adj_flat[rows, cols] == 1
        base_edges = np.column_stack((rows[valid], cols[valid]))

        ones = np.ones((len(base_nodes), 1))
        nodes_homo = np.hstack([base_nodes, ones])
        
        all_nodes_list = []
        all_edges_list = []
        current_offset = 0

        for M in self.matrices_norm:
            M_phys = M.copy()
            M_phys[:3, 3] = M_phys[:3, 3] * L
            trans_homo = (M_phys @ nodes_homo.T).T
            trans_nodes = trans_homo[:, :3]
            all_nodes_list.append(trans_nodes)
            all_edges_list.append(base_edges + current_offset)
            current_offset += len(trans_nodes)

        raw_nodes = np.vstack(all_nodes_list)
        raw_edges = np.vstack(all_edges_list)

        final_nodes, final_edges = self.clean_and_merge(raw_nodes, raw_edges)

        # 【新增】只保留参与连接的点（删除孤立点）
        final_nodes, final_edges = self.remove_isolated_nodes(final_nodes, final_edges)

        return final_nodes, final_edges, name

    def save_to_txt(self, filename, nodes, edges, identifier):
        """
        【修改】写入模式改为 'w' (覆盖)，确保每个文件是独立的
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# ==========================================\n")
            f.write(f"# Data ID: {identifier}\n")
            f.write(f"# ==========================================\n")
            
            f.write("node_data = [\n")
            for i, (x, y, z) in enumerate(nodes):
                f.write(f"    [{i+1}, {x:.6f}, {y:.6f}, {z:.6f}],\n")
            f.write("]\n\n")
            
            f.write("element_conn = [\n")
            for u, v in edges:
                f.write(f"    [{u+1}, {v+1}],\n")
            f.write("]\n\n")

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV 转 Abaqus txt（按群矩阵拼接）")
    parser.add_argument("--csv", default=CSV_PATH, help="输入 architecture CSV")
    parser.add_argument("--out", default=OUTPUT_DIR, help="输出 txt 文件夹")
    parser.add_argument("--group", default=GROUP_NAME, help="群名（如 Aba2 / P222）")
    parser.add_argument("--group-db", default=GROUP_DB_PATH, help="群矩阵 JSON 路径")
    args = parser.parse_args()

    if not args.csv:
        raise ValueError("必须提供 --csv")
    # 1. 初始化生成器
    generator = TrussGenerator(args.csv, group_name=args.group, group_db_path=args.group_db)
    
    # 2. 检查并创建输出文件夹
    if not os.path.exists(args.out):
        os.makedirs(args.out)
        print(f"创建输出目录: {args.out}")
    else:
        print(f"使用现有目录: {args.out}")

    # 3. 获取总行数
    total_rows = len(generator.df)
    print(f"准备处理全部 {total_rows} 个数据...")

    # 4. 循环处理所有行
    for k in range(total_rows):
        # 处理数据
        nodes, edges, name = generator.process_row(k)
        
        if nodes is not None:
            # 【关键】构建文件名：直接使用行索引 k.txt
            file_name = f"{k}.txt"
            full_path = os.path.join(args.out, file_name)
            
            # 保存
            generator.save_to_txt(full_path, nodes, edges, name)
            
            # 打印进度 (每10个打印一次，防止刷屏)
            if k % 10 == 0:
                print(f"[{k}/{total_rows}] 已保存: {file_name} (ID: {name})")

    print("\n所有文件处理完成！")
