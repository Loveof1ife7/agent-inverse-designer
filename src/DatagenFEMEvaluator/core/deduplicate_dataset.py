# -*- coding: utf-8 -*-
import csv
import os

# 配置路径
INPUT_PATH = r"C:\Users\admin\Desktop\3Dtruss\P222_1\P222_1-architecture.csv"
OUTPUT_PATH = r"C:\Users\admin\Desktop\3Dtruss\P222_1\P222_1-architecture_CLEANED.csv"

def clean_dataset_and_reindex():
    # 集合用于存储指纹，指纹是 (邻接矩阵元组)
    seen_adj = set()
    
    count_read = 0
    count_saved = 0
    
    # 邻接矩阵大小 (19*19)
    ADJ_SIZE = 361

    print(f"正在读取原始文件: {INPUT_PATH}")
    print(f"正在写入清洗文件: {OUTPUT_PATH}")

    # 使用 utf-8-sig 防止 Excel 打开乱码，或者 utf-8
    with open(INPUT_PATH, 'r', encoding='utf-8', newline='') as f_in, \
         open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        # 1. 处理表头
        header = next(reader, None)
        if header:
            writer.writerow(header)
        
        # 2. 逐行处理
        for row in reader:
            count_read += 1
            if not row: continue

            # 提取最后 361 列作为指纹 (Topology Fingerprint)
            adj_tuple = tuple(row[-ADJ_SIZE:])
            
            if adj_tuple in seen_adj:
                # 是重复的，跳过
                continue
            else:
                # 是新的，执行写入
                seen_adj.add(adj_tuple)
                
                # ==========================================
                # [关键修改] 重置 ID
                # 将该行的第0列（ID列）修改为当前的保存计数
                # 这样 ID 就会是连续的 0, 1, 2, 3 ...
                # ==========================================
                row[0] = count_saved
                
                writer.writerow(row)
                count_saved += 1
            
            if count_read % 100000 == 0:
                print(f"已扫描 {count_read} 行，当前有效样本 ID 已排到 {count_saved-1} ...")

    print("="*50)
    print("清洗与 ID 重排完成！")
    print(f"原始行数: {count_read}")
    print(f"清洗后行数: {count_saved}")
    print(f"删除了: {count_read - count_saved} 行重复数据")
    print(f"新 ID 范围: 0 到 {count_saved - 1}")
    print(f"新文件保存在: {OUTPUT_PATH}")
    print("="*50)

if __name__ == "__main__":
    clean_dataset_and_reindex()