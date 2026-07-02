# gid_c3d4_pipeline —— GraphMetaMat 结构 C3D4 实体有限元 一体化脚本

一个自包含单文件脚本（`gid_c3d4_pipeline.py`，无外部 `.py` 依赖），把一条 GraphMetaMat 数据集结构从**导入 → 建网格 → 跑仿真 → 出对比图**全程走完，复现数据集的应力应变参考曲线。

---

## 1. 它做什么

```
导入结构 (nodes.csv / struts.csv / meta.json)
  → [可选] N×N×N 阵列（--array，默认 1 = 输入单结构直接算）+ 合并周期交界节点
  → gmsh OCC：每根杆一个圆柱 + 每个节点一个球 → 布尔融合成实心杆架
  → 四面体网格 C3D4（网格尺寸按杆半径缩放，填充到 ~86% 体积）
  → 写 Abaqus/Explicit inp（线弹性 Es=7、底固定 + 顶刚性板压 30%、无摩擦自接触、几何非线性）
  → abaqus 直接调核心求解
  → 从 odb 提取 σ–ε 曲线（σ=F/A=(N·L)²，ε=u/(N·L)）
  → 和 meta.json 里数据集参考曲线对比出图

**默认 `--array 1`：输入什么结构就直接仿真什么**（不堆叠）。
若要复现数据集 6 条参考曲线（参考是 2×2×2 压出来的），加 `--array 2`；
归一化面积 A=(N·L)² 和高度 Lz=N·L 自动跟随 `--array`。
```

---

## 2. 环境依赖

| 依赖 | 说明 |
|---|---|
| Python 3 | 建议 3.9+ |
| gmsh | `pip install gmsh`（纯 wheel，建网格用） |
| numpy | gmsh 依赖 |
| matplotlib | 出对比图用 |
| Abaqus | 求解 + 提取 odb，命令 `abq2025` / `abq2022` / `abaqus` 任一 |

Python 侧只需 gmsh/numpy/matplotlib；`abaqus` 是独立求解器，脚本用子进程调用。

---

## 3. 输入：结构目录

一个目录，含数据集导出的 4 个文件（`reference_curve.csv` 可缺，缺则不画参考线）：

| 文件 | 内容 |
|---|---|
| `nodes.csv` | `node_id, x_mm, y_mm, z_mm`（单胞 [-5,5]³ 毫米坐标，0-based） |
| `struts.csv` | `node_i, node_j, length_mm, radius_mm`（0-based 节点索引） |
| `meta.json` | 半径 `strut_radius_mm`、单胞 `unit_cell_size_L_mm`、相对密度、参考峰值等 |
| `reference_curve.csv` | `strain, stress_normalized_by_Es`（数据集真实曲线，Es=7 原生尺度） |

半径、单胞尺寸、参考峰值都从 `meta.json` 自动读，无需手填。

---

## 4. 用法

### 本地（Windows，自动找 abq2025）
```bash
# 只建网格 + 写 inp，先看体积填充率对不对（不求解）
python gid_c3d4_pipeline.py --struct-dir 导出结构_6条/gid2979 --mesh-only

# 默认单结构（1×1×1）直接算：建网格 → 求解 → 提取 → 出对比图
python gid_c3d4_pipeline.py --struct-dir 导出结构_6条/gid2979 --cpus 8

# 复现数据集参考曲线（参考是 2×2×2 压的）：加 --array 2
python gid_c3d4_pipeline.py --struct-dir 导出结构_6条/gid2979 --array 2 --cpus 8
```

### 服务器（Linux，slurm 坏时直接调核心并行多条）
```bash
# 单条
/path/to/conda/envs/abaqus/bin/python gid_c3d4_pipeline.py \
    --struct-dir data/gid2979 --cpus 8 --abaqus /path/to/abaqus/Commands/abq2022

# 5 条并行，每条 8 核（不走 slurm，直接 setsid 后台）
for g in 2979 2980 2981 2982 2983; do
  setsid python gid_c3d4_pipeline.py --struct-dir data/gid$g --cpus 8 \
      --abaqus /path/.../abq2022 > log_$g.txt 2>&1 < /dev/null &
done
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--struct-dir` | 必填 | 结构目录 |
| `--array` | **1** | N×N×N 阵列堆叠数。默认 1 = 输入单结构直接算；设 2 对齐数据集 2×2×2 参考。归一化 A=(N·L)²、Lz=N·L 自动跟随 |
| `--out` | `struct-dir/c3d4_run` | 输出目录 |
| `--cpus` | 8 | abaqus 求解核数 |
| `--abaqus` | 自动找 | abaqus 命令路径（或设环境变量 `ABAQUS_CMD`） |
| `--k-min` / `--k-max` | 0.310 / 0.775 | 网格尺寸 `CL = k·radius`，默认 = 86% 体积那档 |
| `--young` | 7.0 | 材料模量 Es |
| `--mesh-only` | 关 | 只建网格写 inp |
| `--no-plot` | 关 | 不出图 |

---

## 5. 输出

在 `out/{gid}_c3d4/` 下：
- `Job_{gid}_c3d4.inp` — Abaqus 输入
- `Job_{gid}_c3d4.odb` — 结果
- `data.csv` — σ–ε 曲线（列：`Strain, Disp_mm, Force_N, Stress_MPa, ALLKE, ALLIE, ALLWK, KE_over_IE, Time_s`）
- `out/{gid}_compare.png` — 参考 vs C3D4 对比图

终端打印峰值应力、`FE/ref` 比值、体积填充率。

---

## 6. 严格对齐的 FE 设定（写死在脚本里）

对齐 GraphMetaMat 数据集 `meta.json` 的 `fe_setup`：

| 项 | 值 |
|---|---|
| 单元 | C3D4 四节点四面体 |
| 阵列 | N×N×N（`--array`，默认 1 单结构；数据集参考对应 N=2） |
| 材料 | 线弹性，Es=7，ν=0.3 |
| 边界 | 底板固定，顶刚性板沿 z 压缩到宏观应变 30% |
| 端部 | 每端一个半径厚的帽子 TIE 绑到刚性板 |
| 接触 | 全局无摩擦自接触（general contact） |
| 非线性 | NLGEOM 开，本构线弹性不断裂 |
| 应力 | σ = F/A，A=(N·L)²（N=2 即 400 mm²） |
| 应变 | ε = u/(N·L)（N=2 即 u/20 mm） |
| 求解 | Abaqus/Explicit 准静态 + 固定质量缩放 DT=5e-6 |
| 单位制 | mm-tonne-s-N（σ=MPa，ρ=1.11e-9 t/mm³） |

准静态有效性判据：加载段动能/内能 `KE/IE < 0.01`（`data.csv` 的 `KE_over_IE` 列，忽略 0 应变附近的瞬态尖峰）。

---

## 7. 两个关键结论（已验证）

### 网格必须填够体积才对得上参考
网格尺寸取"平均 2r"（数据集原话）会让细杆欠填、只填 ~51–60% 体积 → 曲线偏软 10%+。
按半径缩放 `CL = 0.31r .. 0.78r` 填到 **~86% 体积**后，C3D4 曲线**全程贴合数据集参考**（gid2978 实测：strain 0.05/0.10/0.20 处 FE/ref = 0.97/0.99/1.02）。
`--k-min`/`--k-max` 控制这个细度；调小 → 更细更满但更慢。

### Es=4 跑后 ×7/4 与 Es=7 直接跑完全等价
仿真是**位移控制**（ε=u/20 定死），材料纯线弹性，变形构型和接触都与 Es 无关，所以应力场**精确正比于 Es**。
用 `--young 4` 跑，再把应力 ×7/4，得到的曲线与 `--young 7` 直接跑逐点重合（gid2978 实测峰值 4.68e-4 vs 4.63e-4，差 1%）。
所以论文 Es=4 与数据集 Es=7 之间只差一个常数因子，不影响归一化曲线形状。

---

## 8. 梁单元（B31）为什么对不上

同结构用 B31 梁替代实体，全程偏软约 20–25%，即使把端部按半径刚化也补不平——梁把节点当成数学点，缺了多杆交汇处那团**实心 hub** 的刚度（实体 FE 里 hub 是满料，实打实缩短了弯曲跨度）。**要复现数据集参考曲线只能用实体 C3D4 + 填够体积。**

---

## 9. 数据集 6 条结构（`导出结构_6条/`）

| 结构 | 节点/杆 | 相对密度 | 杆半径 mm | 参考峰值(Es7) |
|---|---|---|---|---|
| gid2978 | 48 / 96 | 0.0597 | 0.194 | 4.79e-4 |
| gid2979 | 84 / 120 | 0.0546 | 0.202 | 2.53e-4 |
| gid2980 | 78 / 120 | 0.190 | 0.313 | 5.06e-3 |
| gid2981 | 68 / 72 | 0.0505 | 0.252 | 3.18e-4 |
| gid2982 | 66 / 72 | 0.110 | 0.408 | 5.42e-3 |
| gid2983 | 84 / 120 | 0.0614 | 0.186 | 2.73e-4 |
