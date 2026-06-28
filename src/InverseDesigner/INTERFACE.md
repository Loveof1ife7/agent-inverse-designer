# 数据约定

## 0. 在线 InverseDesigner 调用约定

`src.InverseDesigner.InverseDesigner` 的主契约仍然是：

```text
target_property -> explicit structure
```

当前实现支持两层来源：

```text
1. neural backend:
   调用已预训练的结构族逆向设计网络

2. retrieval fallback:
   当神经 backend 未配置、不可用或没有生成候选时，
   回退到原来的最近邻显式结构检索
```

结构族 backend 通过环境变量或构造函数注入：

```text
TPMS:
  INVERSE_TPMS_CKPT=/path/to/tpms_or_diffumeta.ckpt
  INVERSE_TPMS_PROJECT_DIR=third-party/DiffusionMetamaterials
  INVERSE_TPMS_TARGET_KEYS=stiffness_proxy,density_proxy

Truss:
  INVERSE_TRUSS_BACKEND=graphmetamat
  INVERSE_GRAPHMETAMAT_PROJECT_DIR=third-party/GraphMetaMat
  INVERSE_GRAPHMETAMAT_NUM_RUNS=16
  INVERSE_GRAPHMETAMAT_TOP_K=1
  INVERSE_GRAPHMETAMAT_DEVICE=cuda

Truss custom wrapper:
  INVERSE_TRUSS_CALLABLE=my_pkg.truss_wrapper:sample
  INVERSE_TRUSS_COMMAND='python wrapper.py --ckpt {checkpoint_path} --target {target_json} --out {output_json}'
  INVERSE_TRUSS_CKPT=/path/to/truss.ckpt

B-spline:
  INVERSE_BSPLINE_CALLABLE=my_pkg.bspline_wrapper:sample
  INVERSE_BSPLINE_COMMAND='python wrapper.py --ckpt {checkpoint_path} --target {target_json} --out {output_json}'
  INVERSE_BSPLINE_CKPT=/path/to/bspline.ckpt

Voxel:
  INVERSE_VOXEL_CKPT=/path/to/voxel.ckpt
  INVERSE_VOXEL_PROJECT_DIR=third-party/microstructure_generation_3d
  INVERSE_VOXEL_TARGET_KEYS=C11,C12,C44,volume_fraction
```

`CallableBackend` 的 Python wrapper 需接受：

```python
def sample(target_property, checkpoint_path, output_dir, num_samples, backend_config):
    ...
    return {
        "structure_id": "...",
        "coordinates": [[...], ...],  # truss-like structures
        "edges": [[0, 1], ...],
    }
```

也可以返回非 truss 表示，例如：

```python
{
    "structure_id": "...",
    "representation": "density_voxel",
    "voxel_path": ".../sample.npy"
}
```

`CommandBackend` 的命令模板可使用：

```text
{checkpoint_path}
{output_dir}
{output_json}
{target_json}
{target_csv}
{num_samples}
{sample_index}
```

在线 batch 中，如果 `TargetScheduleItem.expected_effect["structure_family"]`
没有指定结构族，`sample_schedule` 会在已配置 neural backend 的结构族之间轮转，
例如：

```text
tpms -> truss -> b_spline -> voxel -> ...
```

这样一个 target schedule 可以自然覆盖多种结构表示。

本文件定义快速对齐阶段建议交付的数据字段。推荐格式是 **JSONL**：每一行是一个样本，保存未 padding 的变长字段；训练时再由 dataloader 根据 batch 内或全局 \(n_{\max}\)、\(C_{\max}\) 做 padding 和 mask。

## 1. 顶点编号约定

顶点编号直接由 `coordinates` 列表顺序决定：

```text
coordinates[0] -> vertex 1
coordinates[1] -> vertex 2
...
coordinates[n-1] -> vertex n
```

因此所有边字段，包括 `edges` 和 `extra_edges`，都使用这个编号体系。数据文件中不再保存 `node_order`、`node_order_method` 或 `serialization_order`。

这个约定的好处是字段更少，沟通更直接。需要注意的是：数据准备阶段必须保证 `coordinates` 的列表顺序稳定，训练和推理都不能再额外重排节点。

## 2. 保留字段总表

| 字段名 | 类型 / 形状 | 必填 | 作用 |
|---|---:|---:|---|
| `sample_id` | string | 是 | 合并原来的 `sample_id` 和 `source_id`，同时承担唯一 ID 和来源追溯作用。 |
| `split` | string | 建议 | 标记 `train` / `val` / `test`。如果用不同文件夹区分 split，可以不写。 |
| `version` | string | 可选 | 数据预处理版本。快速对齐时可选，但正式实验建议保留。 |
| `y` | float array, shape `[31]` | 是 | 目标性质向量，采用 `stress_grid_v1`：在固定 strain grid `[0.00, 0.01, ..., 0.30]` 上采样得到的 stress 曲线。 |
| `n` | int | 是 | 节点数量。也可由 `len(coordinates)` 推出，但显式保存方便校验。 |
| `coordinates` | float array, shape `[n,3]` | 是 | 节点坐标；列表顺序就是顶点编号顺序。当前预处理会按 BFS discovery order 重编号。 |
| `edges` | int array, shape `[m,2]` | 是 | 完整真实拓扑边集合 \(E\)。每条边使用 `coordinates` 对应的顶点编号。 |
| `parent_sequence` | int array, shape `[n-1]` | 是 | Stage 1 的自回归监督信号，表示 spanning tree。默认 root 为 vertex 1；`parent_sequence[j-2] = p_j`，并保证 `1 <= p_j < j`。 |
| `k` | int | 是 | Stage 2 的边数量监督信号，表示 extra-edge 数量。 |
| `extra_edges` | int array, shape `[k,2]` | 是 | Stage 2 的真实补边集合 \(E_{\mathrm{extra}}\)。 |
| `topology_prefix_tokens` | int / string array | 建议 | Stage 3 的 topology prefix。若 tokenizer 由训练侧统一实现，也可由 `n`、`parent_sequence`、`extra_edges` 推出。 |
| `coordinate_tokens` | int array, shape `[3n]` | 建议 | Stage 3 的坐标 token 监督信号。若坐标量化由训练侧统一实现，也可由 `coordinates` 推出。 |

## 3. Property 曲线采样约定

性质文件 `properties/<id>.csv` 是 Abaqus 压缩仿真的原始应力-应变曲线：

```text
Strain,Stress
```

训练数据中的 `y` 不直接保存变长原始曲线，而是统一转换为固定长度采样向量：

```text
representation = stress_grid_v1
strain_grid = [0.00, 0.01, 0.02, ..., 0.30]
y[i] = stress(strain_grid[i])
```

其中 `Stress` 单位为 MPa，`Strain` 无量纲。

CSV 清洗和采样规则：

1. 删除非法行、NaN、inf；
2. 负 stress clamp 到 `0.0`；
3. 按 strain 升序排序；
4. 重复 strain 的 stress 取平均；
5. 使用线性插值采样到固定 `strain_grid`；
6. 如果曲线没有覆盖到 0.30，右侧使用 `hold_last`。

## 4. 派生字段

以下字段不再要求数据同学交付，由训练侧统一派生：

| 派生字段 | 来源 | 用途 |
|---|---|---|
| `tree_edges` | `parent_sequence` | Stage 2 的 tree GNN 输入。 |
| `candidate_edges` | `n` + `parent_sequence` | Extra-Edge Scorer 的候选边集合。 |
| `candidate_count` | `n` | 限制合法的 \(k\) 范围。 |
| `extra_edge_labels` | `candidate_edges` + `extra_edges` | Weighted CE 的 0/1 边标签。 |
| `positive_edge_count` | `k` | Weighted CE 的正样本权重。 |
| `negative_edge_count` | `candidate_count - k` | Weighted CE 的负样本权重。 |
| `node_mask` | `n` | batch padding。 |
| `parent_mask` | `n` | parent sequence padding。 |
| `candidate_edge_mask` | `candidate_count` | candidate edge padding。 |
| `coord_token_mask` | `n` 或 `coordinate_tokens` | coordinate token padding。 |
| `k_valid_mask` | `candidate_count` | mask 掉非法 \(k\) 类别。 |

## 5. 单样本 JSONL 示例

```json
{
  "sample_id": "sim_batch_03/truss_000001",
  "split": "train",
  "version": "inverse_truss_property_grid_v1",
  "y": [0.0, 0.00042, 0.00081],
  "property": {
    "representation": "stress_grid_v1",
    "strain_grid": [0.0, 0.01, 0.02],
    "stress_unit": "MPa",
    "strain_unit": "dimensionless",
    "interpolation": "linear",
    "negative_stress_policy": "clamp_to_zero",
    "right_extrapolation": "hold_last"
  },
  "n": 5,
  "coordinates": [
    [0.10, 0.20, 0.05],
    [0.35, 0.18, 0.12],
    [0.50, 0.55, 0.20],
    [0.70, 0.40, 0.65],
    [0.90, 0.80, 0.75]
  ],
  "edges": [[1, 2], [1, 3], [2, 5], [3, 4], [2, 4], [3, 5]],
  "parent_sequence": [1, 1, 3, 2],
  "k": 2,
  "extra_edges": [[2, 4], [3, 5]],
  "topology_prefix_tokens": ["<N>", 5, "<TREE>", 1, 1, 3, 2, "<EXTRA>", 2, 4, 3, 5, "<COORD>"],
  "coordinate_tokens": [102, 205, 51, 358, 184, 123, 512, 563, 205, 716, 409, 665, 921, 818, 767]
}
```

当前预处理版本使用 `bfs_parent_pointer_reindexed_v2`。这意味着 `coordinates`、
`edges`、`extra_edges` 和 `parent_sequence` 都处在 BFS 重编号后的 1-based
顶点空间；原始节点顺序保存在 `preprocessing.node_order_original_ids`。
