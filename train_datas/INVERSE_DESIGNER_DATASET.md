# InverseDesigner Dataset Interface

本文档定义当前交付给下游 InverseDesigner 训练的数据接口。核心任务是：

```text
target property y -> explicit truss structure
```

也就是模型输入是性质向量，监督目标是显式 truss：

```text
structure = topology + coordinates
```

## 1. Dataset Location

主训练数据：

```text
train_datas/P222_paired_dataset_0_99999_20260620/preprocessed/
  inverse_truss_property_grid_v1_0_9999.jsonl
  inverse_truss_property_grid_v1_0_9999_manifest.json
```

调试小样本：

```text
train_datas/P222_paired_dataset_0_99999_20260620/preprocessed/
  inverse_truss_property_grid_v1_0_99.jsonl
  inverse_truss_property_grid_v1_0_99_manifest.json
```

`0_99` 是 `0_9999` 的前 100 条，schema 和内容完全对齐，可用于 smoke test。

当前主数据统计：

```text
samples: 10000
split: train 9000 / val 500 / test 500
skipped: 0
format: JSONL, one sample per line, no padding
```

更完整的 dataloader 示例见：

```text
train_datas/P222_paired_dataset_0_99999_20260620/preprocessed/DATALOADER.md
```

## 2. One-Line Record Contract

每一行 JSONL 是一个未 padding 的样本。下游 Dataset 读取一行后，应得到：

```text
y:                  FloatTensor[31]
n:                  int
parent_sequence:    LongTensor[n - 1]
k:                  int
extra_edges:        LongTensor[k, 2]
coordinates:        FloatTensor[n, 3]
coordinate_tokens:  LongTensor[3 * n]
edges:              LongTensor[m, 2]
```

推荐训练目标拆分：

```text
condition:
  y

generation targets:
  1. n
  2. parent_sequence
  3. k / extra_edges
  4. coordinates or coordinate_tokens

optional reconstruction target:
  edges
```

## 3. Required Fields

| Field | Type / Shape | Meaning |
|---|---:|---|
| `sample_id` | string | 样本 ID，对应原始 `structures/<id>.txt` 和 `properties/<id>.csv`。 |
| `split` | string | `train` / `val` / `test`。 |
| `version` | string | 当前为 `inverse_truss_property_grid_v1`。 |
| `y` | float array `[31]` | 条件输入，固定 strain grid 上的 stress vector。 |
| `n` | int | 节点数。 |
| `coordinates` | float array `[n, 3]` | 归一化坐标，范围 `[0, 1]`。 |
| `edges` | int array `[m, 2]` | 完整 truss connectivity。 |
| `parent_sequence` | int array `[n - 1]` | spanning tree 的 parent-pointer 监督。 |
| `k` | int | extra-edge 数量。 |
| `extra_edges` | int array `[k, 2]` | 非 tree edges。 |
| `coordinate_tokens` | int array `[3n]` | 10-bit 量化坐标 token。 |
| `property` | object | `y` 的采样规则和 summary metadata。 |
| `preprocessing` | object | 坐标归一化、重编号、量化、原始节点顺序等信息。 |
| `provenance` | object | 原始结构和 property 文件路径。 |

`topology_prefix_tokens` 也会导出，但它是 convenience serialization，不建议作为唯一训练接口。严肃训练中优先使用结构化字段：

```text
n
parent_sequence
k
extra_edges
coordinates / coordinate_tokens
```

## 4. Property Interface

`y` 是 InverseDesigner 的条件输入：

```text
representation = stress_grid_v1
shape = [31]
strain_grid = [0.00, 0.01, ..., 0.30]
y[i] = stress(strain_grid[i])
stress_unit = MPa
strain_unit = dimensionless
```

原始曲线来自：

```text
properties/<sample_id>.csv
```

CSV 格式：

```text
Strain,Stress
```

预处理策略：

```text
invalid row / NaN / inf: drop
duplicate strain: mean stress
negative stress: clamp to 0
interpolation: linear
left_extrapolation: hold_first
right_extrapolation: hold_last
```

训练时直接读取：

```python
y = torch.tensor(row["y"], dtype=torch.float32)
```

不要在普通训练 dataloader 中重新解析原始 CSV。

## 5. Topology Interface

所有 topology index 在 JSONL 中都是 **1-based**：

```text
coordinates[0] -> vertex 1
coordinates[1] -> vertex 2
...
coordinates[n - 1] -> vertex n
```

以下字段均使用 1-based vertex index：

```text
edges
parent_sequence
extra_edges
topology_prefix_tokens
```

推荐训练侧约定：

```text
JSONL on disk: keep 1-based
Dataset.__getitem__: convert to 0-based tensors
padding index: -1
```

Parent-pointer 定义：

```text
parent_sequence = [p_2, p_3, ..., p_n]
len(parent_sequence) = n - 1
```

自回归保证：

```text
for child j in [2, n]:
    1 <= p_j < j
```

所以在生成第 `j` 个节点时，它的 parent 一定已经出现。

当前导出使用：

```text
topology_tree_method = bfs_parent_pointer_reindexed_v2
```

含义：

```text
1. 从原始 node 1 开始 BFS。
2. 邻居按原始编号升序访问。
3. 按 BFS discovery order 重新编号节点。
4. coordinates / edges / parent_sequence / extra_edges 全部使用重编号后的 vertex index。
5. 原始节点顺序保存在 preprocessing.node_order_original_ids，仅用于 traceability。
```

## 6. Coordinate Interface

连续坐标：

```text
coordinates: FloatTensor[n, 3]
range: [0, 1]
normalization: bbox_to_unit_cube
```

离散坐标 token：

```text
coordinate_tokens: LongTensor[3 * n]
quantization_bits: 10
token range: [0, 1023]
flatten order: x1, y1, z1, x2, y2, z2, ...
```

第一版下游模型可以先用连续 `coordinates` 做回归；如果采用 token generation，再使用 `coordinate_tokens`。

## 7. Minimal Loader Sketch

```python
import json
from pathlib import Path

import torch


def to_edge_tensor(edge_list):
    if not edge_list:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(edge_list, dtype=torch.long) - 1


path = Path(
    "train_datas/P222_paired_dataset_0_99999_20260620/"
    "preprocessed/inverse_truss_property_grid_v1_0_9999.jsonl"
)

with path.open("r", encoding="utf-8") as f:
    row = json.loads(next(f))

item = {
    "sample_id": row["sample_id"],
    "split": row["split"],
    "y": torch.tensor(row["y"], dtype=torch.float32),
    "n": torch.tensor(row["n"], dtype=torch.long),
    "coordinates": torch.tensor(row["coordinates"], dtype=torch.float32),
    "edges": to_edge_tensor(row["edges"]),
    "parent_sequence": torch.tensor(row["parent_sequence"], dtype=torch.long) - 1,
    "k": torch.tensor(row["k"], dtype=torch.long),
    "extra_edges": to_edge_tensor(row["extra_edges"]),
    "coordinate_tokens": torch.tensor(row["coordinate_tokens"], dtype=torch.long),
}
```

Batch padding 建议：

```text
variable-length index tensor padding: -1
float tensor padding: 0.0
coordinate token padding: 0, always masked
```

## 8. Validation Constraints

下游读取数据后至少检查：

```python
def check_sample(row):
    n = row["n"]

    assert len(row["y"]) == 31
    assert len(row["coordinates"]) == n
    assert len(row["parent_sequence"]) == n - 1
    assert len(row["coordinate_tokens"]) == 3 * n
    assert row["k"] == len(row["extra_edges"])

    for xyz in row["coordinates"]:
        assert len(xyz) == 3
        assert all(0.0 <= v <= 1.0 for v in xyz)

    for u, v in row["edges"] + row["extra_edges"]:
        assert 1 <= u <= n
        assert 1 <= v <= n
        assert u != v

    for child in range(2, n + 1):
        parent = row["parent_sequence"][child - 2]
        assert 1 <= parent < child
```

当前 10000 样本已通过以上约束抽检。

## 9. Preprocessing Script Pointer

预处理脚本：

```text
src/TrainingDataset/inverse_truss_preprocess.py
```

输入目录约定：

```text
train_datas/P222_paired_dataset_0_99999_20260620/
  structures/<id>.txt
  properties/<id>.csv
```

重新生成 10000 条主训练集：

```bash
python src/TrainingDataset/inverse_truss_preprocess.py \
  --dataset-root train_datas/P222_paired_dataset_0_99999_20260620 \
  --output train_datas/P222_paired_dataset_0_99999_20260620/preprocessed/inverse_truss_property_grid_v1_0_9999.jsonl \
  --manifest train_datas/P222_paired_dataset_0_99999_20260620/preprocessed/inverse_truss_property_grid_v1_0_9999_manifest.json \
  --limit 10000
```

生成 100 条 smoke-test 数据：

```bash
python src/TrainingDataset/inverse_truss_preprocess.py \
  --dataset-root train_datas/P222_paired_dataset_0_99999_20260620 \
  --output train_datas/P222_paired_dataset_0_99999_20260620/preprocessed/inverse_truss_property_grid_v1_0_99.jsonl \
  --manifest train_datas/P222_paired_dataset_0_99999_20260620/preprocessed/inverse_truss_property_grid_v1_0_99_manifest.json \
  --limit 100
```

相关测试：

```bash
python -m pytest tests/test_inverse_truss_preprocess.py -q
```
