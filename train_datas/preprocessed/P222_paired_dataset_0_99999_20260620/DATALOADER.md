# Dataloader Guide for Downstream InverseDesigner Training

This folder contains preprocessed `(property, structure)` pairs for downstream
InverseDesigner training.

Current files:

```text
inverse_truss_property_grid_v1_0_9999.jsonl
inverse_truss_property_grid_v1_0_9999_manifest.json
inverse_truss_property_grid_v1_0_99.jsonl
inverse_truss_property_grid_v1_0_99_manifest.json
```

Use `inverse_truss_property_grid_v1_0_9999.jsonl` as the main training dataset.
The `0_99` file is a small smoke-test/debug subset with the same schema.

Each JSONL line is one unpadded sample. The intended learning problem is:

```text
target property y -> generate truss structure
```

In the current dataset, `y` is already extracted from the stress-strain curve.
The structure supervision is split into topology and geometry:

```text
y
-> n
-> parent_sequence       # spanning tree, autoregressive-safe
-> k and extra_edges     # non-tree connectivity
-> coordinates           # normalized node coordinates in [0, 1]^3
```

## 1. Dataset Contract

For each sample:

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

Recommended model contract:

```text
condition input:
  y

generation targets:
  1. n
  2. parent_sequence
  3. k / extra_edges
  4. coordinates or coordinate_tokens

optional reconstruction target:
  edges
```

`edges` is the full graph connectivity. It equals the union of tree edges implied
by `parent_sequence` and non-tree edges stored in `extra_edges`.

## 2. Property Vector `y`

`y` is the main conditioning vector for InverseDesigner.

```text
representation = stress_grid_v1
shape = [31]
strain_grid = [0.00, 0.01, ..., 0.30]
y[i] = stress(strain_grid[i])
stress_unit = MPa
strain_unit = dimensionless
```

Property preprocessing policy:

```text
interpolation: linear
duplicate_strain_policy: mean_stress
negative_stress_policy: clamp_to_zero
left_extrapolation: hold_first
right_extrapolation: hold_last
```

The current export keeps raw MPa-scale stress values. Downstream training may add
its own normalization, for example dataset-level mean/std normalization of `y`.
If normalization is used, save the statistics with the trained model.

### How to Read Property in Training

For downstream training, read property directly from the preprocessed JSONL
field:

```python
import json
import torch

jsonl_path = (
    "train_datas/P222_paired_dataset_0_99999_20260620/"
    "preprocessed/inverse_truss_property_grid_v1_0_9999.jsonl"
)

with open(jsonl_path, "r", encoding="utf-8") as f:
    row = json.loads(next(f))

y = torch.tensor(row["y"], dtype=torch.float32)

print(row["sample_id"])
print(y.shape)                            # torch.Size([31])
print(row["property"]["representation"])  # stress_grid_v1
print(row["property"]["strain_grid"])     # [0.0, 0.01, ..., 0.3]
```

Training dataloader should usually use:

```python
item["y"] = torch.tensor(row["y"], dtype=torch.float32)
```

Do not re-parse raw property CSV files during normal training. The raw CSV files
are kept for provenance and possible reprocessing:

```text
properties/<sample_id>.csv
```

Raw CSV format:

```csv
Strain,Stress
0,0
0.000157443,8.58064e-06
...
```

The preprocessor converts raw CSV curves into `row["y"]` by:

```text
raw Strain-Stress curve
-> remove/average duplicate strain points
-> clamp negative stress to 0
-> linearly interpolate to fixed strain_grid
-> output 31-dim stress vector y
```

So the final training meaning is:

```text
row["y"][i] = stress(row["property"]["strain_grid"][i])
```

## 3. Topology Indexing Rule

All topology fields in JSONL use **1-based vertex indices**:

```text
coordinates[0] -> vertex 1
coordinates[1] -> vertex 2
...
coordinates[n - 1] -> vertex n
```

Fields using 1-based indices:

```text
edges
parent_sequence
extra_edges
topology_prefix_tokens
```

Recommended training-side convention:

```text
Keep JSONL on disk as 1-based.
Convert to 0-based tensors inside Dataset.__getitem__.
Use -1 as padding for variable-length index tensors.
```

## 4. Parent-Pointer Sequence

`parent_sequence` is the spanning-tree supervision for autoregressive topology
generation.

JSONL format:

```text
parent_sequence = [p_2, p_3, ..., p_n]
len(parent_sequence) = n - 1
```

The current export uses:

```text
topology_tree_method = bfs_parent_pointer_reindexed_v2
```

This means the graph has been reindexed by BFS discovery order before exporting
`coordinates`, `edges`, `parent_sequence`, and `extra_edges`.

Autoregressive guarantee in JSONL 1-based indices:

```text
for child j in [2, n]:
    1 <= p_j < j
```

After converting to 0-based PyTorch indices:

```text
parent_sequence_0based[j - 2] = p_j - 1
child_0based = j - 1
0 <= parent_sequence_0based[j - 2] < child_0based
```

So at generation step `child_0based`, the parent always refers to an already
generated node.

Do **not** assume `parent_sequence` follows the original structure-file node
order. It follows BFS-reindexed node order. The original node ids are saved only
for traceability:

```text
preprocessing.node_order_original_ids
```

This field should normally not be used as model input or supervision.

## 5. Coordinates and Coordinate Tokens

`coordinates` stores normalized geometry:

```text
coordinates: FloatTensor[n, 3]
range: [0, 1]
normalization: bbox_to_unit_cube
```

`coordinate_tokens` stores quantized coordinates:

```text
coordinate_tokens: LongTensor[3 * n]
quantization_bits: 10
token range: [0, 1023]
flatten order: x1, y1, z1, x2, y2, z2, ...
```

Two valid training choices:

```text
continuous regression:
  predict coordinates directly with MSE/L1 loss

discrete token generation:
  predict coordinate_tokens with cross-entropy loss
```

For the first version, using continuous `coordinates` is usually simpler.

## 6. Extra Edges

`extra_edges` stores non-tree edges after removing the spanning-tree edges
implied by `parent_sequence`.

```text
k = len(extra_edges)
extra_edges: LongTensor[k, 2]
```

The downstream model can treat extra-edge prediction in several ways:

```text
simple sequence target:
  generate k, then generate k edge pairs

candidate classification:
  build all candidate undirected pairs i < j excluding tree edges,
  then predict whether each candidate is selected

sparse retrieval/ranking:
  score candidate pairs and choose top-k
```

The dataset does not require a specific extra-edge modeling strategy.

## 7. `topology_prefix_tokens`

`topology_prefix_tokens` is included as a convenience serialization:

```text
["<N>", n, "<TREE>", ..., "<EXTRA>", ..., "<COORD>"]
```

It intentionally mixes strings and integers. For serious training, prefer
building your own tokenizer from structured fields:

```text
n
parent_sequence
k
extra_edges
coordinates / coordinate_tokens
```

## 8. Batch Padding

Samples have variable numbers of nodes and edges. Dataloader should pad within
each batch.

Recommended padded tensors:

```text
y:                       [B, 31]
n:                       [B]
coordinates:             [B, n_max, 3]
node_mask:               [B, n_max]
parent_sequence:         [B, n_max - 1]
parent_mask:             [B, n_max - 1]
edges:                   [B, m_max, 2]
edge_mask:               [B, m_max]
extra_edges:             [B, k_max, 2]
extra_edge_mask:         [B, k_max]
coordinate_tokens:       [B, 3 * n_max]
coord_token_mask:        [B, 3 * n_max]
```

Recommended padding values:

```text
real node/edge indices: 0..n-1 after Dataset conversion
index padding: -1
float padding: 0.0
coordinate token padding: 0, always masked
```

If an embedding layer cannot accept `-1`, either remap padding to a dedicated
padding index before lookup or clamp only after applying the mask.

## 9. Minimal PyTorch Dataset

```python
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def edge_tensor(edge_list, offset):
    if not edge_list:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(edge_list, dtype=torch.long) - offset


class InverseTrussDataset(Dataset):
    def __init__(self, jsonl_path, split=None, to_zero_based=True):
        self.rows = []
        self.to_zero_based = to_zero_based

        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if split is None or row.get("split") == split:
                    self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        offset = 1 if self.to_zero_based else 0

        return {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "y": torch.tensor(row["y"], dtype=torch.float32),
            "n": torch.tensor(row["n"], dtype=torch.long),
            "coordinates": torch.tensor(row["coordinates"], dtype=torch.float32),
            "edges": edge_tensor(row["edges"], offset),
            "parent_sequence": torch.tensor(row["parent_sequence"], dtype=torch.long) - offset,
            "k": torch.tensor(row["k"], dtype=torch.long),
            "extra_edges": edge_tensor(row["extra_edges"], offset),
            "coordinate_tokens": torch.tensor(row["coordinate_tokens"], dtype=torch.long),
        }
```

## 10. Minimal Collate Function

```python
import torch


def pad_1d(items, key, pad_value=0):
    max_len = max(item[key].shape[0] for item in items)
    out = items[0][key].new_full((len(items), max_len), pad_value)
    mask = torch.zeros((len(items), max_len), dtype=torch.bool)

    for i, item in enumerate(items):
        length = item[key].shape[0]
        out[i, :length] = item[key]
        mask[i, :length] = True

    return out, mask


def pad_2d(items, key, pad_value=0):
    max_len = max(item[key].shape[0] for item in items)
    tail_shape = items[0][key].shape[1:]
    out = items[0][key].new_full((len(items), max_len, *tail_shape), pad_value)
    mask = torch.zeros((len(items), max_len), dtype=torch.bool)

    for i, item in enumerate(items):
        length = item[key].shape[0]
        out[i, :length] = item[key]
        mask[i, :length] = True

    return out, mask


def collate_inverse_truss(items):
    y = torch.stack([item["y"] for item in items])
    n = torch.stack([item["n"] for item in items])
    k = torch.stack([item["k"] for item in items])

    coordinates, node_mask = pad_2d(items, "coordinates", pad_value=0.0)
    edges, edge_mask = pad_2d(items, "edges", pad_value=-1)
    parent_sequence, parent_mask = pad_1d(items, "parent_sequence", pad_value=-1)
    extra_edges, extra_edge_mask = pad_2d(items, "extra_edges", pad_value=-1)
    coordinate_tokens, coord_token_mask = pad_1d(items, "coordinate_tokens", pad_value=0)

    return {
        "sample_id": [item["sample_id"] for item in items],
        "split": [item["split"] for item in items],
        "y": y,
        "n": n,
        "k": k,
        "coordinates": coordinates,
        "node_mask": node_mask,
        "edges": edges,
        "edge_mask": edge_mask,
        "parent_sequence": parent_sequence,
        "parent_mask": parent_mask,
        "extra_edges": extra_edges,
        "extra_edge_mask": extra_edge_mask,
        "coordinate_tokens": coordinate_tokens,
        "coord_token_mask": coord_token_mask,
    }
```

## 11. Sanity Checks

Run these checks after loading data and before training:

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

For model-side 0-based tensors:

```python
def check_zero_based_parent_sequence(parent_sequence_0based):
    for pos, parent in enumerate(parent_sequence_0based.tolist()):
        child = pos + 1
        assert 0 <= parent < child
```

Current export summary:

```text
main file: inverse_truss_property_grid_v1_0_9999.jsonl
samples: 10000
split: train 9000 / val 500 / test 500
skipped: 0
schema_version: inverse_truss_property_grid_v1
property representation: stress_grid_v1
topology_tree_method: bfs_parent_pointer_reindexed_v2
index_base: 1
coordinate_normalization: bbox_to_unit_cube
coordinate_quantization_bits: 10
```

Validation summary for the 10000-sample export:

```text
line_count: 10000
y shape: [31]
coordinate range: [0, 1]
parent-pointer rule: for child j in [2, n], 1 <= p_j < j
split: train 9000 / val 500 / test 500
```
