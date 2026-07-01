# DatagenFEMEvaluator Interface

`src/DatagenFEMEvaluator` is the truss-structure dataset generation and
FEM-Eval facade. It wraps the P222 / symmetry-group truss generation pipeline,
the conversion from generated topology records to Abaqus-compatible truss txt
files, optional crystal expansion, and property evaluation through proxy or
Abaqus FEM.

In the closed-loop system, this package is an offline data factory for cold
start and dataset building. It should not be treated as the online exploration
engine; online candidates should come from `InverseDesigner`, and explicit
structures should be validated by the FEM evaluation path.

```
1. 输入 group
   从 symmetry_group_transforms.json 读群矩阵 M_sym、lattice_lengths。
   constraints_solver.py 根据 group 推出 k/p 参数约束。

2. 构造 base 点
   dataset_generator.py 在 19 个模板点上采样：
   A1-A12 边点
   q_front/q_back/... 面点
   v_center 中心点

3. 生成 base 连接关系
   在这些 base 点之间随机选 active nodes，然后先造一棵连通树，再补边。
   这里受到 MAX_BARS 限制。

4. 生成连接时做局部约束检查
   每尝试加一条 base edge，会先把这条边通过 group 展开到 unit cell 的对称副本。
   然后检查：
   - 是否产生有效新边
   - 是否重复
   - 是否违反杆间 clearance
   - 是否超过 max_bars
   - 是否能维持当前构造逻辑

5. base graph 初步连通检查
   base_edges 生成后，检查 base active nodes 连通。

6. group 展开成 unit cell
   apply_symmetry_ops(base_nodes, base_edges, group)。
   合并重合节点、去重边。

7. unit cell / PBC / array 约束检查
   这里还有几层检查：
   - unit cell active nodes 连通
   - PBC 周期识别后连通
   - PBC 后 degree==1 节点剔除
   - 2x2x2 array 连通

8. 输出 CSV
   CSV 保存的是 base_nodes + base adjacency + L。

9. CSV -> unit cell txt
   abaqus_converter.py 再读 CSV，用 group 矩阵展开成 unit cell txt。

10. expand
   crystal_builder.py 把 unit cell txt 复制扩展成等效 4x4x4 crystal txt。

```

This package is the agent-facing scheduler facade for the truss datagen and
FEM-Eval engine under `DatagenFEMEvaluator/core/truss`.

The core math and geometry scripts remain the source of truth.
The public interface here is responsible for:

- launching the core pipeline
- generating truss-structure dataset artifacts
- converting generated truss records into Abaqus-compatible txt files
- evaluating generated or explicit truss structures with proxy or Abaqus FEM
- cross-platform path and environment compatibility
- collecting structured outputs
- exposing KB-friendly generated-data manifests

## Stable Public API

Preferred class:

- `DatagenFEMEvaluator`

Preferred functions:

- `run_auto_generate_4x4x4`
- `run_all_groups_4x4x4`
- `get_interface_contract`

Preferred methods:

- `DatagenFEMEvaluator.auto_generate_4x4x4(config)`
- `DatagenFEMEvaluator.run_group(config)`
- `DatagenFEMEvaluator.run_group_pipeline(config)`
- `DatagenFEMEvaluator.run_all_groups_4x4x4(config)`
- `DatagenFEMEvaluator.run_all_groups(config)`
- `DatagenFEMEvaluator.interface_contract()`
- `DatagenFEMEvaluator.datagen_schema()`
- `DatagenFEMEvaluator.datagen(config)`
- `DatagenFEMEvaluator.bootstrap_dataset_and_kb(datagen_configs, kb_path, output_dir=None)`
- `DatagenFEMEvaluator.fem_evaluate(structures)`
- `DatagenFEMEvaluator.collect_samples(structures, fem_results, target_property, datagen_config)`
- `DatagenFEMEvaluator.evaluate_existing_candidate(candidate, target_property)`

## Input Types

### `DatagenConfig`

`DatagenConfig` is the executable suggestion produced by `AgentExplorer`.
It is also persisted into `KnowledgeBase` metadata so every generated structure keeps its origin and reasoning.

Identity and provenance:

- `suggestion_id: str`
- `parent_sample_id: str`
- `source: str`

Target and expectation:

- `target_property: dict[str, float]`
- `expected_property: dict[str, float]`
- `objective: str`
- `confidence: float`

Runtime controls:

- `group: str`
- `basic_size: int`
- `num_samples: int`
- `workers: int`
- `batch: int`
- `print_every: int`
- `run_dir: str`

Structure semantics:

- `symmetry: str`
- `basic_unit_type: str`
- `unit_cell_type: str`
- `topology_type: str`
- `connectivity_pattern: str`

Search-space controls:

- `max_bars: int`
- `rho_target: float`
- `density_range: tuple[float, float]`
- `parameter_ranges: dict[str, Any]`
- `sampling_strategy: str`
- `constraints: dict[str, Any]`

Agent reasoning:

- `hypothesis: str`
- `reason: str`
- `failure_analysis: dict[str, Any]`
- `exploration_strategy: str`
- `tags: tuple[str, ...]`

### `AutoGenerateConfig`

- `group: str`
- `basic_size: int`
- `samples: int`
- `workers: int`
- `batch: int`
- `print_every: int`
- `group_db: str`
- `run_dir: str`
- `resume: bool`
- `allow_single_process_fallback: bool`

Use this for one symmetry group run.

### `BatchGenerateConfig`

- `workers: int`
- `samples: int`
- `basic_size: int`
- `poll_seconds: int`
- `idle_timeout_minutes: int`
- `group_timeout_minutes: int`
- `stop_on_failure: bool`
- `include_groups: tuple[str, ...]`
- `exclude_groups: tuple[str, ...]`
- `group_db: str`
- `output_root: str`
- `batch_dir: str`
- `resume: bool`
- `allow_single_process_fallback: bool`

Use this for multi-group scheduling.

## Output Types

### `AutoGenerateResult`

- `group`
- `command`
- `exit_code`
- `run_dir`
- `status`
- `summary_path`
- `summary`
- `stdout_path`
- `stderr_path`
- `generated_data_manifest_path`
- `knowledge_base_seed_path`

### `BatchGroupResult`

- `group`
- `index`
- `total`
- `status`
- `exit_code`
- `run_dir`
- `stdout_path`
- `stderr_path`
- `summary_path`
- `summary`
- `generated_data_manifest_path`
- `knowledge_base_seed_path`
- `detail`

### `BatchGenerateResult`

- `output_root`
- `batch_dir`
- `progress_path`
- `groups_total`
- `groups_finished`
- `stop_triggered`
- `results`
- `skipped`

### `BootstrapDatagenResult`

- `output_dir`
- `kb_path`
- `dataset_jsonl_path`
- `summary_path`
- `total_samples`
- `label_counts`
- `run_results`

## Artifact Contract

Single run directory:

- `constraints_<group>.json`
- `<group>-architecture.csv`
- `abaqus_txt/*.txt`
- `crystal_4x4x4/*.txt`
- `summary.json`
- `generated_data_manifest.json`
- `knowledge_base_seed.jsonl`
- `auto_generate.stdout.log`
- `auto_generate.stderr.log`

Batch root:

- `_batch/progress.tsv`
- `_batch/<index>_<group>.log`
- `_batch/<index>_<group>.err.log`
- `<group>/summary.json`
- `<group>/generated_data_manifest.json`
- `<group>/knowledge_base_seed.jsonl`

Bootstrap dataset root:

- `bootstrap_dataset.jsonl`
- `bootstrap_summary.json`
- `runs/<suggestion_id>/summary.json`
- `runs/<suggestion_id>/generated_data_manifest.json`
- `runs/<suggestion_id>/knowledge_base_seed.jsonl`

## Knowledge Base Ingestion

Preferred ingestion file:

- `knowledge_base_seed.jsonl`

Each line is one generated sample with:

- `structure_id`
- `sample_index`
- `csv_row_id`
- `csv_name`
- `group`
- `basic_size`
- `replication`
- `csv_path`
- `constraints_path`

## Closed-Loop Evaluator Methods

`datagen(config)` accepts `DatagenConfig` and returns generated structure records read from `knowledge_base_seed.jsonl`.

`bootstrap_dataset_and_kb(datagen_configs, kb_path, output_dir=None)` is the agent-facing way to build a finite seed dataset and seed knowledge base.
It executes one or more `DatagenConfig` suggestions, evaluates the generated structures, writes `bootstrap_dataset.jsonl`, inserts `KnowledgeSample` rows into the sqlite knowledge base, and persists provenance-rich metadata for later AgentExplorer reasoning.

`fem_evaluate(structures)` currently returns deterministic proxy properties:

- `stiffness_proxy`
- `density_proxy`

This method is the intended replacement point for a real FEM backend.

`collect_samples(...)` converts generated structures and FEM results into `KnowledgeSample` records with labels:

- `success`
- `near_miss`
- `failure`
- `abaqus_txt_path`
- `crystal_txt_path`
- `run_dir`

Recommended primary key:

- `structure_id`

Recommended artifact links:

- `csv_path`
- `abaqus_txt_path`
- `crystal_txt_path`
- `constraints_path`

## Bootstrap Record Semantics

Each line in `bootstrap_dataset.jsonl` is a `KnowledgeSample` and contains:

- `structure_path`
- `target_property`
- `evaluated_property`
- `property_error`
- `label`
- `source`
- `metadata.datagen_config`
- `metadata.agent_suggestion`
- `metadata.artifacts`
- `metadata.bootstrap`

The most important provenance block is `metadata.datagen_config`.
This is the exact executable `DatagenConfig` that produced the structure, including:

- who proposed it: `suggestion_id`, `parent_sample_id`, `source`
- why it exists: `objective`, `hypothesis`, `reason`, `failure_analysis`, `exploration_strategy`
- what search region it came from: `group`, `symmetry`, `max_bars`, `rho_target`, `density_range`, `parameter_ranges`, `constraints`
- what the run expected: `target_property`, `expected_property`, `confidence`

This makes the bootstrap dataset suitable both as:

- a finite exploratory base dataset
- a finite provenance-aware seed knowledge base

## Usage Example

```python
from src.DatagenFEMEvaluator import DatagenFEMEvaluator, AutoGenerateConfig

evaluator = DatagenFEMEvaluator(workspace_root="workspace")

result = evaluator.auto_generate_4x4x4(
    AutoGenerateConfig(
        group="P222",
        samples=8,
        workers=2,
        batch=2,
        print_every=1,
        allow_single_process_fallback=True,
    )
)

print(result.summary_path)
print(result.knowledge_base_seed_path)
```

```python
from src.DatagenFEMEvaluator import DatagenFEMEvaluator
from src.closed_loop_contracts import DatagenConfig

evaluator = DatagenFEMEvaluator(workspace_root="workspace")

result = evaluator.bootstrap_dataset_and_kb(
    datagen_configs=[
        DatagenConfig(
            suggestion_id="bootstrap_001_P222",
            source="bootstrap_seed",
            group="P222",
            symmetry="P222",
            num_samples=4,
            workers=1,
            batch=1,
            print_every=1,
            rho_target=0.1,
            max_bars=10,
            hypothesis="Bootstrap a finite exploratory base dataset.",
            reason="Create a regularized seed dataset and seed knowledge base for downstream search.",
            exploration_strategy="bootstrap_seed",
            tags=("bootstrap_seed", "P222"),
        )
    ],
    kb_path="workspace/seed_kb.sqlite",
    output_dir="workspace/bootstrap_seed",
)

print(result.dataset_jsonl_path)
print(result.summary_path)
print(result.kb_path)
```
