# Agent Material 项目部署与工作流说明

本文档面向把本项目迁移到 Windows 后运行、调试和继续开发。当前项目是一个面向 truss / architected material 的 Python 原型系统，核心目标是：

```text
目标性质 target property -> 反向设计显式结构 explicit structure -> FEM / proxy 评价 -> 数据与知识闭环更新
```

## 1. 项目目录

```text
agent-material/
  src/                         核心 Python 包
  demos/                       可直接运行的演示脚本
  tests/                       pytest 测试
  tools/                       打包、格式转换等工具脚本
  docs/                        设计文档、迁移说明、接口说明
  train_datas/                 训练数据与预处理数据
  requirements.txt             Python 依赖
  setup_and_workflow.md        本文件
```

默认 Windows 轻量包会排除运行缓存、历史实验、`.git`、`__pycache__`、`archive/`、`workspace/`、`experiments/`、原始大规模 `structures/` 和 `properties/`。轻量包会保留预处理 JSONL，足够运行 demo、测试闭环和做 InverseDesigner 数据接口验证。

## 2. 核心模块

### `src/InverseDesigner`

主反向设计模块。当前实现是 deterministic retrieval baseline：

```text
target_property -> 从训练样本中检索最近性质 -> 返回 explicit_structure
```

它不负责 meta 参数搜索。长期可以替换为真正的生成模型，例如：

```text
property curve / summary -> topology + coordinates
```

### `src/AgentExplorer`

辅助探索模块。只有当 `InverseDesigner` 给出的显式结构不能满足目标时，才启动 meta search。当前 demo 默认关闭 LLM，使用确定性 fallback，因此在 Windows 上无需配置外部模型服务也能跑通。

如果启用 LLM，相关环境变量在代码中支持本地 OpenAI-compatible 服务：

```powershell
$env:AGENT_EXPLORER_ENABLE_LLM="1"
$env:AGENT_EXPLORER_BASE_URL="http://127.0.0.1:17777/v1"
$env:AGENT_EXPLORER_API_KEY="..."
$env:AGENT_EXPLORER_MODEL="..."
```

### `src/DatagenFEMEvaluator`

结构生成与评价模块，包含：

```text
meta parameters -> structure generation -> FEM / proxy evaluation -> sample records
```

主要能力：

- symmetry constraints 求解
- truss architecture CSV 生成
- Abaqus txt 转换
- VTK 导出
- proxy FEM 评价
- Abaqus FEM 后端调用
- 批量 group 生成调度

Windows 上建议先使用 `--fem-backend proxy` 验证闭环。Abaqus 后端是可选增强，需要 Windows 机器已经安装 Abaqus 并配置 `abaqus` 命令或 `ABAQUS_CMD` 环境变量。

### `src/KnowledgeBase`

SQLite 知识库。用于保存已评价结构、性质结果、失败原因和检索证据。默认路径通常是：

```text
workspace/knowledge.sqlite
```

### `src/KnowledgeRefiner`

把样本级 evidence 汇总成 agent-facing knowledge，包括统计摘要和可解释证据。它服务于 `AgentExplorer`，不直接作为 `InverseDesigner` 的训练数据。

### `src/ExperimentStore`

append-only JSONL 实验记录仓库。用于保存所有评价 observation，并投影为：

```text
property_result -> explicit_structure       供 InverseDesigner 训练/微调
meta -> structure -> property evidence      供 KnowledgeBase / AgentExplorer 使用
```

### `src/Scheduler`

闭环调度器。核心类是 `StructureDiscoverySystem`，负责把以下模块串起来：

```text
InverseDesigner
AgentExplorer
DatagenFEMEvaluator
KnowledgeBase
KnowledgeRefiner
ExperimentStore
```

### `src/TrainingDataset`

训练数据导出与预处理模块。当前重点数据接口是：

```text
target property y -> explicit truss structure
```

预处理脚本：

```text
src/TrainingDataset/inverse_truss_preprocess.py
```

训练数据说明：

```text
train_datas/INVERSE_DESIGNER_DATASET.md
train_datas/P222_paired_dataset_0_99999_20260620/preprocessed/DATALOADER.md
```

### `src/MetaSpace`

设计空间定义模块，为 `AgentExplorer` 和数据生成提供 group、density、bar 数量等参数空间。

### `src/api.py`、`src/agent_api.py`、`src/cli.py`

对外入口：

- `src/api.py`：Python 函数式 API
- `src/agent_api.py`：面向 agent 的 facade，返回普通 Python dict/list
- `src/cli.py`：命令行入口，可用 `python -m src ...` 调用

## 3. Windows 环境安装

推荐使用 Python 3.10 或 3.11。假设项目解压到：

```text
C:\projects\agent-material
```

PowerShell 中执行：

```powershell
cd C:\projects\agent-material
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 PowerShell 禁止激活虚拟环境：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. 基础验证

运行测试：

```powershell
python -m pytest tests -q
```

检查 CLI：

```powershell
python -m src --help
```

运行最小约束求解：

```powershell
python -m src constraints --group P222
```

运行闭环 demo：

```powershell
python demos\closed_loop_discovery_demo.py `
  --fresh `
  --workspace workspace\win_demo `
  --max-iterations 6 `
  --experiment-budget 8 `
  --agent-batch-size 77 `
  --retrain-trigger 4 `
  --fem-backend proxy `
  --allow-failure
```

demo 输出会写入：

```text
workspace/win_demo/
  knowledge.sqlite
  inverse_retrieval_training_dataset.json
  closed_loop_events.jsonl
  demo_summary.json
  demo_summary.md
```

## 5. 主要运行方式

### 5.1 运行闭环发现 demo

默认读取小样本预处理数据：

```text
train_datas/P222_paired_dataset_0_99999_20260620/preprocessed/inverse_truss_property_grid_v1_0_99.jsonl
```

命令：

```powershell
python demos\closed_loop_discovery_demo.py --fresh --fem-backend proxy --allow-failure
```

指定目标性质：

```powershell
python demos\closed_loop_discovery_demo.py `
  --fresh `
  --target-property "{""stiffness_proxy"":0.30,""density_proxy"":0.02}" `
  --fem-backend proxy `
  --allow-failure
```

使用 10000 条预处理数据：

```powershell
python demos\closed_loop_discovery_demo.py `
  --fresh `
  --preprocessed-jsonl train_datas\P222_paired_dataset_0_99999_20260620\preprocessed\inverse_truss_property_grid_v1_0_9999.jsonl `
  --limit 10000 `
  --fem-backend proxy `
  --allow-failure
```

### 5.2 使用 CLI 跑 closed-loop discovery

```powershell
python -m src discover `
  --target-property "{""stiffness_proxy"":25,""density_proxy"":0.1}" `
  --workspace-root workspace `
  --kb-path workspace\knowledge.sqlite `
  --max-iterations 3
```

### 5.3 生成结构 CSV

```powershell
python -m src generate `
  --output-dir workspace\gen `
  --csv-name P222-architecture.csv `
  --samples 100 `
  --workers 1 `
  --batch 1 `
  --allow-single-process-fallback
```

### 5.4 运行 group pipeline

```powershell
python -m src pipeline P222 `
  --basic-size 4 `
  --samples 100 `
  --workers 1 `
  --batch 1 `
  --rho-target 0.1 `
  --max-bars 10 `
  --allow-single-process-fallback
```

### 5.5 导出 VTK

把 Abaqus txt / truss txt 转成 VTK：

```powershell
python -m src vtk `
  --input workspace\some_txt_dir `
  --output workspace\vtk `
  --glob "*.txt"
```

### 5.6 查询知识库

```powershell
python -m src kb-stats --kb-path workspace\knowledge.sqlite
```

```powershell
python -m src kb-query `
  --kb-path workspace\knowledge.sqlite `
  --type similar `
  --target-property "{""stiffness_proxy"":25,""density_proxy"":0.1}" `
  --top-k 10
```

### 5.7 导出 InverseDesigner 训练数据

```powershell
python -m src export-training-dataset `
  --kb-path workspace\knowledge.sqlite `
  --output workspace\inverse_designer_dataset.json
```

## 6. 数据说明

轻量包包含：

```text
train_datas/
  README.md
  INVERSE_DESIGNER_DATASET.md
  P222_paired_dataset_0_99999_20260620/
    README_dataset.md
    数据说明.md
    preprocessed/
      DATALOADER.md
      inverse_truss_property_grid_v1_0_99.jsonl
      inverse_truss_property_grid_v1_0_99_manifest.json
      inverse_truss_property_grid_v1_0_9999.jsonl
      inverse_truss_property_grid_v1_0_9999_manifest.json
```

轻量包不包含：

```text
train_datas/P222_paired_dataset_0_99999_20260620/structures/
train_datas/P222_paired_dataset_0_99999_20260620/properties/
train_datas/*.zip
train_datas/*.7z
```

如果要重新从原始结构和曲线构建数据集，需要完整数据包或手动拷贝原始数据目录。

## 7. Abaqus 后端

默认 demo 使用：

```text
--fem-backend proxy
```

如果 Windows 上安装了 Abaqus，可以切换：

```powershell
python demos\closed_loop_discovery_demo.py --fresh --fem-backend abaqus
```

Abaqus 命令需要满足以下任一条件：

- `abaqus`、`abq2022` 等命令已经加入 `PATH`
- 设置 `ABAQUS_CMD`

示例：

```powershell
$env:ABAQUS_CMD="C:\SIMULIA\Commands\abaqus.bat"
```

也可以使用：

```text
--fem-backend auto
```

该模式会优先尝试 Abaqus，失败后回退到 proxy。

## 8. 在 Linux 端打包给 Windows

生成轻量包：

```bash
python tools/package_windows.py
```

默认输出：

```text
dist/agent-material-windows-lite.zip
```

生成完整数据包：

```bash
python tools/package_windows.py --include-full-data --output dist/agent-material-windows-full.zip
```

轻量包适合迁移运行和调试；完整包会包含原始 `structures/`、`properties/`，体积明显更大。

## 9. 推荐迁移工作流

```text
1. Linux 端运行 package_windows.py 生成 zip
2. 拷贝 zip 到 Windows
3. 解压到 C:\projects\agent-material
4. 创建 venv 并安装 requirements.txt
5. python -m pytest tests -q
6. python demos\closed_loop_discovery_demo.py --fresh --fem-backend proxy --allow-failure
7. 检查 workspace\win_demo\demo_summary.md
8. 如需真实 FEM，再配置 Abaqus 并切换 --fem-backend abaqus / auto
```

## 10. 常见问题

### 找不到 `src`

请在项目根目录运行命令，或显式设置：

```powershell
$env:PYTHONPATH=(Get-Location).Path
```

### PowerShell JSON 引号报错

PowerShell 中建议使用双引号转义形式：

```powershell
--target-property "{""stiffness_proxy"":0.30,""density_proxy"":0.02}"
```

### demo 未找到成功样本

闭环搜索是实验流程，目标可能较难。用于流程验证时加：

```text
--allow-failure
```

此时即使没有找到 success，也会输出完整 summary 和 workspace 结果。

### Windows 路径包含中文或空格导致问题

建议解压到：

```text
C:\projects\agent-material
```

避免放在桌面、OneDrive 或含中文/空格的路径下。
