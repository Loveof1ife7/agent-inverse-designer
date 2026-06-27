# Windows Migration Guide

This guide describes how to run the lightweight project package on Windows.

## Package Types

```text
lite package:
  source code
  docs
  tests
  demo scripts
  small preprocessed inverse-design dataset

full data package:
  lite package
  raw structures / properties
  large data needed to rebuild pretraining datasets
```

Do not package generated runtime folders by default:

```text
workspace/
experiments/
__pycache__/
.git/
```

## Environment

Recommended:

```powershell
conda activate agent-material
python -m pip install -r requirements.txt
```

Or use a venv:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run tests:

```powershell
python -m pytest tests -q
```

## Current Closed-Loop Direction

The target architecture is:

```text
offline cold start:
  DatagenFEMEvaluator generates large pretraining data
  InverseDesigner learns target/property -> explicit structure

online closed loop:
  final target T
  -> AgentExplorer proposes target schedule
  -> InverseDesigner samples explicit structures
  -> FEMEvaluator evaluates structures
  -> RawExperimentStore records observations
  -> KnowledgeBase stores target-schedule evidence
  -> InverseDesigner replay / finetune dataset is updated
```

`DatagenFEMEvaluator` should not be the online exploration engine. It remains the offline data factory.

## FEM Backend

For Windows smoke tests, proxy FEM is enough:

```text
--fem-backend proxy
```

Abaqus can be used later if available:

```powershell
$env:ABAQUS_CMD = "abaqus"
```

or by ensuring `abaqus` / `abq2022` is on `PATH`.

## Data Layout

The lightweight package may include:

```text
train_datas/
  README.md
  P222_paired_dataset_0_99999_20260620/
    README_dataset.md
    preprocessed/
      inverse_truss_property_grid_v1_0_99.jsonl
      inverse_truss_property_grid_v1_0_99_manifest.json
```

The lightweight package usually excludes:

```text
train_datas/*/structures/
train_datas/*/properties/
train_datas/*.zip
train_datas/*.7z
```

## Migration Priority

1. Verify tests on Windows.
2. Verify explicit-structure FEM evaluation with proxy backend.
3. Verify InverseDesigner can load preprocessed cold-start data.
4. Migrate scheduler semantics from datagen-in-loop to target-schedule loop.
5. Add Abaqus backend only after proxy loop is stable.
