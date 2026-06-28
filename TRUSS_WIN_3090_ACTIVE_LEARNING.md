# Truss Closed Loop: Windows FEM + 3090 GraphMetaMat

This document narrows the closed-loop system to the truss structure family.

Goal:

```text
Windows Agent plans target curves
  -> calls 3090 truss inverse designer
  -> receives candidate truss structures
  -> optionally calls 3090 forward surrogate for quick screening
  -> runs Windows real FEM for truth labels
  -> writes evaluated_samples.jsonl
  -> uploads active data to 3090
  -> 3090 finetunes GraphMetaMat
```

The non-negotiable rule:

```text
Windows FEM response is the training label.
GraphMetaMat forward prediction is only a surrogate for screening / acquisition.
```

## 1. 3090 Interfaces to Expose

The 3090 server should expose two truss model functions.

```text
inverse designer:
  target property -> structure

forward surrogate:
  structure -> predicted property
```

For now these are exposed as SSH-callable Python runners. They can later be wrapped in HTTP without changing request/response schemas.

## 2. Inverse Designer Interface

Runner:

```text
tools/run_inverse_design_job.py
```

Supported family:

```text
structure_family = "truss"
```

Request:

```json
{
  "job_id": "round001_truss_target001",
  "structure_family": "truss",
  "target": {
    "type": "stress_curve",
    "strain_grid": [256 floats],
    "stress": [256 floats]
  },
  "options": {
    "project_dir": "third-party/GraphMetaMat",
    "num_runs": 16,
    "top_k": 4,
    "device": "cuda",
    "timeout_seconds": 3600
  }
}
```

Target curve convention:

```python
strain_grid = np.linspace(0.0, 0.3, 256)
```

If Windows Agent has a target curve on another grid, resample to this grid before submitting.

SSH call from Windows:

```powershell
$root = "/root/autodl-tmp/projects/agent-material"
$job = "round001_truss_target001"

ssh agent-3090 "mkdir -p $root/workspace/remote_inverse_jobs/$job"
scp .\request_truss.json agent-3090:$root/workspace/remote_inverse_jobs/$job/request.json
ssh agent-3090 "cd $root && python tools/run_inverse_design_job.py --request workspace/remote_inverse_jobs/$job/request.json --output workspace/remote_inverse_jobs/$job/response.json"
scp agent-3090:$root/workspace/remote_inverse_jobs/$job/response.json .\response_truss.json
scp -r agent-3090:$root/workspace/remote_inverse_jobs/$job .\remote_inverse_jobs\$job
```

Response contains:

```json
{
  "job_id": "round001_truss_target001",
  "status": "success",
  "structure_family": "truss",
  "candidate": {
    "structure_family": "truss",
    "representation": "graph_truss",
    "coordinates": [[0.0, 0.0, 0.0]],
    "edges": [[0, 1]],
    "edge_radii": [0.04],
    "rho": 0.10,
    "predicted_property": {
      "mae": 0.0,
      "mse": 0.0,
      "jaccard": 0.9,
      "num_nodes": 38,
      "num_edges": 72
    },
    "artifacts": {
      "gpkl": ".../rank_01_sample_000.gpkl",
      "vtk": ".../rank_01_sample_000.vtk",
      "graph_png": ".../rank_01_sample_000_graph.png",
      "curve_png": ".../rank_01_sample_000_curves.png",
      "summary_csv": ".../summary.csv",
      "top_designs_json": ".../top_designs.json"
    }
  }
}
```

The `predicted_property` in this response comes from GraphMetaMat's forward surrogate during inverse search. It is not a true FEM label.

## 3. Forward Surrogate Interface

Runner:

```text
tools/run_truss_forward_predict.py
```

Purpose:

```text
candidate graph -> GraphMetaMat forward-predicted stress curve
```

This is useful for cheap screening, uncertainty-aware acquisition, and sanity checks before Windows spends FEM time.

Request inputs:

```text
--graph:
  .gpkl candidate graph, usually from inverse response candidate.artifacts.gpkl

--out-dir:
  3090 output directory

--output:
  JSON response path

--device:
  cuda or cpu
```

SSH call from Windows:

```powershell
$root = "/root/autodl-tmp/projects/agent-material"
$job = "round001_truss_target001"
$graph = "$root/workspace/remote_inverse_jobs/$job/inverse_designer/graphmetamat_truss_remote/sample_000001/design_exports/rank_01_sample_000.gpkl"

ssh agent-3090 "cd $root && python tools/run_truss_forward_predict.py --graph $graph --out-dir workspace/remote_forward_jobs/$job --output workspace/remote_forward_jobs/$job/forward_response.json --device cuda"
scp agent-3090:$root/workspace/remote_forward_jobs/$job/forward_response.json .\forward_response_truss.json
```

Response:

```json
{
  "status": "success",
  "structure_family": "truss",
  "representation": "graph_truss",
  "input_graph_path": ".../rank_01_sample_000.gpkl",
  "predicted_property": {
    "task": "compression_stress_strain",
    "strain_grid": [256 floats],
    "stress": [256 floats],
    "c_shape": [256 floats],
    "c_magnitude": -1.66,
    "c_shape_std": [256 floats],
    "c_magnitude_std": [1 float],
    "rho": 0.10,
    "num_nodes": 38,
    "num_edges": 72
  }
}
```

Forward surrogate prediction can be used for:

```text
candidate ranking
uncertainty scoring
simulation budget allocation
detecting obviously bad structures
```

It must not be written as `response.stress` in the active dataset.

## 4. Windows Transfer Data Standard

Windows owns the active learning round directory.

Recommended Windows layout:

```text
round001/
  targets/
    target_schedule.json
    truss_target_001.json
  inverse_responses/
    response_truss_target_001.json
  candidates/
    round001_truss_target001_rank01.gpkl
    round001_truss_target001_rank01.vtk
  forward_surrogate/
    forward_response_truss_target001_rank01.json
  fem_runs/
    round001_truss_target001_rank01/
  evaluated_samples.jsonl
  selected_samples.jsonl
```

Target schedule format:

```json
{
  "round_id": "round001",
  "targets": [
    {
      "target_id": "round001_truss_target001",
      "structure_family": "truss",
      "reason": "probe high energy absorption with low plateau stress",
      "target": {
        "type": "stress_curve",
        "strain_grid": [256 floats],
        "stress": [256 floats]
      },
      "inverse_options": {
        "num_runs": 16,
        "top_k": 4,
        "device": "cuda"
      }
    }
  ]
}
```

Candidate identity must be stable across all files:

```text
candidate_id = <target_id>_rank<rank>
```

Example:

```text
round001_truss_target001_rank01
```

## 5. Windows Real FEM Eval Standard

Windows writes:

```text
evaluated_samples.jsonl
```

Each line is one candidate with real FEM results.

Schema:

```json
{
  "candidate_id": "round001_truss_target001_rank01",
  "target_id": "round001_truss_target001",
  "structure_family": "truss",
  "representation": "graph_truss",
  "inverse_designer": {
    "name": "GraphMetaMat",
    "checkpoint_path": "checkpoints/stressstrain_standard_RL.pt",
    "requested_target": {
      "type": "stress_curve",
      "strain_grid": [256 floats],
      "stress": [256 floats]
    },
    "num_runs": 16,
    "top_k": 4
  },
  "structure": {
    "gpkl_path": "candidates/round001_truss_target001_rank01.gpkl",
    "vtk_path": "candidates/round001_truss_target001_rank01.vtk",
    "coordinates": [[0.0, 0.0, 0.0]],
    "edges": [[0, 1]],
    "edge_radii": [0.04],
    "rho": 0.10,
    "num_nodes": 38,
    "num_edges": 72
  },
  "surrogate_prediction": {
    "source": "GraphMetaMat forward ensemble on 3090",
    "strain_grid": [256 floats],
    "stress": [256 floats],
    "c_shape_std": [256 floats],
    "c_magnitude_std": [1 float]
  },
  "evaluation": {
    "eval_status": "success",
    "geometry_status": "valid",
    "fem_status": "success",
    "fidelity": "abaqus",
    "evaluator": "windows_abaqus_truss_compression_v1",
    "failure_reason": "",
    "artifacts": {
      "raw_curve_csv": "fem_runs/.../stress_strain.csv",
      "fem_run_dir": "fem_runs/round001_truss_target001_rank01"
    }
  },
  "response": {
    "task": "compression_stress_strain",
    "strain_grid": [256 floats],
    "stress": [256 floats],
    "curve": [[0.0, 0.0]],
    "relative_density": 0.10
  },
  "metrics": {
    "target_mae": 0.0,
    "target_mse": 0.0,
    "target_jaccard": 1.0,
    "simulation_cost": 0.0
  },
  "metadata": {
    "simulator": "Abaqus on Windows",
    "created_at": "..."
  }
}
```

Required label:

```text
response.strain_grid = np.linspace(0.0, 0.3, 256)
response.stress      = Windows FEM stress resampled to that grid
```

If FEM fails, still write a record:

```json
{
  "evaluation": {
    "eval_status": "failed",
    "geometry_status": "invalid",
    "fem_status": "not_run",
    "failure_reason": "mesh_self_intersection"
  },
  "response": {}
}
```

Failure records improve Agent scheduling and filtering, but do not enter supervised finetuning.

## 6. Active Learning Strategy

Windows Agent plans targets using:

```text
1. user objective
2. previous evaluated_samples.jsonl
3. target-space coverage gaps
4. GraphMetaMat surrogate uncertainty
5. FEM failure modes
6. near-miss structures
```

Round flow:

```text
1. Agent writes target_schedule.json.
2. For each target, Windows calls 3090 inverse designer.
3. Windows downloads candidates.
4. Windows optionally calls 3090 forward predictor for each candidate.
5. Agent ranks candidates for FEM:
     high predicted target match
     high uncertainty if exploration is desired
     topology diversity
     acceptable node/edge/rho/radius constraints
6. Windows FEM evaluates selected candidates.
7. Windows writes evaluated_samples.jsonl.
8. Windows uploads evaluated_samples.jsonl and accepted artifacts to 3090.
9. 3090 builds active dataset version and finetunes.
10. Windows runs benchmark FEM eval before checkpoint promotion.
```

Acquisition score example:

```text
score = w_match * surrogate_target_match
      + w_unc   * surrogate_uncertainty
      + w_div   * topology_diversity
      - w_cost  * estimated_simulation_cost
      - w_fail  * predicted_failure_risk
```

Where:

```text
surrogate_target_match:
  jaccard / mae between surrogate prediction and requested target

surrogate_uncertainty:
  c_shape_std and c_magnitude_std from GraphMetaMat forward ensemble

topology_diversity:
  graph hash, edge set distance, node count, edge count, rho bins

estimated_simulation_cost:
  node count, edge count, mesh complexity, expected Abaqus runtime
```

## 7. Upload to 3090

Windows PowerShell:

```powershell
$root = "/root/autodl-tmp/projects/agent-material"
$round = "round001"

ssh agent-3090 "mkdir -p $root/workspace/truss_active_learning/$round/windows_eval"
scp .\round001\evaluated_samples.jsonl agent-3090:$root/workspace/truss_active_learning/$round/windows_eval/evaluated_samples.jsonl
scp -r .\round001\candidates agent-3090:$root/workspace/truss_active_learning/$round/windows_eval/candidates
scp -r .\round001\fem_runs agent-3090:$root/workspace/truss_active_learning/$round/windows_eval/fem_runs
```

3090 output locations:

```text
workspace/truss_active_learning/round001/
  windows_eval/
  dataset_active_round001/
  logs_forward_round001/
  logs_inverse_round001/
  benchmark/
```

## 8. Finetuning Strategy on 3090

Input:

```text
workspace/truss_active_learning/<round>/windows_eval/evaluated_samples.jsonl
```

Select supervised records:

```text
structure_family == "truss"
evaluation.eval_status == "success"
evaluation.geometry_status == "valid"
evaluation.fem_status == "success"
response.stress length == 256
```

Build GraphMetaMat dataset:

```text
dataset_active_round001/
  train/
    graphs/{GID}.gpkl
    graphs/{GID}_polyhedron.gpkl
    curves/{CID}.pkl
    mapping.tsv
  dev/
  test/
```

Curve projection:

```python
curve = np.stack([record["response"]["strain_grid"], record["response"]["stress"]], axis=-1)
# shape [256, 2]
```

Graph projection:

```text
Use record.structure.gpkl_path if available.
Otherwise rebuild NetworkX graph from coordinates, edges, edge_radii, rho.
Generate node_feats, edge_feats, edge_index before training.
```

Polyhedron requirement:

```text
Forward finetune:
  needs graphs/{GID}.gpkl and curves/{CID}.pkl

Inverse RL finetune:
  can use generated policy and latest forward ensemble

Inverse IL finetune:
  requires graphs/{GID}_polyhedron.gpkl or an action-sequence reconstruction pipeline
```

Recommended finetune order:

```text
1. Finetune forward ensemble on active dataset.
2. Validate forward ensemble on fixed held-out curves.
3. If polyhedron/action labels exist, run inverse IL warm start.
4. Run inverse RL with latest forward ensemble.
5. Run fixed benchmark target curves through inverse designer.
6. Windows evaluates benchmark structures with real FEM.
7. Promote checkpoint only if Windows FEM benchmark improves.
```

Do not promote on surrogate metrics alone.

## 9. 3090 HTTP API Target

The SSH runners are the MVP. The service version should expose the same schemas:

```text
POST /truss/inverse-design/jobs
GET  /truss/inverse-design/jobs/{job_id}
GET  /truss/inverse-design/jobs/{job_id}/artifacts.tar.gz

POST /truss/forward-predict/jobs
GET  /truss/forward-predict/jobs/{job_id}

POST /truss/evaluated-samples
POST /truss/finetune/jobs
GET  /truss/finetune/jobs/{job_id}
```

HTTP should not change the meaning of any field. It only removes SSH orchestration.

## 10. Minimal Truss Round Checklist

Windows:

```text
[ ] Write target_schedule.json with 256-point stress curves.
[ ] Submit inverse jobs to 3090.
[ ] Download gpkl/vtk/png artifacts.
[ ] Optionally call forward predictor for each candidate.
[ ] Select candidates for FEM.
[ ] Run real FEM.
[ ] Resample true stress to np.linspace(0.0, 0.3, 256).
[ ] Write evaluated_samples.jsonl.
[ ] Upload round directory to 3090.
```

3090:

```text
[ ] Run inverse designer jobs.
[ ] Run forward surrogate jobs if requested.
[ ] Build dataset_active_roundXXX.
[ ] Finetune forward ensemble.
[ ] Finetune inverse policy.
[ ] Generate benchmark candidates.
[ ] Wait for Windows benchmark FEM.
[ ] Promote or reject checkpoint.
```

One-line summary:

```text
Windows plans targets and supplies truth; 3090 proposes truss structures, predicts cheaply, and trains.
```

