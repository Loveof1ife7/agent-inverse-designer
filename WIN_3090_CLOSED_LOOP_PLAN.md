# Windows FEM + 3090 Inverse-Designer Closed Loop

This document defines the deployment plan for running the closed-loop material discovery system when:

- Windows owns the agent, scheduler, and real FEM/simulation environment.
- Linux 3090 owns neural inverse designers and finetuning.
- Windows calls the 3090 remotely to generate candidates, evaluates them with the real simulator, then uploads verified labels back to the 3090.

The rule of the system is:

```text
Windows FEM is the truth source.
3090 inverse designers only propose candidate structures.
```

Never use an inverse-designer requested target or surrogate prediction as a finetuning label.

## 1. Role Split

```text
Windows machine
  AgentExplorer / scheduler
  Knowledge and feedback for the active round
  Abaqus / real FEM / high-fidelity simulator
  evaluated_samples.jsonl writer
  active dataset delta packaging

Linux 3090 server
  DiffuMeta TPMS inverse designer
  GraphMetaMat truss inverse designer
  future voxel / B-spline inverse designers
  active dataset builder
  finetuning and checkpoint promotion
```

Online loop:

```text
1. Windows Agent proposes target schedule / parameter path.
2. Windows submits inverse-design jobs to 3090.
3. 3090 returns candidate structures and artifacts.
4. Windows runs real FEM on candidates.
5. Windows writes evaluated_samples.jsonl with real response labels.
6. Windows uploads evaluated_samples.jsonl and accepted artifacts to 3090.
7. 3090 builds active dataset versions and finetunes.
8. Windows switches future jobs to the promoted checkpoint.
```

## 2. Current 3090 Interfaces

Two inverse designers are available on the 3090 side.

### TPMS: DiffuMeta

Guide:

```text
third-party/DiffusionMetamaterials/user_guide.md
third-party/DiffusionMetamaterials/docs/inverse_design_active_learning.md
```

Input target:

```json
{
  "type": "control_points_stress",
  "values": [11 floats]
}
```

Native model call:

```bash
cd third-party/DiffusionMetamaterials
python sample.py \
  --cfg_scale 10.0 \
  --num_samples 200 \
  --model_checkpoint model_checkpoints/model_checkpoint.pth
```

Output candidate:

```json
{
  "structure_family": "tpms",
  "representation": "implicit_equation",
  "equation": "...",
  "tokens": [22 ints],
  "obj_path": "...",
  "png_path": "..."
}
```

Generation validity is only a geometry check. The real label must come from Windows FEM.

### Truss: GraphMetaMat

Guide:

```text
third-party/GraphMetaMat/user_guide.md
```

Input target:

```json
{
  "type": "stress_curve",
  "strain_grid": [256 floats],
  "stress": [256 floats]
}
```

The strain grid must be:

```python
np.linspace(0.0, 0.3, 256)
```

Native model call:

```bash
cd third-party/GraphMetaMat
python run_inverse_designer.py \
  --target path/to/target_curve.csv \
  --out-dir outputs/my_design \
  --num-runs 128 \
  --top-k 8 \
  --device cuda
```

Output candidate:

```json
{
  "structure_family": "truss",
  "representation": "graph_truss",
  "gpkl_path": "...",
  "vtk_path": "...",
  "coordinates": [[...]],
  "edges": [[...]],
  "edge_radii": [...],
  "rho": 0.08,
  "surrogate_metrics": {
    "mae": 0.0,
    "jaccard": 0.92
  }
}
```

The surrogate metrics are useful for filtering but are not truth labels.

## 3. How Windows Codex Uses the 3090

Set up an SSH alias on Windows, for example in `~/.ssh/config`:

```text
Host agent-3090
  HostName <3090-host-or-ip>
  User <linux-user>
  IdentityFile ~/.ssh/id_ed25519
```

Assume the project root on the 3090 is:

```text
/root/autodl-tmp/projects/agent-material
```

### 3.1 SSH Job Runner

The 3090 repo provides:

```text
tools/run_inverse_design_job.py
```

Windows Codex should create a request JSON, upload it, run the job over SSH, then download the response JSON and artifacts.

Remote working directory convention:

```text
workspace/remote_inverse_jobs/<job_id>/
  request.json
  response.json
  inverse_designer/
  knowledge.sqlite
```

### 3.2 TPMS Request Example

`request_tpms.json`:

```json
{
  "job_id": "round001_tpms_target001",
  "structure_family": "tpms",
  "target": {
    "type": "control_points_stress",
    "values": [1.30, 2.72, 4.42, 5.97, 6.55, 6.57, 6.54, 6.49, 6.38, 6.32, 6.19]
  },
  "options": {
    "num_samples": 200,
    "cfg_scale": 10.0,
    "checkpoint_path": "third-party/DiffusionMetamaterials/model_checkpoints/model_checkpoint.pth"
  }
}
```

Windows PowerShell:

```powershell
$root = "/root/autodl-tmp/projects/agent-material"
$job = "round001_tpms_target001"

ssh agent-3090 "mkdir -p $root/workspace/remote_inverse_jobs/$job"
scp .\request_tpms.json agent-3090:$root/workspace/remote_inverse_jobs/$job/request.json
ssh agent-3090 "cd $root && python tools/run_inverse_design_job.py --request workspace/remote_inverse_jobs/$job/request.json --output workspace/remote_inverse_jobs/$job/response.json"
scp agent-3090:$root/workspace/remote_inverse_jobs/$job/response.json .\response_tpms.json
```

### 3.3 Truss Request Example

`request_truss.json`:

```json
{
  "job_id": "round001_truss_target001",
  "structure_family": "truss",
  "target": {
    "type": "stress_curve",
    "strain_grid": [256 floats from 0.0 to 0.3],
    "stress": [256 floats]
  },
  "options": {
    "num_runs": 16,
    "top_k": 4,
    "device": "cuda"
  }
}
```

Windows PowerShell:

```powershell
$root = "/root/autodl-tmp/projects/agent-material"
$job = "round001_truss_target001"

ssh agent-3090 "mkdir -p $root/workspace/remote_inverse_jobs/$job"
scp .\request_truss.json agent-3090:$root/workspace/remote_inverse_jobs/$job/request.json
ssh agent-3090 "cd $root && python tools/run_inverse_design_job.py --request workspace/remote_inverse_jobs/$job/request.json --output workspace/remote_inverse_jobs/$job/response.json"
scp agent-3090:$root/workspace/remote_inverse_jobs/$job/response.json .\response_truss.json
```

The response includes artifact paths on the 3090. To download all generated artifacts:

```powershell
scp -r agent-3090:$root/workspace/remote_inverse_jobs/$job .\remote_inverse_jobs\$job
```

### 3.4 Future HTTP API

SSH mode works now. For higher throughput, wrap the same runner as a 3090 service:

```text
POST /inverse-design/jobs
GET  /inverse-design/jobs/{job_id}
GET  /inverse-design/jobs/{job_id}/artifacts.tar.gz
POST /evaluated-samples
POST /finetune/jobs
GET  /finetune/jobs/{job_id}
```

The request and response JSON should stay identical to the SSH mode.

## 4. Windows FEM Eval Format

Windows must write true labels to:

```text
evaluated_samples.jsonl
```

Each line is one evaluated candidate. This schema is shared by TPMS and Truss.

Common skeleton:

```json
{
  "candidate_id": "...",
  "target_id": "...",
  "structure_family": "tpms | truss",
  "representation": "implicit_equation | graph_truss",
  "inverse_designer": {
    "name": "...",
    "checkpoint_path": "...",
    "requested_target": {}
  },
  "structure": {},
  "evaluation": {
    "eval_status": "success",
    "geometry_status": "valid",
    "fem_status": "success",
    "fidelity": "abaqus",
    "evaluator": "windows_abaqus_v1",
    "failure_reason": "",
    "artifacts": {
      "raw_curve_csv": "...",
      "fem_run_dir": "..."
    }
  },
  "response": {},
  "metrics": {},
  "metadata": {
    "simulator": "Abaqus on Windows",
    "created_at": "..."
  }
}
```

### TPMS response

TPMS records must provide labels aligned to DiffuMeta:

```json
{
  "response": {
    "task": "compression_stress_strain",
    "strain_grid": [40 floats],
    "stress": [40 floats],
    "control_points_stress": [11 floats],
    "label_array": [40 floats],
    "stiffness": [21 floats],
    "moduli": [12 floats],
    "relative_density": 0.10
  }
}
```

Required for basic DiffuMeta finetune:

```text
structure.tokens
response.control_points_stress
response.label_array
```

Optional for multi-objective finetune:

```text
response.stiffness
response.moduli
```

### Truss response

Truss records must provide labels aligned to GraphMetaMat:

```json
{
  "response": {
    "task": "compression_stress_strain",
    "strain_grid": [256 floats],
    "stress": [256 floats],
    "curve": [[0.0, 0.0]],
    "relative_density": 0.10
  }
}
```

Required strain grid:

```python
np.linspace(0.0, 0.3, 256)
```

If Windows FEM produces another sampling grid, resample the stress curve to this grid before writing `evaluated_samples.jsonl`.

## 5. Upload Back to 3090

After Windows finishes real FEM:

```powershell
$root = "/root/autodl-tmp/projects/agent-material"
$round = "round001"

ssh agent-3090 "mkdir -p $root/workspace/active_learning/$round/windows_eval"
scp .\evaluated_samples.jsonl agent-3090:$root/workspace/active_learning/$round/windows_eval/evaluated_samples.jsonl
scp -r .\accepted_artifacts agent-3090:$root/workspace/active_learning/$round/windows_eval/accepted_artifacts
```

3090 then builds model-specific active datasets from this common file.

## 6. Dataset Projection and Finetuning Strategy

### 6.1 TPMS / DiffuMeta projection

Input:

```text
workspace/active_learning/<round>/windows_eval/evaluated_samples.jsonl
```

Select records:

```text
structure_family == "tpms"
evaluation.eval_status == "success"
evaluation.geometry_status == "valid"
evaluation.fem_status == "success"
```

Project to DiffuMeta dataset fields:

```json
{
  "tokens": "structure.tokens",
  "control_points_stress": "response.control_points_stress",
  "label_array": "response.label_array",
  "stiffness": "response.stiffness",
  "moduli": "response.moduli",
  "equations": "structure.equation",
  "provenance": {}
}
```

Recommended dataset versions:

```text
third-party/DiffusionMetamaterials/data/dataset/dataset_active_round001.json
third-party/DiffusionMetamaterials/data/dataset/dataset_active_round002.json
```

Finetune strategy:

```text
1. Keep original dataset.
2. Build dataset_active_roundXXX.json = original + accepted evaluated TPMS samples.
3. Use small learning rate, around 0.1x to 0.2x original.
4. Oversample active samples only moderately.
5. Validate on fixed benchmark targets.
6. Promote checkpoint only if true evaluator benchmark improves.
```

Important:

```text
requested_target.values is provenance only.
response.control_points_stress is the label.
```

### 6.2 Truss / GraphMetaMat projection

Input:

```text
workspace/active_learning/<round>/windows_eval/evaluated_samples.jsonl
```

Select records:

```text
structure_family == "truss"
evaluation.eval_status == "success"
evaluation.geometry_status == "valid"
evaluation.fem_status == "success"
```

Project to GraphMetaMat dataset:

```text
dataset_active_roundXXX/
  train/
    graphs/{GID}.gpkl
    graphs/{GID}_polyhedron.gpkl
    curves/{CID}.pkl
    mapping.tsv
  dev/
  test/
```

`curves/{CID}.pkl`:

```python
curve = np.stack([response["strain_grid"], response["stress"]], axis=-1)  # shape [256, 2]
```

`graphs/{GID}.gpkl`:

```text
Use structure.gpkl_path when available.
Otherwise rebuild a NetworkX graph from coordinates, edges, edge_radii, and rho.
```

`graphs/{GID}_polyhedron.gpkl`:

```text
Required only for inverse IL.
If the active sample only has final truss graph/VTK, do forward finetune and inverse RL, but skip inverse IL for that sample.
```

Finetune strategy:

```text
1. Finetune forward ensemble first, because inverse RL reward depends on it.
2. If polyhedron/action data exists, run inverse IL warm start.
3. Run inverse RL using the latest forward ensemble.
4. Evaluate inverse designer on fixed benchmark target curves.
5. Promote checkpoint only if true Windows FEM benchmark improves.
```

Important:

```text
GraphMetaMat surrogate curve is not a label.
response.stress from Windows FEM is the label.
```

## 7. Agent Policy on Windows

Windows AgentExplorer should choose target schedules using:

```text
prior knowledge
previous evaluated_samples.jsonl
failure modes
coverage gaps
near-miss samples
surrogate uncertainty reported by 3090
human objective
```

Suggested exploration actions:

```text
exploitation:
  sample near the final target

reachability_probe:
  sample targets known to be easy for the current inverse designer

boundary_probe:
  sample sparse or high-error regions

counterfactual:
  perturb one response dimension and test whether the hypothesis holds

curriculum:
  move gradually from reachable targets toward the final target
```

The agent should not directly write training labels. It only proposes targets and decides which evaluated samples are worth adding.

## 8. Checkpoint Promotion

Each finetune round must produce:

```text
checkpoint candidate
training dataset version
fixed benchmark target list
inverse-design outputs for benchmark targets
Windows FEM benchmark evaluations
promotion decision
```

Promote a checkpoint only if:

```text
true FEM target error improves
validity does not collapse
coverage improves or remains acceptable
failure rate does not increase materially
```

Do not promote solely on training loss or surrogate metrics.

## 9. Minimal Round Checklist

```text
Windows:
  [ ] Generate TPMS 11-D control point targets and/or Truss 256-D stress targets.
  [ ] Submit inverse jobs to 3090.
  [ ] Download candidate artifacts.
  [ ] Run real FEM.
  [ ] Write evaluated_samples.jsonl.
  [ ] Upload evaluated samples and accepted artifacts to 3090.

3090:
  [ ] Build active dataset versions.
  [ ] Finetune TPMS / Truss models as applicable.
  [ ] Run fixed benchmark inverse-design jobs.
  [ ] Wait for Windows FEM benchmark labels.
  [ ] Promote or reject checkpoints.
```

One-line summary:

```text
Windows is the planner and physics judge; the 3090 is the neural generator and training factory.
```

