from __future__ import annotations

import json
from pathlib import Path

from src.InverseDesigner import RemoteGraphMetaMatClient, RemoteInverseDesignerConfig, default_truss_finetune_config


def _client(tmp_path: Path) -> RemoteGraphMetaMatClient:
    return RemoteGraphMetaMatClient(
        RemoteInverseDesignerConfig(
            ssh_alias="remote-test",
            remote_root="/remote/project",
            remote_python="/remote/miniconda3/bin/python",
            local_workspace=tmp_path / "remote_jobs",
            dry_run=True,
        )
    )


def test_remote_inverse_uses_module_entrypoint_and_no_bom_json(tmp_path):
    client = _client(tmp_path)

    result = client.run_inverse_design(
        {
            "job_id": "round001_truss_target001",
            "structure_family": "truss",
            "target": {"type": "stress_curve", "stress": [0.0, 1.0]},
            "options": {"num_runs": 1, "top_k": 1, "device": "cuda"},
        },
        download_artifacts=False,
    )

    request_path = Path(result.local_dir) / "request.json"
    raw = request_path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    payload = json.loads(raw.decode("utf-8"))
    assert payload["job_id"] == "round001_truss_target001"

    ssh_commands = [" ".join(item.command) for item in result.commands if item.command and item.command[0] == "ssh"]
    assert any("env -u PYTHONPATH /remote/miniconda3/bin/python -m tools.run_inverse_design_job" in cmd for cmd in ssh_commands)


def test_remote_forward_uses_script_entrypoint_not_module(tmp_path):
    client = _client(tmp_path)

    result = client.run_truss_forward_predict(
        "/remote/project/workspace/remote_inverse_jobs/job/design_exports/rank_01_sample_000.gpkl",
        job_id="round001_truss_target001",
    )

    ssh_commands = [" ".join(item.command) for item in result.commands if item.command and item.command[0] == "ssh"]
    assert any("env -u PYTHONPATH /remote/miniconda3/bin/python tools/run_truss_forward_predict.py" in cmd for cmd in ssh_commands)
    assert not any("-m tools.run_truss_forward_predict" in cmd for cmd in ssh_commands)


def test_remote_finetune_upload_can_run_without_remote_runner(tmp_path):
    evaluated = tmp_path / "evaluated_samples.jsonl"
    evaluated.write_text('{"candidate_id":"c1"}\n', encoding="utf-8")
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / "c1.gpkl").write_bytes(b"graph")
    fem_runs = tmp_path / "fem_runs"
    fem_runs.mkdir()
    (fem_runs / "data.csv").write_text("Strain,Stress\n", encoding="utf-8")
    client = _client(tmp_path)

    result = client.submit_truss_finetune(
        round_id="round001",
        evaluated_samples_path=evaluated,
        candidates_dir=candidates,
        fem_runs_dir=fem_runs,
        run_remote=False,
    )

    assert result.status == "uploaded"
    request_path = Path(result.local_dir) / "request.json"
    config_path = Path(result.local_dir) / "finetune_config.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert request["workspace_root"] == "workspace/truss_active_learning/round001"
    assert request["windows_eval_dir"] == "workspace/truss_active_learning/round001/windows_eval"
    assert request["dataset_output_dir"] == "workspace/truss_active_learning/round001/dataset_active_round001"
    assert request["finetune_config_path"] == "workspace/truss_active_learning/round001/finetune_config.json"
    assert request["run_training"] is False
    assert config["schema_version"] == "truss_finetune_v1"
    assert config["curve"]["length"] == 256
    assert config["curve"]["label_source"] == "windows_fem"
    scp_commands = [" ".join(item.command) for item in result.commands if item.command and item.command[0] == "scp"]
    assert any("evaluated_samples.jsonl" in cmd for cmd in scp_commands)
    assert any("finetune_config.json" in cmd for cmd in scp_commands)
    assert any("-r" in cmd and "candidates" in cmd for cmd in scp_commands)
    assert any("-r" in cmd and "fem_runs" in cmd for cmd in scp_commands)


def test_remote_finetune_training_uses_new_runner_request_path(tmp_path):
    evaluated = tmp_path / "evaluated_samples.jsonl"
    evaluated.write_text('{"candidate_id":"c1"}\n', encoding="utf-8")
    config = default_truss_finetune_config("round002", enable_forward=True)
    client = _client(tmp_path)

    result = client.submit_truss_finetune(
        round_id="round002",
        evaluated_samples_path=evaluated,
        finetune_config=config,
        run_training=True,
        run_remote=True,
    )

    request = json.loads((Path(result.local_dir) / "request.json").read_text(encoding="utf-8"))
    assert request["run_training"] is True
    ssh_commands = [" ".join(item.command) for item in result.commands if item.command and item.command[0] == "ssh"]
    assert any("tools/run_truss_finetune_job.py --request workspace/truss_active_learning/round002/request.json --output workspace/truss_active_learning/round002/finetune_response.json" in cmd for cmd in ssh_commands)
