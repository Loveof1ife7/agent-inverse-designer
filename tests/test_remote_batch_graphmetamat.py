from __future__ import annotations

from pathlib import Path

from src.InverseDesigner import RemoteGraphMetaMatClient, RemoteGraphMetaMatInverseDesigner, RemoteInverseDesignerConfig, RemoteJobResult
from src.closed_loop_contracts import TargetSchedule, TargetScheduleItem
from src.curve_targets import fixed_strain_grid


def test_remote_inverse_batch_builds_dry_run_request_and_uploads_targets(tmp_path):
    client = RemoteGraphMetaMatClient(
        RemoteInverseDesignerConfig(
            ssh_alias="dry-host",
            remote_root="/remote/project",
            remote_python="/remote/python",
            local_workspace=tmp_path / "remote_jobs",
            dry_run=True,
        )
    )

    result = client.run_inverse_design_batch(
        [[0.0, 0.1, 0.2], [0.0, 0.2, 0.4]],
        job_id="batch_inverse_test",
        num_runs=64,
        top_k=4,
        batch_size=2,
        download_artifacts=False,
    )

    local_dir = Path(result.local_dir)
    assert (local_dir / "target_curves.npy").exists()
    assert (local_dir / "request.json").exists()
    request_text = (local_dir / "request.json").read_text(encoding="utf-8")
    assert "target_curve_path" in request_text
    assert "batch_inverse_test/target_curves.npy" in request_text
    assert any("target_curves.npy" in " ".join(cmd.command) for cmd in result.commands)
    assert any("tools.run_inverse_design_job" in " ".join(cmd.command) for cmd in result.commands)


def test_remote_forward_batch_builds_graphs_file_and_worker_command(tmp_path):
    graph = tmp_path / "candidate.gpkl"
    graph.write_bytes(b"fake")
    client = RemoteGraphMetaMatClient(
        RemoteInverseDesignerConfig(
            ssh_alias="dry-host",
            remote_root="/remote/project",
            remote_python="/remote/python",
            local_workspace=tmp_path / "remote_jobs",
            dry_run=True,
        )
    )

    result = client.run_truss_forward_predict_batch(
        [graph, "/remote/project/workspace/already_remote.gpkl"],
        job_id="batch_forward_test",
        batch_size=256,
        gpu_workers=5,
        compact=False,
    )

    local_graphs = Path(result.local_dir) / "graphs.txt"
    assert local_graphs.exists()
    text = local_graphs.read_text(encoding="utf-8")
    assert "candidate.gpkl" in text
    assert "/remote/project/workspace/already_remote.gpkl" in text
    command_text = "\n".join(" ".join(cmd.command) for cmd in result.commands)
    assert "--graphs-file" in command_text
    assert "--batch-size 256" in command_text
    assert "--gpu-workers 5" in command_text


class FakeBatchClient:
    def __init__(self, tmp_path: Path) -> None:
        self.config = RemoteInverseDesignerConfig(local_workspace=tmp_path / "fake")
        self.calls = []

    def run_inverse_design_batch(self, curves, **kwargs):
        self.calls.append({"curves": curves, **kwargs})
        return RemoteJobResult(
            job_id=kwargs["job_id"],
            status="success",
            local_dir=str(self.config.local_workspace / kwargs["job_id"]),
            remote_dir="/remote/inverse/fake",
            response={
                "status": "success",
                "candidates": [
                    {
                        "target_index": 0,
                        "structure_id": "candidate_0",
                        "coordinates": [[0, 0, 0], [1, 0, 0]],
                        "edges": [[0, 1]],
                        "artifacts": {"gpkl": "/remote/inverse/fake/inverse_designer/c0.gpkl"},
                    },
                    {
                        "target_index": 1,
                        "structure_id": "candidate_1",
                        "coordinates": [[0, 0, 0], [0, 1, 0]],
                        "edges": [[0, 1]],
                        "artifacts": {"gpkl": "/remote/inverse/fake/inverse_designer/c1.gpkl"},
                    },
                ],
            },
        )


def _target(scale: float) -> dict:
    return {
        "type": "stress_curve",
        "strain_grid": fixed_strain_grid(8),
        "stress": [scale * index for index in range(8)],
    }


def test_remote_adapter_sample_schedule_uses_batch_candidates(tmp_path):
    client = FakeBatchClient(tmp_path)
    adapter = RemoteGraphMetaMatInverseDesigner(
        client=client,
        workspace_root=tmp_path,
        num_runs=64,
        top_k=2,
        batch_size=2,
    )
    schedule = TargetSchedule(
        schedule_id="batch_schedule",
        final_target=_target(1.0),
        scheduled_targets=[
            TargetScheduleItem(target_id="t0", target_property=_target(1.0)),
            TargetScheduleItem(target_id="t1", target_property=_target(0.9)),
        ],
    )

    records = adapter.sample_schedule(schedule)

    assert len(client.calls) == 1
    assert len(client.calls[0]["curves"]) == 2
    assert len(records) == 2
    assert records[0]["structure"]["structure_id"] == "candidate_0"
    assert records[1]["structure"]["structure_id"] == "candidate_1"
    assert records[0]["structure"]["artifacts"]["remote"]["gpkl"].endswith("c0.gpkl")
