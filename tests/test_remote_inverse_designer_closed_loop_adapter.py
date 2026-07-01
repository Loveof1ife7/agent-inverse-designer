from __future__ import annotations

import json
from pathlib import Path

from src.InverseDesigner import RemoteGraphMetaMatInverseDesigner, RemoteJobResult
from src.InverseDesigner.remote import RemoteInverseDesignerConfig
from src.DatagenFEMEvaluator import DatagenFEMEvaluator
from src.closed_loop_contracts import TargetSchedule, TargetScheduleItem


class FakeRemoteClient:
    def __init__(self, tmp_path: Path):
        self.config = RemoteInverseDesignerConfig(
            ssh_alias="remote-test",
            remote_root="/remote/project",
            remote_python="/remote/python",
            local_workspace=tmp_path / "remote_jobs",
            dry_run=True,
        )
        self.inverse_requests = []
        self.finetune_requests = []

    def run_inverse_design(self, request, *, job_id=None, download_artifacts=True):
        self.inverse_requests.append(
            {
                "request": request,
                "job_id": job_id,
                "download_artifacts": download_artifacts,
            }
        )
        local_dir = self.config.local_workspace / str(job_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        return RemoteJobResult(
            job_id=str(job_id),
            status="success",
            local_dir=str(local_dir),
            remote_dir=f"/remote/project/workspace/remote_inverse_jobs/{job_id}",
            response_path=str(local_dir / "response.json"),
            response={
                "status": "success",
                "candidate": {
                    "structure_id": "remote_candidate_001",
                    "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    "edges": [[0, 1]],
                    "artifacts": {
                        "gpkl": f"/remote/project/workspace/remote_inverse_jobs/{job_id}/inverse_designer/design_exports/rank_01.gpkl"
                    },
                },
            },
        )

    def submit_truss_finetune(self, **kwargs):
        self.finetune_requests.append(kwargs)
        output = Path(kwargs["evaluated_samples_path"])
        return RemoteJobResult(
            job_id="round001_truss_finetune",
            status="uploaded",
            local_dir=str(output.parent),
            remote_dir="/remote/project/workspace/truss_active_learning/round001",
        )


def test_remote_adapter_samples_closed_loop_stress_curve_target(tmp_path):
    client = FakeRemoteClient(tmp_path)
    adapter = RemoteGraphMetaMatInverseDesigner(client=client, workspace_root=tmp_path / "workspace")
    target_property = {
        "stress_curve": [0.0, 0.1, 0.2],
        "strain_grid": [0.0, 0.15, 0.3],
    }

    structure = adapter.sample_structure(target_property)

    assert structure["structure_id"] == "remote_candidate_001"
    assert structure["source"].startswith("inverse_designer_remote")
    assert structure["remote_requested_target"]["type"] == "stress_curve"
    assert structure["artifacts"]["gpkl"].endswith("rank_01.gpkl")
    request = client.inverse_requests[0]["request"]
    assert request["structure_family"] == "truss"
    assert len(request["target"]["strain_grid"]) == 256
    assert len(request["target"]["stress"]) == 256
    assert request["target"]["strain_grid"][0] == 0.0
    assert request["target"]["strain_grid"][-1] == 0.3
    assert request["target"]["stress"][0] == 0.0
    assert request["target"]["stress"][-1] == 0.2


def test_remote_adapter_sample_schedule_preserves_schedule_record(tmp_path):
    client = FakeRemoteClient(tmp_path)
    adapter = RemoteGraphMetaMatInverseDesigner(client=client, workspace_root=tmp_path / "workspace")
    schedule = TargetSchedule(
        schedule_id="schedule001",
        final_target={"stress_curve": [0.0, 0.1], "strain_grid": [0.0, 0.3]},
        scheduled_targets=[
            TargetScheduleItem(
                target_id="target001",
                target_property={"stress_curve": [0.0, 0.1], "strain_grid": [0.0, 0.3]},
                expected_effect={"structure_family": "truss"},
            )
        ],
    )

    records = adapter.sample_schedule(schedule)

    assert len(records) == 1
    assert records[0]["status"] == "sampled"
    assert records[0]["schedule_id"] == "schedule001"
    assert records[0]["structure"]["structure_id"] == "remote_candidate_001"


def test_remote_adapter_finetune_uploads_windows_fem_curve_as_256_point_truth(tmp_path):
    curve_path = tmp_path / "fem_curve.csv"
    curve_path.write_text("Strain,Stress\n0.0,0.0\n0.3,0.6\n", encoding="utf-8")
    gpkl_path = tmp_path / "candidate.gpkl"
    gpkl_path.write_bytes(b"graph")
    client = FakeRemoteClient(tmp_path)
    adapter = RemoteGraphMetaMatInverseDesigner(client=client, workspace_root=tmp_path / "workspace")

    adapter.finetune(
        [
            {
                "sample_id": "obs001",
                "property": {"stiffness_proxy": 1.0},
                "explicit_structure": {
                    "structure_id": "remote_candidate_001",
                    "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    "edges": [[0, 1]],
                    "artifacts": {"gpkl": str(gpkl_path)},
                },
                "validity": {"geometry_status": "valid", "fem_status": "success"},
                "raw_metrics": {"fem_curve_path": str(curve_path)},
            }
        ]
    )

    kwargs = client.finetune_requests[0]
    evaluated_path = Path(kwargs["evaluated_samples_path"])
    rows = [json.loads(line) for line in evaluated_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert len(rows[0]["response"]["strain_grid"]) == 256
    assert len(rows[0]["response"]["stress"]) == 256
    assert rows[0]["response"]["strain_grid"][0] == 0.0
    assert rows[0]["response"]["strain_grid"][-1] == 0.3
    assert rows[0]["response"]["stress"][0] == 0.0
    assert rows[0]["response"]["stress"][-1] == 0.6
    assert kwargs["candidates_dir"].exists()
    assert (kwargs["candidates_dir"] / "candidate.gpkl").exists()


def test_fem_evaluator_accepts_curve_target_without_scalar_error_crash(tmp_path):
    evaluator = DatagenFEMEvaluator(workspace_root=tmp_path / "workspace", fem_backend="proxy")
    result = evaluator.evaluate_explicit_structure(
        {
            "structure_id": "explicit_curve_target",
            "coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "edges": [[0, 1]],
        },
        {"stress_curve": [0.0, 0.1], "strain_grid": [0.0, 0.3]},
    )

    assert result["fem_status"] == "success"
    assert result["property_error"] == {}
    assert result["label"] == "failure"
