from __future__ import annotations

import json
import os
import posixpath
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _safe_name(value: str, default: str = "job") -> str:
    text = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))
    return text[:160] or default


def _write_json_no_bom(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _remote_join(*parts: str) -> str:
    cleaned = [str(part).strip("/") for part in parts if str(part)]
    if not cleaned:
        return ""
    prefix = "/" if str(parts[0]).startswith("/") else ""
    return prefix + posixpath.join(*cleaned)


def _quote_remote(path_or_arg: str) -> str:
    return shlex.quote(str(path_or_arg))


def default_truss_finetune_config(
    round_id: str,
    *,
    remote_python: str = "/root/miniconda3/bin/python",
    min_accepted: int = 1,
    enable_forward: bool = False,
    enable_inverse_rl: bool = False,
    enable_inverse_il: bool = False,
) -> dict[str, Any]:
    """Return a valid ``truss_finetune_v1`` config.

    Stages are explicit but disabled by default. This makes build-only dataset
    projection safe, while still giving callers a concrete schema to enable
    training stages intentionally.
    """

    return {
        "schema_version": "truss_finetune_v1",
        "round_id": str(round_id),
        "structure_family": "truss",
        "curve": {
            "task": "compression_stress_strain",
            "length": 256,
            "strain_min": 0.0,
            "strain_max": 0.3,
            "label_source": "windows_fem",
        },
        "data": {
            "min_accepted": int(min_accepted),
            "require_no_rejected": False,
            "require_polyhedron_for_inverse_il": False,
        },
        "stages": [
            {
                "name": f"forward_finetune_{round_id}",
                "kind": "forward_finetune",
                "enabled": bool(enable_forward),
                "cwd": "third-party/GraphMetaMat",
                "command": [remote_python, "main_forward.py"],
                "timeout_seconds": 86400,
            },
            {
                "name": f"inverse_rl_{round_id}",
                "kind": "inverse_rl",
                "enabled": bool(enable_inverse_rl),
                "cwd": "third-party/GraphMetaMat",
                "command": [remote_python, "main_inverse.py"],
                "timeout_seconds": 86400,
            },
            {
                "name": f"inverse_il_{round_id}",
                "kind": "inverse_il",
                "enabled": bool(enable_inverse_il),
                "cwd": "third-party/GraphMetaMat",
                "command": [remote_python, "main_inverse.py"],
                "timeout_seconds": 86400,
                "requires_polyhedron": True,
                "on_missing_polyhedron": "skip",
            },
        ],
    }


@dataclass
class RemoteInverseDesignerConfig:
    """Windows SSH configuration for the remote GraphMetaMat runners.

    The defaults match the tested 3090/base-conda setup:

    - SSH host: ``data-center-ps``
    - project root: ``/root/autodl-tmp/projects/agent-material``
    - Python: ``/root/miniconda3/bin/python``

    Important launch rule:

    - inverse runner uses ``python -m tools.run_inverse_design_job`` so the
      project root is importable without manually setting ``PYTHONPATH``.
    - forward runner uses ``python tools/run_truss_forward_predict.py`` because
      running it as a module makes the repo-level ``src`` package shadow
      GraphMetaMat's own ``src`` package.
    """

    ssh_alias: str = field(default_factory=lambda: os.getenv("REMOTE_INVERSE_SSH_ALIAS", "data-center-ps"))
    remote_root: str = field(
        default_factory=lambda: os.getenv("REMOTE_INVERSE_ROOT", "/root/autodl-tmp/projects/agent-material")
    )
    remote_python: str = field(default_factory=lambda: os.getenv("REMOTE_INVERSE_PYTHON", "/root/miniconda3/bin/python"))
    local_workspace: Path = field(default_factory=lambda: Path(os.getenv("REMOTE_INVERSE_LOCAL_WORKSPACE", "remote_inverse_jobs")))
    inverse_job_root: str = "workspace/remote_inverse_jobs"
    forward_job_root: str = "workspace/remote_forward_jobs"
    active_learning_root: str = "workspace/truss_active_learning"
    ssh_options: tuple[str, ...] = ()
    timeout_seconds: int | None = None
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["local_workspace"] = str(self.local_workspace)
        return payload


@dataclass
class RemoteCommandResult:
    command: list[str]
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RemoteJobResult:
    job_id: str
    status: str
    local_dir: str
    remote_dir: str
    response_path: str = ""
    response: dict[str, Any] = field(default_factory=dict)
    commands: list[RemoteCommandResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "local_dir": self.local_dir,
            "remote_dir": self.remote_dir,
            "response_path": self.response_path,
            "response": self.response,
            "commands": [item.to_dict() for item in self.commands],
        }


class RemoteGraphMetaMatClient:
    """Windows-side client for remote truss inverse, forward, and finetune jobs."""

    def __init__(self, config: RemoteInverseDesignerConfig | dict[str, Any] | None = None):
        if isinstance(config, RemoteInverseDesignerConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = RemoteInverseDesignerConfig(**config)
        else:
            self.config = RemoteInverseDesignerConfig()
        self.commands: list[RemoteCommandResult] = []

    def run_inverse_design(
        self,
        request: dict[str, Any],
        *,
        job_id: str | None = None,
        download_artifacts: bool = True,
    ) -> RemoteJobResult:
        payload = dict(request)
        resolved_job_id = _safe_name(job_id or str(payload.get("job_id") or "inverse_job"))
        payload["job_id"] = resolved_job_id
        payload.setdefault("structure_family", "truss")
        payload.setdefault("options", {})
        payload["options"].setdefault("project_dir", "third-party/GraphMetaMat")

        local_dir = self._local_job_dir(resolved_job_id)
        remote_dir = self._remote_inverse_dir(resolved_job_id)
        local_request = local_dir / "request.json"
        local_response = local_dir / "response.json"
        _write_json_no_bom(local_request, payload)

        self._ssh(f"mkdir -p {_quote_remote(remote_dir)}")
        self._scp(local_request, f"{self.config.ssh_alias}:{remote_dir}/request.json")
        self._ssh(
            "cd {root} && env -u PYTHONPATH {python} -m tools.run_inverse_design_job "
            "--request {request} --output {output}".format(
                root=_quote_remote(self.config.remote_root),
                python=_quote_remote(self.config.remote_python),
                request=_quote_remote(f"{self.config.inverse_job_root}/{resolved_job_id}/request.json"),
                output=_quote_remote(f"{self.config.inverse_job_root}/{resolved_job_id}/response.json"),
            )
        )
        self._scp(f"{self.config.ssh_alias}:{remote_dir}/response.json", local_response)
        if download_artifacts:
            self._scp(f"{self.config.ssh_alias}:{remote_dir}/inverse_designer", local_dir, recursive=True)

        response = _read_json(local_response) if local_response.exists() else {}
        return RemoteJobResult(
            job_id=resolved_job_id,
            status=str(response.get("status") or "unknown"),
            local_dir=str(local_dir),
            remote_dir=remote_dir,
            response_path=str(local_response),
            response=response,
            commands=list(self.commands),
        )

    def run_inverse_design_batch(
        self,
        target_curves: list[list[float]],
        *,
        job_id: str,
        num_runs: int = 64,
        top_k: int | None = None,
        batch_size: int | None = None,
        device: str = "cuda",
        timeout_seconds: int = 7200,
        download_artifacts: bool = True,
    ) -> RemoteJobResult:
        resolved_job_id = _safe_name(job_id or "inverse_batch")
        local_dir = self._local_job_dir(resolved_job_id)
        remote_dir = self._remote_inverse_dir(resolved_job_id)
        local_targets = local_dir / "target_curves.npy"
        remote_targets = f"{remote_dir}/target_curves.npy"
        curves = np.asarray(target_curves, dtype="float32")
        if curves.ndim != 2 or curves.shape[0] <= 0:
            raise ValueError("target_curves must have shape (N, L)")
        np.save(local_targets, curves)
        request = {
            "job_id": resolved_job_id,
            "structure_family": "truss",
            "target_property": {
                "type": "stress_curve",
                "target_curve_path": remote_targets,
            },
            "options": {
                "project_dir": "third-party/GraphMetaMat",
                "num_runs": int(num_runs),
                "top_k": int(top_k or curves.shape[0]),
                "device": device,
                "batch_size": int(batch_size or curves.shape[0]),
                "timeout_seconds": int(timeout_seconds),
            },
            "workspace_root": f"{self.config.inverse_job_root}/{resolved_job_id}",
        }
        self._ssh(f"mkdir -p {_quote_remote(remote_dir)}")
        self._scp(local_targets, f"{self.config.ssh_alias}:{remote_targets}")
        return self.run_inverse_design(
            request,
            job_id=resolved_job_id,
            download_artifacts=download_artifacts,
        )

    def run_truss_forward_predict(
        self,
        graph_path: str | os.PathLike[str],
        *,
        job_id: str,
        device: str = "cuda",
        download_response: bool = True,
    ) -> RemoteJobResult:
        resolved_job_id = _safe_name(job_id)
        local_dir = self._local_job_dir(resolved_job_id) / "forward_surrogate"
        local_dir.mkdir(parents=True, exist_ok=True)
        remote_dir = self._remote_forward_dir(resolved_job_id)
        remote_graph = self._remote_graph_path(graph_path, remote_dir)
        remote_output = f"{remote_dir}/forward_response.json"
        local_response = local_dir / "forward_response.json"

        self._ssh(f"mkdir -p {_quote_remote(remote_dir)}")
        graph_local = Path(graph_path)
        if graph_local.exists():
            self._scp(graph_local, f"{self.config.ssh_alias}:{remote_graph}")

        self._ssh(
            "cd {root} && env -u PYTHONPATH {python} tools/run_truss_forward_predict.py "
            "--graph {graph} --out-dir {out_dir} --output {output} --device {device}".format(
                root=_quote_remote(self.config.remote_root),
                python=_quote_remote(self.config.remote_python),
                graph=_quote_remote(remote_graph),
                out_dir=_quote_remote(f"{self.config.forward_job_root}/{resolved_job_id}"),
                output=_quote_remote(f"{self.config.forward_job_root}/{resolved_job_id}/forward_response.json"),
                device=_quote_remote(device),
            )
        )
        if download_response:
            self._scp(f"{self.config.ssh_alias}:{remote_output}", local_response)

        response = _read_json(local_response) if local_response.exists() else {}
        return RemoteJobResult(
            job_id=resolved_job_id,
            status=str(response.get("status") or "unknown"),
            local_dir=str(local_dir),
            remote_dir=remote_dir,
            response_path=str(local_response),
            response=response,
            commands=list(self.commands),
        )

    def run_truss_forward_predict_batch(
        self,
        graph_paths: list[str | os.PathLike[str]],
        *,
        job_id: str,
        device: str = "cuda",
        batch_size: int = 256,
        gpu_workers: int = 5,
        compact: bool = False,
        quiet: bool = True,
        download_response: bool = True,
    ) -> RemoteJobResult:
        resolved_job_id = _safe_name(job_id)
        local_dir = self._local_job_dir(resolved_job_id) / "forward_surrogate"
        local_dir.mkdir(parents=True, exist_ok=True)
        remote_dir = self._remote_forward_dir(resolved_job_id)
        local_graphs = local_dir / "graphs.txt"
        remote_graphs = f"{remote_dir}/graphs.txt"
        remote_output = f"{remote_dir}/forward_response.json"
        local_response = local_dir / "forward_response.json"

        self._ssh(f"mkdir -p {_quote_remote(remote_dir)}")
        remote_graph_paths: list[str] = []
        for index, graph_path in enumerate(graph_paths):
            local_graph = Path(graph_path)
            if local_graph.exists():
                remote_graph = f"{remote_dir}/graphs/{index:05d}_{local_graph.name}"
                self._ssh(f"mkdir -p {_quote_remote(f'{remote_dir}/graphs')}")
                self._scp(local_graph, f"{self.config.ssh_alias}:{remote_graph}")
                remote_graph_paths.append(remote_graph)
            else:
                remote_graph_paths.append(str(graph_path).replace("\\", "/"))
        local_graphs.write_text("\n".join(remote_graph_paths) + "\n", encoding="utf-8")
        self._scp(local_graphs, f"{self.config.ssh_alias}:{remote_graphs}")

        flags = []
        if compact:
            flags.append("--compact")
        if quiet:
            flags.append("--quiet")
        self._ssh(
            "cd {root} && env -u PYTHONPATH {python} tools/run_truss_forward_predict.py "
            "--graphs-file {graphs_file} --out-dir {out_dir} --output {output} "
            "--device {device} --batch-size {batch_size} --gpu-workers {gpu_workers} {flags}".format(
                root=_quote_remote(self.config.remote_root),
                python=_quote_remote(self.config.remote_python),
                graphs_file=_quote_remote(f"{self.config.forward_job_root}/{resolved_job_id}/graphs.txt"),
                out_dir=_quote_remote(f"{self.config.forward_job_root}/{resolved_job_id}"),
                output=_quote_remote(f"{self.config.forward_job_root}/{resolved_job_id}/forward_response.json"),
                device=_quote_remote(device),
                batch_size=int(batch_size),
                gpu_workers=int(gpu_workers),
                flags=" ".join(flags),
            )
        )
        if download_response:
            self._scp(f"{self.config.ssh_alias}:{remote_output}", local_response)

        response = _read_json(local_response) if local_response.exists() else {}
        return RemoteJobResult(
            job_id=resolved_job_id,
            status=str(response.get("status") or "unknown"),
            local_dir=str(local_dir),
            remote_dir=remote_dir,
            response_path=str(local_response),
            response=response,
            commands=list(self.commands),
        )

    def submit_truss_finetune(
        self,
        *,
        round_id: str,
        evaluated_samples_path: str | os.PathLike[str],
        candidates_dir: str | os.PathLike[str] | None = None,
        fem_runs_dir: str | os.PathLike[str] | None = None,
        finetune_config: dict[str, Any] | None = None,
        finetune_config_path: str | os.PathLike[str] | None = None,
        run_training: bool = False,
        graphmetamat_project_dir: str = "third-party/GraphMetaMat",
        dataset_output_dir: str | None = None,
        min_accepted: int = 1,
        options: dict[str, Any] | None = None,
        runner: str = "tools/run_truss_finetune_job.py",
        run_remote: bool = True,
        download_response: bool = True,
    ) -> RemoteJobResult:
        """Upload Windows FEM truth and optionally start a remote finetune job.

        The upload location is fixed to:

        ``workspace/truss_active_learning/<round_id>/windows_eval``.

        The remote request follows the truss-specific runner contract:

        ``workspace/truss_active_learning/<round_id>/request.json``

        and the stage config is uploaded to:

        ``workspace/truss_active_learning/<round_id>/finetune_config.json``.

        ``run_remote=False`` is useful for upload-only sync. When
        ``run_remote=True``, the remote runner is expected to accept:

        ``--request <request.json> --output <response.json>``.
        """

        resolved_round = _safe_name(round_id, default="round")
        local_round_dir = self.config.local_workspace / resolved_round
        local_round_dir.mkdir(parents=True, exist_ok=True)
        remote_round_relative = f"{self.config.active_learning_root}/{resolved_round}"
        remote_round_dir = _remote_join(self.config.remote_root, remote_round_relative)
        remote_eval_relative = f"{remote_round_relative}/windows_eval"
        remote_eval_dir = _remote_join(self.config.remote_root, remote_eval_relative)
        remote_request_relative = f"{remote_round_relative}/request.json"
        remote_config_relative = f"{remote_round_relative}/finetune_config.json"
        remote_response_relative = f"{remote_round_relative}/finetune_response.json"
        remote_request = _remote_join(self.config.remote_root, remote_request_relative)
        remote_config = _remote_join(self.config.remote_root, remote_config_relative)
        remote_response = _remote_join(self.config.remote_root, remote_response_relative)
        local_request = local_round_dir / "request.json"
        local_config = local_round_dir / "finetune_config.json"
        local_response = local_round_dir / "finetune_response.json"

        if finetune_config is not None and finetune_config_path is not None:
            raise ValueError("Pass either finetune_config or finetune_config_path, not both.")
        if finetune_config is None:
            if finetune_config_path is not None:
                finetune_config = _read_json(finetune_config_path)
            else:
                finetune_config = default_truss_finetune_config(
                    resolved_round,
                    remote_python=self.config.remote_python,
                    min_accepted=min_accepted,
                )
        finetune_config = dict(finetune_config)
        finetune_config.setdefault("round_id", resolved_round)
        _write_json_no_bom(local_config, finetune_config)

        dataset_output = dataset_output_dir or f"{remote_round_relative}/dataset_active_{resolved_round}"
        request = {
            "round_id": resolved_round,
            "workspace_root": remote_round_relative,
            "windows_eval_dir": remote_eval_relative,
            "dataset_output_dir": dataset_output,
            "graphmetamat_project_dir": graphmetamat_project_dir,
            "finetune_config_path": remote_config_relative,
            "run_training": bool(run_training),
        }
        request.update(dict(options or {}))
        _write_json_no_bom(local_request, request)

        self._ssh(f"mkdir -p {_quote_remote(remote_eval_dir)}")
        self._scp(evaluated_samples_path, f"{self.config.ssh_alias}:{remote_eval_dir}/evaluated_samples.jsonl")
        if candidates_dir:
            self._scp(candidates_dir, f"{self.config.ssh_alias}:{remote_eval_dir}/candidates", recursive=True)
        if fem_runs_dir:
            self._scp(fem_runs_dir, f"{self.config.ssh_alias}:{remote_eval_dir}/fem_runs", recursive=True)
        self._scp(local_request, f"{self.config.ssh_alias}:{remote_request}")
        self._scp(local_config, f"{self.config.ssh_alias}:{remote_config}")

        if run_remote:
            self._ssh(
                "cd {root} && if test -f {runner}; then "
                "env -u PYTHONPATH {python} {runner} --request {request} --output {output}; "
                "else echo 'Missing remote finetune runner: {runner_raw}' >&2; exit 127; fi".format(
                    root=_quote_remote(self.config.remote_root),
                    runner=_quote_remote(runner),
                    runner_raw=runner,
                    python=_quote_remote(self.config.remote_python),
                    request=_quote_remote(remote_request_relative),
                    output=_quote_remote(remote_response_relative),
                )
            )
            if download_response:
                self._scp(f"{self.config.ssh_alias}:{remote_response}", local_response)

        response = _read_json(local_response) if local_response.exists() else {"status": "uploaded"}
        return RemoteJobResult(
            job_id=f"{resolved_round}_truss_finetune",
            status=str(response.get("status") or "uploaded"),
            local_dir=str(local_round_dir),
            remote_dir=remote_round_dir,
            response_path=str(local_response) if local_response.exists() else "",
            response=response,
            commands=list(self.commands),
        )

    def _local_job_dir(self, job_id: str) -> Path:
        path = self.config.local_workspace / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _remote_inverse_dir(self, job_id: str) -> str:
        return _remote_join(self.config.remote_root, self.config.inverse_job_root, job_id)

    def _remote_forward_dir(self, job_id: str) -> str:
        return _remote_join(self.config.remote_root, self.config.forward_job_root, job_id)

    def _remote_graph_path(self, graph_path: str | os.PathLike[str], remote_dir: str) -> str:
        local = Path(graph_path)
        if local.exists():
            return f"{remote_dir}/{local.name}"
        return str(graph_path).replace("\\", "/")

    def _ssh(self, remote_command: str) -> RemoteCommandResult:
        return self._run(["ssh", *self.config.ssh_options, self.config.ssh_alias, remote_command])

    def _scp(self, source: str | os.PathLike[str], destination: str | os.PathLike[str], recursive: bool = False) -> RemoteCommandResult:
        command = ["scp", *self.config.ssh_options]
        if recursive:
            command.append("-r")
        command.extend([str(source), str(destination)])
        return self._run(command)

    def _run(self, command: list[str]) -> RemoteCommandResult:
        if self.config.dry_run:
            result = RemoteCommandResult(command=list(command), skipped=True)
            self.commands.append(result)
            return result
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=self.config.timeout_seconds,
        )
        result = RemoteCommandResult(
            command=list(command),
            returncode=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        self.commands.append(result)
        if completed.returncode != 0:
            raise RuntimeError(
                "Remote command failed with exit code {code}: {cmd}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".format(
                    code=completed.returncode,
                    cmd=" ".join(command),
                    stdout=completed.stdout.strip(),
                    stderr=completed.stderr.strip(),
                )
            )
        return result


def _remote_project_relative(remote_root: str, path: str) -> str:
    root = remote_root.rstrip("/") + "/"
    if path.startswith(root):
        return path[len(root) :]
    return path


__all__ = [
    "default_truss_finetune_config",
    "RemoteCommandResult",
    "RemoteGraphMetaMatClient",
    "RemoteInverseDesignerConfig",
    "RemoteJobResult",
]
