from __future__ import annotations

import json
from pathlib import Path
import logging
import os
import subprocess
import sys
from types import SimpleNamespace

from common.io_utils import read_json, write_json
from scripts.worker import run_queue_loop_sftp


def test_parse_args_uses_default_ssh_files(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_queue_loop_sftp.py"])
    args = run_queue_loop_sftp.parse_args()
    assert args.ssh_key == run_queue_loop_sftp.DEFAULT_SSH_KEY
    assert args.known_hosts == run_queue_loop_sftp.DEFAULT_KNOWN_HOSTS


class _FakeSFTP:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = {"/"}

    def _normalize(self, path: str) -> str:
        if path != "/" and path.endswith("/"):
            return path.rstrip("/")
        return path

    def stat(self, remote_path: str):
        path = self._normalize(remote_path)
        if path not in self.dirs and path not in self.files:
            raise FileNotFoundError(path)
        return True

    def mkdir(self, remote_path: str):
        self.dirs.add(self._normalize(remote_path))

    def listdir(self, remote_path: str):
        parent = self._normalize(remote_path)
        prefix = f"{parent}/" if parent != "/" else "/"
        names = []
        for path in self.files:
            if not path.startswith(prefix):
                continue
            remainder = path[len(prefix) :]
            if "/" not in remainder:
                names.append(remainder)
        return names

    def rename(self, source: str, destination: str):
        source = self._normalize(source)
        destination = self._normalize(destination)
        if source not in self.files:
            raise FileNotFoundError(source)
        self.files[destination] = self.files.pop(source)

    def get(self, remote_path: str, local_path: str):
        data = self.files[self._normalize(remote_path)]
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def put(self, local_path: str, remote_path: str):
        self.files[self._normalize(remote_path)] = Path(local_path).read_bytes()

    def close(self):
        return None


class _FakeSSHClient:
    def __init__(self, sftp: _FakeSFTP):
        self.sftp = sftp
        self.connected = False

    def load_host_keys(self, _path: str):
        return None

    def set_missing_host_key_policy(self, _policy):
        return None

    def connect(self, **_kwargs):
        self.connected = True

    def open_sftp(self):
        return self.sftp

    def close(self):
        self.connected = False


def _install_fake_paramiko(monkeypatch, sftp: _FakeSFTP) -> None:
    fake_paramiko = SimpleNamespace(
        SSHClient=lambda: _FakeSSHClient(sftp),
        RejectPolicy=type("RejectPolicy", (), {}),
    )
    monkeypatch.setitem(sys.modules, "paramiko", fake_paramiko)


def _ctx(tmp_path: Path, monkeypatch, sftp: _FakeSFTP) -> run_queue_loop_sftp.LoopContext:
    queue_root = tmp_path / "queue"
    logs_dir = tmp_path / "logs"
    ssh_key = tmp_path / "id_ed25519"
    known_hosts = tmp_path / "known_hosts"
    ssh_key.write_text("dummy", encoding="utf-8")
    known_hosts.write_text("dummy", encoding="utf-8")
    for subdir in ("pending", "running", "done", "failed", "results"):
        (queue_root / subdir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        run_queue_loop_sftp,
        "load_worker_config",
        lambda _path: SimpleNamespace(worker_name="test_worker"),
    )
    _install_fake_paramiko(monkeypatch, sftp)
    logger = logging.getLogger(f"test_sftp_loop_{tmp_path.name}")
    logger.handlers = [logging.NullHandler()]
    return run_queue_loop_sftp.LoopContext(
        worker_name="test_worker",
        worker_config_path=tmp_path / "worker.yaml",
        queue_root=queue_root,
        logs_dir=logs_dir,
        poll_interval_sec=1,
        job_timeout_sec=10,
        stale_running_sec=1,
        sftp=run_queue_loop_sftp.SftpSettings(
            server_host="10.15.89.240",
            server_port=22031,
            username="zhanghf",
            remote_queue_root="/remote/queue",
            ssh_key=ssh_key,
            known_hosts=known_hosts,
        ),
        logger=logger,
    )


def _remote_job_payload(job_id: str) -> dict:
    return {
        "job_id": job_id,
        "template_name": "si_template_v1",
        "variables": {"Din": "55um"},
        "output_filename": f"{job_id}.s2p",
        "submit_time": "2026-04-22T08:00:00Z",
        "metrics_to_extract": ["max_s11_db"],
    }


def _seed_remote_pending(sftp: _FakeSFTP, job_id: str) -> None:
    sftp.files[f"/remote/queue/pending/{job_id}.json"] = (
        json.dumps(_remote_job_payload(job_id)).encode("utf-8")
    )


def test_claim_one_remote_job_renames_and_downloads(monkeypatch, tmp_path: Path) -> None:
    sftp = _FakeSFTP()
    _seed_remote_pending(sftp, "job_001")
    ctx = _ctx(tmp_path, monkeypatch, sftp)

    claimed = run_queue_loop_sftp.claim_one_remote_job(ctx)

    assert claimed is not None
    assert claimed.local_job_path == ctx.queue_root / "running" / "job_001.json"
    assert "/remote/queue/pending/job_001.json" not in sftp.files
    assert "/remote/queue/running/job_001.json" in sftp.files
    assert read_json(claimed.local_job_path)["job_id"] == "job_001"


def test_success_path_uploads_and_leaves_remote_archive_to_server(
    monkeypatch, tmp_path: Path
) -> None:
    sftp = _FakeSFTP()
    _seed_remote_pending(sftp, "job_002")
    ctx = _ctx(tmp_path, monkeypatch, sftp)

    def _fake_run(*_args, **_kwargs):
        job_path = ctx.queue_root / "running" / "job_002.json"
        (ctx.queue_root / "results" / "job_002.s2p").write_text("touchstone", encoding="utf-8")
        (ctx.queue_root / "results" / "job_002.log").write_text("log", encoding="utf-8")
        write_json(
            ctx.queue_root / "done" / "job_002.result.json",
            {
                "job_id": "job_002",
                "status": "done",
                "worker_name": "test_worker",
                "metrics": {"max_s11_db": -1.0},
                "artifacts": {
                    "touchstone": "results/job_002.s2p",
                    "csv": None,
                    "log": "results/job_002.log",
                },
            },
        )
        job_path.unlink()
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_queue_loop_sftp.subprocess, "run", _fake_run)

    assert run_queue_loop_sftp.run_loop(ctx, once=True) == 0

    assert not (ctx.queue_root / "done" / "job_002.result.json").exists()
    assert (ctx.queue_root / "archive" / "done" / "job_002.result.json").exists()
    assert "/remote/queue/done/job_002.result.json" in sftp.files
    assert "/remote/queue/results/job_002.s2p" in sftp.files
    assert "/remote/queue/running/job_002.json" in sftp.files
    assert "/remote/queue/archive/running/job_002.json" not in sftp.files


def test_failed_subprocess_result_uploads_and_archives(monkeypatch, tmp_path: Path) -> None:
    sftp = _FakeSFTP()
    _seed_remote_pending(sftp, "job_003")
    ctx = _ctx(tmp_path, monkeypatch, sftp)

    def _fake_run(*_args, **_kwargs):
        job_path = ctx.queue_root / "running" / "job_003.json"
        (ctx.queue_root / "results" / "job_003.log").write_text("log", encoding="utf-8")
        write_json(
            ctx.queue_root / "failed" / "job_003.result.json",
            {
                "job_id": "job_003",
                "status": "failed",
                "worker_name": "test_worker",
                "metrics": None,
                "artifacts": {"touchstone": None, "csv": None, "log": "results/job_003.log"},
                "error_message": "mesh_failure",
            },
        )
        job_path.unlink()
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")

    monkeypatch.setattr(run_queue_loop_sftp.subprocess, "run", _fake_run)

    assert run_queue_loop_sftp.run_loop(ctx, once=True) == 0

    assert (ctx.queue_root / "archive" / "failed" / "job_003.result.json").exists()
    assert "/remote/queue/failed/job_003.result.json" in sftp.files
    assert "/remote/queue/results/job_003.log" in sftp.files
    assert "/remote/queue/running/job_003.json" in sftp.files
    assert "/remote/queue/archive/running/job_003.json" not in sftp.files


def test_timeout_writes_failed_result_uploads_and_archives(monkeypatch, tmp_path: Path) -> None:
    sftp = _FakeSFTP()
    _seed_remote_pending(sftp, "job_004")
    ctx = _ctx(tmp_path, monkeypatch, sftp)

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="worker", timeout=10, output="out", stderr="err")

    monkeypatch.setattr(run_queue_loop_sftp.subprocess, "run", _timeout)

    assert run_queue_loop_sftp.run_loop(ctx, once=True) == 0

    archived = ctx.queue_root / "archive" / "failed" / "job_004.result.json"
    assert archived.exists()
    assert read_json(archived)["error_message"] == "job_timeout"
    assert (ctx.queue_root / "archive" / "running" / "job_004.json").exists()
    assert "/remote/queue/failed/job_004.result.json" in sftp.files
    assert "/remote/queue/running/job_004.json" in sftp.files
    assert "/remote/queue/archive/running/job_004.json" not in sftp.files


def test_stale_local_running_job_uploads_and_archives(monkeypatch, tmp_path: Path) -> None:
    sftp = _FakeSFTP()
    sftp.files["/remote/queue/running/job_005.json"] = (
        json.dumps(_remote_job_payload("job_005")).encode("utf-8")
    )
    ctx = _ctx(tmp_path, monkeypatch, sftp)
    local_running = ctx.queue_root / "running" / "job_005.json"
    write_json(local_running, _remote_job_payload("job_005"))
    os.utime(local_running, (100, 100))

    assert run_queue_loop_sftp.run_loop(ctx, once=True) == 0

    archived = ctx.queue_root / "archive" / "failed" / "job_005.result.json"
    assert archived.exists()
    assert read_json(archived)["error_message"] == "stale_running_job"
    assert (ctx.queue_root / "archive" / "running" / "job_005.json").exists()
    assert "/remote/queue/failed/job_005.result.json" in sftp.files
    assert "/remote/queue/running/job_005.json" in sftp.files
    assert "/remote/queue/archive/running/job_005.json" not in sftp.files


def test_archived_results_are_not_reuploaded(monkeypatch, tmp_path: Path) -> None:
    sftp = _FakeSFTP()
    ctx = _ctx(tmp_path, monkeypatch, sftp)
    (ctx.queue_root / "archive" / "done").mkdir(parents=True, exist_ok=True)
    write_json(
        ctx.queue_root / "archive" / "done" / "job_006.result.json",
        {
            "job_id": "job_006",
            "status": "done",
            "worker_name": "test_worker",
            "metrics": {},
            "artifacts": {},
        },
    )

    assert run_queue_loop_sftp.run_loop(ctx, once=True) == 0
    assert sftp.files == {}


def test_unarchived_local_result_is_uploaded_once_and_archived(
    monkeypatch, tmp_path: Path
) -> None:
    sftp = _FakeSFTP()
    sftp.files["/remote/queue/running/job_007.json"] = (
        json.dumps(_remote_job_payload("job_007")).encode("utf-8")
    )
    ctx = _ctx(tmp_path, monkeypatch, sftp)
    (ctx.queue_root / "results" / "job_007.log").write_text("log", encoding="utf-8")
    write_json(
        ctx.queue_root / "failed" / "job_007.result.json",
        {
            "job_id": "job_007",
            "status": "failed",
            "worker_name": "test_worker",
            "metrics": None,
            "artifacts": {"touchstone": None, "csv": None, "log": "results/job_007.log"},
            "error_message": "mesh_failure",
        },
    )

    assert run_queue_loop_sftp.run_loop(ctx, once=True) == 0

    assert not (ctx.queue_root / "failed" / "job_007.result.json").exists()
    assert (ctx.queue_root / "archive" / "failed" / "job_007.result.json").exists()
    assert "/remote/queue/failed/job_007.result.json" in sftp.files
    assert "/remote/queue/running/job_007.json" in sftp.files
    assert "/remote/queue/archive/running/job_007.json" not in sftp.files
