from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from common.io_utils import write_json
from worker.queue_loop import SftpResultUploader, SftpSettings, find_result_json_for_job


class _FakeSFTP:
    def __init__(self):
        self.files: list[tuple[str, str]] = []
        self.dirs: set[str] = set()

    def stat(self, remote_path: str):
        if remote_path not in self.dirs:
            raise FileNotFoundError(remote_path)
        return True

    def mkdir(self, remote_path: str):
        self.dirs.add(remote_path)

    def put(self, local_path: str, remote_path: str):
        self.files.append((local_path, remote_path))

    def close(self):
        return None


class _FakeSSHClient:
    def __init__(self):
        self.sftp = _FakeSFTP()
        self.connected = False
        self.known_hosts_loaded = None
        self.policy = None
        self.connect_kwargs = None

    def load_host_keys(self, path: str):
        self.known_hosts_loaded = path

    def set_missing_host_key_policy(self, policy):
        self.policy = policy

    def connect(self, **kwargs):
        self.connected = True
        self.connect_kwargs = kwargs
        if kwargs.get("hostname") == "unknown-host":
            raise RuntimeError("Unknown host key")

    def open_sftp(self):
        return self.sftp

    def close(self):
        self.connected = False


def _install_fake_paramiko(monkeypatch, fake_client: _FakeSSHClient) -> None:
    fake_paramiko = SimpleNamespace(
        SSHClient=lambda: fake_client,
        RejectPolicy=type("RejectPolicy", (), {}),
    )
    monkeypatch.setitem(sys.modules, "paramiko", fake_paramiko)


def _write_done_bundle(queue_root: Path, job_id: str) -> Path:
    (queue_root / "done").mkdir(parents=True, exist_ok=True)
    (queue_root / "results").mkdir(parents=True, exist_ok=True)
    (queue_root / "results" / f"{job_id}.s2p").write_text("touchstone", encoding="utf-8")
    (queue_root / "results" / f"{job_id}.csv").write_text("csv", encoding="utf-8")
    (queue_root / "results" / f"{job_id}.log").write_text("log", encoding="utf-8")
    result_json = queue_root / "done" / f"{job_id}.result.json"
    write_json(
        result_json,
        {
            "job_id": job_id,
            "status": "done",
            "artifacts": {
                "touchstone": f"results/{job_id}.s2p",
                "csv": f"results/{job_id}.csv",
                "log": f"results/{job_id}.log",
            },
        },
    )
    return result_json


def _uploader(
    *,
    queue_root: Path,
    server_host: str,
    server_port: int,
    username: str,
    private_key_path: Path,
    known_hosts_path: Path,
    remote_queue_root: str,
) -> SftpResultUploader:
    return SftpResultUploader(
        queue_root=queue_root,
        settings=SftpSettings(
            server_host=server_host,
            server_port=server_port,
            username=username,
            remote_queue_root=remote_queue_root,
            ssh_key=private_key_path,
            known_hosts=known_hosts_path,
        ),
    )


def test_upload_done_bundle_and_path_mirroring(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    result_json = _write_done_bundle(queue_root, "job_001")
    ssh_key = tmp_path / "id_ed25519"
    known_hosts = tmp_path / "known_hosts"
    ssh_key.write_text("dummy", encoding="utf-8")
    known_hosts.write_text("dummy", encoding="utf-8")

    fake_client = _FakeSSHClient()
    _install_fake_paramiko(monkeypatch, fake_client)

    summary = _uploader(
        queue_root=queue_root,
        server_host="10.15.89.240",
        server_port=22031,
        username="zhanghf",
        private_key_path=ssh_key,
        known_hosts_path=known_hosts,
        remote_queue_root="/home/zhanghf/projects/vector/queue",
    ).upload(result_json)

    assert summary.status == "uploaded"
    assert any(path.endswith("/done/job_001.result.json") for path in summary.uploaded_remote_paths)
    assert any(path.endswith("/results/job_001.s2p") for path in summary.uploaded_remote_paths)
    assert any(path.endswith("/results/job_001.csv") for path in summary.uploaded_remote_paths)
    assert any(path.endswith("/results/job_001.log") for path in summary.uploaded_remote_paths)


def test_upload_failed_bundle_log_only(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    (queue_root / "failed").mkdir(parents=True, exist_ok=True)
    (queue_root / "results").mkdir(parents=True, exist_ok=True)
    (queue_root / "results" / "job_002.log").write_text("log", encoding="utf-8")
    result_json = queue_root / "failed" / "job_002.result.json"
    write_json(
        result_json,
        {
            "job_id": "job_002",
            "status": "failed",
            "artifacts": {"log": "results/job_002.log"},
            "error_message": "mesh_failure",
        },
    )
    ssh_key = tmp_path / "id_rsa"
    known_hosts = tmp_path / "known_hosts"
    ssh_key.write_text("dummy", encoding="utf-8")
    known_hosts.write_text("dummy", encoding="utf-8")

    fake_client = _FakeSSHClient()
    _install_fake_paramiko(monkeypatch, fake_client)

    summary = _uploader(
        queue_root=queue_root,
        server_host="10.15.89.240",
        server_port=22031,
        username="zhanghf",
        private_key_path=ssh_key,
        known_hosts_path=known_hosts,
        remote_queue_root="/home/zhanghf/projects/vector/queue",
    ).upload(result_json)
    assert any(path.endswith("/failed/job_002.result.json") for path in summary.uploaded_remote_paths)
    assert any(path.endswith("/results/job_002.log") for path in summary.uploaded_remote_paths)
    assert all(not path.endswith(".s2p") for path in summary.uploaded_remote_paths)


def test_missing_artifact_fails_fast(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    (queue_root / "done").mkdir(parents=True, exist_ok=True)
    result_json = queue_root / "done" / "job_003.result.json"
    write_json(
        result_json,
        {
            "job_id": "job_003",
            "status": "done",
            "artifacts": {
                "touchstone": "results/job_003.s2p",
                "csv": "results/job_003.csv",
                "log": "results/job_003.log",
            },
        },
    )
    ssh_key = tmp_path / "id_key"
    known_hosts = tmp_path / "known_hosts"
    ssh_key.write_text("dummy", encoding="utf-8")
    known_hosts.write_text("dummy", encoding="utf-8")

    fake_client = _FakeSSHClient()
    _install_fake_paramiko(monkeypatch, fake_client)

    with pytest.raises(FileNotFoundError):
        _uploader(
            queue_root=queue_root,
            server_host="10.15.89.240",
            server_port=22031,
            username="zhanghf",
            private_key_path=ssh_key,
            known_hosts_path=known_hosts,
            remote_queue_root="/home/zhanghf/projects/vector/queue",
        ).upload(result_json)


def test_strict_host_key_unknown_host_raises(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    result_json = _write_done_bundle(queue_root, "job_004")
    ssh_key = tmp_path / "id_key"
    known_hosts = tmp_path / "known_hosts"
    ssh_key.write_text("dummy", encoding="utf-8")
    known_hosts.write_text("dummy", encoding="utf-8")

    fake_client = _FakeSSHClient()
    _install_fake_paramiko(monkeypatch, fake_client)

    with pytest.raises(RuntimeError):
        _uploader(
            queue_root=queue_root,
            server_host="unknown-host",
            server_port=22031,
            username="zhanghf",
            private_key_path=ssh_key,
            known_hosts_path=known_hosts,
            remote_queue_root="/home/zhanghf/projects/vector/queue",
        ).upload(result_json)


def test_find_result_json_for_job_prefers_existing_state(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    done_dir = queue_root / "done"
    failed_dir = queue_root / "failed"
    done_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    done_path = done_dir / "job_005.result.json"
    done_path.write_text("{}", encoding="utf-8")
    selected = find_result_json_for_job(queue_root, "job_005")
    assert selected == done_path
