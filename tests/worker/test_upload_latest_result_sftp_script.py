from __future__ import annotations

from pathlib import Path
import os
import sys
import time

from common.io_utils import write_json
from scripts.worker import upload_latest_result_sftp


def test_parse_args_uses_default_ssh_files(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["upload_latest_result_sftp.py"])
    args = upload_latest_result_sftp.parse_args()
    assert args.ssh_key == upload_latest_result_sftp.DEFAULT_SSH_KEY
    assert args.known_hosts == upload_latest_result_sftp.DEFAULT_KNOWN_HOSTS


def test_find_latest_result_json_across_done_failed(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    done_dir = queue_root / "done"
    failed_dir = queue_root / "failed"
    done_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    older = done_dir / "a.result.json"
    newer = failed_dir / "b.result.json"
    write_json(older, {"job_id": "a"})
    old_time = time.time() - 10
    os.utime(older, (old_time, old_time))
    write_json(newer, {"job_id": "b"})
    selected = upload_latest_result_sftp._find_latest_result_json(queue_root)
    assert selected == newer


def test_run_upload_with_explicit_result_json_bypasses_latest(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    queue_root = tmp_path / "queue"
    result_json = queue_root / "done" / "job_001.result.json"
    result_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(result_json, {"job_id": "job_001"})

    captured = {}

    def _fake_upload(**kwargs):
        captured.update(kwargs)
        return type(
            "Summary",
            (),
            {
                "result_json_path": kwargs["result_json_path"],
                "uploaded_remote_paths": ["/home/zhanghf/projects/vector/queue/done/job_001.result.json"],
                "total_bytes": 123,
                "status": "uploaded",
            },
        )()

    monkeypatch.setattr(upload_latest_result_sftp, "upload_job_bundle_via_sftp", _fake_upload)
    rc = upload_latest_result_sftp.run_upload(
        queue_root=queue_root,
        server_host="10.15.89.240",
        server_port=22031,
        username="zhanghf",
        remote_queue_root="/home/zhanghf/projects/vector/queue",
        ssh_key=tmp_path / "id",
        known_hosts=tmp_path / "known_hosts",
        result_json=result_json,
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert captured["result_json_path"] == result_json
    assert "selected_result_json=" in out
    assert "uploaded=/home/zhanghf/projects/vector/queue/done/job_001.result.json" in out
    assert "status=uploaded" in out


def test_run_upload_no_results_returns_error(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    rc = upload_latest_result_sftp.run_upload(
        queue_root=queue_root,
        server_host="10.15.89.240",
        server_port=22031,
        username="zhanghf",
        remote_queue_root="/home/zhanghf/projects/vector/queue",
        ssh_key=tmp_path / "id",
        known_hosts=tmp_path / "known_hosts",
        result_json=None,
    )
    assert rc == 2
