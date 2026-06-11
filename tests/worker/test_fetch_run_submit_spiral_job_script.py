from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from common.io_utils import write_json
from scripts.worker import fetch_run_submit_job
from worker.queue_loop import ClaimedJob


def _claim(job_id: str, running_path: Path) -> ClaimedJob:
    return ClaimedJob(
        job_id=job_id,
        job_name=running_path.name,
        local_job_path=running_path,
        remote_running_path=f"/home/zhanghf/projects/vector/queue/running/{running_path.name}",
    )


def _install_fake_remote_queue(
    monkeypatch, claims: list[ClaimedJob | None | BaseException]
) -> None:
    class _FakeRemoteQueueClient:
        def __init__(self, _settings, _local_queue_root: Path) -> None:
            pass

        def claim_one(self) -> ClaimedJob | None:
            claim = claims.pop(0)
            if isinstance(claim, BaseException):
                raise claim
            return claim

    monkeypatch.setattr(fetch_run_submit_job, "SftpRemoteQueueClient", _FakeRemoteQueueClient)


def _install_fake_uploader(
    monkeypatch,
    uploaded_remote_paths: list[str],
    failures: list[BaseException] | None = None,
) -> list[Path]:
    uploaded_result_paths: list[Path] = []
    pending_failures = list(failures or [])

    class _FakeUploader:
        def __init__(self, *, queue_root: Path, settings) -> None:
            self.queue_root = queue_root
            self.settings = settings

        def upload(self, result_json_path: Path):
            if pending_failures:
                raise pending_failures.pop(0)
            uploaded_result_paths.append(result_json_path)
            return SimpleNamespace(uploaded_remote_paths=uploaded_remote_paths)

    monkeypatch.setattr(fetch_run_submit_job, "SftpResultUploader", _FakeUploader)
    return uploaded_result_paths


def _install_fake_processor(monkeypatch, process_func) -> None:
    class _FakeProcessor:
        @classmethod
        def default(cls, *, cfg, template_registry, global_logger):
            return cls()

        def process(self, path: Path) -> None:
            process_func(path)

    monkeypatch.setattr(fetch_run_submit_job, "ClaimedJobProcessor", _FakeProcessor)


def test_no_pending_remote_job_returns_zero(monkeypatch, tmp_path: Path, capsys) -> None:
    cfg = SimpleNamespace(
        queue_root=tmp_path / "queue",
        logs_dir=tmp_path / "logs",
        poll_interval_sec=1,
    )
    monkeypatch.setattr(
        fetch_run_submit_job,
        "_load_runtime",
        lambda _worker_config, _queue_root_override: (cfg, object(), object()),
    )
    _install_fake_remote_queue(monkeypatch, [None])

    rc = fetch_run_submit_job.run(
        worker_config=tmp_path / "worker.yaml",
        queue_root=None,
        server_host="10.15.89.240",
        server_port=22031,
        username="zhanghf",
        ssh_key=tmp_path / "id",
        known_hosts=tmp_path / "known_hosts",
        remote_queue_root="/home/zhanghf/projects/vector/queue",
        wait_when_empty=False,
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "status=no_remote_pending" in out


def test_waits_then_claims_when_pending_appears(monkeypatch, tmp_path: Path, capsys) -> None:
    queue_root = tmp_path / "queue"
    running_path = queue_root / "running" / "job_wait_001.json"
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_path.write_text('{"job_id":"job_wait_001"}', encoding="utf-8")
    cfg = SimpleNamespace(queue_root=queue_root, logs_dir=tmp_path / "logs", poll_interval_sec=2)
    monkeypatch.setattr(
        fetch_run_submit_job,
        "_load_runtime",
        lambda _worker_config, _queue_root_override: (cfg, object(), object()),
    )
    claims = [
        None,
        _claim("job_wait_001", running_path),
    ]
    _install_fake_remote_queue(monkeypatch, claims)
    sleeps: list[int] = []
    monkeypatch.setattr(fetch_run_submit_job.time, "sleep", lambda sec: sleeps.append(sec))

    def _process(path):
        assert path == running_path
        (queue_root / "done").mkdir(parents=True, exist_ok=True)
        write_json(
            queue_root / "done" / "job_wait_001.result.json",
            {"job_id": "job_wait_001", "status": "done", "artifacts": {"log": "results/job_wait_001.log"}},
        )

    _install_fake_processor(monkeypatch, _process)
    _install_fake_uploader(monkeypatch, ["one"])
    rc = fetch_run_submit_job.run(
        worker_config=tmp_path / "worker.yaml",
        queue_root=None,
        server_host="10.15.89.240",
        server_port=22031,
        username="zhanghf",
        ssh_key=tmp_path / "id",
        known_hosts=tmp_path / "known_hosts",
        remote_queue_root="/home/zhanghf/projects/vector/queue",
        once=True,
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert sleeps == [2]
    assert "status=waiting_for_remote_pending poll_interval_sec=2" in out
    assert "claimed=/home/zhanghf/projects/vector/queue/running/job_wait_001.json" in out


def test_waits_after_remote_connect_failure_then_claims(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    queue_root = tmp_path / "queue"
    running_path = queue_root / "running" / "job_retry_001.json"
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_path.write_text('{"job_id":"job_retry_001"}', encoding="utf-8")
    cfg = SimpleNamespace(queue_root=queue_root, logs_dir=tmp_path / "logs", poll_interval_sec=2)
    monkeypatch.setattr(
        fetch_run_submit_job,
        "_load_runtime",
        lambda _worker_config, _queue_root_override: (cfg, object(), object()),
    )
    _install_fake_remote_queue(
        monkeypatch,
        [
            ConnectionError("SFTP connect failed after 5 attempts"),
            _claim("job_retry_001", running_path),
        ],
    )
    sleeps: list[int] = []
    monkeypatch.setattr(fetch_run_submit_job.time, "sleep", lambda sec: sleeps.append(sec))

    def _process(path):
        assert path == running_path
        (queue_root / "done").mkdir(parents=True, exist_ok=True)
        write_json(
            queue_root / "done" / "job_retry_001.result.json",
            {"job_id": "job_retry_001", "status": "done", "artifacts": {"log": "results/job_retry_001.log"}},
        )

    _install_fake_processor(monkeypatch, _process)
    _install_fake_uploader(monkeypatch, ["one"])

    rc = fetch_run_submit_job.run(
        worker_config=tmp_path / "worker.yaml",
        queue_root=None,
        server_host="10.15.89.240",
        server_port=22031,
        username="zhanghf",
        ssh_key=tmp_path / "id",
        known_hosts=tmp_path / "known_hosts",
        remote_queue_root="/home/zhanghf/projects/vector/queue",
        once=True,
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert sleeps == [2]
    assert "status=sftp_connect_failed error_class=ConnectionError" in out
    assert "poll_interval_sec=2" in out
    assert "claimed=/home/zhanghf/projects/vector/queue/running/job_retry_001.json" in out


def test_run_loop_processes_multiple_jobs_without_exiting(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    queue_root = tmp_path / "queue"
    first_running = queue_root / "running" / "job_loop_001.json"
    second_running = queue_root / "running" / "job_loop_002.json"
    first_running.parent.mkdir(parents=True, exist_ok=True)
    first_running.write_text('{"job_id":"job_loop_001"}', encoding="utf-8")
    second_running.write_text('{"job_id":"job_loop_002"}', encoding="utf-8")
    cfg = SimpleNamespace(queue_root=queue_root, logs_dir=tmp_path / "logs", poll_interval_sec=1)
    monkeypatch.setattr(
        fetch_run_submit_job,
        "_load_runtime",
        lambda _worker_config, _queue_root_override: (cfg, object(), object()),
    )
    claims = [
        _claim("job_loop_001", first_running),
        _claim("job_loop_002", second_running),
    ]
    _install_fake_remote_queue(monkeypatch, claims)

    def _process(path):
        job_id = path.stem
        (queue_root / "done").mkdir(parents=True, exist_ok=True)
        write_json(
            queue_root / "done" / f"{job_id}.result.json",
            {"job_id": job_id, "status": "done", "artifacts": {"log": f"results/{job_id}.log"}},
        )

    _install_fake_processor(monkeypatch, _process)
    _install_fake_uploader(monkeypatch, ["one"])

    rc = fetch_run_submit_job.run(
        worker_config=tmp_path / "worker.yaml",
        queue_root=None,
        server_host="10.15.89.240",
        server_port=22031,
        username="zhanghf",
        ssh_key=tmp_path / "id",
        known_hosts=tmp_path / "known_hosts",
        remote_queue_root="/home/zhanghf/projects/vector/queue",
        max_jobs=2,
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "claimed=/home/zhanghf/projects/vector/queue/running/job_loop_001.json" in out
    assert "claimed=/home/zhanghf/projects/vector/queue/running/job_loop_002.json" in out


def test_happy_path_claim_run_upload_leaves_remote_archive_to_server(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    queue_root = tmp_path / "queue"
    running_path = queue_root / "running" / "job_001.json"
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_path.write_text('{"job_id":"job_001"}', encoding="utf-8")

    cfg = SimpleNamespace(queue_root=queue_root, logs_dir=tmp_path / "logs", poll_interval_sec=1)
    monkeypatch.setattr(
        fetch_run_submit_job,
        "_load_runtime",
        lambda _worker_config, _queue_root_override: (cfg, object(), object()),
    )
    _install_fake_remote_queue(monkeypatch, [_claim("job_001", running_path)])

    def _process(path):
        assert path == running_path
        (queue_root / "done").mkdir(parents=True, exist_ok=True)
        write_json(
            queue_root / "done" / "job_001.result.json",
            {"job_id": "job_001", "status": "done", "artifacts": {"log": "results/job_001.log"}},
        )

    _install_fake_processor(monkeypatch, _process)
    _install_fake_uploader(monkeypatch, ["a", "b"])
    rc = fetch_run_submit_job.run(
        worker_config=tmp_path / "worker.yaml",
        queue_root=None,
        server_host="10.15.89.240",
        server_port=22031,
        username="zhanghf",
        ssh_key=tmp_path / "id",
        known_hosts=tmp_path / "known_hosts",
        remote_queue_root="/home/zhanghf/projects/vector/queue",
        once=True,
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "claimed=/home/zhanghf/projects/vector/queue/running/job_001.json" in out
    assert "status=done" in out
    assert "uploaded_count=2" in out
    assert "remote_archive=server_managed" in out


def test_failed_result_upload_path(monkeypatch, tmp_path: Path, capsys) -> None:
    queue_root = tmp_path / "queue"
    running_path = queue_root / "running" / "job_002.json"
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_path.write_text('{"job_id":"job_002"}', encoding="utf-8")
    cfg = SimpleNamespace(queue_root=queue_root, logs_dir=tmp_path / "logs", poll_interval_sec=1)
    monkeypatch.setattr(
        fetch_run_submit_job,
        "_load_runtime",
        lambda _worker_config, _queue_root_override: (cfg, object(), object()),
    )
    _install_fake_remote_queue(monkeypatch, [_claim("job_002", running_path)])

    def _process(path):
        assert path == running_path
        (queue_root / "failed").mkdir(parents=True, exist_ok=True)
        write_json(
            queue_root / "failed" / "job_002.result.json",
            {"job_id": "job_002", "status": "failed", "artifacts": {"log": "results/job_002.log"}},
        )

    _install_fake_processor(monkeypatch, _process)
    _install_fake_uploader(monkeypatch, ["a"])
    rc = fetch_run_submit_job.run(
        worker_config=tmp_path / "worker.yaml",
        queue_root=None,
        server_host="10.15.89.240",
        server_port=22031,
        username="zhanghf",
        ssh_key=tmp_path / "id",
        known_hosts=tmp_path / "known_hosts",
        remote_queue_root="/home/zhanghf/projects/vector/queue",
        once=True,
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "status=failed" in out


def test_retries_upload_connection_failure_before_continuing(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    queue_root = tmp_path / "queue"
    running_path = queue_root / "running" / "job_upload_retry_001.json"
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_path.write_text('{"job_id":"job_upload_retry_001"}', encoding="utf-8")
    cfg = SimpleNamespace(queue_root=queue_root, logs_dir=tmp_path / "logs", poll_interval_sec=1)
    monkeypatch.setattr(
        fetch_run_submit_job,
        "_load_runtime",
        lambda _worker_config, _queue_root_override: (cfg, object(), object()),
    )
    _install_fake_remote_queue(monkeypatch, [_claim("job_upload_retry_001", running_path)])

    def _process(path):
        assert path == running_path
        (queue_root / "done").mkdir(parents=True, exist_ok=True)
        write_json(
            queue_root / "done" / "job_upload_retry_001.result.json",
            {
                "job_id": "job_upload_retry_001",
                "status": "done",
                "artifacts": {"log": "results/job_upload_retry_001.log"},
            },
        )

    _install_fake_processor(monkeypatch, _process)
    uploaded = _install_fake_uploader(
        monkeypatch,
        ["a"],
        failures=[ConnectionError("SFTP connect failed after 5 attempts")],
    )
    sleeps: list[int] = []
    monkeypatch.setattr(fetch_run_submit_job.time, "sleep", lambda sec: sleeps.append(sec))

    rc = fetch_run_submit_job.run(
        worker_config=tmp_path / "worker.yaml",
        queue_root=None,
        server_host="10.15.89.240",
        server_port=22031,
        username="zhanghf",
        ssh_key=tmp_path / "id",
        known_hosts=tmp_path / "known_hosts",
        remote_queue_root="/home/zhanghf/projects/vector/queue",
        once=True,
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert sleeps == [1]
    assert len(uploaded) == 1
    assert "status=remote_upload_connect_failed_retrying poll_interval_sec=1" in out
    assert "status=done" in out


def test_removes_stale_failed_result_before_processing(monkeypatch, tmp_path: Path, capsys) -> None:
    queue_root = tmp_path / "queue"
    running_path = queue_root / "running" / "job_003.json"
    failed_path = queue_root / "failed" / "job_003.result.json"
    running_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    running_path.write_text('{"job_id":"job_003"}', encoding="utf-8")
    write_json(failed_path, {"job_id": "job_003", "status": "failed"})

    cfg = SimpleNamespace(queue_root=queue_root, logs_dir=tmp_path / "logs", poll_interval_sec=1)
    monkeypatch.setattr(
        fetch_run_submit_job,
        "_load_runtime",
        lambda _worker_config, _queue_root_override: (cfg, object(), object()),
    )
    _install_fake_remote_queue(monkeypatch, [_claim("job_003", running_path)])

    def _process(path):
        assert path == running_path
        assert not failed_path.exists()
        (queue_root / "failed").mkdir(parents=True, exist_ok=True)
        write_json(
            queue_root / "failed" / "job_003.result.json",
            {"job_id": "job_003", "status": "failed", "artifacts": {"log": "results/job_003.log"}},
        )

    _install_fake_processor(monkeypatch, _process)
    _install_fake_uploader(monkeypatch, ["x"])
    rc = fetch_run_submit_job.run(
        worker_config=tmp_path / "worker.yaml",
        queue_root=None,
        server_host="10.15.89.240",
        server_port=22031,
        username="zhanghf",
        ssh_key=tmp_path / "id",
        known_hosts=tmp_path / "known_hosts",
        remote_queue_root="/home/zhanghf/projects/vector/queue",
        once=True,
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "removed_stale_result=" in out
    assert "status=failed" in out
