from __future__ import annotations

import json
from pathlib import Path
import time

from worker.queue_loop import (
    ClaimedJob,
    JobResult,
    StaleJobPolicy,
    SubprocessJobRunner,
    WorkerLoopService,
)


class ListReporter:
    def __init__(self) -> None:
        self.events: list[str] = []

    def event(self, message: str) -> None:
        self.events.append(message)


class FakeRemoteQueue:
    def __init__(self, local_queue_root: Path) -> None:
        self.local_queue_root = local_queue_root
        self.pending = [
            {
                "job_id": "demo_job_001",
                "remote_running_path": "running/demo_job_001.json",
            }
        ]

    def claim_one(self) -> ClaimedJob | None:
        if not self.pending:
            return None
        payload = self.pending.pop(0)
        job_path = self.local_queue_root / "running" / f"{payload['job_id']}.json"
        job_path.parent.mkdir(parents=True, exist_ok=True)
        job_path.write_text(json.dumps({"job_id": payload["job_id"]}), encoding="utf-8")
        return ClaimedJob(
            job_id=payload["job_id"],
            job_name=job_path.name,
            local_job_path=job_path,
            remote_running_path=payload["remote_running_path"],
        )

    def running_path_for_local_job(self, job_path: Path) -> str:
        return f"running/{job_path.name}"


class FileLocalQueueStore:
    def __init__(self, queue_root: Path, stale_policy: StaleJobPolicy) -> None:
        self.queue_root = queue_root
        self.stale_policy = stale_policy
        for name in (
            "running",
            "done",
            "failed",
            "archive/done",
            "archive/failed",
            "archive/running",
        ):
            (queue_root / name).mkdir(parents=True, exist_ok=True)

    def stale_running_jobs(self) -> list[Path]:
        now = time.time()
        return [
            path
            for path in sorted((self.queue_root / "running").glob("*.json"))
            if self.stale_policy.is_stale(path, now)
        ]

    def unuploaded_results(self) -> list[Path]:
        return sorted((self.queue_root / "done").glob("*.result.json")) + sorted(
            (self.queue_root / "failed").glob("*.result.json")
        )

    def job_id_from_file(self, job_path: Path) -> str:
        return json.loads(job_path.read_text(encoding="utf-8"))["job_id"]

    def job_id_from_result(self, result_json_path: Path) -> str | None:
        value = json.loads(result_json_path.read_text(encoding="utf-8")).get("job_id")
        return value if isinstance(value, str) else None

    def find_result_json(self, job_id: str) -> Path | None:
        for state in ("done", "failed"):
            result_path = self.queue_root / state / f"{job_id}.result.json"
            if result_path.exists():
                return result_path
        return None

    def write_failure_result(self, job_path: Path, error_message: str, detail: str = "") -> Path:
        job_id = self.job_id_from_file(job_path)
        result_path = self.queue_root / "failed" / f"{job_id}.result.json"
        result_path.write_text(
            json.dumps({"job_id": job_id, "status": "failed", "error_message": error_message}),
            encoding="utf-8",
        )
        return result_path

    def archive_result(self, result_json_path: Path) -> Path:
        state = result_json_path.parent.name
        target = self.queue_root / "archive" / state / result_json_path.name
        result_json_path.replace(target)
        return target

    def archive_running_job(self, job_path: Path) -> Path | None:
        if not job_path.exists():
            return None
        target = self.queue_root / "archive" / "running" / job_path.name
        job_path.replace(target)
        return target


class FakeJobRunner:
    def __init__(self, queue_root: Path) -> None:
        self.queue_root = queue_root

    def run(self, claimed: ClaimedJob) -> JobResult:
        result_path = self.queue_root / "done" / f"{claimed.job_id}.result.json"
        result_path.write_text(
            json.dumps(
                {
                    "job_id": claimed.job_id,
                    "status": "done",
                    "metrics": {"max_s11_db": -0.2},
                }
            ),
            encoding="utf-8",
        )
        return JobResult(claimed.job_id, result_path, "done")


class FakeResultUploader:
    def __init__(self) -> None:
        self.uploaded: list[Path] = []

    def upload(self, result_json_path: Path) -> None:
        self.uploaded.append(result_json_path)


class _MinimalLocalStore:
    def __init__(self, result_json_path: Path) -> None:
        self.result_json_path = result_json_path
        self.failure_writes: list[tuple[Path, str, str]] = []

    def find_result_json(self, _job_id: str) -> Path | None:
        if self.result_json_path.exists():
            return self.result_json_path
        return None

    def write_failure_result(self, job_path: Path, error_message: str, detail: str = "") -> Path:
        self.failure_writes.append((job_path, error_message, detail))
        self.result_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.result_json_path.write_text(
            json.dumps({"job_id": "demo_job_002", "status": "failed"}),
            encoding="utf-8",
        )
        return self.result_json_path


def test_worker_loop_service_runs_one_fake_job_then_idles(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    local_store = FileLocalQueueStore(
        queue_root=queue_root,
        stale_policy=StaleJobPolicy(stale_after_sec=3600),
    )
    remote_queue = FakeRemoteQueue(queue_root)
    uploader = FakeResultUploader()
    reporter = ListReporter()
    service = WorkerLoopService(
        remote_queue=remote_queue,
        local_store=local_store,
        runner=FakeJobRunner(queue_root),
        uploader=uploader,
        reporter=reporter,
    )

    assert service.run_once() is True
    assert service.run_once() is False

    assert (queue_root / "archive" / "done" / "demo_job_001.result.json").exists()
    assert (queue_root / "archive" / "running" / "demo_job_001.json").exists()
    assert [path.name for path in uploader.uploaded] == ["demo_job_001.result.json"]
    assert reporter.events[-1] == "idle"


def test_subprocess_runner_uses_generic_fetched_job_entrypoint(tmp_path: Path) -> None:
    repo_root = tmp_path
    running_job_path = repo_root / "queue" / "running" / "demo_job_002.json"
    running_job_path.parent.mkdir(parents=True, exist_ok=True)
    running_job_path.write_text(json.dumps({"job_id": "demo_job_002"}), encoding="utf-8")
    done_result_path = repo_root / "queue" / "done" / "demo_job_002.result.json"
    done_result_path.parent.mkdir(parents=True, exist_ok=True)
    done_result_path.write_text(
        json.dumps({"job_id": "demo_job_002", "status": "done"}),
        encoding="utf-8",
    )
    local_store = _MinimalLocalStore(done_result_path)
    commands: list[list[str]] = []

    def _fake_run(command, **_kwargs):
        commands.append(command)
        return type("CP", (), {"returncode": 0})()

    runner = SubprocessJobRunner(
        repo_root=repo_root,
        worker_config_path=repo_root / "configs" / "worker.yaml",
        timeout_sec=30,
        local_store=local_store,
        run_command=_fake_run,
    )
    result = runner.run(
        ClaimedJob(
            job_id="demo_job_002",
            job_name="demo_job_002.json",
            local_job_path=running_job_path,
            remote_running_path="running/demo_job_002.json",
        )
    )

    assert result.status == "done"
    assert commands
    assert commands[0][1] == "-m"
    assert commands[0][2] == "worker.run_fetched_job_cli"
