from __future__ import annotations

from datetime import datetime
from pathlib import Path

from common.io_utils import atomic_write_json
from worker.schemas import ArtifactPaths, FailureResult, JobRequest, JobResult
from worker.utils import WorkerError


def _result_path(queue_root: Path, state: str, job_id: str) -> Path:
    return queue_root / state / f"{job_id}.result.json"


def _remove_running_file(running_job_path: Path) -> None:
    if running_job_path.exists():
        running_job_path.unlink()


def write_success_result(
    queue_root: Path,
    worker_name: str,
    job: JobRequest,
    running_job_path: Path,
    start_time: datetime,
    finish_time: datetime,
    metrics: dict[str, float],
    artifacts: ArtifactPaths,
) -> Path:
    out_path = _result_path(queue_root, "done", job.job_id)
    if out_path.exists():
        raise WorkerError("unexpected_exception", f"Result already exists: {out_path}")

    result = JobResult(
        job_id=job.job_id,
        status="done",
        worker_name=worker_name,
        start_time=start_time,
        finish_time=finish_time,
        metrics=metrics,
        artifacts=artifacts,
        error_message=None,
    )
    atomic_write_json(out_path, result.model_dump(mode="json"))
    _remove_running_file(running_job_path)
    return out_path


def write_failure_result(
    queue_root: Path,
    worker_name: str,
    job_id: str,
    running_job_path: Path,
    start_time: datetime,
    finish_time: datetime,
    artifacts: ArtifactPaths,
    error_message: str,
) -> Path:
    out_path = _result_path(queue_root, "failed", job_id)
    if out_path.exists():
        raise WorkerError("unexpected_exception", f"Result already exists: {out_path}")

    result = FailureResult(
        job_id=job_id,
        worker_name=worker_name,
        start_time=start_time,
        finish_time=finish_time,
        artifacts=artifacts,
        error_message=error_message,
    )
    atomic_write_json(out_path, result.model_dump(mode="json"))
    _remove_running_file(running_job_path)
    return out_path
