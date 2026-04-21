from datetime import UTC, datetime
from pathlib import Path

from worker.schemas import ArtifactPaths, JobRequest
from worker.state_manager import write_failure_result, write_success_result


def _job() -> JobRequest:
    return JobRequest.model_validate(
        {
            "job_id": "bclf_000001",
            "design_name": "BCLF",
            "template_name": "bclf_template_v1",
            "setup_name": "Setup1",
            "sweep_name": "Sweep1",
            "variables": {"L": "7.8mm"},
            "metrics_to_extract": ["max_s11_db"],
            "artifact_options": {"export_touchstone": True, "export_csv": True},
            "submit_time": "2026-04-21T12:00:00Z",
            "retry_count": 0,
        }
    )


def _mkdir_queue(root: Path) -> None:
    for name in ("pending", "running", "done", "failed", "results"):
        (root / name).mkdir(parents=True, exist_ok=True)


def test_success_json_written(tmp_path: Path) -> None:
    _mkdir_queue(tmp_path)
    running = tmp_path / "running" / "job.json"
    running.write_text("{}", encoding="utf-8")
    out = write_success_result(
        queue_root=tmp_path,
        worker_name="hfss_win01",
        job=_job(),
        running_job_path=running,
        start_time=datetime.now(UTC),
        finish_time=datetime.now(UTC),
        metrics={"max_s11_db": -9.3},
        artifacts=ArtifactPaths(
            touchstone="results/a.s2p", csv="results/a.csv", log="results/a.log"
        ),
    )
    assert out.exists()
    assert not running.exists()


def test_failure_json_written(tmp_path: Path) -> None:
    _mkdir_queue(tmp_path)
    running = tmp_path / "running" / "job.json"
    running.write_text("{}", encoding="utf-8")
    out = write_failure_result(
        queue_root=tmp_path,
        worker_name="hfss_win01",
        job_id="bclf_000002",
        running_job_path=running,
        start_time=datetime.now(UTC),
        finish_time=datetime.now(UTC),
        artifacts=ArtifactPaths(log="results/b.log"),
        error_message="mesh_failure",
    )
    assert out.exists()
    assert not running.exists()
