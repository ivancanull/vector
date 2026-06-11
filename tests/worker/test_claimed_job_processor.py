from __future__ import annotations

from pathlib import Path

from common.io_utils import read_json, write_json
from worker.schemas import ArtifactPaths, JobResult, TemplatesConfig, WorkerConfig
from worker.template_registry import TemplateRegistry
from worker.utils import WorkerError, utc_now
from worker.worker_main import (
    ClaimedJobProcessor,
    JobLoggerFactory,
    MetricExtractor,
    _resolve_artifact_path,
)


def _worker_config(tmp_path: Path) -> WorkerConfig:
    queue_root = tmp_path / "queue"
    return WorkerConfig(
        worker_name="worker-a",
        queue_root=queue_root,
        templates_yaml=tmp_path / "templates.yaml",
        poll_interval_sec=10,
        claim_retry_sec=5,
        results_dir=queue_root / "results",
        logs_dir=queue_root / "logs",
        default_export_touchstone=True,
        default_export_csv=False,
    )


def _registry(tmp_path: Path) -> TemplateRegistry:
    project_file = tmp_path / "template.aedt"
    project_file.write_text("placeholder", encoding="utf-8")
    return TemplateRegistry(
        TemplatesConfig(
            templates={
                "si_template_v1": {
                    "project_path": project_file,
                    "design_name": "HFSSDesign1",
                    "setup_name": "Setup1",
                    "sweep_name": "Sweep",
                    "hfss_sweep": {
                        "sweep_type": "interpolating",
                        "sweep_data": "LINC 0GHz 4GHz 201",
                    },
                }
            }
        )
    )


def _job_payload() -> dict:
    return {
        "job_id": "job-001",
        "template_name": "si_template_v1",
        "variables": {"Din": "55um", "H": "2um", "S": "1.5um", "W": "15um"},
        "output_filename": "job-001.s2p",
        "metrics_to_extract": ["max_s11_db"],
        "artifact_options": {"export_touchstone": True, "export_csv": False},
        "submit_time": "2026-04-24T08:00:00Z",
        "retry_count": 0,
    }


class StaticMetricExtractor(MetricExtractor):
    def extract(self, touchstone_path: Path, metrics: list[str], *, job_id: str) -> dict[str, float]:
        assert touchstone_path.exists()
        return {"max_s11_db": -1.0}


class SuccessExecutor:
    def __init__(self, cfg: WorkerConfig, *, with_touchstone: bool = True) -> None:
        self.cfg = cfg
        self.with_touchstone = with_touchstone

    def execute(self, job, template, worker_cfg) -> JobResult:
        artifacts = ArtifactPaths(touchstone=None, csv=None, log=None)
        if self.with_touchstone:
            touchstone = self.cfg.results_dir / "job-001.s2p"
            touchstone.parent.mkdir(parents=True, exist_ok=True)
            touchstone.write_text("touchstone", encoding="utf-8")
            artifacts.touchstone = "results/job-001.s2p"
        now = utc_now()
        return JobResult(
            job_id=job.job_id,
            status="done",
            worker_name=worker_cfg.worker_name,
            start_time=now,
            finish_time=now,
            metrics=None,
            artifacts=artifacts,
            error_message=None,
        )


class WorkerErrorExecutor:
    def execute(self, job, template, worker_cfg) -> JobResult:
        raise WorkerError("mesh_failure", "mesh failed")


class UnexpectedExecutor:
    def execute(self, job, template, worker_cfg) -> JobResult:
        raise RuntimeError("boom")


class FakeJobLogger:
    handlers: list = []

    def info(self, *args, **kwargs) -> None:
        return None

    def exception(self, *args, **kwargs) -> None:
        return None

    def error(self, *args, **kwargs) -> None:
        return None


class FakeJobLoggerFactory(JobLoggerFactory):
    def create(self, logs_dir: Path, job_id: str):
        log_path = logs_dir / "hfss_runs" / f"{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("job log\n", encoding="utf-8")
        return FakeJobLogger(), log_path


def _processor(tmp_path: Path, executor) -> tuple[ClaimedJobProcessor, WorkerConfig]:
    cfg = _worker_config(tmp_path)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    processor = ClaimedJobProcessor(
        cfg=cfg,
        template_registry=_registry(tmp_path),
        global_logger=type("Logger", (), {"info": lambda self, *args, **kwargs: None})(),
        executor=executor,
        metric_extractor=StaticMetricExtractor(),
        logger_factory=FakeJobLoggerFactory(),
    )
    return processor, cfg


def _running_job(cfg: WorkerConfig, payload: dict | None = None) -> Path:
    running_path = cfg.queue_root / "running" / "job-001.json"
    write_json(running_path, payload or _job_payload())
    return running_path


def test_claimed_job_processor_success_writes_done_and_removes_running(tmp_path: Path) -> None:
    cfg = _worker_config(tmp_path)
    processor, cfg = _processor(tmp_path, SuccessExecutor(cfg))
    running_path = _running_job(cfg)

    processor.process(running_path)

    assert not running_path.exists()
    result = read_json(cfg.queue_root / "done" / "job-001.result.json")
    assert result["status"] == "done"
    assert result["metrics"] == {"max_s11_db": -1.0}
    assert result["artifacts"]["log"] == "results/job-001.log"


def test_claimed_job_processor_worker_error_writes_failed(tmp_path: Path) -> None:
    processor, cfg = _processor(tmp_path, WorkerErrorExecutor())
    running_path = _running_job(cfg)

    processor.process(running_path)

    assert not running_path.exists()
    result = read_json(cfg.queue_root / "failed" / "job-001.result.json")
    assert result["status"] == "failed"
    assert result["error_message"] == "mesh_failure"


def test_claimed_job_processor_unexpected_exception_writes_failed(tmp_path: Path) -> None:
    processor, cfg = _processor(tmp_path, UnexpectedExecutor())
    running_path = _running_job(cfg)

    processor.process(running_path)

    result = read_json(cfg.queue_root / "failed" / "job-001.result.json")
    assert result["error_message"] == "unexpected_exception"


def test_claimed_job_processor_missing_touchstone_writes_metric_failure(tmp_path: Path) -> None:
    cfg = _worker_config(tmp_path)
    processor, cfg = _processor(tmp_path, SuccessExecutor(cfg, with_touchstone=False))
    running_path = _running_job(cfg)

    processor.process(running_path)

    result = read_json(cfg.queue_root / "failed" / "job-001.result.json")
    assert result["error_message"] == "metric_extraction_failure"


def test_resolve_artifact_path_rebases_relative_queue_artifact(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"

    assert _resolve_artifact_path(queue_root, "results/job-001.s2p") == queue_root / "results" / "job-001.s2p"


def test_resolve_artifact_path_preserves_external_artifact(tmp_path: Path) -> None:
    touchstone = tmp_path / "ansys_export" / "job-001.s2p"

    assert _resolve_artifact_path(tmp_path / "queue", str(touchstone)) == touchstone
