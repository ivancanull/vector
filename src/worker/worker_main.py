from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path

from worker.hfss_api import execute_job
from worker.job_loader import load_job
from worker.logger import get_job_logger
from worker.metrics import compute_metrics
from worker.schemas import ArtifactPaths, JobRequest
from worker.state_manager import write_failure_result, write_success_result
from worker.template_registry import TemplateRegistry
from worker.utils import WorkerError, as_posix_rel, utc_now


def _extract_job_id_fallback(running_job_path: Path) -> str:
    try:
        with running_job_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and isinstance(payload.get("job_id"), str):
            return payload["job_id"]
    except Exception:
        pass
    return running_job_path.stem


def _copy_job_log(job_logger, job_log_src: Path, log_artifact_target: Path) -> None:
    for handler in job_logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    log_artifact_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(job_log_src, log_artifact_target)


def _persist_failure_result(
    *,
    queue_root: Path,
    worker_name: str,
    job_id: str,
    running_job_path: Path,
    start_time,
    finish_time,
    result_artifacts: ArtifactPaths,
    job_logger,
    job_log_src: Path,
    results_dir: Path,
    error_message: str,
) -> None:
    log_artifact_target = results_dir / f"{job_id}.log"
    _copy_job_log(job_logger, job_log_src, log_artifact_target)
    result_artifacts.log = as_posix_rel(log_artifact_target, queue_root)
    write_failure_result(
        queue_root=queue_root,
        worker_name=worker_name,
        job_id=job_id,
        running_job_path=running_job_path,
        start_time=start_time,
        finish_time=finish_time,
        artifacts=result_artifacts,
        error_message=error_message,
    )


def process_claimed_job(
    running_job_path: Path,
    cfg,
    template_registry: TemplateRegistry,
    global_logger,
) -> None:
    start_time = utc_now()
    job_id = _extract_job_id_fallback(running_job_path)
    job_logger, job_log_src = get_job_logger(cfg.logs_dir, job_id)
    job_logger.info("job claimed path=%s", running_job_path)

    result_artifacts = ArtifactPaths(
        touchstone=None,
        csv=None,
        log=as_posix_rel(cfg.results_dir / f"{job_id}.log", cfg.queue_root),
    )

    try:
        job: JobRequest = load_job(running_job_path, template_registry)
        job_id = job.job_id
        job_logger.info("job parsed job_id=%s template=%s", job.job_id, job.template_name)
        job_logger.info("raw job content=%s", running_job_path.read_text(encoding="utf-8"))
        job_logger.info("variables=%s", job.variables)
        template = template_registry.resolve_for_job(job)
        job_logger.info(
            "resolved template project=%s design=%s setup=%s sweep=%s",
            template.project_path,
            template.design_name,
            template.setup_name,
            template.sweep_name,
        )
        job_logger.info("solve started")
        execution_result = execute_job(job, template, cfg)
        job_logger.info("solve finished")
        result_artifacts = execution_result.artifacts
        job_logger.info("artifact export finished artifacts=%s", result_artifacts.model_dump())
        touchstone_path = (
            cfg.queue_root / result_artifacts.touchstone
            if result_artifacts.touchstone
            else None
        )
        if touchstone_path is None:
            raise WorkerError("metric_extraction_failure", "Touchstone artifact missing")
        metrics = compute_metrics(touchstone_path, job.metrics_to_extract, job_context={"job_id": job.job_id})
        job_logger.info("metric extraction finished metrics=%s", metrics)
        finish_time = utc_now()
        log_artifact_target = cfg.results_dir / f"{job_id}.log"
        _copy_job_log(job_logger, job_log_src, log_artifact_target)
        result_artifacts.log = as_posix_rel(log_artifact_target, cfg.queue_root)
        write_success_result(
            queue_root=cfg.queue_root,
            worker_name=cfg.worker_name,
            job=job,
            running_job_path=running_job_path,
            start_time=start_time,
            finish_time=finish_time,
            metrics=metrics,
            artifacts=result_artifacts,
        )
        global_logger.info("job marked done job_id=%s", job_id)
    except WorkerError as exc:
        finish_time = utc_now()
        job_logger.exception("job failed with WorkerError=%s", exc.failure_class)
        _persist_failure_result(
            queue_root=cfg.queue_root,
            worker_name=cfg.worker_name,
            job_id=job_id,
            running_job_path=running_job_path,
            start_time=start_time,
            finish_time=finish_time,
            result_artifacts=result_artifacts,
            job_logger=job_logger,
            job_log_src=job_log_src,
            results_dir=cfg.results_dir,
            error_message=exc.failure_class,
        )
        global_logger.info("job marked failed job_id=%s error=%s", job_id, exc.failure_class)
    except Exception:
        finish_time = utc_now()
        traceback_text = traceback.format_exc()
        job_logger.error("unexpected exception\n%s", traceback_text)
        _persist_failure_result(
            queue_root=cfg.queue_root,
            worker_name=cfg.worker_name,
            job_id=job_id,
            running_job_path=running_job_path,
            start_time=start_time,
            finish_time=finish_time,
            result_artifacts=result_artifacts,
            job_logger=job_logger,
            job_log_src=job_log_src,
            results_dir=cfg.results_dir,
            error_message="unexpected_exception",
        )
        global_logger.info("job marked failed job_id=%s error=unexpected_exception", job_id)
