from __future__ import annotations

import json
import logging
import shutil
import traceback
from pathlib import Path

from worker.hfss_api import HfssJobExecutor
from worker.job_loader import load_job
from worker.logger import get_job_logger
from worker.metrics import compute_metrics
from worker.schemas import ArtifactPaths, JobRequest, WorkerConfig
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


class MetricExtractor:
    def extract(self, touchstone_path: Path, metrics: list[str], *, job_id: str) -> dict[str, float]:
        return compute_metrics(touchstone_path, metrics, job_context={"job_id": job_id})


class ResultWriter:
    def write_success(
        self,
        *,
        cfg: WorkerConfig,
        job: JobRequest,
        running_job_path: Path,
        start_time,
        finish_time,
        metrics: dict[str, float],
        artifacts: ArtifactPaths,
    ) -> Path:
        return write_success_result(
            queue_root=cfg.queue_root,
            worker_name=cfg.worker_name,
            job=job,
            running_job_path=running_job_path,
            start_time=start_time,
            finish_time=finish_time,
            metrics=metrics,
            artifacts=artifacts,
        )

    def write_failure(
        self,
        *,
        cfg: WorkerConfig,
        job_id: str,
        running_job_path: Path,
        start_time,
        finish_time,
        artifacts: ArtifactPaths,
        error_message: str,
    ) -> Path:
        return write_failure_result(
            queue_root=cfg.queue_root,
            worker_name=cfg.worker_name,
            job_id=job_id,
            running_job_path=running_job_path,
            start_time=start_time,
            finish_time=finish_time,
            artifacts=artifacts,
            error_message=error_message,
        )


class JobLoggerFactory:
    def create(self, logs_dir: Path, job_id: str):
        return get_job_logger(logs_dir, job_id)


class ClaimedJobProcessor:
    def __init__(
        self,
        *,
        cfg: WorkerConfig,
        template_registry: TemplateRegistry,
        global_logger: logging.Logger,
        executor: HfssJobExecutor,
        metric_extractor: MetricExtractor | None = None,
        result_writer: ResultWriter | None = None,
        logger_factory: JobLoggerFactory | None = None,
    ) -> None:
        self.cfg = cfg
        self.template_registry = template_registry
        self.global_logger = global_logger
        self.executor = executor
        self.metric_extractor = metric_extractor or MetricExtractor()
        self.result_writer = result_writer or ResultWriter()
        self.logger_factory = logger_factory or JobLoggerFactory()

    @classmethod
    def default(
        cls,
        *,
        cfg: WorkerConfig,
        template_registry: TemplateRegistry,
        global_logger: logging.Logger,
    ) -> "ClaimedJobProcessor":
        return cls(
            cfg=cfg,
            template_registry=template_registry,
            global_logger=global_logger,
            executor=HfssJobExecutor.default(),
        )

    def process(self, running_job_path: Path) -> None:
        start_time = utc_now()
        job_id = _extract_job_id_fallback(running_job_path)
        job_logger, job_log_src = self.logger_factory.create(self.cfg.logs_dir, job_id)
        job_logger.info("job claimed path=%s", running_job_path)

        result_artifacts = ArtifactPaths(
            touchstone=None,
            csv=None,
            log=as_posix_rel(self.cfg.results_dir / f"{job_id}.log", self.cfg.queue_root),
        )

        try:
            job: JobRequest = load_job(running_job_path, self.template_registry)
            job_id = job.job_id
            job_logger.info("job parsed job_id=%s template=%s", job.job_id, job.template_name)
            job_logger.info("raw job content=%s", running_job_path.read_text(encoding="utf-8"))
            job_logger.info("variables=%s", job.variables)
            template = self.template_registry.resolve_for_job(job)
            job_logger.info(
                "resolved template project=%s design=%s setup=%s sweep=%s",
                template.project_path,
                template.design_name,
                template.setup_name,
                template.sweep_name,
            )
            job_logger.info("solve started")
            execution_result = self.executor.execute(job, template, self.cfg)
            job_logger.info("solve finished")
            result_artifacts = execution_result.artifacts
            job_logger.info("artifact export finished artifacts=%s", result_artifacts.model_dump())
            touchstone_path = (
                self.cfg.queue_root / result_artifacts.touchstone
                if result_artifacts.touchstone
                else None
            )
            if touchstone_path is None:
                raise WorkerError("metric_extraction_failure", "Touchstone artifact missing")
            metrics = self.metric_extractor.extract(touchstone_path, job.metrics_to_extract, job_id=job.job_id)
            job_logger.info("metric extraction finished metrics=%s", metrics)
            finish_time = utc_now()
            log_artifact_target = self.cfg.results_dir / f"{job_id}.log"
            _copy_job_log(job_logger, job_log_src, log_artifact_target)
            result_artifacts.log = as_posix_rel(log_artifact_target, self.cfg.queue_root)
            self.result_writer.write_success(
                cfg=self.cfg,
                job=job,
                running_job_path=running_job_path,
                start_time=start_time,
                finish_time=finish_time,
                metrics=metrics,
                artifacts=result_artifacts,
            )
            self.global_logger.info("job marked done job_id=%s", job_id)
        except WorkerError as exc:
            finish_time = utc_now()
            job_logger.exception("job failed with WorkerError=%s", exc.failure_class)
            self._persist_failure_result(
                job_id=job_id,
                running_job_path=running_job_path,
                start_time=start_time,
                finish_time=finish_time,
                result_artifacts=result_artifacts,
                job_logger=job_logger,
                job_log_src=job_log_src,
                error_message=exc.failure_class,
            )
            self.global_logger.info("job marked failed job_id=%s error=%s", job_id, exc.failure_class)
        except Exception:
            finish_time = utc_now()
            traceback_text = traceback.format_exc()
            job_logger.error("unexpected exception\n%s", traceback_text)
            self._persist_failure_result(
                job_id=job_id,
                running_job_path=running_job_path,
                start_time=start_time,
                finish_time=finish_time,
                result_artifacts=result_artifacts,
                job_logger=job_logger,
                job_log_src=job_log_src,
                error_message="unexpected_exception",
            )
            self.global_logger.info("job marked failed job_id=%s error=unexpected_exception", job_id)

    def _persist_failure_result(
        self,
        *,
        job_id: str,
        running_job_path: Path,
        start_time,
        finish_time,
        result_artifacts: ArtifactPaths,
        job_logger,
        job_log_src: Path,
        error_message: str,
    ) -> None:
        log_artifact_target = self.cfg.results_dir / f"{job_id}.log"
        _copy_job_log(job_logger, job_log_src, log_artifact_target)
        result_artifacts.log = as_posix_rel(log_artifact_target, self.cfg.queue_root)
        self.result_writer.write_failure(
            cfg=self.cfg,
            job_id=job_id,
            running_job_path=running_job_path,
            start_time=start_time,
            finish_time=finish_time,
            artifacts=result_artifacts,
            error_message=error_message,
        )
