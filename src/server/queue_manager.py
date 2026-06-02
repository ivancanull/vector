from pathlib import Path

from common.io_utils import read_json, write_json
from common.job_types import BatchSimulationJob, JobStatus, SimulationJob


class QueueManager:
    def __init__(self, queue_root: str | Path = "queue"):
        self.root = Path(queue_root)
        self.pending = self.root / JobStatus.PENDING.value
        self.running = self.root / JobStatus.RUNNING.value
        self.done = self.root / JobStatus.DONE.value
        self.failed = self.root / JobStatus.FAILED.value
        self.results = self.root / "results"

    def _write_worker_job_json(self, job: SimulationJob, path: Path) -> None:
        write_json(
            path,
            {
                "job_id": job.job_id,
                "template_name": job.template_name,
                "variables": job.variables,
                "output_filename": job.output_filename,
                "submit_time": job.submit_time,
                "metrics_to_extract": job.metrics_to_extract,
                "design_name": job.design_name,
                "setup_name": job.setup_name,
                "sweep_name": job.sweep_name,
                "frequency_points_ghz": job.frequency_points_ghz,
                "artifact_options": job.artifact_options,
                "retry_count": job.retry_count,
            },
        )

    def _validate_worker_job(self, job: SimulationJob) -> None:
        if not job.job_id:
            raise ValueError("job_id must be non-empty")
        if not job.template_name:
            raise ValueError("template_name must be non-empty")
        if not job.variables:
            raise ValueError("variables must be non-empty")
        if not job.output_filename:
            raise ValueError("output_filename must be non-empty")
        if not job.metrics_to_extract:
            raise ValueError("metrics_to_extract must be non-empty")

    def enqueue(self, job: SimulationJob) -> Path:
        self._validate_worker_job(job)
        path = self.pending / f"{job.job_id}.json"
        self._write_worker_job_json(job, path)
        return path

    def enqueue_batch(self, batch_job: BatchSimulationJob) -> list[Path]:
        if not batch_job.batch_id:
            raise ValueError("batch_id must be non-empty")
        if not batch_job.template_name:
            raise ValueError("template_name must be non-empty")
        if not batch_job.cases:
            raise ValueError("cases must be non-empty")
        if not batch_job.metrics_to_extract:
            raise ValueError("metrics_to_extract must be non-empty")

        emitted: list[Path] = []
        for idx, case in enumerate(batch_job.cases):
            suffix = case.case_id if case.case_id else str(idx)
            job_id = f"{batch_job.batch_id}_{suffix}"
            worker_job = SimulationJob(
                job_id=job_id,
                template_name=batch_job.template_name,
                variables=case.variables,
                output_filename=case.output_filename,
                metrics_to_extract=batch_job.metrics_to_extract,
                design_name=batch_job.design_name,
                setup_name=batch_job.setup_name,
                sweep_name=batch_job.sweep_name,
                frequency_points_ghz=batch_job.frequency_points_ghz,
                artifact_options=batch_job.artifact_options,
                retry_count=batch_job.retry_count,
            )
            self._validate_worker_job(worker_job)
            path = self.pending / f"{job_id}.json"
            self._write_worker_job_json(worker_job, path)
            emitted.append(path)
        return emitted

    def list_pending(self) -> list[Path]:
        self.pending.mkdir(parents=True, exist_ok=True)
        return sorted(self.pending.glob("*.json"))

    def load_job(self, path: str | Path) -> SimulationJob:
        payload = read_json(path)
        if "template_name" in payload and "variables" in payload and "output_filename" in payload:
            return SimulationJob(
                job_id=payload["job_id"],
                template_name=payload["template_name"],
                variables=payload["variables"],
                output_filename=payload["output_filename"],
                metrics_to_extract=payload.get("metrics_to_extract", []),
                submit_time=payload.get("submit_time", ""),
                design_name=payload.get("design_name"),
                setup_name=payload.get("setup_name"),
                sweep_name=payload.get("sweep_name"),
                frequency_points_ghz=payload.get("frequency_points_ghz"),
                artifact_options=payload.get("artifact_options", {}),
                retry_count=payload.get("retry_count", 0),
                status=JobStatus(payload.get("status", JobStatus.PENDING.value)),
            )

        return SimulationJob(
            job_id=payload["job_id"],
            template_name=payload.get("design_type", "legacy_template"),
            variables=payload.get("parameters", {}),
            output_filename=f"{payload['job_id']}.s2p",
            metrics_to_extract=[],
            submit_time="",
            design_name=payload.get("design_type"),
            status=JobStatus(payload.get("status", JobStatus.PENDING.value)),
        )
