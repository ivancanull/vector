from pathlib import Path

from src.common.job_types import JobStatus, SimulationJob
from src.common.io_utils import read_json, write_json


class QueueManager:
    def __init__(self, queue_root: str | Path = "queue"):
        self.root = Path(queue_root)
        self.pending = self.root / JobStatus.PENDING.value
        self.running = self.root / JobStatus.RUNNING.value
        self.done = self.root / JobStatus.DONE.value
        self.failed = self.root / JobStatus.FAILED.value
        self.results = self.root / "results"

    def enqueue(self, job: SimulationJob) -> Path:
        path = self.pending / f"{job.job_id}.json"
        write_json(path, {
            "job_id": job.job_id,
            "design_type": job.design_type,
            "parameters": job.parameters,
            "status": job.status.value,
            "metadata": job.metadata,
        })
        return path

    def list_pending(self) -> list[Path]:
        self.pending.mkdir(parents=True, exist_ok=True)
        return sorted(self.pending.glob("*.json"))

    def load_job(self, path: str | Path) -> SimulationJob:
        payload = read_json(path)
        return SimulationJob(
            job_id=payload["job_id"],
            design_type=payload["design_type"],
            parameters=payload["parameters"],
            status=JobStatus(payload.get("status", JobStatus.PENDING.value)),
            metadata=payload.get("metadata", {}),
        )
