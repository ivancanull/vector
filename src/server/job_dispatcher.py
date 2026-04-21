from pathlib import Path

from .queue_manager import QueueManager


class JobDispatcher:
    """Moves jobs from pending to running for worker pickup."""

    def __init__(self, queue: QueueManager):
        self.queue = queue

    def dispatch_next(self) -> Path | None:
        pending_jobs = self.queue.list_pending()
        if not pending_jobs:
            return None

        job_file = pending_jobs[0]
        target = self.queue.running / job_file.name
        self.queue.running.mkdir(parents=True, exist_ok=True)
        job_file.rename(target)
        return target
