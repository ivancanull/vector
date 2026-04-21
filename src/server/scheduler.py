from .db import ExperimentDB
from .job_dispatcher import JobDispatcher
from .optimizer import Optimizer
from .queue_manager import QueueManager
from .result_ingestor import ResultIngestor
from .trainer import Trainer


class Scheduler:
    def __init__(self, queue_root: str = "queue", db_path: str = "database/experiments.db"):
        self.queue = QueueManager(queue_root)
        self.db = ExperimentDB(db_path)
        self.optimizer = Optimizer()
        self.dispatcher = JobDispatcher(self.queue)
        self.ingestor = ResultIngestor(self.db, self.queue.results)
        self.trainer = Trainer()

    def bootstrap(self) -> None:
        self.db.init_schema()
        for directory in [
            self.queue.pending,
            self.queue.running,
            self.queue.done,
            self.queue.failed,
            self.queue.results,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
