from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(slots=True)
class SimulationJob:
    """Single simulation task exchanged between Linux master and Windows worker."""

    job_id: str
    design_type: str
    parameters: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
