from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(slots=True)
class SimulationJob:
    """Single simulation task exchanged between Linux master and Windows worker."""

    job_id: str
    template_name: str
    variables: dict[str, str]
    output_filename: str
    metrics_to_extract: list[str] = field(default_factory=list)
    submit_time: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    design_name: str | None = None
    setup_name: str | None = None
    sweep_name: str | None = None
    frequency_points_ghz: list[float] | None = None
    artifact_options: dict[str, bool | None] = field(default_factory=dict)
    retry_count: int = 0
    status: JobStatus = JobStatus.PENDING


@dataclass(slots=True)
class BatchCase:
    case_id: str
    variables: dict[str, str]
    output_filename: str


@dataclass(slots=True)
class BatchSimulationJob:
    batch_id: str
    template_name: str
    cases: list[BatchCase]
    metrics_to_extract: list[str]
    design_name: str | None = None
    setup_name: str | None = None
    sweep_name: str | None = None
    frequency_points_ghz: list[float] | None = None
    artifact_options: dict[str, bool | None] = field(default_factory=dict)
    retry_count: int = 0
