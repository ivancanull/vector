from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ResultPayload:
    job_id: str
    metrics: dict[str, float]
    artifacts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DispatchRecord:
    job_id: str
    worker_id: str
    detail: dict[str, Any] = field(default_factory=dict)
