from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from worker.schemas import JobRequest
from worker.template_registry import TemplateRegistry
from worker.utils import WorkerError


def _map_validation_to_failure(error_text: str) -> str:
    if "Variable" in error_text and "units" in error_text:
        return "invalid_variable"
    return "invalid_job_json"


def load_job(job_path: Path, template_registry: TemplateRegistry) -> JobRequest:
    try:
        with job_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise WorkerError("invalid_job_json", f"Invalid job JSON: {exc}") from exc

    try:
        job = JobRequest.model_validate(payload)
    except ValidationError as exc:
        failure_class = _map_validation_to_failure(str(exc))
        raise WorkerError(failure_class, f"Job validation failed: {exc}") from exc

    if not template_registry.has_template(job.template_name):
        raise WorkerError("unknown_template", f"Unknown template: {job.template_name}")

    return job
