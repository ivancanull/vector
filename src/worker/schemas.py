from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.units import has_explicit_unit

SUPPORTED_METRICS = {
    "center_freq",
    "bandwidth",
    "max_s11_db",
    "min_s11_db",
    "min_s21_db",
    "max_s21_db",
    "q_peak",
    "inductance_at_freq",
    "srf",
    "passband_ripple",
    "stopband_attenuation",
}


class ArtifactOptions(BaseModel):
    export_touchstone: bool | None = None
    export_csv: bool | None = None


class MetricRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_metric_name(cls, value: str) -> str:
        if value not in SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric: {value}")
        return value


class JobRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    job_id: str = Field(min_length=1)
    design_name: str = Field(min_length=1)
    template_name: str = Field(min_length=1)
    setup_name: str | None = None
    sweep_name: str | None = None
    variables: dict[str, str]
    metrics_to_extract: list[str]
    artifact_options: ArtifactOptions = Field(default_factory=ArtifactOptions)
    submit_time: datetime
    retry_count: int = Field(default=0, ge=0)

    @field_validator("variables")
    @classmethod
    def validate_variables(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("variables must be non-empty")
        for var_name, var_value in value.items():
            if not isinstance(var_value, str):
                raise ValueError(f"Variable {var_name} must be a string")
            if not has_explicit_unit(var_value):
                raise ValueError(f"Variable {var_name} must include explicit units")
        return value

    @field_validator("metrics_to_extract")
    @classmethod
    def validate_metrics(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("metrics_to_extract must contain at least one metric")
        for metric in value:
            MetricRequest(name=metric)
        return value


class TemplateConfig(BaseModel):
    project_path: Path
    design_name: str
    setup_name: str
    sweep_name: str


class TemplatesConfig(BaseModel):
    templates: dict[str, TemplateConfig]


class WorkerConfig(BaseModel):
    worker_name: str
    queue_root: Path
    templates_yaml: Path
    poll_interval_sec: int = Field(gt=0)
    claim_retry_sec: int = Field(gt=0)
    results_dir: Path
    logs_dir: Path
    non_graphical: bool = True
    new_desktop: bool = False
    close_desktop_after_job: bool = False
    restart_aedt_every_n_jobs: int = Field(default=0, ge=0)
    job_timeout_sec: int = Field(default=7200, gt=0)
    max_retry_count: int = Field(default=2, ge=0)
    default_export_touchstone: bool = True
    default_export_csv: bool = True
    preserve_local_project_copy: bool = False


class ArtifactPaths(BaseModel):
    touchstone: str | None = None
    csv: str | None = None
    log: str | None = None


class JobResult(BaseModel):
    job_id: str
    status: Literal["done", "failed"]
    worker_name: str
    start_time: datetime
    finish_time: datetime
    metrics: dict[str, float] | None = None
    artifacts: ArtifactPaths
    error_message: str | None = None


class FailureResult(JobResult):
    status: Literal["failed"] = "failed"
    metrics: None = None
