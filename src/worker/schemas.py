from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    design_name: str | None = None
    template_name: str = Field(min_length=1)
    setup_name: str | None = None
    sweep_name: str | None = None
    frequency_points_ghz: list[float] | None = None
    variables: dict[str, str]
    output_filename: str = Field(min_length=1)
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

    @field_validator("output_filename")
    @classmethod
    def validate_output_filename(cls, value: str) -> str:
        suffix = Path(value).suffix.lower()
        if not (suffix.startswith(".s") and suffix.endswith("p")):
            raise ValueError("output_filename must be a Touchstone file name like .s2p")
        return value

    @field_validator("frequency_points_ghz")
    @classmethod
    def validate_frequency_points(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("frequency_points_ghz must be non-empty when provided")
        points = sorted({float(item) for item in value})
        for point in points:
            if not math.isfinite(point) or point <= 0:
                raise ValueError("frequency_points_ghz values must be finite and > 0")
        return points


class TemplateConfig(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: Path
    design_name: str
    setup_name: str
    sweep_name: str
    hfss_sweep: "HfssSweepConfig"


class HfssSweepConfig(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    sweep_type: Literal["interpolating", "discrete", "fast"]
    sweep_data: str | None = Field(default=None, min_length=1)
    start_freq: str | None = Field(default=None, min_length=1)
    stop_freq: str | None = Field(default=None, min_length=1)
    point_count: int | None = Field(default=None, gt=1)

    @field_validator("start_freq", "stop_freq")
    @classmethod
    def validate_freq_units(cls, value: str | None) -> str | None:
        if value is not None and not has_explicit_unit(value):
            raise ValueError("Frequency values must include explicit units")
        return value

    @field_validator("sweep_data")
    @classmethod
    def validate_sweep_data(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("sweep_data must be non-empty when provided")
        return value

    @model_validator(mode="after")
    def validate_definition(self) -> "HfssSweepConfig":
        has_sweep_data = self.sweep_data is not None
        has_range = (
            self.start_freq is not None
            or self.stop_freq is not None
            or self.point_count is not None
        )
        if has_sweep_data == has_range:
            raise ValueError(
                "Provide either sweep_data or start_freq/stop_freq/point_count"
            )
        if has_range and (
            self.start_freq is None or self.stop_freq is None or self.point_count is None
        ):
            raise ValueError(
                "start_freq, stop_freq, and point_count are required together"
            )
        return self


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
