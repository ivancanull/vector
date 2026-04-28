import json
from pathlib import Path

import pytest

from worker.job_loader import load_job
from worker.schemas import TemplateConfig, TemplatesConfig
from worker.template_registry import TemplateRegistry
from worker.utils import WorkerError


def _registry(tmp_path: Path) -> TemplateRegistry:
    project_file = tmp_path / "template.aedt"
    project_file.write_text("placeholder", encoding="utf-8")
    templates = TemplatesConfig(
        templates={
            "bclf_template_v1": TemplateConfig(
                project_path=project_file,
                design_name="BCLF_Design",
                setup_name="Setup1",
                sweep_name="Sweep1",
                hfss_sweep={
                    "sweep_type": "interpolating",
                    "sweep_data": "LINC 0GHz 4GHz 201",
                },
            )
        }
    )
    return TemplateRegistry(templates)


def _valid_payload() -> dict:
    return {
        "job_id": "bclf_000001",
        "design_name": "BCLF",
        "template_name": "bclf_template_v1",
        "setup_name": "Setup1",
        "sweep_name": "Sweep1",
        "variables": {"L": "7.8mm", "W": "1.0mm"},
        "output_filename": "bclf_000001.s2p",
        "metrics_to_extract": ["max_s11_db", "min_s21_db"],
        "artifact_options": {"export_touchstone": True, "export_csv": True},
        "submit_time": "2026-04-21T12:00:00Z",
        "retry_count": 0,
    }


def test_valid_job_loads(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    path = tmp_path / "job.json"
    path.write_text(json.dumps(_valid_payload()), encoding="utf-8")
    job = load_job(path, registry)
    assert job.job_id == "bclf_000001"


def test_missing_required_field_fails(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    payload = _valid_payload()
    del payload["job_id"]
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkerError) as exc:
        load_job(path, registry)
    assert exc.value.failure_class == "invalid_job_json"


def test_bad_unit_fails(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    payload = _valid_payload()
    payload["variables"]["L"] = "7.8"
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkerError) as exc:
        load_job(path, registry)
    assert exc.value.failure_class == "invalid_variable"


def test_unknown_metric_fails(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    payload = _valid_payload()
    payload["metrics_to_extract"] = ["foo_metric"]
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkerError):
        load_job(path, registry)


def test_missing_output_filename_fails(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    payload = _valid_payload()
    del payload["output_filename"]
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkerError) as exc:
        load_job(path, registry)
    assert exc.value.failure_class == "invalid_job_json"
