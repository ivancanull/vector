from __future__ import annotations

from pathlib import Path

import pytest

from worker.hfss_api import HfssArtifactPlanner, HfssJobExecutor
from worker.schemas import HfssSweepConfig, JobRequest, WorkerConfig
from worker.setup_designer import ResolvedHfssSetup
from worker.template_copy import TemplateProjectCopyManager
from worker.template_registry import ResolvedTemplate
from worker.utils import WorkerError


def _worker_config(tmp_path: Path, *, copy: bool, preserve: bool = False) -> WorkerConfig:
    queue_root = tmp_path / "queue"
    return WorkerConfig(
        worker_name="worker-copy",
        queue_root=queue_root,
        templates_yaml=tmp_path / "templates.yaml",
        poll_interval_sec=1,
        claim_retry_sec=1,
        results_dir=queue_root / "results",
        logs_dir=tmp_path / "logs",
        default_export_touchstone=True,
        default_export_csv=False,
        copy_template_per_job=copy,
        template_copy_root=tmp_path / "aedt_copies",
        preserve_local_project_copy=preserve,
    )


def _template(project_path: Path) -> ResolvedTemplate:
    return ResolvedTemplate(
        template_name="spiral_inductor_t3_template_v1",
        project_path=project_path,
        design_name="SpiralInductorT3",
        setup_name="Setup1",
        sweep_name="Sweep",
        hfss_sweep=HfssSweepConfig(
            sweep_type="interpolating",
            sweep_data="LINC 0.1GHz 30GHz 601",
        ),
    )


def _job() -> JobRequest:
    return JobRequest(
        job_id="spiral_job_001",
        template_name="spiral_inductor_t3_template_v1",
        variables={"Din": "60um", "W": "15um", "S": "1.5um", "H": "2um"},
        output_filename="spiral_job_001.s2p",
        submit_time="2026-06-04T08:00:00Z",
        metrics_to_extract=["max_s11_db"],
        artifact_options={"export_touchstone": True, "export_csv": False},
    )


def test_template_copy_disabled_uses_original_project(tmp_path: Path) -> None:
    project = tmp_path / "template.aedt"
    project.write_text("golden", encoding="utf-8")
    manager = TemplateProjectCopyManager()

    prepared = manager.prepare(
        _template(project),
        worker_cfg=_worker_config(tmp_path, copy=False),
        job_id="job",
    )

    assert prepared.template.project_path == project
    assert prepared.work_dir is None
    prepared.cleanup()
    assert project.exists()


def test_template_copy_enabled_copies_only_aedt_and_cleans_up(tmp_path: Path) -> None:
    project = tmp_path / "template.aedt"
    project.write_text("golden", encoding="utf-8")
    (tmp_path / "template.aedt.lock").write_text("lock", encoding="utf-8")
    (tmp_path / "template.aedt.auto").write_text("auto", encoding="utf-8")
    (tmp_path / "template.aedtresults").mkdir()
    manager = TemplateProjectCopyManager()

    prepared = manager.prepare(
        _template(project),
        worker_cfg=_worker_config(tmp_path, copy=True),
        job_id="spiral/job:001",
    )
    assert prepared.template.project_path != project
    assert prepared.template.project_path.read_text(encoding="utf-8") == "golden"
    assert not (prepared.work_dir / "template.aedt.lock").exists()
    assert not (prepared.work_dir / "template.aedt.auto").exists()
    assert not (prepared.work_dir / "template.aedtresults").exists()

    prepared.cleanup()

    assert project.exists()
    assert prepared.work_dir is not None
    assert not prepared.work_dir.exists()


def test_template_copy_preserve_keeps_job_copy(tmp_path: Path) -> None:
    project = tmp_path / "template.aedt"
    project.write_text("golden", encoding="utf-8")
    manager = TemplateProjectCopyManager()

    prepared = manager.prepare(
        _template(project),
        worker_cfg=_worker_config(tmp_path, copy=True, preserve=True),
        job_id="job001",
    )
    prepared.cleanup()

    assert prepared.work_dir is not None
    assert prepared.work_dir.exists()
    assert prepared.template.project_path.exists()


class _FakeSetupDesigner:
    def ensure_setup(self, _hfss, *, setup_name, sweep_name, sweep_config):
        return ResolvedHfssSetup(setup_name=setup_name, sweep_name=sweep_name)


class _FakeSession:
    hfss = object()

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.closed: tuple[bool, bool] | None = None

    def apply_variables(self, _variables):
        return None

    def analyze(self, _setup_name: str):
        if self.fail:
            raise WorkerError("mesh_failure", "mesh failed")

    def export_touchstone(self, _setup_name: str, _sweep_name: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("s2p", encoding="utf-8")
        return out_path

    def close(self, close_desktop: bool, close_projects: bool = False) -> None:
        self.closed = (close_desktop, close_projects)


class _FakeSessionFactory:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.opened_project_paths: list[Path] = []
        self.sessions: list[_FakeSession] = []

    def open(self, template_cfg, _worker_cfg):
        self.opened_project_paths.append(template_cfg.project_path)
        session = _FakeSession(fail=self.fail)
        self.sessions.append(session)
        return session


def test_hfss_executor_opens_job_copy_and_cleans_after_success(tmp_path: Path) -> None:
    project = tmp_path / "template.aedt"
    project.write_text("golden", encoding="utf-8")
    cfg = _worker_config(tmp_path, copy=True)
    factory = _FakeSessionFactory()
    executor = HfssJobExecutor(
        session_factory=factory,
        setup_designer=_FakeSetupDesigner(),
        artifact_planner=HfssArtifactPlanner(),
    )

    result = executor.execute(_job(), _template(project), cfg)

    assert result.status == "done"
    assert factory.opened_project_paths[0] != project
    assert not factory.opened_project_paths[0].parent.exists()
    assert factory.sessions[0].closed == (True, True)
    assert project.exists()


def test_hfss_executor_cleans_job_copy_after_failure(tmp_path: Path) -> None:
    project = tmp_path / "template.aedt"
    project.write_text("golden", encoding="utf-8")
    cfg = _worker_config(tmp_path, copy=True)
    factory = _FakeSessionFactory(fail=True)
    executor = HfssJobExecutor(
        session_factory=factory,
        setup_designer=_FakeSetupDesigner(),
        artifact_planner=HfssArtifactPlanner(),
    )

    with pytest.raises(WorkerError):
        executor.execute(_job(), _template(project), cfg)

    assert factory.opened_project_paths[0] != project
    assert not factory.opened_project_paths[0].parent.exists()
    assert factory.sessions[0].closed == (True, True)
    assert project.exists()
