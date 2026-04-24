from __future__ import annotations

from pathlib import Path

import pytest

from worker.schemas import (
    ArtifactOptions,
    HfssSetupConfig,
    HfssSweepConfig,
    JobRequest,
    TemplatesConfig,
    WorkerConfig,
)
from worker.template_registry import TemplateRegistry
from worker.utils import WorkerError


class FakeSweep:
    def __init__(self, name: str):
        self.name = name
        self.props: dict[str, object] = {}
        self.updated = 0

    def update(self) -> None:
        self.updated += 1


class FakeSetup:
    def __init__(self, name: str):
        self.name = name
        self.props: dict[str, object] = {}
        self.sweeps: list[FakeSweep] = []
        self.updated = 0

    def enable_adaptive_setup_single(self, freq=None, max_passes=None, max_delta_s=None) -> bool:
        if freq is not None:
            self.props["Frequency"] = freq
        if max_passes is not None:
            self.props["MaximumPasses"] = max_passes
        if max_delta_s is not None:
            self.props["MaxDeltaS"] = max_delta_s
        return True

    def add_sweep(self, name: str, sweep_type: str | None = None) -> FakeSweep:
        sweep = FakeSweep(name)
        if sweep_type is not None:
            sweep.props["Type"] = sweep_type
        self.sweeps.append(sweep)
        return sweep

    def create_frequency_sweep(
        self,
        start_frequency=None,
        stop_frequency=None,
        num_of_freq_points=None,
        name: str | None = None,
        sweep_type: str = "Discrete",
    ) -> FakeSweep:
        sweep = self.add_sweep(name or "Sweep", sweep_type)
        sweep.props["RangeType"] = "LinearCount"
        sweep.props["RangeStart"] = start_frequency
        sweep.props["RangeEnd"] = stop_frequency
        sweep.props["RangeCount"] = num_of_freq_points
        return sweep

    def create_linear_step_sweep(
        self,
        start_frequency=None,
        stop_frequency=None,
        step_size=None,
        name: str | None = None,
        sweep_type: str = "Discrete",
    ) -> FakeSweep:
        sweep = self.add_sweep(name or "Sweep", sweep_type)
        sweep.props["RangeType"] = "LinearStep"
        sweep.props["RangeStart"] = start_frequency
        sweep.props["RangeEnd"] = stop_frequency
        sweep.props["RangeStep"] = step_size
        return sweep

    def get_sweep(self, name: str | None = None):
        if name is None:
            return self.sweeps[0] if self.sweeps else False
        for sweep in self.sweeps:
            if sweep.name == name:
                return sweep
        return False

    def delete_sweep(self, name: str) -> bool:
        for index, sweep in enumerate(self.sweeps):
            if sweep.name == name:
                del self.sweeps[index]
                return True
        return False

    def update(self) -> None:
        self.updated += 1


class FakeHfss:
    def __init__(self):
        self.setups: list[FakeSetup] = []

    def create_setup(self, name: str) -> FakeSetup:
        setup = FakeSetup(name)
        self.setups.append(setup)
        return setup


def _template_payload(project_file: Path) -> dict[str, object]:
    return {
        "project_path": project_file,
        "design_name": "HFSSDesign1",
        "setup_name": "Setup1",
        "sweep_name": "Sweep1",
        "hfss_setup": {
            "solution_frequency": "2.4GHz",
            "max_delta_s": 0.02,
            "maximum_passes": 20,
            "minimum_passes": 1,
            "minimum_converged_passes": 1,
            "percent_refinement": 30,
            "basis_order": 1,
        },
        "hfss_sweep": {
            "sweep_type": "interpolating",
            "sweep_data": "LINC 0GHz 4GHz 201",
        },
    }


def _hfss_setup_config() -> HfssSetupConfig:
    return HfssSetupConfig(
        solution_frequency="2.4GHz",
        max_delta_s=0.02,
        maximum_passes=20,
        minimum_passes=1,
        minimum_converged_passes=1,
        percent_refinement=30,
        basis_order=1,
    )


def _hfss_sweep_config() -> HfssSweepConfig:
    return HfssSweepConfig(
        sweep_type="interpolating",
        sweep_data="LINC 0GHz 4GHz 201",
    )


def _worker_config(tmp_path: Path) -> WorkerConfig:
    queue_root = tmp_path / "queue"
    results_dir = queue_root / "results"
    logs_dir = queue_root / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return WorkerConfig(
        worker_name="worker-a",
        queue_root=queue_root,
        templates_yaml=tmp_path / "templates.yaml",
        poll_interval_sec=10,
        claim_retry_sec=5,
        results_dir=results_dir,
        logs_dir=logs_dir,
        default_export_touchstone=True,
        default_export_csv=False,
    )


def _job() -> JobRequest:
    return JobRequest(
        job_id="job-001",
        template_name="si_template_v1",
        variables={"Din": "55um", "H": "2um", "S": "1.5um", "W": "15um"},
        output_filename="job-001.s2p",
        metrics_to_extract=["max_s11_db"],
        artifact_options=ArtifactOptions(export_touchstone=True, export_csv=False),
        submit_time="2026-04-24T08:00:00Z",
        retry_count=0,
        setup_name="JobSetup",
        sweep_name="JobSweep",
    )


def test_hfss_sweep_config_accepts_range_definition() -> None:
    cfg = HfssSweepConfig(
        sweep_type="discrete",
        start_freq="1GHz",
        stop_freq="4GHz",
        point_count=101,
    )
    assert cfg.start_freq == "1GHz"
    assert cfg.sweep_data is None


def test_hfss_sweep_config_rejects_mixed_definition() -> None:
    with pytest.raises(ValueError):
        HfssSweepConfig(
            sweep_type="fast",
            sweep_data="LINC 1GHz 4GHz 201",
            start_freq="1GHz",
            stop_freq="4GHz",
            point_count=101,
        )


def test_template_registry_carries_hfss_setup_config(tmp_path: Path) -> None:
    project_file = tmp_path / "template.aedt"
    project_file.write_text("placeholder", encoding="utf-8")
    templates = TemplatesConfig(
        templates={"si_template_v1": _template_payload(project_file)}
    )
    registry = TemplateRegistry(templates)

    resolved = registry.resolve_for_job(_job())

    assert resolved.setup_name == "JobSetup"
    assert resolved.sweep_name == "JobSweep"
    assert resolved.hfss_setup.solution_frequency == "2.4GHz"
    assert resolved.hfss_sweep.sweep_data == "LINC 0GHz 4GHz 201"


def test_setup_designer_creates_missing_setup_and_sweep() -> None:
    from worker.setup_designer import HfssSetupDesigner

    hfss = FakeHfss()
    designer = HfssSetupDesigner()
    resolved = designer.ensure_setup(
        hfss,
        setup_name="Setup1",
        sweep_name="Sweep1",
        setup_config=_hfss_setup_config(),
        sweep_config=_hfss_sweep_config(),
    )

    setup = hfss.setups[0]
    sweep = setup.sweeps[0]
    assert resolved.setup_name == "Setup1"
    assert resolved.sweep_name == "Sweep1"
    assert setup.props["Frequency"] == "2.4GHz"
    assert setup.props["MaximumPasses"] == 20
    assert sweep.props["Type"] == "Interpolating"
    assert sweep.props["RangeType"] == "LinearCount"
    assert sweep.props["RangeStart"] == "0GHz"
    assert sweep.props["RangeEnd"] == "4GHz"
    assert sweep.props["RangeCount"] == 201


def test_setup_designer_modifies_existing_setup_in_place() -> None:
    from worker.setup_designer import HfssSetupDesigner

    hfss = FakeHfss()
    setup = hfss.create_setup("Setup1")
    setup.props["Frequency"] = "1GHz"
    sweep = setup.add_sweep("Sweep1")
    sweep.props["RangeStart"] = "1GHz"

    designer = HfssSetupDesigner()
    designer.ensure_setup(
        hfss,
        setup_name="Setup1",
        sweep_name="Sweep1",
        setup_config=_hfss_setup_config(),
        sweep_config=_hfss_sweep_config(),
    )

    assert len(hfss.setups) == 1
    assert setup.props["Frequency"] == "2.4GHz"
    recreated = setup.get_sweep("Sweep1")
    assert recreated is not sweep
    assert recreated.props["RangeType"] == "LinearCount"
    assert recreated.props["RangeStart"] == "0GHz"
    assert recreated.props["RangeEnd"] == "4GHz"
    assert recreated.props["RangeCount"] == 201


def test_setup_designer_raises_when_setup_cannot_create_sweep() -> None:
    from worker.setup_designer import HfssSetupDesigner

    class NoSweepSetup:
        def __init__(self):
            self.name = "Setup1"
            self.props = {}
            self.sweeps = []

        def enable_adaptive_setup_single(self, **kwargs) -> bool:
            return True

        def get_sweep(self, name: str | None = None):
            return False

        def delete_sweep(self, name: str) -> bool:
            return False

        def update(self) -> None:
            return None

    class NoSweepHfss:
        def __init__(self):
            self.setups = [NoSweepSetup()]

    with pytest.raises(WorkerError) as exc:
        HfssSetupDesigner().ensure_setup(
            NoSweepHfss(),
            setup_name="Setup1",
            sweep_name="Sweep1",
            setup_config=_hfss_setup_config(),
            sweep_config=_hfss_sweep_config(),
        )
    assert exc.value.failure_class == "setup_definition_failure"


def test_execute_job_calls_setup_designer_before_analysis_and_export(
    tmp_path: Path,
) -> None:
    from worker.hfss_api import HfssArtifactPlanner, HfssJobExecutor

    events: list[str] = []
    project_file = tmp_path / "template.aedt"
    project_file.write_text("placeholder", encoding="utf-8")
    templates = TemplatesConfig(
        templates={"si_template_v1": _template_payload(project_file)}
    )
    template = TemplateRegistry(templates).resolve_for_job(_job())
    cfg = _worker_config(tmp_path)

    class FakeDesigner:
        def ensure_setup(self, hfss, **kwargs):
            events.append(f"ensure:{kwargs['setup_name']}:{kwargs['sweep_name']}")
            return type("Resolved", (), {"setup_name": kwargs["setup_name"], "sweep_name": kwargs["sweep_name"]})()

    class FakeSession:
        hfss = object()

        def apply_variables(self, variables):
            events.append("apply")

        def analyze(self, setup_name: str):
            events.append(f"analyze:{setup_name}")

        def export_touchstone(self, setup_name: str, sweep_name: str, out_path: Path) -> Path:
            events.append(f"export:{setup_name}:{sweep_name}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("s2p", encoding="utf-8")
            return out_path

        def close(self, close_desktop: bool) -> None:
            events.append("close")

    class FakeSessionFactory:
        def open(self, template_cfg, worker_cfg):
            return FakeSession()

    executor = HfssJobExecutor(
        session_factory=FakeSessionFactory(),
        setup_designer=FakeDesigner(),
        artifact_planner=HfssArtifactPlanner(),
    )
    result = executor.execute(_job(), template, cfg)

    assert result.status == "done"
    assert events == [
        "apply",
        "ensure:JobSetup:JobSweep",
        "analyze:JobSetup",
        "export:JobSetup:JobSweep",
        "close",
    ]
