from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from worker.spiral_template_builder import (
    PyAEDTSpiralTemplateGenerator,
    SpiralGeometry,
    SpiralTemplateBuilder,
    SpiralTemplateDeviceDefinition,
    SpiralTouchstoneMetricsExtractor,
    estimate_square_spiral,
    square_spiral_outer_diameter_um,
)


DEVICE_DEFINITION = Path("configs/spiral_inductor_template_device_v1.yaml")


def test_spiral_definition_loads_required_variables_and_nominal_estimate() -> None:
    definition = SpiralTemplateDeviceDefinition.load(DEVICE_DEFINITION)

    assert set(("Din", "W", "S", "H")).issubset(definition.variables)
    assert definition.nominal_variables() == {
        "Din": "60um",
        "W": "15um",
        "S": "1.5um",
        "H": "2um",
    }

    estimate = definition.nominal_estimate()
    assert estimate.dout_um == pytest.approx(156.0)
    assert estimate.davg_um == pytest.approx(108.0)
    assert estimate.inductance_nh == pytest.approx(1.2752442113604712)
    assert estimate.reactance_ohm_at_f0 == pytest.approx(40.06297845942966)


def test_square_spiral_estimator_matches_skill_formula() -> None:
    estimate = estimate_square_spiral(din_um=60, w_um=15, s_um=1.5, turns=3, f0_ghz=5)

    assert square_spiral_outer_diameter_um(din_um=60, w_um=15, s_um=1.5, turns=3) == pytest.approx(156)
    assert estimate.inductance_nh == pytest.approx(1.2752442113604712)
    assert estimate.reactance_ohm_at_f0 == pytest.approx(40.06297845942966)


def test_spiral_geometry_uses_parameterized_expressions() -> None:
    definition = SpiralTemplateDeviceDefinition.load(DEVICE_DEFINITION)
    geometry = SpiralGeometry(definition)

    assert definition.max_outer_diameter_um() == pytest.approx(186.0)
    points = geometry.centerline_points()
    serialized = " ".join(coord for point in points for coord in point)

    assert "Din/2" in serialized
    assert "W + S" in serialized
    assert "H/2" in serialized
    assert "60um" not in serialized
    assert "15um" not in serialized
    assert len(points) == 13


def test_spiral_builder_calls_fake_pyaedt_in_expected_order() -> None:
    definition = SpiralTemplateDeviceDefinition.load(DEVICE_DEFINITION)
    events: list[tuple] = []
    hfss = FakeHfss(events)

    SpiralTemplateBuilder(definition).build(hfss)

    variable_events = [event for event in events if event[0] == "variable"]
    assert [event[1] for event in variable_events] == ["Din", "W", "S", "H"]

    first_geometry_idx = next(
        i
        for i, event in enumerate(events)
        if event[0] == "create_box" and event[3].get("name") == "SpiralTraceSegment1"
    )
    last_variable_idx = max(i for i, event in enumerate(events) if event[0] == "variable")
    assert last_variable_idx < first_geometry_idx

    trace_event = events[first_geometry_idx]
    assert trace_event[3]["material"] == "copper"
    assert "Din/2" in str(trace_event)

    port_events = [event for event in events if event[0] == "lumped_port"]
    assert [event[1]["name"] for event in port_events] == ["Port1", "Port2"]
    assert all(event[1]["create_port_sheet"] is False for event in port_events)
    for event in port_events:
        start, end = event[1]["integration_line"]
        assert start != end
        assert sum(1 for lhs, rhs in zip(start, end) if lhs != rhs) == 1
    assert any(event[0] == "radiation" for event in events)
    assert any(
        event[0] == "unite"
        and "SpiralTraceSegment1" in event[1]
        and "InnerTerminalPad" in event[1]
        and "InnerBridgeTrace" in event[1]
        for event in events
    )
    assert any(event[0] == "frequency_sweep" and event[1]["name"] == "Sweep" for event in events)


def test_spiral_builder_prefers_public_lumped_port_api() -> None:
    definition = SpiralTemplateDeviceDefinition.load(DEVICE_DEFINITION)
    events: list[tuple] = []

    class FakeHfssWithPrivateBoundary(FakeHfss):
        def _create_boundary(self, *args, **kwargs):
            events.append(("private_boundary", args, kwargs))
            return "private"

    SpiralTemplateBuilder(definition).build(FakeHfssWithPrivateBoundary(events))

    assert [event[1]["name"] for event in events if event[0] == "lumped_port"] == ["Port1", "Port2"]
    assert not any(event[0] == "private_boundary" for event in events)


def test_spiral_generator_saves_project_with_fake_factory(tmp_path: Path) -> None:
    definition = SpiralTemplateDeviceDefinition.load(DEVICE_DEFINITION)
    definition = replace(
        definition,
        template_output=replace(
            definition.template_output,
            project_path=tmp_path / "spiral_inductor_template_v1.aedt",
        ),
    )
    events: list[tuple] = []
    fake_hfss = FakeHfss(events)
    factory = FakeHfssFactory(fake_hfss)

    generated = PyAEDTSpiralTemplateGenerator(hfss_factory=factory).generate(
        definition,
        overwrite=True,
    )

    assert generated == tmp_path / "spiral_inductor_template_v1.aedt"
    assert events[-2] == ("save_project", str(tmp_path / "spiral_inductor_template_v1.aedt"))
    assert events[-1] == ("release_desktop", False)


def test_spiral_generator_overwrite_removes_existing_project_artifacts(tmp_path: Path) -> None:
    definition = SpiralTemplateDeviceDefinition.load(DEVICE_DEFINITION)
    project_path = tmp_path / "spiral_inductor_template_v1.aedt"
    definition = replace(
        definition,
        template_output=replace(definition.template_output, project_path=project_path),
    )
    project_path.write_text("old", encoding="utf-8")
    project_path.with_name(project_path.name + ".auto").write_text("old", encoding="utf-8")
    project_path.with_name(project_path.name + ".lock").write_text("old", encoding="utf-8")
    project_path.with_name(project_path.name + "results").mkdir()
    fake_hfss = FakeHfss([])

    PyAEDTSpiralTemplateGenerator(hfss_factory=FakeHfssFactory(fake_hfss)).generate(
        definition,
        overwrite=True,
    )

    assert not project_path.with_name(project_path.name + ".auto").exists()
    assert not project_path.with_name(project_path.name + ".lock").exists()
    assert not project_path.with_name(project_path.name + "results").exists()


def test_generate_cli_dry_run_does_not_import_pyaedt() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/worker/generate_spiral_template.py",
            "--device-definition",
            str(DEVICE_DEFINITION),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    out = result.stdout
    assert '"template": "spiral_inductor_t3_template_v1"' in out
    assert '"inductance_nh": 1.2752442113604712' in out


def test_touchstone_debug_extractor_returns_nominal_metrics(tmp_path: Path) -> None:
    pytest.importorskip("skrf")
    touchstone = tmp_path / "simple.s2p"
    touchstone.write_text(
        "\n".join(
            [
                "# GHZ S RI R 50",
                "1 0 0.1 0.01 0 0.01 0 0 0.1",
                "5 0 0.2 0.02 0 0.02 0 0 0.2",
                "10 0 -0.1 0.02 0 0.02 0 0 -0.1",
            ]
        ),
        encoding="utf-8",
    )

    metrics = SpiralTouchstoneMetricsExtractor().extract(touchstone, target_ghz=5)

    assert metrics.freq_range_ghz == pytest.approx((1.0, 10.0))
    assert metrics.l_at_target_nh is not None
    assert metrics.q_at_target is not None
    assert metrics.srf_ghz is not None
    assert metrics.s21_at_target_db is not None


class FakeMaterial:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeMaterials:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events
        self.materials: dict[str, FakeMaterial] = {}

    def checkifmaterialexists(self, name: str) -> bool:
        return name in self.materials

    def add_material(self, name: str) -> FakeMaterial:
        self.events.append(("material", name))
        material = FakeMaterial(name)
        self.materials[name] = material
        return material

    def __getitem__(self, name: str) -> FakeMaterial:
        return self.materials[name]


class FakeModeler:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events
        self.model_units = None

    def create_box(self, origin, sizes, **kwargs):
        self.events.append(("create_box", origin, sizes, kwargs))
        return kwargs.get("name", "Box")

    def create_polyline(self, points, **kwargs):
        self.events.append(("create_polyline", points, kwargs))
        return kwargs.get("name", "Polyline")

    def create_rectangle(self, orientation, origin, sizes, **kwargs):
        self.events.append(("create_rectangle", orientation, origin, sizes, kwargs))
        return kwargs.get("name", "Rectangle")

    def unite(self, assignments):
        self.events.append(("unite", assignments))
        return True


class FakeSweep:
    pass


class FakeSetup:
    def __init__(self, name: str, events: list[tuple]) -> None:
        self.name = name
        self.events = events

    def get_sweep(self, _name: str):
        return None

    def create_frequency_sweep(self, **kwargs):
        self.events.append(("frequency_sweep", kwargs))
        return FakeSweep()

    def create_linear_step_sweep(self, **kwargs):
        self.events.append(("linear_step_sweep", kwargs))
        return FakeSweep()


class FakeHfss(dict):
    def __init__(self, events: list[tuple]) -> None:
        super().__init__()
        self.events = events
        self.modeler = FakeModeler(events)
        self.materials = FakeMaterials(events)
        self.setups: list[FakeSetup] = []
        self.solution_type = "DrivenModal"

    def __setitem__(self, key, value) -> None:
        self.events.append(("variable", key, value))
        super().__setitem__(key, value)

    def create_setup(self, name: str) -> FakeSetup:
        self.events.append(("create_setup", name))
        setup = FakeSetup(name, self.events)
        self.setups.append(setup)
        return setup

    def lumped_port(self, **kwargs):
        self.events.append(("lumped_port", kwargs))
        return kwargs["name"]

    def assign_radiation_boundary_to_objects(self, assignment, **kwargs):
        self.events.append(("radiation", assignment, kwargs))
        return kwargs.get("name", "Rad1")

    def assign_perfecte_to_sheets(self, assignment, **kwargs):
        self.events.append(("perfect_e", assignment, kwargs))
        return kwargs.get("name", "PerfE_Ground")

    def save_project(self, path: str) -> None:
        self.events.append(("save_project", path))

    def release_desktop(self, *, close_projects: bool, close_desktop: bool) -> None:
        assert close_projects is False
        self.events.append(("release_desktop", close_desktop))


class FakeHfssFactory:
    def __init__(self, hfss: FakeHfss) -> None:
        self.hfss = hfss

    def open(self, definition, *, non_graphical: bool, new_desktop: bool):
        assert definition.template_output.template_name == "spiral_inductor_t3_template_v1"
        assert non_graphical is True
        assert new_desktop is False
        return self.hfss
