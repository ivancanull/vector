from pathlib import Path
import os

import pytest

from worker.hfss_api import (
    HfssSession,
    HfssSessionFactory,
)
from worker.utils import WorkerError


def test_basic_task_set_variables() -> None:
    class FakeHfss(dict):
        pass

    hfss = FakeHfss()
    HfssSession(hfss).apply_variables(
        {
            "Din": "55um",
            "H": "2um",
            "S": "1.5um",
            "W": "15um",
        },
    )
    assert hfss["Din"] == "55um"
    assert hfss["H"] == "2um"
    assert hfss["S"] == "1.5um"
    assert hfss["W"] == "15um"


def test_basic_task_analyze_setup_success() -> None:
    class FakeSetup:
        is_solved = True

    class FakeHfss:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def analyze_setup(self, name: str, *, blocking: bool):
            self.calls.append((name, blocking))
            return True

        def get_setup(self, name: str):
            assert name == "Setup1"
            return FakeSetup()

    hfss = FakeHfss()
    HfssSession(hfss).analyze("Setup1")

    assert hfss.calls == [("Setup1", True)]


def test_basic_task_analyze_setup_falls_back_to_setup_object() -> None:
    class FakeSetup:
        is_solved = True

        def analyze(self, *, blocking: bool):
            assert blocking is True
            return True

    class FakeHfss:
        def get_setup(self, name: str):
            assert name == "Setup1"
            return FakeSetup()

    HfssSession(FakeHfss()).analyze("Setup1")


def test_basic_task_analyze_setup_failure_raises_worker_error() -> None:
    class FakeHfss:
        def analyze_setup(self, _name: str, *, blocking: bool):
            assert blocking is True
            return False

    with pytest.raises(WorkerError) as exc:
        HfssSession(FakeHfss()).analyze("Setup1")
    assert exc.value.failure_class == "mesh_failure"


def test_basic_task_export_touchstone_uses_user_guide_signature(tmp_path: Path) -> None:
    out_path = tmp_path / "result.s2p"

    class FakeHfss:
        def export_touchstone(self, **kwargs):
            assert kwargs["setup"] == "Setup1"
            assert kwargs["sweep"] == "Sweep"
            assert kwargs["output_file"] == str(out_path)
            return str(out_path)

    exported = HfssSession(FakeHfss()).export_touchstone("Setup1", "Sweep", out_path)
    assert exported == out_path


def test_basic_task_export_touchstone_passes_absolute_output_file(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out_path = Path("results") / "result.s2p"
    expected_output = out_path.resolve()

    class FakeHfss:
        def export_touchstone(self, **kwargs):
            assert kwargs["output_file"] == str(expected_output)
            expected_output.write_text("s2p", encoding="utf-8")
            return str(expected_output)

    exported = HfssSession(FakeHfss()).export_touchstone("Setup1", "Sweep", out_path)

    assert exported == out_path


def test_basic_task_open_hfss_session_supports_new_desktop_signature(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeHfss:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    factory = HfssSessionFactory()
    monkeypatch.setattr(factory, "import_hfss_class", lambda: FakeHfss)
    template = type(
        "Template",
        (),
        {"project_path": r"D:\ansys\spiral_inductor_example.aedt", "design_name": "HFSSDesign1"},
    )()
    cfg = type("Cfg", (), {"non_graphical": True, "new_desktop": False})()
    factory.open(template, cfg)
    assert captured["project"] == r"D:\ansys\spiral_inductor_example.aedt"
    assert captured["design"] == "HFSSDesign1"
    assert captured["non_graphical"] is True
    assert "new_desktop_session" in captured


def test_basic_task_open_hfss_session_retries_project_lock(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeHfss:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("Project is locked. Close or remove the lock before proceeding.")

    factory = HfssSessionFactory()
    monkeypatch.setattr(factory, "import_hfss_class", lambda: FakeHfss)
    template = type(
        "Template",
        (),
        {"project_path": r"D:\ansys\spiral_inductor_example.aedt", "design_name": "HFSSDesign1"},
    )()
    cfg = type("Cfg", (), {"non_graphical": True, "new_desktop": False})()
    factory.open(template, cfg)
    assert len(calls) == 2
    assert "remove_lock" not in calls[0]
    assert calls[1]["remove_lock"] is True


def test_basic_task_circuit_setup_sweep_definition_data() -> None:
    class FakeSetup:
        def __init__(self, name: str):
            self.name = name
            self.props = {"SweepDefinition": {"Data": ""}}

    class FakeCircuit:
        def create_setup(self, name: str):
            return FakeSetup(name)

    circuit = FakeCircuit()
    setup1 = circuit.create_setup("MyLNA")
    setup1.props["SweepDefinition"]["Data"] = "LINC 0GHz 4GHz 201"

    assert setup1.name == "MyLNA"
    assert setup1.props["SweepDefinition"]["Data"] == "LINC 0GHz 4GHz 201"


def test_example_live_spiral_inductor_load_and_create_setup() -> None:
    if os.environ.get("RUN_AEDT_LIVE_TESTS") != "1":
        pytest.skip("Set RUN_AEDT_LIVE_TESTS=1 to run live AEDT integration examples.")

    project_path = Path(r"D:\ansys\spiral_inductor_example.aedt")
    if not project_path.exists():
        pytest.skip(f"Live project file not found: {project_path}")

    core = pytest.importorskip("ansys.aedt.core")
    Hfss = core.Hfss

    def _list_setup_names(hfss_obj) -> list[str]:
        names = getattr(hfss_obj, "existing_analysis_setups", None)
        if isinstance(names, list):
            return [n for n in names if isinstance(n, str)]
        return [s.name for s in getattr(hfss_obj, "setups", []) if hasattr(s, "name")]

    def _delete_mylna_setups(hfss_obj) -> None:
        for setup_name in _list_setup_names(hfss_obj):
            if setup_name == "MyLNA" or setup_name.startswith("MyLNA_"):
                hfss_obj.delete_setup(setup_name)

    hfss = None
    created_setup_name: str | None = None
    try:
        hfss = Hfss(
            project=str(project_path),
            design="HFSSDesign1",
            non_graphical=True,
            new_desktop=False,
        )
        # Keep the project clean by removing stale tutorial setups from prior runs.
        _delete_mylna_setups(hfss)
        setup1 = hfss.create_setup("MyLNA")
        created_setup_name = setup1.name
        sweep_definition = setup1.props.get("SweepDefinition")
        if not isinstance(sweep_definition, dict):
            setup1.props["SweepDefinition"] = {"Data": "LINC 0GHz 4GHz 201"}
        else:
            setup1.props["SweepDefinition"]["Data"] = "LINC 0GHz 4GHz 201"

        assert setup1.name == "MyLNA" or setup1.name.startswith("MyLNA_")
        assert setup1.props["SweepDefinition"]["Data"] == "LINC 0GHz 4GHz 201"
    finally:
        if hfss is not None:
            if created_setup_name:
                try:
                    hfss.delete_setup(created_setup_name)
                except Exception:
                    pass
            hfss.release_desktop(close_projects=False, close_desktop=False)
