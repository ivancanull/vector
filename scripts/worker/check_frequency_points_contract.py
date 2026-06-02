from __future__ import annotations

import inspect
import math
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class ContractCheckError(RuntimeError):
    pass


def _require_parameters(callable_obj: Any, expected: set[str], label: str) -> inspect.Signature:
    signature = inspect.signature(callable_obj)
    missing = expected.difference(signature.parameters)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ContractCheckError(f"{label} is missing parameters: {missing_text}")
    return signature


def _import_aedt_api() -> tuple[str, str, type, type]:
    try:
        import pyaedt  # type: ignore

        from pyaedt.modules.SolveSetup import SetupHFSS  # type: ignore

        return "pyaedt", getattr(pyaedt, "__version__", "unknown"), pyaedt.Hfss, SetupHFSS
    except ModuleNotFoundError:
        pass

    try:
        import ansys.aedt.core as aedt  # type: ignore

        from ansys.aedt.core.modules.solve_setup import SetupHFSS  # type: ignore

        return "ansys.aedt.core", getattr(aedt, "__version__", "unknown"), aedt.Hfss, SetupHFSS
    except ModuleNotFoundError as exc:
        raise ContractCheckError("Neither pyaedt nor ansys.aedt.core is importable") from exc


def check_installed_aedt_contract() -> list[str]:
    package_name, version, Hfss, SetupHFSS = _import_aedt_api()
    lines = [f"AEDT package: {package_name} {version}"]

    export_signature = _require_parameters(
        Hfss.export_touchstone,
        {"setup", "sweep", "output_file"},
        "Hfss.export_touchstone",
    )
    lines.append(f"Hfss.export_touchstone signature: {export_signature}")

    single_point_signature = _require_parameters(
        SetupHFSS.create_single_point_sweep,
        {"name", "unit", "freq", "save_single_field", "save_fields"},
        "SetupHFSS.create_single_point_sweep",
    )
    lines.append(f"SetupHFSS.create_single_point_sweep signature: {single_point_signature}")

    count_signature = _require_parameters(
        SetupHFSS.create_frequency_sweep,
        {"start_frequency", "stop_frequency", "num_of_freq_points", "name", "sweep_type"},
        "SetupHFSS.create_frequency_sweep",
    )
    lines.append(f"SetupHFSS.create_frequency_sweep signature: {count_signature}")
    return lines


def check_job_contract() -> list[str]:
    from common.io_utils import read_json
    from common.job_types import SimulationJob
    from server.queue_manager import QueueManager
    from worker.schemas import ArtifactOptions, JobRequest

    job = JobRequest(
        job_id="uaw_frequency_contract",
        template_name="bclf_template_v1",
        variables={"L": "7.8mm", "W": "1.0mm"},
        output_filename="uaw_frequency_contract.s2p",
        metrics_to_extract=["max_s11_db", "min_s21_db"],
        artifact_options=ArtifactOptions(export_touchstone=True, export_csv=False),
        submit_time="2026-06-02T00:00:00Z",
        frequency_points_ghz=[5.0, 3.5, 5.0, 4.2],
    )
    expected_points = [3.5, 4.2, 5.0]
    if job.frequency_points_ghz != expected_points:
        raise ContractCheckError(
            f"JobRequest normalized {job.frequency_points_ghz}, expected {expected_points}"
        )

    for invalid_points in ([3.5, 0.0], [3.5, math.inf], []):
        try:
            JobRequest(
                job_id="invalid_frequency_contract",
                template_name="bclf_template_v1",
                variables={"L": "7.8mm"},
                output_filename="invalid_frequency_contract.s2p",
                metrics_to_extract=["max_s11_db"],
                submit_time="2026-06-02T00:00:00Z",
                frequency_points_ghz=list(invalid_points),
            )
        except ValueError:
            continue
        raise ContractCheckError(f"Invalid frequency_points_ghz accepted: {invalid_points}")

    with tempfile.TemporaryDirectory() as tmp:
        queue = QueueManager(Path(tmp) / "queue")
        path = queue.enqueue(
            SimulationJob(
                job_id="uaw_frequency_contract",
                template_name="bclf_template_v1",
                variables={"L": "7.8mm", "W": "1.0mm"},
                output_filename="uaw_frequency_contract.s2p",
                metrics_to_extract=["max_s11_db", "min_s21_db"],
                submit_time="2026-06-02T00:00:00Z",
                frequency_points_ghz=expected_points,
            )
        )
        payload = read_json(path)
        if payload["frequency_points_ghz"] != expected_points:
            raise ContractCheckError("QueueManager did not preserve frequency_points_ghz")

    return [
        f"JobRequest normalizes frequency_points_ghz to {expected_points}",
        "JobRequest rejects empty, non-positive, and non-finite frequency points",
        "QueueManager preserves frequency_points_ghz in canonical worker JSON",
    ]


class SpySweep:
    def __init__(self, name: str, props: dict[str, Any]) -> None:
        self.name = name
        self.props = props


class SpySetup:
    def __init__(self, name: str) -> None:
        self.name = name
        self.sweep: SpySweep | None = None
        self.single_point_kwargs: dict[str, Any] | None = None
        self.count_kwargs: dict[str, Any] | None = None

    def get_sweep(self, name: str | None = None) -> bool:
        return False

    def delete_sweep(self, name: str) -> bool:
        return True

    def create_single_point_sweep(self, **kwargs: Any) -> SpySweep:
        self.single_point_kwargs = dict(kwargs)
        self.sweep = SpySweep(kwargs.get("name", "Sweep"), {"kind": "single_point"})
        return self.sweep

    def create_frequency_sweep(self, **kwargs: Any) -> SpySweep:
        self.count_kwargs = dict(kwargs)
        self.sweep = SpySweep(kwargs.get("name", "Sweep"), {"kind": "frequency_count"})
        return self.sweep


class SpyHfss:
    def __init__(self) -> None:
        self.setups: list[SpySetup] = []

    def create_setup(self, name: str) -> SpySetup:
        setup = SpySetup(name)
        self.setups.append(setup)
        return setup


def check_setup_designer_contract() -> list[str]:
    from worker.schemas import HfssSweepConfig
    from worker.setup_designer import HfssSetupDesigner

    designer = HfssSetupDesigner()
    sweep_config = HfssSweepConfig(
        sweep_type="interpolating",
        sweep_data="LINC 3.5GHz 8.7GHz 401",
    )

    hfss_override = SpyHfss()
    designer.ensure_setup(
        hfss_override,
        setup_name="Setup1",
        sweep_name="Sweep1",
        sweep_config=sweep_config,
        frequency_points_ghz=[5.0, 3.5, 5.0, 4.2],
    )
    override_kwargs = hfss_override.setups[0].single_point_kwargs
    expected_override = {
        "name": "Sweep1",
        "unit": "GHz",
        "freq": [3.5, 4.2, 5.0],
        "save_single_field": False,
        "save_fields": False,
    }
    if override_kwargs != expected_override:
        raise ContractCheckError(
            f"Single-point sweep kwargs {override_kwargs}, expected {expected_override}"
        )

    hfss_template = SpyHfss()
    designer.ensure_setup(
        hfss_template,
        setup_name="Setup1",
        sweep_name="Sweep1",
        sweep_config=sweep_config,
        frequency_points_ghz=None,
    )
    count_kwargs = hfss_template.setups[0].count_kwargs
    expected_count = {
        "start_frequency": "3.5GHz",
        "stop_frequency": "8.7GHz",
        "num_of_freq_points": 401,
        "name": "Sweep1",
        "sweep_type": "Interpolating",
    }
    if count_kwargs != expected_count:
        raise ContractCheckError(f"Template sweep kwargs {count_kwargs}, expected {expected_count}")

    return [
        "HfssSetupDesigner uses create_single_point_sweep for job frequency overrides",
        "HfssSetupDesigner keeps template LINC sweep when no job frequency override is present",
    ]


def main() -> int:
    print("UAW frequency_points_ghz contract check")
    print(f"Python executable: {sys.executable}")
    checks = [
        check_installed_aedt_contract,
        check_job_contract,
        check_setup_designer_contract,
    ]
    for check in checks:
        for line in check():
            print(f"PASS: {line}")
    print("PASS: frequency_points_ghz contract is compatible with local PyAEDT/AEDT API")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
