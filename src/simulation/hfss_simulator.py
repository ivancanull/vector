from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

try:
    from ..complex_func import FunctionFromTouchstone
    from ..worker.hfss_api import HfssJobExecutor
    from ..worker.schemas import JobRequest, JobResult, WorkerConfig
    from ..worker.template_registry import ResolvedTemplate
    from ..worker.utils import WorkerError, utc_now
except ImportError:
    from src.complex_func import FunctionFromTouchstone
    from worker.hfss_api import HfssJobExecutor
    from worker.schemas import JobRequest, JobResult, WorkerConfig
    from worker.template_registry import ResolvedTemplate
    from worker.utils import WorkerError, utc_now

from .core import SimulationError, SimulationInput, SimulationOutput, Simulator


class HfssExecutor(Protocol):
    def execute(
        self,
        job: JobRequest,
        template_cfg: ResolvedTemplate,
        worker_cfg: WorkerConfig,
    ) -> JobResult:
        ...


class RealHfssSimulator(Simulator):
    """Synchronous optimization-facing adapter around the existing HFSS worker executor."""

    def __init__(
        self,
        *,
        template_cfg: ResolvedTemplate,
        worker_cfg: WorkerConfig,
        port_num: int,
        unit_map: dict[str, str],
        variable_order: list[str],
        metrics_to_extract: list[str] | None = None,
        executor: HfssExecutor | None = None,
    ) -> None:
        self.template_cfg = template_cfg
        self.worker_cfg = worker_cfg
        self.port_num = port_num
        self.unit_map = dict(unit_map)
        self.variable_order = list(variable_order)
        self.metrics_to_extract = metrics_to_extract or ["max_s11_db"]
        self.executor = executor or HfssJobExecutor.default()

    def simulate(self, request: SimulationInput) -> SimulationOutput:
        if list(request.variable_order) != self.variable_order:
            raise SimulationError(
                request.case_id,
                "invalid_variables",
                f"Expected variable order {self.variable_order}, got {request.variable_order}",
            )

        job = self._build_job(request)
        try:
            result = self.executor.execute(job, self.template_cfg, self.worker_cfg)
        except WorkerError as exc:
            raise SimulationError(request.case_id, exc.failure_class, exc.message) from exc

        if result.status != "done":
            raise SimulationError(
                request.case_id,
                "simulation_failed",
                result.error_message or "HFSS simulation failed",
            )

        touchstone_path = self._resolve_touchstone_path(request.case_id, result)
        freqs, s_params = self._parse_touchstone(request.case_id, touchstone_path)
        return SimulationOutput(
            case_id=request.case_id,
            geo_params=request.geo_params(),
            freqs=freqs,
            s_params=s_params,
            artifacts={"touchstone": touchstone_path},
        )

    def _build_job(self, request: SimulationInput) -> JobRequest:
        output_filename = f"{request.case_id}.s{self.port_num}p"
        return JobRequest(
            job_id=request.case_id,
            design_name=self.template_cfg.design_name,
            template_name=self.template_cfg.template_name,
            setup_name=self.template_cfg.setup_name,
            sweep_name=self.template_cfg.sweep_name,
            variables=self._format_variables(request),
            output_filename=output_filename,
            metrics_to_extract=self.metrics_to_extract,
            artifact_options={},
            submit_time=utc_now(),
            retry_count=0,
        )

    def _format_variables(self, request: SimulationInput) -> dict[str, str]:
        variables: dict[str, str] = {}
        for name in self.variable_order:
            if name not in request.variables:
                raise SimulationError(
                    request.case_id,
                    "invalid_variables",
                    f"Missing simulation variable: {name}",
                )
            unit = self.unit_map.get(name)
            if not unit:
                raise SimulationError(
                    request.case_id,
                    "invalid_variables",
                    f"Missing HFSS unit for variable: {name}",
                )
            variables[name] = f"{request.variables[name]:g}{unit}"
        return variables

    def _resolve_touchstone_path(self, case_id: str, result: JobResult) -> Path:
        touchstone = result.artifacts.touchstone
        if touchstone is None:
            raise SimulationError(case_id, "missing_artifact", "HFSS result has no Touchstone artifact")

        path = Path(touchstone)
        if not path.is_absolute():
            path = self.worker_cfg.queue_root / path
        if not path.exists():
            raise SimulationError(case_id, "missing_artifact", f"Touchstone artifact not found: {path}")
        return path

    def _parse_touchstone(self, case_id: str, touchstone_path: Path) -> tuple[np.ndarray, np.ndarray]:
        parser = FunctionFromTouchstone(port_num=self.port_num)
        try:
            parser.parse(str(touchstone_path))
            s_params = parser.evaluate_all()
        except Exception as exc:
            raise SimulationError(
                case_id,
                "artifact_parse_failure",
                f"Failed to parse Touchstone artifact {touchstone_path}: {exc}",
            ) from exc

        return np.asarray(parser.freq, dtype=float), np.asarray(s_params, dtype=complex)
