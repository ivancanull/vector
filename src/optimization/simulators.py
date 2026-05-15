from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

from common.io_utils import read_json
from common.job_types import SimulationJob
from server.queue_manager import QueueManager

from simulation.core import SimulationInput
from simulation.result_store import SimulationResultStore

from .datasets import PassiveDataset


SimulationStatus = Literal["done", "failed"]


@dataclass(frozen=True)
class SimulationResult:
    freqs: np.ndarray | None
    s_params: np.ndarray | None
    status: SimulationStatus
    error: str | None = None
    metadata: dict[str, Any] | None = None
    actual_geometry: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.actual_geometry is not None:
            object.__setattr__(self, "actual_geometry", np.asarray(self.actual_geometry, dtype=float).reshape(-1))
        if self.freqs is not None:
            object.__setattr__(self, "freqs", np.asarray(self.freqs, dtype=float))
        if self.s_params is not None:
            object.__setattr__(self, "s_params", np.asarray(self.s_params, dtype=complex))


class Simulator(ABC):
    def __init__(self) -> None:
        self.num_calls = 0
        self.num_success = 0
        self.num_failures = 0

    @abstractmethod
    def simulate(self, geometry: np.ndarray) -> SimulationResult:
        raise NotImplementedError


class ReplaySimulator(Simulator):
    """Offline simulator that replays stored responses from an existing dataset."""

    def __init__(self, *, dataset: PassiveDataset) -> None:
        super().__init__()
        self.dataset = dataset

    def simulate(self, geometry: np.ndarray) -> SimulationResult:
        self.num_calls += 1
        try:
            idx = self.dataset.find_index_by_geometry(geometry)
            freqs, s_params = self.dataset.get_response_by_geometry(geometry)
        except (ValueError, IndexError) as exc:
            self.num_failures += 1
            return SimulationResult(
                freqs=None,
                s_params=None,
                status="failed",
                error=str(exc),
                metadata=None,
            )

        self.num_success += 1
        return SimulationResult(
            freqs=freqs,
            s_params=s_params,
            status="done",
            error=None,
            metadata={"dataset_index": idx, "case_id": self.dataset.get_case_id(idx)},
            actual_geometry=self.dataset.get_geometry(idx),
        )


class HfssSimulator(Simulator):
    """Adapter that exposes geometry-only simulate API over existing HFSS simulator stack."""

    def __init__(self, *, backend: Any, variable_order: list[str]) -> None:
        super().__init__()
        self.backend = backend
        self.variable_order = list(variable_order)

    def simulate(self, geometry: np.ndarray) -> SimulationResult:
        self.num_calls += 1

        values = np.asarray(geometry, dtype=float).reshape(-1)
        if len(values) != len(self.variable_order):
            self.num_failures += 1
            return SimulationResult(
                freqs=None,
                s_params=None,
                status="failed",
                error=(
                    f"Geometry dimension mismatch: got {len(values)}, expected {len(self.variable_order)}"
                ),
            )

        variables = {name: float(values[i]) for i, name in enumerate(self.variable_order)}
        request = _GeometryRequest(
            case_id=f"sim_{self.num_calls:06d}",
            variables=variables,
            variable_order=self.variable_order,
        )

        try:
            output = self.backend.simulate(request)
        except Exception as exc:
            # Backend is an external execution boundary (HFSS / worker adapter).
            self.num_failures += 1
            return SimulationResult(freqs=None, s_params=None, status="failed", error=str(exc))

        self.num_success += 1
        return SimulationResult(
            freqs=output.freqs,
            s_params=output.s_params,
            status="done",
            error=None,
            metadata={"case_id": getattr(output, "case_id", request.case_id), "artifacts": getattr(output, "artifacts", None)},
            actual_geometry=request.geo_params(),
        )


class ServerQueueSimulator(Simulator):
    """Queue-backed real simulator with cache-first touchstone lookup."""

    def __init__(
        self,
        *,
        queue_root: str,
        variable_order: list[str],
        unit_map: dict[str, str],
        template_name: str,
        port_num: int,
        result_store: SimulationResultStore,
        design_name: str | None = None,
        setup_name: str | None = None,
        sweep_name: str | None = None,
        metrics_to_extract: list[str] | None = None,
        poll_interval_sec: float = 2.0,
        job_timeout_sec: float = 7200.0,
        run_id: str = "bo_online",
        mm_resolution: float = 0.01,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.queue_root = str(queue_root)
        self.variable_order = list(variable_order)
        self.unit_map = dict(unit_map)
        self.template_name = template_name
        self.design_name = design_name
        self.setup_name = setup_name
        self.sweep_name = sweep_name
        self.metrics_to_extract = list(metrics_to_extract or ["max_s11_db"])
        self.port_num = int(port_num)
        self.result_store = result_store
        self.poll_interval_sec = float(poll_interval_sec)
        self.job_timeout_sec = float(job_timeout_sec)
        self.run_id = run_id
        self.mm_resolution = float(mm_resolution)
        self.verbose = bool(verbose)
        if self.mm_resolution <= 0:
            raise ValueError("mm_resolution must be > 0")

        self.queue = QueueManager(self.queue_root)
        self._next_case_id = 0
        self._context_iteration: int | None = None

    def set_context(self, *, run_id: str, iteration: int) -> None:
        self.run_id = str(run_id)
        self._context_iteration = int(iteration)

    def simulate(self, geometry: np.ndarray) -> SimulationResult:
        self.num_calls += 1
        start = time.perf_counter()
        values = np.asarray(geometry, dtype=float).reshape(-1)
        self._log(f"simulate call={self.num_calls} run_id={self.run_id} geometry={values.tolist()}")
        if len(values) != len(self.variable_order):
            self.num_failures += 1
            self._log(
                f"failed: geometry dimension mismatch got={len(values)} expected={len(self.variable_order)}"
            )
            return SimulationResult(
                freqs=None,
                s_params=None,
                status="failed",
                error=f"Geometry dimension mismatch: got {len(values)}, expected {len(self.variable_order)}",
            )

        case_id = self._build_case_id()
        quantized_variables = self._quantize_variables(
            {name: float(values[i]) for i, name in enumerate(self.variable_order)}
        )
        actual_geometry = np.array([quantized_variables[name] for name in self.variable_order], dtype=float)
        self._log(
            f"case_id={case_id} quantized_geometry={actual_geometry.tolist()} vars={quantized_variables}"
        )

        request = SimulationInput(
            case_id=case_id,
            variable_order=list(self.variable_order),
            variables=quantized_variables,
        )

        cached = self.result_store.lookup(request)
        if cached is not None:
            self._log(f"cache hit: case_id={case_id} touchstone={cached.touchstone_path}")
            try:
                freqs, s_params = self._load_touchstone(cached.touchstone_path)
            except Exception as exc:
                self.num_failures += 1
                return SimulationResult(
                    freqs=None,
                    s_params=None,
                    status="failed",
                    error=f"Failed to parse cached touchstone: {exc}",
                    metadata={"cache_hit": True, "case_id": case_id},
                )
            self.num_success += 1
            self._log(
                f"cache done: case_id={case_id} elapsed={float(time.perf_counter() - start):.3f}s"
            )
            return SimulationResult(
                freqs=freqs,
                s_params=s_params,
                status="done",
                metadata={
                    "cache_hit": True,
                    "job_id": case_id,
                    "case_id": case_id,
                    "result_json_path": None,
                    "wait_seconds": 0.0,
                    "touchstone_path": str(cached.touchstone_path),
                    "simulator_elapsed_sec": float(time.perf_counter() - start),
                },
                actual_geometry=actual_geometry,
            )

        job = self._build_job(case_id, request.variables)
        self._log(
            f"enqueue: job_id={job.job_id} template={job.template_name} output={job.output_filename} "
            f"queue_root={self.queue_root}"
        )
        self.queue.enqueue(job)
        wait_res = self._wait_job_result(case_id)
        self._log(
            f"wait result: case_id={case_id} status={wait_res.get('status')} "
            f"wait={float(wait_res.get('wait_seconds', 0.0)):.3f}s "
            f"result_json={wait_res.get('result_json_path')}"
        )
        if wait_res["status"] != "done":
            self.num_failures += 1
            self._log(
                f"failed: case_id={case_id} error={wait_res.get('error', 'simulation_failed')}"
            )
            return SimulationResult(
                freqs=None,
                s_params=None,
                status="failed",
                error=str(wait_res.get("error", "simulation_failed")),
                metadata={
                    "cache_hit": False,
                    "job_id": case_id,
                    "case_id": case_id,
                    "result_json_path": str(wait_res.get("result_json_path")) if wait_res.get("result_json_path") else None,
                    "wait_seconds": float(wait_res["wait_seconds"]),
                    "touchstone_path": None,
                    "simulator_elapsed_sec": float(time.perf_counter() - start),
                },
            )

        result_json_path = wait_res["result_json_path"]
        payload = read_json(result_json_path)
        artifact_touchstone = payload.get("artifacts", {}).get("touchstone")
        if not isinstance(artifact_touchstone, str) or not artifact_touchstone:
            self.num_failures += 1
            self._log(f"failed: case_id={case_id} missing touchstone artifact in result payload")
            return SimulationResult(
                freqs=None,
                s_params=None,
                status="failed",
                error="Result JSON has no touchstone artifact",
                metadata={
                    "cache_hit": False,
                    "job_id": case_id,
                    "case_id": case_id,
                    "result_json_path": str(result_json_path),
                    "wait_seconds": float(wait_res["wait_seconds"]),
                },
            )

        touchstone_path = wait_res["queue_root"] / artifact_touchstone
        try:
            saved = self.result_store.save(
                request=request,
                source_touchstone=touchstone_path,
                metadata={"source": "queue"},
            )
            freqs, s_params = self._load_touchstone(saved.touchstone_path)
        except Exception as exc:
            self.num_failures += 1
            self._log(f"failed: case_id={case_id} parse/save touchstone error={exc}")
            return SimulationResult(
                freqs=None,
                s_params=None,
                status="failed",
                error=f"Failed to parse/save touchstone: {exc}",
                metadata={
                    "cache_hit": False,
                    "job_id": case_id,
                    "case_id": case_id,
                    "result_json_path": str(result_json_path),
                    "wait_seconds": float(wait_res["wait_seconds"]),
                    "touchstone_path": str(touchstone_path),
                    "simulator_elapsed_sec": float(time.perf_counter() - start),
                },
            )

        self.num_success += 1
        self._log(
            f"done: case_id={case_id} touchstone={saved.touchstone_path} "
            f"elapsed={float(time.perf_counter() - start):.3f}s"
        )
        return SimulationResult(
            freqs=freqs,
            s_params=s_params,
            status="done",
            metadata={
                "cache_hit": False,
                "job_id": case_id,
                "case_id": case_id,
                "result_json_path": str(result_json_path),
                "wait_seconds": float(wait_res["wait_seconds"]),
                "touchstone_path": str(saved.touchstone_path),
                "simulator_elapsed_sec": float(time.perf_counter() - start),
            },
            actual_geometry=actual_geometry,
        )

    def _quantize_variables(self, variables: dict[str, float]) -> dict[str, float]:
        quantized: dict[str, float] = {}
        for name in self.variable_order:
            value = float(variables[name])
            unit = self.unit_map.get(name, "")
            if unit == "mm":
                snapped = round(value / self.mm_resolution) * self.mm_resolution
                quantized[name] = float(round(snapped, 8))
            else:
                quantized[name] = value
        return quantized

    def _build_case_id(self) -> str:
        if self._context_iteration is not None:
            case_id = f"{self.run_id}_it{self._context_iteration:04d}"
            self._context_iteration = None
            return case_id
        case_id = f"{self.run_id}_it{self._next_case_id:04d}"
        self._next_case_id += 1
        return case_id

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[ServerQueueSimulator] {message}")

    def _build_job(self, case_id: str, variables: dict[str, float]) -> SimulationJob:
        formatted: dict[str, str] = {}
        for name in self.variable_order:
            unit = self.unit_map.get(name)
            if not unit:
                raise ValueError(f"Missing unit for variable {name}")
            formatted[name] = f"{variables[name]:g}{unit}"
        return SimulationJob(
            job_id=case_id,
            template_name=self.template_name,
            variables=formatted,
            output_filename=f"{case_id}.s{self.port_num}p",
            metrics_to_extract=list(self.metrics_to_extract),
            design_name=self.design_name,
            setup_name=self.setup_name,
            sweep_name=self.sweep_name,
        )

    def _wait_job_result(self, case_id: str) -> dict[str, Any]:
        queue_root = self.queue.root
        done_path = queue_root / "done" / f"{case_id}.result.json"
        failed_path = queue_root / "failed" / f"{case_id}.result.json"
        start = time.perf_counter()
        while True:
            if done_path.exists():
                archived_running = self._archive_running_job(case_id, queue_root)
                return {
                    "status": "done",
                    "result_json_path": done_path,
                    "wait_seconds": float(time.perf_counter() - start),
                    "queue_root": queue_root,
                    "archived_running_path": str(archived_running) if archived_running else None,
                }
            if failed_path.exists():
                payload = read_json(failed_path)
                archived_running = self._archive_running_job(case_id, queue_root)
                return {
                    "status": "failed",
                    "result_json_path": failed_path,
                    "wait_seconds": float(time.perf_counter() - start),
                    "queue_root": queue_root,
                    "error": payload.get("error_message", "simulation_failed"),
                    "archived_running_path": str(archived_running) if archived_running else None,
                }
            if (time.perf_counter() - start) >= self.job_timeout_sec:
                return {
                    "status": "failed",
                    "result_json_path": None,
                    "wait_seconds": float(time.perf_counter() - start),
                    "queue_root": queue_root,
                    "error": "job_timeout",
                }
            time.sleep(self.poll_interval_sec)

    def _load_touchstone(self, touchstone_path: Any) -> tuple[np.ndarray, np.ndarray]:
        # Support both import styles used across scripts/notebooks.
        try:
            from src.complex_func import FunctionFromTouchstone
        except ImportError:
            from complex_func import FunctionFromTouchstone

        parser = FunctionFromTouchstone(port_num=self.port_num)
        parser.parse(str(touchstone_path))
        s_params = parser.evaluate_all()
        return np.asarray(parser.freq, dtype=float), np.asarray(s_params, dtype=complex)

    @staticmethod
    def _archive_running_job(case_id: str, queue_root: Path) -> Path | None:
        running_path = queue_root / "running" / f"{case_id}.json"
        if not running_path.exists():
            return None
        archive_dir = queue_root / "archive" / "running"
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = archive_dir / running_path.name
        if target.exists():
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            target = archive_dir / f"{running_path.stem}.{ts}{running_path.suffix}"
        running_path.replace(target)
        return target


@dataclass(frozen=True)
class _GeometryRequest:
    case_id: str
    variables: dict[str, float]
    variable_order: list[str]

    def geo_params(self) -> np.ndarray:
        return np.array([self.variables[name] for name in self.variable_order], dtype=float)
