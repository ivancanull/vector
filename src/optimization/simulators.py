from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .datasets import PassiveDataset


SimulationStatus = Literal["done", "failed"]


@dataclass(frozen=True)
class SimulationResult:
    freqs: np.ndarray | None
    s_params: np.ndarray | None
    status: SimulationStatus
    error: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
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
        )


@dataclass(frozen=True)
class _GeometryRequest:
    case_id: str
    variables: dict[str, float]
    variable_order: list[str]

    def geo_params(self) -> np.ndarray:
        return np.array([self.variables[name] for name in self.variable_order], dtype=float)
