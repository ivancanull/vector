from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SimulationInput:
    case_id: str
    variables: dict[str, float]
    variable_order: list[str]

    def geo_params(self) -> np.ndarray:
        missing = [name for name in self.variable_order if name not in self.variables]
        if missing:
            raise SimulationError(
                self.case_id,
                "invalid_variables",
                f"Missing simulation variables: {missing}",
            )
        return np.array([self.variables[name] for name in self.variable_order], dtype=float)


@dataclass(frozen=True)
class SimulationOutput:
    case_id: str
    geo_params: np.ndarray
    freqs: np.ndarray
    s_params: np.ndarray
    artifacts: dict[str, Path] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "geo_params", np.asarray(self.geo_params, dtype=float))
        object.__setattr__(self, "freqs", np.asarray(self.freqs, dtype=float))
        object.__setattr__(self, "s_params", np.asarray(self.s_params, dtype=complex))

    def to_dataset_item(self) -> dict:
        return {
            "caseid": self.case_id,
            "geo_params": self.geo_params,
            "s_params": self.s_params,
            "sampled_indices": None,
        }


@dataclass
class SimulationError(Exception):
    case_id: str
    failure_class: str
    message: str

    def __str__(self) -> str:
        return f"{self.case_id}: {self.message}"


class Simulator(ABC):
    @abstractmethod
    def simulate(self, request: SimulationInput) -> SimulationOutput:
        pass
