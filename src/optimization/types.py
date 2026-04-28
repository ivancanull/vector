from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvaluationRecord:
    dataset_index: int
    case_id: str
    geo_params: np.ndarray
    metrics: dict[str, float]
    objective: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "geo_params", np.asarray(self.geo_params, dtype=float))
