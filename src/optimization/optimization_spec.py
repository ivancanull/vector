from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constraints import Constraint
from .metrics import BandMaxMetric, BandMeanMetric, BandMinMetric, UnionBandMaxMetric, UnionBandMeanMetric
from .objectives import (
    LowerBoundHingePenalty,
    MaximizeMetric,
    MinimizeMetric,
    Objective,
    ObjectiveTerm,
    UpperBoundHingePenalty,
    band_ghz_to_hz,
)


@dataclass(frozen=True)
class SpecResults:
    """Array-based optimization result consumed by generic optimizers."""

    objective: np.ndarray
    constraints: np.ndarray
    metrics: dict[str, float]
    objective_components: dict[str, float]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        objective = np.asarray(self.objective, dtype=float).reshape(-1)
        constraints = np.asarray(self.constraints, dtype=float).reshape(-1)
        if objective.size == 0:
            raise ValueError("SpecResults.objective must contain at least one value")
        if not np.all(np.isfinite(objective)):
            raise ValueError("SpecResults.objective must be finite")
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "metrics", {str(k): float(v) for k, v in self.metrics.items()})
        object.__setattr__(
            self,
            "objective_components",
            {str(k): float(v) for k, v in self.objective_components.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def scalar_objective(self) -> float:
        if self.objective.shape != (1,):
            raise ValueError(
                f"Only scalar objectives are supported now; got objective shape {self.objective.shape}"
            )
        return float(self.objective[0])


@dataclass(frozen=True)
class OptimizationSpec:
    objective_terms: list[ObjectiveTerm]
    constraints: list[Constraint]
    frequency_bands_ghz: dict[str, float | tuple[float, float]]

    def evaluate(self, freqs, s_params) -> SpecResults:
        result = self.to_objective().evaluate(freqs, s_params)
        return SpecResults(
            objective=np.asarray([float(result.objective)], dtype=float),
            constraints=np.asarray(
                [float(v) for v in result.constraint_violations.values()],
                dtype=float,
            ),
            metrics=result.metrics,
            objective_components=result.objective_components,
            metadata={
                "constraint_metrics": result.constraint_metrics,
                "constraint_violations": result.constraint_violations,
                "feasible": result.feasible,
                "constraint_margin_min": result.constraint_margin_min,
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "optimization_spec",
            "spec": self.__class__.__name__,
            "minimize": True,
            "frequency_bands_ghz": dict(self.frequency_bands_ghz),
            "terms": [
                {
                    "name": t.name,
                    "kind": t.__class__.__name__,
                    "metric": t.metric.name,
                    "weight": t.weight,
                }
                for t in self.objective_terms
            ],
            "constraints": [
                {
                    "name": c.name,
                    "kind": c.__class__.__name__,
                    "metric": c.metric.name,
                }
                for c in self.constraints
            ],
        }

    def to_objective(self) -> Objective:
        return Objective(terms=self.objective_terms, constraints=self.constraints)


class BCLFOptimizationSpec(OptimizationSpec):
    """BCLF spec with penalty-augmented objective and no explicit constraints."""

    def __init__(
        self,
        *,
        passband_ghz: tuple[float, float] = (5.2, 5.9),
        lower_stopband_ghz: tuple[float, float] = (3.6, 4.8),
        upper_stopband_ghz: tuple[float, float] = (6.5, 8.7),
        min_passband_s21_db: float = -7.0,
        max_passband_s11_db: float = -7.0,
        max_stopband_s21_db: float = -18.0,
        s11_weight: float = 0.5,
        stopband_weight: float = 0.5,
        penalty_weight: float = 5.0,
    ) -> None:
        passband_hz = band_ghz_to_hz(passband_ghz)
        lower_stopband_hz = band_ghz_to_hz(lower_stopband_ghz)
        upper_stopband_hz = band_ghz_to_hz(upper_stopband_ghz)
        stopband_union_hz = (lower_stopband_hz, upper_stopband_hz)

        a_mean_metric = BandMeanMetric("S21", passband_hz)
        b_mean_metric = BandMeanMetric("S11", passband_hz)
        c_mean_metric = UnionBandMeanMetric("S21", stopband_union_hz)

        a_worst_metric = BandMinMetric("S21", passband_hz)
        b_worst_metric = BandMaxMetric("S11", passband_hz)
        c_worst_metric = UnionBandMaxMetric("S21", stopband_union_hz)

        objective_terms = [
            # S_score = a_mean - 0.5*b_mean - 0.5*c_mean.
            # The optimizer minimizes objective, so these terms encode -S_score.
            MaximizeMetric(name="score_a_mean_passband_s21", metric=a_mean_metric, weight=1.0),
            MinimizeMetric(name="score_b_mean_passband_s11", metric=b_mean_metric, weight=float(s11_weight)),
            MinimizeMetric(name="score_c_mean_stopband_s21", metric=c_mean_metric, weight=float(stopband_weight)),
            # P = hinge_a + hinge_b + hinge_c ; objective adds penalty_weight * P.
            LowerBoundHingePenalty(
                name="penalty_a_worst_passband_s21",
                metric=a_worst_metric,
                bound=float(min_passband_s21_db),
                weight=float(penalty_weight),
            ),
            UpperBoundHingePenalty(
                name="penalty_b_worst_passband_s11",
                metric=b_worst_metric,
                bound=float(max_passband_s11_db),
                weight=float(penalty_weight),
            ),
            UpperBoundHingePenalty(
                name="penalty_c_worst_stopband_s21",
                metric=c_worst_metric,
                bound=float(max_stopband_s21_db),
                weight=float(penalty_weight),
            ),
        ]
        frequency_bands_ghz = {
            "passband": passband_ghz,
            "lower_stopband": lower_stopband_ghz,
            "upper_stopband": upper_stopband_ghz,
        }
        object.__setattr__(self, "objective_terms", objective_terms)
        object.__setattr__(self, "constraints", [])
        object.__setattr__(self, "frequency_bands_ghz", frequency_bands_ghz)


def build_bclf_optimization_spec(**kwargs) -> OptimizationSpec:
    """Compatibility wrapper for legacy callers. Prefer BCLFOptimizationSpec(...)."""
    return BCLFOptimizationSpec(**kwargs)


def build_bclf_objective(**kwargs) -> Objective:
    """Compatibility wrapper for legacy callers. Prefer BCLFOptimizationSpec(...).to_objective()."""
    return BCLFOptimizationSpec(**kwargs).to_objective()
