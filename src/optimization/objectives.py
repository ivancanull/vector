from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .constraints import Constraint, GreaterEqualConstraint, LessEqualConstraint
from .metrics import (
    AtFrequencyMetric,
    BandMaxMetric,
    BandMeanMetric,
    BandMinMetric,
    BandRippleMetric,
    PeakFrequencyMetric,
    SParamMetric,
)
from .sparam_response import SParameterResponse

ObjectiveDirection = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class ObjectiveResult:
    value: float
    objectives: dict[str, float]
    constraints: dict[str, float]
    feasible: bool
    metrics: dict[str, float]

    @property
    def objective(self) -> float:
        # Backward-compatible alias used by ReplayOptimizer.
        return self.value


@dataclass(frozen=True)
class ObjectiveTerm:
    name: str
    metric: SParamMetric
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("weight must be > 0")

    def evaluate(self, response: SParameterResponse) -> tuple[float, dict[str, float]]:
        raise NotImplementedError


@dataclass(frozen=True)
class MaximizeMetric(ObjectiveTerm):
    def evaluate(self, response: SParameterResponse) -> tuple[float, dict[str, float]]:
        metric_value = float(self.metric.compute(response))
        term_value = -self.weight * metric_value
        return float(term_value), {
            "term": self.name,
            "metric_name": self.metric.name,
            "metric_value": metric_value,
            "term_value": float(term_value),
        }


@dataclass(frozen=True)
class MinimizeMetric(ObjectiveTerm):
    def evaluate(self, response: SParameterResponse) -> tuple[float, dict[str, float]]:
        metric_value = float(self.metric.compute(response))
        term_value = self.weight * metric_value
        return float(term_value), {
            "term": self.name,
            "metric_name": self.metric.name,
            "metric_value": metric_value,
            "term_value": float(term_value),
        }


@dataclass(frozen=True)
class TargetMetric(ObjectiveTerm):
    target: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scale <= 0:
            raise ValueError("scale must be > 0")

    def evaluate(self, response: SParameterResponse) -> tuple[float, dict[str, float]]:
        metric_value = float(self.metric.compute(response))
        term_value = self.weight * abs(metric_value - float(self.target)) / float(self.scale)
        return float(term_value), {
            "term": self.name,
            "metric_name": self.metric.name,
            "metric_value": metric_value,
            "target": float(self.target),
            "scale": float(self.scale),
            "term_value": float(term_value),
        }


class ObjectiveFunction:
    """Small objective engine: minimize weighted terms + positive constraint penalty."""

    def __init__(
        self,
        terms: list[ObjectiveTerm],
        constraints: list[Constraint] | None = None,
        penalty_weight: float = 100.0,
    ) -> None:
        if not terms:
            raise ValueError("ObjectiveFunction requires at least one objective term")
        if penalty_weight < 0:
            raise ValueError("penalty_weight must be >= 0")
        self.terms = list(terms)
        self.constraints = list(constraints or [])
        self.penalty_weight = float(penalty_weight)

    def evaluate(self, freqs: np.ndarray | SParameterResponse, s_params: np.ndarray | None = None) -> ObjectiveResult:
        response = self._to_response(freqs, s_params)

        metrics: dict[str, float] = {}
        objectives: dict[str, float] = {}
        constraints: dict[str, float] = {}

        raw_objective = 0.0
        for term in self.terms:
            term_value, info = term.evaluate(response)
            metric_name = str(info["metric_name"])
            metric_value = float(info["metric_value"])
            metrics[metric_name] = metric_value
            objectives[term.name] = float(term_value)
            raw_objective += float(term_value)

        total_positive_violation = 0.0
        feasible = True
        for cons in self.constraints:
            violation, info = cons.violation(response)
            metric_name = str(info["metric_name"])
            metric_value = float(info["metric_value"])
            metrics[metric_name] = metric_value
            constraints[cons.name] = float(violation)
            if violation > 0:
                feasible = False
                total_positive_violation += float(violation)

        penalty = self.penalty_weight * total_positive_violation
        value = raw_objective + penalty

        metrics["raw_objective"] = float(raw_objective)
        metrics["total_positive_violation"] = float(total_positive_violation)
        metrics["penalty"] = float(penalty)

        return ObjectiveResult(
            value=float(value),
            objectives=objectives,
            constraints=constraints,
            feasible=bool(feasible),
            metrics=metrics,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "objective_function",
            "minimize": True,
            "penalty_weight": self.penalty_weight,
            "terms": [
                {
                    "name": t.name,
                    "kind": t.__class__.__name__,
                    "metric": t.metric.name,
                    "weight": t.weight,
                }
                for t in self.terms
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

    @staticmethod
    def _to_response(freqs: np.ndarray | SParameterResponse, s_params: np.ndarray | None) -> SParameterResponse:
        if isinstance(freqs, SParameterResponse):
            if s_params is not None:
                raise ValueError("s_params must be None when evaluate is called with SParameterResponse")
            return freqs
        if s_params is None:
            raise ValueError("s_params must be provided when evaluate is called with raw freqs")
        return SParameterResponse.from_sparams(freqs, s_params)


class Objective(ObjectiveFunction):
    """Backward-compatible alias used by optimizer wiring."""


# Legacy metric names kept for existing code/tests.
class MaxS11DbMetric(SParamMetric):
    @property
    def name(self) -> str:
        return "max_s11_db"

    def compute(self, response: SParameterResponse) -> float:
        return BandMaxMetric("S11").compute(response)


class MinS21DbMetric(SParamMetric):
    @property
    def name(self) -> str:
        return "min_s21_db"

    def compute(self, response: SParameterResponse) -> float:
        return BandMinMetric("S21").compute(response)


@dataclass(frozen=True)
class PassbandMeanS21DbMetric(SParamMetric):
    passband_hz: tuple[float, float]

    @property
    def name(self) -> str:
        return "mean_s21_db"

    def compute(self, response: SParameterResponse) -> float:
        return BandMeanMetric("S21", self.passband_hz).compute(response)


class BCLFObjective(ObjectiveFunction):
    """Convenience constructor for the project BCLF spec."""

    def __init__(
        self,
        *,
        passband_hz: tuple[float, float],
        lower_stopband_hz: tuple[float, float],
        upper_stopband_hz: tuple[float, float],
        target_center_hz: float,
        min_passband_s21_db: float,
        max_passband_s11_db: float,
        max_stopband_s21_db: float,
        center_weight: float = 2.0,
        penalty_weight: float = 100.0,
    ) -> None:
        passband_metric = BandMeanMetric("S21", passband_hz)
        s11_passband_metric = BandMaxMetric("S11", passband_hz)
        stopband_low_metric = BandMaxMetric("S21", lower_stopband_hz)
        stopband_high_metric = BandMaxMetric("S21", upper_stopband_hz)
        peak_metric = PeakFrequencyMetric("S21", passband_hz)

        super().__init__(
            terms=[
                MaximizeMetric(
                    name="maximize_passband_s21_mean",
                    metric=passband_metric,
                    weight=1.0,
                ),
                TargetMetric(
                    name="align_s21_peak_frequency",
                    metric=peak_metric,
                    target=float(target_center_hz),
                    scale=float(passband_hz[1] - passband_hz[0]),
                    weight=float(center_weight),
                ),
            ],
            constraints=[
                GreaterEqualConstraint(
                    name="passband_s21_mean_geq",
                    metric=passband_metric,
                    lower_bound=float(min_passband_s21_db),
                ),
                LessEqualConstraint(
                    name="passband_s11_max_leq",
                    metric=s11_passband_metric,
                    upper_bound=float(max_passband_s11_db),
                ),
                LessEqualConstraint(
                    name="lower_stopband_s21_max_leq",
                    metric=stopband_low_metric,
                    upper_bound=float(max_stopband_s21_db),
                ),
                LessEqualConstraint(
                    name="upper_stopband_s21_max_leq",
                    metric=stopband_high_metric,
                    upper_bound=float(max_stopband_s21_db),
                ),
            ],
            penalty_weight=float(penalty_weight),
        )
        self.passband_hz = passband_hz
        self.lower_stopband_hz = lower_stopband_hz
        self.upper_stopband_hz = upper_stopband_hz


def build_bclf_objective(
    *,
    passband_hz: tuple[float, float] = (5.2e9, 5.9e9),
    lower_stopband_hz: tuple[float, float] = (3.6e9, 4.8e9),
    upper_stopband_hz: tuple[float, float] = (6.5e9, 8.7e9),
    target_center_hz: float = 5.55e9,
    min_passband_s21_db: float = -5.0,
    max_passband_s11_db: float = -10.0,
    max_stopband_s21_db: float = -25.0,
    center_weight: float = 2.0,
    penalty_weight: float = 100.0,
) -> BCLFObjective:
    return BCLFObjective(
        passband_hz=passband_hz,
        lower_stopband_hz=lower_stopband_hz,
        upper_stopband_hz=upper_stopband_hz,
        target_center_hz=target_center_hz,
        min_passband_s21_db=min_passband_s21_db,
        max_passband_s11_db=max_passband_s11_db,
        max_stopband_s21_db=max_stopband_s21_db,
        center_weight=center_weight,
        penalty_weight=penalty_weight,
    )


__all__ = [
    "AtFrequencyMetric",
    "BandMaxMetric",
    "BandMeanMetric",
    "BandMinMetric",
    "BandRippleMetric",
    "BCLFObjective",
    "Constraint",
    "GreaterEqualConstraint",
    "LessEqualConstraint",
    "MaxS11DbMetric",
    "MaximizeMetric",
    "Metric",
    "MinS21DbMetric",
    "MinimizeMetric",
    "Objective",
    "ObjectiveFunction",
    "ObjectiveResult",
    "ObjectiveTerm",
    "PassbandMeanS21DbMetric",
    "PeakFrequencyMetric",
    "SParamMetric",
    "SParameterResponse",
    "TargetMetric",
    "build_bclf_objective",
]

# Backward-compatible type alias.
Metric = SParamMetric
