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


def ghz_to_hz(value_ghz: float) -> float:
    return float(value_ghz) * 1e9


def band_ghz_to_hz(band_ghz: tuple[float, float]) -> tuple[float, float]:
    return (ghz_to_hz(band_ghz[0]), ghz_to_hz(band_ghz[1]))


@dataclass(frozen=True)
class ObjectiveResult:
    value: float
    objective_components: dict[str, float]
    constraint_metrics: dict[str, float]
    constraint_violations: dict[str, float]
    feasible: bool | None
    constraint_margin_min: float | None
    metrics: dict[str, float]

    @property
    def objective(self) -> float:
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


@dataclass(frozen=True)
class LowerBoundHingePenalty(ObjectiveTerm):
    bound: float = 0.0

    def evaluate(self, response: SParameterResponse) -> tuple[float, dict[str, float]]:
        metric_value = float(self.metric.compute(response))
        hinge = max(0.0, float(self.bound) - metric_value)
        term_value = self.weight * hinge
        return float(term_value), {
            "term": self.name,
            "metric_name": self.metric.name,
            "metric_value": metric_value,
            "bound": float(self.bound),
            "hinge_value": float(hinge),
            "term_value": float(term_value),
        }


@dataclass(frozen=True)
class UpperBoundHingePenalty(ObjectiveTerm):
    bound: float = 0.0

    def evaluate(self, response: SParameterResponse) -> tuple[float, dict[str, float]]:
        metric_value = float(self.metric.compute(response))
        hinge = max(0.0, metric_value - float(self.bound))
        term_value = self.weight * hinge
        return float(term_value), {
            "term": self.name,
            "metric_name": self.metric.name,
            "metric_value": metric_value,
            "bound": float(self.bound),
            "hinge_value": float(hinge),
            "term_value": float(term_value),
        }


class ObjectiveFunction:
    """Minimize weighted objective terms; constraints are handled separately."""

    def __init__(
        self,
        terms: list[ObjectiveTerm],
        constraints: list[Constraint] | None = None,
    ) -> None:
        if not terms:
            raise ValueError("ObjectiveFunction requires at least one objective term")
        self.terms = list(terms)
        self.constraints = list(constraints or [])

    def evaluate(self, freqs: np.ndarray | SParameterResponse, s_params: np.ndarray | None = None) -> ObjectiveResult:
        response = self._to_response(freqs, s_params)

        objective_components: dict[str, float] = {}
        constraint_metrics: dict[str, float] = {}
        constraint_violations: dict[str, float] = {}
        metrics: dict[str, float] = {}

        aggregate_objective = 0.0
        for term in self.terms:
            term_value, info = term.evaluate(response)
            metric_name = str(info["metric_name"])
            metric_value = float(info["metric_value"])
            metrics[metric_name] = metric_value
            objective_components[term.name] = float(term_value)
            aggregate_objective += float(term_value)

        feasible: bool | None = True
        margin_values: list[float] = []
        for cons in self.constraints:
            violation, info = cons.violation(response)
            metric_name = str(info["metric_name"])
            metric_value = float(info["metric_value"])
            metrics[metric_name] = metric_value
            constraint_metrics[metric_name] = metric_value
            constraint_violations[cons.name] = float(violation)
            margin_values.append(float(-violation))
            if violation > 0:
                feasible = False

        metrics["aggregate_objective"] = float(aggregate_objective)
        # Backward-compatible diagnostics expected by older analysis notebooks.
        metrics["raw_objective"] = float(aggregate_objective)
        metrics["total_positive_violation"] = float(
            sum(max(0.0, v) for v in constraint_violations.values())
        )
        metrics["penalty"] = 0.0

        if not self.constraints:
            feasible = None
            constraint_margin_min = None
        else:
            constraint_margin_min = float(min(margin_values)) if margin_values else float("inf")

        return ObjectiveResult(
            value=float(aggregate_objective),
            objective_components=objective_components,
            constraint_metrics=constraint_metrics,
            constraint_violations=constraint_violations,
            feasible=feasible,
            constraint_margin_min=constraint_margin_min,
            metrics=metrics,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "objective_function",
            "minimize": True,
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


# Backward-compatible generic alias.
Metric = SParamMetric

__all__ = [
    "AtFrequencyMetric",
    "BandMaxMetric",
    "BandMeanMetric",
    "BandMinMetric",
    "BandRippleMetric",
    "Constraint",
    "GreaterEqualConstraint",
    "LessEqualConstraint",
    "MaximizeMetric",
    "Metric",
    "MinimizeMetric",
    "Objective",
    "ObjectiveFunction",
    "ObjectiveResult",
    "ObjectiveTerm",
    "PeakFrequencyMetric",
    "SParamMetric",
    "SParameterResponse",
    "TargetMetric",
    "LowerBoundHingePenalty",
    "UpperBoundHingePenalty",
    "ghz_to_hz",
    "band_ghz_to_hz",
]
