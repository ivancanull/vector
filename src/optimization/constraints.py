from __future__ import annotations

from dataclasses import dataclass

from .metrics import SParamMetric
from .sparam_response import SParameterResponse


@dataclass(frozen=True)
class Constraint:
    name: str
    metric: SParamMetric

    def violation(self, response: SParameterResponse) -> tuple[float, dict[str, float]]:
        raise NotImplementedError


@dataclass(frozen=True)
class LessEqualConstraint(Constraint):
    upper_bound: float

    def violation(self, response: SParameterResponse) -> tuple[float, dict[str, float]]:
        metric_value = float(self.metric.compute(response))
        v = metric_value - float(self.upper_bound)
        return float(v), {
            "constraint": self.name,
            "metric_name": self.metric.name,
            "metric_value": metric_value,
            "upper_bound": float(self.upper_bound),
            "violation": float(v),
        }


@dataclass(frozen=True)
class GreaterEqualConstraint(Constraint):
    lower_bound: float

    def violation(self, response: SParameterResponse) -> tuple[float, dict[str, float]]:
        metric_value = float(self.metric.compute(response))
        v = float(self.lower_bound) - metric_value
        return float(v), {
            "constraint": self.name,
            "metric_name": self.metric.name,
            "metric_value": metric_value,
            "lower_bound": float(self.lower_bound),
            "violation": float(v),
        }
