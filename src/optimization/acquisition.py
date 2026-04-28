from __future__ import annotations

from abc import ABC, abstractmethod
from math import erf, pi, sqrt

import numpy as np


class AcquisitionFunction(ABC):
    @abstractmethod
    def score(
        self,
        mean: np.ndarray,
        uncertainty: np.ndarray | None,
        *,
        best_y: float | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class UCBAcquisition(AcquisitionFunction):
    def __init__(self, beta: float = 2.0) -> None:
        self.beta = float(beta)

    def score(
        self,
        mean: np.ndarray,
        uncertainty: np.ndarray | None,
        *,
        best_y: float | None = None,
    ) -> np.ndarray:
        _ = best_y
        mu = np.asarray(mean, dtype=float).reshape(-1)
        if uncertainty is None:
            # Optimizer minimizes objective; larger score should mean better candidate.
            return -mu
        sigma = np.asarray(uncertainty, dtype=float).reshape(-1)
        # LCB transformed to a maximization score.
        return -mu + self.beta * sigma


class EIAcquisition(AcquisitionFunction):
    def __init__(self, xi: float = 0.01) -> None:
        self.xi = float(xi)

    @staticmethod
    def _phi(z: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * z * z) / sqrt(2.0 * pi)

    @staticmethod
    def _Phi(z: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))

    def score(
        self,
        mean: np.ndarray,
        uncertainty: np.ndarray | None,
        *,
        best_y: float | None = None,
    ) -> np.ndarray:
        mu = np.asarray(mean, dtype=float).reshape(-1)
        if uncertainty is None or best_y is None:
            return -mu

        sigma = np.asarray(uncertainty, dtype=float).reshape(-1)
        sigma_safe = np.maximum(sigma, 1e-12)

        # Improvement for minimization: lower objective is better.
        improvement = best_y - mu - self.xi
        z = improvement / sigma_safe
        ei = improvement * self._Phi(z) + sigma_safe * self._phi(z)

        # Keep EI well-defined when uncertainty collapses.
        mask_zero = sigma <= 1e-12
        if np.any(mask_zero):
            ei[mask_zero] = np.maximum(0.0, improvement[mask_zero])

        return ei
