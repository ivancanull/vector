from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class CandidateSpace(ABC):
    """Common candidate-provider and geometry-transform interface.

    Candidate spaces own the physical domain and the reversible transform used by
    model-based optimizers. Public methods return physical geometry values;
    optimizers call normalize before passing values to BoTorch.
    """

    def __init__(self, *, variable_order: list[str], match_atol: float = 1e-9, match_rtol: float = 1e-9) -> None:
        if not variable_order:
            raise ValueError("variable_order must be non-empty")
        self.variable_order = list(variable_order)
        self.match_atol = float(match_atol)
        self.match_rtol = float(match_rtol)
        self._x_min = np.zeros(len(self.variable_order), dtype=float)
        self._x_span = np.ones(len(self.variable_order), dtype=float)

    @abstractmethod
    def remaining(self) -> np.ndarray:
        raise NotImplementedError

    def sample_initial(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be >= 0")
        candidates = self.remaining()
        if n == 0 or len(candidates) == 0:
            return np.empty((0, len(self.variable_order)), dtype=float)
        count = min(int(n), len(candidates))
        positions = rng.choice(np.arange(len(candidates)), size=count, replace=False)
        return np.asarray(candidates[positions], dtype=float)

    @abstractmethod
    def remove(self, geometry: np.ndarray) -> None:
        raise NotImplementedError

    def normalize(self, geometry: np.ndarray) -> np.ndarray:
        X = self._as_matrix(geometry)
        Xn = (X - self._x_min) / self._x_span
        Xn = np.clip(Xn, 0.0, 1.0)
        return Xn[0] if np.asarray(geometry).ndim == 1 else Xn

    def denormalize(self, geometry_normalized: np.ndarray) -> np.ndarray:
        Xn = self._as_matrix(geometry_normalized)
        X = Xn * self._x_span + self._x_min
        return X[0] if np.asarray(geometry_normalized).ndim == 1 else X

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        raise NotImplementedError

    def _set_transform_from_bounds(self, lows: np.ndarray, highs: np.ndarray) -> None:
        lows = np.asarray(lows, dtype=float).reshape(-1)
        highs = np.asarray(highs, dtype=float).reshape(-1)
        if len(lows) != len(self.variable_order) or len(highs) != len(self.variable_order):
            raise ValueError("normalization bounds dimension mismatch")
        span = highs - lows
        span = np.where(np.abs(span) < 1e-12, 1.0, span)
        self._x_min = lows
        self._x_span = span

    def _set_transform_from_pool(self, pool: np.ndarray) -> None:
        matrix = self._validate_matrix(pool)
        self._set_transform_from_bounds(matrix.min(axis=0), matrix.max(axis=0))

    def _validate_matrix(self, rows: np.ndarray) -> np.ndarray:
        matrix = np.asarray(rows, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("candidate matrix must be 2D")
        if matrix.shape[1] != len(self.variable_order):
            raise ValueError(
                f"candidate dimension mismatch: got {matrix.shape[1]}, expected {len(self.variable_order)}"
            )
        return matrix

    def _as_matrix(self, rows: np.ndarray) -> np.ndarray:
        arr = np.asarray(rows, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return self._validate_matrix(arr)

    def _remove_from_pool(self, pool: np.ndarray, geometry: np.ndarray) -> np.ndarray:
        if len(pool) == 0:
            return pool
        target = np.asarray(geometry, dtype=float).reshape(-1)
        if target.shape[0] != pool.shape[1]:
            raise ValueError(f"geometry dimension mismatch: got {target.shape[0]}, expected {pool.shape[1]}")
        keep = [
            not np.allclose(row, target, atol=self.match_atol, rtol=self.match_rtol)
            for row in np.asarray(pool, dtype=float)
        ]
        return np.asarray(pool[keep], dtype=float)

    def _base_state(self) -> dict[str, Any]:
        return {
            "variable_order": list(self.variable_order),
            "match_atol": self.match_atol,
            "match_rtol": self.match_rtol,
            "x_min": self._x_min.tolist(),
            "x_span": self._x_span.tolist(),
        }

    def _load_base_state(self, payload: dict[str, Any]) -> None:
        if list(payload.get("variable_order", [])) != self.variable_order:
            raise ValueError("candidate-space variable_order mismatch")
        self._x_min = np.asarray(payload.get("x_min", self._x_min), dtype=float).reshape(-1)
        self._x_span = np.asarray(payload.get("x_span", self._x_span), dtype=float).reshape(-1)


class DiscreteCandidateSpace(CandidateSpace):
    """Finite candidate space for lookup/fake/replay-style domains."""

    def __init__(
        self,
        *,
        variable_order: list[str],
        candidate_geometries: np.ndarray,
        match_atol: float = 1e-9,
        match_rtol: float = 1e-9,
    ) -> None:
        super().__init__(variable_order=variable_order, match_atol=match_atol, match_rtol=match_rtol)
        self._remaining = self._validate_matrix(candidate_geometries)
        self._set_transform_from_pool(self._remaining)

    def remaining(self) -> np.ndarray:
        return np.asarray(self._remaining, dtype=float).copy()

    def remove(self, geometry: np.ndarray) -> None:
        self._remaining = self._remove_from_pool(self._remaining, geometry)

    def state_dict(self) -> dict[str, Any]:
        payload = self._base_state()
        payload.update(
            {
                "type": self.__class__.__name__,
                "remaining": self._remaining.tolist(),
            }
        )
        return payload

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self._load_base_state(payload)
        self._remaining = self._validate_matrix(np.asarray(payload.get("remaining", []), dtype=float))

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "variable_order": list(self.variable_order),
            "num_remaining": int(len(self._remaining)),
            "match_atol": self.match_atol,
            "match_rtol": self.match_rtol,
            "normalization": {
                "x_min": self._x_min.tolist(),
                "x_span": self._x_span.tolist(),
            },
        }


class ContinuousCandidateSpace(DiscreteCandidateSpace):
    """Finite candidate pool sampled from continuous bounds."""

    def __init__(
        self,
        *,
        variable_order: list[str],
        variable_bounds: dict[str, tuple[float, float]],
        pool_size: int,
        seed: int | None = None,
        resolution: float | dict[str, float] | None = None,
        match_atol: float = 1e-9,
        match_rtol: float = 1e-9,
    ) -> None:
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0")
        self.variable_bounds = {k: (float(v[0]), float(v[1])) for k, v in variable_bounds.items()}
        self.pool_size = int(pool_size)
        self.seed = seed
        self.resolution = resolution
        rows = self._sample_pool(variable_order=variable_order, rng=np.random.default_rng(seed))
        super().__init__(
            variable_order=variable_order,
            candidate_geometries=rows,
            match_atol=match_atol,
            match_rtol=match_rtol,
        )
        lows = np.asarray([self.variable_bounds[name][0] for name in self.variable_order], dtype=float)
        highs = np.asarray([self.variable_bounds[name][1] for name in self.variable_order], dtype=float)
        self._set_transform_from_bounds(lows, highs)

    def sample_initial(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0:
            return np.empty((0, len(self.variable_order)), dtype=float)
        return self.sample_uniform(n, rng)

    def sample_uniform(
        self,
        n: int,
        rng: np.random.Generator,
        *,
        bounds_normalized: np.ndarray | None = None,
    ) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0:
            return np.empty((0, len(self.variable_order)), dtype=float)
        bounds = self.normalized_bounds() if bounds_normalized is None else np.asarray(bounds_normalized, dtype=float)
        if bounds.shape != (2, len(self.variable_order)):
            raise ValueError("bounds_normalized shape mismatch")
        Xn = rng.uniform(bounds[0], bounds[1], size=(int(n), len(self.variable_order)))
        return np.asarray(self.project(self.denormalize(Xn)), dtype=float)

    def _sample_pool(self, *, variable_order: list[str], rng: np.random.Generator) -> np.ndarray:
        rows: list[list[float]] = []
        for _ in range(self.pool_size):
            row: list[float] = []
            for name in variable_order:
                if name not in self.variable_bounds:
                    raise ValueError(f"Missing bounds for variable {name}")
                low, high = self.variable_bounds[name]
                if low >= high:
                    raise ValueError(f"Invalid bounds for {name}: ({low}, {high})")
                value = float(rng.uniform(low, high))
                row.append(self._snap_value(name, value))
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def _snap_value(self, name: str, value: float) -> float:
        if self.resolution is None:
            return float(value)
        step = self.resolution.get(name) if isinstance(self.resolution, dict) else self.resolution
        if step is None:
            return float(value)
        step = float(step)
        if step <= 0:
            raise ValueError("resolution values must be > 0")
        return float(round(round(value / step) * step, 8))

    def project(self, geometry: np.ndarray) -> np.ndarray:
        """Clip physical geometry to bounds and snap to the configured resolution."""

        X = self._as_matrix(geometry)
        rows: list[list[float]] = []
        for row in X:
            projected_row: list[float] = []
            for idx, name in enumerate(self.variable_order):
                low, high = self.variable_bounds[name]
                value = float(np.clip(float(row[idx]), low, high))
                value = self._snap_value(name, value)
                projected_row.append(float(np.clip(value, low, high)))
            rows.append(projected_row)
        out = np.asarray(rows, dtype=float)
        return out[0] if np.asarray(geometry).ndim == 1 else out

    def normalized_bounds(self) -> np.ndarray:
        """Return the optimizer-facing normalized box for continuous acquisition."""

        return np.vstack(
            [
                np.zeros(len(self.variable_order), dtype=float),
                np.ones(len(self.variable_order), dtype=float),
            ]
        )

    def state_dict(self) -> dict[str, Any]:
        payload = super().state_dict()
        payload.update(
            {
                "type": self.__class__.__name__,
                "variable_bounds": self.variable_bounds,
                "pool_size": self.pool_size,
                "seed": self.seed,
                "resolution": self.resolution,
            }
        )
        return payload

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "variable_bounds": self.variable_bounds,
                "pool_size": self.pool_size,
                "seed": self.seed,
                "resolution": self.resolution,
            }
        )
        return payload
