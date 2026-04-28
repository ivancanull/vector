from __future__ import annotations

import pickle
from abc import ABC
from pathlib import Path
from typing import Any

import numpy as np

from .objectives import Objective, build_bclf_objective


class PassiveDataset(ABC):
    """Passive-device data container used by replay and online optimization."""

    def __init__(
        self,
        *,
        samples: list[dict[str, Any]],
        variable_order: list[str],
        freqs: np.ndarray,
        device_name: str | None = None,
        port_names: list[str] | None = None,
        default_train_indices: list[int] | None = None,
        default_candidate_indices: list[int] | None = None,
        default_test_indices: list[int] | None = None,
        match_atol: float = 1e-9,
        match_rtol: float = 1e-9,
    ) -> None:
        if not samples:
            raise ValueError("samples must be non-empty")
        if not variable_order:
            raise ValueError("variable_order must be non-empty")

        self.variable_order = list(variable_order)
        self.device_name = device_name
        self.port_names = list(port_names) if port_names is not None else None
        self.freqs = np.asarray(freqs, dtype=float).reshape(-1)
        if self.freqs.size == 0:
            raise ValueError("freqs must be non-empty")

        self.match_atol = float(match_atol)
        self.match_rtol = float(match_rtol)

        self.samples = [self.preprocess_sample(item) for item in samples]
        self.X_pool = np.asarray([item["geo_params"] for item in self.samples], dtype=float)
        self.S_pool = [np.asarray(item["s_params"], dtype=complex) for item in self.samples]

        n = len(self.samples)
        self.default_train_indices = self._normalize_indices(default_train_indices or [], n)
        if default_candidate_indices is None:
            inferred = [i for i in range(n) if i not in self.default_train_indices]
            self.default_candidate_indices = inferred
        else:
            self.default_candidate_indices = self._normalize_indices(default_candidate_indices, n)
        self.default_test_indices = self._normalize_indices(default_test_indices or [], n)

    @classmethod
    def from_pickle(
        cls,
        path: str | Path,
        *,
        variable_order: list[str],
        freqs: np.ndarray,
        device_name: str | None = None,
        port_names: list[str] | None = None,
        default_test_indices: list[int] | None = None,
        match_atol: float = 1e-9,
        match_rtol: float = 1e-9,
    ) -> "PassiveDataset":
        payload_path = Path(path)
        with payload_path.open("rb") as f:
            payload = pickle.load(f)

        samples, train_indices = cls._parse_payload(payload)
        return cls(
            samples=samples,
            variable_order=variable_order,
            freqs=freqs,
            device_name=device_name,
            port_names=port_names,
            default_train_indices=train_indices,
            default_test_indices=default_test_indices,
            match_atol=match_atol,
            match_rtol=match_rtol,
        )

    @staticmethod
    def _parse_payload(payload: Any) -> tuple[list[dict[str, Any]], list[int]]:
        if isinstance(payload, dict):
            if "dataset" not in payload:
                raise ValueError("Dataset payload dict must include 'dataset'")
            samples = payload["dataset"]
            train_indices = payload.get("history_idx") or payload.get("sample_idx") or []
        elif isinstance(payload, list):
            samples = payload
            train_indices = []
        else:
            raise ValueError("Unsupported dataset payload type")

        if not isinstance(samples, list) or not samples:
            raise ValueError("dataset must be a non-empty list")

        n = len(samples)
        train = [int(i) for i in train_indices if isinstance(i, int) and 0 <= int(i) < n]
        return samples, sorted(set(train))

    @staticmethod
    def _normalize_indices(indices: list[int], n: int) -> list[int]:
        normalized = sorted(set(int(i) for i in indices))
        for idx in normalized:
            if idx < 0 or idx >= n:
                raise IndexError(f"Dataset index out of range: {idx}")
        return normalized

    def preprocess_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        if "geo_params" not in sample or "s_params" not in sample:
            raise ValueError("Each sample must include 'geo_params' and 's_params'")
        return sample

    def __len__(self) -> int:
        return len(self.samples)

    def get_geometry(self, index: int) -> np.ndarray:
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Dataset index out of range: {index}")
        return np.asarray(self.X_pool[index], dtype=float)

    def get_case_id(self, index: int) -> str:
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Dataset index out of range: {index}")
        return str(self.samples[index].get("caseid", index))

    def find_index_by_geometry(self, geometry: np.ndarray) -> int:
        target = np.asarray(geometry, dtype=float).reshape(-1)
        for idx, candidate in enumerate(self.X_pool):
            if candidate.shape != target.shape:
                continue
            if np.allclose(candidate, target, atol=self.match_atol, rtol=self.match_rtol):
                return idx
        raise ValueError(f"No sample matches geometry={target.tolist()}")

    def get_response_by_geometry(self, geometry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = self.find_index_by_geometry(geometry)
        s_params = np.asarray(self.S_pool[idx], dtype=complex)
        if s_params.ndim != 3 or s_params.shape[2] != len(self.freqs):
            raise ValueError(
                f"Invalid s_params shape at index {idx}: expected (*, *, {len(self.freqs)}), got {s_params.shape}"
            )
        return np.asarray(self.freqs, dtype=float), s_params

    def append(self, sample: dict[str, Any]) -> int:
        item = self.preprocess_sample(sample)
        geo = np.asarray(item["geo_params"], dtype=float)
        s_params = np.asarray(item["s_params"], dtype=complex)
        if s_params.ndim != 3 or s_params.shape[2] != len(self.freqs):
            raise ValueError(
                f"Expected s_params frequency axis length {len(self.freqs)}, got shape {s_params.shape}"
            )

        self.samples.append(item)
        self.X_pool = np.vstack([self.X_pool, geo.reshape(1, -1)])
        self.S_pool.append(s_params)
        return len(self.samples) - 1


class BCLFDataset(PassiveDataset):
    """Thin BCLF-specific dataset adapter with sensible defaults."""

    DEFAULT_VARIABLE_ORDER = ["Din", "S", "W"]

    @staticmethod
    def default_objective() -> Objective:
        return build_bclf_objective()

    @classmethod
    def from_pickle(
        cls,
        path: str | Path,
        *,
        freqs: np.ndarray,
        variable_order: list[str] | None = None,
        device_name: str | None = "bclf",
        port_names: list[str] | None = None,
        default_test_indices: list[int] | None = None,
        match_atol: float = 1e-9,
        match_rtol: float = 1e-9,
    ) -> "BCLFDataset":
        return super().from_pickle(
            path,
            variable_order=variable_order or cls.DEFAULT_VARIABLE_ORDER,
            freqs=freqs,
            device_name=device_name,
            port_names=port_names,
            default_test_indices=default_test_indices,
            match_atol=match_atol,
            match_rtol=match_rtol,
        )
