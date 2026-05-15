from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .core import SimulationError, SimulationInput, SimulationOutput, Simulator


class DatasetSimulator(Simulator):
    """Virtual simulator backed by existing generated S-parameter data."""

    def __init__(
        self,
        *,
        dataset_path: str | Path,
        freqs: np.ndarray,
        port_num: int,
        param_start_idx: int,
        variable_order: list[str],
        match_atol: float = 1e-9,
        match_rtol: float = 1e-9,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.freqs = np.asarray(freqs, dtype=float)
        self.port_num = port_num
        self.param_start_idx = param_start_idx
        self.variable_order = list(variable_order)
        self.match_atol = match_atol
        self.match_rtol = match_rtol
        self.dataset = self._load_dataset()

    def simulate(self, request: SimulationInput) -> SimulationOutput:
        expected_order = list(self.variable_order)
        if list(request.variable_order) != expected_order:
            raise SimulationError(
                request.case_id,
                "invalid_variables",
                f"Expected variable order {expected_order}, got {request.variable_order}",
            )

        target_geo_params = request.geo_params()
        sample = self._find_sample(request.case_id, target_geo_params)
        return SimulationOutput(
            case_id=request.case_id,
            geo_params=np.asarray(sample["geo_params"], dtype=float),
            freqs=self.freqs,
            s_params=self._normalize_s_params(request.case_id, sample["s_params"]),
            artifacts=None,
        )

    def _load_dataset(self) -> list[dict[str, Any]]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        if self.dataset_path.is_file():
            with self.dataset_path.open("rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and "dataset" in payload:
                payload = payload["dataset"]
            if not isinstance(payload, list):
                raise ValueError(f"Unsupported dataset pickle payload: {self.dataset_path}")
            return payload

        # Support both import styles used across scripts/notebooks.
        try:
            from src.complex_func import FunctionFromTouchstone
        except ImportError:
            from complex_func import FunctionFromTouchstone
        from loader import load_dataset

        parser = FunctionFromTouchstone(port_num=self.port_num)
        return load_dataset(
            str(self.dataset_path),
            self.port_num,
            parser,
            self.param_start_idx,
            force_regen=False,
            random_sample=False,
        )

    def _find_sample(self, case_id: str, target_geo_params: np.ndarray) -> dict[str, Any]:
        for sample in self.dataset:
            geo_params = np.asarray(sample.get("geo_params"), dtype=float)
            if geo_params.shape != target_geo_params.shape:
                continue
            if np.allclose(
                geo_params,
                target_geo_params,
                atol=self.match_atol,
                rtol=self.match_rtol,
            ):
                return sample

        raise SimulationError(
            case_id,
            "sample_not_found",
            f"No generated sample matches geo_params={target_geo_params.tolist()}",
        )

    def _normalize_s_params(self, case_id: str, s_params: np.ndarray) -> np.ndarray:
        values = np.asarray(s_params, dtype=complex)
        expected_shape = (self.port_num, self.port_num, len(self.freqs))
        if values.shape != expected_shape:
            raise SimulationError(
                case_id,
                "invalid_sample",
                f"Expected s_params shape {expected_shape}, got {values.shape}",
            )
        return values
