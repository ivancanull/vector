from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from src.complex_func import FunctionFromTouchstone
except ImportError:  # pragma: no cover - notebook/src path import style
    from complex_func import FunctionFromTouchstone


DEFAULT_BCLF_VARIABLE_ORDER = ["L", "L1", "L2", "W", "W1", "W2", "S1", "S2"]


@dataclass(frozen=True)
class BCLFResponseDataset:
    """BCLF geometry/response matrix loaded from cached Touchstone files."""

    variable_order: list[str]
    freqs: np.ndarray
    X: np.ndarray
    s_params: np.ndarray
    filenames: list[str]
    keys: list[str]
    case_ids: list[str]

    def __post_init__(self) -> None:
        freqs = np.asarray(self.freqs, dtype=float).reshape(-1)
        X = np.asarray(self.X, dtype=float)
        s_params = np.asarray(self.s_params, dtype=complex)
        if X.ndim != 2:
            raise ValueError("X must be a 2D geometry matrix")
        if X.shape[1] != len(self.variable_order):
            raise ValueError("X dimension must match variable_order")
        if s_params.ndim != 4 or s_params.shape[0] != X.shape[0] or s_params.shape[-1] != len(freqs):
            raise ValueError("s_params must have shape (n, ports, ports, n_freq)")
        object.__setattr__(self, "freqs", freqs)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "s_params", s_params)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_freqs(self) -> int:
        return int(self.freqs.shape[0])

    def subset(self, indices: Iterable[int]) -> "BCLFResponseDataset":
        idx = np.asarray(list(indices), dtype=int)
        return BCLFResponseDataset(
            variable_order=list(self.variable_order),
            freqs=self.freqs,
            X=self.X[idx],
            s_params=self.s_params[idx],
            filenames=[self.filenames[i] for i in idx],
            keys=[self.keys[i] for i in idx],
            case_ids=[self.case_ids[i] for i in idx],
        )


class BCLFTouchstoneDatasetLoader:
    """Independent modeling-layer loader for BCLF cache directories."""

    def __init__(
        self,
        *,
        dataset_dir: str | Path,
        variable_order: list[str] | None = None,
        index_filename: str = "index.json",
        port_num: int = 2,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.variable_order = list(variable_order or DEFAULT_BCLF_VARIABLE_ORDER)
        self.index_filename = index_filename
        self.port_num = int(port_num)

    def load(self, *, limit: int | None = None) -> BCLFResponseDataset:
        index_path = self.dataset_dir / self.index_filename
        if not index_path.exists():
            raise FileNotFoundError(f"Missing cache index: {index_path}")
        with index_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        records = payload.get("records", {})
        if not isinstance(records, dict) or not records:
            raise ValueError(f"No records found in {index_path}")

        parser = FunctionFromTouchstone(port_num=self.port_num)
        X_rows: list[list[float]] = []
        s_rows: list[np.ndarray] = []
        filenames: list[str] = []
        keys: list[str] = []
        case_ids: list[str] = []
        freqs_ref: np.ndarray | None = None

        for key, record in records.items():
            if limit is not None and len(X_rows) >= int(limit):
                break
            if not isinstance(record, dict):
                continue
            variables = record.get("variables", {})
            filename = str(record.get("filename", ""))
            if not isinstance(variables, dict) or not filename:
                continue
            path = self.dataset_dir / filename
            if not path.exists():
                continue

            parser.parse(str(path))
            freqs = np.asarray(parser.freq, dtype=float).reshape(-1)
            if freqs_ref is None:
                freqs_ref = freqs
            elif freqs.shape != freqs_ref.shape or not np.allclose(freqs, freqs_ref):
                raise ValueError(f"Frequency grid mismatch in {path}")

            X_rows.append([float(variables[name]) for name in self.variable_order])
            s_rows.append(np.asarray(parser.evaluate_all(), dtype=complex))
            filenames.append(filename)
            keys.append(str(key))
            case_ids.append(str(record.get("case_id", len(case_ids))))

        if freqs_ref is None or not X_rows:
            raise ValueError(f"No usable Touchstone records found in {self.dataset_dir}")

        return BCLFResponseDataset(
            variable_order=list(self.variable_order),
            freqs=freqs_ref,
            X=np.asarray(X_rows, dtype=float),
            s_params=np.asarray(s_rows, dtype=complex),
            filenames=filenames,
            keys=keys,
            case_ids=case_ids,
        )


def response_db(s_params: np.ndarray, row: int, col: int) -> np.ndarray:
    mag = np.maximum(np.abs(np.asarray(s_params)[..., row, col, :]), 1e-12)
    return 20.0 * np.log10(mag)
