from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SParameterResponse:
    """Container for one simulated response in dB-domain curves."""

    freq_hz: np.ndarray
    s_db: dict[str, np.ndarray]
    s_params: np.ndarray | None = None

    def __post_init__(self) -> None:
        freq = np.asarray(self.freq_hz, dtype=float).reshape(-1)
        if freq.size == 0:
            raise ValueError("freq_hz must be non-empty")

        normalized: dict[str, np.ndarray] = {}
        for name, values in self.s_db.items():
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size != freq.size:
                raise ValueError(f"{name} length {arr.size} does not match freq_hz length {freq.size}")
            normalized[str(name).upper()] = arr

        object.__setattr__(self, "freq_hz", freq)
        object.__setattr__(self, "s_db", normalized)
        if self.s_params is not None:
            raw = _normalize_sparams_shape(self.s_params, freq_count=freq.size)
            object.__setattr__(self, "s_params", raw)

    @classmethod
    def from_sparams(
        cls,
        freq_hz: np.ndarray,
        s_params: np.ndarray,
        *,
        port_map: dict[str, tuple[int, int]] | None = None,
    ) -> "SParameterResponse":
        freq = np.asarray(freq_hz, dtype=float).reshape(-1)
        s_vals = _normalize_sparams_shape(s_params, freq_count=freq.size)

        mapping = port_map or {
            "S11": (0, 0),
            "S21": (1, 0),
            "S12": (0, 1),
            "S22": (1, 1),
        }

        curves: dict[str, np.ndarray] = {}
        for name, (i, j) in mapping.items():
            ii = int(i)
            jj = int(j)
            if ii < 0 or jj < 0 or ii >= s_vals.shape[0] or jj >= s_vals.shape[1]:
                continue
            mag = np.maximum(np.abs(s_vals[ii, jj]), 1e-12)
            curves[str(name).upper()] = 20.0 * np.log10(mag)

        if not curves:
            raise ValueError("No valid S-parameter curves available from provided port_map")
        return cls(freq_hz=freq, s_db=curves, s_params=s_vals)

    def get_db(self, name: str) -> np.ndarray:
        key = str(name).upper()
        if key not in self.s_db:
            raise KeyError(f"Missing S-parameter curve: {key}")
        return self.s_db[key]

    def band_freqs(self, band_hz: tuple[float, float]) -> np.ndarray:
        lo, hi = _validate_band(band_hz)
        mask = (self.freq_hz >= lo) & (self.freq_hz <= hi)
        if not np.any(mask):
            raise ValueError(f"No samples inside band {band_hz}")
        return self.freq_hz[mask]

    def band_values(self, name: str, band_hz: tuple[float, float]) -> np.ndarray:
        values = self.get_db(name)
        lo, hi = _validate_band(band_hz)
        mask = (self.freq_hz >= lo) & (self.freq_hz <= hi)
        if not np.any(mask):
            raise ValueError(f"No samples inside band {band_hz}")
        return values[mask]


def _validate_band(band_hz: tuple[float, float]) -> tuple[float, float]:
    if len(band_hz) != 2:
        raise ValueError("band_hz must contain exactly two values")
    lo, hi = float(band_hz[0]), float(band_hz[1])
    if lo >= hi:
        raise ValueError("band_hz must satisfy low < high")
    return lo, hi


def _normalize_sparams_shape(s_params: np.ndarray, *, freq_count: int) -> np.ndarray:
    values = np.asarray(s_params, dtype=complex)
    if values.ndim != 3:
        raise ValueError("s_params must be a 3D array shaped (ports, ports, freqs)")
    if values.shape[2] == freq_count:
        return values
    if values.shape[0] == freq_count and values.shape[1] == values.shape[2]:
        return np.moveaxis(values, 0, 2)
    raise ValueError(
        f"s_params frequency axis length {values.shape[2]} does not match freq_hz length {freq_count}"
    )
