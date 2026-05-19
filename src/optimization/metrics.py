from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sparam_response import SParameterResponse


class SParamMetric:
    name: str

    def compute(self, response: SParameterResponse) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class AtFrequencyMetric(SParamMetric):
    s_name: str
    freq_hz: float

    @property
    def name(self) -> str:
        return f"{self.s_name.lower()}_at_{self.freq_hz:.0f}hz"

    def compute(self, response: SParameterResponse) -> float:
        freq = response.freq_hz
        idx = int(np.argmin(np.abs(freq - float(self.freq_hz))))
        return float(response.get_db(self.s_name)[idx])


@dataclass(frozen=True)
class BandMeanMetric(SParamMetric):
    s_name: str
    band_hz: tuple[float, float]

    @property
    def name(self) -> str:
        return f"{self.s_name.lower()}_mean_{_fmt_band(self.band_hz)}"

    def compute(self, response: SParameterResponse) -> float:
        return float(np.mean(response.band_values(self.s_name, self.band_hz)))


@dataclass(frozen=True)
class BandMaxMetric(SParamMetric):
    s_name: str
    band_hz: tuple[float, float] | None = None

    @property
    def name(self) -> str:
        if self.band_hz is None:
            return f"{self.s_name.lower()}_max"
        return f"{self.s_name.lower()}_max_{_fmt_band(self.band_hz)}"

    def compute(self, response: SParameterResponse) -> float:
        vals = response.get_db(self.s_name) if self.band_hz is None else response.band_values(self.s_name, self.band_hz)
        return float(np.max(vals))


@dataclass(frozen=True)
class BandMinMetric(SParamMetric):
    s_name: str
    band_hz: tuple[float, float] | None = None

    @property
    def name(self) -> str:
        if self.band_hz is None:
            return f"{self.s_name.lower()}_min"
        return f"{self.s_name.lower()}_min_{_fmt_band(self.band_hz)}"

    def compute(self, response: SParameterResponse) -> float:
        vals = response.get_db(self.s_name) if self.band_hz is None else response.band_values(self.s_name, self.band_hz)
        return float(np.min(vals))


@dataclass(frozen=True)
class BandRippleMetric(SParamMetric):
    s_name: str
    band_hz: tuple[float, float]

    @property
    def name(self) -> str:
        return f"{self.s_name.lower()}_ripple_{_fmt_band(self.band_hz)}"

    def compute(self, response: SParameterResponse) -> float:
        vals = response.band_values(self.s_name, self.band_hz)
        return float(np.max(vals) - np.min(vals))


@dataclass(frozen=True)
class PeakFrequencyMetric(SParamMetric):
    s_name: str
    search_band_hz: tuple[float, float]

    @property
    def name(self) -> str:
        return f"{self.s_name.lower()}_peak_freq_{_fmt_band(self.search_band_hz)}"

    def compute(self, response: SParameterResponse) -> float:
        freqs = response.band_freqs(self.search_band_hz)
        vals = response.band_values(self.s_name, self.search_band_hz)
        pos = int(np.argmax(vals))
        return float(freqs[pos])


@dataclass(frozen=True)
class UnionBandMeanMetric(SParamMetric):
    s_name: str
    bands_hz: tuple[tuple[float, float], tuple[float, float]]

    @property
    def name(self) -> str:
        b0, b1 = self.bands_hz
        return f"{self.s_name.lower()}_mean_union_{_fmt_band(b0)}__{_fmt_band(b1)}"

    def compute(self, response: SParameterResponse) -> float:
        vals0 = response.band_values(self.s_name, self.bands_hz[0])
        vals1 = response.band_values(self.s_name, self.bands_hz[1])
        vals = np.concatenate([vals0, vals1], axis=0)
        return float(np.mean(vals))


@dataclass(frozen=True)
class UnionBandMaxMetric(SParamMetric):
    s_name: str
    bands_hz: tuple[tuple[float, float], tuple[float, float]]

    @property
    def name(self) -> str:
        b0, b1 = self.bands_hz
        return f"{self.s_name.lower()}_max_union_{_fmt_band(b0)}__{_fmt_band(b1)}"

    def compute(self, response: SParameterResponse) -> float:
        vals0 = response.band_values(self.s_name, self.bands_hz[0])
        vals1 = response.band_values(self.s_name, self.bands_hz[1])
        vals = np.concatenate([vals0, vals1], axis=0)
        return float(np.max(vals))


@dataclass(frozen=True)
class ResponseShapeEnvelopeSegment:
    """One frequency interval with optional linear lower and upper dB envelopes."""

    band_hz: tuple[float, float]
    lower_start_db: float | None = None
    lower_end_db: float | None = None
    upper_start_db: float | None = None
    upper_end_db: float | None = None

    def __post_init__(self) -> None:
        lo, hi = _validate_band(self.band_hz)
        object.__setattr__(self, "band_hz", (lo, hi))
        has_lower = self.lower_start_db is not None or self.lower_end_db is not None
        has_upper = self.upper_start_db is not None or self.upper_end_db is not None
        if not has_lower and not has_upper:
            raise ValueError("ResponseShapeEnvelopeSegment requires at least one envelope bound")
        if (self.lower_start_db is None) != (self.lower_end_db is None):
            raise ValueError("lower envelope requires both start and end values")
        if (self.upper_start_db is None) != (self.upper_end_db is None):
            raise ValueError("upper envelope requires both start and end values")

    def mask(self, freq_hz: np.ndarray) -> np.ndarray:
        lo, hi = self.band_hz
        return (freq_hz >= lo) & (freq_hz <= hi)

    def lower_at(self, freq_hz: np.ndarray) -> np.ndarray | None:
        if self.lower_start_db is None or self.lower_end_db is None:
            return None
        return self._linear_at(freq_hz, float(self.lower_start_db), float(self.lower_end_db))

    def upper_at(self, freq_hz: np.ndarray) -> np.ndarray | None:
        if self.upper_start_db is None or self.upper_end_db is None:
            return None
        return self._linear_at(freq_hz, float(self.upper_start_db), float(self.upper_end_db))

    def _linear_at(self, freq_hz: np.ndarray, start_db: float, end_db: float) -> np.ndarray:
        lo, hi = self.band_hz
        t = (freq_hz - lo) / (hi - lo)
        return start_db + t * (end_db - start_db)


@dataclass(frozen=True)
class ResponseShapeEnvelopeMetric(SParamMetric):
    s_name: str
    segments: tuple[ResponseShapeEnvelopeSegment, ...]
    label: str = "response_shape"
    aggregation: str = "mean_squared_hinge"

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("ResponseShapeEnvelopeMetric requires at least one segment")
        if self.aggregation != "mean_squared_hinge":
            raise ValueError("Only mean_squared_hinge aggregation is supported")
        object.__setattr__(self, "segments", tuple(self.segments))

    @property
    def name(self) -> str:
        return f"{self.s_name.lower()}_{self.label}_{self.aggregation}"

    def compute(self, response: SParameterResponse) -> float:
        freq = response.freq_hz
        values = response.get_db(self.s_name)
        violations: list[np.ndarray] = []

        for segment in self.segments:
            mask = segment.mask(freq)
            if not np.any(mask):
                continue
            segment_freq = freq[mask]
            segment_values = values[mask]

            lower = segment.lower_at(segment_freq)
            if lower is not None:
                violations.append(np.maximum(0.0, lower - segment_values))

            upper = segment.upper_at(segment_freq)
            if upper is not None:
                violations.append(np.maximum(0.0, segment_values - upper))

        if not violations:
            raise ValueError(f"No samples inside response-shape envelope for {self.s_name}")
        all_violations = np.concatenate(violations, axis=0)
        return float(np.mean(np.square(all_violations)))


def _fmt_band(band: tuple[float, float]) -> str:
    return f"{float(band[0]):.0f}_{float(band[1]):.0f}"


def _validate_band(band_hz: tuple[float, float]) -> tuple[float, float]:
    if len(band_hz) != 2:
        raise ValueError("band_hz must contain exactly two values")
    lo, hi = float(band_hz[0]), float(band_hz[1])
    if lo >= hi:
        raise ValueError("band_hz must satisfy low < high")
    return lo, hi
