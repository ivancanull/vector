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


def _fmt_band(band: tuple[float, float]) -> str:
    return f"{float(band[0]):.0f}_{float(band[1]):.0f}"
