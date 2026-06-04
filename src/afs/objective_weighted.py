from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from src.poles_init import basic_poles_initialize
    from src.utils import frequency_to_s
    from src.vector_fitting import evaluate, vector_fit
except ImportError:  # pragma: no cover - notebook/src path import style
    from poles_init import basic_poles_initialize
    from utils import frequency_to_s
    from vector_fitting import evaluate, vector_fit


BCLF_BOUNDARY_GHZ = np.asarray([3.6, 4.4, 4.8, 5.4, 5.8], dtype=float)


@dataclass(frozen=True)
class ObjectiveWeightedAFSConfig:
    sample_count: int = 20
    initial_anchor_count: int = 9
    min_spacing_indices: int = 2
    s11_weight: float = 0.5
    s21_weight: float = 1.0
    boundary_weight: float = 2.0
    passband_weight: float = 1.5
    stopband_weight: float = 1.25
    passband_ghz: tuple[float, float] = (4.8, 5.4)
    lower_stopband_ghz: tuple[float, float] = (3.6, 4.4)
    upper_stopband_ghz: tuple[float, float] = (5.8, 8.7)
    boundary_ghz: tuple[float, ...] = tuple(float(x) for x in BCLF_BOUNDARY_GHZ)
    boundary_window_indices: int = 2

    def __post_init__(self) -> None:
        if int(self.sample_count) <= 0:
            raise ValueError("sample_count must be > 0")
        if int(self.initial_anchor_count) <= 0:
            raise ValueError("initial_anchor_count must be > 0")
        if int(self.min_spacing_indices) < 0:
            raise ValueError("min_spacing_indices must be >= 0")


@dataclass(frozen=True)
class VFFitConfig:
    n_poles_real: int = 4
    n_poles_complex: int = 8
    max_iter: int = 30
    tol: float = 1e-7
    order_policy: str = "fixed"

    def order_for_sample_count(self, sample_count: int) -> dict[str, int | str]:
        sample_count = int(sample_count)
        if sample_count <= 0:
            raise ValueError("sample_count must be > 0")
        if self.order_policy == "fixed":
            n_real = int(self.n_poles_real)
            n_complex = int(self.n_poles_complex)
        elif self.order_policy == "pole_residue_budget":
            actual_poles = max(1, sample_count // 2)
            n_complex = actual_poles // 2
            n_real = actual_poles % 2
        else:
            raise ValueError(f"unknown VF order policy: {self.order_policy!r}")
        actual_poles = int(n_real + 2 * n_complex)
        return {
            "vf_order_policy": str(self.order_policy),
            "vf_actual_poles": actual_poles,
            "vf_n_poles_real": int(n_real),
            "vf_n_poles_complex": int(n_complex),
            "vf_pole_residue_terms": int(2 * actual_poles),
        }


@dataclass(frozen=True)
class ObjectiveWeightedAFSResult:
    indices: np.ndarray
    frequency_points_ghz: list[float]
    score: np.ndarray
    residual_component: np.ndarray
    uncertainty_component: np.ndarray
    band_weights: np.ndarray
    metadata: dict[str, Any]


def channel_db(s_params: np.ndarray, row: int, col: int) -> np.ndarray:
    mag = np.maximum(np.abs(np.asarray(s_params)[..., row, col, :]), 1e-12)
    return 20.0 * np.log10(mag)


def uncertainty_profile_from_samples(response_samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(response_samples, dtype=complex)
    if samples.ndim != 4:
        raise ValueError("response_samples must have shape (mc, ports, ports, n_freq)")
    s11 = channel_db(samples, 0, 0)
    s21 = channel_db(samples, 0, 1)
    return 0.5 * np.std(s11, axis=0) + np.std(s21, axis=0)


def objective_band_weights(freqs_hz: np.ndarray, config: ObjectiveWeightedAFSConfig | None = None) -> np.ndarray:
    config = config or ObjectiveWeightedAFSConfig()
    freqs_ghz = np.asarray(freqs_hz, dtype=float).reshape(-1) / 1e9
    weights = np.ones(freqs_ghz.shape, dtype=float)
    passband = (freqs_ghz >= config.passband_ghz[0]) & (freqs_ghz <= config.passband_ghz[1])
    low_stop = (freqs_ghz >= config.lower_stopband_ghz[0]) & (
        freqs_ghz <= config.lower_stopband_ghz[1]
    )
    high_stop = (freqs_ghz >= config.upper_stopband_ghz[0]) & (
        freqs_ghz <= config.upper_stopband_ghz[1]
    )
    weights[passband] = np.maximum(weights[passband], float(config.passband_weight))
    weights[low_stop | high_stop] = np.maximum(weights[low_stop | high_stop], float(config.stopband_weight))
    for freq_ghz in config.boundary_ghz:
        center = nearest_frequency_index(freqs_hz, float(freq_ghz) * 1e9)
        lo = max(0, center - int(config.boundary_window_indices))
        hi = min(len(weights), center + int(config.boundary_window_indices) + 1)
        weights[lo:hi] = np.maximum(weights[lo:hi], float(config.boundary_weight))
    return weights


def nearest_frequency_index(freqs_hz: np.ndarray, target_hz: float) -> int:
    freqs = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if freqs.size == 0:
        raise ValueError("freqs_hz must be non-empty")
    return int(np.argmin(np.abs(freqs - float(target_hz))))


def objective_weighted_anchor_indices(
    freqs_hz: np.ndarray,
    n_points: int,
    config: ObjectiveWeightedAFSConfig | None = None,
) -> np.ndarray:
    config = config or ObjectiveWeightedAFSConfig()
    freqs = np.asarray(freqs_hz, dtype=float).reshape(-1)
    n_points = int(n_points)
    if n_points > len(freqs):
        raise ValueError("n_points exceeds available frequencies")
    anchor_count = min(n_points, int(config.initial_anchor_count))
    priority = _unique_preserving_order(
        [0, len(freqs) - 1]
        + [nearest_frequency_index(freqs, float(freq_ghz) * 1e9) for freq_ghz in config.boundary_ghz]
        + _uniform_fill_indices(len(freqs), max(anchor_count, 2))
    )
    selected = priority[:anchor_count]
    if len(selected) < anchor_count:
        for idx in _uniform_fill_indices(len(freqs), len(freqs)):
            if idx not in selected:
                selected.append(idx)
            if len(selected) == anchor_count:
                break
    return np.asarray(sorted(selected), dtype=int)


def select_objective_weighted_afs_indices(
    freqs_hz: np.ndarray,
    proxy_s_params: np.ndarray,
    uncertainty_profile: np.ndarray,
    *,
    n_points: int | None = None,
    config: ObjectiveWeightedAFSConfig | None = None,
    vf_config: VFFitConfig | None = None,
) -> ObjectiveWeightedAFSResult:
    config = config or ObjectiveWeightedAFSConfig()
    vf_config = vf_config or VFFitConfig()
    freqs = np.asarray(freqs_hz, dtype=float).reshape(-1)
    proxy = np.asarray(proxy_s_params, dtype=complex)
    uncertainty = np.asarray(uncertainty_profile, dtype=float).reshape(-1)
    n_points = int(config.sample_count if n_points is None else n_points)
    if proxy.ndim != 3 or proxy.shape[-1] != len(freqs):
        raise ValueError("proxy_s_params must have shape (ports, ports, n_freq)")
    if uncertainty.shape != freqs.shape:
        raise ValueError("uncertainty_profile must have one value per frequency")
    if n_points > len(freqs):
        raise ValueError("n_points exceeds available frequencies")

    selected = set(int(i) for i in objective_weighted_anchor_indices(freqs, n_points, config))
    band_weights = objective_band_weights(freqs, config)
    last_score = np.zeros(len(freqs), dtype=float)
    last_residual = np.zeros(len(freqs), dtype=float)
    last_uncertainty = np.zeros(len(freqs), dtype=float)
    vf_failures = 0

    while len(selected) < n_points:
        current = np.asarray(sorted(selected), dtype=int)
        unsampled = np.asarray([idx for idx in range(len(freqs)) if idx not in selected], dtype=int)
        distances = np.min(np.abs(unsampled[:, None] - current[None, :]), axis=1)
        candidates = unsampled[distances >= int(config.min_spacing_indices)]
        if candidates.size == 0:
            candidates = unsampled

        try:
            pred = reconstruct_sparams_vf(
                freqs[current],
                proxy[:, :, current],
                freqs,
                config=vf_config,
            )
            residual = objective_vf_residual_db(proxy, pred, config)
        except Exception:
            # Early low-count VF can be numerically fragile; keep OWAFS moving by
            # falling back to the uncertainty term for this one acquisition step.
            vf_failures += 1
            residual = np.zeros(len(freqs), dtype=float)

        residual_component = np.zeros(len(freqs), dtype=float)
        uncertainty_component = np.zeros(len(freqs), dtype=float)
        residual_component[candidates] = _rank01(residual[candidates])
        uncertainty_component[candidates] = _rank01(np.maximum(uncertainty[candidates], 0.0))
        score = np.zeros(len(freqs), dtype=float)
        score[candidates] = band_weights[candidates] * 0.5 * (
            residual_component[candidates] + uncertainty_component[candidates]
        )
        next_idx = int(candidates[np.argmax(score[candidates])])
        selected.add(next_idx)
        last_score = score
        last_residual = residual_component
        last_uncertainty = uncertainty_component

    indices = np.asarray(sorted(selected), dtype=int)
    return ObjectiveWeightedAFSResult(
        indices=indices,
        frequency_points_ghz=[float(freqs[i] / 1e9) for i in indices],
        score=last_score,
        residual_component=last_residual,
        uncertainty_component=last_uncertainty,
        band_weights=band_weights,
        metadata={
            "method": "owafs_bnn",
            "sample_count": int(n_points),
            "initial_anchor_count": int(config.initial_anchor_count),
            "min_spacing_indices": int(config.min_spacing_indices),
            "vf_failures_during_selection": int(vf_failures),
            "score_formula": "band_weight * 0.5 * (rank01(vf_residual_db) + rank01(mc_uncertainty_db))",
            "band_weights": {
                "boundary": float(config.boundary_weight),
                "passband": float(config.passband_weight),
                "stopband": float(config.stopband_weight),
                "else": 1.0,
            },
        },
    )


def objective_vf_residual_db(
    proxy_s_params: np.ndarray,
    reconstructed_s_params: np.ndarray,
    config: ObjectiveWeightedAFSConfig | None = None,
) -> np.ndarray:
    config = config or ObjectiveWeightedAFSConfig()
    proxy = np.asarray(proxy_s_params, dtype=complex)
    pred = np.asarray(reconstructed_s_params, dtype=complex)
    s11_err = np.abs(channel_db(proxy, 0, 0) - channel_db(pred, 0, 0))
    s21_err = np.abs(channel_db(proxy, 0, 1) - channel_db(pred, 0, 1))
    return float(config.s11_weight) * s11_err + float(config.s21_weight) * s21_err


def reconstruct_sparams_vf(
    sample_freqs_hz: np.ndarray,
    sample_s_params: np.ndarray,
    target_freqs_hz: np.ndarray,
    *,
    config: VFFitConfig | None = None,
) -> np.ndarray:
    config = config or VFFitConfig()
    sample_freqs = np.asarray(sample_freqs_hz, dtype=float).reshape(-1)
    target_freqs = np.asarray(target_freqs_hz, dtype=float).reshape(-1)
    samples = np.asarray(sample_s_params, dtype=complex)
    if samples.ndim != 3 or samples.shape[-1] != len(sample_freqs):
        raise ValueError("sample_s_params must have shape (ports, ports, n_sample_freq)")
    if target_freqs.size == 0:
        raise ValueError("target_freqs_hz must be non-empty")
    ports = int(samples.shape[0])
    if samples.shape[1] != ports:
        raise ValueError("sample_s_params must be square in port dimensions")

    order = config.order_for_sample_count(len(sample_freqs))
    init_poles = basic_poles_initialize(
        freqs_range=(float(target_freqs[0]), float(target_freqs[-1])),
        n_poles_real=int(order["vf_n_poles_real"]),
        n_poles_complex=int(order["vf_n_poles_complex"]),
    )
    pred = np.zeros((ports, ports, len(target_freqs)), dtype=complex)
    for row in range(ports):
        for col in range(ports):
            poles, residues, d, h = vector_fit(
                sample_freqs,
                samples[row, col, :],
                init_poles=init_poles,
                max_iter=int(config.max_iter),
                tol=float(config.tol),
                atol=float(config.tol),
                rtol=float(config.tol),
            )
            pred[row, col, :] = evaluate(poles, residues, d, h, frequency_to_s(target_freqs))
    return pred


def write_touchstone_ri(
    path: str | Path,
    freqs_hz: np.ndarray,
    s_params: np.ndarray,
    *,
    comment: str | None = None,
) -> Path:
    path = Path(path)
    freqs = np.asarray(freqs_hz, dtype=float).reshape(-1)
    arr = np.asarray(s_params, dtype=complex)
    if arr.ndim != 3 or arr.shape[-1] != len(freqs):
        raise ValueError("s_params must have shape (ports, ports, n_freq)")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("s_params must be square in port dimensions")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        if comment:
            fh.write(f"! {comment}\n")
        fh.write("# Hz S RI R 50\n")
        for pos, freq in enumerate(freqs):
            values = [f"{float(freq):.12g}"]
            for row in range(arr.shape[0]):
                for col in range(arr.shape[1]):
                    value = arr[row, col, pos]
                    values.append(f"{float(np.real(value)):.12g}")
                    values.append(f"{float(np.imag(value)):.12g}")
            fh.write(" ".join(values) + "\n")
    return path


def _uniform_fill_indices(n_freqs: int, n_points: int) -> list[int]:
    return [int(i) for i in np.round(np.linspace(0, int(n_freqs) - 1, int(n_points))).astype(int)]


def _unique_preserving_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        idx = int(value)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def _rank01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=float)
    safe = np.where(finite, arr, np.nanmin(arr[finite]))
    if arr.size == 1 or float(np.max(safe) - np.min(safe)) <= 1e-12:
        return np.zeros(arr.shape, dtype=float)
    unique = np.unique(safe)
    if unique.size == 1:
        return np.zeros(arr.shape, dtype=float)
    rank_by_value = {float(value): pos / float(unique.size - 1) for pos, value in enumerate(unique)}
    return np.asarray([rank_by_value[float(value)] for value in safe], dtype=float)
