from __future__ import annotations

from pathlib import Path

import numpy as np

import afs.objective_weighted as owafs


def _response(n_freq: int) -> np.ndarray:
    s = np.zeros((2, 2, n_freq), dtype=complex)
    x = np.linspace(0.0, 1.0, n_freq)
    s[0, 0, :] = 0.1 + 0.02 * x
    s[0, 1, :] = 0.8 - 0.1 * x
    s[1, 0, :] = s[0, 1, :]
    s[1, 1, :] = s[0, 0, :]
    return s


def test_objective_weighted_anchors_include_endpoints_and_bclf_boundaries() -> None:
    freqs = np.linspace(3.5e9, 8.7e9, 401)
    indices = owafs.objective_weighted_anchor_indices(freqs, 20)
    selected_ghz = {round(float(freqs[i] / 1e9), 3) for i in indices}

    assert len(indices) == 9
    assert indices[0] == 0
    assert indices[-1] == 400
    for boundary in [3.6, 4.4, 4.8, 5.4, 5.8]:
        nearest = round(float(freqs[owafs.nearest_frequency_index(freqs, boundary * 1e9)] / 1e9), 3)
        assert nearest in selected_ghz


def test_selector_uses_residual_and_uncertainty_terms(monkeypatch) -> None:
    freqs = np.linspace(3.5e9, 8.7e9, 10)
    proxy = np.zeros((2, 2, 10), dtype=complex)
    proxy[0, 0, :] = 0.1
    proxy[0, 1, :] = 0.1
    proxy[0, 1, 4] = 1.0
    uncertainty = np.zeros(10, dtype=float)
    uncertainty[6] = 10.0

    def _zero_reconstruction(sample_freqs, sample_s, target_freqs, *, config=None):  # type: ignore[no-untyped-def]
        return np.zeros((2, 2, len(target_freqs)), dtype=complex)

    monkeypatch.setattr(owafs, "reconstruct_sparams_vf", _zero_reconstruction)
    result = owafs.select_objective_weighted_afs_indices(
        freqs,
        proxy,
        uncertainty,
        n_points=4,
        config=owafs.ObjectiveWeightedAFSConfig(
            sample_count=4,
            initial_anchor_count=2,
            min_spacing_indices=0,
            boundary_ghz=(),
        ),
    )

    assert result.indices.tolist() == [0, 4, 6, 9]
    assert result.metadata["score_formula"].startswith("band_weight")


def test_touchstone_writer_emits_parseable_ri_file(tmp_path: Path) -> None:
    freqs = np.asarray([1.0e9, 2.0e9], dtype=float)
    s_params = _response(len(freqs))
    path = owafs.write_touchstone_ri(tmp_path / "demo.s2p", freqs, s_params, comment="demo")

    text = path.read_text(encoding="utf-8")
    assert "# Hz S RI R 50" in text
    assert len([line for line in text.splitlines() if line and not line.startswith(("!", "#"))]) == 2
