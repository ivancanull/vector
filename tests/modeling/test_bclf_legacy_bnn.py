from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from modeling import (
    BCLFResponseDataset,
    BCLFTouchstoneDatasetLoader,
    FourierBNNResponseSurrogate,
    LegacyBNNResponseSurrogate,
)


def _synthetic_dataset(n: int = 5, n_freqs: int = 3) -> BCLFResponseDataset:
    freqs = np.linspace(1.0, 3.0, n_freqs)
    base = np.linspace(0.0, 1.0, n)
    X = np.column_stack(
        [
            7.7 + 0.2 * base,
            8.0 + 0.2 * base,
            7.8 + 0.2 * base,
            0.9 + 0.2 * base,
            0.7 + 0.2 * base,
            0.9 + 0.2 * base,
            0.13 + 0.03 * base,
            0.55 + 0.05 * base,
        ]
    )
    s_params = np.zeros((n, 2, 2, n_freqs), dtype=complex)
    for idx in range(n):
        curve = (idx + 1) * 0.01 + np.linspace(0.0, 0.02, n_freqs)
        s_params[idx, 0, 0, :] = curve + 0.1j * curve
        s_params[idx, 0, 1, :] = 0.5 * curve + 0.2j * curve
        s_params[idx, 1, 0, :] = s_params[idx, 0, 1, :]
    return BCLFResponseDataset(
        variable_order=["L", "L1", "L2", "W", "W1", "W2", "S1", "S2"],
        freqs=freqs,
        X=X,
        s_params=s_params,
        filenames=[f"case_{i}.s2p" for i in range(n)],
        keys=[f"k{i}" for i in range(n)],
        case_ids=[f"c{i}" for i in range(n)],
    )


def test_loader_reads_index_records_with_touchstone_parser(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ext"
    dataset_dir.mkdir()
    (dataset_dir / "case.s2p").write_text("dummy", encoding="utf-8")
    (dataset_dir / "index.json").write_text(
        json.dumps(
            {
                "records": {
                    "case": {
                        "case_id": "case0",
                        "variables": {
                            "L": 7.7,
                            "L1": 8.0,
                            "L2": 7.8,
                            "W": 0.9,
                            "W1": 0.7,
                            "W2": 0.9,
                            "S1": 0.13,
                            "S2": 0.55,
                        },
                        "filename": "case.s2p",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeParser:
        def __init__(self, port_num):
            self.port_num = port_num
            self.freq = np.asarray([1.0, 2.0])

        def parse(self, filename):
            self.filename = filename

        def evaluate_all(self):
            return np.ones((2, 2, 2), dtype=complex)

    import modeling.bclf_dataset as module

    monkeypatch.setattr(module, "FunctionFromTouchstone", FakeParser)
    loaded = BCLFTouchstoneDatasetLoader(dataset_dir=dataset_dir).load()

    assert loaded.n_samples == 1
    assert loaded.X.shape == (1, 8)
    assert loaded.s_params.shape == (1, 2, 2, 2)
    assert loaded.case_ids == ["case0"]


def test_legacy_bnn_adapter_trains_and_predicts_sparam_shape() -> None:
    dataset = _synthetic_dataset(n=3, n_freqs=2)
    model = LegacyBNNResponseSurrogate(
        hidden_dims=[8],
        epochs=1,
        batch_size=4,
        mc_samples=2,
        seed=11,
        device="cpu",
    )

    model.fit(dataset, [0, 1])
    pred = model.predict_s_params(dataset, [2], mc_samples=2)

    assert pred.shape == (1, 2, 2, 2)
    assert np.all(np.isfinite(pred.real))
    assert np.allclose(pred[:, 1, 0, :], pred[:, 0, 1, :])
    assert model.input_dim == 9


def test_legacy_bnn_db_mode_save_load_predicts(tmp_path: Path) -> None:
    dataset = _synthetic_dataset(n=3, n_freqs=2)
    model = LegacyBNNResponseSurrogate(
        hidden_dims=[8],
        epochs=1,
        batch_size=4,
        mc_samples=2,
        seed=12,
        device="cpu",
        target_mode="db",
    )

    model.fit(dataset, [0, 1])
    pred = model.predict_s_params(dataset, [2], mc_samples=2)
    path = tmp_path / "legacy_bnn_db.pt"
    model.save(path, metadata={"split": "unit"})
    loaded = LegacyBNNResponseSurrogate.load(path, device="cpu")
    loaded_pred = loaded.predict_s_params(dataset, [2], mc_samples=2)

    assert loaded.target_mode == "db"
    assert pred.shape == (1, 2, 2, 2)
    assert loaded_pred.shape == (1, 2, 2, 2)
    assert np.all(np.isfinite(loaded_pred.real))
    assert np.all(loaded_pred.real >= 0.0)


def test_legacy_bnn_update_warm_starts_same_model() -> None:
    dataset = _synthetic_dataset(n=4, n_freqs=2)
    model = LegacyBNNResponseSurrogate(
        hidden_dims=[8],
        epochs=1,
        batch_size=4,
        mc_samples=2,
        seed=13,
        device="cpu",
        target_mode="db",
    )

    model.fit(dataset, [0, 1], epochs=1)
    model_id = id(model.model)
    model.update(dataset, [0, 1, 2], epochs=1)
    pred = model.predict_s_params(dataset, [3], mc_samples=2)

    assert id(model.model) == model_id
    assert model.trained_epochs == 2
    assert pred.shape == (1, 2, 2, 2)
    assert np.all(np.isfinite(pred.real))


def test_legacy_bnn_save_load_after_update_preserves_optimizer(tmp_path: Path) -> None:
    dataset = _synthetic_dataset(n=4, n_freqs=2)
    model = LegacyBNNResponseSurrogate(
        hidden_dims=[8],
        epochs=1,
        batch_size=4,
        mc_samples=2,
        seed=14,
        device="cpu",
        target_mode="db",
    )

    model.fit(dataset, [0, 1], epochs=1)
    model.update(dataset, [0, 1, 2], epochs=1)
    path = tmp_path / "updated_bnn.pt"
    model.save(path)
    loaded = LegacyBNNResponseSurrogate.load(path, device="cpu")
    loaded.update(dataset, [0, 1, 2, 3], epochs=1)
    pred = loaded.predict_s_params(dataset, [3], mc_samples=2)

    assert loaded.trained_epochs == 3
    assert loaded._optimizer is not None
    assert pred.shape == (1, 2, 2, 2)
    assert np.all(np.isfinite(pred.real))


def test_fourier_bnn_builds_expected_input_features() -> None:
    dataset = _synthetic_dataset(n=3, n_freqs=3)
    model = FourierBNNResponseSurrogate(
        hidden_dims=[8],
        epochs=1,
        batch_size=4,
        mc_samples=2,
        seed=15,
        device="cpu",
        target_mode="db",
        frequency_min_hz=dataset.freqs.min(),
        frequency_max_hz=dataset.freqs.max(),
    )

    flat = model._build_flat_inputs(dataset, np.asarray([0], dtype=int))
    repeated = model._build_flat_inputs(dataset, np.asarray([0], dtype=int))

    assert model.input_dim == 19
    assert flat.shape == (dataset.n_freqs, 19)
    assert np.allclose(flat, repeated)
    assert np.allclose(flat[:, :8], dataset.X[0].reshape(1, -1))
    assert np.all(flat[:, 8] >= 0.0)
    assert np.all(flat[:, 8] <= 1.0)
    assert not np.allclose(flat[0, 9:], flat[1, 9:])


def test_fourier_bnn_fit_save_load_and_update(tmp_path: Path) -> None:
    dataset = _synthetic_dataset(n=4, n_freqs=3)
    model = FourierBNNResponseSurrogate(
        hidden_dims=[8],
        epochs=1,
        batch_size=4,
        mc_samples=2,
        seed=16,
        device="cpu",
        target_mode="db",
        fourier_bands=[1, 2],
        include_raw_frequency=True,
    )

    model.fit(dataset, [0, 1], epochs=1)
    pred = model.predict_s_params(dataset, [2], mc_samples=2)
    path = tmp_path / "fourier_bnn.pt"
    model.save(path, metadata={"variant": "fourier"})
    with pytest.raises(ValueError, match="matching surrogate class"):
        LegacyBNNResponseSurrogate.load(path, device="cpu")
    loaded = FourierBNNResponseSurrogate.load(path, device="cpu")
    loaded.update(dataset, [0, 1, 2], epochs=1)
    loaded_pred = loaded.predict_s_params(dataset, [3], mc_samples=2)

    assert pred.shape == (1, 2, 2, 3)
    assert loaded.input_dim == 13
    assert loaded.surrogate_variant == "fourier"
    assert loaded.fourier_bands == [1.0, 2.0]
    assert loaded.trained_epochs == 2
    assert loaded_pred.shape == (1, 2, 2, 3)
    assert np.all(np.isfinite(loaded_pred.real))
