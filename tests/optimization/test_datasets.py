from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from optimization.datasets import BCLFDataset, PassiveDataset


def _write_payload(path: Path) -> None:
    def mk_sparams(s11_db: float, s21_db: float) -> np.ndarray:
        s = np.zeros((2, 2, 3), dtype=complex)
        s[0, 0, :] = 10 ** (s11_db / 20.0)
        s[1, 0, :] = 10 ** (s21_db / 20.0)
        return s

    payload = {
        "dataset": [
            {"caseid": 0, "geo_params": [55.0, 1.5, 15.0], "s_params": mk_sparams(-10.0, -5.0)},
            {"caseid": 1, "geo_params": [56.0, 1.6, 16.0], "s_params": mk_sparams(-20.0, -4.0)},
            {"caseid": 2, "geo_params": [57.0, 1.7, 17.0], "s_params": mk_sparams(-15.0, -8.0)},
        ],
        "history_idx": [1],
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def test_passive_dataset_loads_container_metadata_and_default_splits(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.pkl"
    _write_payload(dataset_path)

    ds = PassiveDataset.from_pickle(
        dataset_path,
        variable_order=["Din", "S", "W"],
        freqs=np.array([1.0, 2.0, 3.0]),
        device_name="demo_device",
        port_names=["P1", "P2"],
    )

    assert len(ds) == 3
    assert ds.device_name == "demo_device"
    assert ds.port_names == ["P1", "P2"]
    assert ds.default_train_indices == [1]
    assert ds.default_candidate_indices == [0, 2]


def test_passive_dataset_geometry_and_response_access(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.pkl"
    _write_payload(dataset_path)

    ds = PassiveDataset.from_pickle(
        dataset_path,
        variable_order=["Din", "S", "W"],
        freqs=np.array([1.0, 2.0, 3.0]),
    )

    geo = ds.get_geometry(0)
    assert geo.tolist() == [55.0, 1.5, 15.0]

    idx = ds.find_index_by_geometry(np.array([56.0, 1.6, 16.0]))
    assert idx == 1

    freqs, s_params = ds.get_response_by_geometry(np.array([57.0, 1.7, 17.0]))
    assert freqs.tolist() == [1.0, 2.0, 3.0]
    assert s_params.shape == (2, 2, 3)


def test_passive_dataset_append_adds_new_sample(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.pkl"
    _write_payload(dataset_path)

    ds = PassiveDataset.from_pickle(
        dataset_path,
        variable_order=["Din", "S", "W"],
        freqs=np.array([1.0, 2.0, 3.0]),
    )

    s = np.zeros((2, 2, 3), dtype=complex)
    new_idx = ds.append({"caseid": 9, "geo_params": [58.0, 1.8, 18.0], "s_params": s})

    assert new_idx == 3
    assert len(ds) == 4
    assert ds.get_case_id(3) == "9"
    assert ds.get_geometry(3).tolist() == [58.0, 1.8, 18.0]


def test_passive_dataset_rejects_shape_mismatch(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.pkl"
    _write_payload(dataset_path)

    ds = PassiveDataset.from_pickle(
        dataset_path,
        variable_order=["Din", "S", "W"],
        freqs=np.array([1.0, 2.0]),
    )

    with pytest.raises(ValueError, match="Invalid s_params shape"):
        ds.get_response_by_geometry(np.array([55.0, 1.5, 15.0]))


def test_bclf_dataset_defaults_apply(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.pkl"
    _write_payload(dataset_path)

    ds = BCLFDataset.from_pickle(dataset_path, freqs=np.array([1.0, 2.0, 3.0]))

    assert ds.variable_order == ["Din", "S", "W"]
    assert ds.device_name == "bclf"
    desc = ds.default_objective().describe()
    assert desc["type"] == "objective_function"
    assert desc["minimize"] is True
    assert len(desc["terms"]) >= 1
    assert len(desc["constraints"]) == 4
