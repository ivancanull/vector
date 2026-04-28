from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from optimization.acquisition import UCBAcquisition
from optimization.datasets import PassiveDataset
from optimization.objectives import MaxS11DbMetric, MinimizeMetric, Objective
from optimization.recorder import ExperimentRecorder
from optimization.replay_optimizer import ReplayOptimizer
from optimization.simulators import SimulationResult, Simulator
from optimization.surrogates import SurrogateModel


class CountingLinearSurrogate(SurrogateModel):
    def __init__(self) -> None:
        self.fit_calls = 0
        self.coef = np.array([0.0])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.fit_calls += 1
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float)
        A = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.coef = coef

    def predict(self, X_pool: np.ndarray) -> np.ndarray:
        X = np.asarray(X_pool, dtype=float)
        A = np.column_stack([np.ones(len(X)), X])
        return A @ self.coef

    def predict_uncertainty(self, X_pool: np.ndarray):
        X = np.asarray(X_pool, dtype=float)
        return np.full(len(X), 0.05, dtype=float)


class FakeSimulator(Simulator):
    def __init__(self, freqs: np.ndarray) -> None:
        super().__init__()
        self.freqs = np.asarray(freqs, dtype=float)

    def simulate(self, geometry: np.ndarray) -> SimulationResult:
        self.num_calls += 1
        geo = np.asarray(geometry, dtype=float)
        s11_db = -5.0 - geo[0]
        s_params = np.zeros((2, 2, len(self.freqs)), dtype=complex)
        s_params[0, 0, :] = 10 ** (s11_db / 20.0)
        s_params[1, 0, :] = 10 ** (-6.0 / 20.0)
        self.num_success += 1
        return SimulationResult(freqs=self.freqs, s_params=s_params, status="done")


def _write_replay_payload(path: Path) -> None:
    dataset = []
    for idx, s11_db in enumerate([-5.0, -10.0, -15.0, -20.0, -12.0, -8.0]):
        s_params = np.zeros((2, 2, 4), dtype=complex)
        s_params[0, 0, :] = 10 ** (s11_db / 20.0)
        s_params[1, 0, :] = 10 ** (-6.0 / 20.0)
        dataset.append(
            {
                "caseid": idx,
                "geo_params": [float(idx), float(idx) / 10.0, 1.0],
                "s_params": s_params,
                "sampled_indices": None,
            }
        )

    payload = {"dataset": dataset, "history_idx": []}
    with path.open("wb") as f:
        pickle.dump(payload, f)


def test_replay_optimizer_controller_flow_and_retrain_every(tmp_path: Path) -> None:
    dataset_path = tmp_path / "replay.pkl"
    _write_replay_payload(dataset_path)

    dataset = PassiveDataset.from_pickle(
        dataset_path,
        variable_order=["Din", "S", "W"],
        freqs=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    simulator = FakeSimulator(freqs=np.array([1.0, 2.0, 3.0, 4.0]))
    objective = Objective([MinimizeMetric(name="min_max_s11", metric=MaxS11DbMetric())])

    recorder = ExperimentRecorder(output_dir=tmp_path / "records")
    surrogate = CountingLinearSurrogate()
    optimizer = ReplayOptimizer(
        dataset=dataset,
        simulator=simulator,
        objective=objective,
        surrogate=surrogate,
        acquisition=UCBAcquisition(beta=1.0),
        recorder=recorder,
        initial_indices=[0],
        candidate_indices=[0, 1, 2, 3, 4, 5],
        budget=3,
        retrain_every=2,
        n_init=2,
        seed=11,
        run_id="unit_replay",
    )

    result = optimizer.run()

    # init:2 + bo budget:3
    assert len(result.history) == 5
    assert simulator.num_calls == 5
    assert surrogate.fit_calls == 2  # steps 0 and 2
    assert len(result.state.train_indices) == 5
    assert len(result.state.candidate_indices) == 1
    assert result.best_trial["best_so_far"] == min(item["objective"] for item in result.history)

    with result.recorder_output.json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["config"]["simulator"] == "FakeSimulator"
    assert payload["config"]["objective"]["terms"][0]["metric"] == "max_s11_db"


def test_optimizer_does_not_directly_query_dataset_response(tmp_path: Path) -> None:
    dataset_path = tmp_path / "replay.pkl"
    _write_replay_payload(dataset_path)

    dataset = PassiveDataset.from_pickle(
        dataset_path,
        variable_order=["Din", "S", "W"],
        freqs=np.array([1.0, 2.0, 3.0, 4.0]),
    )

    # If optimizer reaches into dataset response APIs directly, this test should fail.
    def _forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Optimizer should not call dataset.get_response_by_geometry directly")

    dataset.get_response_by_geometry = _forbidden  # type: ignore[method-assign]

    optimizer = ReplayOptimizer(
        dataset=dataset,
        simulator=FakeSimulator(freqs=np.array([1.0, 2.0, 3.0, 4.0])),
        objective=Objective([MinimizeMetric(name="min_max_s11", metric=MaxS11DbMetric())]),
        surrogate=CountingLinearSurrogate(),
        acquisition=UCBAcquisition(beta=1.0),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records2"),
        initial_indices=[0],
        candidate_indices=[0, 1, 2],
        budget=1,
        retrain_every=1,
        n_init=1,
        seed=7,
        run_id="boundary",
    )

    result = optimizer.run()
    assert len(result.history) == 2
