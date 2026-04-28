from __future__ import annotations

from pathlib import Path

import numpy as np

from optimization.acquisition import UCBAcquisition
from optimization.objectives import MaxS11DbMetric, MinimizeMetric, Objective
from optimization.online_optimizer import OnlineBOOptimizer
from optimization.recorder import ExperimentRecorder
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


class FakeOnlineSimulator(Simulator):
    def __init__(self, freqs: np.ndarray) -> None:
        super().__init__()
        self.freqs = np.asarray(freqs, dtype=float)

    def simulate(self, geometry: np.ndarray) -> SimulationResult:
        self.num_calls += 1
        geo = np.asarray(geometry, dtype=float)
        s11_db = -5.0 - 0.5 * geo[0]
        s_params = np.zeros((2, 2, len(self.freqs)), dtype=complex)
        s_params[0, 0, :] = 10 ** (s11_db / 20.0)
        s_params[1, 0, :] = 10 ** (-6.0 / 20.0)
        self.num_success += 1
        return SimulationResult(
            freqs=self.freqs,
            s_params=s_params,
            status="done",
            metadata={"cache_hit": False, "job_id": f"job_{self.num_calls}"},
        )


def _build_optimizer(tmp_path: Path, *, run_id: str, resume: bool = False) -> OnlineBOOptimizer:
    return OnlineBOOptimizer(
        variable_order=["x"],
        variable_bounds={"x": (0.0, 1.0)},
        pool_size=16,
        simulator=FakeOnlineSimulator(freqs=np.array([1.0, 2.0, 3.0])),
        objective=Objective([MinimizeMetric(name="min_max_s11", metric=MaxS11DbMetric())]),
        surrogate=CountingLinearSurrogate(),
        acquisition=UCBAcquisition(beta=1.0),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records"),
        budget=5,
        retrain_every=1,
        n_init=2,
        seed=42,
        run_id=run_id,
        checkpoint_path=tmp_path / f"{run_id}.checkpoint.json",
        resume=resume,
    )


def test_online_optimizer_deterministic_and_timing_fields(tmp_path: Path) -> None:
    opt1 = _build_optimizer(tmp_path, run_id="online1")
    res1 = opt1.run()

    opt2 = _build_optimizer(tmp_path, run_id="online2")
    res2 = opt2.run()

    g1 = [tuple(row["geometry"]) for row in res1.history]
    g2 = [tuple(row["geometry"]) for row in res2.history]
    assert g1 == g2

    for row in res1.history:
        assert "t_engine_sec" in row
        assert "t_simulator_sec" in row
        assert "t_iteration_sec" in row
        assert "t_engine_cum_sec" in row
        assert "t_simulator_cum_sec" in row

    assert res1.state.best_objective is not None
    assert res1.best_trial["best_so_far"] == min(float(item["objective"]) for item in res1.history)


def test_online_optimizer_checkpoint_resume(tmp_path: Path) -> None:
    run_id = "online_resume"
    first = _build_optimizer(tmp_path, run_id=run_id)
    first_res = first.run()

    resumed = _build_optimizer(tmp_path, run_id=run_id, resume=True)
    resumed_res = resumed.run()

    assert len(resumed_res.history) == len(first_res.history)
    assert [tuple(r["geometry"]) for r in resumed_res.history] == [
        tuple(r["geometry"]) for r in first_res.history
    ]
    assert resumed_res.state.best_objective == first_res.state.best_objective
