from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from optimization.bnn_acquisition import (
    BNNCheckpointError,
    BNNExpectedImprovementAcquisition,
    BNNExpectedImprovementBackend,
    ScalarBNNConfig,
    ScalarBNNRegressor,
)
from optimization.botorch_acquisition import GPqLogEIAcquisitionBackend
from optimization.botorch_acquisition import AcquisitionFitResult
from optimization.candidate_space import ContinuousCandidateSpace, DiscreteCandidateSpace
from optimization.datasets import PassiveDataset
from optimization.optimization_spec import OptimizationSpec, SpecResults
from optimization.optimizer_base import SelectionResult
from optimization.optimizer_bo import OWAFSBNNOptimizer, OptimizerBoTorch
from optimization.recorder import ExperimentRecorder
from optimization.simulators import ReplaySimulator, SimulationResult, Simulator


class ScalarResponseSpec(OptimizationSpec):
    def __init__(self) -> None:
        object.__setattr__(self, "objective_terms", [])
        object.__setattr__(self, "constraints", [])
        object.__setattr__(self, "frequency_bands_ghz", {})

    def evaluate(self, freqs, s_params) -> SpecResults:
        _ = freqs
        value = float(np.mean(np.abs(np.asarray(s_params)[0, 0, :])))
        return SpecResults(
            objective=np.asarray([value], dtype=float),
            constraints=np.asarray([], dtype=float),
            metrics={"objective": value},
            objective_components={"objective": value},
            metadata={},
        )

    def to_objective(self):  # pragma: no cover
        raise AssertionError("online optimizer should use evaluate")


class QuadraticSimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()
        self.freqs = np.asarray([3.5e9, 8.7e9], dtype=float)

    def simulate(self, geometry: np.ndarray) -> SimulationResult:
        self.num_calls += 1
        x = np.asarray(geometry, dtype=float).reshape(-1)
        value = float(0.02 + (x[0] - 7.8) ** 2 + (x[1] - 8.1) ** 2)
        s_params = np.zeros((2, 2, len(self.freqs)), dtype=complex)
        s_params[0, 0, :] = value + 0j
        s_params[0, 1, :] = 0.5 * value + 0j
        s_params[1, 0, :] = s_params[0, 1, :]
        self.num_success += 1
        return SimulationResult(
            freqs=self.freqs,
            s_params=s_params,
            status="done",
            metadata={"cache_hit": False, "job_id": f"bnn_bo_{self.num_calls}"},
            actual_geometry=x,
        )

    def simulate_afs(self, geometry: np.ndarray, prediction) -> SimulationResult:
        _ = prediction
        result = self.simulate(geometry)
        metadata = dict(result.metadata)
        metadata["afs"] = True
        return SimulationResult(
            freqs=result.freqs,
            s_params=result.s_params,
            status=result.status,
            metadata=metadata,
            actual_geometry=result.actual_geometry,
        )

    def simulate_full_validation(self, geometry: np.ndarray, *, case_id: str) -> SimulationResult:
        result = self.simulate(geometry)
        metadata = dict(result.metadata)
        metadata.update({"full_validation": True, "case_id": case_id})
        return SimulationResult(
            freqs=result.freqs,
            s_params=result.s_params,
            status=result.status,
            metadata=metadata,
            actual_geometry=result.actual_geometry,
        )


def _bclf_like_candidates() -> np.ndarray:
    return np.asarray(
        [
            [7.70, 8.00, 7.80, 0.90, 0.70, 0.90, 0.13, 0.55],
            [7.80, 8.10, 7.90, 1.00, 0.80, 1.00, 0.145, 0.575],
            [7.90, 8.20, 8.00, 1.10, 0.90, 1.10, 0.16, 0.60],
            [7.75, 8.15, 7.85, 0.95, 0.75, 0.95, 0.14, 0.56],
        ],
        dtype=float,
    )


def _bclf_like_passive_dataset() -> PassiveDataset:
    freqs = np.asarray([3.5e9, 8.7e9], dtype=float)
    samples = []
    for idx, x in enumerate(_bclf_like_candidates()):
        value = float(0.02 + (x[0] - 7.8) ** 2 + (x[1] - 8.1) ** 2)
        s_params = np.zeros((2, 2, len(freqs)), dtype=complex)
        s_params[0, 0, :] = value + 0j
        s_params[0, 1, :] = 0.5 * value + 0j
        s_params[1, 0, :] = s_params[0, 1, :]
        samples.append({"geo_params": x, "s_params": s_params, "caseid": f"case_{idx}"})
    return PassiveDataset(
        samples=samples,
        variable_order=["L", "L1", "L2", "W", "W1", "W2", "S1", "S2"],
        freqs=freqs,
    )


def _response_history(dataset: PassiveDataset, indices: list[int]) -> list[dict]:
    history = []
    for i in indices:
        freqs, s_params = dataset.get_response_by_geometry(dataset.get_geometry(i))
        history.append(
            {
                "geometry": dataset.get_geometry(i).tolist(),
                "_response_freqs": freqs,
                "_response_s_params": s_params,
            }
        )
    return history


@pytest.mark.parametrize("acquisition", ["qlogei", "qei", "ei"])
def test_bnn_response_soft_ei_ranks_candidates_when_hard_ei_is_zero(acquisition: str) -> None:
    class FixedResponseSurrogate:
        surrogate_variant = "raw"
        trained_epochs = 0

        def predict_s_params(self, candidate_dataset, candidate_indices, *, mc_samples, return_samples):
            _ = return_samples
            n_candidates = len(candidate_indices)
            values = np.asarray([10.0, 8.0, 6.0], dtype=float)[:n_candidates]
            samples = np.zeros((int(mc_samples), n_candidates, 2, 2, candidate_dataset.n_freqs), dtype=complex)
            samples[:, :, 0, 0, :] = values.reshape(1, n_candidates, 1)
            return samples.mean(axis=0), samples

    freqs = np.asarray([3.5e9, 8.7e9], dtype=float)
    backend = BNNExpectedImprovementBackend(
        {"acquisition": acquisition, "mc_samples": 3},
        device="cpu",
    )
    backend._torch = object()  # type: ignore[attr-defined]
    backend._device = "cpu"  # type: ignore[attr-defined]
    backend._surrogate = FixedResponseSurrogate()  # type: ignore[assignment]
    backend._train_dataset = SimpleNamespace(
        variable_order=["L", "L1", "L2", "W", "W1", "W2", "S1", "S2"],
        freqs=freqs,
        n_freqs=len(freqs),
    )
    backend._optimization_spec = ScalarResponseSpec()
    backend._last_fit_mode = "test"
    backend._last_train_count = 1

    candidates = _bclf_like_candidates()[:3]
    scores = backend.score_candidates(
        candidates_actual=candidates,
        candidates_normalized=candidates,
        history=[{"objective": 0.0}],
        tr_debug={"candidate_source": "resampled_continuous_bounds"},
    )
    result = backend.select(
        candidates_actual=candidates,
        candidates_normalized=candidates,
        history=[{"objective": 0.0}],
        tr_debug={"candidate_source": "resampled_continuous_bounds"},
    )

    debug = result.selection_debug or {}
    assert scores.predicted_mean.shape == (3,)
    assert scores.predicted_uncertainty.shape == (3,)
    assert scores.acquisition_score.shape == (3,)
    assert np.isfinite(scores.acquisition_score).all()
    assert debug["selected_pos"] == 2
    assert debug["selected_pos"] == int(np.argmax(scores.acquisition_score))
    assert debug["soft_ei_tau"] == 1.0
    assert np.isfinite(float(debug["soft_improvement_min"]))
    assert np.isfinite(float(debug["soft_improvement_max"]))
    assert float(debug["soft_improvement_max"]) > float(debug["soft_improvement_min"]) > 0.0
    if acquisition == "qlogei":
        assert float(result.acquisition_score) > np.log(np.finfo(float).tiny)


def test_scalar_bnn_acquisition_scores_candidates() -> None:
    config = ScalarBNNConfig(hidden_dims=(8,), epochs=1, batch_size=2, mc_samples=4, bayesian_loss_weight=1e-5)
    surrogate = ScalarBNNRegressor(input_dim=2, config=config, seed=3, device="cpu")
    X = np.asarray([[0.0, 0.0], [0.25, 0.75], [1.0, 1.0]], dtype=np.float32)
    y = np.asarray([[1.0], [0.0], [1.0]], dtype=np.float32)
    surrogate.fit(X, y)

    import torch

    acq = BNNExpectedImprovementAcquisition(surrogate, best_objective=0.5, mc_samples=4)
    values = acq(torch.tensor(X).unsqueeze(1))

    assert values.shape == (3,)
    assert torch.isfinite(values).all()


def test_bnn_backend_warm_starts_like_fewshot_schedule() -> None:
    backend = BNNExpectedImprovementBackend(
        {
            "hidden_dims": [8],
            "epochs": 2,
            "update_epochs": 1,
            "initial_train_size": 2,
            "batch_size": 2,
            "mc_samples": 4,
            "bayesian_loss_weight": 1e-5,
        },
        device="cpu",
    )
    dataset = _bclf_like_passive_dataset()
    X2 = np.asarray([dataset.get_geometry(i) for i in [0, 1]], dtype=np.float32)
    y2 = np.asarray([[0.1], [0.0]], dtype=np.float32)
    history2 = []
    for i in [0, 1]:
        freqs, s_params = dataset.get_response_by_geometry(dataset.get_geometry(i))
        history2.append({"geometry": dataset.get_geometry(i).tolist(), "_response_freqs": freqs, "_response_s_params": s_params})
    X3 = np.asarray([dataset.get_geometry(i) for i in [0, 1, 2]], dtype=np.float32)
    y3 = np.asarray([[0.1], [0.0], [0.1]], dtype=np.float32)
    history3 = []
    for i in [0, 1, 2]:
        freqs, s_params = dataset.get_response_by_geometry(dataset.get_geometry(i))
        history3.append({"geometry": dataset.get_geometry(i).tolist(), "_response_freqs": freqs, "_response_s_params": s_params})

    first = backend.fit(
        X2,
        y2,
        seed=3,
        X_actual=X2,
        history=history2,
        optimization_spec=ScalarResponseSpec(),
        variable_order=dataset.variable_order,
    )
    first_model_id = id(backend._surrogate.model)  # type: ignore[union-attr]
    second = backend.fit(
        X3,
        y3,
        seed=3,
        X_actual=X3,
        history=history3,
        optimization_spec=ScalarResponseSpec(),
        variable_order=dataset.variable_order,
    )

    assert first.ready
    assert second.ready
    assert backend._last_fit_mode == "warm_start_update"
    assert id(backend._surrogate.model) == first_model_id  # type: ignore[union-attr]
    assert backend._surrogate.trained_epochs == 3  # type: ignore[union-attr]


def test_bnn_backend_loads_checkpoint_without_initial_retraining(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "response_bnn.pt"
    dataset = _bclf_like_passive_dataset()
    indices = [0, 1]
    X = np.asarray([dataset.get_geometry(i) for i in indices], dtype=np.float32)
    y = np.asarray([[0.1], [0.0]], dtype=np.float32)
    history = _response_history(dataset, indices)
    base_config = {
        "hidden_dims": [8],
        "epochs": 2,
        "update_epochs": 1,
        "initial_train_size": 2,
        "batch_size": 2,
        "mc_samples": 4,
        "bayesian_loss_weight": 1e-5,
        "checkpoint_path": checkpoint_path,
    }
    training_backend = BNNExpectedImprovementBackend(
        {**base_config, "force_retrain": True},
        device="cpu",
    )

    first = training_backend.fit(
        X,
        y,
        seed=23,
        X_actual=X,
        history=history,
        optimization_spec=ScalarResponseSpec(),
        variable_order=dataset.variable_order,
    )
    assert first.ready
    saved_epochs = training_backend._surrogate.trained_epochs  # type: ignore[union-attr]
    loading_backend = BNNExpectedImprovementBackend(
        {**base_config, "require_checkpoint": True},
        device="cpu",
    )
    second = loading_backend.fit(
        X,
        y,
        seed=23,
        X_actual=X,
        history=history,
        optimization_spec=ScalarResponseSpec(),
        variable_order=dataset.variable_order,
    )

    assert second.ready
    assert checkpoint_path.exists()
    assert loading_backend._last_fit_mode == "checkpoint_load"
    assert loading_backend._surrogate.trained_epochs == saved_epochs  # type: ignore[union-attr]
    assert loading_backend._surrogate.checkpoint_metadata["train_count"] == 2  # type: ignore[union-attr]


def test_bnn_backend_can_fit_and_load_fourier_surrogate(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "fourier_response_bnn.pt"
    dataset = _bclf_like_passive_dataset()
    indices = [0, 1]
    X = np.asarray([dataset.get_geometry(i) for i in indices], dtype=np.float32)
    y = np.asarray([[0.1], [0.0]], dtype=np.float32)
    config = {
        "hidden_dims": [8],
        "epochs": 1,
        "update_epochs": 1,
        "initial_train_size": 2,
        "batch_size": 2,
        "mc_samples": 4,
        "bayesian_loss_weight": 1e-5,
        "checkpoint_path": checkpoint_path,
        "surrogate_variant": "fourier",
        "fourier_bands": [1, 2],
        "include_raw_frequency": True,
    }
    training_backend = BNNExpectedImprovementBackend(
        {**config, "force_retrain": True},
        device="cpu",
    )

    first = training_backend.fit(
        X,
        y,
        seed=31,
        X_actual=X,
        history=_response_history(dataset, indices),
        optimization_spec=ScalarResponseSpec(),
        variable_order=dataset.variable_order,
    )
    loading_backend = BNNExpectedImprovementBackend(
        {**config, "require_checkpoint": True},
        device="cpu",
    )
    second = loading_backend.fit(
        X,
        y,
        seed=31,
        X_actual=X,
        history=_response_history(dataset, indices),
        optimization_spec=ScalarResponseSpec(),
        variable_order=dataset.variable_order,
    )

    assert first.ready
    assert second.ready
    assert checkpoint_path.exists()
    assert training_backend._surrogate.surrogate_variant == "fourier"  # type: ignore[union-attr]
    assert training_backend._surrogate.input_dim == 13  # type: ignore[union-attr]
    assert loading_backend._last_fit_mode == "checkpoint_load"
    assert loading_backend._surrogate.surrogate_variant == "fourier"  # type: ignore[union-attr]


def test_bnn_backend_requires_checkpoint_when_configured(tmp_path: Path) -> None:
    dataset = _bclf_like_passive_dataset()
    indices = [0, 1]
    X = np.asarray([dataset.get_geometry(i) for i in indices], dtype=np.float32)
    y = np.asarray([[0.1], [0.0]], dtype=np.float32)
    backend = BNNExpectedImprovementBackend(
        {
            "hidden_dims": [8],
            "epochs": 1,
            "batch_size": 2,
            "mc_samples": 4,
            "bayesian_loss_weight": 1e-5,
            "checkpoint_path": tmp_path / "missing.pt",
            "require_checkpoint": True,
        },
        device="cpu",
    )

    with pytest.raises(BNNCheckpointError, match="required"):
        backend.fit(
            X,
            y,
            seed=29,
            X_actual=X,
            history=_response_history(dataset, indices),
            optimization_spec=ScalarResponseSpec(),
            variable_order=dataset.variable_order,
        )


def test_optimizer_botorch_can_use_bnn_acquisition(tmp_path: Path) -> None:
    candidates = _bclf_like_candidates()
    optimizer = OptimizerBoTorch(
        candidate_space=DiscreteCandidateSpace(
            variable_order=["L", "L1", "L2", "W", "W1", "W2", "S1", "S2"],
            candidate_geometries=candidates,
        ),
        simulator=QuadraticSimulator(),
        optimization_spec=ScalarResponseSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records"),
        budget=1,
        n_init=2,
        seed=11,
        run_id="bnn_botorch",
        checkpoint_path=tmp_path / "bnn_botorch.checkpoint.json",
        acquisition_backend=BNNExpectedImprovementBackend(
            {
                "hidden_dims": [8],
                "epochs": 1,
                "batch_size": 2,
                "mc_samples": 4,
                "bayesian_loss_weight": 1e-5,
            },
            device="cpu",
        ),
    )

    result = optimizer.run()

    assert len(result.history) == 3
    assert result.history[-1]["optimizer_backend"] == "botorch"
    assert result.history[-1]["selection_debug"]["mode"] == "bnn_qlogei_discrete"
    assert result.history[-1]["selection_debug"]["bnn_runtime"]["surrogate"] == "response"
    assert np.isfinite(float(result.history[-1]["acquisition_score"]))


def test_bnn_acquisition_resamples_bounds_for_continuous_space(tmp_path: Path) -> None:
    variable_order = ["L", "L1", "L2", "W", "W1", "W2", "S1", "S2"]
    bounds = {
        "L": (7.7, 7.9),
        "L1": (8.0, 8.2),
        "L2": (7.8, 8.0),
        "W": (0.9, 1.1),
        "W1": (0.7, 0.9),
        "W2": (0.9, 1.1),
        "S1": (0.13, 0.16),
        "S2": (0.55, 0.60),
    }
    optimizer = OptimizerBoTorch(
        candidate_space=ContinuousCandidateSpace(
            variable_order=variable_order,
            variable_bounds=bounds,
            pool_size=12,
            seed=29,
            resolution=0.01,
        ),
        simulator=QuadraticSimulator(),
        optimization_spec=ScalarResponseSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records_bnn_continuous"),
        budget=1,
        n_init=2,
        seed=29,
        run_id="bnn_continuous_resampled",
        checkpoint_path=tmp_path / "bnn_continuous_resampled.checkpoint.json",
        acquisition_backend=BNNExpectedImprovementBackend(
            {
                "hidden_dims": [8],
                "epochs": 1,
                "batch_size": 2,
                "mc_samples": 4,
                "bayesian_loss_weight": 1e-5,
            },
            device="cpu",
        ),
    )

    result = optimizer.run()

    post_init = result.history[-1]
    assert post_init["selection_debug"]["mode"] == "bnn_qlogei_resampled"
    assert post_init["selection_debug"]["candidate_source"] == "resampled_continuous_bounds"
    assert post_init["selection_debug"]["resampled_candidate_target"] == 12
    assert post_init["selection_debug"]["resampled_candidate_count"] > 0
    assert all(abs(float(value) / 0.01 - round(float(value) / 0.01)) < 1e-8 for value in post_init["requested_geometry"])


def test_owafs_bnn_resamples_bounds_for_continuous_space(tmp_path: Path) -> None:
    variable_order = ["L", "L1", "L2", "W", "W1", "W2", "S1", "S2"]
    bounds = {
        "L": (7.7, 7.9),
        "L1": (8.0, 8.2),
        "L2": (7.8, 8.0),
        "W": (0.9, 1.1),
        "W1": (0.7, 0.9),
        "W2": (0.9, 1.1),
        "S1": (0.13, 0.16),
        "S2": (0.55, 0.60),
    }
    candidate_space = ContinuousCandidateSpace(
        variable_order=variable_order,
        variable_bounds=bounds,
        pool_size=12,
        seed=31,
        resolution=0.01,
    )
    optimizer = OWAFSBNNOptimizer(
        candidate_space=candidate_space,
        simulator=QuadraticSimulator(),
        optimization_spec=ScalarResponseSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records_owafs_bnn_continuous"),
        budget=2,
        n_init=2,
        seed=31,
        run_id="owafs_bnn_continuous_resampled",
        checkpoint_path=tmp_path / "owafs_bnn_continuous_resampled.checkpoint.json",
        acquisition_backend=BNNExpectedImprovementBackend(
            {
                "hidden_dims": [8],
                "epochs": 1,
                "batch_size": 2,
                "mc_samples": 4,
                "bayesian_loss_weight": 1e-5,
            },
            device="cpu",
        ),
    )

    result = optimizer.run()

    guided = [row for row in result.history if row["iteration"] >= 2 and row.get("phase") != "validation"]
    assert len(guided) == 2
    for row in guided:
        debug = row["selection_debug"]
        assert debug["mode"] == "bnn_qlogei_resampled"
        assert debug["candidate_source"] == "resampled_continuous_bounds"
        assert debug["resampled_candidate_target"] == 12
        assert debug["resampled_candidate_count"] > 0
        assert debug["soft_ei_tau"] == 1.0
        assert all(abs(float(value) / 0.01 - round(float(value) / 0.01)) < 1e-8 for value in row["requested_geometry"])


def test_default_optimizer_botorch_uses_gp_qlogei_backend(tmp_path: Path) -> None:
    candidates = _bclf_like_candidates()
    optimizer = OptimizerBoTorch(
        candidate_space=DiscreteCandidateSpace(
            variable_order=["L", "L1", "L2", "W", "W1", "W2", "S1", "S2"],
            candidate_geometries=candidates,
        ),
        simulator=QuadraticSimulator(),
        optimization_spec=ScalarResponseSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records_gp"),
        budget=1,
        n_init=2,
        seed=13,
        run_id="gp_default",
        checkpoint_path=tmp_path / "gp_default.checkpoint.json",
    )

    result = optimizer.run()

    assert isinstance(optimizer.acquisition_backend, GPqLogEIAcquisitionBackend)
    assert result.history[-1]["selection_debug"]["mode"] == "gp_qlogei_discrete"


def test_optimizer_botorch_can_use_gp_qlogei_backend(tmp_path: Path) -> None:
    candidates = _bclf_like_candidates()
    optimizer = OptimizerBoTorch(
        candidate_space=DiscreteCandidateSpace(
            variable_order=["L", "L1", "L2", "W", "W1", "W2", "S1", "S2"],
            candidate_geometries=candidates,
        ),
        simulator=QuadraticSimulator(),
        optimization_spec=ScalarResponseSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records_gp_qlogei"),
        budget=1,
        n_init=2,
        seed=19,
        run_id="gp_qlogei",
        checkpoint_path=tmp_path / "gp_qlogei.checkpoint.json",
        acquisition_backend=GPqLogEIAcquisitionBackend(),
    )

    result = optimizer.run()

    assert result.history[-1]["selection_debug"]["mode"] == "gp_qlogei_discrete"


def test_gp_qlogei_score_candidates_matches_select() -> None:
    candidates = _bclf_like_candidates()
    backend = GPqLogEIAcquisitionBackend()
    X_train = np.asarray([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=float)
    X_score = np.asarray([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.25, 0.75]], dtype=float)
    y_train = np.asarray([[4.0], [1.0], [3.0]], dtype=float)
    fit = backend.fit(X_train, y_train, seed=7)
    assert fit.ready

    history = [{"objective": float(value)} for value in y_train.reshape(-1)]
    scores = backend.score_candidates(
        candidates_actual=candidates,
        candidates_normalized=X_score,
        history=history,
        tr_debug={"candidate_source": "unit_test"},
    )
    selected = backend.select(
        candidates_actual=candidates,
        candidates_normalized=X_score,
        history=history,
        tr_debug={"candidate_source": "unit_test"},
    )

    assert scores.predicted_mean.shape == (len(candidates),)
    assert scores.predicted_uncertainty.shape == (len(candidates),)
    assert scores.acquisition_score.shape == (len(candidates),)
    assert np.isfinite(scores.acquisition_score).all()
    assert selected.selection_debug is not None
    assert selected.selection_debug["selected_pos"] == int(np.argmax(scores.acquisition_score))


def test_gp_qlogei_uses_continuous_acqf_for_continuous_space(tmp_path: Path) -> None:
    variable_order = ["L", "L1", "L2", "W", "W1", "W2", "S1", "S2"]
    bounds = {
        "L": (7.7, 7.9),
        "L1": (8.0, 8.2),
        "L2": (7.8, 8.0),
        "W": (0.9, 1.1),
        "W1": (0.7, 0.9),
        "W2": (0.9, 1.1),
        "S1": (0.13, 0.16),
        "S2": (0.55, 0.60),
    }
    optimizer = OptimizerBoTorch(
        candidate_space=ContinuousCandidateSpace(
            variable_order=variable_order,
            variable_bounds=bounds,
            pool_size=8,
            seed=23,
            resolution=0.01,
        ),
        simulator=QuadraticSimulator(),
        optimization_spec=ScalarResponseSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records_gp_continuous"),
        budget=1,
        n_init=2,
        seed=23,
        run_id="gp_qlogei_continuous",
        checkpoint_path=tmp_path / "gp_qlogei_continuous.checkpoint.json",
        acquisition_backend=GPqLogEIAcquisitionBackend(
            num_restarts=2,
            raw_samples=8,
            maxiter=5,
            fallback_samples=16,
        ),
    )

    result = optimizer.run()

    post_init = result.history[-1]
    assert post_init["selection_debug"]["mode"] == "gp_qlogei_continuous"
    assert post_init["selection_debug"]["kernel_runtime"]["optimizer"] == "optimize_acqf"
    assert all(abs(float(value) / 0.01 - round(float(value) / 0.01)) < 1e-8 for value in post_init["requested_geometry"])


def test_gp_and_bnn_backends_run_on_tiny_replay_dataset_without_repeats(tmp_path: Path) -> None:
    dataset = _bclf_like_passive_dataset()
    candidates = _bclf_like_candidates()

    methods = {
        "gp": GPqLogEIAcquisitionBackend(),
        "bnn": BNNExpectedImprovementBackend(
            {"hidden_dims": [8], "epochs": 1, "batch_size": 2, "mc_samples": 4, "bayesian_loss_weight": 1e-5},
            device="cpu",
        ),
    }
    for name, backend in methods.items():
        optimizer = OptimizerBoTorch(
            candidate_space=DiscreteCandidateSpace(variable_order=dataset.variable_order, candidate_geometries=candidates),
            simulator=ReplaySimulator(dataset=dataset),
            optimization_spec=ScalarResponseSpec(),
            recorder=ExperimentRecorder(output_dir=tmp_path / f"records_{name}"),
            budget=1,
            n_init=2,
            seed=17,
            run_id=f"replay_{name}",
            checkpoint_path=tmp_path / f"replay_{name}.checkpoint.json",
            acquisition_backend=backend,
        )

        result = optimizer.run()
        requested = [tuple(row["requested_geometry"]) for row in result.history]
        actual = [tuple(row["geometry"]) for row in result.history]

        assert len(result.history) == 3
        assert len(set(requested)) == len(requested)
        assert len(set(actual)) == len(actual)


def test_owafs_bnn_validation_row_does_not_consume_budget_and_replaces_training_label(
    tmp_path: Path,
) -> None:
    class FakeBackend:
        name = "fake_bnn"

        def __init__(self) -> None:
            self.fit_case_ids: list[list[str]] = []

        def fit(self, X, y, *, history, **kwargs):  # type: ignore[no-untyped-def]
            _ = X, y, kwargs
            self.fit_case_ids.append([str(row.get("case_id")) for row in history])
            return AcquisitionFitResult(ready=True)

        def select(self, *, candidates_actual, candidates_normalized, history, tr_debug):  # type: ignore[no-untyped-def]
            _ = candidates_normalized, history, tr_debug
            return SelectionResult(requested_geometry=np.asarray(candidates_actual[0], dtype=float))

        def predict_response_profile(self, geometry):  # type: ignore[no-untyped-def]
            _ = geometry
            return object()

    class FakeOWAFSSimulator(Simulator):
        def __init__(self) -> None:
            super().__init__()
            self.freqs = np.asarray([1.0, 2.0], dtype=float)
            self.context_iteration = 0

        def set_context(self, *, run_id: str, iteration: int) -> None:
            _ = run_id
            self.context_iteration = int(iteration)

        def _result(self, geometry: np.ndarray, value: float, case_id: str, source: str) -> SimulationResult:
            s_params = np.zeros((2, 2, len(self.freqs)), dtype=complex)
            s_params[0, 0, :] = value
            self.num_success += 1
            return SimulationResult(
                freqs=self.freqs,
                s_params=s_params,
                status="done",
                metadata={"case_id": case_id, "job_id": case_id, "response_source": source},
                actual_geometry=np.asarray(geometry, dtype=float),
            )

        def simulate(self, geometry: np.ndarray) -> SimulationResult:
            self.num_calls += 1
            return self._result(geometry, 1.0, f"init_{self.context_iteration}", "full")

        def simulate_afs(self, geometry: np.ndarray, prediction) -> SimulationResult:  # type: ignore[no-untyped-def]
            _ = prediction
            self.num_calls += 1
            return self._result(geometry, 0.05, f"afs_{self.context_iteration}", "afs_vf_reconstructed")

        def simulate_full_validation(self, geometry: np.ndarray, *, case_id: str) -> SimulationResult:
            self.num_calls += 1
            return self._result(geometry, 0.1, case_id, "full_validation")

    candidates = _bclf_like_candidates()
    backend = FakeBackend()
    optimizer = OWAFSBNNOptimizer(
        candidate_space=DiscreteCandidateSpace(
            variable_order=["L", "L1", "L2", "W", "W1", "W2", "S1", "S2"],
            candidate_geometries=candidates,
        ),
        simulator=FakeOWAFSSimulator(),
        optimization_spec=ScalarResponseSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records_owafs"),
        budget=2,
        n_init=1,
        seed=3,
        run_id="owafs_fake",
        checkpoint_path=tmp_path / "owafs_fake.checkpoint.json",
        acquisition_backend=backend,  # type: ignore[arg-type]
    )

    result = optimizer.run()
    budgeted = [row for row in result.history if row.get("counts_toward_budget", True)]
    validations = [row for row in result.history if not row.get("counts_toward_budget", True)]
    shadow_rows = [row for row in result.history if row.get("exclude_from_model_training")]

    assert len(budgeted) == 3
    assert len(validations) == 2
    assert all(row["phase"] == "validation" for row in validations)
    assert shadow_rows and all(row["exclude_from_state"] for row in shadow_rows)
    assert result.best_trial["phase"] == "validation"
    assert result.state.best_objective == pytest.approx(0.1)
    assert any("afs_1" not in fit_ids and "afs_2" not in fit_ids for fit_ids in backend.fit_case_ids[1:])
    assert any(any(case_id.endswith("_full_validation") for case_id in fit_ids) for fit_ids in backend.fit_case_ids[1:])
