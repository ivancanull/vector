from __future__ import annotations

from pathlib import Path

import numpy as np

from optimization.bnn_acquisition import (
    BNNExpectedImprovementAcquisition,
    BNNExpectedImprovementBackend,
    ScalarBNNConfig,
    ScalarBNNRegressor,
)
from optimization.botorch_acquisition import GPqLogEIAcquisitionBackend
from optimization.candidate_space import DiscreteCandidateSpace
from optimization.datasets import PassiveDataset
from optimization.optimization_spec import OptimizationSpec, SpecResults
from optimization.optimizer_bo import OptimizerBoTorch
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
