from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optimization.candidate_space import ContinuousCandidateSpace, DiscreteCandidateSpace
from optimization.optimization_spec import OptimizationSpec, SpecResults
from optimization.optimizer_bo import OptimizerBoTorch
from optimization.recorder import ExperimentRecorder
from optimization.simulators import SimulationResult, Simulator


class ScalarEncodedSpec(OptimizationSpec):
    def __init__(self) -> None:
        object.__setattr__(self, "objective_terms", [])
        object.__setattr__(self, "constraints", [])
        object.__setattr__(self, "frequency_bands_ghz", {})
        self.evaluate_calls = 0

    def evaluate(self, freqs, s_params) -> SpecResults:
        self.evaluate_calls += 1
        value = float(np.real(np.asarray(s_params)[0, 0, 0]))
        return SpecResults(
            objective=np.asarray([value], dtype=float),
            constraints=np.asarray([], dtype=float),
            metrics={"encoded_objective": value},
            objective_components={"encoded_objective": value},
            metadata={},
        )

    def to_objective(self):  # pragma: no cover - should never be called by new online optimizer.
        raise AssertionError("Online optimizer must call OptimizationSpec.evaluate, not to_objective")


class RoundingFakeSimulator(Simulator):
    def __init__(self, freqs: np.ndarray, *, step: float = 0.5) -> None:
        super().__init__()
        self.freqs = np.asarray(freqs, dtype=float)
        self.step = float(step)

    def simulate(self, geometry: np.ndarray) -> SimulationResult:
        self.num_calls += 1
        requested = np.asarray(geometry, dtype=float).reshape(-1)
        actual = np.round(requested / self.step) * self.step
        objective = float((actual[0] - 1.0) ** 2)
        s_params = np.zeros((2, 2, len(self.freqs)), dtype=complex)
        s_params[0, 0, :] = objective + 0j
        self.num_success += 1
        return SimulationResult(
            freqs=self.freqs,
            s_params=s_params,
            status="done",
            metadata={"cache_hit": False, "job_id": f"job_{self.num_calls}"},
            actual_geometry=actual,
        )


class LookupFakeSimulator(Simulator):
    def __init__(self, freqs: np.ndarray, supported: np.ndarray) -> None:
        super().__init__()
        self.freqs = np.asarray(freqs, dtype=float)
        self.supported = np.asarray(supported, dtype=float)

    def simulate(self, geometry: np.ndarray) -> SimulationResult:
        self.num_calls += 1
        requested = np.asarray(geometry, dtype=float).reshape(-1)
        distances = np.sum((self.supported - requested) ** 2, axis=1)
        actual = self.supported[int(np.argmin(distances))]
        objective = float(np.sum((actual - np.array([0.0, 1.0])) ** 2))
        s_params = np.zeros((2, 2, len(self.freqs)), dtype=complex)
        s_params[0, 0, :] = objective + 0j
        self.num_success += 1
        return SimulationResult(
            freqs=self.freqs,
            s_params=s_params,
            status="done",
            metadata={"cache_hit": False, "job_id": f"lookup_{self.num_calls}"},
            actual_geometry=actual,
        )


def test_spec_results_shape_validation() -> None:
    result = SpecResults(
        objective=np.asarray([[1.0]]),
        constraints=np.asarray([], dtype=float),
        metrics={"m": 1.0},
        objective_components={"c": 1.0},
        metadata={},
    )
    assert result.objective.shape == (1,)
    assert result.scalar_objective() == 1.0

    with pytest.raises(ValueError, match="at least one"):
        SpecResults(
            objective=np.asarray([], dtype=float),
            constraints=np.asarray([], dtype=float),
            metrics={},
            objective_components={},
            metadata={},
        )


def test_candidate_space_normalization_roundtrip() -> None:
    continuous = ContinuousCandidateSpace(
        variable_order=["x"],
        variable_bounds={"x": (0.0, 2.0)},
        pool_size=8,
        seed=1,
    )
    x = np.asarray([[0.0], [1.0], [2.0]])
    assert np.allclose(continuous.denormalize(continuous.normalize(x)), x)

    discrete = DiscreteCandidateSpace(
        variable_order=["x", "y"],
        candidate_geometries=np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
    )
    values = np.asarray([[0.0, 1.0], [4.0, 5.0]])
    assert np.allclose(discrete.denormalize(discrete.normalize(values)), values)


def test_online_optimizer_uses_spec_evaluate_and_actual_geometry(tmp_path: Path) -> None:
    spec = ScalarEncodedSpec()
    candidate_space = ContinuousCandidateSpace(
        variable_order=["x"],
        variable_bounds={"x": (0.0, 2.0)},
        pool_size=10,
        seed=42,
    )
    optimizer = OptimizerBoTorch(
        candidate_space=candidate_space,
        simulator=RoundingFakeSimulator(freqs=np.array([1.0, 2.0, 3.0]), step=0.5),
        optimization_spec=spec,
        recorder=ExperimentRecorder(output_dir=tmp_path / "records"),
        budget=3,
        n_init=2,
        seed=123,
        run_id="online_actual_geometry",
        checkpoint_path=tmp_path / "online_actual_geometry.checkpoint.json",
    )

    result = optimizer.run()

    assert spec.evaluate_calls == len(result.history)
    for row in result.history:
        actual = np.asarray(row["geometry"], dtype=float)
        assert np.allclose(actual / 0.5, np.round(actual / 0.5))
        assert np.allclose(row["geometry_normalized"], candidate_space.normalize(actual))
        assert row["requested_geometry"] != row["geometry"] or row["selection_debug"]["actual_geometry_source"]

    assert result.state.best_objective is not None
    assert result.recorder_output.json_path.exists()
    assert result.recorder_output.csv_path.exists()


def test_discrete_candidate_space_with_limited_fake_simulator(tmp_path: Path) -> None:
    supported = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=float)
    candidate_space = DiscreteCandidateSpace(variable_order=["x", "y"], candidate_geometries=supported)
    optimizer = OptimizerBoTorch(
        candidate_space=candidate_space,
        simulator=LookupFakeSimulator(freqs=np.array([1.0, 2.0, 3.0]), supported=supported),
        optimization_spec=ScalarEncodedSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records"),
        budget=2,
        n_init=1,
        seed=7,
        run_id="online_discrete",
        checkpoint_path=tmp_path / "online_discrete.checkpoint.json",
    )

    result = optimizer.run()
    supported_tuples = {tuple(row) for row in supported.tolist()}
    assert all(tuple(row["geometry"]) in supported_tuples for row in result.history)
    assert len(candidate_space.remaining()) < len(supported)


def test_online_optimizer_checkpoint_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resume.checkpoint.json"
    kwargs = dict(
        candidate_space=ContinuousCandidateSpace(
            variable_order=["x"],
            variable_bounds={"x": (0.0, 2.0)},
            pool_size=10,
            seed=4,
        ),
        simulator=RoundingFakeSimulator(freqs=np.array([1.0, 2.0, 3.0]), step=0.5),
        optimization_spec=ScalarEncodedSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records"),
        budget=3,
        n_init=2,
        seed=4,
        run_id="resume",
        checkpoint_path=checkpoint,
    )
    first = OptimizerBoTorch(**kwargs).run()

    resumed = OptimizerBoTorch(
        candidate_space=ContinuousCandidateSpace(
            variable_order=["x"],
            variable_bounds={"x": (0.0, 2.0)},
            pool_size=10,
            seed=4,
        ),
        simulator=RoundingFakeSimulator(freqs=np.array([1.0, 2.0, 3.0]), step=0.5),
        optimization_spec=ScalarEncodedSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records_resumed"),
        budget=3,
        n_init=2,
        seed=4,
        run_id="resume",
        checkpoint_path=checkpoint,
        resume=True,
    ).run()

    assert len(resumed.history) == len(first.history)
    assert resumed.state.best_objective == first.state.best_objective
    assert [row["geometry"] for row in resumed.history] == [row["geometry"] for row in first.history]
