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


def test_continuous_candidate_space_initial_sampling_is_not_limited_by_pool_size() -> None:
    continuous = ContinuousCandidateSpace(
        variable_order=["x", "y"],
        variable_bounds={"x": (0.0, 1.0), "y": (2.0, 3.0)},
        pool_size=1,
        seed=1,
        resolution=0.1,
    )

    samples = continuous.sample_initial(5, np.random.default_rng(2))

    assert samples.shape == (5, 2)
    assert np.all(samples[:, 0] >= 0.0) and np.all(samples[:, 0] <= 1.0)
    assert np.all(samples[:, 1] >= 2.0) and np.all(samples[:, 1] <= 3.0)
    assert np.allclose(samples / 0.1, np.round(samples / 0.1))


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


def test_online_optimizer_uses_fixed_initial_geometries_and_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fixed_init.checkpoint.json"
    initial_geometries = np.asarray([[1.5], [0.0]], dtype=float)
    metadata = [{"source": "shared_lhs_cache"}, {"source": "shared_lhs_cache"}]

    first = OptimizerBoTorch(
        candidate_space=DiscreteCandidateSpace(
            variable_order=["x"],
            candidate_geometries=np.asarray([[0.0], [0.5], [1.0], [1.5], [2.0]], dtype=float),
        ),
        simulator=RoundingFakeSimulator(freqs=np.array([1.0, 2.0, 3.0]), step=0.1),
        optimization_spec=ScalarEncodedSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records"),
        budget=1,
        n_init=2,
        seed=11,
        run_id="fixed_init",
        checkpoint_path=checkpoint,
        initial_geometries=initial_geometries,
        initial_geometry_metadata=metadata,
    ).run()

    assert [row["requested_geometry"] for row in first.history[:2]] == [[1.5], [0.0]]
    assert [row["selection_debug"]["mode"] for row in first.history[:2]] == [
        "fixed_initial_geometry",
        "fixed_initial_geometry",
    ]
    assert first.history[0]["selection_debug"]["source"] == "shared_lhs_cache"

    resumed = OptimizerBoTorch(
        candidate_space=DiscreteCandidateSpace(
            variable_order=["x"],
            candidate_geometries=np.asarray([[0.0], [0.5], [1.0], [1.5], [2.0]], dtype=float),
        ),
        simulator=RoundingFakeSimulator(freqs=np.array([1.0, 2.0, 3.0]), step=0.1),
        optimization_spec=ScalarEncodedSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records_resumed"),
        budget=1,
        n_init=2,
        seed=11,
        run_id="fixed_init",
        checkpoint_path=checkpoint,
        resume=True,
        initial_geometries=initial_geometries,
        initial_geometry_metadata=metadata,
    ).run()

    assert len(resumed.history) == len(first.history)
    assert [row["requested_geometry"] for row in resumed.history[:2]] == [[1.5], [0.0]]


def test_trust_region_filters_candidates_in_deterministic_path(tmp_path: Path) -> None:
    # Ordered points so deterministic fallback picks first candidate in filtered subset.
    pool = np.asarray([[0.0], [0.3], [0.7], [1.0]], dtype=float)
    space = DiscreteCandidateSpace(variable_order=["x"], candidate_geometries=pool)
    opt = OptimizerBoTorch(
        candidate_space=space,
        simulator=RoundingFakeSimulator(freqs=np.array([1.0]), step=0.1),
        optimization_spec=ScalarEncodedSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records"),
        budget=1,
        n_init=0,
        seed=0,
        run_id="tr_filter",
        checkpoint_path=tmp_path / "tr_filter.checkpoint.json",
        use_trust_region=True,
        tr_length_init=0.2,
    )
    # Force model-not-ready path and explicit center.
    opt._model_ready = False
    opt._tr_center_norm = np.asarray([0.65], dtype=float)
    sel = opt._select_next(space.remaining())
    # in normalized [0,1], 0.7 is inside; 0.3 is outside for length=0.2
    assert np.allclose(sel.requested_geometry, np.asarray([0.7]))
    assert sel.selection_debug is not None
    assert sel.selection_debug.get("tr_filter_applied") is True
    assert int(sel.selection_debug.get("tr_candidates_in_region")) >= 1


def test_trust_region_state_roundtrip_and_backward_compat(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tr_state.checkpoint.json"
    kwargs = dict(
        candidate_space=ContinuousCandidateSpace(
            variable_order=["x"],
            variable_bounds={"x": (0.0, 2.0)},
            pool_size=12,
            seed=5,
        ),
        simulator=RoundingFakeSimulator(freqs=np.array([1.0, 2.0]), step=0.5),
        optimization_spec=ScalarEncodedSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records"),
        budget=2,
        n_init=1,
        seed=5,
        run_id="tr_state",
        checkpoint_path=checkpoint,
        use_trust_region=True,
    )
    first = OptimizerBoTorch(**kwargs).run()
    payload = checkpoint.read_text(encoding="utf-8")
    assert '"trust_region"' in payload

    resumed = OptimizerBoTorch(
        candidate_space=ContinuousCandidateSpace(
            variable_order=["x"],
            variable_bounds={"x": (0.0, 2.0)},
            pool_size=12,
            seed=5,
        ),
        simulator=RoundingFakeSimulator(freqs=np.array([1.0, 2.0]), step=0.5),
        optimization_spec=ScalarEncodedSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records_resumed"),
        budget=2,
        n_init=1,
        seed=5,
        run_id="tr_state",
        checkpoint_path=checkpoint,
        resume=True,
        use_trust_region=True,
    ).run()
    assert len(first.history) == len(resumed.history)

    # Backward-compat simulation: remove trust_region section and ensure resume does not crash.
    import json

    old_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    old_payload.pop("trust_region", None)
    checkpoint.write_text(json.dumps(old_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    compat = OptimizerBoTorch(
        candidate_space=ContinuousCandidateSpace(
            variable_order=["x"],
            variable_bounds={"x": (0.0, 2.0)},
            pool_size=12,
            seed=5,
        ),
        simulator=RoundingFakeSimulator(freqs=np.array([1.0, 2.0]), step=0.5),
        optimization_spec=ScalarEncodedSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records_compat"),
        budget=2,
        n_init=1,
        seed=5,
        run_id="tr_state",
        checkpoint_path=checkpoint,
        resume=True,
        use_trust_region=True,
    )
    assert compat.use_trust_region is True
