from __future__ import annotations

from pathlib import Path

import numpy as np

from optimization.candidate_space import DiscreteCandidateSpace
from optimization.optimization_spec import OptimizationSpec, SpecResults
from optimization.optimizer_random import OptimizerRandom
from optimization.recorder import ExperimentRecorder
from optimization.simulators import SimulationResult, Simulator


class EncodedScalarSpec(OptimizationSpec):
    def __init__(self) -> None:
        object.__setattr__(self, "objective_terms", [])
        object.__setattr__(self, "constraints", [])
        object.__setattr__(self, "frequency_bands_ghz", {})

    def evaluate(self, freqs, s_params) -> SpecResults:
        value = float(np.real(np.asarray(s_params)[0, 0, 0]))
        return SpecResults(
            objective=np.asarray([value], dtype=float),
            constraints=np.asarray([], dtype=float),
            metrics={"encoded_objective": value},
            objective_components={"encoded_objective": value},
            metadata={},
        )


class LookupSimulator(Simulator):
    def __init__(self, freqs: np.ndarray) -> None:
        super().__init__()
        self.freqs = np.asarray(freqs, dtype=float)

    def simulate(self, geometry: np.ndarray) -> SimulationResult:
        self.num_calls += 1
        actual = np.asarray(geometry, dtype=float).reshape(-1)
        objective = float(np.sum((actual - np.array([1.0, 0.0])) ** 2))
        s_params = np.zeros((2, 2, len(self.freqs)), dtype=complex)
        s_params[0, 0, :] = objective + 0j
        self.num_success += 1
        return SimulationResult(
            freqs=self.freqs,
            s_params=s_params,
            status="done",
            actual_geometry=actual,
            metadata={"case_id": f"random_{self.num_calls:03d}", "cache_hit": False},
        )


class EmptyResponseSimulator(Simulator):
    def simulate(self, geometry: np.ndarray) -> SimulationResult:
        self.num_calls += 1
        self.num_success += 1
        return SimulationResult(
            freqs=np.asarray([], dtype=float),
            s_params=np.zeros((2, 2, 0), dtype=complex),
            status="done",
            actual_geometry=np.asarray(geometry, dtype=float).reshape(-1),
            metadata={"case_id": f"empty_{self.num_calls:03d}", "cache_hit": False},
        )


def test_optimizer_random_runs_with_general_candidate_space(tmp_path: Path) -> None:
    candidate_space = DiscreteCandidateSpace(
        variable_order=["x", "y"],
        candidate_geometries=np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]]),
    )
    optimizer = OptimizerRandom(
        candidate_space=candidate_space,
        simulator=LookupSimulator(freqs=np.asarray([1.0, 2.0, 3.0])),
        optimization_spec=EncodedScalarSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records"),
        budget=3,
        n_init=1,
        seed=2,
        run_id="random_general",
        checkpoint_path=tmp_path / "random_general.checkpoint.json",
    )

    result = optimizer.run()

    assert len(result.history) == 4
    assert result.state.best_objective == min(float(row["objective"]) for row in result.history)
    assert all(row["optimizer_backend"] == "random" for row in result.history)
    assert all(row["selection_debug"]["mode"] in {"init", "random"} for row in result.history)
    assert result.recorder_output.json_path.exists()
    assert result.checkpoint_path.exists()


def test_optimizer_random_marks_empty_simulation_response_failed(tmp_path: Path) -> None:
    candidate_space = DiscreteCandidateSpace(
        variable_order=["x", "y"],
        candidate_geometries=np.asarray([[0.0, 0.0]], dtype=float),
    )
    spec = EncodedScalarSpec()
    optimizer = OptimizerRandom(
        candidate_space=candidate_space,
        simulator=EmptyResponseSimulator(),
        optimization_spec=spec,
        recorder=ExperimentRecorder(output_dir=tmp_path / "records"),
        budget=1,
        seed=2,
        run_id="random_empty_response",
        checkpoint_path=tmp_path / "random_empty_response.checkpoint.json",
    )

    result = optimizer.run()

    assert len(result.history) == 1
    trial = result.history[0]
    assert trial["simulation_status"] == "failed"
    assert trial["simulation_error"] == "simulation response has no frequency samples"
    assert trial["objective"] == float("inf")
    assert result.state.evaluated_geometries == []
