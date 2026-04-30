from __future__ import annotations

from pathlib import Path

import numpy as np

from optimization.candidate_space import DiscreteCandidateSpace
from optimization.datasets import PassiveDataset
from optimization.optimization_spec import OptimizationSpec, SpecResults
from optimization.optimizer_bo import OptimizerBoTorch
from optimization.optimizer_random import OptimizerRandom
from optimization.recorder import ExperimentRecorder
from optimization.simulators import ReplaySimulator


class EncodedDatasetSpec(OptimizationSpec):
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


def _dataset() -> PassiveDataset:
    freqs = np.asarray([1.0, 2.0, 3.0], dtype=float)
    samples = []
    for i, x in enumerate([0.0, 0.5, 1.0, 1.5, 2.0]):
        s_params = np.zeros((2, 2, len(freqs)), dtype=complex)
        s_params[0, 0, :] = float((x - 1.0) ** 2) + 0j
        s_params[1, 0, :] = -6.0 + 0j
        samples.append({"caseid": f"case_{i}", "geo_params": [x], "s_params": s_params})
    return PassiveDataset(samples=samples, variable_order=["x"], freqs=freqs)


def _candidate_space(dataset: PassiveDataset) -> DiscreteCandidateSpace:
    return DiscreteCandidateSpace(
        variable_order=dataset.variable_order,
        candidate_geometries=np.asarray([dataset.get_geometry(i) for i in range(len(dataset))], dtype=float),
    )


def test_replay_simulator_with_general_botorch_optimizer(tmp_path: Path) -> None:
    dataset = _dataset()
    optimizer = OptimizerBoTorch(
        candidate_space=_candidate_space(dataset),
        simulator=ReplaySimulator(dataset=dataset),
        optimization_spec=EncodedDatasetSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "bo"),
        budget=3,
        n_init=2,
        seed=0,
        run_id="replay_general_bo",
        checkpoint_path=tmp_path / "replay_general_bo.checkpoint.json",
    )

    result = optimizer.run()

    assert len(result.history) == 5
    assert result.state.best_objective == min(float(row["objective"]) for row in result.history)
    assert all(row["optimizer_backend"] == "botorch" for row in result.history)
    assert all(row["selection_debug"]["actual_geometry_source"] == "simulation.actual_geometry" for row in result.history)


def test_replay_simulator_with_general_random_optimizer(tmp_path: Path) -> None:
    dataset = _dataset()
    optimizer = OptimizerRandom(
        candidate_space=_candidate_space(dataset),
        simulator=ReplaySimulator(dataset=dataset),
        optimization_spec=EncodedDatasetSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "random"),
        budget=3,
        n_init=1,
        seed=1,
        run_id="replay_general_random",
        checkpoint_path=tmp_path / "replay_general_random.checkpoint.json",
    )

    result = optimizer.run()

    assert len(result.history) == 4
    assert result.state.best_objective == min(float(row["objective"]) for row in result.history)
    assert all(row["optimizer_backend"] == "random" for row in result.history)
