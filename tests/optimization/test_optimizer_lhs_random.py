from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from optimization.candidate_space import DiscreteCandidateSpace
from optimization.optimization_spec import OptimizationSpec, SpecResults
from optimization.optimizer_lhs_random import OptimizerLHSRandom
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
    def __init__(self) -> None:
        super().__init__()
        self.freqs = np.asarray([1.0, 2.0, 3.0], dtype=float)

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
            metadata={"case_id": f"lhs_{self.num_calls:03d}", "cache_hit": False},
        )


def _candidate_space() -> DiscreteCandidateSpace:
    return DiscreteCandidateSpace(
        variable_order=["x", "y"],
        candidate_geometries=np.asarray(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [1.5, 0.5],
                [2.0, 1.0],
                [2.5, 1.0],
            ],
            dtype=float,
        ),
    )


def _run_lhs(tmp_path: Path, *, seed: int) -> list[dict]:
    optimizer = OptimizerLHSRandom(
        candidate_space=_candidate_space(),
        simulator=LookupSimulator(),
        optimization_spec=EncodedScalarSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / f"records_{seed}"),
        budget=3,
        n_init=2,
        seed=seed,
        run_id=f"lhs_seed_{seed}",
        checkpoint_path=tmp_path / f"lhs_seed_{seed}.checkpoint.json",
    )
    return optimizer.run().history


def test_optimizer_lhs_random_runs_with_general_candidate_space(tmp_path: Path) -> None:
    history = _run_lhs(tmp_path, seed=11)

    assert len(history) == 5
    assert all(row["optimizer_backend"] == "lhs_random" for row in history)
    assert min(float(row["objective"]) for row in history) == history[-1]["best_so_far"]
    assert all(row["selection_debug"]["mode"] == "lhs_random" for row in history)


def test_optimizer_lhs_random_is_deterministic_for_seed(tmp_path: Path) -> None:
    first = _run_lhs(tmp_path / "first", seed=7)
    second = _run_lhs(tmp_path / "second", seed=7)

    first_geometries = [row["requested_geometry"] for row in first]
    second_geometries = [row["requested_geometry"] for row in second]

    assert first_geometries == second_geometries


def test_optimizer_lhs_random_uses_lhs_for_initial_trials(tmp_path: Path) -> None:
    history = _run_lhs(tmp_path, seed=3)

    init_rows = [row for row in history if row["phase"] == "init"]

    assert len(init_rows) == 2
    assert [row["selection_debug"]["mode"] for row in init_rows] == ["lhs_random", "lhs_random"]
    assert [row["selection_debug"]["lhs_index"] for row in init_rows] == [0, 1]


def test_optimizer_lhs_random_does_not_repeat_candidates(tmp_path: Path) -> None:
    history = _run_lhs(tmp_path, seed=19)

    selected = [tuple(row["requested_geometry"]) for row in history]

    assert len(selected) == len(set(selected))


def test_optimizer_lhs_random_resume_uses_checkpoint_seed_without_rng_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "lhs_seed_7.checkpoint.json"
    first = OptimizerLHSRandom(
        candidate_space=_candidate_space(),
        simulator=LookupSimulator(),
        optimization_spec=EncodedScalarSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records_first"),
        budget=1,
        seed=7,
        run_id="lhs_seed_7",
        checkpoint_path=checkpoint,
    )
    first.run()

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload.pop("rng_state", None)
    checkpoint.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    resumed = OptimizerLHSRandom(
        candidate_space=_candidate_space(),
        simulator=LookupSimulator(),
        optimization_spec=EncodedScalarSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records_resumed"),
        budget=3,
        seed=999,
        run_id="lhs_seed_7",
        checkpoint_path=checkpoint,
        resume=True,
    )
    expected = OptimizerLHSRandom(
        candidate_space=_candidate_space(),
        simulator=LookupSimulator(),
        optimization_spec=EncodedScalarSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / "records_expected"),
        budget=3,
        seed=7,
        run_id="lhs_seed_7_expected",
        checkpoint_path=tmp_path / "lhs_seed_7_expected.checkpoint.json",
    )

    assert resumed.seed == 7
    assert np.allclose(resumed._lhs_sequence, expected._lhs_sequence)
    assert int(resumed._rng.integers(0, 10000)) == int(np.random.default_rng(7).integers(0, 10000))
