from __future__ import annotations

from pathlib import Path

import numpy as np

from optimization.candidate_space import DiscreteCandidateSpace
from optimization.optimization_spec import OptimizationSpec, SpecResults
from optimization.optimizer_ann import OptimizerANN
from optimization.recorder import ExperimentRecorder
from optimization.simulators import SimulationResult, Simulator


class ResponseEncodedSpec(OptimizationSpec):
    def __init__(self) -> None:
        object.__setattr__(self, "objective_terms", [])
        object.__setattr__(self, "constraints", [])
        object.__setattr__(self, "frequency_bands_ghz", {})
        self.evaluate_calls = 0

    def evaluate(self, freqs, s_params) -> SpecResults:
        self.evaluate_calls += 1
        _ = freqs
        values = np.asarray(s_params, dtype=complex)
        objective = float(np.real(values[1, 0, 0]))
        return SpecResults(
            objective=np.asarray([objective], dtype=float),
            constraints=np.asarray([], dtype=float),
            metrics={"encoded_objective": objective},
            objective_components={"encoded_objective": objective},
            metadata={},
        )

    def to_objective(self):  # pragma: no cover
        raise AssertionError("test spec uses evaluate directly")


class ResponseFakeSimulator(Simulator):
    def __init__(self, freqs: np.ndarray) -> None:
        super().__init__()
        self.freqs = np.asarray(freqs, dtype=float)

    def simulate(self, geometry: np.ndarray) -> SimulationResult:
        self.num_calls += 1
        actual = np.asarray(geometry, dtype=float).reshape(-1)
        objective = float((actual[0] - 0.25) ** 2 + (actual[1] - 0.75) ** 2)
        s_params = np.zeros((2, 2, len(self.freqs)), dtype=complex)
        s_params[0, 0, :] = objective + 0.1j
        s_params[0, 1, :] = objective + 0.2j
        s_params[1, 0, :] = objective + 0.2j
        self.num_success += 1
        return SimulationResult(
            freqs=self.freqs,
            s_params=s_params,
            status="done",
            metadata={"cache_hit": False, "job_id": f"ann_{self.num_calls}"},
            actual_geometry=actual,
        )


def _optimizer(tmp_path: Path, *, checkpoint: Path | None = None, resume: bool = False) -> OptimizerANN:
    candidates = np.asarray([[0.0, 0.0], [0.25, 0.75], [1.0, 1.0]], dtype=float)
    return OptimizerANN(
        candidate_space=DiscreteCandidateSpace(variable_order=["x", "y"], candidate_geometries=candidates),
        simulator=ResponseFakeSimulator(freqs=np.linspace(1.0, 2.0, 401)),
        optimization_spec=ResponseEncodedSpec(),
        recorder=ExperimentRecorder(output_dir=tmp_path / ("records_resumed" if resume else "records")),
        budget=1,
        n_init=1,
        seed=5,
        run_id="ann_online",
        checkpoint_path=checkpoint or tmp_path / "ann_online.checkpoint.json",
        resume=resume,
        epochs=1,
        batch_size=1,
        lr=1e-3,
        device="cpu",
        model_config={"dropout": 0.0},
    )


def test_optimizer_ann_runs_and_uses_ann_selection(tmp_path: Path) -> None:
    result = _optimizer(tmp_path).run()

    assert len(result.history) == 2
    assert result.history[-1]["optimizer_backend"] == "ann"
    assert result.history[-1]["selection_debug"]["mode"] == "ann_predicted_objective"
    assert result.checkpoint_path.with_suffix(".ann_training.npz").exists()


def test_optimizer_ann_resume_loads_sidecar_and_preserves_history(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resume_ann.checkpoint.json"
    first = _optimizer(tmp_path, checkpoint=checkpoint).run()
    resumed = _optimizer(tmp_path, checkpoint=checkpoint, resume=True).run()

    assert len(resumed.history) == len(first.history)
    assert resumed.state.best_objective == first.state.best_objective
    assert checkpoint.with_suffix(".ann_training.npz").exists()


def test_optimizer_ann_select_failure_falls_back_deterministically(tmp_path: Path) -> None:
    opt = _optimizer(tmp_path)
    opt._model_ready = True
    opt._freqs = np.linspace(1.0, 2.0, 401)

    selection = opt._select_next(opt.candidate_space.remaining())

    assert np.allclose(selection.requested_geometry, np.asarray([0.0, 0.0]))
    assert selection.selection_debug is not None
    assert selection.selection_debug["mode"] == "deterministic_select_error"
    assert "select_error" in selection.selection_debug
