from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from simulation import DatasetSimulator, RealHfssSimulator, SimulationError, SimulationInput
from worker.schemas import (
    ArtifactPaths,
    HfssSetupConfig,
    HfssSweepConfig,
    JobResult,
    WorkerConfig,
)
from worker.template_registry import ResolvedTemplate
from worker.utils import utc_now


def _write_dataset_pickle(path: Path) -> np.ndarray:
    s_params = np.zeros((2, 2, 3), dtype=complex)
    s_params[0, 0] = np.array([0.1, 0.2, 0.3]) + 1j * np.array([0.0, 0.1, 0.2])
    dataset = [
        {
            "caseid": 7,
            "geo_params": [55.0, 2.0, 15.0],
            "s_params": s_params,
            "sampled_indices": None,
        }
    ]
    with path.open("wb") as f:
        pickle.dump({"dataset": dataset, "sample_idx": [7], "available_idx": [], "history_idx": [7]}, f)
    return s_params


def test_dataset_simulator_returns_matching_generated_sample(tmp_path: Path) -> None:
    dataset_path = tmp_path / "online_dataset.pkl"
    expected_s_params = _write_dataset_pickle(dataset_path)
    simulator = DatasetSimulator(
        dataset_path=dataset_path,
        freqs=np.array([1.0, 2.0, 3.0]),
        port_num=2,
        param_start_idx=0,
        variable_order=["Din", "H", "W"],
    )

    output = simulator.simulate(
        SimulationInput(
            case_id="case_a",
            variables={"Din": 55.0, "H": 2.0, "W": 15.0},
            variable_order=["Din", "H", "W"],
        )
    )

    assert output.case_id == "case_a"
    np.testing.assert_allclose(output.geo_params, [55.0, 2.0, 15.0])
    np.testing.assert_allclose(output.freqs, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(output.s_params, expected_s_params)
    assert output.s_params.dtype == complex
    assert output.to_dataset_item()["sampled_indices"] is None


def test_dataset_simulator_raises_when_no_sample_matches(tmp_path: Path) -> None:
    dataset_path = tmp_path / "online_dataset.pkl"
    _write_dataset_pickle(dataset_path)
    simulator = DatasetSimulator(
        dataset_path=dataset_path,
        freqs=np.array([1.0, 2.0, 3.0]),
        port_num=2,
        param_start_idx=0,
        variable_order=["Din", "H", "W"],
    )

    with pytest.raises(SimulationError) as exc:
        simulator.simulate(
            SimulationInput(
                case_id="missing",
                variables={"Din": 56.0, "H": 2.0, "W": 15.0},
                variable_order=["Din", "H", "W"],
            )
        )

    assert exc.value.failure_class == "sample_not_found"
    assert "No generated sample matches" in exc.value.message


def test_real_hfss_simulator_formats_job_and_parses_touchstone(tmp_path: Path) -> None:
    touchstone_path = tmp_path / "queue" / "results" / "case_001.s2p"
    touchstone_path.parent.mkdir(parents=True)
    touchstone_path.write_text(
        "# Hz S RI R 50\n"
        "1.0 0.1 0.0 0.2 0.0 0.3 0.0 0.4 0.0\n"
        "2.0 0.5 0.1 0.6 0.1 0.7 0.1 0.8 0.1\n",
        encoding="utf-8",
    )

    captured = {}

    class FakeExecutor:
        def execute(self, job, template_cfg, worker_cfg):
            captured["job"] = job
            captured["template_cfg"] = template_cfg
            captured["worker_cfg"] = worker_cfg
            return JobResult(
                job_id=job.job_id,
                status="done",
                worker_name="fake",
                start_time=utc_now(),
                finish_time=utc_now(),
                metrics=None,
                artifacts=ArtifactPaths(touchstone="results/case_001.s2p"),
                error_message=None,
            )

    simulator = RealHfssSimulator(
        template_cfg=_template(),
        worker_cfg=_worker_config(tmp_path / "queue"),
        port_num=2,
        unit_map={"Din": "um", "H": "um"},
        variable_order=["Din", "H"],
        executor=FakeExecutor(),
    )

    output = simulator.simulate(
        SimulationInput(
            case_id="case_001",
            variables={"Din": 55.0, "H": 2.0},
            variable_order=["Din", "H"],
        )
    )

    job = captured["job"]
    assert job.job_id == "case_001"
    assert job.variables == {"Din": "55um", "H": "2um"}
    assert job.output_filename == "case_001.s2p"
    assert captured["template_cfg"].template_name == "spiral"
    assert captured["worker_cfg"].queue_root == tmp_path / "queue"
    np.testing.assert_allclose(output.freqs, [1.0, 2.0])
    assert output.s_params.shape == (2, 2, 2)
    np.testing.assert_allclose(output.s_params[0, 0], [0.1 + 0j, 0.5 + 0.1j])
    assert output.artifacts == {"touchstone": touchstone_path}


def _template() -> ResolvedTemplate:
    return ResolvedTemplate(
        template_name="spiral",
        project_path=Path("template.aedt"),
        design_name="HFSSDesign1",
        setup_name="Setup1",
        sweep_name="Sweep",
        hfss_setup=HfssSetupConfig(solution_frequency="1GHz"),
        hfss_sweep=HfssSweepConfig(sweep_type="interpolating", sweep_data="LINC 1GHz 2GHz 2"),
    )


def _worker_config(queue_root: Path) -> WorkerConfig:
    return WorkerConfig(
        worker_name="fake",
        queue_root=queue_root,
        templates_yaml=queue_root / "templates.yaml",
        poll_interval_sec=1,
        claim_retry_sec=1,
        results_dir=queue_root / "results",
        logs_dir=queue_root / "logs",
    )
