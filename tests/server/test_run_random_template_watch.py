from __future__ import annotations

import argparse
from pathlib import Path

from common.io_utils import read_json, write_json
from scripts.server import run_random_template_watch as script


def _args(queue_root: Path, **overrides) -> argparse.Namespace:
    values = {
        "template_name": "bclf_template_v1",
        "queue_root": queue_root,
        "batch_id": "random_batch_test",
        "count": 5,
        "seed": 123,
        "timeout_sec": 0,
        "poll_interval_sec": 1,
        "design_name": "HFSSDesign1",
        "setup_name": "Setup1",
        "sweep_name": "Sweep",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_bclf_generation_is_seed_deterministic() -> None:
    first = script.generate_random_bclf_experiment(
        batch_id="batch",
        count=5,
        seed=123,
        template_name="bclf_template_v1",
        design_name="HFSSDesign1",
        setup_name="Setup1",
        sweep_name="Sweep",
    )
    second = script.generate_random_bclf_experiment(
        batch_id="batch",
        count=5,
        seed=123,
        template_name="bclf_template_v1",
        design_name="HFSSDesign1",
        setup_name="Setup1",
        sweep_name="Sweep",
    )
    assert [case.variables for case in first.cases] == [case.variables for case in second.cases]
    assert [case.output_filename for case in first.cases] == [case.output_filename for case in second.cases]


def test_si_generation_is_seed_deterministic() -> None:
    first = script.generate_random_spiral_experiment(
        batch_id="batch",
        count=5,
        seed=123,
        template_name="si_template_v1",
        design_name="HFSSDesign1",
        setup_name="Setup1",
        sweep_name="Sweep",
    )
    second = script.generate_random_spiral_experiment(
        batch_id="batch",
        count=5,
        seed=123,
        template_name="si_template_v1",
        design_name="HFSSDesign1",
        setup_name="Setup1",
        sweep_name="Sweep",
    )
    assert [case.variables for case in first.cases] == [case.variables for case in second.cases]
    assert [case.output_filename for case in first.cases] == [case.output_filename for case in second.cases]


def test_new_spiral_template_generation_uses_template_specific_design() -> None:
    exp = script._build_experiment(
        template_name="spiral_inductor_t3_template_v1",
        batch_id="batch",
        count=2,
        seed=123,
        design_name="SpiralInductorT3",
        setup_name="Setup1",
        sweep_name="Sweep",
    )

    assert exp.template_name == "spiral_inductor_t3_template_v1"
    assert exp.design_name == "SpiralInductorT3"
    assert exp.setup_name == "Setup1"
    assert exp.sweep_name == "Sweep"
    assert len(exp.cases) == 2


def test_submit_writes_expected_pending_jobs_for_template(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    exp = script.generate_random_bclf_experiment(
        batch_id="bclf_random_test",
        count=5,
        seed=123,
        template_name="bclf_template_v1",
        design_name="HFSSDesign1",
        setup_name="Setup1",
        sweep_name="Sweep",
    )

    emitted, expected_jobs, batch_id = script.submit_random_batch(queue_root, exp)
    assert batch_id == "bclf_random_test"
    assert len(emitted) == 5
    assert len(expected_jobs) == 5
    payload = read_json(emitted[0])
    assert payload["template_name"] == "bclf_template_v1"
    assert payload["output_filename"].endswith(".s2p")


def test_watch_results_collects_done_and_failed(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    (queue_root / "done").mkdir(parents=True)
    (queue_root / "failed").mkdir(parents=True)
    expected = {
        "job_done": {"L": "7.70mm"},
        "job_failed": {"L": "7.90mm"},
    }
    write_json(
        queue_root / "done" / "job_done.result.json",
        {
            "job_id": "job_done",
            "status": "done",
            "metrics": {"max_s11_db": -0.1},
            "artifacts": {"touchstone": "results/job_done.s2p"},
            "error_message": None,
        },
    )
    write_json(
        queue_root / "failed" / "job_failed.result.json",
        {
            "job_id": "job_failed",
            "status": "failed",
            "metrics": None,
            "artifacts": {"log": "results/job_failed.log"},
            "error_message": "mesh_failure",
        },
    )

    finished, missing = script.watch_results(
        queue_root=queue_root,
        expected_jobs=expected,
        timeout_sec=0,
        poll_interval_sec=1,
    )

    assert missing == set()
    assert finished["job_done"].metrics == {"max_s11_db": -0.1}
    assert finished["job_failed"].error_message == "mesh_failure"


def test_run_with_unsupported_template_returns_two(tmp_path: Path) -> None:
    args = _args(tmp_path / "queue", template_name="unknown_template")
    assert script.run(args) == 2


def test_run_timeout_path_no_export_side_effect(tmp_path: Path) -> None:
    args = _args(tmp_path / "queue", template_name="si_template_v1", timeout_sec=0)
    rc = script.run(args)
    assert rc == 1
    # no export-copy path in unified script; ensure only queue tree exists
    assert (tmp_path / "queue" / "pending").exists()
    assert not (tmp_path / "data").exists()
