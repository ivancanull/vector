from __future__ import annotations

import argparse
from pathlib import Path

from common.io_utils import read_json, write_json
from scripts.server import submit_random_spiral_watch as script


def _args(queue_root: Path, **overrides) -> argparse.Namespace:
    values = {
        "queue_root": queue_root,
        "batch_id": "spiral_random_test",
        "count": 5,
        "seed": 123,
        "timeout_sec": 0,
        "poll_interval_sec": 1,
        "template_name": "si_template_v1",
        "design_name": "HFSSDesign1",
        "setup_name": "Setup1",
        "sweep_name": "Sweep",
        "din_min_um": 50.0,
        "din_max_um": 70.0,
        "s_min_um": 1.0,
        "s_max_um": 2.0,
        "w_min_um": 12.0,
        "w_max_um": 18.0,
        "h_um": 2.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_generate_random_spiral_batch_is_seed_deterministic() -> None:
    ranges = script.RandomSpiralRanges()
    first = script.generate_random_spiral_batch(
        batch_id="batch",
        count=5,
        seed=123,
        ranges=ranges,
        template_name="si_template_v1",
        design_name="HFSSDesign1",
        setup_name="Setup1",
        sweep_name="Sweep",
    )
    second = script.generate_random_spiral_batch(
        batch_id="batch",
        count=5,
        seed=123,
        ranges=ranges,
        template_name="si_template_v1",
        design_name="HFSSDesign1",
        setup_name="Setup1",
        sweep_name="Sweep",
    )

    assert [case.variables for case in first.cases] == [case.variables for case in second.cases]
    assert [case.output_filename for case in first.cases] == [
        case.output_filename for case in second.cases
    ]


def test_submit_random_batch_writes_five_valid_pending_jobs(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    batch = script.generate_random_spiral_batch(
        batch_id="spiral_random_test",
        count=5,
        seed=123,
        ranges=script.RandomSpiralRanges(),
        template_name="si_template_v1",
        design_name="HFSSDesign1",
        setup_name="Setup1",
        sweep_name="Sweep",
    )

    emitted = script.submit_random_batch(queue_root, batch)

    assert len(emitted) == 5
    assert [path.name for path in emitted] == [
        "spiral_random_test_c01.json",
        "spiral_random_test_c02.json",
        "spiral_random_test_c03.json",
        "spiral_random_test_c04.json",
        "spiral_random_test_c05.json",
    ]
    payload = read_json(emitted[0])
    assert payload["job_id"] == "spiral_random_test_c01"
    assert payload["output_filename"].endswith(".s2p")
    assert payload["variables"]["Din"].endswith("um")
    assert payload["variables"]["H"] == "2.00um"


def test_watch_results_collects_done_and_failed(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    (queue_root / "done").mkdir(parents=True)
    (queue_root / "failed").mkdir(parents=True)
    expected = {
        "job_done": {"Din": "55um", "H": "2um", "S": "1.5um", "W": "15um"},
        "job_failed": {"Din": "60um", "H": "2um", "S": "1.2um", "W": "14um"},
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


def test_watch_results_times_out_when_result_missing(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    expected = {"job_missing": {"Din": "55um"}}

    finished, missing = script.watch_results(
        queue_root=queue_root,
        expected_jobs=expected,
        timeout_sec=0,
        poll_interval_sec=1,
    )

    assert finished == {}
    assert missing == {"job_missing"}


def test_run_returns_two_for_invalid_ranges(tmp_path: Path) -> None:
    args = _args(tmp_path / "queue", din_min_um=80.0, din_max_um=70.0)
    assert script.run(args) == 2
