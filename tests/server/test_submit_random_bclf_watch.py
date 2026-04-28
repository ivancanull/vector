from __future__ import annotations

import argparse
from pathlib import Path

from common.io_utils import read_json, write_json
from scripts.server import submit_random_bclf_watch as script


def _args(queue_root: Path, **overrides) -> argparse.Namespace:
    values = {
        "queue_root": queue_root,
        "export_dir": queue_root / "exported",
        "batch_id": "bclf_random_test",
        "count": 5,
        "seed": 123,
        "timeout_sec": 0,
        "poll_interval_sec": 1,
        "template_name": "bclf_template_v1",
        "design_name": "HFSSDesign1",
        "setup_name": "Setup1",
        "sweep_name": "Sweep",
        "unit": "mm",
        "l_min": 7.7,
        "l_max": 7.9,
        "l1_min": 8.0,
        "l1_max": 8.2,
        "l2_min": 7.8,
        "l2_max": 8.0,
        "w_min": 0.9,
        "w_max": 1.1,
        "w1_min": 0.7,
        "w1_max": 0.9,
        "w2_min": 0.9,
        "w2_max": 1.1,
        "s1_min": 0.13,
        "s1_max": 0.16,
        "s2_min": 0.55,
        "s2_max": 0.6,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_generate_random_bclf_batch_is_seed_deterministic() -> None:
    ranges = script.BclfVariableRanges()
    first = script.generate_random_bclf_batch(
        batch_id="batch",
        count=5,
        seed=123,
        ranges=ranges,
        template_name="bclf_template_v1",
        design_name="HFSSDesign1",
        setup_name="Setup1",
        sweep_name="Sweep",
    )
    second = script.generate_random_bclf_batch(
        batch_id="batch",
        count=5,
        seed=123,
        ranges=ranges,
        template_name="bclf_template_v1",
        design_name="HFSSDesign1",
        setup_name="Setup1",
        sweep_name="Sweep",
    )

    assert [case.variables for case in first.cases] == [case.variables for case in second.cases]
    assert [case.output_filename for case in first.cases] == [case.output_filename for case in second.cases]


def test_submit_random_bclf_batch_writes_five_pending_jobs(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    batch = script.generate_random_bclf_batch(
        batch_id="bclf_random_test",
        count=5,
        seed=123,
        ranges=script.BclfVariableRanges(),
        template_name="bclf_template_v1",
        design_name="HFSSDesign1",
        setup_name="Setup1",
        sweep_name="Sweep",
    )

    emitted = script.submit_random_batch(queue_root, batch)

    assert len(emitted) == 5
    assert [path.name for path in emitted] == [
        "bclf_random_test_c01.json",
        "bclf_random_test_c02.json",
        "bclf_random_test_c03.json",
        "bclf_random_test_c04.json",
        "bclf_random_test_c05.json",
    ]
    payload = read_json(emitted[0])
    assert payload["job_id"] == "bclf_random_test_c01"
    assert payload["output_filename"].endswith(".s2p")
    assert payload["variables"]["L"].endswith("mm")
    assert payload["setup_name"] == "Setup1"
    assert payload["sweep_name"] == "Sweep"


def test_watch_results_bclf_collects_done_and_failed(tmp_path: Path) -> None:
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


def test_export_done_touchstones_copies_to_export_dir(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    src_results = queue_root / "results"
    src_results.mkdir(parents=True)
    src_touchstone = src_results / "sample.s2p"
    src_touchstone.write_text("dummy", encoding="utf-8")

    finished = {
        "job1": script.ResultSummary(
            job_id="job1",
            status="done",
            variables={},
            metrics={"max_s11_db": -1.0},
            error_message=None,
            artifacts={"touchstone": "results/sample.s2p"},
            result_path=queue_root / "done" / "job1.result.json",
        ),
        "job2": script.ResultSummary(
            job_id="job2",
            status="failed",
            variables={},
            metrics=None,
            error_message="mesh_failure",
            artifacts={"log": "results/job2.log"},
            result_path=queue_root / "failed" / "job2.result.json",
        ),
    }

    export_dir = tmp_path / "data" / "bandpass_coupled_line_filter_export" / "ext"
    copied = script.export_done_touchstones(
        queue_root=queue_root,
        finished=finished,
        export_dir=export_dir,
    )

    assert len(copied) == 1
    assert copied[0] == export_dir / "sample.s2p"
    assert copied[0].exists()


def test_run_returns_two_for_invalid_ranges(tmp_path: Path) -> None:
    args = _args(tmp_path / "queue", l_min=8.1, l_max=7.9)
    assert script.run(args) == 2
