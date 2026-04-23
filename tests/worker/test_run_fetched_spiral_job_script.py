from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from common.io_utils import read_json, write_json
from scripts.server.submit_spiral_batch import submit_batch
from scripts.worker import run_fetched_spiral_job


def test_run_fetched_job_missing_or_invalid_file(tmp_path: Path) -> None:
    missing = tmp_path / "queue" / "running" / "missing.json"
    assert run_fetched_spiral_job.run_fetched_job(missing, tmp_path / "worker.yaml", True) == 2

    invalid = tmp_path / "queue" / "running" / "invalid.json"
    invalid.parent.mkdir(parents=True, exist_ok=True)
    invalid.write_text("{ invalid json", encoding="utf-8")
    assert run_fetched_spiral_job.run_fetched_job(invalid, tmp_path / "worker.yaml", True) == 2


def test_retry_path_runs_once_and_applies_fix(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    job_file = queue_root / "running" / "demo_c01.json"
    write_json(
        job_file,
        {
            "job_id": "demo_c01",
            "template_name": "si_template_v1",
            "variables": {"Din": "55um", "H": "2um", "S": "1.5um", "W": "15um"},
            "output_filename": "demo_output",
            "submit_time": "2026-04-22T00:00:00Z",
            "metrics_to_extract": ["max_s11_db"],
        },
    )

    calls = {"count": 0}

    def _fake_run_once(path: Path, _worker_cfg_path: Path):
        calls["count"] += 1
        path_payload = read_json(path)
        failed_path = queue_root / "failed" / "demo_c01.result.json"
        done_path = queue_root / "done" / "demo_c01.result.json"
        if calls["count"] == 1:
            write_json(
                failed_path,
                {
                    "job_id": "demo_c01",
                    "status": "failed",
                    "error_message": "invalid_job_json",
                    "artifacts": {"log": "results/demo_c01.log"},
                },
            )
        else:
            assert path_payload["output_filename"] == "demo_output.s2p"
            write_json(
                done_path,
                {
                    "job_id": "demo_c01",
                    "status": "done",
                    "artifacts": {
                        "touchstone": "results/demo_output.s2p",
                        "log": "results/demo_c01.log",
                    },
                },
            )
        return SimpleNamespace(queue_root=queue_root)

    monkeypatch.setattr(run_fetched_spiral_job, "_run_once", _fake_run_once)

    rc = run_fetched_spiral_job.run_fetched_job(job_file, tmp_path / "worker.yaml", True)
    assert rc == 0
    assert calls["count"] == 2


def test_local_queue_submit_then_run_fetched_job(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "batch_id": "spiral_demo",
                "template_name": "si_template_v1",
                "metrics_to_extract": ["max_s11_db"],
                "cases": [
                    {
                        "case_id": "c01",
                        "variables": {
                            "Din": "55um",
                            "H": "2um",
                            "S": "1.5um",
                            "W": "15um",
                        },
                        "output_filename": "spiral_demo_c01.s2p",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    emitted = submit_batch(batch, queue_root)
    assert len(emitted) == 1
    pending_path = emitted[0]
    running_path = queue_root / "running" / pending_path.name
    running_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.rename(running_path)

    def _fake_run_once(path: Path, _worker_cfg_path: Path):
        path_payload = read_json(path)
        assert path_payload["template_name"] == "si_template_v1"
        assert path_payload["output_filename"] == "spiral_demo_c01.s2p"
        write_json(
            queue_root / "done" / "spiral_demo_c01.result.json",
            {
                "job_id": "spiral_demo_c01",
                "status": "done",
                "artifacts": {
                    "touchstone": "results/spiral_demo_c01.s2p",
                    "log": "results/spiral_demo_c01.log",
                },
            },
        )
        return SimpleNamespace(queue_root=queue_root)

    monkeypatch.setattr(run_fetched_spiral_job, "_run_once", _fake_run_once)
    rc = run_fetched_spiral_job.run_fetched_job(running_path, tmp_path / "worker.yaml", True)
    assert rc == 0
    assert (queue_root / "done" / "spiral_demo_c01.result.json").exists()


def test_run_fetched_job_removes_stale_result_before_first_attempt(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    queue_root = tmp_path / "queue"
    job_file = queue_root / "running" / "demo_stale_c01.json"
    stale_failed = queue_root / "failed" / "demo_stale_c01.result.json"
    write_json(
        job_file,
        {
            "job_id": "demo_stale_c01",
            "template_name": "si_template_v1",
            "variables": {"Din": "55um", "H": "2um", "S": "1.5um", "W": "15um"},
            "output_filename": "demo_stale.s2p",
            "submit_time": "2026-04-22T00:00:00Z",
            "metrics_to_extract": ["max_s11_db"],
        },
    )
    write_json(stale_failed, {"job_id": "demo_stale_c01", "status": "failed"})

    def _fake_run_once(path: Path, _worker_cfg_path: Path):
        assert path == job_file
        assert not stale_failed.exists()
        write_json(
            queue_root / "done" / "demo_stale_c01.result.json",
            {
                "job_id": "demo_stale_c01",
                "status": "done",
                "artifacts": {
                    "touchstone": "results/demo_stale.s2p",
                    "log": "results/demo_stale_c01.log",
                },
            },
        )
        return SimpleNamespace(queue_root=queue_root)

    monkeypatch.setattr(run_fetched_spiral_job, "_run_once", _fake_run_once)
    rc = run_fetched_spiral_job.run_fetched_job(job_file, tmp_path / "worker.yaml", True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "removed_stale_result=" in out
