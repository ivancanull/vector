from __future__ import annotations

from pathlib import Path

from common.io_utils import write_json
from scripts.worker import cleanup_test_results


def test_cleanup_deletes_only_test_prefix_results_and_artifacts(tmp_path: Path, capsys) -> None:
    queue_root = tmp_path / "queue"
    test_result = queue_root / "failed" / "test_case_001.result.json"
    exp_result = queue_root / "done" / "exp_case_001.result.json"
    test_s2p = queue_root / "results" / "test_case_001.s2p"
    test_csv = queue_root / "results" / "test_case_001.csv"
    test_log = queue_root / "results" / "test_case_001.log"
    exp_s2p = queue_root / "results" / "exp_case_001.s2p"
    exp_log = queue_root / "results" / "exp_case_001.log"

    test_s2p.parent.mkdir(parents=True, exist_ok=True)
    for path in (test_s2p, test_csv, test_log, exp_s2p, exp_log):
        path.write_text("x", encoding="utf-8")

    write_json(
        test_result,
        {
            "job_id": "test_case_001",
            "status": "failed",
            "artifacts": {
                "touchstone": "results/test_case_001.s2p",
                "csv": "results/test_case_001.csv",
                "log": "results/test_case_001.log",
            },
        },
    )
    write_json(
        exp_result,
        {
            "job_id": "exp_case_001",
            "status": "done",
            "artifacts": {
                "touchstone": "results/exp_case_001.s2p",
                "log": "results/exp_case_001.log",
            },
        },
    )

    rc = cleanup_test_results.cleanup_results(queue_root, job_prefix="test_", dry_run=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "matched_results=1" in out
    assert "deleted_files=4" in out
    assert not test_result.exists()
    assert not test_s2p.exists()
    assert not test_csv.exists()
    assert not test_log.exists()
    assert exp_result.exists()
    assert exp_s2p.exists()
    assert exp_log.exists()


def test_cleanup_dry_run_keeps_files(tmp_path: Path, capsys) -> None:
    queue_root = tmp_path / "queue"
    result_path = queue_root / "done" / "test_case_002.result.json"
    artifact_path = queue_root / "results" / "test_case_002.log"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("x", encoding="utf-8")
    write_json(
        result_path,
        {
            "job_id": "test_case_002",
            "status": "done",
            "artifacts": {"log": "results/test_case_002.log"},
        },
    )

    rc = cleanup_test_results.cleanup_results(queue_root, job_prefix="test_", dry_run=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "mode=dry_run" in out
    assert "matched_results=1" in out
    assert "deleted_files=2" in out
    assert result_path.exists()
    assert artifact_path.exists()
