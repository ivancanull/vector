from pathlib import Path

from worker.queue_watcher import try_claim_one_job


def _mkdir_queue(root: Path) -> None:
    for name in ("pending", "running", "done", "failed", "results"):
        (root / name).mkdir(parents=True, exist_ok=True)


def test_claim_one_pending_job(tmp_path: Path) -> None:
    _mkdir_queue(tmp_path)
    job_path = tmp_path / "pending" / "job1.json"
    job_path.write_text('{"job_id":"job1"}', encoding="utf-8")

    claimed = try_claim_one_job(tmp_path)
    assert claimed is not None
    assert claimed.name == "job1.json"
    assert claimed.exists()
    assert not job_path.exists()


def test_double_claim_prevented(tmp_path: Path) -> None:
    _mkdir_queue(tmp_path)
    job_path = tmp_path / "pending" / "job1.json"
    job_path.write_text('{"job_id":"job1"}', encoding="utf-8")

    first = try_claim_one_job(tmp_path)
    second = try_claim_one_job(tmp_path)

    assert first is not None
    assert second is None
