from __future__ import annotations

from pathlib import Path

from scripts.worker import run_local_queue_worker as script
from scripts.worker.run_local_queue_worker import _resolve_results_dir


def test_local_queue_worker_rebases_default_results_dir(tmp_path: Path) -> None:
    original_queue = tmp_path / "queue"
    original_results = original_queue / "results"
    runtime_queue = tmp_path / ".runtime" / "run" / "queue"

    assert (
        _resolve_results_dir(
            original_queue_root=original_queue,
            original_results_dir=original_results,
            queue_root=runtime_queue,
        )
        == runtime_queue / "results"
    )


def test_local_queue_worker_preserves_external_results_dir(tmp_path: Path) -> None:
    original_queue = tmp_path / "queue"
    external_results = tmp_path / "ansys_export"
    runtime_queue = tmp_path / ".runtime" / "run" / "queue"

    assert (
        _resolve_results_dir(
            original_queue_root=original_queue,
            original_results_dir=external_results,
            queue_root=runtime_queue,
        )
        == external_results
    )


def test_local_queue_worker_default_queue_root_is_runtime_queue(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(script, "DEFAULT_QUEUE_ROOT", tmp_path / ".runtime" / "queue")

    args = script.parse_args([])

    assert args.queue_root == tmp_path / ".runtime" / "queue"
