from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from common.io_utils import read_json, write_json
from scripts.worker import validate_spiral_t3_queue_cases as script
from worker.schemas import WorkerConfig


def _worker_config(tmp_path: Path, queue_root: Path) -> WorkerConfig:
    return WorkerConfig(
        worker_name="hfss_win01_local",
        queue_root=queue_root,
        templates_yaml=tmp_path / "templates.yaml",
        poll_interval_sec=1,
        claim_retry_sec=1,
        results_dir=queue_root / "results",
        logs_dir=tmp_path / "logs",
        default_export_touchstone=True,
        default_export_csv=False,
    )


def test_fixed_batch_payload_uses_new_spiral_template() -> None:
    payload = script.build_fixed_batch_payload()

    assert payload["template_name"] == "spiral_inductor_t3_template_v1"
    assert payload["design_name"] == "SpiralInductorT3"
    assert payload["setup_name"] == "Setup1"
    assert payload["sweep_name"] == "Sweep"
    assert payload["metrics_to_extract"] == ["max_s11_db", "min_s21_db"]
    assert len(payload["cases"]) == 5
    assert payload["cases"][0]["variables"] == {
        "Din": "60um",
        "W": "15um",
        "S": "1.5um",
        "H": "2um",
    }


def test_parse_args_defaults_use_runtime_outside_queue() -> None:
    args = script.parse_args([])

    assert ".runtime" in str(args.queue_root)
    assert ".runtime" in str(args.slot_root)
    assert "local_spiral_t3_parallel_validation" in str(args.queue_root)
    assert "local_spiral_t3_parallel_slots" in str(args.slot_root)


def test_dry_run_prints_batch_without_queue_side_effect(tmp_path: Path, capsys) -> None:
    args = argparse.Namespace(
        worker_config=tmp_path / "worker.yaml",
        queue_root=tmp_path / "queue",
        batch_id="dry",
        template_name="spiral_inductor_t3_template_v1",
        design_name="SpiralInductorT3",
        setup_name="Setup1",
        sweep_name="Sweep",
        dry_run=True,
        submit_only=False,
        parallel_slots=1,
    )

    rc = script.run(args)
    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert out["batch_id"] == "dry"
    assert out["template_name"] == "spiral_inductor_t3_template_v1"
    assert not (tmp_path / "queue").exists()


def test_local_validation_processes_jobs_through_worker_path(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"

    class FakeProcessor:
        def process(self, running_path: Path) -> None:
            payload = read_json(running_path)
            job_id = payload["job_id"]
            running_path.unlink()
            write_json(
                queue_root / "done" / f"{job_id}.result.json",
                {
                    "job_id": job_id,
                    "status": "done",
                    "metrics": {"max_s11_db": -1.0, "min_s21_db": -20.0},
                    "artifacts": {
                        "touchstone": f"results/{payload['output_filename']}",
                        "csv": f"results/{Path(payload['output_filename']).with_suffix('.csv').name}",
                        "log": f"results/{job_id}.log",
                    },
                    "error_message": None,
                },
            )

    monkeypatch.setattr(
        script,
        "_load_processor",
        lambda _worker_config, queue_root_override: (
            SimpleNamespace(queue_root=queue_root_override),
            FakeProcessor(),
        ),
    )
    args = argparse.Namespace(
        worker_config=tmp_path / "worker.yaml",
        queue_root=queue_root,
        batch_id="spiral_t3_test",
        template_name="spiral_inductor_t3_template_v1",
        design_name="SpiralInductorT3",
        setup_name="Setup1",
        sweep_name="Sweep",
        dry_run=False,
        submit_only=False,
        parallel_slots=1,
    )

    rc = script.run(args)

    assert rc == 0
    summary = read_json(queue_root / "results" / "spiral_t3_test_summary.json")
    assert len(summary["jobs"]) == 5
    assert all(row["state"] == "done" for row in summary["jobs"])
    assert summary["jobs"][0]["touchstone"] == "results/spiral_t3_nominal.s2p"
    assert summary["parallel_slots"] == 1


def test_build_local_slot_worker_config_payload_isolates_slot(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    cfg = _worker_config(tmp_path, queue_root)

    payload = script.build_local_slot_worker_config_payload(
        cfg,
        slot_index=2,
        slot_root=tmp_path / "slots",
    )

    assert payload["worker_name"] == "hfss_win01_local_local_slot_02"
    assert payload["queue_root"] == str(queue_root)
    assert payload["results_dir"] == str(queue_root / "results")
    assert payload["logs_dir"].endswith("slot_02\\logs") or payload["logs_dir"].endswith("slot_02/logs")
    assert payload["new_desktop"] is True
    assert payload["close_desktop_after_job"] is True
    assert payload["copy_template_per_job"] is True
    assert payload["template_copy_root"].endswith("slot_02\\aedt_copies") or payload["template_copy_root"].endswith("slot_02/aedt_copies")


def test_process_pending_jobs_in_slot_claims_each_job_once(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    cfg = SimpleNamespace(queue_root=queue_root, worker_name="slot_a")
    processed: list[str] = []
    for idx in range(3):
        write_json(queue_root / "pending" / f"job_{idx}.json", {"job_id": f"job_{idx}"})

    class FakeProcessor:
        def process(self, running_path: Path) -> None:
            payload = read_json(running_path)
            job_id = payload["job_id"]
            processed.append(job_id)
            running_path.unlink()
            write_json(
                queue_root / "done" / f"{job_id}.result.json",
                {
                    "job_id": job_id,
                    "status": "done",
                    "worker_name": "slot_a",
                    "artifacts": {"touchstone": f"results/{job_id}.s2p"},
                    "error_message": None,
                },
            )

    first = script.process_pending_jobs_in_slot(cfg, FakeProcessor())
    second = script.process_pending_jobs_in_slot(cfg, FakeProcessor())

    assert sorted(processed) == ["job_0", "job_1", "job_2"]
    assert len(first) == 3
    assert second == []
    assert not list((queue_root / "pending").glob("*.json"))
    assert not list((queue_root / "running").glob("*.json"))


def test_claim_one_pending_respects_existing_claim_lock(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    pending = queue_root / "pending" / "job_001.json"
    write_json(pending, {"job_id": "job_001"})
    (queue_root / "pending" / "job_001.json.claim").write_text("locked", encoding="utf-8")

    assert script._claim_one_pending(queue_root) is None
    assert pending.exists()

    (queue_root / "pending" / "job_001.json.claim").unlink()
    claimed = script._claim_one_pending(queue_root)

    assert claimed == queue_root / "running" / "job_001.json"
    assert claimed.exists()
    assert not pending.exists()
    assert not (queue_root / "pending" / "job_001.json.claim").exists()


def test_parallel_validation_launches_local_slots_and_writes_summary(monkeypatch, tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    cfg = _worker_config(tmp_path, queue_root)
    launched: list[list[str]] = []

    monkeypatch.setattr(
        script,
        "_load_processor",
        lambda _worker_config, queue_root_override: (
            cfg.model_copy(
                update={
                    "queue_root": queue_root_override,
                    "results_dir": queue_root_override / "results",
                }
            ),
            object(),
        ),
    )

    class FakeProcess:
        def __init__(self, command, **_kwargs) -> None:
            launched.append(command)

        def wait(self, timeout=None):
            while True:
                running_path = script._claim_one_pending(queue_root)
                if running_path is None:
                    break
                payload = read_json(running_path)
                job_id = payload["job_id"]
                running_path.unlink()
                write_json(
                    queue_root / "done" / f"{job_id}.result.json",
                    {
                        "job_id": job_id,
                        "status": "done",
                        "worker_name": "fake_slot",
                        "artifacts": {"touchstone": f"results/{payload['output_filename']}"},
                        "error_message": None,
                    },
                )
            return 0

        def poll(self):
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    monkeypatch.setattr(script.subprocess, "Popen", FakeProcess)
    args = argparse.Namespace(
        worker_config=tmp_path / "worker.yaml",
        queue_root=queue_root,
        batch_id="spiral_t3_parallel_test",
        template_name="spiral_inductor_t3_template_v1",
        design_name="SpiralInductorT3",
        setup_name="Setup1",
        sweep_name="Sweep",
        dry_run=False,
        submit_only=False,
        parallel_slots=2,
        slot_root=tmp_path / "slots",
    )

    rc = script.run(args)

    assert rc == 0
    assert len(launched) == 2
    assert all("--local-slot-runner" in command for command in launched)
    assert all("--queue-root" in command for command in launched)
    assert all(
        command[command.index("--queue-root") + 1] == str(queue_root)
        for command in launched
    )
    summary = read_json(queue_root / "results" / "spiral_t3_parallel_test_summary.json")
    assert summary["parallel_slots"] == 2
    assert len(summary["slot_configs"]) == 2
    assert len(summary["jobs"]) == 5
    assert all(row["state"] == "done" for row in summary["jobs"])
