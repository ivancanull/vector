from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from common.io_utils import write_json
from scripts.worker import fetch_run_submit_job_parallel as script


def _base_config() -> dict:
    return {
        "worker_name": "hfss_win01",
        "queue_root": "D:/vector/queue",
        "templates_yaml": "D:/vector/configs/templates.yaml",
        "poll_interval_sec": 2,
        "claim_retry_sec": 1,
        "results_dir": "D:/vector/queue/results",
        "logs_dir": "D:/vector/logs",
        "non_graphical": True,
        "new_desktop": False,
        "close_desktop_after_job": False,
        "job_timeout_sec": 7200,
        "default_export_touchstone": True,
        "default_export_csv": False,
        "preserve_local_project_copy": False,
    }


def test_build_slot_config_payload_isolates_each_slot(tmp_path: Path) -> None:
    payload = script.build_slot_config_payload(
        _base_config(),
        slot_index=2,
        slot_root=tmp_path / "slots",
    )

    assert payload["worker_name"] == "hfss_win01_slot_02"
    assert payload["queue_root"].endswith("slot_02\\queue") or payload["queue_root"].endswith("slot_02/queue")
    assert payload["results_dir"].endswith("slot_02\\queue\\results") or payload["results_dir"].endswith("slot_02/queue/results")
    assert payload["logs_dir"].endswith("slot_02\\logs") or payload["logs_dir"].endswith("slot_02/logs")
    assert payload["new_desktop"] is True
    assert payload["close_desktop_after_job"] is True
    assert payload["copy_template_per_job"] is True
    assert payload["template_copy_root"].endswith("slot_02\\aedt_copies") or payload["template_copy_root"].endswith("slot_02/aedt_copies")


def test_default_slot_root_uses_runtime_outside_queue() -> None:
    assert ".runtime" in str(script.DEFAULT_SLOT_ROOT)
    assert "parallel_worker_slots" in str(script.DEFAULT_SLOT_ROOT)


def test_write_slot_worker_configs_emits_unique_configs(tmp_path: Path) -> None:
    base_path = tmp_path / "worker.yaml"
    with base_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_base_config(), f)

    slots = script.write_slot_worker_configs(
        base_worker_config=base_path,
        slot_root=tmp_path / "slots",
        slots=2,
    )

    assert [slot.name for slot in slots] == ["slot_01", "slot_02"]
    first = yaml.safe_load(slots[0].worker_config.read_text(encoding="utf-8"))
    second = yaml.safe_load(slots[1].worker_config.read_text(encoding="utf-8"))
    assert first["worker_name"] != second["worker_name"]
    assert first["queue_root"] != second["queue_root"]
    assert first["template_copy_root"] != second["template_copy_root"]
    assert first["copy_template_per_job"] is True
    assert second["copy_template_per_job"] is True


def test_run_launches_existing_worker_once_per_slot(monkeypatch, tmp_path: Path) -> None:
    base_path = tmp_path / "worker.yaml"
    write_json(base_path, _base_config())
    launched: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command, **_kwargs) -> None:
            launched.append(command)

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    monkeypatch.setattr(script.subprocess, "Popen", FakeProcess)
    args = argparse.Namespace(
        worker_config=base_path,
        slots=2,
        slot_root=tmp_path / "slots",
        server_host="10.15.89.240",
        server_port=22031,
        username="zhanghf",
        ssh_key=tmp_path / "id",
        known_hosts=tmp_path / "known_hosts",
        remote_queue_root="/remote/queue",
        sftp_connect_timeout_sec=12,
        sftp_connect_attempts=7,
        sftp_connect_retry_delay_sec=1.5,
        wait_when_empty=False,
        poll_interval_sec=3,
        idle_status_interval_sec=11,
        once=True,
    )

    rc = script.run(args)

    assert rc == 0
    assert len(launched) == 2
    assert all("fetch_run_submit_job.py" in command[1] for command in launched)
    assert all("--once" in command for command in launched)
    assert all("--no-wait-when-empty" in command for command in launched)
    for command in launched:
        assert command[command.index("--sftp-connect-timeout-sec") + 1] == "12"
        assert command[command.index("--sftp-connect-attempts") + 1] == "7"
        assert command[command.index("--sftp-connect-retry-delay-sec") + 1] == "1.5"
    assert launched[0][launched[0].index("--worker-config") + 1] != launched[1][launched[1].index("--worker-config") + 1]
