from __future__ import annotations

from pathlib import Path

import numpy as np

from common.io_utils import read_json
from scripts.worker import run_inductor_local_candidate_space as script


def _base_worker_config(path: Path, queue_root: Path) -> Path:
    worker_config = path / "worker.yaml"
    worker_config.write_text(
        "\n".join(
            [
                "worker_name: hfss_win01_local",
                f"queue_root: {queue_root.as_posix()}",
                f"templates_yaml: {(path / 'templates.yaml').as_posix()}",
                "poll_interval_sec: 2",
                "claim_retry_sec: 1",
                f"results_dir: {(queue_root / 'results').as_posix()}",
                f"logs_dir: {(path / 'logs').as_posix()}",
                "non_graphical: true",
                "new_desktop: false",
                "close_desktop_after_job: false",
                "restart_aedt_every_n_jobs: 10",
                "job_timeout_sec: 7200",
                "max_retry_count: 2",
                "default_export_touchstone: true",
                "default_export_csv: false",
                "copy_template_per_job: false",
                "template_copy_root: null",
                "preserve_local_project_copy: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return worker_config


def _um_to_float(value: str) -> float:
    assert value.endswith("um")
    return float(value[:-2])


def _assert_on_grid(value: float, step: float) -> None:
    assert abs(value / step - round(value / step)) < 1e-8


def test_inductor_candidate_lhs_is_deterministic_and_in_range() -> None:
    bounds = script.InductorDesignBounds()
    first = script.generate_lhs_geometries(count=20, seed=0, bounds=bounds)
    second = script.generate_lhs_geometries(count=20, seed=0, bounds=bounds)

    np.testing.assert_allclose(first, second)
    assert first.shape == (20, 4)
    assert np.all((first[:, 0] >= 50.0) & (first[:, 0] <= 70.0))
    assert np.all((first[:, 1] >= 12.0) & (first[:, 1] <= 18.0))
    assert np.all((first[:, 2] >= 1.0) & (first[:, 2] <= 2.0))
    assert np.all(first[:, 3] == 2.0)
    assert np.all(first[:, 0] > 2.0 * (first[:, 1] + first[:, 2]))
    for col in range(3):
        np.testing.assert_allclose(first[:, col] / 0.1, np.round(first[:, col] / 0.1), atol=1e-8)


def test_inductor_candidate_script_writes_local_queue_and_summary(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    export_dir = tmp_path / "export"
    worker_config = _base_worker_config(tmp_path, tmp_path / "queue")

    rc = script.main(
        [
            "--count",
            "3",
            "--seed",
            "0",
            "--run-id",
            "inductor_test",
            "--runtime-root",
            str(runtime_root),
            "--worker-config",
            str(worker_config),
            "--export-dir",
            str(export_dir),
        ]
    )

    assert rc == 0
    pending = sorted((runtime_root / "queue" / "pending").glob("*.json"))
    assert len(pending) == 3
    first = read_json(pending[0])
    assert first["template_name"] == "spiral_inductor_t3_template_v1"
    assert set(first["variables"]) == {"Din", "W", "S", "H"}
    assert all(str(value).endswith("um") for value in first["variables"].values())
    _assert_on_grid(_um_to_float(first["variables"]["Din"]), 0.1)
    _assert_on_grid(_um_to_float(first["variables"]["W"]), 0.1)
    _assert_on_grid(_um_to_float(first["variables"]["S"]), 0.1)
    assert first["output_filename"].startswith("spiral_inductor_example_HFSSDesign000_Din_")
    assert first["output_filename"].endswith(".s2p")

    generated_cfg = (runtime_root / "worker_configs" / "inductor_test.yaml").read_text(encoding="utf-8")
    assert f"results_dir: {export_dir}" in generated_cfg
    assert "copy_template_per_job: true" in generated_cfg

    summary = read_json(runtime_root / "queue" / "results" / "inductor_test_inductor_candidate_summary.json")
    assert summary["status_counts"] == {"pending_or_missing": 3}
    assert summary["export_dir"] == str(export_dir)
    assert summary["resolution_um"] == {"Din": 0.1, "H": None, "S": 0.1, "W": 0.1}
    assert summary["rows"][0]["resolution_um"] == "Din=0.1um;W=0.1um;S=0.1um;H=fixed"
    assert (runtime_root / "queue" / "results" / "inductor_test_inductor_candidate_summary.csv").exists()

    jobs_manifest = read_json(runtime_root / "queue" / "results" / "inductor_test_inductor_candidate_jobs.json")
    assert jobs_manifest["resolution_um"] == {"Din": 0.1, "H": None, "S": 0.1, "W": 0.1}


def test_inductor_candidate_script_defaults_to_runtime_queue(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(script, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(script, "DEFAULT_RUNTIME_ROOT", tmp_path / ".runtime")
    monkeypatch.setattr(script, "DEFAULT_QUEUE_ROOT", tmp_path / ".runtime" / "queue")
    worker_config = _base_worker_config(tmp_path, tmp_path / "queue")

    args = script.parse_args(["--run-id", "default_queue", "--worker-config", str(worker_config)])
    paths = script.build_paths(args)

    assert paths["runtime_root"] == tmp_path / ".runtime"
    assert paths["queue_root"] == tmp_path / ".runtime" / "queue"
    assert paths["logs_dir"] == tmp_path / ".runtime" / "logs" / "default_queue"
    assert paths["worker_config"] == tmp_path / ".runtime" / "worker_configs" / "default_queue.yaml"
    assert paths["summary_csv"] == tmp_path / ".runtime" / "queue" / "results" / "default_queue_inductor_candidate_summary.csv"


def test_inductor_candidate_script_default_export_dir_is_ext() -> None:
    args = script.parse_args([])

    assert args.export_dir == Path("D:/ansys/spiral_inductor_s_parameters_export/ext")


def test_inductor_candidate_script_default_resolution() -> None:
    args = script.parse_args([])

    assert script.build_resolution(args) == {"Din": 0.1, "W": 0.1, "S": 0.1, "H": None}
    assert (
        script.build_output_filename(0, {"Din": 62.5, "S": 1.0, "W": 13.5, "H": 2.0})
        == "spiral_inductor_example_HFSSDesign000_Din_62d5_S_1_W_13d5.s2p"
    )


def test_inductor_candidate_script_run_worker_uses_single_thread_local_runner(monkeypatch, tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    export_dir = tmp_path / "export"
    worker_config = _base_worker_config(tmp_path, tmp_path / "queue")
    commands: list[list[str]] = []

    class Completed:
        stdout = "processed_count=0\n"
        stderr = ""
        returncode = 0

    def fake_run(command, **kwargs):
        commands.append(list(command))
        return Completed()

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    rc = script.main(
        [
            "--count",
            "2",
            "--run-id",
            "inductor_test",
            "--runtime-root",
            str(runtime_root),
            "--worker-config",
            str(worker_config),
            "--export-dir",
            str(export_dir),
            "--run-worker",
        ]
    )

    assert rc == 0
    assert len(commands) == 1
    command = commands[0]
    assert "run_local_queue_worker.py" in command[1]
    assert "--max-jobs" in command
    assert command[command.index("--max-jobs") + 1] == "2"
    assert "--no-wait" in command
    assert "--parallel-slots" not in command


def test_inductor_candidate_script_dry_run_has_no_queue_side_effect(tmp_path: Path, capsys) -> None:
    runtime_root = tmp_path / "runtime"
    worker_config = _base_worker_config(tmp_path, tmp_path / "queue")

    rc = script.main(
        [
            "--count",
            "2",
            "--run-id",
            "dry",
            "--runtime-root",
            str(runtime_root),
            "--worker-config",
            str(worker_config),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "spiral_inductor_example_HFSSDesign000" in out
    assert not runtime_root.exists()


def test_assign_target_rank_prefers_target_l_then_high_q() -> None:
    rows = [
        {"status": "done", "target_L_error_nH": 0.1, "Q_5GHz": 10.0},
        {"status": "done", "target_L_error_nH": 0.1, "Q_5GHz": 20.0},
        {"status": "done", "target_L_error_nH": 0.2, "Q_5GHz": 50.0},
        {"status": "failed", "target_L_error_nH": 0.0, "Q_5GHz": 100.0},
    ]

    script._assign_target_rank(rows)

    assert rows[1]["target_rank"] == 1
    assert rows[0]["target_rank"] == 2
    assert rows[2]["target_rank"] == 3
    assert rows[3].get("target_rank") is None


def test_inductor_candidate_script_does_not_call_hfss_directly() -> None:
    source = Path("scripts/worker/run_inductor_local_candidate_space.py").read_text(encoding="utf-8")

    assert "DiscreteCandidateSpace" in source
    assert "run_local_queue_worker.py" in source
    assert "Hfss(" not in source
    assert "HfssJobExecutor" not in source
    assert "ansys.aedt" not in source
