from pathlib import Path

from worker.artifact_exporter import build_artifact_paths_with_name
from worker.schemas import WorkerConfig


def _worker_cfg(tmp_path: Path) -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "worker_name": "hfss_win01",
            "queue_root": str(tmp_path / "queue"),
            "templates_yaml": str(tmp_path / "templates.yaml"),
            "poll_interval_sec": 1,
            "claim_retry_sec": 1,
            "results_dir": str(tmp_path / "queue" / "results"),
            "logs_dir": str(tmp_path / "logs"),
            "job_timeout_sec": 60,
        }
    )


def test_build_artifact_paths_uses_output_filename_basename(tmp_path: Path) -> None:
    cfg = _worker_cfg(tmp_path)
    paths = build_artifact_paths_with_name(
        job_id="batch_01_c001",
        cfg=cfg,
        output_filename="spiral_Din55_S1d5_W15.s2p",
    )
    assert paths.touchstone.name == "spiral_Din55_S1d5_W15.s2p"
    assert paths.csv.name == "spiral_Din55_S1d5_W15.csv"
    assert paths.log.name == "batch_01_c001.log"


def test_build_artifact_paths_falls_back_to_job_id_when_name_missing(tmp_path: Path) -> None:
    cfg = _worker_cfg(tmp_path)
    paths = build_artifact_paths_with_name(
        job_id="batch_01_c002",
        cfg=cfg,
        output_filename=None,
    )
    assert paths.touchstone.name == "batch_01_c002.s2p"
    assert paths.csv.name == "batch_01_c002.csv"
