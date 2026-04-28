from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from worker.schemas import ArtifactPaths, WorkerConfig
from worker.utils import WorkerError, as_posix_rel


@dataclass(frozen=True)
class LocalArtifactPaths:
    touchstone: Path
    csv: Path
    log: Path


def build_artifact_paths_with_name(
    job_id: str, cfg: WorkerConfig, output_filename: str | None
) -> LocalArtifactPaths:
    results_dir = cfg.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    if output_filename:
        touchstone_name = Path(output_filename).name
        touchstone_path = results_dir / touchstone_name
        stem = Path(touchstone_name).stem
        csv_path = results_dir / f"{stem}.csv"
    else:
        touchstone_path = results_dir / f"{job_id}.s2p"
        csv_path = results_dir / f"{job_id}.csv"

    return LocalArtifactPaths(
        touchstone=touchstone_path,
        csv=csv_path,
        log=results_dir / f"{job_id}.log",
    )


def export_csv_from_touchstone(touchstone_path: Path, csv_path: Path) -> Path:
    try:
        import numpy as np
        import skrf as rf

        network = rf.Network(str(touchstone_path))
    except Exception as exc:
        raise WorkerError("export_failure", f"Failed to parse touchstone for CSV: {exc}") from exc

    frequency_hz = network.f
    s11_db = 20.0 * np.log10(np.abs(network.s[:, 0, 0]))
    if network.number_of_ports >= 2:
        s21_db = 20.0 * np.log10(np.abs(network.s[:, 1, 0]))
    else:
        s21_db = np.full_like(s11_db, np.nan)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("frequency_hz,s11_db,s21_db\n")
        for freq, s11, s21 in zip(frequency_hz, s11_db, s21_db):
            f.write(f"{freq:.12g},{s11:.6f},{s21:.6f}\n")
        f.flush()
    return csv_path


def verify_artifacts_exist(
    paths: LocalArtifactPaths, need_touchstone: bool, need_csv: bool
) -> None:
    if need_touchstone and not paths.touchstone.exists():
        raise WorkerError("export_failure", f"Missing touchstone: {paths.touchstone}")
    if need_csv and not paths.csv.exists():
        raise WorkerError("export_failure", f"Missing csv: {paths.csv}")


def to_result_artifacts(
    paths: LocalArtifactPaths,
    queue_root: Path,
    *,
    include_touchstone: bool,
    include_csv: bool,
    include_log: bool = True,
) -> ArtifactPaths:
    return ArtifactPaths(
        touchstone=as_posix_rel(paths.touchstone, queue_root) if include_touchstone else None,
        csv=as_posix_rel(paths.csv, queue_root) if include_csv else None,
        log=as_posix_rel(paths.log, queue_root) if include_log else None,
    )
