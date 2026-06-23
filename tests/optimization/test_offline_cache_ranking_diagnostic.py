from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("studies/bnn_acquisition/scripts/diagnose_offline_cache_ranking.py")
VARIABLE_ORDER = ["L", "L1", "L2", "W", "W1", "W2", "S1", "S2"]


def _write_config(path: Path) -> None:
    path.write_text(
        """
data:
  regularization:
    min: [7.7, 8.0, 7.8, 0.9, 0.7, 0.9, 0.13, 0.55, 3.5e+9]
    max: [7.9, 8.2, 8.0, 1.1, 0.9, 1.1, 0.16, 0.6, 8.7e+9]
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_s2p(path: Path, scale: float) -> None:
    rows = []
    for freq in [3.5, 3.6, 4.8, 5.4, 8.7]:
        s11 = 0.2 * scale
        s21 = 0.05 * scale
        rows.append(
            f"{freq:.3f} {s11:.6f} 0 {s21:.6f} 0 {s21:.6f} 0 {s11:.6f} 0\n"
        )
    path.write_text("# GHz S MA R 50\n" + "".join(rows), encoding="utf-8")


def _make_cache(tmp_path: Path) -> tuple[Path, list[list[float]]]:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    geometries = [
        [7.80, 8.10, 7.90, 1.00, 0.80, 1.00, 0.15, 0.57],
        [7.82, 8.12, 7.92, 1.02, 0.82, 1.02, 0.15, 0.58],
    ]
    records = {}
    for idx, geometry in enumerate(geometries):
        filename = f"case_{idx}.s2p"
        _write_s2p(cache_dir / filename, scale=1.0 + idx)
        variables = {name: geometry[pos] for pos, name in enumerate(VARIABLE_ORDER)}
        key = "|".join(f"{name}={variables[name]:.2f}".rstrip("0").rstrip(".") for name in VARIABLE_ORDER)
        records[key] = {"case_id": f"case_{idx}", "variables": variables, "filename": filename}
    (cache_dir / "index.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    return cache_dir, geometries


def _write_checkpoint(path: Path, geometry: list[float]) -> None:
    payload = {
        "history": [
            {
                "iteration": 0,
                "phase": "init",
                "geometry": geometry,
                "objective": 1.0,
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_offline_cache_ranking_dry_run_rehydrates_tiny_cache(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    cache_dir, geometries = _make_cache(tmp_path)
    checkpoint = tmp_path / "checkpoint.json"
    _write_checkpoint(checkpoint, geometries[0])

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--config-path",
            str(config),
            "--cache-dir",
            str(cache_dir),
            "--history-checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "out"),
            "--report-dir",
            str(tmp_path / "reports"),
            "--figure-dir",
            str(tmp_path / "figures"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "train_rows: 1" in result.stdout
    assert "heldout_candidates: 1" in result.stdout
    assert "dry_run: true" in result.stdout


def test_offline_cache_ranking_fails_when_training_geometry_missing(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    cache_dir, geometries = _make_cache(tmp_path)
    checkpoint = tmp_path / "checkpoint.json"
    missing = list(geometries[0])
    missing[0] = 7.89
    _write_checkpoint(checkpoint, missing)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--config-path",
            str(config),
            "--cache-dir",
            str(cache_dir),
            "--history-checkpoint",
            str(checkpoint),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing from cache index" in result.stderr
