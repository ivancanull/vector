from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


def _load_plot_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "studies" / "online_owafs" / "scripts" / "plot_owafs_geometry_coverage.py"
    spec = importlib.util.spec_from_file_location("plot_owafs_geometry_coverage", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fixture_csvs(
    root: Path,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    warmup_count: int = 2,
    budgeted_count: int = 4,
    incomplete_seed: int | None = None,
) -> None:
    initial_dir = root / "experiments" / "online_lhsdb_warmup" / "initial_designs"
    owafs_dir = root / "experiments" / "online_lhsdb_warmup" / "owafs_bnn"
    initial_dir.mkdir(parents=True)
    owafs_dir.mkdir(parents=True)

    warmup_rows = []
    for seed in seeds:
        for idx in range(warmup_count):
            geometry = np.full(8, 0.05 * seed + 0.1 * idx, dtype=float)
            warmup_rows.append(
                {
                    "target_seed": seed,
                    "initial_design_mode": "lhs_database_random",
                    "init_idx": idx,
                    "source_objective": float(10 * seed - idx),
                    "source_best_so_far": float(10 * seed - idx),
                    "source_lhs_seed": seed,
                    "source_lhs_iteration": idx,
                    "source_case_id": f"lhs_seed{seed}_it{idx:04d}",
                    "nearest_distance_normalized": 0.1,
                    "cache_required": True,
                    "cache_path": "",
                    "geometry": str(geometry.tolist()),
                }
            )
    pd.DataFrame(warmup_rows).to_csv(initial_dir / "warmup_initial_points.csv", index=False)

    for seed in seeds:
        rows = []
        rows_to_write = budgeted_count - 1 if incomplete_seed == seed else budgeted_count
        for iteration in range(rows_to_write):
            geometry = np.full(8, 0.05 * seed + 0.1 * iteration, dtype=float)
            rows.append(
                {
                    "iteration": iteration,
                    "phase": "init" if iteration < warmup_count else "bo",
                    "geometry": str(geometry.tolist()),
                    "geometry_normalized": str(geometry.tolist()),
                    "case_id": f"owafs_seed{seed}_it{iteration:04d}",
                    "objective": float(5 * seed - iteration),
                    "best_so_far": float(5 * seed - iteration),
                    "counts_toward_budget": "True",
                }
            )
        if incomplete_seed != seed:
            geometry = np.full(8, 0.05 * seed + 0.2, dtype=float)
            rows.append(
                {
                    "iteration": warmup_count,
                    "phase": "validation",
                    "geometry": str(geometry.tolist()),
                    "geometry_normalized": str(geometry.tolist()),
                    "case_id": f"owafs_seed{seed}_validation",
                    "objective": float(5 * seed - 100),
                    "best_so_far": float(5 * seed - 100),
                    "counts_toward_budget": "False",
                }
            )
        pd.DataFrame(rows).to_csv(owafs_dir / f"owafs_bnn_online_lhsdb_warmup_seed{seed}.csv", index=False)


def test_geometry_helpers_parse_project_and_measure_distance() -> None:
    module = _load_plot_module()

    vector = module.parse_vector("[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]")
    assert vector.shape == (8,)

    normalized = module.normalize_geometry(np.ones(8), np.zeros(8), np.full(8, 2.0))
    np.testing.assert_allclose(normalized, np.full(8, 0.5))

    pca = module.fit_pca(np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=float))
    projected = pca.transform(np.asarray([[0.0, 0.0]], dtype=float))
    assert projected.shape == (1, 2)

    distances = module.nearest_distances(
        np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=float),
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float),
    )
    np.testing.assert_allclose(distances, [0.0, 1.0])

    mask = module.parse_budget_mask(pd.Series(["True", "False", "1", "0", True, False, np.nan]))
    assert mask.tolist() == [True, False, True, False, True, False, True]


def test_plotter_generates_expected_outputs(tmp_path: Path) -> None:
    module = _load_plot_module()
    _write_fixture_csvs(tmp_path)
    figure_dir = tmp_path / "figures"
    summary_dir = tmp_path / "diagnostics"

    plotter = module.OWAFSGeometryCoveragePlotter(
        repo_root=tmp_path,
        expected_budgeted_trials=4,
        warmup_count=2,
        geometry_min=np.zeros(8),
        geometry_max=np.ones(8),
        figure_dir=figure_dir,
        summary_dir=summary_dir,
    )
    outputs = plotter.run()

    output_names = {path.name for path in outputs}
    assert output_names == {
        "owafs_geometry_coverage_pca_seed0_4.pdf",
        "owafs_geometry_coverage_pca_seed0_4.png",
        "owafs_geometry_distance_to_lhs_seed0_4.pdf",
        "owafs_geometry_distance_to_lhs_seed0_4.png",
        "owafs_geometry_coverage_summary_seed0_4.csv",
    }
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0

    summary = pd.read_csv(summary_dir / "owafs_geometry_coverage_summary_seed0_4.csv")
    assert summary["seed"].tolist() == [0, 1, 2, 3, 4]
    assert summary["budgeted_trials"].tolist() == [4, 4, 4, 4, 4]


def test_plotter_rejects_incomplete_seed_without_outputs(tmp_path: Path) -> None:
    module = _load_plot_module()
    _write_fixture_csvs(tmp_path, incomplete_seed=2)
    figure_dir = tmp_path / "figures"
    summary_dir = tmp_path / "diagnostics"

    plotter = module.OWAFSGeometryCoveragePlotter(
        repo_root=tmp_path,
        expected_budgeted_trials=4,
        warmup_count=2,
        geometry_min=np.zeros(8),
        geometry_max=np.ones(8),
        figure_dir=figure_dir,
        summary_dir=summary_dir,
    )

    with pytest.raises(RuntimeError, match="Seed 2 is not complete"):
        plotter.run()
    assert not figure_dir.exists()
    assert not summary_dir.exists()
