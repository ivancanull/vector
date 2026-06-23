from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pandas as pd
import pytest


def _load_plot_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "studies" / "online_comparison" / "scripts" / "plot_owafs_vs_bo_lhs_bnn_objective.py"
    spec = importlib.util.spec_from_file_location("plot_owafs_vs_bo_lhs_bnn_objective", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_comparison_fixture(output_dir: Path, *, forbidden_bnn: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_rows = []
    per_run_rows = []
    methods = ("OWAFS", "BNN", "LHS", "BO")
    for method_idx, method in enumerate(methods):
        for trial in range(1, 81):
            baseline = 10.0 * method_idx - 0.5 * trial
            curve_rows.append(
                {
                    "method": method,
                    "iteration": trial - 1,
                    "trial": trial,
                    "n": 2,
                    "median": baseline,
                    "mean": baseline,
                    "q25": baseline - 1.0,
                    "q75": baseline + 1.0,
                    "min": baseline - 2.0,
                    "max": baseline + 2.0,
                }
            )
        for seed in (0, 1):
            if method == "BNN" and forbidden_bnn:
                source = f"/tmp/experiments/online_lhs15_shared_init/bnn_lhs15_shared_seed{seed}.csv"
            elif method == "BNN":
                source = f"/tmp/experiments/bnn_online_general/bnn_online_general_seed{seed}.csv"
            else:
                source = f"/tmp/experiments/{method.lower()}_online_general/{method.lower()}_seed{seed}.csv"
            per_run_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "source_csv": source,
                    "budgeted_rows": 80,
                    "final_best_objective": float(10.0 * method_idx - seed),
                }
            )

    pd.DataFrame(curve_rows).to_csv(output_dir / "owafs_vs_bo_lhs_bnn_best_so_far_by_iteration.csv", index=False)
    pd.DataFrame(per_run_rows).to_csv(output_dir / "owafs_vs_bo_lhs_bnn_per_run.csv", index=False)


def test_objective_plotter_generates_expected_outputs(tmp_path: Path) -> None:
    module = _load_plot_module()
    data_dir = tmp_path / "comparison"
    output_dir = tmp_path / "figures"
    _write_comparison_fixture(data_dir)

    plotter = module.OWAFSObjectivePlotter(repo_root=tmp_path, data_dir=data_dir, output_dir=output_dir)
    outputs = plotter.run()

    assert {path.name for path in outputs} == {
        "owafs_vs_bo_lhs_bnn_best_so_far_objective.pdf",
        "owafs_vs_bo_lhs_bnn_best_so_far_objective.png",
        "owafs_vs_bo_lhs_bnn_final_best_objective.pdf",
        "owafs_vs_bo_lhs_bnn_final_best_objective.png",
    }
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0


def test_objective_plotter_rejects_previous_lhs_bnn_source(tmp_path: Path) -> None:
    module = _load_plot_module()
    data_dir = tmp_path / "comparison"
    output_dir = tmp_path / "figures"
    _write_comparison_fixture(data_dir, forbidden_bnn=True)

    plotter = module.OWAFSObjectivePlotter(repo_root=tmp_path, data_dir=data_dir, output_dir=output_dir)
    with pytest.raises(RuntimeError, match="Forbidden previous-LHS BNN source"):
        plotter.run()
