from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pandas as pd
import pytest


def _load_report_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "studies" / "online_owafs" / "scripts" / "summarize_owafs_experiment_status.py"
    spec = importlib.util.spec_from_file_location("summarize_owafs_experiment_status", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fixture_csvs(root: Path, *, forbidden_bnn: bool = False) -> None:
    owafs_dir = root / "experiments" / "online_lhsdb_warmup" / "owafs_bnn"
    diagnostics_dir = owafs_dir / "diagnostics"
    comparison_dir = owafs_dir / "comparison"
    bnn_source = (
        "/tmp/experiments/online_lhs15_shared_init/bnn_lhs15_shared_seed0.csv"
        if forbidden_bnn
        else "/tmp/experiments/bnn_online_general/bnn_online_general_seed0.csv"
    )
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "seed": 0,
                "status": "completed",
                "num_budgeted_trials": 80,
                "num_validations": 1,
                "best_objective": -22.0,
            },
            {
                "seed": 1,
                "status": "completed",
                "num_budgeted_trials": 80,
                "num_validations": 1,
                "best_objective": -17.0,
            },
            {
                "seed": 2,
                "status": "completed",
                "num_budgeted_trials": 80,
                "num_validations": 1,
                "best_objective": -6.0,
            },
            {
                "seed": 3,
                "status": "completed",
                "num_budgeted_trials": 80,
                "num_validations": 1,
                "best_objective": -23.8,
            },
            {
                "seed": 4,
                "status": "completed",
                "num_budgeted_trials": 80,
                "num_validations": 2,
                "best_objective": -16.8,
            },
        ]
    ).to_csv(owafs_dir / "campaign_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "seed": 0,
                "warmup_best_objective": -16.0,
                "owafs_best_objective": -22.0,
                "owafs_best_iteration": 44,
                "online_median_distance_to_lhs": 0.68,
                "online_max_distance_to_lhs": 1.04,
            },
            {
                "seed": 1,
                "warmup_best_objective": -11.0,
                "owafs_best_objective": -17.0,
                "owafs_best_iteration": 32,
                "online_median_distance_to_lhs": 0.66,
                "online_max_distance_to_lhs": 0.97,
            },
            {
                "seed": 2,
                "warmup_best_objective": 0.02,
                "owafs_best_objective": -6.8,
                "owafs_best_iteration": 41,
                "online_median_distance_to_lhs": 0.67,
                "online_max_distance_to_lhs": 1.03,
            },
            {
                "seed": 3,
                "warmup_best_objective": -13.5,
                "owafs_best_objective": -23.8,
                "owafs_best_iteration": 55,
                "online_median_distance_to_lhs": 0.70,
                "online_max_distance_to_lhs": 1.02,
            },
            {
                "seed": 4,
                "warmup_best_objective": -12.0,
                "owafs_best_objective": -16.8,
                "owafs_best_iteration": 66,
                "online_median_distance_to_lhs": 0.65,
                "online_max_distance_to_lhs": 0.95,
            },
        ]
    ).to_csv(diagnostics_dir / "owafs_geometry_coverage_summary_seed0_4.csv", index=False)

    pd.DataFrame(
        [
            {
                "method": "BNN",
                "n_runs": 3,
                "seeds": "0,1,2",
                "median_final_best_objective": -19.16,
                "mean_final_best_objective": -20.50,
                "best_min_objective": -25.0,
                "worst_max_objective": -17.34,
                "median_improvement_15_to_80": 3.53,
            },
            {
                "method": "LHS",
                "n_runs": 5,
                "seeds": "0,1,2,3,4",
                "median_final_best_objective": -19.16,
                "mean_final_best_objective": -18.81,
                "best_min_objective": -25.14,
                "worst_max_objective": -9.85,
                "median_improvement_15_to_80": 17.27,
            },
            {
                "method": "OWAFS",
                "n_runs": 5,
                "seeds": "0,1,2,3,4",
                "median_final_best_objective": -17.41,
                "mean_final_best_objective": -17.62,
                "best_min_objective": -23.8,
                "worst_max_objective": -6.80,
                "median_improvement_15_to_80": 5.95,
            },
            {
                "method": "BO",
                "n_runs": 5,
                "seeds": "0,1,2,3,4",
                "median_final_best_objective": -14.66,
                "mean_final_best_objective": -15.65,
                "best_min_objective": -23.81,
                "worst_max_objective": -8.05,
                "median_improvement_15_to_80": 5.95,
            },
        ]
    ).to_csv(comparison_dir / "owafs_vs_bo_lhs_bnn_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "method": "BNN",
                "seed": 0,
                "source_csv": bnn_source,
                "budgeted_rows": 80,
                "final_best_objective": -25.0,
            },
            {
                "method": "OWAFS",
                "seed": 0,
                "source_csv": "/tmp/experiments/online_lhsdb_warmup/owafs_bnn/seed0.csv",
                "budgeted_rows": 80,
                "final_best_objective": -22.0,
            },
        ]
    ).to_csv(comparison_dir / "owafs_vs_bo_lhs_bnn_per_run.csv", index=False)


def test_status_report_is_generated_from_csv_sources(tmp_path: Path) -> None:
    module = _load_report_module()
    _write_fixture_csvs(tmp_path)
    output = tmp_path / "report.md"

    reporter = module.OWAFSExperimentStatusReporter(repo_root=tmp_path, output_path=output)
    path = reporter.run()

    text = path.read_text(encoding="utf-8")
    assert path == output
    assert "OWAFS 实验进度与后续实验计划" in text
    assert "previous-LHS / `online_lhs15_shared` 那批 BNN" in text
    assert "描述性 pooled evidence" in text
    assert "| 0 | 80 | 1 | -16.00 | -22.00 | 44 | 6.00 | 0.68 | 1.04 |" in text
    assert "BNN online_general seed0-2 median final best 为 `-19.16`" in text
    assert "OWAFS seed0-4 median final best 为 `-17.41`" in text
    assert "## 当前比较方法说明" in text
    assert "| OWAFS-BNN | 当前研究方法" in text
    assert "| BNN online_general | 主要神经 surrogate baseline" in text
    assert "## 后续实验方法设计" in text
    assert "| Exploration ablation | 检查当前算法是否探索扩张不足。" in text
    assert "P0: Same-warmup BNN baseline" in text
    assert "OWAFS seed completion" not in text


def test_status_report_rejects_previous_lhs_bnn_source(tmp_path: Path) -> None:
    module = _load_report_module()
    _write_fixture_csvs(tmp_path, forbidden_bnn=True)

    reporter = module.OWAFSExperimentStatusReporter(repo_root=tmp_path, output_path=tmp_path / "report.md")
    with pytest.raises(RuntimeError, match="Forbidden previous-LHS BNN source"):
        reporter.run()
