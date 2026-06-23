from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pandas as pd
import pytest


def _load_compare_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "studies" / "online_comparison" / "scripts" / "compare_owafs_vs_bo_lhs_bnn.py"
    spec = importlib.util.spec_from_file_location("compare_owafs_vs_bo_lhs_bnn", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_run_csv(path: Path, *, seed: int, rows: int = 80, include_validation: bool = False, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    best = float("inf")
    for iteration in range(rows):
        objective = float(offset + seed * 10 - iteration)
        best = min(best, objective)
        records.append(
            {
                "iteration": iteration,
                "phase": "init" if iteration == 0 else "bo",
                "case_id": f"{path.stem}_it{iteration:04d}",
                "objective": objective,
                "best_so_far": best,
                "cache_hit": "True",
                "t_engine_sec": 1.0,
                "t_simulator_sec": 2.0,
                "counts_toward_budget": "True",
            }
        )
    if include_validation:
        records.append(
            {
                "iteration": rows - 1,
                "phase": "validation",
                "case_id": f"{path.stem}_validation",
                "objective": -1000.0,
                "best_so_far": -1000.0,
                "cache_hit": "False",
                "t_engine_sec": 1.0,
                "t_simulator_sec": 2.0,
                "counts_toward_budget": "False",
            }
        )
    pd.DataFrame(records).to_csv(path, index=False)


def _write_fixture_csvs(root: Path, *, incomplete_bnn_seed: int | None = None) -> None:
    for seed in (0, 1, 2, 3, 4):
        _write_run_csv(
            root
            / "experiments"
            / "online_lhsdb_warmup"
            / "owafs_bnn"
            / f"owafs_bnn_online_lhsdb_warmup_seed{seed}.csv",
            seed=seed,
            include_validation=True,
            offset=-20.0,
        )

    for seed in (0, 1, 2):
        bnn_rows = 79 if seed == incomplete_bnn_seed else 80
        _write_run_csv(
            root / "experiments" / "bnn_online_general" / f"bnn_online_general_seed{seed}.csv",
            seed=seed,
            rows=bnn_rows,
            offset=-10.0,
        )

    for seed in (0, 1):
        _write_run_csv(
            root / "experiments" / "bo_online_general" / f"bo_online_general_seed{seed}.csv",
            seed=seed,
            offset=0.0,
        )
        _write_run_csv(
            root / "experiments" / "lhs_online_general" / f"lhs_online_general_seed{seed}.csv",
            seed=seed,
            offset=-5.0,
        )


def test_comparer_writes_expected_outputs_and_excludes_previous_lhs_bnn(tmp_path: Path) -> None:
    module = _load_compare_module()
    _write_fixture_csvs(tmp_path)
    output_dir = tmp_path / "comparison"

    comparer = module.OWAFSOnlineGeneralComparer(
        repo_root=tmp_path,
        output_dir=output_dir,
        expected_budgeted_trials=80,
    )
    outputs = comparer.run()

    assert {path.name for path in outputs} == {
        "owafs_vs_bo_lhs_bnn_per_run.csv",
        "owafs_vs_bo_lhs_bnn_summary.csv",
        "owafs_vs_bo_lhs_bnn_ranked_runs.csv",
        "owafs_vs_bo_lhs_bnn_best_so_far_by_iteration.csv",
    }
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0

    per_run = pd.read_csv(output_dir / "owafs_vs_bo_lhs_bnn_per_run.csv")
    assert set(per_run["method"]) == {"OWAFS", "BO", "LHS", "BNN"}
    assert per_run["budgeted_rows"].eq(80).all()
    assert per_run.loc[per_run["method"] == "OWAFS", "validation_rows"].eq(1).all()

    bnn_sources = per_run.loc[per_run["method"] == "BNN", "source_csv"].astype(str)
    assert bnn_sources.str.contains("experiments/bnn_online_general").all()
    assert not bnn_sources.str.contains("online_lhs15_shared|lhs15_shared|previous_lhs|previous-lhs").any()

    summary = pd.read_csv(output_dir / "owafs_vs_bo_lhs_bnn_summary.csv")
    assert summary.set_index("method").loc["OWAFS", "seeds"] == "0,1,2,3,4"
    assert summary.set_index("method").loc["BNN", "seeds"] == "0,1,2"
    ranked = pd.read_csv(output_dir / "owafs_vs_bo_lhs_bnn_ranked_runs.csv")
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))


def test_comparer_rejects_incomplete_required_bnn_seed(tmp_path: Path) -> None:
    module = _load_compare_module()
    _write_fixture_csvs(tmp_path, incomplete_bnn_seed=2)

    comparer = module.OWAFSOnlineGeneralComparer(
        repo_root=tmp_path,
        output_dir=tmp_path / "comparison",
        expected_budgeted_trials=80,
    )

    with pytest.raises(RuntimeError, match="BNN seed 2 is incomplete"):
        comparer.run()


def test_forbidden_bnn_path_is_rejected(tmp_path: Path) -> None:
    module = _load_compare_module()
    forbidden = tmp_path / "experiments" / "online_lhs15_shared_init" / "bnn_lhs15_shared_seed0.csv"

    with pytest.raises(RuntimeError, match="Forbidden BNN baseline path"):
        module.validate_bnn_path(forbidden)
