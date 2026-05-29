from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
PLAYGROUND = ROOT / "experiments" / "hfss_skill_playground"


def _load_analyzer():
    path = PLAYGROUND / "scripts" / "analyze_playground.py"
    spec = importlib.util.spec_from_file_location("hfss_skill_playground_analyzer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_playground_cases_define_research_protocol() -> None:
    data = yaml.safe_load((PLAYGROUND / "cases.yaml").read_text(encoding="utf-8"))

    assert data["source_case_pool"] == "skills/hfss-rf-skill/experiments/public_hfss_validation_cases.yaml"
    assert set(("baseline", "current_skill", "improved_skill")).issubset(data["methods"])
    assert data["protocol"]["max_attempts"] == 2
    assert data["protocol"]["max_repairs"] == 1
    assert data["budgets"]["pilot_hfss_calls"] == 20
    assert len(data["selected_cases"]) >= 6
    for case in data["selected_cases"]:
        assert case["id"]
        assert case["layer"] in {"case_reproduction", "closed_loop_optimization"}
        assert case["prompt_goal"]
        assert case["design_variables"]
        assert case["objective"]["direction"] in {"minimize", "maximize"}
        assert isinstance(case["objective"]["target_threshold"], (int, float))
        assert case["verifier_rules"]


def test_playground_analyzer_summarizes_reproduction_and_optimization(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    shutil.copy(PLAYGROUND / "cases.yaml", tmp_path / "cases.yaml")
    shutil.copy(PLAYGROUND / "failure_taxonomy.yaml", tmp_path / "failure_taxonomy.yaml")
    run_dir = tmp_path / "runs" / "dipole_antenna_return_loss" / "current_skill" / "0"
    run_dir.mkdir(parents=True)
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "case_id": "dipole_antenna_return_loss",
                "method": "current_skill",
                "seed": 0,
                "reproduction": {
                    "syntax_pass": True,
                    "import_pass": True,
                    "hfss_build_success": True,
                    "setup_success": True,
                    "solve_success": True,
                    "result_export_success": True,
                    "verifier_pass": True,
                    "human_edit_count": 1,
                    "repair_count": 1,
                    "failure_type": None,
                },
                "optimization": {
                    "trials": [
                        {"call": 1, "true_objective": -4.0, "best_so_far": -4.0, "wall_clock_sec": 10.0},
                        {"call": 2, "true_objective": -11.0, "best_so_far": -11.0, "wall_clock_sec": 25.0},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    objectives = analyzer.load_case_objectives(tmp_path)
    penalties = analyzer.load_failure_penalties(tmp_path)
    row = analyzer.summarize_run(run_dir / "evaluation.json", objectives, penalties)

    assert row["target_reached"] is True
    assert row["calls_to_target"] == 2
    assert row["wall_clock_time_to_target_sec"] == 25.0
    assert row["final_best_true_objective"] == -11.0
    assert row["failure_adjusted_score"] == -11.0

    summary = tmp_path / "summary.csv"
    report = tmp_path / "report.md"
    analyzer.write_summary([row], summary)
    analyzer.write_report([row], report)
    assert "current_skill" in summary.read_text(encoding="utf-8")
    assert "HFSS RF Skill Playground Report" in report.read_text(encoding="utf-8")
