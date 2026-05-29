# HFSS RF Skill Research Playground

This playground evaluates generated RF skills from a research perspective. It
separates two questions:

- Can the skill reliably reproduce public HFSS/PyAEDT cases?
- Can the generated workflow help closed-loop optimization reach better true
  HFSS objectives faster?

The scaffold is intentionally small. It stores case protocols and analysis code;
large run outputs stay local under `runs/`.

## Directory Layout

```text
experiments/hfss_skill_playground/
  cases.yaml
  failure_taxonomy.yaml
  scripts/analyze_playground.py
  runs/<case_id>/<method>/<seed>/evaluation.json
  runs/<case_id>/<method>/<seed>/history.csv
  summary.csv
```

## Methods

Use the same case and seed set for each method:

- `baseline`: generic prompt, no RF skill context.
- `current_skill`: current `hfss-rf-skill`.
- `improved_skill`: skill after lessons from prior failures.
- `reference`: optional human reference script upper bound.

## Run Record

Each run should write `evaluation.json` with reproduction fields and either an
optimization history or a pointer to `history.csv`.

Minimal shape:

```json
{
  "case_id": "spiral_inductor_l_q",
  "method": "current_skill",
  "seed": 0,
  "reproduction": {
    "syntax_pass": true,
    "import_pass": true,
    "hfss_build_success": true,
    "setup_success": true,
    "solve_success": true,
    "result_export_success": true,
    "verifier_pass": true,
    "human_edit_count": 0,
    "repair_count": 0,
    "failure_type": null
  },
  "optimization": {
    "target_reached": true,
    "trials": [
      {"call": 1, "true_objective": 4.0, "best_so_far": 4.0, "wall_clock_sec": 100},
      {"call": 2, "true_objective": 2.0, "best_so_far": 2.0, "wall_clock_sec": 220}
    ]
  }
}
```

## Analysis

```powershell
python experiments/hfss_skill_playground/scripts/analyze_playground.py
```

The analyzer writes:

- `summary.csv`: one row per run.
- `playground_report.generated.md`: method-level medians, success rates, and
  failure counts.

Primary research metrics:

- median final true objective;
- median HFSS calls to target;
- median wall-clock time to target;
- area under the best-so-far curve;
- solve/export/verifier success rates;
- failure-adjusted score.
