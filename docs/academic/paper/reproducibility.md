# Reproducibility Policy

This repo uses a pragmatic reproducibility standard for paper iteration. The
goal is traceable claims, stable experiment contracts, and rerunnable notebooks,
not byte-level environment locking.

## Required Fields Per Experiment Card

Each paper-facing card in [experiments.md](experiments.md) must include:

- `experiment_id`
- `status` (`draft` | `paper-candidate` | `camera-ready`)
- primary notebook or implementation reference
- seed policy and actual seeds used
- objective/protocol reference
- artifact links
- inclusion rule for reported figures/tables

## Artifact Conventions

| Family | Required artifacts |
|---|---|
| BO online | `experiments/bo_online_general/campaign_summary.csv`, per-seed `.json`, `.csv`, `.checkpoint.json` |
| LHS online | `experiments/lhs_online_general/campaign_summary.csv`, per-seed `.json`, `.csv`, `.checkpoint.json` |
| Turbo online | `experiments/turbo_online_general/campaign_summary.csv`, per-seed `.json`, `.csv`, `.checkpoint.json` |
| BO/LHS/Turbo comparison | summary CSVs under `experiments/bo_lhs_turbo_online_comparison/`; plots under `studies/online_comparison/figures/` |
| BNN online | `experiments/bnn_online_general/campaign_summary.csv`, per-seed `.json`, `.csv`, `.checkpoint.json` once the full campaign is run |
| BNN offline feasibility | summary CSVs under `experiments/bnn_bclf_offline_feasibility/`; plots under `studies/bnn_surrogate/figures/` |
| Few-shot BNN replay | `fewshot_metrics_by_variant.csv` under `experiments/`; learning/scatter plots under `studies/bnn_surrogate/figures/` |
| Response-shape metric | `summary_by_seed_size_method.csv` and `topk_summary.csv` under `experiments/`; plots under `studies/bnn_surrogate/figures/` |
| BNN EI replay | `comparison_summary.csv` and `comparison_runs.csv` under `experiments/`; best-so-far plots under `studies/bnn_acquisition/figures/` |
| Acquisition quality | `acquisition_quality_summary.csv` and row-level CSVs under `experiments/`; rank/objective plots under `studies/bnn_acquisition/figures/` |

## Paper Figure Export Convention

Paper-facing figures should be exported through `src/paper_figures.py` rather
than ad hoc `savefig` calls. The exporter uses IEEE column widths and chooses a
layout from deterministic content hints:

- `single`: one panel, at most three curves, short labels, compact legend.
- `double`: multi-panel figures, heatmaps, dense tick labels, four or more
  methods/curves, large legends, or response plots with markers/band shading.

Do not generate both single- and double-column PDFs unless a figure explicitly
requests both layouts. Each exported paper figure must write a manifest row
with figure id, chosen layout, reason, format, dimensions, and path. Diagnostic
plots and article-ready assets should live under the appropriate
`studies/<study>/figures/` directory; CSV/JSON/checkpoint/model artifacts remain
under `experiments/`.

## Inclusion Gates

- BO online: include seeds with `status == completed` and `num_trials >= 80`.
- LHS online: include seeds with `status == completed` and `num_trials >= 80`.
- Turbo online: include seeds with `status == completed` and `num_trials >= 80`.
- BO/LHS/Turbo comparison: include only seed/method pairs that pass the online gate for the full configured seed set.
- BNN online: include only completed rows under `experiments/bnn_online_general/`; current five-seed results support a completed baseline but not BNN superiority over LHS/Turbo.
- Replay comparisons: include only rows whose `status == completed` and whose `num_trials` equals the configured `target_trials` when that field exists.
- Offline surrogate studies: use the exact split/method rows named in each experiment card.

## Execution Environment

Use the `vector` conda environment for notebook execution. The user-level
`jupyter` executable may point to a Python that lacks required packages.

Recommended command pattern:

```bash
/home/zhanghf/anaconda3/envs/vector/bin/python -m jupyter nbconvert \
  --to notebook --execute notebooks/<name>.ipynb \
  --output <name>.executed.ipynb \
  --ExecutePreprocessor.kernel_name=vector \
  --ExecutePreprocessor.timeout=-1
```

Queue-backed online notebooks submit real HFSS jobs through the server/worker
queue. Run them only when the worker is available and the intended campaign
state is clean.

## Numeric Summary Rule

Every number copied into paper text must be recomputed from artifacts at the
time the paper docs are updated. For online summaries, aggregate the
completion-gated rows from `campaign_summary.csv`; for replay summaries,
aggregate the method rows from the named comparison CSV.

Example checks:

```bash
python - <<'PY'
import pandas as pd

bo = pd.read_csv("experiments/bo_online_general/campaign_summary.csv")
bo = bo[(bo["status"] == "completed") & (bo["num_trials"] >= 80)]
print(bo["best_objective"].agg(["count", "median", "mean", "min", "max"]))

replay = pd.read_csv("experiments/bnn_ei_replay/comparison_summary.csv")
print(replay.groupby("method_label")["best_objective"].agg(["count", "median", "mean", "min", "max"]))
PY
```

## Update Rules

- Any new result used in paper text, figures, or tables must update one
  experiment card in `experiments.md` and one claim entry in
  `results_index.md`.
- Any new plotted claim must list exact plot paths in `results_index.md`.
- If an objective, band, seed policy, or artifact root changes, update
  `docs/academic/experiments/bclf_optimization_target.md` first, then update paper docs.
- Do not promote BNN online superiority claims under the current completed
  campaign; report the negative/limited comparison explicitly.
