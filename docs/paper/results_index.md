# Results Index (Claim to Evidence)

This file maps paper claims to concrete artifacts and experiment cards.

Rules:
- Every claim used in paper text/figure/table must include:
  - source experiment card
  - exact artifact file paths
  - inclusion rule reference

---

## Claim Entries

### C001: Online BO campaign can be resumed across seeds while preserving per-seed histories

- experiment card:
  - [experiments.md#experiment-card-online-bo-seed-campaign](experiments.md#experiment-card-online-bo-seed-campaign)
- evidence artifacts:
  - `experiments/bo_online/campaign_summary.csv`
  - `experiments/bo_online/bo_online_seed0.checkpoint.json`
  - `experiments/bo_online/bo_online_seed1.checkpoint.json`
  - `experiments/bo_online/bo_online_seed2.checkpoint.json`
- inclusion rule:
  - use only seeds satisfying campaign completion rule in the experiment card.
- status: `draft`

### C002: Replay grid summaries provide cross-setting, multi-seed comparison tables

- experiment card:
  - [experiments.md#experiment-card-bo-replay-grid-study](experiments.md#experiment-card-bo-replay-grid-study)
- evidence artifacts:
  - `experiments/bo_replay/optimization_grid_summary.csv`
  - `experiments/bo_replay/optimization_grid_runs.csv`
  - representative per-run files: `experiments/bo_replay/grid_ucb_beta2_n4_seed1.csv`, `experiments/bo_replay/grid_ei_xi005_n4_seed1.csv`
- inclusion rule:
  - include all seeds for each compared setting; apply one aggregation rule consistently.
- status: `draft`

---

## Promotion Checklist

Before changing a claim entry from `draft` to `paper-candidate`:
- Verify all artifact paths exist.
- Verify experiment-card required fields are complete.
- Verify inclusion rule is explicit and reproducible.
