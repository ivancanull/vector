# Reproducibility Policy

This repo uses a pragmatic standard for paper iteration:

- Required for each reported experiment:
  - config reference
  - seed policy and actual seeds
  - artifact links (`csv/json` summaries and plots where applicable)

- Not required at this stage:
  - full environment lock and byte-level build reproducibility

## Required Fields Per Experiment Card

For each card in [experiments.md](experiments.md):
- `experiment_id`
- `status` (`draft` | `paper-candidate` | `camera-ready`)
- `config references`
- `seed policy`
- `seeds used`
- `artifact links`

## Artifact Conventions

- Online campaign artifacts:
  - `experiments/bo_online/campaign_summary.csv`
  - `experiments/bo_online/bo_online_seed*.json`
  - `experiments/bo_online/bo_online_seed*.csv`
  - `experiments/bo_online/bo_online_seed*.checkpoint.json`
- Replay/grid artifacts:
  - `experiments/bo_replay/optimization_grid_summary.csv`
  - `experiments/bo_replay/optimization_grid_runs.csv`
  - `experiments/bo_replay/grid_*.json`
  - `experiments/bo_replay/grid_*.csv`

## Update Rules

- Any new result used in paper text, figures, or tables must update:
  - one experiment card in `experiments.md`
  - and one claim entry in `results_index.md`
- If a card is `paper-candidate`, all required fields must be complete.
