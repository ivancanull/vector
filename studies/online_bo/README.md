# Online BO Study

Bayesian optimization online and replay assets.

- Scripts: `studies/online_bo/scripts`
- Data: `experiments/bo_online_general`, `experiments/bo_replay`
- Figures: `studies/online_comparison/figures/bo_lhs_online_comparison`,
  `studies/online_comparison/figures/bo_lhs_turbo_online_comparison`
- Notebooks remain in `notebooks/`.

Run from the repository root:

```bash
python studies/online_bo/scripts/run_bo_online_experiment.py --backend server
```

For online continuous runs, GP logEI is optimized over the normalized bounded
geometry domain and the selected point is snapped to the configured resolution
before cache lookup or HFSS submission. The fixed finite-candidate path is kept
for replay/discrete experiments.

When you already have a tmux window open, use the wrapper below. It writes a
timestamped log and status file under `logs/`, but does not create tmux sessions
or windows:

```bash
studies/online_bo/scripts/run_bo_online_in_tmux_window.sh --backend server
```

Use `--reset-checkpoint` only when intentionally rerunning a seed from scratch.
