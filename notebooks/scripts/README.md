# Notebook Scripts Moved

Experiment scripts now live under `studies/` so each study keeps its scripts,
reports, and figures together. Notebook files remain under `notebooks/`.

Use these entry points:

- BNN online runner: `studies/online_bnn/scripts/run_bnn_online_experiment.py`
- BNN tmux launcher: `studies/online_bnn/scripts/run_bnn_online_tmux.sh`
- LHSDB warmup runner: `studies/online_owafs/scripts/run_online_lhsdb_warmup.py`
- OWAFS geometry diagnostics: `studies/online_owafs/scripts/plot_owafs_geometry_coverage.py`
- OWAFS status report: `studies/online_owafs/scripts/summarize_owafs_experiment_status.py`
- Online comparison tables: `studies/online_comparison/scripts/compare_owafs_vs_bo_lhs_bnn.py`
- Online comparison figures: `studies/online_comparison/scripts/plot_owafs_vs_bo_lhs_bnn_objective.py`

Data artifacts remain under `experiments/`.
