# Online Comparison Study

Cross-method comparison assets for BO, LHS, Turbo, BNN, LHS15 shared-init, and
OWAFS/LHSDB warmup experiments.

- Scripts: `studies/online_comparison/scripts`
- Comparison CSVs: `experiments/online_lhsdb_warmup/owafs_bnn/comparison`
- Figures: `studies/online_comparison/figures`
- Data sources: `experiments/bo_online_general`, `experiments/lhs_online_general`,
  `experiments/bnn_online_general`, and `experiments/online_lhsdb_warmup`

Typical refresh:

```bash
python studies/online_comparison/scripts/compare_owafs_vs_bo_lhs_bnn.py
python studies/online_comparison/scripts/plot_owafs_vs_bo_lhs_bnn_objective.py
```
