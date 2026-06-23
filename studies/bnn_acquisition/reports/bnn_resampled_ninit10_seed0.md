# Offline Cache Ranking Diagnostic

- Generated: `2026-06-10T06:13:56.639606+00:00`
- Candidates: `1602`
- True best objective: `-25.1413`
- Ranking CSV: `/home/zhanghf/projects/vector/experiments/bnn_acquisition_offline_cache_ranking/bnn_resampled_ninit10_seed0/ranking.csv`
- Figure: `/home/zhanghf/projects/vector/studies/bnn_acquisition/figures/bnn_resampled_ninit10_seed0_top1_regret.png`

| Method | Spearman | True-best rank | Top-1 true objective | Top-1 regret | Top-10 overlap |
|---|---:|---:|---:|---:|---:|
| bnn_predicted | 0.331 | 206 | 23.9483 | 49.0896 | 0/10 |
| bnn_ei | 0.322 | 161 | 23.9483 | 49.0896 | 0/10 |
| gp_predicted | 0.306 | 313 | 5.28626 | 30.4275 | 0/10 |
| gp_ei | 0.144 | 896 | -0.0918047 | 25.0495 | 0/10 |
