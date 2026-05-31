# Results Index (Claim to Evidence)

This file maps paper claims to concrete artifacts and experiment cards.

Rules:
- Every claim used in paper text, figure, or table must include exact artifact
  paths.
- Online claims must state the completion gate used for included seeds.
- BNN online superiority is not a valid claim until the full BNN online campaign
  is complete.

## Claim Entries

### C001: The current paper target is a minimized BCLF scalar objective over S11/S21 bands

- experiment card:
  - [experiments.md#current-online-protocol](experiments.md#current-online-protocol)
- evidence artifacts:
  - `docs/bclf_optimization_target.md`
  - `src/optimization/optimization_spec.py`
  - `notebooks/bo_online_experiment.ipynb`
  - `notebooks/lhs_online_experiment.ipynb`
  - `notebooks/turbo_online_experiment.ipynb`
  - `notebooks/bnn_online_experiment.ipynb`
- inclusion rule:
  - use the frequency bands and limits documented in `docs/bclf_optimization_target.md`.
- status: `paper-candidate`

### C002: BO online has a completed five-seed baseline under the current protocol

- experiment card:
  - [experiments.md#experiment-card-bo-online-baseline](experiments.md#experiment-card-bo-online-baseline)
- evidence artifacts:
  - `experiments/bo_online_general/campaign_summary.csv`
  - `experiments/bo_online_general/bo_online_general_seed0.json`
  - `experiments/bo_online_general/bo_online_general_seed1.json`
  - `experiments/bo_online_general/bo_online_general_seed2.json`
  - `experiments/bo_online_general/bo_online_general_seed3.json`
  - `experiments/bo_online_general/bo_online_general_seed4.json`
- inclusion rule:
  - include rows with `status == completed` and `num_trials >= 80`.
- current numeric summary:
  - 5 completed seeds; median final best `-14.657`, mean final best `-15.649`.
- status: `paper-candidate`

### C003: LHS online has a completed five-seed baseline under the current protocol

- experiment card:
  - [experiments.md#experiment-card-lhs-online-baseline](experiments.md#experiment-card-lhs-online-baseline)
- evidence artifacts:
  - `experiments/lhs_online_general/campaign_summary.csv`
  - `experiments/lhs_online_general/lhs_online_general_seed0.json`
  - `experiments/lhs_online_general/lhs_online_general_seed1.json`
  - `experiments/lhs_online_general/lhs_online_general_seed2.json`
  - `experiments/lhs_online_general/lhs_online_general_seed3.json`
  - `experiments/lhs_online_general/lhs_online_general_seed4.json`
- inclusion rule:
  - include rows with `status == completed` and `num_trials >= 80`.
- current numeric summary:
  - 5 completed seeds; median final best `-19.162`, mean final best `-18.815`.
- status: `paper-candidate`

### C004: The BNN response surrogate is feasible but held-out ranking remains imperfect

- experiment card:
  - [experiments.md#experiment-card-bnn-offline-response-surrogate-feasibility](experiments.md#experiment-card-bnn-offline-response-surrogate-feasibility)
- evidence artifacts:
  - `experiments/bnn_bclf_offline_feasibility/holdout_accuracy_metrics.csv`
  - `experiments/bnn_bclf_offline_feasibility/holdout_train_test_objective_true_vs_pred.png`
  - `experiments/bnn_bclf_offline_feasibility/holdout_best_response_prediction.png`
- inclusion rule:
  - use the train/test rows from `holdout_accuracy_metrics.csv`.
- current numeric summary:
  - test objective MAE `22.63`, test Spearman `0.38`, test S21 MAE `2.97 dB`, top-10 recall `0.60`.
- status: `paper-candidate`

### C005: Few-shot BNN warm-start updates improve the offline online-learning signal

- experiment card:
  - [experiments.md#experiment-card-few-shot-bnn-online-learning-replay](experiments.md#experiment-card-few-shot-bnn-online-learning-replay)
- evidence artifacts:
  - `experiments/bnn_bclf_fewshot_online_learning/fewshot_metrics_by_variant.csv`
  - `experiments/bnn_bclf_fewshot_online_learning/fewshot_learning_curves_by_variant.png`
  - `experiments/bnn_bclf_fewshot_online_learning/fewshot_objective_scatter_final_by_variant.png`
- inclusion rule:
  - compare `split == test` rows at initial iteration `0` and final iteration `50`.
- current numeric summary:
  - Fourier final test objective MAE `25.05`, Spearman `0.44`, S21 MAE `2.37 dB`, top-10 recall `0.30`.
- status: `paper-candidate`

### C006: Fourier BNN response features are competitive for response-shape metric prediction

- experiment card:
  - [experiments.md#experiment-card-gp-vs-bnn-response-shape-metric](experiments.md#experiment-card-gp-vs-bnn-response-shape-metric)
- evidence artifacts:
  - `experiments/gp_vs_bnn_response_shape_metric/summary_by_seed_size_method.csv`
  - `experiments/gp_vs_bnn_response_shape_metric/topk_summary.csv`
  - `experiments/gp_vs_bnn_response_shape_metric/metric_error_by_train_size.png`
- inclusion rule:
  - aggregate by `train_size` and `method` across seeds `0..4`.
- current numeric summary:
  - at train size 40, Fourier BNN mean metric MAE `6.68` and mean Spearman `0.31`; GP mean metric MAE `6.64` and mean Spearman `0.21`.
- status: `paper-candidate`

### C007: BNN EI replay is operational but does not beat GP on median/mean final best

- experiment card:
  - [experiments.md#experiment-card-bnn-ei-replay-optimizer](experiments.md#experiment-card-bnn-ei-replay-optimizer)
- evidence artifacts:
  - `experiments/bnn_ei_replay/comparison_summary.csv`
  - `experiments/bnn_ei_replay/comparison_runs.csv`
  - `experiments/bnn_ei_replay/best_so_far_by_method.png`
- inclusion rule:
  - include rows with `status == completed` and `num_trials == target_trials`.
- current numeric summary:
  - GP logEI median final best `-22.77`; BNN EI median final best `-22.08`.
- status: `paper-candidate`

### C008: Current BNN logEI selection quality is weaker than GP logEI in offline snapshots

- experiment card:
  - [experiments.md#experiment-card-acquisition-quality-snapshot](experiments.md#experiment-card-acquisition-quality-snapshot)
- evidence artifacts:
  - `experiments/acquisition_quality_gp_qlogei_vs_bnn_qlogei/acquisition_quality_summary.csv`
  - `experiments/acquisition_quality_gp_qlogei_vs_bnn_qlogei/acquisition_quality_rows.csv`
  - `experiments/acquisition_quality_gp_qlogei_vs_bnn_qlogei/rank_percentile_by_train_size.png`
  - `experiments/acquisition_quality_gp_qlogei_vs_bnn_qlogei/selected_objective_by_train_size.png`
- inclusion rule:
  - compare `gp_qlogei` and `bnn_qlogei` rows over the same 200 snapshots.
- current numeric summary:
  - BNN logEI median selected objective `27.39` vs GP logEI `9.60`; BNN logEI median regret `51.24` vs GP logEI `33.45`.
- status: `paper-candidate`

### C009: BNN online is implemented as a protocol, not yet as a performance result

- experiment card:
  - [experiments.md#experiment-card-bnn-online-campaign](experiments.md#experiment-card-bnn-online-campaign)
- evidence artifacts:
  - `notebooks/bnn_online_experiment.ipynb`
  - `docs/bclf_bnn_optimization_plan.md`
  - `tests/optimization/test_bnn_botorch_optimizer.py`
- inclusion rule:
  - do not report BNN online performance until `experiments/bnn_online_general/campaign_summary.csv` exists with completed 80-trial rows for the intended seed set.
- status: `draft`

### C010: Current BO/Turbo online settings do not outperform LHS in the five-seed comparison

- experiment card:
  - [experiments.md#experiment-card-bolhsturbo-online-comparison](experiments.md#experiment-card-bolhsturbo-online-comparison)
- evidence artifacts:
  - `experiments/bo_lhs_turbo_online_comparison/bo_lhs_turbo_final_summary.csv`
  - `experiments/bo_lhs_turbo_online_comparison/bo_lhs_turbo_per_seed_final.csv`
  - `experiments/bo_lhs_turbo_online_comparison/bo_lhs_turbo_best_so_far_by_iteration.csv`
  - `experiments/bo_lhs_turbo_online_comparison/bo_lhs_turbo_best_so_far_median.png`
  - `experiments/bo_lhs_turbo_online_comparison/bo_lhs_turbo_final_best_by_seed.png`
- inclusion rule:
  - include only methods/seeds whose campaign summaries have `status == completed` and `num_trials >= 80` for seeds `0..4`.
- current numeric summary:
  - median final best: BO `-14.657`, LHS `-19.162`, Turbo `-19.161`; LHS wins 3 of 5 seeds.
- status: `paper-candidate`

## Promotion Checklist

Before changing a claim entry from `draft` to `paper-candidate`:
- Verify all artifact paths exist.
- Verify experiment-card required fields are complete.
- Verify inclusion rules are explicit and reproducible.
- Recompute numeric summaries from CSV/JSON artifacts.
