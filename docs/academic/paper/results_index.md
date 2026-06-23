# Results Index (Claim to Evidence)

This file maps paper claims to concrete artifacts and experiment cards.

Rules:
- Every claim used in paper text, figure, or table must include exact artifact
  paths.
- Online claims must state the completion gate used for included seeds.
- BNN online superiority is not supported by the current completed five-seed
  campaign.
- OWAFS superiority over BNN/LHS/Turbo is not supported under the current
  five-seed previous-LHS warmup diagnostic.

## Claim Entries

### C001: The current paper target is a minimized BCLF scalar objective over S11/S21 bands

- experiment card:
  - [experiments.md#current-online-protocol](experiments.md#current-online-protocol)
- evidence artifacts:
  - `docs/academic/experiments/bclf_optimization_target.md`
  - `src/optimization/optimization_spec.py`
  - `notebooks/bo_online_experiment.ipynb`
  - `notebooks/lhs_online_experiment.ipynb`
  - `notebooks/turbo_online_experiment.ipynb`
  - `notebooks/bnn_online_experiment.ipynb`
- inclusion rule:
  - use the frequency bands and limits documented in `docs/academic/experiments/bclf_optimization_target.md`.
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
  - `studies/bnn_surrogate/figures/bnn_bclf_offline_feasibility/holdout_train_test_objective_true_vs_pred.png`
  - `studies/bnn_surrogate/figures/bnn_bclf_offline_feasibility/holdout_best_response_prediction.png`
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
  - `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_learning_curves_by_variant.png`
  - `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_objective_scatter_final_by_variant.png`
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
  - `studies/bnn_surrogate/figures/gp_vs_bnn_response_shape_metric/metric_error_by_train_size.png`
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
  - `studies/bnn_acquisition/figures/bnn_ei_replay/best_so_far_by_method.png`
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
  - `studies/bnn_acquisition/figures/acquisition_quality_gp_qlogei_vs_bnn_qlogei/rank_percentile_by_train_size.png`
  - `studies/bnn_acquisition/figures/acquisition_quality_gp_qlogei_vs_bnn_qlogei/selected_objective_by_train_size.png`
- inclusion rule:
  - compare `gp_qlogei` and `bnn_qlogei` rows over the same 200 snapshots.
- current numeric summary:
  - BNN logEI median selected objective `27.39` vs GP logEI `9.60`; BNN logEI median regret `51.24` vs GP logEI `33.45`.
- status: `paper-candidate`

### C009: BNN online has a completed five-seed baseline but does not beat LHS/Turbo

- experiment card:
  - [experiments.md#experiment-card-bnn-online-campaign](experiments.md#experiment-card-bnn-online-campaign)
- evidence artifacts:
  - `notebooks/bnn_online_experiment.ipynb`
  - `docs/academic/research_plans/bclf_bnn_optimization_plan.md`
  - `tests/optimization/test_bnn_botorch_optimizer.py`
  - `experiments/bnn_online_general/campaign_summary.csv`
  - `experiments/bnn_online_general/bnn_online_general_seed0.csv`
  - `experiments/bnn_online_general/bnn_online_general_seed1.csv`
  - `experiments/bnn_online_general/bnn_online_general_seed2.csv`
  - `experiments/bnn_online_general/bnn_online_general_seed3.csv`
  - `experiments/bnn_online_general/bnn_online_general_seed4.csv`
- inclusion rule:
  - include rows with `status == completed` and `num_trials >= 80`.
  - use only `experiments/bnn_online_general`; explicitly exclude previous-LHS, `online_lhs15_shared`, `lhs15_shared`, `previous_lhs`, and `previous-lhs` BNN artifacts.
- current numeric summary:
  - BNN online_general has 5 completed seeds; median final best `-17.3415`, mean final best `-17.4018`.
  - per-seed best objectives are seed0 `-24.9994`, seed1 `-19.1619`, seed2 `-17.3415`, seed3 `-16.2997`, and seed4 `-9.2064`.
  - BNN is better than BO by median final best (`-17.3415` vs `-14.6571`), but does not beat LHS (`-19.1619`) or Turbo (`-19.1610`).
- status: `paper-candidate`

### C010: Current BO/Turbo online settings do not outperform LHS in the five-seed comparison

- experiment card:
  - [experiments.md#experiment-card-bolhsturbo-online-comparison](experiments.md#experiment-card-bolhsturbo-online-comparison)
- evidence artifacts:
  - `experiments/bo_lhs_turbo_online_comparison/bo_lhs_turbo_final_summary.csv`
  - `experiments/bo_lhs_turbo_online_comparison/bo_lhs_turbo_per_seed_final.csv`
  - `experiments/bo_lhs_turbo_online_comparison/bo_lhs_turbo_best_so_far_by_iteration.csv`
  - `studies/online_comparison/figures/bo_lhs_turbo_online_comparison/bo_lhs_turbo_best_so_far_median.png`
  - `studies/online_comparison/figures/bo_lhs_turbo_online_comparison/bo_lhs_turbo_final_best_by_seed.png`
- inclusion rule:
  - include only methods/seeds whose campaign summaries have `status == completed` and `num_trials >= 80` for seeds `0..4`.
- current numeric summary:
  - median final best: BO `-14.657`, LHS `-19.162`, Turbo `-19.161`; LHS wins 3 of 5 seeds.
- best-candidate response check:
  - global best candidates: BO seed `1` iter `38` objective `-23.814`, LHS seed `0` iter `23` objective `-25.141`, Turbo seed `3` iter `59` objective `-22.517`.
  - LHS has the best scalar objective; BO and LHS satisfy the current response thresholds. Turbo has strong stopband attenuation but passband `S11` max `-6.419 dB`, slightly above the `-7 dB` target.
- status: `paper-candidate`

### C011: Objective-Weighted + Uncertainty-Aware AFS improves objective preservation

- experiment card:
  - [experiments.md#experiment-card-afs-vf-reconstruction-sampling-diagnostic](experiments.md#experiment-card-afs-vf-reconstruction-sampling-diagnostic)
- evidence artifacts:
  - `experiments/uaw_afs_ext_offline_reproduction/acceptance_checks.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/sample_count_sweep_summary.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/sample_count_best_methods.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/sample_count_rank_summary.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/sample_count_regime_summary.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/sample_count_sweep_summary_order_matched.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/sample_count_best_methods_order_matched.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/sample_count_rank_summary_order_matched.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/sample_count_regime_summary_order_matched.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/vf_order_policy_rank_comparison.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/vf_order_policy_takeaways.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/acceptance_checks_order_matched.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/sample_count_sweep_rows_deployable_owafs.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/sample_count_sweep_summary_deployable_owafs.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/sample_count_rank_summary_deployable_owafs.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/reconstruction_rows_deployable_owafs.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/acceptance_checks_deployable_owafs.csv`
  - `experiments/uaw_afs_ext_offline_reproduction/paper_figures/paper_figure_manifest.csv`
  - `docs/academic/paper/assets/uaw_afs_ext_offline_reproduction/paper_figure_manifest.csv`
  - `studies/uaw_afs/figures/paper_assets/uaw_afs_ext_offline_reproduction/uaw_afs_sampling_sweep_order_matched_ieee_double.pdf`
  - `studies/uaw_afs/figures/paper_assets/uaw_afs_ext_offline_reproduction/uaw_afs_sampling_sweep_order_matched_ieee_double.png`
  - `studies/uaw_afs/figures/uaw_afs_ext_offline_reproduction/plots/sample_count_vs_mean_db_mae.png`
  - `studies/uaw_afs/figures/uaw_afs_ext_offline_reproduction/plots/sample_count_vs_objective_abs_error.png`
  - `studies/uaw_afs/figures/uaw_afs_ext_offline_reproduction/plots/sample_count_rank_heatmap.png`
  - `studies/uaw_afs/figures/uaw_afs_ext_offline_reproduction/plots/order_matched_sample_count_vs_mean_db_mae.png`
  - `studies/uaw_afs/figures/uaw_afs_ext_offline_reproduction/plots/order_matched_sample_count_vs_objective_abs_error.png`
  - `studies/uaw_afs/figures/uaw_afs_ext_offline_reproduction/plots/order_matched_sample_count_rank_heatmap.png`
  - `studies/uaw_afs/figures/uaw_afs_ext_offline_reproduction/plots/order_matched_delta_vs_uniform.png`
  - `studies/uaw_afs/figures/uaw_afs_ext_offline_reproduction/plots/deployable_owafs_sample_count_vs_mean_db_mae.png`
  - `studies/uaw_afs/figures/uaw_afs_ext_offline_reproduction/plots/deployable_owafs_sample_count_vs_objective_abs_error.png`
- inclusion rule:
  - offline replay diagnostic only; use cached BCLF `ext` responses, seeds `[0, 1, 2, 3, 4]`, and sampling counts `4..20`.
  - fixed-order and order-matched sweeps each contain `17000/17000` expected rows and `68/68` summarized sample-count/method pairs.
  - order-matched rerun uses `actual_poles = floor(n / 2)` and records
    `vf_pole_residue_terms <= sample_count` for every row.
  - deployable Objective-Weighted + Uncertainty-Aware AFS sweep contains
    `21250/21250` expected rows and `85/85` summarized sample-count/method
    pairs; its acceptance checks record that `owafs_bnn` selects frequencies
    using BNN proxy residuals plus uncertainty, without full-wave truth.
  - paper figure `uaw_afs_sampling_sweep_order_matched` uses the auto IEEE
    layout rule and is exported as double-column only.
  - do not interpret this as a completed online HFSS optimizer comparison.
- current numeric summary:
  - fixed-order baseline: for median mean dB MAE, uniform has the best mean rank `1.71` and wins `12/17`; Uncertainty-Aware AFS has mean rank `2.94`.
  - fixed-order baseline: for median objective absolute error, Objective-Weighted AFS has mean rank `1.29` and wins `13/17`; Uncertainty-Aware AFS has mean rank `2.71`.
  - order-matched rerun: for median mean dB MAE, uniform remains best by mean rank `1.82` and wins `10/17`; Uncertainty-Aware AFS has mean rank `3.00`.
  - order-matched rerun: for median objective absolute error, Objective-Weighted AFS has mean rank `1.59` and wins `11/17`; Uncertainty-Aware AFS has mean rank `1.94`.
  - deployable Objective-Weighted + Uncertainty-Aware AFS rerun: for median objective absolute error, the combined method has the best mean rank `1.47` and wins `11/17`; the truth-informed Objective-Weighted AFS diagnostic has mean rank `1.94`; Uncertainty-Aware AFS has mean rank `2.88`.
  - deployable Objective-Weighted + Uncertainty-Aware AFS rerun: for median mean dB MAE, uniform remains best by mean rank `1.94` and wins `10/17`; the combined method has mean rank `4.24`, so it should be described as objective-preserving rather than full-curve-reconstruction optimal.
  - interpretation: Uncertainty-Aware AFS does not dominate simple baselines. Objective-weighted residual signals are useful for preserving the BCLF objective, and the combined objective-weighted plus uncertainty-aware variant makes that signal deployable for offline replay, but full-curve dB MAE and objective-error optimization remain different targets.
- status: `draft`

### C012: OWAFS improves its 15-point previous-LHS warmup but does not beat LHS/Turbo

- experiment card:
  - [experiments.md#experiment-card-owafs-previous-lhs-warmup-online-diagnostic](experiments.md#experiment-card-owafs-previous-lhs-warmup-online-diagnostic)
- evidence artifacts:
  - `experiments/online_lhsdb_warmup/initial_designs/warmup_initial_points.csv`
  - `experiments/online_lhsdb_warmup/owafs_bnn/campaign_summary.csv`
  - `experiments/online_lhsdb_warmup/owafs_bnn/owafs_bnn_online_lhsdb_warmup_seed0.csv`
  - `experiments/online_lhsdb_warmup/owafs_bnn/owafs_bnn_online_lhsdb_warmup_seed1.csv`
  - `experiments/online_lhsdb_warmup/owafs_bnn/owafs_bnn_online_lhsdb_warmup_seed2.csv`
  - `experiments/online_lhsdb_warmup/owafs_bnn/owafs_bnn_online_lhsdb_warmup_seed3.csv`
  - `experiments/online_lhsdb_warmup/owafs_bnn/owafs_bnn_online_lhsdb_warmup_seed4.csv`
  - `experiments/online_lhsdb_warmup/owafs_bnn/diagnostics/owafs_geometry_coverage_summary_seed0_4.csv`
  - `experiments/online_lhsdb_warmup/owafs_bnn/comparison/owafs_vs_bo_lhs_bnn_per_run.csv`
  - `experiments/online_lhsdb_warmup/owafs_bnn/comparison/owafs_vs_bo_lhs_bnn_summary.csv`
  - `studies/online_owafs/reports/OWAFS_EXPERIMENT_STATUS_AND_NEXT_STEPS.md`
- inclusion rule:
  - include OWAFS seeds `0..4` after each has exactly `80` budgeted rows; filter validation rows from budgeted trajectories.
  - BNN baseline must use only `experiments/bnn_online_general/bnn_online_general_seed{0..4}.csv`.
  - explicitly exclude previous-LHS, `online_lhs15_shared`, `lhs15_shared`, `previous_lhs`, and `previous-lhs` BNN artifacts.
  - BO/LHS/Turbo/BNN online_general seeds `0..4` are descriptive references; this is not a paired initialization test because OWAFS starts from a different 15-point warmup.
- current numeric summary:
  - OWAFS seed0 improves from warmup best `-16.43` to `-22.08`.
  - OWAFS seed1 improves from warmup best `-11.47` to `-17.41`.
  - OWAFS seed2 improves from warmup best `0.02` to `-6.80`.
  - OWAFS seed3 improves from warmup best `-2.06` to `-23.82`.
  - OWAFS seed4 improves from warmup best `6.64` to `-16.80`.
  - OWAFS seed0-4 median final best is `-17.41`, mean final best is `-17.38`.
  - BNN online_general seed0-4 median final best is `-17.34`, mean final best is `-17.40`.
  - BO/LHS/Turbo online_general seed0-4 median final best values are `-14.66`, `-19.16`, and `-19.16`.
  - interpretation: this supports warmup improvement and partial competitiveness with BO/BNN, but it does not support OWAFS superiority over LHS or Turbo.
- status: `paper-candidate`

### C013: Current OWAFS geometry does not show sustained unknown-region expansion

- experiment card:
  - [experiments.md#experiment-card-owafs-previous-lhs-warmup-online-diagnostic](experiments.md#experiment-card-owafs-previous-lhs-warmup-online-diagnostic)
- evidence artifacts:
  - `experiments/online_lhsdb_warmup/owafs_bnn/diagnostics/owafs_geometry_coverage_summary_seed0_4.csv`
  - `studies/online_owafs/figures/online_lhsdb_warmup/owafs_geometry_coverage_pca_seed0_4.pdf`
  - `studies/online_owafs/figures/online_lhsdb_warmup/owafs_geometry_coverage_pca_seed0_4.png`
  - `studies/online_owafs/figures/online_lhsdb_warmup/owafs_geometry_distance_to_lhs_seed0_4.pdf`
  - `studies/online_owafs/figures/online_lhsdb_warmup/owafs_geometry_distance_to_lhs_seed0_4.png`
- inclusion rule:
  - fit the PCA projection jointly over seed0-4 warmup and OWAFS budgeted points.
  - compute each OWAFS online point's normalized Euclidean nearest distance to the same seed's original 15 LHS warmup points.
  - validation rows are excluded from the budgeted trajectory.
- current numeric summary:
  - median distance-to-LHS is `0.69`, `0.66`, `0.67`, `0.73`, and `0.71` for seeds `0..4`.
  - maximum distance-to-LHS is `1.04`, `0.97`, `1.03`, `1.09`, and `1.11` for seeds `0..4`.
  - best iterations are `44`, `32`, `41`, `18`, and `26`, so the best points occur before or near the middle of the budget rather than as late exploration discoveries.
  - interpretation: OWAFS does move away from exact warmup duplicates, but the current nearest-distance trajectories do not show sustained expansion into unknown geometry regions.
- status: `draft`

## Promotion Checklist

Before changing a claim entry from `draft` to `paper-candidate`:
- Verify all artifact paths exist.
- Verify experiment-card required fields are complete.
- Verify inclusion rules are explicit and reproducible.
- Recompute numeric summaries from CSV/JSON artifacts.
