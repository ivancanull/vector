# Experiments

This page tracks paper-facing experiment families. Treat notebooks and files
under `experiments/` as the data-generation source of truth.

## Current Online Protocol

| Setting | BO | LHS | Turbo | BNN online |
|---|---|---|---|---|
| Notebook | `notebooks/bo_online_experiment.ipynb` | `notebooks/lhs_online_experiment.ipynb` | `notebooks/turbo_online_experiment.ipynb` | `notebooks/bnn_online_experiment.ipynb` |
| Optimizer | `OptimizerBoTorch` | `OptimizerLHSRandom` | `OptimizerBoTorch` with trust-region selection | `OptimizerBoTorch` |
| Selection | GP logEI | LHS random sequence | trust-region GP logEI | BNN response logEI |
| Simulator | `ServerQueueSimulator` | `ServerQueueSimulator` | `ServerQueueSimulator` | `ServerQueueSimulator` |
| Candidate space | `ContinuousCandidateSpace` | `ContinuousCandidateSpace` | `ContinuousCandidateSpace` | `ContinuousCandidateSpace` |
| Pool size | `1500` | `1500` | `1500` | `1500` |
| Seeds | `[0, 1, 2, 3, 4]` | `[0, 1, 2, 3, 4]` | `[0, 1, 2, 3, 4]` | `[0, 1, 2, 3, 4]` |
| Target trials | `80` | `80` | `80` | `80` |
| Output root | `experiments/bo_online_general/` | `experiments/lhs_online_general/` | `experiments/turbo_online_general/` | `experiments/bnn_online_general/` |

All four online families use the current BCLF target in
[BCLF Optimization Target And Online Settings](../experiments/bclf_optimization_target.md).
The optimizer minimizes scalar objective; lower `best_so_far` is better.

The OWAFS previous-LHS warmup diagnostic below is intentionally tracked
separately from this online-general protocol because it starts from 15
previous-LHS warmup points instead of the 30-point online-general
initialization.

## Experiment Card: BO Online Baseline

- experiment_id: `bo_online_general_v1`
- status: `paper-candidate`
- primary notebook: `notebooks/bo_online_experiment.ipynb`
- output root: `experiments/bo_online_general/`

### Research Question

Can GP logEI operate as a queue-backed online baseline for BCLF optimization
under the current target and 80-trial seed budget?

### Setup

- Method label: `global_gp_qlogei`
- `N_INIT = 30`, `BUDGET = 50`, total `80` trials per seed
- Seeds included now: `0, 1, 2, 3, 4`
- Completion gate: `status == completed` and `num_trials >= 80`

### Current Reportable Result

From `experiments/bo_online_general/campaign_summary.csv`, all 5 seeds satisfy
the completion gate.

| Seeds | Median final best | Mean final best | Best seed result | Worst seed result |
|---:|---:|---:|---:|---:|
| 5 | -14.657 | -15.649 | -23.814 | -8.052 |

### Evidence Artifacts

- `experiments/bo_online_general/campaign_summary.csv`
- `experiments/bo_online_general/bo_online_general_seed0.json`
- `experiments/bo_online_general/bo_online_general_seed1.json`
- `experiments/bo_online_general/bo_online_general_seed2.json`
- `experiments/bo_online_general/bo_online_general_seed3.json`
- `experiments/bo_online_general/bo_online_general_seed4.json`

## Experiment Card: LHS Online Baseline

- experiment_id: `lhs_online_general_v1`
- status: `paper-candidate`
- primary notebook: `notebooks/lhs_online_experiment.ipynb`
- output root: `experiments/lhs_online_general/`

### Research Question

How does a model-free LHS random baseline perform under the same objective,
candidate-space, simulator, and per-seed budget contract as BO?

### Setup

- Method label: `lhs_random`
- Total budget: `80` trials per seed
- Seeds included now: `0, 1, 2, 3, 4`
- Completion gate: `status == completed` and `num_trials >= 80`

### Current Reportable Result

From `experiments/lhs_online_general/campaign_summary.csv`, all 5 seeds satisfy
the completion gate.

| Seeds | Median final best | Mean final best | Best seed result | Worst seed result |
|---:|---:|---:|---:|---:|
| 5 | -19.162 | -18.815 | -25.141 | -9.847 |

### Evidence Artifacts

- `experiments/lhs_online_general/campaign_summary.csv`
- `experiments/lhs_online_general/lhs_online_general_seed0.json`
- `experiments/lhs_online_general/lhs_online_general_seed1.json`
- `experiments/lhs_online_general/lhs_online_general_seed2.json`
- `experiments/lhs_online_general/lhs_online_general_seed3.json`
- `experiments/lhs_online_general/lhs_online_general_seed4.json`

## Experiment Card: Turbo Online Baseline

- experiment_id: `turbo_online_general_v1`
- status: `paper-candidate`
- primary notebook: `notebooks/turbo_online_experiment.ipynb`
- output root: `experiments/turbo_online_general/`

### Research Question

Does a trust-region GP logEI policy improve online BCLF optimization relative
to the global BO and LHS baselines under the same 80-trial seed budget?

### Setup

- Method label: `trust_region_gp_qlogei`
- `N_INIT = 30`, `BUDGET = 50`, total `80` trials per seed
- Seeds included now: `0, 1, 2, 3, 4`
- Completion gate: `status == completed` and `num_trials >= 80`

### Current Reportable Result

From `experiments/turbo_online_general/campaign_summary.csv`, all 5 seeds
satisfy the completion gate.

| Seeds | Median final best | Mean final best | Best seed result | Worst seed result |
|---:|---:|---:|---:|---:|
| 5 | -19.161 | -17.524 | -22.517 | -9.206 |

### Evidence Artifacts

- `experiments/turbo_online_general/campaign_summary.csv`
- `experiments/turbo_online_general/turbo_online_general_seed0.json`
- `experiments/turbo_online_general/turbo_online_general_seed1.json`
- `experiments/turbo_online_general/turbo_online_general_seed2.json`
- `experiments/turbo_online_general/turbo_online_general_seed3.json`
- `experiments/turbo_online_general/turbo_online_general_seed4.json`

## Experiment Card: BO/LHS/Turbo Online Comparison

- experiment_id: `bo_lhs_turbo_online_comparison_v1`
- status: `paper-candidate`
- primary notebook: `notebooks/online_lhsdb_warmup_comparison.ipynb`
- output root: `experiments/bo_lhs_turbo_online_comparison/`

### Research Question

Do the current surrogate-assisted BO/Turbo policies outperform the model-free
LHS baseline under the same completed five-seed online protocol?

### Setup

- Methods: BO `global_gp_qlogei`, LHS `lhs_random`, Turbo `trust_region_gp_qlogei`
- Seeds: `0, 1, 2, 3, 4`
- Inclusion gate: `status == completed` and `num_trials >= 80` for every method
- Primary signals: final `best_objective`, per-seed winner, and best-so-far curves

### Current Reportable Result

From `bo_lhs_turbo_final_summary.csv`, LHS has the best median final objective
and wins 3 of 5 seeds. BO wins seed `1`; BO and Turbo tie on seed `3`.

| Method | Seeds | Median final best | Mean final best | Best seed result | Worst seed result |
|---|---:|---:|---:|---:|---:|
| BO | 5 | -14.657 | -15.649 | -23.814 | -8.052 |
| LHS | 5 | -19.162 | -18.815 | -25.141 | -9.847 |
| Turbo | 5 | -19.161 | -17.524 | -22.517 | -9.206 |

This result does not support a claim that the current BO or Turbo setup
outperforms LHS. It should be reported as a limitation of the current optimizer
configuration and a motivation for controlled replay or broader-seed follow-up,
not as evidence that optimization is generally ineffective.

### Best-Candidate Response Check

The response-shape check notebook
`notebooks/bclf_best_candidate_sp_range_compare.ipynb` compares the global best
candidate from each method using cached Touchstone responses. LHS has the best
scalar objective among these three best candidates.

| Method | Best seed | Iteration | Objective | Passband S21 min | Passband S11 max | Low-stop S21 max | High-stop S21 max |
|---|---:|---:|---:|---:|---:|---:|---:|
| BO | 1 | 38 | -23.814 | -1.440 | -7.518 | -19.487 | -16.328 |
| LHS | 0 | 23 | -25.141 | -1.020 | -9.372 | -18.932 | -15.646 |
| Turbo | 3 | 59 | -22.517 | -1.759 | -6.419 | -23.194 | -18.704 |

BO and LHS satisfy the current passband/stopband thresholds. Turbo gives strong
stopband attenuation, but its passband `S11` maximum is slightly above the
`-7 dB` target line.

### Evidence Artifacts

- `experiments/bo_lhs_turbo_online_comparison/bo_lhs_turbo_final_summary.csv`
- `experiments/bo_lhs_turbo_online_comparison/bo_lhs_turbo_per_seed_final.csv`
- `experiments/bo_lhs_turbo_online_comparison/bo_lhs_turbo_best_so_far_by_iteration.csv`
- `studies/online_comparison/figures/bo_lhs_turbo_online_comparison/bo_lhs_turbo_best_so_far_median.png`
- `studies/online_comparison/figures/bo_lhs_turbo_online_comparison/bo_lhs_turbo_final_best_by_seed.png`
- `studies/online_comparison/figures/bo_lhs_turbo_online_comparison/best_candidate_sp_range/bclf_best_candidate_s11_s21_fullband.png`
- `studies/online_comparison/figures/bo_lhs_turbo_online_comparison/best_candidate_sp_range/bclf_best_candidate_band_metric_range_bars.png`
- `experiments/bo_lhs_turbo_online_comparison/best_candidate_sp_range/bclf_best_candidate_band_metrics.csv`

## Experiment Card: BNN Offline Response-Surrogate Feasibility

- experiment_id: `bclf_bnn_offline_surrogate_v1`
- status: `paper-candidate`
- primary notebook: `notebooks/bnn_bclf_offline_feasibility.ipynb`
- output root: `experiments/bnn_bclf_offline_feasibility/`

### Research Question

Can a Bayesian neural network learn BCLF `S11` and `S21` response curves well
enough to justify optimizer-side response-surrogate experiments?

### Setup

- Model: `modeling.LegacyBNNResponseSurrogate`
- Target mode: direct dB response prediction
- Split: 80 percent train, 20 percent test
- Seed: `0`

### Current Reportable Result

From `holdout_accuracy_metrics.csv`, the response surrogate fits training
responses strongly and retains useful but imperfect held-out objective ranking.

| Split | Samples | Objective MAE | Objective Spearman | S11 MAE | S21 MAE | Top-10 recall |
|---|---:|---:|---:|---:|---:|---:|
| train | 126 | 5.84 | 0.97 | 0.16 dB | 1.39 dB | 0.90 |
| test | 31 | 22.63 | 0.38 | 0.81 dB | 2.97 dB | 0.60 |

### Evidence Artifacts

- `experiments/bnn_bclf_offline_feasibility/holdout_accuracy_metrics.csv`
- `studies/bnn_surrogate/figures/bnn_bclf_offline_feasibility/holdout_train_test_objective_true_vs_pred.png`
- `studies/bnn_surrogate/figures/bnn_bclf_offline_feasibility/holdout_best_response_prediction.png`

## Experiment Card: Few-Shot BNN Online-Learning Replay

- experiment_id: `bclf_bnn_fewshot_online_learning_v1`
- status: `paper-candidate`
- primary notebook: `notebooks/bnn_bclf_fewshot_online_learning.ipynb`
- output root: `experiments/bnn_bclf_fewshot_online_learning/`

### Research Question

Does warm-start BNN training improve response/objective prediction as simulated
online samples grow from 30 to 80?

### Setup

- Offline replay over cached BCLF responses
- Initial training set: 30 samples
- First fit: 800 epochs
- Incremental updates: 20 epochs per added sample
- Compared variants: raw frequency input and Fourier frequency features

### Current Reportable Result

From `fewshot_metrics_by_variant.csv`, warm-start updates improve ranking and
response prediction. The Fourier variant gives the strongest final test
objective MAE and S21 MAE among the two BNN variants.

| Variant | Test point | Train samples | Objective MAE | Objective Spearman | S21 MAE | Top-10 recall |
|---|---|---:|---:|---:|---:|---:|
| raw | initial | 30 | 31.31 | 0.26 | 3.79 dB | 0.10 |
| raw | final | 80 | 30.60 | 0.43 | 2.89 dB | 0.30 |
| Fourier | initial | 30 | 30.49 | 0.21 | 3.18 dB | 0.20 |
| Fourier | final | 80 | 25.05 | 0.44 | 2.37 dB | 0.30 |

### Evidence Artifacts

- `experiments/bnn_bclf_fewshot_online_learning/fewshot_metrics_by_variant.csv`
- `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_learning_curves_by_variant.png`
- `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_objective_scatter_final_by_variant.png`

## Experiment Card: GP vs BNN Response-Shape Metric

- experiment_id: `gp_vs_bnn_response_shape_metric_v1`
- status: `paper-candidate`
- primary notebook: `notebooks/gp_vs_bnn_response_shape_metric.ipynb`
- output root: `experiments/gp_vs_bnn_response_shape_metric/`

### Research Question

How do direct GP metric prediction, raw BNN response prediction, and Fourier BNN
response prediction compare on response-shape metric estimation?

### Setup

- Train sizes: `10`, `20`, `30`, `40`
- Seeds: `0, 1, 2, 3, 4`
- Methods: direct GP metric, raw BNN response metric, Fourier BNN response metric
- Primary signals: metric MAE, Spearman rank correlation, top-10 recall

### Current Reportable Result

From `summary_by_seed_size_method.csv`, Fourier BNN is competitive with the GP
metric predictor in MAE and has stronger mean Spearman correlation at train size
40. Raw BNN is fragile at train size 10.

| Train size | Method | Mean metric MAE | Mean Spearman | Mean top-10 recall |
|---:|---|---:|---:|---:|
| 10 | GP direct metric | 7.35 | 0.16 | 0.08 |
| 10 | raw BNN response | 72.48 | 0.01 | 0.00 |
| 10 | Fourier BNN response | 7.32 | 0.23 | 0.06 |
| 40 | GP direct metric | 6.64 | 0.21 | 0.08 |
| 40 | raw BNN response | 8.56 | 0.26 | 0.04 |
| 40 | Fourier BNN response | 6.68 | 0.31 | 0.10 |

### Evidence Artifacts

- `experiments/gp_vs_bnn_response_shape_metric/summary_by_seed_size_method.csv`
- `experiments/gp_vs_bnn_response_shape_metric/topk_summary.csv`
- `studies/bnn_surrogate/figures/gp_vs_bnn_response_shape_metric/metric_error_by_train_size.png`

## Experiment Card: BNN EI Replay Optimizer

- experiment_id: `bnn_ei_replay_optimizer_v1`
- status: `paper-candidate`
- primary notebook: `notebooks/bnn_ei_replay_experiment.ipynb`
- output root: `experiments/bnn_ei_replay/`

### Research Question

Can the BNN response-surrogate acquisition backend run inside the same optimizer
loop as GP logEI, and how does it compare in replay?

### Setup

- Methods: GP logEI and BNN EI
- Seeds: `0, 1, 2, 3, 4`
- Target trials: `48` per method/seed
- Completion gate: `status == completed` and `num_trials == target_trials`

### Current Reportable Result

From `comparison_summary.csv`, BNN EI is operational and ties or beats GP on
some seeds, but GP has stronger median and mean final best objective.

| Method | Seeds | Median final best | Mean final best | Best | Worst |
|---|---:|---:|---:|---:|---:|
| GP logEI | 5 | -22.77 | -21.80 | -23.85 | -16.43 |
| BNN EI | 5 | -22.08 | -19.55 | -22.77 | -8.07 |

### Evidence Artifacts

- `experiments/bnn_ei_replay/comparison_summary.csv`
- `experiments/bnn_ei_replay/comparison_runs.csv`
- `studies/bnn_acquisition/figures/bnn_ei_replay/best_so_far_by_method.png`

## Experiment Card: Acquisition-Quality Snapshot

- experiment_id: `acquisition_quality_gp_vs_bnn_v1`
- status: `paper-candidate`
- primary notebook: `notebooks/acquisition_quality_gp_vs_bnn.ipynb`
- output roots:
  - `experiments/acquisition_quality_gp_vs_bnn/`
  - `experiments/acquisition_quality_gp_qlogei_vs_bnn_qlogei/`

### Research Question

When GP and BNN acquisitions are evaluated on the same offline snapshots, which
method selects lower-objective candidates more often?

### Setup

- Snapshot count: `200`
- Reported logEI comparison: `gp_qlogei` vs `bnn_qlogei`
- Primary signals: selected objective, rank percentile, improvement rate, regret

### Current Reportable Result

From `acquisition_quality_gp_qlogei_vs_bnn_qlogei/acquisition_quality_summary.csv`,
BNN logEI has weaker selection quality than GP logEI in the current snapshot
study. This helps explain why BNN replay optimization has not consistently
beaten GP.

| Method | Snapshots | Median selected objective | Median rank percentile | Improvement rate | Median regret |
|---|---:|---:|---:|---:|---:|
| GP logEI | 200 | 9.60 | 0.22 | 0.055 | 33.45 |
| BNN logEI | 200 | 27.39 | 0.46 | 0.045 | 51.24 |

### Evidence Artifacts

- `experiments/acquisition_quality_gp_qlogei_vs_bnn_qlogei/acquisition_quality_summary.csv`
- `experiments/acquisition_quality_gp_qlogei_vs_bnn_qlogei/acquisition_quality_rows.csv`
- `studies/bnn_acquisition/figures/acquisition_quality_gp_qlogei_vs_bnn_qlogei/rank_percentile_by_train_size.png`
- `studies/bnn_acquisition/figures/acquisition_quality_gp_qlogei_vs_bnn_qlogei/selected_objective_by_train_size.png`

## Experiment Card: BNN Online Campaign

- experiment_id: `bnn_online_general_v1`
- status: `paper-candidate`
- primary notebook: `notebooks/bnn_online_experiment.ipynb`
- output root: `experiments/bnn_online_general/`

### Research Question

Can the response-surrogate BNN acquisition improve online BCLF optimization
relative to BO, LHS, and Turbo under the same queue-backed protocol?

### Setup

- Method label: `bnn_response_qlogei`
- `N_INIT = 30`, `BUDGET = 50`, total `80` trials per seed
- First BNN fit: 800 epochs after the initial 30 cases
- Incremental learning: warm-start 20 epochs after each new online case
- Seeds: `[0, 1, 2, 3, 4]`

### Current Reportable Result

From `experiments/bnn_online_general/campaign_summary.csv`, all five seeds
satisfy `status == completed` and `num_trials == 80`.

| Seed | Best objective | Cache hit ratio | Updated at |
|---:|---:|---:|---|
| 0 | -24.9994 | 0.7250 | 2026-06-08T05:40:41Z |
| 1 | -19.1619 | 0.6000 | 2026-06-08T05:40:41Z |
| 2 | -17.3415 | 0.6625 | 2026-06-08T05:40:41Z |
| 3 | -16.2997 | 0.6125 | 2026-06-08T08:42:02Z |
| 4 | -9.2064 | 0.6125 | 2026-06-08T13:15:14Z |

Five-seed aggregate:

| Method | Seeds | Median final best | Mean final best | Best seed result | Worst seed result |
|---|---:|---:|---:|---:|---:|
| BNN online_general | 5 | -17.3415 | -17.4018 | -24.9994 | -9.2064 |
| BO online_general | 5 | -14.6571 | -15.6494 | -23.8139 | -8.0524 |
| LHS online_general | 5 | -19.1619 | -18.8150 | -25.1413 | -9.8473 |
| Turbo online_general | 5 | -19.1610 | -17.5235 | -22.5171 | -9.2064 |

The completed BNN campaign is better than BO by median final best, but it does
not outperform LHS or Turbo under the current five-seed online-general
comparison. The supported claim is therefore a completed BNN baseline and a
negative/limited performance result, not BNN online superiority.

### Evidence Artifacts

- `notebooks/bnn_online_experiment.ipynb`
- `docs/academic/research_plans/bclf_bnn_optimization_plan.md`
- `tests/optimization/test_bnn_botorch_optimizer.py`
- `experiments/bnn_online_general/campaign_summary.csv`
- `experiments/bnn_online_general/bnn_online_general_seed0.csv`
- `experiments/bnn_online_general/bnn_online_general_seed1.csv`
- `experiments/bnn_online_general/bnn_online_general_seed2.csv`
- `experiments/bnn_online_general/bnn_online_general_seed3.csv`
- `experiments/bnn_online_general/bnn_online_general_seed4.csv`

## Experiment Card: OWAFS Previous-LHS Warmup Online Diagnostic

- experiment_id: `owafs_online_lhsdb_warmup_v1`
- status: `paper-candidate`
- primary scripts:
  - `studies/online_owafs/scripts/plot_owafs_geometry_coverage.py`
  - `studies/online_comparison/scripts/compare_owafs_vs_bo_lhs_bnn.py`
  - `studies/online_comparison/scripts/plot_owafs_vs_bo_lhs_bnn_objective.py`
  - `studies/online_owafs/scripts/summarize_owafs_experiment_status.py`
- output root: `experiments/online_lhsdb_warmup/owafs_bnn/`

### Research Question

Can Objective-Weighted + Uncertainty-Aware AFS improve from a 15-point
previous-LHS warmup, and does the current objective-weighted online policy
expand into previously unexplored geometry regions?

### Setup

- Method label: Objective-Weighted + Uncertainty-Aware AFS, stored as
  `owafs_bnn`
- Initial design: 15 previous-LHS warmup points per seed
- Budgeted trials: `80` per seed, with validation rows filtered from budgeted
  trajectories
- Completed OWAFS seeds: `[0, 1, 2, 3, 4]`
- Baseline comparison:
  - BNN: `experiments/bnn_online_general/bnn_online_general_seed*.csv`, complete seeds `0..4`
  - BO: `experiments/bo_online_general/bo_online_general_seed*.csv`, complete seeds `0..4`
  - LHS: `experiments/lhs_online_general/lhs_online_general_seed*.csv`, complete seeds `0..4`
  - Turbo: `experiments/turbo_online_general/turbo_online_general_seed*.csv`, complete seeds `0..4`
- Explicit exclusion: do not use previous-LHS, `online_lhs15_shared`,
  `lhs15_shared`, or `previous_lhs` BNN artifacts as the BNN baseline.

The seed count is now five for OWAFS and the online-general references, but the
initialization is still not paired: OWAFS starts from 15 previous-LHS warmup
points, while online-general BO/LHS/Turbo/BNN use the 30-point online-general
initialization.

### Current Reportable Result

From `owafs_geometry_coverage_summary_seed0_4.csv`, all five OWAFS seeds
improve over their own 15-point previous-LHS warmup:

| Seed | Warmup best | OWAFS best | Best iter | Improvement | Median dist-to-LHS | Max dist-to-LHS |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -16.43 | -22.08 | 44 | 5.64 | 0.69 | 1.04 |
| 1 | -11.47 | -17.41 | 32 | 5.95 | 0.66 | 0.97 |
| 2 | 0.02 | -6.80 | 41 | 6.82 | 0.67 | 1.03 |
| 3 | -2.06 | -23.82 | 18 | 21.76 | 0.73 | 1.09 |
| 4 | 6.64 | -16.80 | 26 | 23.44 | 0.71 | 1.11 |

Combining `campaign_summary.csv` from BNN/BO/LHS/Turbo with the completed
OWAFS warmup summary, the current descriptive pooled comparison is:

| Method | Runs | Seeds | Median final best | Mean final best | Best run | Worst run |
|---|---:|---|---:|---:|---:|---:|
| BNN | 5 | 0,1,2,3,4 | -17.34 | -17.40 | -25.00 | -9.21 |
| LHS | 5 | 0,1,2,3,4 | -19.16 | -18.81 | -25.14 | -9.85 |
| Turbo | 5 | 0,1,2,3,4 | -19.16 | -17.52 | -22.52 | -9.21 |
| OWAFS | 5 | 0,1,2,3,4 | -17.41 | -17.38 | -23.82 | -6.80 |
| BO | 5 | 0,1,2,3,4 | -14.66 | -15.65 | -23.81 | -8.05 |

The supported interpretation is narrow. OWAFS improves the previous-LHS warmup
for all five seeds, and its median final best is better than the five-seed BO
median. It is essentially tied with the BNN median, slightly weaker by mean, and
does not outperform LHS or Turbo. Seed `2` remains the weak OWAFS run. The
geometry diagnostic shows median distances of about `0.66-0.73` from the
original 15 warmup points, so the online samples are
not exact warmup duplicates; nevertheless, the distance curves do not show
sustained expansion over iteration, and the best points appear in the middle of
the budget rather than as late exploration discoveries.

### Claim Boundary

Supported:
- OWAFS can improve from its own 15-point previous-LHS warmup.

Partially supported:
- OWAFS is better than BO and close to BNN under the current descriptive
  comparison, but initialization differs and stability is weak.

Not supported:
- OWAFS superiority over BNN, LHS, or Turbo.
- Stronger global geometry coverage or sustained unknown-region expansion.

### Future Experiment Plan

| Experiment | Priority | Purpose |
|---|---|---|
| Same-warmup BNN baseline | P0 | Run `online_lhsdb_warmup/bnn` seeds `0..4` with the exact same 15 previous-LHS warmup points as OWAFS. |
| Exploration ablation | P1 | Compare current, more-exploratory, and coverage-regularized OWAFS variants using final best, best-so-far curves, and nearest-distance curves. |
| Objective-weighting ablation | P1 | Disable or weaken objective weighting to test whether seed `2` collapses from early exploitation. |
| Warmup fairness ablation | P2 | Repeat with different previous-LHS warmup draws, such as offsets `100` and `200`. |
| Validation consistency | P2 | Run full validation for best candidates and report budgeted best separately from validation best. |

### Evidence Artifacts

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
- `experiments/online_lhsdb_warmup/owafs_bnn/comparison/owafs_vs_bo_lhs_bnn_ranked_runs.csv`
- `experiments/online_lhsdb_warmup/owafs_bnn/comparison/owafs_vs_bo_lhs_bnn_best_so_far_by_iteration.csv`
- `studies/online_owafs/figures/online_lhsdb_warmup/owafs_geometry_coverage_pca_seed0_4.pdf`
- `studies/online_owafs/figures/online_lhsdb_warmup/owafs_geometry_distance_to_lhs_seed0_4.pdf`
- `studies/online_comparison/figures/online_lhsdb_warmup/owafs_vs_bo_bnn_best_so_far_objective.pdf`
- `studies/online_comparison/figures/online_lhsdb_warmup/owafs_vs_bo_bnn_final_best_objective.pdf`
- `studies/online_owafs/reports/OWAFS_EXPERIMENT_STATUS_AND_NEXT_STEPS.md`

## Experiment Card: AFS VF Reconstruction Sampling Diagnostic

- experiment_id: `uaw_afs_ext_offline_reproduction_v1`
- status: `draft`
- primary notebook: `notebooks/uaw_afs_ext_offline_reproduction.ipynb`
- output root: `experiments/uaw_afs_ext_offline_reproduction/`

### Paper-Facing Claim

For BCLF response reconstruction, Uncertainty-Aware AFS, stored internally as
`uaw`, is not well aligned with vector-fitting error. Objective-Weighted AFS,
stored internally as `oracle_vf_error`, remains a retrospective diagnostic
because it uses full-wave truth during frequency selection. The newer
Objective-Weighted + Uncertainty-Aware AFS selector, stored internally as
`owafs_bnn`, is deployable in the offline replay sense: it does not use
full-wave truth to choose frequencies, and instead combines BNN proxy VF
residuals with BNN Monte Carlo uncertainty. In the latest replay,
Objective-Weighted + Uncertainty-Aware AFS is strongest for preserving the
scalar BCLF objective, but not for full-curve dB reconstruction error.

### Protocol

- Data: offline replay over the cached BCLF `ext` Touchstone dataset.
- Recorded latest sweeps: seeds `[0, 1, 2, 3, 4]`; each
  sample-count/method summary aggregates `250` replay cases.
- Frequency grid: `401` full-wave points from `3.5` to `8.7 GHz`.
- Sampling-count sweep: every integer `n = 4..20`.
- VF order policies:
  - fixed-order baseline: `4` real poles plus `8` complex pole pairs, i.e.
    `20` actual poles and matching residues for every `n`.
  - order-matched rerun: `actual_poles = floor(n / 2)`, with mostly
    complex-conjugate pairs plus one real pole when needed, so
    `poles + residues <= n`.
- Methods:
  - `uniform`: deterministic endpoint-inclusive uniform frequency selection.
  - `random`: endpoint-inclusive random interior frequency selection.
  - Uncertainty-Aware AFS, stored internally as `uaw`:
    endpoint-inclusive uncertainty-weighted random selection using BNN MC
    variance in `S11` and `S21` dB.
  - Objective-Weighted AFS, stored internally as `oracle_vf_error`:
    retrospective adaptive selector using the full-wave truth to add points
    where current VF reconstruction error is largest after objective weighting.
  - Objective-Weighted + Uncertainty-Aware AFS, stored internally as
    `owafs_bnn`: deployable adaptive selector using BNN mean-response VF
    residuals and BNN MC uncertainty, without consulting the full-wave
    response during frequency selection.

Objective-Weighted AFS starts from endpoints, BCLF band-boundary anchors, and
uniform-fill anchors, then repeatedly fits VF on the sampled points and adds
the unsampled frequency with the largest objective-weighted residual:
`0.5 * |S11_true_dB - S11_vf_dB| + |S21_true_dB - S21_vf_dB|`. Boundary,
passband, and stopband weights emphasize objective-relevant regions. Because
the full-wave truth is used during selection, this method is only an upper
bound for method design.

Objective-Weighted + Uncertainty-Aware AFS uses the same anchor policy but
replaces full-truth residuals with a BNN proxy score:
`band_weight * 0.5 * (rank01(vf_residual_db) + rank01(mc_uncertainty_db))`.
The VF residual term compares the BNN mean response with its own VF
reconstruction, and the uncertainty term comes from BNN MC samples. The true
full response is used only after selection to evaluate the reconstructed
response and scalar objective in this offline diagnostic.

### Result

The current fixed-order, order-matched, and deployable
Objective-Weighted + Uncertainty-Aware AFS sweeps all pass their acceptance
checks. The fixed-order and order-matched sweeps each contain `17000/17000`
expected rows and `68/68` summarized sample-count/method pairs. The deployable
combined AFS sweep contains `21250/21250` expected rows and `85/85` summarized
pairs. Successful reconstructions have `401` output points, all reported
metrics are finite, and the deployable checks confirm that `owafs_bnn` rows
use BNN proxy residuals plus uncertainty rather than full-wave truth for
selection.

| Sweep | Metric | Best method | Best mean rank | Best wins | Uncertainty-Aware AFS mean rank | Objective-Weighted AFS mean rank | Objective-Weighted + Uncertainty-Aware AFS mean rank |
|---|---|---|---:|---:|---:|---:|---:|
| fixed 20-pole, 4 methods | median mean dB MAE | uniform | 1.71 | 12/17 | 2.94 | 2.88 | n/a |
| fixed 20-pole, 4 methods | median objective abs. error | Objective-Weighted AFS | 1.29 | 13/17 | 2.71 | 1.29 | n/a |
| order matched, 4 methods | median mean dB MAE | uniform | 1.82 | 10/17 | 3.00 | 3.06 | n/a |
| order matched, 4 methods | median objective abs. error | Objective-Weighted AFS | 1.59 | 11/17 | 1.94 | 1.59 | n/a |
| deployable combined AFS, 5 methods | median mean dB MAE | uniform | 1.94 | 10/17 | 2.94 | 3.18 | 4.24 |
| deployable combined AFS, 5 methods | median objective abs. error | Objective-Weighted + Uncertainty-Aware AFS | 1.47 | 11/17 | 2.88 | 1.94 | 1.47 |

Order matching tests whether the previous conclusion was caused by using too
many VF poles at small `n`. It does not make Uncertainty-Aware AFS dominant:
uniform remains best for full-curve dB MAE, and Objective-Weighted AFS remains
best for objective error in the four-method order-matched diagnostic. The new
deployable Objective-Weighted + Uncertainty-Aware AFS result is narrower but
more useful for online design: it has the best objective-error rank in the
five-method replay, including ahead of the truth-informed diagnostic under
this metric, while performing poorly on full-curve dB MAE. The supported
conclusion is therefore objective-specific: practical AFS should target
objective-relevant VF residuals and model uncertainty, but this evidence does
not claim best full-response reconstruction or completed online optimizer
superiority.

### Paper Figure

- figure_id: `uaw_afs_sampling_sweep_order_matched`
- layout policy: auto-selected IEEE double-column PDF, reason
  `multi_panel_dense_methods`
- PDF: `studies/uaw_afs/figures/paper_assets/uaw_afs_ext_offline_reproduction/uaw_afs_sampling_sweep_order_matched_ieee_double.pdf`
- preview: `studies/uaw_afs/figures/paper_assets/uaw_afs_ext_offline_reproduction/uaw_afs_sampling_sweep_order_matched_ieee_double.png`
- manifest: `docs/academic/paper/assets/uaw_afs_ext_offline_reproduction/paper_figure_manifest.csv`

Caption draft: Order-matched AFS reconstruction sweep on the BCLF offline
`ext` dataset. Each curve reports the median over `250` replay cases for
sampling budgets `n=4..20`; bands show the interquartile range.
Uniform and random sampling remain competitive for full-curve reconstruction
error, while the non-deployable Objective-Weighted AFS diagnostic best
preserves the BCLF objective among the four methods shown.
Uncertainty-Aware AFS does not dominate either baseline, indicating that
practical AFS should target VF residuals and objective-relevant frequency bands
rather than uncertainty alone.

### Evidence Artifacts

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
- `studies/uaw_afs/figures/uaw_afs_ext_offline_reproduction/plots/deployable_owafs_sample_count_vs_mean_db_mae.png`
- `studies/uaw_afs/figures/uaw_afs_ext_offline_reproduction/plots/deployable_owafs_sample_count_vs_objective_abs_error.png`
- `experiments/uaw_afs_ext_offline_reproduction/paper_figures/paper_figure_manifest.csv`
- `docs/academic/paper/assets/uaw_afs_ext_offline_reproduction/paper_figure_manifest.csv`
- `studies/uaw_afs/figures/paper_assets/uaw_afs_ext_offline_reproduction/uaw_afs_sampling_sweep_order_matched_ieee_double.pdf`
- `studies/uaw_afs/figures/paper_assets/uaw_afs_ext_offline_reproduction/uaw_afs_sampling_sweep_order_matched_ieee_double.png`
