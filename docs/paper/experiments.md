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
[BCLF Optimization Target And Online Settings](../bclf_optimization_target.md).
The optimizer minimizes scalar objective; lower `best_so_far` is better.

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
- primary notebook: `notebooks/bo_lhs_online_comparison.ipynb`
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
- `experiments/bo_lhs_turbo_online_comparison/bo_lhs_turbo_best_so_far_median.png`
- `experiments/bo_lhs_turbo_online_comparison/bo_lhs_turbo_final_best_by_seed.png`
- `experiments/bo_lhs_turbo_online_comparison/best_candidate_sp_range/bclf_best_candidate_s11_s21_fullband.png`
- `experiments/bo_lhs_turbo_online_comparison/best_candidate_sp_range/bclf_best_candidate_band_metric_range_bars.png`
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
- `experiments/bnn_bclf_offline_feasibility/holdout_train_test_objective_true_vs_pred.png`
- `experiments/bnn_bclf_offline_feasibility/holdout_best_response_prediction.png`

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
- `experiments/bnn_bclf_fewshot_online_learning/fewshot_learning_curves_by_variant.png`
- `experiments/bnn_bclf_fewshot_online_learning/fewshot_objective_scatter_final_by_variant.png`

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
- `experiments/gp_vs_bnn_response_shape_metric/metric_error_by_train_size.png`

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
- `experiments/bnn_ei_replay/best_so_far_by_method.png`

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
- `experiments/acquisition_quality_gp_qlogei_vs_bnn_qlogei/rank_percentile_by_train_size.png`
- `experiments/acquisition_quality_gp_qlogei_vs_bnn_qlogei/selected_objective_by_train_size.png`

## Experiment Card: BNN Online Campaign

- experiment_id: `bnn_online_general_v1`
- status: `draft`
- primary notebook: `notebooks/bnn_online_experiment.ipynb`
- planned output root: `experiments/bnn_online_general/`

### Research Question

Can the response-surrogate BNN acquisition improve online BCLF optimization
relative to BO, LHS, and Turbo under the same queue-backed protocol?

### Planned Setup

- Method label: `bnn_response_qlogei`
- `N_INIT = 30`, `BUDGET = 50`, total `80` trials per seed
- First BNN fit: 800 epochs after the initial 30 cases
- Incremental learning: warm-start 20 epochs after each new online case
- Seeds: `[0, 1, 2, 3, 4]`

### Current Status

The notebook and backend are implemented. The full queue-backed campaign is not
complete, so no BNN online performance claim should be made yet.

### Evidence Artifacts

- `notebooks/bnn_online_experiment.ipynb`
- `docs/bclf_bnn_optimization_plan.md`
- `tests/optimization/test_bnn_botorch_optimizer.py`

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
- PDF: `docs/paper/assets/uaw_afs_ext_offline_reproduction/uaw_afs_sampling_sweep_order_matched_ieee_double.pdf`
- preview: `docs/paper/assets/uaw_afs_ext_offline_reproduction/uaw_afs_sampling_sweep_order_matched_ieee_double.png`
- manifest: `docs/paper/assets/uaw_afs_ext_offline_reproduction/paper_figure_manifest.csv`

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
- `experiments/uaw_afs_ext_offline_reproduction/plots/deployable_owafs_sample_count_vs_mean_db_mae.png`
- `experiments/uaw_afs_ext_offline_reproduction/plots/deployable_owafs_sample_count_vs_objective_abs_error.png`
- `experiments/uaw_afs_ext_offline_reproduction/paper_figures/paper_figure_manifest.csv`
- `docs/paper/assets/uaw_afs_ext_offline_reproduction/paper_figure_manifest.csv`
- `docs/paper/assets/uaw_afs_ext_offline_reproduction/uaw_afs_sampling_sweep_order_matched_ieee_double.pdf`
- `docs/paper/assets/uaw_afs_ext_offline_reproduction/uaw_afs_sampling_sweep_order_matched_ieee_double.png`
