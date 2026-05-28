# Experiments

This page tracks paper-facing experiment families. Treat notebooks and files
under `experiments/` as the data-generation source of truth.

## Current Online Protocol

| Setting | BO | LHS | BNN online |
|---|---|---|---|
| Notebook | `notebooks/bo_online_experiment.ipynb` | `notebooks/lhs_online_experiment.ipynb` | `notebooks/bnn_online_experiment.ipynb` |
| Optimizer | `OptimizerBoTorch` | `OptimizerLHSRandom` | `OptimizerBoTorch` |
| Selection | GP qLogEI | LHS random sequence | BNN response qLogEI |
| Simulator | `ServerQueueSimulator` | `ServerQueueSimulator` | `ServerQueueSimulator` |
| Candidate space | `ContinuousCandidateSpace` | `ContinuousCandidateSpace` | `ContinuousCandidateSpace` |
| Pool size | `1500` | `1500` | `1500` |
| Seeds | `[0, 1, 2, 3, 4]` | `[0, 1, 2, 3, 4]` | `[0, 1, 2, 3, 4]` |
| Target trials | `80` | `80` | `80` |
| Output root | `experiments/bo_online_general/` | `experiments/lhs_online_general/` | `experiments/bnn_online_general/` |

All three online families use the current BCLF target in
[BCLF Optimization Target And Online Settings](../bclf_optimization_target.md).
The optimizer minimizes scalar objective; lower `best_so_far` is better.

## Experiment Card: BO Online Baseline

- experiment_id: `bo_online_general_v1`
- status: `paper-candidate`
- primary notebook: `notebooks/bo_online_experiment.ipynb`
- output root: `experiments/bo_online_general/`

### Research Question

Can GP qLogEI operate as a queue-backed online baseline for BCLF optimization
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
- status: `draft`
- primary notebook: `notebooks/lhs_online_experiment.ipynb`
- output root: `experiments/lhs_online_general/`

### Research Question

How does a model-free LHS random baseline perform under the same objective,
candidate-space, simulator, and per-seed budget contract as BO?

### Setup

- Method label: `lhs_random`
- Total budget: `80` trials per seed
- Current completed summary rows: seeds `0, 1, 2`
- Completion gate: `status == completed` and `num_trials >= 80`

### Current Reportable Result

From `experiments/lhs_online_general/campaign_summary.csv`, 3 seeds currently
satisfy the completion gate. This is a partial online baseline and should not be
used as a final fair comparison against BO until the configured seed set is
complete or the paper explicitly labels the comparison as partial.

| Completed seeds | Median final best | Mean final best | Best seed result | Worst seed result |
|---:|---:|---:|---:|---:|
| 3 | -19.162 | -20.206 | -25.141 | -16.314 |

### Evidence Artifacts

- `experiments/lhs_online_general/campaign_summary.csv`
- `experiments/lhs_online_general/lhs_online_general_seed0.json`
- `experiments/lhs_online_general/lhs_online_general_seed1.json`
- `experiments/lhs_online_general/lhs_online_general_seed2.json`

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
loop as GP qLogEI, and how does it compare in replay?

### Setup

- Methods: GP qLogEI and BNN EI
- Seeds: `0, 1, 2, 3, 4`
- Target trials: `48` per method/seed
- Completion gate: `status == completed` and `num_trials == target_trials`

### Current Reportable Result

From `comparison_summary.csv`, BNN EI is operational and ties or beats GP on
some seeds, but GP has stronger median and mean final best objective.

| Method | Seeds | Median final best | Mean final best | Best | Worst |
|---|---:|---:|---:|---:|---:|
| GP qLogEI | 5 | -22.77 | -21.80 | -23.85 | -16.43 |
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
- Reported qLogEI comparison: `gp_qlogei` vs `bnn_qlogei`
- Primary signals: selected objective, rank percentile, improvement rate, regret

### Current Reportable Result

From `acquisition_quality_gp_qlogei_vs_bnn_qlogei/acquisition_quality_summary.csv`,
BNN qLogEI has weaker selection quality than GP qLogEI in the current snapshot
study. This helps explain why BNN replay optimization has not consistently
beaten GP.

| Method | Snapshots | Median selected objective | Median rank percentile | Improvement rate | Median regret |
|---|---:|---:|---:|---:|---:|
| GP qLogEI | 200 | 9.60 | 0.22 | 0.055 | 33.45 |
| BNN qLogEI | 200 | 27.39 | 0.46 | 0.045 | 51.24 |

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
relative to BO and LHS under the same queue-backed protocol?

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
