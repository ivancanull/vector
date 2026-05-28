# BCLF BNN Optimization Plan

This plan tracks how to turn the current BCLF BNN surrogate work into an online
optimizer that can be compared fairly against BO and LHS.

The target is to outperform BO under the same online BCLF protocol, but that is
an exploration target, not an assumed result. Each stage should either improve
BNN performance, identify the limiting factor, or narrow the next experiment.

## Fixed Online Comparison Protocol

Use the current BO/LHS protocol in
[BCLF Optimization Target And Online Settings](bclf_optimization_target.md):

- objective: current `BCLFOptimizationSpec(...)`;
- design vector: `[L, L1, L2, W, W1, W2, S1, S2]`;
- variable bounds: first eight `data.regularization.min/max` entries from
  `configs/filter_bnn_online_uaw_uaw.yaml`;
- candidate space: `ContinuousCandidateSpace`;
- pool size: `1500`;
- geometry resolution: `0.01 mm`;
- seeds: `[0, 1, 2, 3, 4]`;
- target trials: `80` evaluated trials per seed;
- simulator backend: queue-backed `ServerQueueSimulator`;
- primary metric: `best_so_far` over iteration;
- artifact pattern: one JSON/CSV/checkpoint per seed plus campaign summary.

BNN results should be compared against:

| Method | Current artifact root | Primary signal |
|---|---|---|
| BO | `experiments/bo_online_general/` | global GP qLogEI `best_so_far` |
| LHS | `experiments/lhs_online_general/` | Latin-hypercube random `best_so_far` |
| BNN | `experiments/bnn_online_general/` | BNN acquisition `best_so_far` |

## BNN Online Baseline Settings

The first queue-backed BNN run should be a simple online baseline, not an
offline-pretrained model:

- `N_INIT = 30`;
- `BUDGET = 50`, so each seed reaches `80` evaluated trials;
- `RETRAIN_EVERY = 1`, so the BNN is updated after every completed online case
  once the initial design is finished;
- `epochs = 800` for the first BNN fit;
- `update_epochs = 20` for each warm-start incremental update;
- `warm_start = True`;
- `initial_train_size = 30`;
- `acquisition = "qlogei"`;
- `mc_samples = 64`.

Training schedule:

| Phase | Iterations | Completed cases available | BNN training behavior |
|---|---:|---:|---|
| Initial design | `0` to `29` | grows from `1` to `30` | no BNN-guided selection; collect online response arrays |
| First BNN fit | before iteration `30` | `30` | train from scratch for `800` epochs |
| Incremental online learning | `30` to `79` | grows from `31` to `80` | warm-start from the existing BNN and train `20` epochs after each new case |

This schedule is the baseline contract for `notebooks/bnn_online_experiment.ipynb`.
If later experiments change the training schedule, record the hypothesis and
only change one factor at a time.

## Current Offline Work

### 1. 80/20 BNN Response Surrogate Feasibility

Source:

- notebook: `notebooks/bnn_bclf_offline_feasibility.ipynb`
- artifacts: `experiments/bnn_bclf_offline_feasibility/`
- summary doc: [BNN Surrogate Experiments](bnn_surrogate_experiments.md)

What is implemented:

- `LegacyBNNResponseSurrogate` predicts `S11`/`S21` response curves in dB.
- Training uses geometry plus frequency as input.
- Objective is computed from predicted response curves, not learned directly as
  only a scalar.

Current result:

| Split | Samples | Objective MAE | Objective Spearman | S11 MAE | S21 MAE | Top-10 recall |
|---|---:|---:|---:|---:|---:|---:|
| train | 126 | 5.84 | 0.97 | 0.16 dB | 1.39 dB | 0.90 |
| test | 31 | 22.63 | 0.38 | 0.81 dB | 2.97 dB | 0.60 |

Interpretation:

- The response surrogate is viable enough for optimizer-side experiments.
- Generalization is still imperfect, especially for objective ranking on held-out
  cases.
- This is surrogate validation only; it is not an optimization win.

### 2. Few-Shot Offline Online-Learning Simulation

Source:

- notebook: `notebooks/bnn_bclf_fewshot_online_learning.ipynb`
- artifacts: `experiments/bnn_bclf_fewshot_online_learning/`
- summary doc: [BNN Surrogate Experiments](bnn_surrogate_experiments.md)

What is implemented:

- offline replay of online learning over cached BCLF responses;
- warm-start BNN updates as new samples are added;
- raw and Fourier-frequency response surrogate variants.

Current raw-surrogate result:

| Test point | Train samples | Objective MAE | Objective Spearman | S11 MAE | S21 MAE | Top-10 recall |
|---|---:|---:|---:|---:|---:|---:|
| initial | 30 | 34.74 | -0.07 | 1.15 dB | 4.55 dB | 0.10 |
| final | 80 | 26.21 | 0.23 | 0.83 dB | 2.47 dB | 0.20 |

Current raw vs Fourier result:

| Variant | Test point | Train samples | Objective MAE | Objective Spearman | S21 MAE | Top-10 recall |
|---|---|---:|---:|---:|---:|---:|
| raw | initial | 30 | 31.31 | 0.26 | 3.79 dB | 0.10 |
| raw | final | 80 | 30.60 | 0.43 | 2.89 dB | 0.30 |
| Fourier | initial | 30 | 30.49 | 0.21 | 3.18 dB | 0.20 |
| Fourier | final | 80 | 25.05 | 0.44 | 2.37 dB | 0.30 |

Interpretation:

- Warm-start updates work.
- Fourier frequency features look better than raw frequency features in this
  replay setting.
- Ranking quality improves but is still not strong enough to claim online
  optimizer superiority.

### 3. GP vs Raw/Fourier BNN Response-Shape Metric

Source:

- notebook: `notebooks/gp_vs_bnn_response_shape_metric.ipynb`
- artifacts: `experiments/gp_vs_bnn_response_shape_metric/`
- summary doc:
  [BCLF Response-Shape BNN Experiments](bclf_response_shape_bnn_experiments.md)

What is implemented:

- `BCLFResponseShapeOptimizationSpec` for shape-level response evaluation;
- direct GP scalar metric baseline;
- raw BNN and Fourier BNN response surrogates;
- checkpoint loading for repeated train-size sweeps.

Current aggregate signal:

| Train size | Method | Mean metric MAE | Mean Spearman | Mean top-10 recall |
|---:|---|---:|---:|---:|
| 10 | GP direct metric | 7.35 | 0.16 | 0.08 |
| 10 | raw BNN response | 72.48 | 0.01 | 0.00 |
| 10 | Fourier BNN response | 7.32 | 0.23 | 0.06 |
| 40 | GP direct metric | 6.64 | 0.21 | 0.08 |
| 40 | raw BNN response | 8.56 | 0.26 | 0.04 |
| 40 | Fourier BNN response | 6.68 | 0.31 | 0.10 |

Interpretation:

- Raw BNN is fragile at very small training sizes.
- Fourier BNN is competitive with the GP metric predictor on response-shape MAE
  and has stronger rank correlation at larger train sizes.
- This is the strongest offline signal for using Fourier response features in a
  future online BNN optimizer.

### 4. BNN EI Replay Optimizer

Source:

- notebook: `notebooks/bnn_ei_replay_experiment.ipynb`
- artifacts: `experiments/bnn_ei_replay/`
- implementation: `optimization.BNNExpectedImprovementBackend`

What is implemented:

- `OptimizerBoTorch` can use `BNNExpectedImprovementBackend`;
- the backend trains a response BNN from optimizer history;
- predicted response samples are scored through the same `OptimizationSpec`;
- acquisition supports `qlogei`, `qei`/`ei`, and `lcb`;
- tests cover warm-start behavior and replay execution without repeats.

Current replay result:

| Method | Seeds | Median best | Mean best | Best | Worst | Median runtime |
|---|---:|---:|---:|---:|---:|---:|
| GP qLogEI | 5 | -22.77 | -21.80 | -23.85 | -16.43 | 29.41 s |
| BNN EI | 5 | -22.08 | -19.55 | -22.77 | -8.07 | 45.01 s |

Seed-level outcome:

- BNN tied GP on seeds `0` and `3`;
- BNN beat GP on seed `1`;
- GP beat BNN on seeds `2` and `4`;
- GP is better on median and mean final best objective.

Interpretation:

- The optimizer path exists and can be reused for online work.
- Current BNN EI does not yet beat GP qLogEI in replay.
- Runtime is higher for BNN because repeated response-surrogate training and
  sampling are expensive.

### 5. Acquisition-Quality Snapshot Comparisons

Source:

- notebook: `notebooks/acquisition_quality_gp_vs_bnn.ipynb`
- artifacts:
  - `experiments/acquisition_quality_gp_vs_bnn/`
  - `experiments/acquisition_quality_gp_qlogei_vs_bnn_qlogei/`

Current result:

| Comparison | Method | Median selected objective | Median rank percentile | Improvement rate | Median regret |
|---|---|---:|---:|---:|---:|
| EI-style | GP qLogEI | 13.37 | 0.34 | 0.060 | 37.22 |
| EI-style | BNN EI | 23.75 | 0.53 | 0.035 | 46.52 |
| qLogEI-style | GP qLogEI | 9.60 | 0.22 | 0.055 | 33.45 |
| qLogEI-style | BNN qLogEI | 27.39 | 0.46 | 0.045 | 51.24 |

Interpretation:

- BNN acquisition is currently weaker than GP acquisition on the offline
  snapshot task.
- This explains why replay optimization has not consistently beaten GP.
- Online work should start from the existing backend but improve the selection
  strategy before spending many queue-backed evaluations.

## Current Boundary

### Already Implemented

- response-level BNN surrogate training and prediction;
- warm-start online-learning simulation;
- raw and Fourier response surrogate variants;
- response-shape metric evaluation;
- BNN acquisition backend integrated into `OptimizerBoTorch`;
- full online BNN notebook scaffold with the fixed baseline training schedule;
- replay optimizer comparison against GP qLogEI;
- acquisition-quality diagnostics.

Concrete code/notebook pieces already present:

| Area | Existing artifact |
|---|---|
| BNN response surrogate | `modeling.LegacyBNNResponseSurrogate` |
| Fourier response surrogate | `modeling.FourierBNNResponseSurrogate` |
| BNN response dataset | `modeling.BCLFResponseDataset` |
| Response-shape objective | `optimization.BCLFResponseShapeOptimizationSpec` |
| Optimizer-side BNN acquisition | `optimization.BNNExpectedImprovementBackend` |
| BO-compatible optimizer loop | `optimization.OptimizerBoTorch` with pluggable acquisition backend |
| LHS baseline | `optimization.OptimizerLHSRandom` |
| Offline surrogate feasibility notebook | `notebooks/bnn_bclf_offline_feasibility.ipynb` |
| Few-shot BNN update notebook | `notebooks/bnn_bclf_fewshot_online_learning.ipynb` |
| GP vs BNN response-shape notebook | `notebooks/gp_vs_bnn_response_shape_metric.ipynb` |
| BNN EI replay notebook | `notebooks/bnn_ei_replay_experiment.ipynb` |
| BNN online notebook | `notebooks/bnn_online_experiment.ipynb` |
| Acquisition-quality notebook | `notebooks/acquisition_quality_gp_vs_bnn.ipynb` |
| Core BNN acquisition tests | `tests/optimization/test_bnn_botorch_optimizer.py` |

### Needs Implementation

These are the remaining implementation pieces after the first online notebook:

| Priority | Item | Notes |
|---:|---|---|
| 1 | Configurable BNN acquisition surrogate | Current `BNNExpectedImprovementBackend` uses raw `LegacyBNNResponseSurrogate`; add Fourier selection if online or replay evidence says this is the next limiting factor. |
| 1 | Current-objective re-score/rerun path | Some older offline artifacts used earlier BCLF bands; final BNN decisions need current `BCLFOptimizationSpec` settings. |
| 3 | Optional offline pretraining path | Only add after the simple online baseline is running. |

### Needs Running

Run these in order; do not jump straight to full online unless the earlier
checks pass.

| Order | Run | Purpose | Expected output |
|---:|---|---|---|
| 1 | Unit tests for BNN acquisition | Verify current backend still works after edits. | `pytest tests/optimization/test_bnn_botorch_optimizer.py` |
| 2 | BNN online notebook parse/smoke | Validate notebook wiring, artifacts, checkpoint/resume, and no repeated candidates before spending queue time. | small `experiments/bnn_online_general/` smoke outputs if replay is used |
| 3 | Full online BNN campaign | Fair comparison against BO/LHS over seeds `[0, 1, 2, 3, 4]` with `30` initial cases, first `800`-epoch BNN fit, then `20`-epoch incremental updates. | full campaign summary and per-seed artifacts |
| 4 | Final comparison analysis | Decide whether BNN beats BO or needs another iteration. | BO/LHS/BNN `best_so_far` plot and table |

### Not Yet Proven

- BNN beats BO under the current BO/LHS online protocol;
- BNN acquisition chooses better candidates than GP qLogEI in offline snapshots;
- Fourier response features are wired into online BNN acquisition;
- completed BNN online campaign artifacts exist under
  `experiments/bnn_online_general/`.

Important caveat:

- Some older offline artifacts used earlier BCLF bands. Before making any
  final claim, rerun or re-score the selected BNN path with the current
  objective in `docs/bclf_optimization_target.md`.

## Online-Focused Execution Plan

### Stage 1: Lock BNN Online Baseline

Implement or update:

- `notebooks/bnn_online_experiment.ipynb`;
- artifact root `experiments/bnn_online_general/`;
- method label `bnn_response_qlogei`;
- `OptimizerBoTorch` with `BNNExpectedImprovementBackend`;
- same `ContinuousCandidateSpace`, simulator, seeds, pool size, resolution, and
  target trial count as BO/LHS;
- BNN training schedule fixed to first fit `800` epochs after the initial
  `30` cases, then warm-start `20` epochs after every incremental online case.

Exit condition:

- one smoke seed runs with a tiny budget on replay or queue cache without
  breaking checkpoint/resume.

### Stage 2: Online Candidate Strategy Search In Replay

Use replay only when the first online baseline needs diagnosis. The baseline
training schedule remains fixed unless this stage is explicitly about changing
the schedule.

Run a small replay grid over:

| Factor | Values |
|---|---|
| surrogate features | raw response, Fourier response |
| acquisition | `qlogei`, `qei`, `lcb` |
| initial training | production `N_INIT=30`, exploratory values only for replay |
| update policy | production first fit `800` epochs then `20`-epoch warm-start updates; exploratory alternatives one at a time |
| MC samples | low-cost smoke value, production value |

Exit condition:

- choose one online BNN configuration that improves over the current BNN EI
  replay baseline or has a clear reason to test online.

### Stage 3: Queue-Backed Single-Seed Online Run

Run seed `0` first with the full online target:

- `SEEDS = [0]`;
- target trials `80`;
- first BNN fit after `30` initial cases uses `800` epochs;
- each later incremental update uses `20` epochs;
- same simulator/cache/template settings as BO/LHS;
- save per-trial BNN diagnostics in `selection_debug`.

Decision rule:

- continue only if the BNN curve is competitive with BO or clearly better than
  LHS early enough to justify more simulator time.

### Stage 4: Full Online BNN Campaign

Run seeds `[0, 1, 2, 3, 4]` with the selected configuration:

- `N_INIT = 30`;
- `BUDGET = 50`;
- first BNN fit after the initial cases: `800` epochs;
- warm-start incremental learning: `20` epochs per new online case.

Required outputs:

- `experiments/bnn_online_general/campaign_summary.csv`;
- `experiments/bnn_online_general/bnn_online_general_seed{seed}.json`;
- `experiments/bnn_online_general/bnn_online_general_seed{seed}.csv`;
- `experiments/bnn_online_general/bnn_online_general_seed{seed}.checkpoint.json`;
- comparison plot/table against BO and LHS `best_so_far`.

Acceptance rule:

- BNN can be called better than BO only if it has lower median final
  `best_so_far` across the same seeds, or a clearly documented early-budget win
  such as lower median `best_so_far` by trial 40.

### Stage 5: Beat-BO Iteration Loop

If BNN does not beat BO, change one factor per round:

- Fourier response acquisition instead of raw response acquisition;
- acquisition mode and exploration strength;
- pretraining from cached EXT data before online updates;
- training schedule and warm-start policy;
- candidate shortlist size and ranking rule;
- response-shape objective vs current band objective.

For each round, record:

- hypothesis;
- changed setting;
- replay result;
- online result if run;
- keep/revert decision.

## Next Implementation Target

The next concrete task is to run `notebooks/bnn_online_experiment.ipynb` with
the full online defaults. The notebook uses `BNNExpectedImprovementBackend` as
the method-specific change and keeps the first version simple. Use the resulting
online evidence to decide whether to add Fourier acquisition or offline
pretraining.

## Notebook Execution

Jupyter execution is available in the current environment:

```bash
jupyter nbconvert --to notebook --execute notebooks/<name>.ipynb \
  --output <name>.executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

For programmatic execution, `nbformat` and `nbclient` are also installed.

Execution policy:

- smoke/replay notebooks can be run directly as validation;
- full offline notebooks can be run when runtime is acceptable;
- queue-backed online notebooks should only run with explicit intent because
  they submit real simulator jobs and may take a long time.
