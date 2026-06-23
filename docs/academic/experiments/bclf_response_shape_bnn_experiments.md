# BCLF Response-Shape BNN Experiments

This document summarizes the current BCLF response-shape surrogate experiments
and the optimizer-side BNN acquisition contract.

## Response-Shape Objective

`BCLFResponseShapeOptimizationSpec` evaluates the full response shape instead of
only worst/mean band metrics.

- `S21` is checked across the full frequency grid:
  lower stopband, passband, upper stopband, and linear transition regions.
- `S11` is checked in the passband by default.
- Violations use squared hinge aggregation, so the optimizer still sees one
  scalar objective where lower is better.
- The existing `BCLFOptimizationSpec` remains unchanged for old experiments.

Default BCLF thresholds used by the response-shape experiments:

```python
SPEC_KWARGS = {
    "passband_ghz": (4.8, 5.4),
    "lower_stopband_ghz": (3.6, 4.4),
    "upper_stopband_ghz": (5.8, 8.7),
    "min_passband_s21_db": -5.0,
    "max_passband_s11_db": -7.0,
    "max_stopband_s21_db": -15.0,
}
```

## Surrogate Variants

The raw BNN response surrogate is `LegacyBNNResponseSurrogate`. It uses:

```text
[geometry(8), raw frequency_hz]
```

The Fourier BNN response surrogate is `FourierBNNResponseSurrogate`. It uses:

```text
[geometry(8), f_norm, sin(2*pi*B*f_norm), cos(2*pi*B*f_norm)]
```

Default Fourier settings:

```python
FOURIER_KWARGS = {
    "fourier_bands": [1, 2, 4, 8, 16],
    "include_raw_frequency": True,
}
```

The Fourier variant defaults to BCLF frequency bounds `3.5e9..8.7e9` for
normalizing `f_norm`. If a dataset uses a different frequency grid, pass
`frequency_min_hz=dataset.freqs.min()` and `frequency_max_hz=dataset.freqs.max()`.

## GP vs Raw/Fourier BNN Response-Shape Metric

Notebook:

```text
notebooks/gp_vs_bnn_response_shape_metric.ipynb
```

Purpose:

- Compare a GP that directly predicts the scalar response-shape metric.
- Compare raw BNN and Fourier BNN response surrogates that predict S11/S21 curves
  first, then compute the metric from predicted responses.
- Evaluate prediction accuracy on the remaining test set.
- Compare selected top-5/top-10 cases by predicted metric using their true metric.

Methods written to result tables:

```text
gp_direct_metric
bnn_response_metric_raw
bnn_response_metric_fourier
```

Full-mode settings:

```python
SEEDS = [0, 1, 2, 3, 4]
TRAIN_SIZES = [10, 20, 30, 40]
BNN_KWARGS_FULL = {
    "hidden_dims": [256, 256, 256],
    "epochs": 800,
    "batch_size": 1024,
    "lr": 1e-3,
    "bayesian_loss_weight": 1e-4,
    "mc_samples": 50,
    "target_mode": "db",
}
```

Outputs:

```text
experiments/gp_vs_bnn_response_shape_metric/prediction_rows.csv
experiments/gp_vs_bnn_response_shape_metric/summary_by_seed_size_method.csv
experiments/gp_vs_bnn_response_shape_metric/topk_selection_rows.csv
experiments/gp_vs_bnn_response_shape_metric/topk_summary.csv
experiments/gp_vs_bnn_response_shape_metric/bnn_checkpoints_raw/
experiments/gp_vs_bnn_response_shape_metric/bnn_checkpoints_fourier/
```

The notebook supports checkpoint loading and continued learning:

- Exact checkpoint exists: load it.
- Exact checkpoint missing: optionally load the closest smaller train-size
  checkpoint for the same seed and variant, then call `update(...)`.
- Raw and Fourier checkpoints are stored in separate directories to avoid mixing
  incompatible model shapes.

## Few-Shot Online Learning: Raw vs Fourier BNN

Notebook:

```text
notebooks/bnn_bclf_fewshot_online_learning.ipynb
```

Purpose:

- Compare raw and Fourier BNN response surrogates under the same online-learning
  schedule.
- Use identical seed, test split, initial train set, remaining pool, and per-step
  added case for both variants.
- Track prediction accuracy and response-shape objective error over online
  updates.

Full-mode settings:

```python
INITIAL_SAMPLES = 30
NUM_UPDATES = 50
INITIAL_EPOCHS = 800
UPDATE_EPOCHS = 20
TEST_FRACTION = 0.2
BASE_BNN_KWARGS = {
    "epochs": INITIAL_EPOCHS,
    "batch_size": 1024,
    "lr": 1e-3,
    "bayesian_loss_weight": 1e-4,
    "mc_samples": 50,
    "hidden_dims": [256, 256, 256],
    "target_mode": "db",
}
```

Main outputs:

```text
experiments/bnn_bclf_fewshot_online_learning/fewshot_metrics_by_variant.csv
experiments/bnn_bclf_fewshot_online_learning/fewshot_prediction_rows_by_variant.csv
experiments/bnn_bclf_fewshot_online_learning/fewshot_selected_trace_by_variant.csv
experiments/bnn_bclf_fewshot_online_learning/fewshot_prediction_cache_by_variant.npz
experiments/bnn_bclf_fewshot_online_learning/fewshot_bnn_db_raw_seed0_final.pt
experiments/bnn_bclf_fewshot_online_learning/fewshot_bnn_db_fourier_seed0_final.pt
```

Plots:

- `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_learning_curves_by_variant.png`
- `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_raw_objective_scatter_initial.png`
- `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_raw_objective_scatter_final.png`
- `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_fourier_objective_scatter_initial.png`
- `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_fourier_objective_scatter_final.png`
- `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_test_response_initial_by_variant.png`
- `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_test_response_final_by_variant.png`

## Optimizer BNN logEI Contract

`BNNExpectedImprovementBackend` now uses a response surrogate, not a scalar
objective BNN, for optimizer-side BNN logEI.

Strict path:

1. Optimizer history keeps in-memory `freqs` and `s_params` for completed trials.
2. The BNN backend builds a `BCLFResponseDataset` from those responses.
3. `LegacyBNNResponseSurrogate` trains on S11/S21 response curves.
4. For each remaining candidate, the BNN predicts response samples.
5. Each response sample is scored by the same `optimization_spec.evaluate(...)`
   used for real simulations.
6. logEI is estimated from sampled metrics:

```text
improvement = max(best_objective - sampled_metric, 0)
logEI = log(mean(improvement))
```

Selection debug includes:

```text
selection_debug["mode"] == "bnn_qlogei_discrete"
selection_debug["bnn_runtime"]["surrogate"] == "response"
```

Important implementation detail:

- In-memory history keeps `_response_freqs` and `_response_s_params`.
- Recorder and checkpoint serialization drop those private response arrays so
  JSON/CSV artifacts remain serializable.
- The backend currently uses `LegacyBNNResponseSurrogate` with `target_mode="db"`.
  Fourier response acquisition can be added by making the backend surrogate class
  configurable, but raw response-surrogate metric scoring is the enforced default.

## Checkpoint and Artifact Policy

Experiment outputs are generated artifacts and should not be committed.

- `.gitignore` ignores `experiments/`.
- If experiment files were already tracked, remove them from Git with:

```bash
git rm --cached -r experiments/<experiment_dir>
```

This keeps local files on disk while removing them from future commits.

For reusable checkpoint-loading conventions, see:

```text
.codex/experiment_checkpoint_loading.md
```

## Smoke Validation

Recommended quick checks before running full campaigns:

```bash
pytest tests/optimization/test_objectives.py
pytest tests/modeling/test_bclf_legacy_bnn.py
pytest tests/optimization/test_bnn_botorch_optimizer.py
```

Notebook smoke modes should use:

- `EXPERIMENT_MODE = "smoke"`
- small hidden dims such as `[16]`
- `epochs = 1`
- `mc_samples = 2`

The full campaigns are intentionally not part of automated tests.
