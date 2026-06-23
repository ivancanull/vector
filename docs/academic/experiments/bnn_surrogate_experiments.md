# BNN Surrogate Experiments

This page summarizes the BNN surrogate experiments for BCLF S-parameter
modeling. The current conclusion is that the legacy BNN can serve as a useful
surrogate for BCLF `S11`/`S21` response modeling on the cached Touchstone data.
These experiments validate the response surrogate itself; they do not by
themselves claim online optimization improvement.

## Research Question

Can a Bayesian neural network learn BCLF S-parameter responses well enough to
replace or support repeated simulator calls during offline screening and future
optimization studies?

The experiments focus on:

- predicting `S11` and `S21` response curves;
- training directly in dB for stable response-shape learning;
- evaluating train/test accuracy on cached responses;
- testing few-shot online learning by adding one new sample at a time.

## Shared Dataset And Target

- Dataset root:
  - `data/bandpass_coupled_line_filter_export/ext`
- Index:
  - `data/bandpass_coupled_line_filter_export/ext/index.json`
- Loader:
  - `modeling.BCLFTouchstoneDatasetLoader`
- Design vector order:
  - `L, L1, L2, W, W1, W2, S1, S2`
- Frequency grid:
  - loaded from Touchstone files; current cache uses 401 frequency points.
- Response channels:
  - `S11 = (0, 0)`
  - `S21 = (0, 1)`, with reciprocal fill to `(1, 0)` for objective evaluation.
- Objective reference:
  - `BCLFOptimizationSpec` with the current BCLF passband/stopband settings used
    by the notebooks.

## Surrogate Model

The experiments use the legacy Bayesian neural network response surrogate:

- class: `modeling.LegacyBNNResponseSurrogate`
- Bayesian layers: `MultiOutputBNN` with `BayesianLinear`
- loss: MSE plus `BKLLoss`
- input features:
  - `[L, L1, L2, W, W1, W2, S1, S2, frequency_hz]`
- preferred target mode:
  - `target_mode="db"`
  - the model directly learns `S11_dB` and `S21_dB`.

Direct dB training is the preferred setting. Earlier real/imag training could
produce noisy or flat response curves in small-data settings. Direct dB targets
provided visibly better agreement with the true response shape, especially for
`S11` notches and `S21` passband/roll-off behavior.

The surrogate automatically uses CUDA when available through PyTorch unless
`device` is explicitly set.

## Experiment 1: 80/20 Offline Surrogate Accuracy

Primary notebook:

- `notebooks/bnn_bclf_offline_feasibility.ipynb`

Purpose:

- train the dB BNN surrogate on 80% of cached BCLF samples;
- evaluate response prediction accuracy on both train and held-out test splits;
- compare true and predicted scalar objectives computed from the predicted
  response curves.

Default training settings:

```python
TARGET_MODE = "db"
HOLDOUT_TRAIN_FRACTION = 0.8
HOLDOUT_SEED = 0

BNN_KWARGS = {
    "epochs": 1000,
    "batch_size": 1024,
    "lr": 1e-3,
    "bayesian_loss_weight": 1e-4,
    "mc_samples": 50,
    "hidden_dims": [256, 256, 256],
    "target_mode": TARGET_MODE,
}
```

Artifacts:

- `experiments/bnn_bclf_offline_feasibility/legacy_bnn_db_holdout_seed0.pt`
- `experiments/bnn_bclf_offline_feasibility/holdout_split.json`
- `experiments/bnn_bclf_offline_feasibility/legacy_bnn_db_holdout_predictions.npz`
- `experiments/bnn_bclf_offline_feasibility/holdout_accuracy_metrics.csv`
- `experiments/bnn_bclf_offline_feasibility/holdout_train_predictions.csv`
- `experiments/bnn_bclf_offline_feasibility/holdout_test_predictions.csv`
- `studies/bnn_surrogate/figures/bnn_bclf_offline_feasibility/holdout_train_test_objective_true_vs_pred.png`
- `studies/bnn_surrogate/figures/bnn_bclf_offline_feasibility/holdout_train_best_response_prediction.png`
- `studies/bnn_surrogate/figures/bnn_bclf_offline_feasibility/holdout_test_best_response_prediction.png`

Observation:

- The dB-trained BNN reproduces the main BCLF response structure.
- `S11` notch locations and depths are captured much better than with the
  earlier noisy per-frequency dropout behavior.
- `S21` passband shape and stopband roll-off are qualitatively aligned with the
  true response.
- The completed experiment supports the claim that the BNN can serve as an
  S-parameter response surrogate on this dataset.

Result figures:

![80/20 train/test objective true-vs-predicted comparison](../../../studies/bnn_surrogate/figures/bnn_bclf_offline_feasibility/holdout_train_test_objective_true_vs_pred.png)

![80/20 train response prediction example](../../../studies/bnn_surrogate/figures/bnn_bclf_offline_feasibility/holdout_train_best_response_prediction.png)

![80/20 held-out test response prediction example](../../../studies/bnn_surrogate/figures/bnn_bclf_offline_feasibility/holdout_test_best_response_prediction.png)

## Experiment 2: Few-Shot Online Learning Simulation

Primary notebook:

- `notebooks/bnn_bclf_fewshot_online_learning.ipynb`

Purpose:

- simulate online learning over the cached dataset without launching HFSS;
- start from a few randomly selected samples;
- add one new cached sample at each iteration;
- warm-start the same BNN instead of retraining from scratch;
- track how train/test response accuracy changes as new data arrives.

Default full-mode settings:

```python
EXPERIMENT_MODE = "full"
TARGET_MODE = "db"
INITIAL_SAMPLES = 8
NUM_UPDATES = 80
INITIAL_EPOCHS = 100
UPDATE_EPOCHS = 10
TEST_FRACTION = 0.2
```

Training policy:

- initial fit uses 8 random samples and 100 epochs;
- each update adds 1 random unseen sample;
- each update warm-starts the existing model for 10 additional epochs;
- the model keeps optimizer and network state across updates.

Artifacts:

- `experiments/bnn_bclf_fewshot_online_learning/fewshot_config.json`
- `experiments/bnn_bclf_fewshot_online_learning/fewshot_metrics.csv`
- `experiments/bnn_bclf_fewshot_online_learning/fewshot_selected_trace.csv`
- `experiments/bnn_bclf_fewshot_online_learning/fewshot_prediction_rows.csv`
- `experiments/bnn_bclf_fewshot_online_learning/fewshot_prediction_cache.npz`
- `experiments/bnn_bclf_fewshot_online_learning/fewshot_bnn_db_seed0_final.pt`
- `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_learning_curves.png`
- `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_objective_scatter_initial.png`
- `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_objective_scatter_final.png`
- `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_test_response_initial.png`
- `studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_test_response_final.png`

Metrics tracked per iteration:

- number of training samples;
- cumulative training epochs;
- train/test `S11_dB` MAE/RMSE;
- train/test `S21_dB` MAE/RMSE;
- train/test scalar objective MAE/RMSE;
- objective Spearman and top-k recall where meaningful.

Observation:

- This experiment tests whether the BNN can improve incrementally as more data
  is added.
- It uses the same dB response target as the completed 80/20 surrogate study.
- It is an offline replay of online learning, so it validates the update
  strategy and surrogate behavior before connecting to live HFSS data.

Result figures:

![Few-shot online learning curves](../../../studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_learning_curves.png)

![Few-shot initial objective true-vs-predicted comparison](../../../studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_objective_scatter_initial.png)

![Few-shot final objective true-vs-predicted comparison](../../../studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_objective_scatter_final.png)

![Few-shot initial held-out test response prediction](../../../studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_test_response_initial.png)

![Few-shot final held-out test response prediction](../../../studies/bnn_surrogate/figures/bnn_bclf_fewshot_online_learning/fewshot_test_response_final.png)

## Current Conclusion

The completed experiments show that the dB-trained legacy BNN is a viable
surrogate for BCLF S-parameter modeling on the cached dataset. It can reproduce
important response structures such as `S11` notches and `S21` passband behavior,
and it supports warm-start updates for few-shot online-learning studies.

This supports using the BNN surrogate as a candidate component for future
surrogate-assisted optimization. Future claims about optimization performance
should come from a separate optimization experiment that uses this surrogate for
candidate ranking or acquisition and compares against random/GP baselines.

## Caveats

- These are surrogate-modeling experiments, not online HFSS optimization
  results.
- In `target_mode="db"`, the model predicts magnitude responses in dB and
  reconstructs nonnegative magnitudes for objective evaluation; it does not
  recover response phase.
- Representative plots should be reported together with the full train/test
  metrics CSVs.
- If stronger physical consistency is required, compare this dB BNN against a
  curve-level model or vector-fitting parameter surrogate.
