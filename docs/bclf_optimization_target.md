# BCLF Optimization Target And Online Settings

This page documents the current BCLF optimization target, parameter constraints,
and online campaign settings used by the BO and LHS notebooks.

The BO/LHS comparison intentionally keeps the BCLF target and evaluation
contract fixed; the method-specific differences are documented below so runs do
not silently mix bands, thresholds, budgets, or artifact locations.

## Dataset And Response Channels

- Dataset/cache root: `data/bandpass_coupled_line_filter_export/ext`
- Config source: `configs/filter_bnn_online_uaw_uaw.yaml`
- Frequency grid:
  - minimum: `3.5e9 Hz`
  - maximum: `8.713e9 Hz`
  - step: `0.013e9 Hz`
- S-parameter channels used:
  - `S11`
  - `S21`

The optimizer evaluates simulated or replayed responses on this frequency grid.
Responses are interpreted in dB for the objective and constraint metrics.

## Design Vector

The geometry vector is ordered as:

```text
x = [L, L1, L2, W, W1, W2, S1, S2]
```

Parameter bounds are taken from the first eight entries of
`data.regularization.min` and `data.regularization.max` in
`configs/filter_bnn_online_uaw_uaw.yaml`. The ninth config entry is a frequency
bound and is not part of the BCLF geometry vector.

| Variable | Min | Max | Unit |
|---|---:|---:|---|
| `L` | 7.7 | 7.9 | mm |
| `L1` | 8.0 | 8.2 | mm |
| `L2` | 7.8 | 8.0 | mm |
| `W` | 0.9 | 1.1 | mm |
| `W1` | 0.7 | 0.9 | mm |
| `W2` | 0.9 | 1.1 | mm |
| `S1` | 0.13 | 0.16 | mm |
| `S2` | 0.55 | 0.60 | mm |

## Frequency Bands

The current online notebooks construct `BCLFOptimizationSpec` with these bands:

| Band | Range |
|---|---:|
| Passband `Bp` | `[4.8, 5.4] GHz` |
| Lower stopband `Bsl` | `[3.6, 4.4] GHz` |
| Upper stopband `Bsu` | `[5.8, 8.7] GHz` |

No center-frequency constraint is used by these current online BCLF targets.

## Objective And Limits

The human-facing goal is to improve passband transmission while satisfying
passband return-loss and stopband attenuation limits:

- improve average passband `S21`;
- keep worst passband `S21 >= -5.0 dB`;
- keep worst passband `S11 <= -7.0 dB`;
- keep worst stopband `S21 <= -15.0 dB`.

The concrete notebook settings are:

```python
BCLFOptimizationSpec(
    passband_ghz=(4.8, 5.4),
    lower_stopband_ghz=(3.6, 4.4),
    upper_stopband_ghz=(5.8, 8.7),
    min_passband_s21_db=-5.0,
    max_passband_s11_db=-7.0,
    max_stopband_s21_db=-15.0,
    s11_weight=0.5,
    stopband_weight=0.5,
)
```

`BCLFOptimizationSpec` emits a scalar objective for the optimizer to minimize.
Internally, higher-quality passband transmission reduces the minimized scalar
objective, while hinge-style penalties increase it when the S-parameter limits
are violated.

## Sign Convention And Reporting

- Human interpretation: better designs have stronger passband `S21`, acceptable
  passband `S11`, and sufficiently low stopband `S21`.
- Optimizer convention: lower scalar `objective` is better.
- Convergence metric: use `best_so_far` over iteration, not raw per-iteration
  objective alone.

## Current BO/LHS Online Comparison

Source notebooks:

- BO: `notebooks/bo_online_experiment.ipynb`
- LHS: `notebooks/lhs_online_experiment.ipynb`

### Held Fixed Between BO And LHS

| Setting | Current value |
|---|---|
| BCLF objective spec | `BCLFOptimizationSpec(...)` settings shown above |
| Design vector | `[L, L1, L2, W, W1, W2, S1, S2]` |
| Variable bounds | first eight `data.regularization.min/max` entries from `configs/filter_bnn_online_uaw_uaw.yaml` |
| Candidate space | `ContinuousCandidateSpace` |
| Candidate pool size | `1500` |
| Geometry resolution | `0.01 mm` candidate snapping and simulator `mm_resolution` |
| Candidate-space seed policy | fresh same-seed runs use the campaign seed for pool generation |
| Campaign seeds | `[0, 1, 2, 3, 4]` |
| Target trial count | `80` evaluated trials per seed |
| Simulator backend | queue-backed `ServerQueueSimulator` with `SIMULATOR_BACKEND='server'` |
| Queue/template | `queue/`, `template_name='bclf_template_v1'` |
| Result cache | `data/bandpass_coupled_line_filter_export/ext` |
| Ports/channels | `port_num=2`, objective uses `S11` and `S21` |
| Unit map | all geometry variables are sent as `mm` |
| Extracted worker metrics | `max_s11_db`, `min_s21_db` |
| Resume policy | `RESUME_IF_CHECKPOINT=True` with one checkpoint per seed |
| Completion gate | history length must reach the target trial count |
| Main comparison signal | `best_so_far` over iteration |

The two methods therefore differ only in candidate-selection policy and
method-specific run bookkeeping. They use the same objective, finite sampled
candidate-pool construction, simulator adapter, seed list, response channels,
and per-seed artifact contract.

### Method-Specific Settings

| Setting | BO online | LHS online |
|---|---|---|
| Method label | `global_gp_qlogei` | `lhs_random` |
| Optimizer class | `OptimizerBoTorch` | `OptimizerLHSRandom` |
| Selection policy | global GP qLogEI over remaining candidates | deterministic Latin-hypercube sequence in normalized space, selecting the nearest remaining candidate |
| Initial/model split | `N_INIT=30`, then `BUDGET=50` BO trials | no model-initialization phase; `TOTAL_BUDGET=80` LHS trials |
| Total trials | `N_INIT + BUDGET = 80` | `TOTAL_BUDGET = 80` |
| Model fitting | GP surrogate refit every step with `RETRAIN_EVERY=1` | no surrogate model or acquisition fit |
| Optimizer seed | campaign seed | checkpoint seed on resume, otherwise campaign seed |
| Reset flag | `RESET_CHECKPOINT=True` in the current BO notebook | `RESET_CHECKPOINT=False` in the current LHS notebook |
| Output root | `experiments/bo_online_general/` | `experiments/lhs_online_general/` |
| Run ID pattern | `bo_online_general_seed{seed}` | `lhs_online_general_seed{seed}` |
| Campaign summary | `experiments/bo_online_general/campaign_summary.csv` | `experiments/lhs_online_general/campaign_summary.csv` |
| Selection diagnostics | model/acquisition fields when available plus BoTorch debug state | `lhs_index`, candidate-pool size, nearest normalized distance, selected position |

## Artifact Roots

Each online campaign records per-seed JSON/CSV/checkpoint artifacts under its
own output root:

| Method | Output root | Run ID pattern | Campaign summary |
|---|---|---|---|
| BO | `experiments/bo_online_general/` | `bo_online_general_seed{seed}` | `experiments/bo_online_general/campaign_summary.csv` |
| LHS | `experiments/lhs_online_general/` | `lhs_online_general_seed{seed}` | `experiments/lhs_online_general/campaign_summary.csv` |
