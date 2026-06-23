# BCLF Optimization Target And Online Settings

This page documents the current BCLF optimization target, parameter constraints,
and online campaign settings used by the BO, LHS, and Turbo notebooks.

The BO/LHS/Turbo comparison intentionally keeps the BCLF target and evaluation
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

## Current BO/LHS/Turbo Online Comparison

Source notebooks:

- BO: `notebooks/bo_online_experiment.ipynb`
- LHS: `notebooks/lhs_online_experiment.ipynb`
- Turbo: `notebooks/turbo_online_experiment.ipynb`
- comparison notebook: `notebooks/online_lhsdb_warmup_comparison.ipynb`

### Held Fixed Between BO, LHS, And Turbo

| Setting | Current value |
|---|---|
| BCLF objective spec | `BCLFOptimizationSpec(...)` settings shown above |
| Design vector | `[L, L1, L2, W, W1, W2, S1, S2]` |
| Variable bounds | first eight `data.regularization.min/max` entries from `configs/filter_bnn_online_uaw_uaw.yaml` |
| Candidate space | `ContinuousCandidateSpace` |
| Acquisition candidate batch size | `1500` for BNN-style sampled-batch acquisition; BO/Turbo GP acquisition now optimizes continuously over normalized bounds |
| Geometry resolution | `0.01 mm` candidate snapping and simulator `mm_resolution` |
| Candidate-space seed policy | fresh same-seed runs use the campaign seed for initialization and sampled acquisition batches |
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

The methods therefore differ only in candidate-selection policy and
method-specific run bookkeeping. They use the same objective, bounded geometry
domain, simulator adapter, seed list, response channels, resolution, and
per-seed artifact contract. Existing completed comparison artifacts were
generated before the continuous-acquisition refactor; new BO/Turbo reruns use
continuous `optimize_acqf`, while BNN sampled-batch acquisition resamples from
bounds before each guided selection.

### Method-Specific Settings

| Setting | BO online | LHS online | Turbo online |
|---|---|---|---|
| Method label | `global_gp_qlogei` | `lhs_random` | `trust_region_gp_qlogei` |
| Optimizer class | `OptimizerBoTorch` | `OptimizerLHSRandom` | `OptimizerBoTorch` with trust-region selection |
| Selection policy | global GP logEI via continuous `optimize_acqf`, snapped to resolution | deterministic Latin-hypercube sequence in normalized space, selecting the nearest available sampled candidate | trust-region GP logEI via continuous `optimize_acqf` inside trust-region bounds, snapped to resolution |
| Initial/model split | `N_INIT=30`, then `BUDGET=50` BO trials | no model-initialization phase; `TOTAL_BUDGET=80` LHS trials | `N_INIT=30`, then `BUDGET=50` Turbo trials |
| Total trials | `N_INIT + BUDGET = 80` | `TOTAL_BUDGET = 80` | `N_INIT + BUDGET = 80` |
| Model fitting | GP surrogate refit every step with `RETRAIN_EVERY=1` | no surrogate model or acquisition fit | GP surrogate refit every step with trust-region state |
| Optimizer seed | campaign seed | checkpoint seed on resume, otherwise campaign seed | campaign seed |
| Output root | `experiments/bo_online_general/` | `experiments/lhs_online_general/` | `experiments/turbo_online_general/` |
| Run ID pattern | `bo_online_general_seed{seed}` | `lhs_online_general_seed{seed}` | `turbo_online_general_seed{seed}` |
| Campaign summary | `experiments/bo_online_general/campaign_summary.csv` | `experiments/lhs_online_general/campaign_summary.csv` | `experiments/turbo_online_general/campaign_summary.csv` |
| Selection diagnostics | model/acquisition fields plus continuous BoTorch debug state | `lhs_index`, candidate-batch size, nearest normalized distance, selected position | trust-region state plus continuous BoTorch debug state |

### Current Five-Seed Comparison Result

The completed 80-trial comparison over seeds `[0, 1, 2, 3, 4]` is summarized by
`experiments/bo_lhs_turbo_online_comparison/bo_lhs_turbo_final_summary.csv` and
`experiments/bo_lhs_turbo_online_comparison/bo_lhs_turbo_per_seed_final.csv`.
Lower objective is better.

| Method | Completed seeds | Median final best | Mean final best | Best seed result | Worst seed result |
|---|---:|---:|---:|---:|---:|
| BO | 5 | -14.657 | -15.649 | -23.814 | -8.052 |
| LHS | 5 | -19.162 | -18.815 | -25.141 | -9.847 |
| Turbo | 5 | -19.161 | -17.524 | -22.517 | -9.206 |

Per-seed winners are LHS on seeds `0`, `2`, and `4`; BO on seed `1`; and a
BO/Turbo tie on seed `3`. Under this current protocol, the evidence does not
support claiming that BO or Turbo outperforms the LHS baseline. This should be
treated as a limitation of the current optimizer setup, not as proof that
surrogate-assisted optimization is generally useless for BCLF design.

### Best-Candidate S-Parameter Shape Check

The notebook
[`notebooks/bclf_best_candidate_sp_range_compare.ipynb`](../../../notebooks/bclf_best_candidate_sp_range_compare.ipynb)
selects each method's global best finite-objective candidate across the five
completed seeds, loads its cached `.s2p` response, and plots full-band `S21` and
`S11` plus band-metric range bars. It does not submit worker jobs or run HFSS.

| Method | Best seed | Iteration | Case ID | Objective |
|---|---:|---:|---|---:|
| BO | 1 | 38 | `bo_online_general_seed1_it0038` | -23.814 |
| LHS | 0 | 23 | `lhs_online_general_seed0_it0023` | -25.141 |
| Turbo | 3 | 59 | `turbo_online_general_seed3_it0059` | -22.517 |

Constraint-facing extrema from the best candidates are:

| Method | Passband S21 min | Passband S11 max | Low-stop S21 max | High-stop S21 max |
|---|---:|---:|---:|---:|
| BO | -1.440 dB | -7.518 dB | -19.487 dB | -16.328 dB |
| LHS | -1.020 dB | -9.372 dB | -18.932 dB | -15.646 dB |
| Turbo | -1.759 dB | -6.419 dB | -23.194 dB | -18.704 dB |

The BO and LHS best candidates satisfy the passband and stopband thresholds
used by the current target (`S21 >= -5 dB` in passband, `S11 <= -7 dB` in
passband, stopband `S21 <= -15 dB`). The Turbo best candidate has stronger
stopband attenuation than BO/LHS in this summary, but its passband `S11` maximum
is `-6.419 dB`, slightly above the `-7 dB` target line; its scalar objective is
therefore worse than the LHS and BO best candidates.

Generated local artifacts:

- `studies/online_comparison/figures/bo_lhs_turbo_online_comparison/best_candidate_sp_range/bclf_best_candidate_s11_s21_fullband.png`
- `studies/online_comparison/figures/bo_lhs_turbo_online_comparison/best_candidate_sp_range/bclf_best_candidate_band_metric_range_bars.png`
- `experiments/bo_lhs_turbo_online_comparison/best_candidate_sp_range/bclf_best_candidate_band_metrics.csv`
- `experiments/bo_lhs_turbo_online_comparison/best_candidate_sp_range/bclf_best_candidate_sp_range_summary.json`

## Artifact Roots

Each online campaign records per-seed JSON/CSV/checkpoint artifacts under its
own output root:

| Method | Output root | Run ID pattern | Campaign summary |
|---|---|---|---|
| BO | `experiments/bo_online_general/` | `bo_online_general_seed{seed}` | `experiments/bo_online_general/campaign_summary.csv` |
| LHS | `experiments/lhs_online_general/` | `lhs_online_general_seed{seed}` | `experiments/lhs_online_general/campaign_summary.csv` |
| Turbo | `experiments/turbo_online_general/` | `turbo_online_general_seed{seed}` | `experiments/turbo_online_general/campaign_summary.csv` |
| BNN | `experiments/bnn_online_general/` | `bnn_online_general_seed{seed}` | `experiments/bnn_online_general/campaign_summary.csv` |

The BNN online-general campaign is complete for seeds `[0, 1, 2, 3, 4]` with
`80` trials per seed. Its median final best objective is `-17.3415`, which is
better than BO's median but not better than the LHS or Turbo medians under the
current protocol.
