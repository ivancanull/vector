# BCLF TuRBO Optimization Target

This page documents the optimization target and parameter constraints used by
`notebooks/turbo_online_experiment.ipynb`.

This is the TuRBO-specific online BCLF target. It is intentionally separate from
older BO/paper objective variants so comparisons do not silently mix different
bands or thresholds.

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

The TuRBO notebook constructs `BCLFOptimizationSpec` with these bands:

| Band | Range |
|---|---:|
| Passband `Bp` | `[4.8, 5.4] GHz` |
| Lower stopband `Bsl` | `[3.6, 4.4] GHz` |
| Upper stopband `Bsu` | `[5.8, 8.7] GHz` |

No center-frequency constraint is used by this TuRBO target.

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

The TuRBO campaign records per-seed JSON/CSV/checkpoint artifacts under:

```text
experiments/turbo_online_general/
```

Per-seed run IDs follow:

```text
turbo_online_general_seed{seed}
```
