# Methods

## Problem Setup

We optimize a bandpass coupled-line filter (BCLF) whose HFSS evaluation is
expensive enough that every simulator call must be tracked as part of the
experiment budget.

The design vector is ordered as:

```text
x = [L, L1, L2, W, W1, W2, S1, S2]
```

The current online target uses `BCLFOptimizationSpec` with:

| Item | Value |
|---|---|
| Passband | `[4.8, 5.4] GHz` |
| Lower stopband | `[3.6, 4.4] GHz` |
| Upper stopband | `[5.8, 8.7] GHz` |
| Passband transmission limit | worst `S21 >= -5.0 dB` |
| Passband return-loss limit | worst `S11 <= -7.0 dB` |
| Stopband attenuation limit | worst stopband `S21 <= -15.0 dB` |

Human-facing design quality improves when passband `S21` is high, passband
`S11` is low, and stopband `S21` is low. The optimizer-facing scalar objective
is minimized. All convergence plots and tables should therefore use
`best_so_far`, where lower is better.

The full target, bounds, queue settings, and artifact roots are maintained in
[BCLF Optimization Target And Online Settings](../bclf_optimization_target.md).

## Shared Optimization Backbone

All online methods use the same simulator-facing loop:

1. Sample or select one candidate from a finite candidate pool.
2. Send the geometry through a simulator adapter.
3. Score the returned `S11` and `S21` response with the same
   `BCLFOptimizationSpec`.
4. Append a trial row to the run history.
5. Save JSON, CSV, and checkpoint artifacts.

The backbone keeps HFSS and queue interactions behind adapter layers. The online
campaigns use `ServerQueueSimulator`; replay and smoke experiments use replay
simulators over cached response datasets. Checkpoint completion is judged by
history length reaching the target trial count, not by status labels alone.

## Baseline Methods

### BO: GP logEI

The BO baseline uses `OptimizerBoTorch` with a Gaussian-process surrogate and
global logEI selection over the remaining candidate pool.

| Setting | Value |
|---|---|
| Notebook | `notebooks/bo_online_experiment.ipynb` |
| Optimizer | `OptimizerBoTorch` |
| Method label | `global_gp_qlogei` |
| Initial trials | `N_INIT = 30` |
| BO trials | `BUDGET = 50` |
| Total trials per seed | `80` |
| Seeds | `[0, 1, 2, 3, 4]` |
| Output root | `experiments/bo_online_general/` |

### LHS Random

The LHS baseline uses `OptimizerLHSRandom` with the same target, geometry
bounds, candidate-pool size, queue-backed simulator, seed convention, and
artifact contract as BO. It has no surrogate or acquisition model.

| Setting | Value |
|---|---|
| Notebook | `notebooks/lhs_online_experiment.ipynb` |
| Optimizer | `OptimizerLHSRandom` |
| Method label | `lhs_random` |
| Total trials per seed | `80` |
| Seeds | `[0, 1, 2, 3, 4]` |
| Output root | `experiments/lhs_online_general/` |

### Turbo: Trust-Region GP logEI

The Turbo baseline uses the same GP logEI backbone as BO, but constrains
candidate selection with trust-region state. It shares the same target,
candidate pool, simulator, seed convention, and artifact contract.

| Setting | Value |
|---|---|
| Notebook | `notebooks/turbo_online_experiment.ipynb` |
| Optimizer | `OptimizerBoTorch` with trust-region selection |
| Method label | `trust_region_gp_qlogei` |
| Initial trials | `N_INIT = 30` |
| Turbo trials | `BUDGET = 50` |
| Total trials per seed | `80` |
| Seeds | `[0, 1, 2, 3, 4]` |
| Output root | `experiments/turbo_online_general/` |

## BNN Response Modeling

The BNN work is response-level rather than scalar-objective-only. The
`LegacyBNNResponseSurrogate` learns `S11` and `S21` response curves from
geometry plus frequency inputs, then derives objective values by passing
predicted responses through the same `BCLFOptimizationSpec`.

Two response-surrogate variants are implemented:

| Variant | Implementation | Purpose |
|---|---|---|
| Raw frequency BNN | `modeling.LegacyBNNResponseSurrogate` | Baseline response surrogate |
| Fourier frequency BNN | `modeling.FourierBNNResponseSurrogate` | Frequency-feature variant for response-shape experiments |

The current optimizer-side BNN acquisition path is
`optimization.BNNExpectedImprovementBackend`. It builds a response dataset from
completed optimizer history rows, samples predicted responses for remaining
candidates, scores those sampled responses with the same optimization spec, and
selects by EI-style acquisition value.

## BNN Online Baseline Protocol

The full online BNN notebook is implemented but its queue-backed campaign has
not been completed. The planned production protocol is:

| Setting | Value |
|---|---|
| Notebook | `notebooks/bnn_online_experiment.ipynb` |
| Optimizer | `OptimizerBoTorch` with `BNNExpectedImprovementBackend` |
| Method label | `bnn_response_qlogei` |
| Initial trials | `N_INIT = 30` |
| BNN-guided trials | `BUDGET = 50` |
| First BNN fit | after 30 initial cases, train 800 epochs |
| Incremental learning | warm-start 20 epochs after each new online case |
| Seeds | `[0, 1, 2, 3, 4]` |
| Output root | `experiments/bnn_online_general/` |

Until the full online campaign finishes, paper claims should describe BNN
feasibility, replay behavior, and acquisition-quality diagnostics only. They
should not make a final online ranking claim.

## Frequency-Sampling Diagnostic

The offline AFS reconstruction diagnostic includes Objective-Weighted AFS as a
retrospective method. It is stored in experiment artifacts as
`oracle_vf_error`, but paper-facing text should use the descriptive method name
Objective-Weighted AFS. The uncertainty-only baseline is Uncertainty-Aware AFS,
stored internally as `uaw`.

Objective-Weighted AFS repeatedly fits VF on the currently sampled frequency
points, compares the VF reconstruction against the full cached response, and
adds the unsampled frequency with the largest objective-weighted residual:

```text
score(f) = band_weight(f) *
           (0.5 * |S11_true_dB(f) - S11_vf_dB(f)|
            + |S21_true_dB(f) - S21_vf_dB(f)|)
```

The band weights emphasize BCLF passband, stopband, and boundary regions. Since
the method uses the full response during selection, it is not a deployable
online sampler; it is a diagnostic for the headroom available to practical
objective-weighted AFS designs.

The deployable combined variant is Objective-Weighted + Uncertainty-Aware AFS,
stored internally as `owafs_bnn`. It uses the same endpoint, BCLF boundary, and
uniform-fill anchors, but selects later frequency points without inspecting the
full-wave truth. Its score combines a BNN-proxy VF residual with BNN Monte
Carlo uncertainty:

```text
score(f) = band_weight(f) * 0.5 *
           (rank01(vf_residual_db(f)) + rank01(mc_uncertainty_db(f)))
```

Here `vf_residual_db` compares the BNN mean response with a VF reconstruction
fit to the currently selected points, while `mc_uncertainty_db` is estimated
from BNN response samples. Full-wave responses are used only after selection to
evaluate the offline reconstruction and objective error.

## Paper Claim Boundaries

- Supported now: method descriptions, objective protocol, BO online baseline,
  completed LHS and Turbo online baselines, BO/LHS/Turbo comparison, offline
  BNN feasibility, few-shot BNN updating, response-shape replay studies, BNN EI
  replay, and acquisition-quality diagnostics.
- Supported limitation: under the current completed five-seed comparison, BO
  and Turbo do not outperform LHS.
- Not supported yet: BNN online performance superiority, final
  BO-vs-LHS-vs-Turbo-vs-BNN online ranking, or any conclusion based on
  incomplete BNN queue-backed runs.

Operational implementation guidance is maintained in
[Optimization Experiment Backbone](../optimization_experiment_backbone.md).
