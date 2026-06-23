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
[BCLF Optimization Target And Online Settings](../experiments/bclf_optimization_target.md).

## Shared Optimization Backbone

All online methods use the same simulator-facing loop:

1. Sample or select one candidate from the configured geometry domain.
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
global logEI selection. For online `ContinuousCandidateSpace` campaigns, the GP
backend optimizes the acquisition over normalized continuous bounds with
BoTorch `optimize_acqf`, then snaps the selected geometry to the configured
fabrication resolution before cache lookup or HFSS submission. Replay and
finite-pool experiments still use discrete acquisition over the available
candidate set.

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
bounds, queue-backed simulator, seed convention, resolution, and artifact
contract as BO. It has no surrogate or acquisition model.

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
continuous acquisition bounds with trust-region state. It shares the same
target, simulator, seed convention, resolution, and artifact contract.

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
completed optimizer history rows. For online `ContinuousCandidateSpace` runs,
each BNN-guided step resamples a fresh batch from the normalized geometry
bounds, projects it back to physical geometry, snaps it to the configured
resolution, predicts responses for that transient batch, scores those sampled
responses with the same optimization spec, and selects by EI-style acquisition
value. The fixed finite-candidate path is reserved for replay/discrete
experiments.

## BNN Online Baseline Protocol

The full online BNN notebook is implemented and the current
`bnn_online_general` campaign has completed seeds `[0, 1, 2, 3, 4]` with
`80` trials per seed. The production protocol is:

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

The completed five-seed BNN baseline has median final best `-17.3415` and mean
final best `-17.4018`. It is better than BO by median final best, but does not
outperform LHS or Turbo under the current online-general protocol. Do not use
previous-LHS, `online_lhs15_shared`, or `lhs15_shared` BNN artifacts as the BNN
baseline for this comparison.

## OWAFS Previous-LHS Warmup Protocol

The online OWAFS diagnostic tests whether the deployable
Objective-Weighted + Uncertainty-Aware AFS idea can improve from a small
previous-LHS initial design. The run uses the same BCLF scalar objective as the
other online campaigns, but its initialization differs: each seed starts from
15 warmup points taken from the previous-LHS database rather than from the
`N_INIT = 30` online-general protocol.

| Setting | Value |
|---|---|
| Output root | `experiments/online_lhsdb_warmup/owafs_bnn/` |
| Initial design source | `experiments/online_lhsdb_warmup/initial_designs/warmup_initial_points.csv` |
| Warmup points per seed | `15` previous-LHS points |
| Budgeted online trials per seed | `80` |
| Validation rows per completed seed | `1-2` in current artifacts |
| Completed seeds | `[0, 1, 2, 3, 4]` |
| Diagnostic plots | `studies/online_owafs/figures/online_lhsdb_warmup/` |

The paper-facing interpretation should separate three questions:

1. Does OWAFS improve its own 15-point previous-LHS warmup?
2. Does OWAFS look competitive with the current online-general baselines under
   descriptive, unpaired seed pooling?
3. Does OWAFS expand geometric coverage beyond the warmup region?

The current evidence supports only the first question. It partially supports
competitiveness with BO and BNN, because the OWAFS median final best is better
than the five-seed BO median and close to the five-seed BNN median, but OWAFS
seed `2` is weak and the initialization differs. It does not support
superiority over LHS or Turbo, and it does not support a claim of stronger
global coverage.

## OWAFS Comparison Methods

The current OWAFS comparison uses the following method roles and boundaries:

| Method | Role | Data source | Boundary |
|---|---|---|---|
| OWAFS-BNN | Current method under diagnosis; objective-weighted adaptive sampling after 15 previous-LHS warmup points. | `experiments/online_lhsdb_warmup/owafs_bnn` seeds `0,1,2,3,4` | Valid for testing warmup improvement; not yet a superiority claim. |
| BNN online_general | Main neural surrogate baseline for this diagnostic. | `experiments/bnn_online_general` seeds `0,1,2,3,4` | Excludes previous-LHS / `online_lhs15_shared` BNN runs; initialization differs from OWAFS. |
| BO online_general | Conventional BO reference. | `experiments/bo_online_general` seeds `0,1,2,3,4` | Five-seed descriptive reference, not paired by warmup. |
| LHS online_general | Coverage-oriented random design reference. | `experiments/lhs_online_general` seeds `0,1,2,3,4` | Useful as a coverage reference, not a same-warmup baseline. |

The geometry diagnostic fits one PCA projection across all seed0-4 warmup and
OWAFS budgeted points. It plots the original 15 LHS warmup points, subsequent
OWAFS online points, and each seed's best point in the shared PCA plane. The
nearest-distance curve reports each OWAFS online point's normalized Euclidean
distance to the same seed's 15 original warmup points. This diagnostic is meant
to test exploration expansion, not objective quality alone.

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
  replay, acquisition-quality diagnostics, completed BNN online-general
  baseline, and OWAFS seed0-4 warmup-improvement diagnostics.
- Supported limitation: under the current completed five-seed comparison, BO
  and Turbo do not outperform LHS.
- Supported limitation: current OWAFS improves its own 15-point previous-LHS
  warmup but is not yet stronger than the LHS/Turbo references and does not
  show sustained geometric expansion.
- Not supported: BNN online performance superiority, OWAFS superiority over
  BNN/LHS/Turbo, or a strict final BO-vs-LHS-vs-Turbo-vs-BNN-vs-OWAFS ranking
  that ignores the different OWAFS initialization.

Operational implementation guidance is maintained in
[Optimization Experiment Backbone](../../engineering/optimization/optimization_experiment_backbone.md).
