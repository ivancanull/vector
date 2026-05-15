# Experiments

This page tracks paper-facing experiment families using the canonical experiment-card format.

Template:
- [templates/experiment_card.md](templates/experiment_card.md)

Migration note:
- Canonical objective construction is class-init style: `BCLFOptimizationSpec(...).to_objective()`.
- Compatibility wrappers `build_bclf_optimization_spec(...)` and `build_bclf_objective(...)` are still available for legacy callers.

## Experiment Settings Summary

| Setting | BO Online Seed Campaign | BO Replay Grid Study |
| --- | --- | --- |
| Primary notebook | `notebooks/bo_online_campaign.ipynb` | `notebooks/bo_replay_experiment.ipynb` |
| Optimizer | `OptimizerBoTorch` | `OptimizerBoTorch` over `DiscreteCandidateSpace` (+ `OptimizerRandom` baseline) |
| Simulator source | Queue-backed real simulator | Replay dataset simulator |
| Objective function | `BCLFOptimizationSpec(...).to_objective()` | `BCLFOptimizationSpec(...).to_objective()` (aligned with online) |
| Passband / stopband | `Bp=[5.2 GHz,5.9 GHz]`, `Bsl=[3.6 GHz,4.8 GHz]`, `Bsu=[6.5 GHz,8.7 GHz]` | Same as online |
| Objective definition | `BCLFOptimizationSpec(...).to_objective()` (penalty-augmented objective, no explicit BoTorch constraints) | Same as online |
| Acquisition (Replay BO) | N/A | `qLogEI` (with PoF bootstrap only when constraints exist and no feasible train points) |
| Seed policy | Fixed campaign seed list with resume | Multi-seed sweeps per method setting |
| Completion / inclusion gate | `status==completed` and `num_trials >= N_INIT + BUDGET` | Include all configured seeds per setting; same aggregation rule across settings |
| Main outputs | `experiments/bo_online/campaign_summary.csv`, per-seed `csv/json/checkpoint` | `experiments/bo_replay/optimization_grid_summary.csv`, `optimization_grid_runs.csv`, per-run `grid_*.csv/json` |

---

## Objective Update (2026-05-06): EXT-Informed Narrow Passband

### New Objective Settings

- passband: `4.8-5.2 GHz`
- passband constraints: `S21 >= -5 dB`, `S11 <= -7 dB`
- stopbands (online BO notebook): unchanged
  - lower stopband: `3.6-4.8 GHz`
  - upper stopband: `6.5-8.7 GHz`
  - stopband constraint term: `max_stopband_s21_db=-15 dB` (as currently configured in notebook)

### Experiment Steps (Notebook Workflow)

1. Run `notebooks/fake_average_passband.ipynb`:
   - load all `ext/` cases from `data/bandpass_coupled_line_filter_export/ext/index.json`
   - evaluate new fixed-band objective on `4.8-5.2 GHz`
   - plot all EXT curves and a separate block for satisfying-only curves
   - print satisfying case count and filenames.
2. Use satisfying-region evidence to set objective in online BO notebook.
3. Run `notebooks/bo_online_experiment.ipynb` with updated `BCLFOptimizationSpec` passband settings.

### Current Observation Under New Band

- In EXT set (184 valid cases), strict full-band feasibility under the new objective is sparse but non-zero:
  - satisfying cases: `2`
- This matches the intended regime: narrow feasible region is acceptable.

---

## TR-BO A/B Update (2026-05-06)

### Variant Definition

- Baseline: existing `OptimizerBoTorch` global-discrete `qLogEI` selection.
- TR-BO: single-region trust-region mode (TuRBO-1 style) in the same optimizer:
  - local candidate filtering in normalized space (`L_inf` trust box),
  - adaptive trust-region length expansion/shrink by success/failure streaks,
  - fallback to global pool if trust-region slice is empty.

### Fixed A/B Setup

- Notebook: `notebooks/bo_online_experiment.ipynb`
- Backend: queue-backed online simulator (`server`)
- Seed/budget: `SEED=0`, `N_INIT=8`, `BUDGET=40`
- Objective settings (same for both variants):
  - passband `4.8-5.2 GHz`
  - passband constraints `S21>=-5 dB`, `S11<=-7 dB`
  - lower stopband `3.6-4.8 GHz`
  - upper stopband `5.8-8.7 GHz`

### Artifact Naming and Outputs

- Run IDs:
  - baseline: `bo_online_general_seed0_baseline`
  - trust-region: `bo_online_general_seed0_trbo1`
- Output family per variant:
  - checkpoint: `experiments/bo_online_general/<run_id>.checkpoint.json`
  - recorder outputs: `<run_id>.json` and `<run_id>.csv`
- Notebook comparison outputs:
  - objective-by-iteration overlay,
  - best-so-far overlay,
  - compact summary table (`final best`, `first feasible iter`, `cache_hit_ratio`, runtime totals).

---

## Offline BNN Surrogate Feasibility

- experiment_id: `bclf_bnn_offline_surrogate_v1`
- status: `promising-feasibility`
- primary notebook: `notebooks/bnn_bclf_offline_feasibility.ipynb`
- output root: `experiments/bnn_bclf_offline_feasibility/`

### Research Question

Can the legacy Bayesian neural network learn BCLF `S11`/`S21` responses from
the cached Touchstone dataset well enough to serve as a response surrogate?

### Current Setup

- Dataset: `data/bandpass_coupled_line_filter_export/ext`
- Split: 80% train, 20% test
- Model: `modeling.LegacyBNNResponseSurrogate`
- Target mode: direct dB prediction, `target_mode="db"`
- Inputs: geometry variables plus frequency
- Default training: 1000 epochs, batch size 1024, hidden dims `[256, 256, 256]`

### Current Observation

The dB-trained BNN surrogate reproduces the main response-curve structure in
the current train comparison, including the `S11` notch and `S21` passband
shape. This is a useful feasibility signal for surrogate modeling, but it is
not yet an online optimization result.

Full experiment details and artifact paths are documented in
[BNN Surrogate Experiments](../bnn_surrogate_experiments.md).

---

## Experiment Card: Online BO Seed Campaign

- experiment_id: `bo_online_seed_campaign_v1`
- status: `paper-candidate`
- owner: `team`
- last_updated_utc: `2026-04-29`

### Research Question / Hypothesis

- Can queue-backed online BO reliably improve objective values across multiple seeds under real simulator latency constraints?

### Method Variant

- `OptimizerBoTorch` with queue-backed simulator.
- Completion policy for campaign summary rows should require target trial count:
  - `num_trials >= N_INIT + BUDGET`

### Dataset / Simulator Source

- Real simulator via server/worker queue flow.
- Source context:
  - [../server_worker_submit_receive_flow.md](../server_worker_submit_receive_flow.md)
  - [../simulation_interface.md](../simulation_interface.md)

### Config References

- Notebook:
  - `notebooks/bo_online_campaign.ipynb` (campaign config and run cells).
- Output root:
  - `experiments/bo_online/`

### Current Optimization Goal (BCLF, BO Online)

Let design vector be `x`, and let `S11(f, x)`, `S21(f, x)` be simulated responses in dB.

Optimization target (spec-defined penalty objective):

```text
Given S11_dB(f), S21_dB(f):

a_mean = mean_{f in Bp} S21_dB(f)
b_mean = mean_{f in Bp} S11_dB(f)
c_mean = mean_{f in (Bsl union Bsu)} S21_dB(f)

a_worst = min_{f in Bp} S21_dB(f)
b_worst = max_{f in Bp} S11_dB(f)
c_worst = max_{f in (Bsl union Bsu)} S21_dB(f)

S_score = a_mean - 0.5 * b_mean - 0.5 * c_mean
P = max(0, -7 - a_worst) + max(0, b_worst + 7) + max(0, c_worst + 18)
J = S_score - 5.0 * P

Optimization uses this penalty-augmented objective directly.
No explicit constraints are passed to BoTorch for this BCLF variant.
```

Band definitions:
- passband `Bp = [5.2 GHz, 5.9 GHz]`
- lower stopband `Bsl = [3.6 GHz, 4.8 GHz]`
- upper stopband `Bsu = [6.5 GHz, 8.7 GHz]`

Source of truth:
- `notebooks/bo_online_campaign.ipynb` objective config cell (`BCLFOptimizationSpec(...).to_objective()`).

### Seed Policy and Seeds Used

- Policy: fixed campaign seed list, run pending seeds with resume support.
- Known seeds in current artifacts: `0, 1, 2` completed/partial histories, `3` in progress artifacts.

### Artifact Links

- Campaign summary:
  - `experiments/bo_online/campaign_summary.csv`
- Per-seed files:
  - `experiments/bo_online/bo_online_seed0.json`
  - `experiments/bo_online/bo_online_seed1.json`
  - `experiments/bo_online/bo_online_seed2.json`
  - `experiments/bo_online/bo_online_seed0.csv`
  - `experiments/bo_online/bo_online_seed1.csv`
  - `experiments/bo_online/bo_online_seed2.csv`

### Inclusion Rule for Paper Figures/Tables

- Include only seeds with:
  - summary `status == completed`
  - and `num_trials >= N_INIT + BUDGET`

### Known Caveats

- Existing summary rows may contain stale status from older notebook logic and should be refreshed before final figure generation.

---

## Experiment Card: BO Replay Grid Study

- experiment_id: `bo_replay_grid_v1`
- status: `paper-candidate`
- owner: `team`
- last_updated_utc: `2026-04-29`

### Research Question / Hypothesis

- How do acquisition settings (for example UCB/EI variants and init counts) change optimization performance trends across seeds?

### Method Variant

- Replay/grid sweeps over acquisition hyperparameters and initialization counts.
- Multi-seed comparison based on aggregated summary artifacts.

### Dataset / Simulator Source

- Replay/offline experiment artifacts under `experiments/bo_replay/`.

### Config References

- Replay analysis notebook:
  - `notebooks/bo_replay_experiment.ipynb`
- Replay/grid artifact context:
  - `experiments/bo_replay/optimization_grid_runs.csv`
  - `experiments/bo_replay/optimization_grid_summary.csv`
  - per-run files `grid_*.json` and `grid_*.csv`

### Current Optimization Goal (Replay, aligned with BO Online)

- Replay now uses the same objective constructor and parameterization as BO online:
  - `BCLFOptimizationSpec(...).to_objective()`
  - `passband_ghz=(5.2 GHz, 5.9 GHz)`
  - `lower_stopband_ghz=(3.6 GHz, 4.8 GHz)`
  - `upper_stopband_ghz=(6.5 GHz, 8.7 GHz)`
  - `min_passband_s21_db=-7.0`
  - `max_passband_s11_db=-7.0`
  - `max_stopband_s21_db=-18.0`
  - `s11_weight=0.5`
  - `stopband_weight=0.5`
- Objective is penalty-augmented S-parameter score (`J` above), with no explicit BoTorch constraints for this BCLF variant.
- In this no-constraint mode, recorder field `feasible` is `None` (not `True/False`) and should not be used as a convergence signal.
- Convergence signal for comparison is `best_so_far`.

### BO vs Random Baseline

- `notebooks/bo_replay_experiment.ipynb` now includes an in-notebook baseline comparison:
  - `OptimizerBoTorch` (BoTorch, qLogEI-based acquisition, with `DiscreteCandidateSpace` for replay-style pools)
  - `OptimizerRandom` (same candidate-space/simulator/spec/budget/seed contracts)
- Comparison plots include per-iteration `objective` and `best_so_far`.

### BO vs Random Figure Observation (Evidence of BO Effectiveness)

- In the per-iteration objective plot, BO and Random are both non-monotonic, but BO samples lower-objective regions more consistently while Random shows larger variance and more high-objective spikes.
- In the `best_so_far` convergence plot (the primary optimization metric), both methods improve quickly in early iterations (to about `-27.x`), but only BO continues to improve afterward.
- BO reaches a lower final best value (about `-29.6`) than Random (about `-27.6`) under the same budget/seed setup.
- Interpretation: BO’s benefit is not that every single trial is better, but that model-guided selection produces better cumulative best-so-far improvement than random sampling.

### Best-Result Band Comparison Figure

![Online vs Replay best full-band comparison](assets/online_vs_replay_best_fullband.png)

### Observations

- Both best results keep the strongest transmission behavior inside passband (`~5.2-5.9 GHz`), consistent with the `a = min S21(passband)` term.
- Replay-best shows a broader high-transmission region in passband for `S21`, while online-best drops faster near the upper passband edge.
- In low stopband (`3.6-4.8 GHz`), online-best remains noticeably closer to 0 dB than replay-best, which is consistent with larger low-stopband violation in online runs.
- In high stopband (`6.5-8.7 GHz`), both curves attenuate strongly; online-best reaches slightly deeper attenuation at the far high-frequency end.
- `S11` behavior differs by shape: replay-best has deeper/narrower notches in passband, while online-best is smoother but with less extreme minima; both remain near 0 dB outside passband.
- Overall, the visual gap is dominated by low-stopband leakage and passband-width/shape differences.

### Seed Policy and Seeds Used

- Policy: seed sweeps per setting.
- Observed seeds in grid files: `1..5` across multiple configurations.

### Artifact Links

- Aggregates:
  - `experiments/bo_replay/optimization_grid_summary.csv`
  - `experiments/bo_replay/optimization_grid_runs.csv`
- Latest replay notebook run:
  - `experiments/bo_replay/bo_replay_experiment_latest_nb.json`
  - `experiments/bo_replay/bo_replay_experiment_latest_nb.csv`
- Example fixed-band summaries:
  - `experiments/bo_replay/fixed_ucb_beta2_n0_objective_band.csv`
  - `experiments/bo_replay/fixed_ucb_beta2_n2_objective_band.csv`

### Inclusion Rule for Paper Figures/Tables

- For each compared method setting, include all configured seeds in that setting.
- Use the same aggregation rule across all settings in a figure/table.

### Known Caveats

- Ensure comparisons do not mix run families with different objective definitions unless explicitly labeled as such.

---

## Experiment Card: Fake Simulator Repeatability (One-vs-One)

- experiment_id: `fake_sim_repeatability_bclf_v1`
- status: `paper-candidate`
- owner: `team`
- last_updated_utc: `2026-05-06`

### Research Question / Hypothesis

- For the same BCLF geometry parameters, does the expected generated result in `ext/` reproduce the original exported response shape (`S11`, `S21`)?

### Method Variant

- One-vs-one comparison only:
  - baseline file from `data/bandpass_coupled_line_filter_export/` (outside `ext`)
  - expected parameter-matched file from `data/bandpass_coupled_line_filter_export/ext/`
- Compare `S11` and `S21` in dB after parsing Touchstone format (`RI` / `MA` / `DB`).

### Dataset / Simulator Source

- Original export dataset:
  - `data/bandpass_coupled_line_filter_export/*.s2p`
- Generated/ext dataset:
  - `data/bandpass_coupled_line_filter_export/ext/*.s2p`
- Notebook driver:
  - `notebooks/fake_simulator_repeatability_experiment.ipynb`

### Config References

- Primary notebook:
  - `notebooks/fake_simulator_repeatability_experiment.ipynb`
- BCLF queue template reference used by the same notebook flow:
  - `bclf_template_v1`

### Seed Policy and Seeds Used

- Seed policy: not applicable (deterministic file-pair comparison).
- Selection policy: first valid baseline case with required vars `L, L1, L2, W, W1, W2, S1, S2`; expected ext path is derived from those values.

### Artifact Links

- Notebook:
  - `notebooks/fake_simulator_repeatability_experiment.ipynb`
- Compared pair (example reported case):
  - `data/bandpass_coupled_line_filter_export/Bandpass_Coupled_Line_Filter_export_Bandpass_Coupled_Line_Filter_L_7d7_L1_8d0_L2_7d8_W_0d9_W1_0d7_W2_0d9_S1_0d13_S2_0d55.s2p`
  - `data/bandpass_coupled_line_filter_export/ext/Bandpass_Coupled_Line_Filter_export_Bandpass_Coupled_Line_Filter_L_7d7_L1_8d0_L2_7d8_W_0d9_W1_0d7_W2_0d9_S1_0d13_S2_0d55.s2p`
- Summary note:
  - `docs/fake_simulator_repeatability_summary.md`

### Inclusion Rule for Paper Figures/Tables

- Include only one-vs-one pairs where:
  - required BCLF vars are parseable from baseline filename,
  - expected `ext` file exists for the exact parameter-derived name,
  - numeric row shapes match and frequency grids align.

### Known Caveats

- Matching parameter names does not guarantee bitwise-identical responses; observed differences are strongest around `S11` notch depth.
- Header metadata differences (for example `Sweep` vs `Sweep1`, generation timestamp) are not used as inclusion/exclusion criteria.
