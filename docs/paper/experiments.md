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
