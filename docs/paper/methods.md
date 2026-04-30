# Methods

## Problem Setup

We optimize parameterized HFSS design variables for bandpass filter performance under expensive simulator evaluations.

Core outcomes:
- Maximize objective derived from passband/stopband and center-frequency requirements.
- Minimize simulator calls while maintaining design quality.

## System Components

- Optimization layer (`src/optimization/`):
  - `OptimizerBoTorch` for queue-backed real simulation flow.
  - Replay/grid workflows for offline comparative studies.
- Simulation layer:
  - Queue-backed server/worker execution for real HFSS evaluations.
  - Optional replay/fake simulators for controlled experiments.
- Result recording:
  - Per-run `json/csv/checkpoint` artifacts in `experiments/`.

## Method Portfolio (Extensible)

This section is intentionally structured for multiple methods. Add new method families as peer subsections, not as edits inside BO-only text.

### M1: Bayesian Optimization (current baseline family)

High-level BO loop:
1. Initialize with `N_INIT` random or seeded trials.
2. Fit/update GP surrogate.
3. Select next candidate by acquisition policy (for example UCB/EI variants).
4. Evaluate candidate (queue-backed simulator or replay simulator).
5. Record trial history and update best-so-far objective.

### M2: Future Method Slot

- Method name: `TBD`
- Core idea: `TBD`
- Why it differs from M1: `TBD`
- Expected advantage/risk: `TBD`

### M3: Future Method Slot

- Method name: `TBD`
- Core idea: `TBD`
- Why it differs from M1/M2: `TBD`
- Expected advantage/risk: `TBD`

Objective details should be defined in experiment configs/notebooks and cited in each experiment card.

## Interface Boundaries

- External tool interactions (HFSS, queue I/O) are treated as adapter-layer responsibilities.
- Paper-facing method claims should reference:
  - algorithm settings in notebooks/configs,
  - and concrete output artifacts under `experiments/`.

## Assumptions and Constraints

- Stochastic behavior is controlled by explicit seed management per run family.
- Campaign completion criteria for online experiments should be evaluated by target trial count (`N_INIT + BUDGET`) rather than status alone.

## Method Comparison Hooks

Use these hooks when adding non-BO methods:
- Shared objective definition across compared methods (or explicitly documented differences).
- Shared seed policy and reporting protocol.
- Shared artifact contract (`json/csv` summary + per-run history links).

## Current Objective Spec Reference (BO Online, BCLF)

- The current BO Online BCLF objective is documented in:
  - [experiments.md](experiments.md), section `Current Optimization Goal (BCLF, BO Online)`.
- When the objective changes, update that section first, then update claim mappings in `results_index.md`.

## Pointers to Operational Docs

- Queue and SFTP behavior: [../sftp_implementation.md](../sftp_implementation.md)
- Server/worker flow: [../server_worker_submit_receive_flow.md](../server_worker_submit_receive_flow.md)
- Simulation API context: [../simulation_interface.md](../simulation_interface.md)
