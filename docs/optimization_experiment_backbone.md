# Optimization Experiment Backbone

This page summarizes the reusable experiment structure for adding optimization algorithms beyond the current BO, ANN, and random baselines.

## Core Contract

All optimization algorithms should plug into the same OOP backbone:

- `OptimizerBase` owns the experiment loop, checkpointing, resume behavior, timing, best-so-far tracking, and recorder integration.
- Subclasses implement `plan_next(iteration)` and return one or more `SelectionResult` objects.
- `CandidateSpace` owns geometry domains, remaining-candidate bookkeeping, and normalization/denormalization.
- `Simulator` owns evaluation, whether replay/offline, HFSS adapter, or queue-backed server execution.
- `OptimizationSpec` owns objective and metric evaluation from `(freqs, s_params)`.
- `ExperimentRecorder` writes the shared JSON/CSV artifact contract.

This means new algorithms should usually be new `OptimizerBase` subclasses, not new experiment loops.

## Data Flow

The shared loop is:

1. Choose initial samples through `CandidateSpace.sample_initial(...)` for `N_INIT`.
2. Ask the algorithm subclass for candidates through `plan_next(...)`.
3. Evaluate candidates through `Simulator.simulate(...)`.
4. Convert simulator output to objective/metrics through `OptimizationSpec.evaluate(...)`.
5. Update candidate-space removals, best objective, and run state.
6. Save checkpoint after every iteration.
7. Finalize JSON/CSV artifacts through `ExperimentRecorder`.

The optimizer minimizes `objective`. `best_so_far` is recorded per trial and should be the primary convergence trace for comparing methods.

## Extension Points

Use these classes as the stable boundaries when adding another algorithm:

- `OptimizerBase`: subclass for the algorithm policy. Keep model fitting, acquisition, evolutionary state, RL policy state, or heuristic scoring inside the subclass.
- `SelectionResult`: return requested geometry plus optional diagnostics such as predicted mean, uncertainty, acquisition score, or algorithm-specific `selection_debug`.
- `CandidateSpace`: reuse `DiscreteCandidateSpace` for replay pools and `ContinuousCandidateSpace` for sampled bounded online campaigns. Add a new candidate-space class only when the domain representation changes.
- `Simulator`: keep external calls behind adapters. Algorithm code should not call queue, HFSS, Touchstone, or dataset APIs directly.
- `OptimizationSpec`: keep objectives shared across methods unless the experiment card explicitly documents a different goal.
- `ExperimentRecorder`: preserve recorder fields so downstream plots can compare `objective`, `best_so_far`, timing, cache hits, and backend metadata across algorithms.

## Current Algorithm Families

- `OptimizerBoTorch`: GP/BoTorch optimizer with logEI-style discrete acquisition.
- `OptimizerANN`: response-surrogate optimizer using an ANN/TCN model and the same base loop. It stores ANN training sidecar data next to the checkpoint.
- `OptimizerRandom`: random baseline over the same candidate-space/simulator/spec contract.

Any new method should set a unique `optimizer_backend` string so CSV/JSON outputs can be grouped reliably.

## Campaign Pattern

Multi-seed experiments should use the same campaign rules:

- Run an explicit seed list, for example `[0, 1, 2, 3, 4]`.
- Use one `run_id` and checkpoint path per seed.
- Treat a seed as completed only when history length is at least `N_INIT + BUDGET`.
- Resume incomplete checkpoints instead of restarting them.
- Write a campaign summary CSV with seed, run ID, status, number of trials, best objective, timing, cache-hit ratio, and artifact paths.
- Aggregate method performance by `best_so_far` over iteration, not only by final objective.

This pattern keeps online campaigns robust to interruptions and makes offline/replay comparisons reproducible.

## Artifact Contract

Every method should produce:

- Per-run checkpoint: resumable state, candidate-space state, history, timing, simulator stats, and RNG state.
- Per-run JSON: config, summary, and full trial list.
- Per-run CSV: tabular trial history with common recorder fields.
- Optional campaign summary CSV: one row per seed/run.

Required common trial fields for method comparison:

- `iteration`, `step`, `phase`
- `requested_geometry`, `geometry`, `geometry_normalized`
- `objective`, `best_so_far`, `feasible`, `constraint_margin_min`
- `predicted_mean`, `predicted_uncertainty`, `acquisition_score` when meaningful
- `optimizer_backend`, `selection_debug`
- `simulation_status`, `simulation_error`, `cache_hit`, `job_id`, `case_id`
- `t_engine_sec`, `t_simulator_sec`, cumulative timing fields

New method-specific diagnostics should go into `selection_debug` unless they need first-class CSV columns.

## Implementation Checklist for a New Algorithm

1. Create a subclass such as `OptimizerMyMethod(OptimizerBase)`.
2. Set `optimizer_backend = "my_method"`.
3. Add algorithm parameters to `__init__`, validate them, then call `super().__init__(...)`.
4. Implement `plan_next(iteration)` using `self._history`, `self.state`, and `self.candidate_space.remaining()`.
5. Return `SelectionResult(requested_geometry=..., selection_debug={...})`.
6. Override `collect_post_simulation_data`, `resume_from_checkpoint`, or `_save_checkpoint` only if the method has extra state that must survive resume.
7. Add focused tests that use `ReplaySimulator` or a small fake simulator so the method is exercised without queue/HFSS dependencies.
8. Add or update a notebook/script campaign that instantiates only the optimizer subclass while reusing the same candidate space, simulator, spec, recorder, seed policy, and artifact layout.

## Comparison Rules

For fair comparisons across algorithms:

- Keep objective/spec, candidate bounds, budget, seed list, and simulator backend identical unless the experiment card documents the difference.
- Compare best-so-far curves and final best objective by seed.
- Report failed simulations separately from algorithm failures.
- Keep raw external tool calls inside simulator or worker adapters, not optimizer subclasses.
- Do not fork the experiment loop for convenience; extend the backbone unless a new evaluation mode truly requires a different lifecycle.

For the current BCLF online campaign target, geometry bounds, and BO/LHS
comparison settings, see
[BCLF Optimization Target And Online Settings](bclf_optimization_target.md).

For the offline BNN response-surrogate feasibility study, see
[BNN Surrogate Experiments](bnn_surrogate_experiments.md).
