# HFSS Worker Side Design (Windows Worker)

## Purpose

The HFSS worker runs on Windows and executes simulations requested by the Linux master.

## Responsibilities

- poll pending jobs from the shared queue
- run HFSS simulation for each job
- write result payloads back to `queue/results/`
- move finished jobs to `queue/done/` or `queue/failed/`

## KISS Implementation Rules

- One job in, one HFSS run out: keep control flow linear and explicit.
- Validate and parse input once, then execute once, then persist result once.
- Keep worker wrappers thin around PyAEDT (`open -> set vars -> solve -> export -> close`).
- Fail fast on invalid input or solve/export errors; always write one failure result.
- Keep logs actionable: include job id, setup/sweep target, and failure class.
- Prefer one shared helper for repeated side effects (for example failure result persistence).

## Minimal Execution Flow

1. Claim one JSON job from `queue/running/`.
2. Parse and validate the job payload.
3. Resolve template file + design + setup + sweep.
4. Execute HFSS once:
   open session -> apply variables -> solve setup -> export touchstone/csv -> close session.
5. Compute metrics from touchstone.
6. Persist exactly one result JSON (`done` or `failed`) and one log artifact.

## Worker Module Layout (KISS)

- `src/worker/worker_main.py`: orchestration loop and result persistence.
- `src/worker/hfss_api.py`: thin PyAEDT execution wrapper.
- `src/worker/job_loader.py`: JSON load and schema validation.
- `src/worker/template_registry.py`: template lookup and resolution.
- `src/worker/state_manager.py`: write done/failed result state.
- `src/worker/metrics.py`: post-process touchstone metrics.
- `src/worker/param_strings.py`: recorded-script style parameter/export naming helpers.

## Contract

- input job file: JSON in `queue/running/*.json`
- output result file: JSON in `queue/results/*.json`
- required job fields: `job_id`, `template_name`, `variables`, `output_filename`, `submit_time`, `metrics_to_extract`
