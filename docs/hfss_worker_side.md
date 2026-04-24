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
- When calling PyAEDT functions, check the official PyAEDT API docs instead of relying on memory.
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

## Worker Code Structure Tree

```text
scripts/worker/
  fetch_run_submit_spiral_job.py    # long-running SFTP worker: claim remote job, run, upload result
  run_queue_loop_sftp.py            # object-oriented SFTP loop wrapper and smoke-test entrypoint
  run_fetched_spiral_job.py         # local debug runner for an already-fetched queue/running job
  upload_latest_result_sftp.py      # manual upload helper for latest or explicit result JSON
  cleanup_test_results.py           # safe cleanup for disposable test_* outputs

src/worker/
  worker_main.py                    # ClaimedJobProcessor: load, resolve, execute, metrics, persist
  hfss_api.py                       # HfssJobExecutor/session classes: open, set variables, solve, export
  artifact_exporter.py              # local artifact paths, CSV export, artifact verification
  metrics.py                        # touchstone metric extraction

  schemas.py                        # Pydantic contracts: job, template, worker config, results
  config.py                         # YAML config loading
  job_loader.py                     # job JSON load/validation and validation error mapping
  template_registry.py              # template lookup and job-specific design/setup/sweep resolution

  state_manager.py                  # writes done/failed result JSONs and removes local running job
  logger.py                         # global/job log setup
  utils.py                          # WorkerError, timestamps, queue-relative paths
  param_strings.py                  # spiral parameter/export name helpers

  queue_loop.py                     # SFTP worker service, local queue store, subprocess runner, uploader
```

### Core Runtime Path

- `fetch_run_submit_spiral_job.py` and `run_queue_loop_sftp.py` are the remote SFTP worker entrypoints.
- `run_fetched_spiral_job.py` is the local single-job debug path.
- `worker_main.ClaimedJobProcessor.process()` is the canonical per-job execution path.
- `hfss_api.HfssJobExecutor.execute()` is the only place that should drive PyAEDT/HFSS execution.
- `queue_loop.SftpResultUploader` owns uploading result bundles over SFTP.

### Queue Loop Components

- `SftpRemoteQueueClient`: claims remote pending jobs into remote `running/` and downloads them locally.
- `FileLocalQueueStore`: finds stale local jobs/results and archives local queue files after upload.
- `SubprocessJobRunner`: runs `run_fetched_spiral_job.py` in a subprocess with timeout handling.
- `SftpResultUploader`: uploads result JSON plus referenced artifacts (`touchstone`, `csv`, `log`).
- `WorkerLoopService`: coordinates stale-job handling, unuploaded-result handling, remote claiming, running, uploading, and local archiving.

## Contract

- input job file: JSON in `queue/running/*.json`
- output result file: JSON in `queue/done/*.result.json` or `queue/failed/*.result.json`
- artifact files: `queue/results/*`
- required job fields: `job_id`, `template_name`, `variables`, `output_filename`, `submit_time`, `metrics_to_extract`
