# Development Log

This document tracks implementation status, operational decisions, and near-term plans for the server-worker HFSS queue workflow.

## Current Decisions

- Remote queue ownership: the server owns final archiving of remote `running/<job>.json` after it reads uploaded results.
- Worker queue ownership: the Windows worker may archive local result JSONs and local running job JSONs after upload.
- Job naming: use `test_` job IDs for disposable tests and `exp_` job IDs for experiment outputs that should be kept.
- Local cleanup safety: cleanup tools should delete only explicitly prefixed jobs and referenced artifacts under the selected `queue_root`.

## Implemented

### SFTP Remote Job Flow

- Worker claims remote jobs by renaming remote `pending/<job>.json` to remote `running/<job>.json`.
- Worker downloads claimed jobs to local `queue/running/`.
- Worker runs HFSS locally and writes local `queue/done/*.result.json` or `queue/failed/*.result.json`.
- Worker uploads result JSONs plus referenced artifacts (`s2p`, `csv`, `log`) back to the remote queue root.
- Worker no longer archives remote `running/<job>.json`; this is now server-managed.
- `queue_loop.py` is now the single worker-side SFTP queue implementation.
- SFTP upload bundle logic now lives inside `SftpResultUploader`; the old `upload_job_bundle_via_sftp` function API has been removed.

### Fetch-Run-Submit Worker

- `scripts/worker/fetch_run_submit_spiral_job.py` continuously waits for remote pending jobs by default.
- Added `--once` for one-job smoke tests.
- Added quieter idle status output:
  - first idle message prints `status=waiting_for_remote_pending`.
  - repeated idle messages are throttled by `--idle-status-interval-sec`.
- Added stale local result cleanup before rerunning a claimed job ID.

### Local Retry and Debug Flow

- `scripts/worker/run_fetched_spiral_job.py` removes stale local result files before first run and before retry.
- The retry path can correct common setup/sweep mismatch failures by forcing `Setup1` / `Sweep`.
- Verified a previous `SI` / `Sweep1` export failure succeeds after retry with `Setup1` / `Sweep`.

### Test/Experiment Output Separation

- Added `scripts/worker/cleanup_test_results.py`.
- It deletes only result JSONs whose `job_id` starts with a configurable prefix, defaulting to `test_`.
- It deletes referenced artifacts only if they are inside the selected `queue_root`.
- It supports `--dry-run`.

### Documentation

- Updated SFTP docs to reflect server-managed remote archive behavior.
- Updated script README for continuous fetch-run-submit behavior and cleanup commands.
- Added this development log as the living implementation/planning tracker.

## Known Issues and Watch Items

- Old remote jobs may still contain stale setup/sweep values such as `SI` / `Sweep1`; prefer new jobs using template defaults or `Setup1` / `Sweep`.
- The one-shot fetch-run-submit script is now a continuous worker by default; use `--once` when debugging a single job.
- Remote `running/<job>.json` files should remain until the server confirms result ingestion and archives them.
- Local queue may contain previous development results; use `cleanup_test_results.py --dry-run` before applying cleanup.

## Planned

- Add or update server-side logic to detect uploaded result JSONs and archive matching remote `running/<job>.json`.
- Add server-side tests for result ingestion and remote running archive behavior.
- Consider a single documented queue lifecycle diagram covering server pending, worker running, result upload, and server archive.
- Add a recommended naming convention for `test_`, `exp_`, and batch job IDs in the job contract docs.
- Consider a worker-side option to disable stale result cleanup for protected experiment prefixes.

## Useful Commands

Run the continuous fetch-run-submit worker:

```powershell
.\.venv\Scripts\python.exe scripts\worker\fetch_run_submit_spiral_job.py `
  --worker-config configs\worker.spiral_local.yaml `
  --server-host 10.15.89.240 `
  --server-port 22031 `
  --username zhanghf `
  --ssh-key C:\Users\Huifan\.ssh\id_ed25519_vector `
  --known-hosts C:\Users\Huifan\.ssh\known_hosts `
  --remote-queue-root /home/zhanghf/projects/vector/queue
```

Run one job and exit:

```powershell
.\.venv\Scripts\python.exe scripts\worker\fetch_run_submit_spiral_job.py `
  --worker-config configs\worker.spiral_local.yaml `
  --server-host 10.15.89.240 `
  --server-port 22031 `
  --username zhanghf `
  --ssh-key C:\Users\Huifan\.ssh\id_ed25519_vector `
  --known-hosts C:\Users\Huifan\.ssh\known_hosts `
  --remote-queue-root /home/zhanghf/projects/vector/queue `
  --once
```

Preview cleanup of test outputs:

```powershell
.\.venv\Scripts\python.exe scripts\worker\cleanup_test_results.py --queue-root queue --job-prefix test_ --dry-run
```

Apply cleanup of test outputs:

```powershell
.\.venv\Scripts\python.exe scripts\worker\cleanup_test_results.py --queue-root queue --job-prefix test_
```
