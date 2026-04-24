# Server/Worker Submit-Receive Flow

This document summarizes how jobs are submitted by the server, received by workers, executed, uploaded, and observed back on the server.

## Protocol Illustration (Text)

```text
                 HFSS Job Queue Protocol (Text Diagram)
                 ======================================

 SERVER SIDE                                              WORKER SIDE
 -----------                                              -----------

 +------------------------------+
 | submit_spiral_batch.py       |
 | / QueueManager.enqueue_*     |
 +--------------+---------------+
                | write job JSON
                v
        queue/pending/<job_id>.json
                |
                | (SFTP list + atomic rename = claim lock)
                |----------------------------------------------+
                |                                              |
                v                                              v
        remote pending/                                  remote running/
       <job_id>.json    -- rename -->                 <job_id>.json
                                                           |
                                                           | sftp.get
                                                           v
                                                  local queue/running/
                                                     <job_id>.json
                                                           |
                                                           | ClaimedJobProcessor.process()
                                                           | HfssJobExecutor.execute() + metrics
                                                           v
                                 +--------------------------------------------------+
                                 | local result persistence                         |
                                 | - queue/done/<job_id>.result.json   OR          |
                                 | - queue/failed/<job_id>.result.json             |
                                 | - queue/results/<artifacts: sNp/csv/log>        |
                                 +--------------------------------------------------+
                                                           |
                                                           | SftpResultUploader.upload()
                                                           | (upload result JSON + artifacts)
                                                           v
                         remote queue root mirrors local relative paths:
                         - done/<job_id>.result.json or failed/<job_id>.result.json
                         - results/<artifact files>
                                                           |
                                                           | worker prints:
                                                           | remote_archive=server_managed
                                                           v
 +----------------------------------------------------------------------------------+
 | SERVER WATCH/INGEST (e.g. submit_random_spiral_watch.py)                        |
 | - poll done/failed result JSON                                                   |
 | - read status + metrics + artifacts                                              |
 | - archive remote running/<job_id>.json                                           |
 +----------------------------------------------------------------------------------+
```
## 1) Server Submits Jobs

- The server creates worker-compatible JSON jobs under `queue/pending/`.
- `QueueManager.enqueue(...)` writes one job file: `queue/pending/<job_id>.json`.
- `QueueManager.enqueue_batch(...)` expands batch cases into one pending file per case.
- Main submit entrypoints: `scripts/server/submit_spiral_batch.py`, `scripts/server/submit_random_spiral_watch.py` (submit + watch).

Job payload fields written for workers include:
- `job_id`
- `template_name`
- `variables`
- `output_filename`
- `metrics_to_extract`
- optional HFSS overrides (`design_name`, `setup_name`, `sweep_name`)

## 2) Worker Receives (Claims) Jobs

There are two active worker entry styles, both using shared queue logic:

- `scripts/worker/fetch_run_submit_spiral_job.py`
- This path pulls remote jobs via SFTP from `pending/`, claims by rename to `running/`, downloads locally into `queue/running/`, runs in-process via `ClaimedJobProcessor.process(...)`, then uploads result bundles.
- `scripts/worker/run_queue_loop_sftp.py`
- This path uses `WorkerLoopService` from `src/worker/queue_loop.py`, claims via `SftpRemoteQueueClient.claim_one()`, runs via `SubprocessJobRunner` (`run_fetched_spiral_job.py`), and applies loop archive policies.
- Core claim implementation: `src/worker/queue_loop.py` -> `SftpRemoteQueueClient.claim_one()`.

## 3) Worker Executes and Persists Result

- Worker execution orchestration: `src/worker/worker_main.py` -> `ClaimedJobProcessor.process(...)`.
- Job loading/validation: `load_job(...)`.
- HFSS solve/export: `HfssJobExecutor.execute(...)`.
- Metric extraction: `compute_metrics(...)`.
- Result persistence:
- Success -> `queue/done/<job_id>.result.json`.
- Failure -> `queue/failed/<job_id>.result.json`.
- Shared write path: `src/worker/state_manager.py`.
- On result write, the local `queue/running/<job>.json` is removed.
- Artifacts are written in `queue/results/` and referenced from result JSON (`touchstone`, `csv`, `log`).

## 4) Worker Uploads Result Bundle

- Upload logic is centralized in `src/worker/queue_loop.py` -> `SftpResultUploader.upload(...)`.
- Upload bundle includes result JSON (`done/*.result.json` or `failed/*.result.json`) and referenced artifacts from `artifacts` (`touchstone`, `csv`, `log`).
- Files are uploaded by mirroring queue-relative paths to remote queue root.

## 5) Server Receives Results

- Server-side watcher scripts read `queue/done` and `queue/failed` for completion.
- Example: `scripts/server/submit_random_spiral_watch.py` polls expected job IDs and prints summaries.
- Ownership rule in current flow: worker does not archive remote `running/<job>.json`; server reads uploaded results and archives remote running claims.

## 6) Queue State Lifecycle

- Submit: `pending/<job>.json` (server)
- Claim: `running/<job>.json` (worker claims by rename)
- Execute: success -> `done/<job>.result.json`; failure -> `failed/<job>.result.json`
- Artifacts: `results/<...>`
- Upload: worker pushes done/failed result + artifacts back to remote queue root
- Archive: remote running archive is server-managed; local archive behavior depends on worker entrypoint/service mode.

## 7) Key Files

- Server submit/write: `src/server/queue_manager.py`, `scripts/server/submit_spiral_batch.py`, `scripts/server/submit_random_spiral_watch.py`.
- Worker claim/run/upload: `src/worker/queue_loop.py`, `scripts/worker/fetch_run_submit_spiral_job.py`, `scripts/worker/run_queue_loop_sftp.py`, `scripts/worker/run_fetched_spiral_job.py`, `src/worker/worker_main.py`, `src/worker/state_manager.py`.
