# Scripts

Notebook-specific runners live in `notebooks/scripts/`; see
`notebooks/scripts/README.md`.

## Server: Submit Spiral Batch

Submit one batch file and emit one per-case job JSON into `queue/pending`.

```powershell
.\.venv\Scripts\python.exe scripts\server\submit_spiral_batch.py --batch templates\queue_batch_submit_template.json
```

Optional queue root override:

```powershell
.\.venv\Scripts\python.exe scripts\server\submit_spiral_batch.py --batch <path-to-batch.json> --queue-root queue
```

## Worker: Validate New Spiral Template Cases Locally

Run fixed `spiral_inductor_t3_template_v1` cases through the same local queue
job path used by BCLF. The script opens the existing AEDT template and changes
job variables; it does not regenerate the AEDT file.

```powershell
.\.venv\Scripts\python.exe scripts\worker\validate_spiral_t3_queue_cases.py `
  --worker-config configs\worker.yaml `
  --queue-root .runtime\local_spiral_t3_parallel_validation
```

Preview the fixed batch without touching the queue:

```powershell
.\.venv\Scripts\python.exe scripts\worker\validate_spiral_t3_queue_cases.py --dry-run
```

## Server: Submit Random Cases And Watch (Unified)

Submit random cases and watch worker results using one script selected by template.
Ranges are built into the script (not CLI inputs).

SI / spiral example:

```bash
cd /home/zhanghf/projects/vector
PYTHONPATH=src:. python scripts/server/run_random_template_watch.py \
  --template-name si_template_v1 \
  --queue-root queue \
  --count 5 \
  --seed 123 \
  --timeout-sec 7200 \
  --poll-interval-sec 10
```

BCLF example:

```bash
cd /home/zhanghf/projects/vector
PYTHONPATH=src:. python scripts/server/run_random_template_watch.py \
  --template-name bclf_template_v1 \
  --queue-root queue \
  --count 5 \
  --seed 123 \
  --timeout-sec 7200 \
  --poll-interval-sec 10
```

New spiral template example:

```bash
cd /home/zhanghf/projects/vector
PYTHONPATH=src:. python scripts/server/run_random_template_watch.py \
  --template-name spiral_inductor_t3_template_v1 \
  --queue-root queue \
  --count 5 \
  --seed 123 \
  --timeout-sec 7200 \
  --poll-interval-sec 10
```

This script does not copy/export artifacts. Results stay in queue directories for external transfer scripts.

See `docs/server_worker_coherent_test.md` for the paired Windows worker command and troubleshooting notes.

## Remote Fetch + Run + Submit

Continuously fetch jobs from remote `pending`, run locally, and upload each result bundle. The server owns archiving remote `running/<job>.json` files after it reads the results.

```powershell
.\.venv\Scripts\python.exe scripts\worker\fetch_run_submit_job.py `
  --worker-config configs\worker.yaml `
  --server-host 10.15.89.240 `
  --server-port 22031 `
  --username zhanghf `
  --ssh-key C:\Users\Huifan\.ssh\id_ed25519_vector `
  --known-hosts C:\Users\Huifan\.ssh\known_hosts `
  --remote-queue-root /home/zhanghf/projects/vector/queue `
  --sftp-connect-attempts 5 `
  --sftp-connect-retry-delay-sec 5
```

When remote `pending` is empty, this script waits quietly and only prints periodic idle status. Add `--once` to process one job and exit, or `--no-wait-when-empty` to exit immediately when no job is pending.
Transient SFTP connection failures are retried before the worker gives up; use
`--sftp-connect-attempts`, `--sftp-connect-timeout-sec`, and
`--sftp-connect-retry-delay-sec` to tune unstable VPN/proxy links.

## Remote Fetch + Run + Submit (Parallel, Isolated AEDT Copies)

Launch multiple worker slots without changing the original worker entrypoint.
Each slot uses its own local queue/log/results directories and each job opens a
fresh copy of the golden AEDT template, so concurrent jobs do not write the same
`.aedt/.aedtresults/.lock` files.

```powershell
.\.venv\Scripts\python.exe scripts\worker\fetch_run_submit_job_parallel.py `
  --worker-config configs\worker.yaml `
  --slots 2 `
  --slot-root .runtime\parallel_worker_slots `
  --server-host 10.15.89.240 `
  --server-port 22031 `
  --username zhanghf `
  --ssh-key C:\Users\Huifan\.ssh\id_ed25519_vector `
  --known-hosts C:\Users\Huifan\.ssh\known_hosts `
  --remote-queue-root /home/zhanghf/projects/vector/queue `
  --sftp-connect-attempts 5 `
  --sftp-connect-retry-delay-sec 5
```

Add `--once` to let each slot claim at most one remote job. To roll back, stop
the parallel launcher and run the single-worker `fetch_run_submit_job.py`
command above.

## Worker: Live Local Spiral Parallel Validation

Run the fixed new-spiral cases locally with isolated AEDT copies and export
Touchstone files under `.runtime/`, keeping the server-facing `queue/`
directory untouched.

```powershell
$stamp = Get-Date -Format yyyyMMdd_HHmmss
.\.venv\Scripts\python.exe scripts\worker\validate_spiral_t3_queue_cases.py `
  --worker-config configs\worker.yaml `
  --queue-root ".runtime\live_spiral_t3_parallel_$stamp" `
  --batch-id "live_spiral_t3_parallel_$stamp" `
  --parallel-slots 2 `
  --slot-root ".runtime\live_spiral_t3_parallel_slots_$stamp"
```

## Inductor: Notebook-First Local Candidate Run

Use `notebooks/inductor_local_candidate_space_run.ipynb` for the first Spiral
Inductor dataset pass. The notebook defines the `CandidateSpace`, writes local
queue jobs under `.runtime/`, optionally launches `run_local_queue_worker.py`,
and analyzes exported `.s2p` files. HFSS still runs only through the worker.

The notebook exports Touchstone files directly to:

```text
D:/ansys/spiral_inductor_s_parameters_export/ext
```

The generated filenames follow the legacy export style:

```text
spiral_inductor_example_HFSSDesign000_Din_60_S_1d5_W_15.s2p
```

Script version, single-thread worker:

```powershell
.\.venv\Scripts\python.exe scripts\worker\run_inductor_local_candidate_space.py `
  --count 2 `
  --seed 0 `
  --run-worker
```

Omit `--run-worker` to only generate pending jobs and summary scaffolding.
By default, this script writes the local offline queue to `.runtime/queue`,
with `pending/running/done/failed/results` matching the server-facing `queue/`
layout. You do not need to pass a run id; the script creates a timestamped job
prefix and writes summaries under `.runtime/queue/results`. Touchstone files
default to `D:/ansys/spiral_inductor_s_parameters_export/ext`.
Initial Inductor candidates are snapped to `Din=0.1um`, `W=0.1um`,
`S=0.1um`, with `H=2um` fixed.

To process existing local pending jobs directly:

```powershell
.\.venv\Scripts\python.exe scripts\worker\run_local_queue_worker.py --no-wait
```

## Worker: Cleanup Test Results

Delete local result JSON/artifact files only for jobs whose `job_id` starts with `test_` (keeps `exp_*`).

```powershell
.\.venv\Scripts\python.exe scripts\worker\cleanup_test_results.py --queue-root queue --job-prefix test_
```

Preview only (no deletion):

```powershell
.\.venv\Scripts\python.exe scripts\worker\cleanup_test_results.py --queue-root queue --job-prefix test_ --dry-run
```

## Queue Outputs

- Submitted job files: `queue/pending/*.json`
- Claimed job files: `queue/running/*.json`
- Success results: `queue/done/*.result.json`
- Failure results: `queue/failed/*.result.json`
- Artifacts/logs: `queue/results/*`

## Notebook: Online BO Campaign

Use `notebooks/bo_online_campaign.ipynb` for real-simulator BO campaigns with resume support.

- Each execution runs all pending seeds (or only `FORCE_SEED` when set).
- Completed seeds are tracked in `experiments/bo_online/campaign_summary.csv`.
- Per-seed checkpoints are kept as `experiments/bo_online/bo_online_seed{seed}.checkpoint.json`.
- Re-run the notebook later to continue remaining seeds automatically.
