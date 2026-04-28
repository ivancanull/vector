# Server-Worker Coherent Test Workflow

This guide documents two linked workflows for testing the Linux server queue and Windows HFSS worker together.

## Idea 1: Worker SFTP Queue Loop

The worker loop runs on the Windows HFSS machine. It claims jobs from the Linux server queue over SFTP, runs each job locally, uploads the result bundle, and archives uploaded local result JSONs.

Recommended Windows command:

```powershell
cd D:\vector
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\worker\run_queue_loop_sftp.py `
  --worker-config configs\worker.yaml `
  --server-host 10.15.89.240 `
  --server-port 22031 `
  --username zhanghf `
  --remote-queue-root /home/zhanghf/projects/vector/queue
```

Default SSH files are `%USERPROFILE%\.ssh\id_ed25519` and `%USERPROFILE%\.ssh\known_hosts` on Windows. Add `--ssh-key` and `--known-hosts` only when the worker uses different paths.

One-iteration smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\worker\run_queue_loop_sftp.py `
  --worker-config configs\worker.yaml `
  --server-host 10.15.89.240 `
  --server-port 22031 `
  --username zhanghf `
  --remote-queue-root /home/zhanghf/projects/vector/queue `
  --once
```

Queue lifecycle:

- Server writes jobs to `queue/pending/*.json`.
- Worker claims a job by renaming remote `pending/<job>.json` to remote `running/<job>.json`.
- Worker downloads the claimed job to local `queue/running/<job>.json`.
- Worker writes local `queue/done/<job>.result.json` or `queue/failed/<job>.result.json`.
- Worker uploads the result JSON and referenced artifacts to the server queue.
- Worker archives uploaded local result JSONs under `queue/archive/done` or `queue/archive/failed`.

Timeout and stale handling:

- A subprocess timeout writes a normal failed result with `error_message="job_timeout"`.
- A stale local running job writes a normal failed result with `error_message="stale_running_job"`.
- These failures use the same result contract as HFSS failures, so server-side tools can parse them.

## Idea 2: Server Five Random Cases Submit-and-Watch

The server random test script generates five spiral inductor cases, submits them to `queue/pending`, watches only those submitted job IDs, and prints a final console summary.

Recommended Linux command:

```bash
cd /home/zhanghf/projects/vector
PYTHONPATH=src:. python scripts/server/submit_random_spiral_watch.py \
  --queue-root queue \
  --count 5 \
  --seed 123 \
  --timeout-sec 7200 \
  --poll-interval-sec 10
```

Default random ranges are intentionally near the checked-in examples:

- `Din`: `50-70um`
- `H`: fixed `2um`
- `S`: `1.0-2.0um`
- `W`: `12-18um`

Result parsing:

- The watcher checks `queue/done/<job_id>.result.json` and `queue/failed/<job_id>.result.json`.
- It ignores unrelated result files from other batches.
- It exits with code `0` when every submitted job has either a done or failed result.
- It exits with code `1` if timeout is reached before all submitted jobs return.

## End-to-End Test Procedure

1. On the Linux server, inspect the queue:

```bash
cd /home/zhanghf/projects/vector
find queue -maxdepth 2 -type f | sort
```

2. On the Windows worker, start the worker loop command from Idea 1.

3. On the Linux server, submit and watch five random cases:

```bash
PYTHONPATH=src:. python scripts/server/submit_random_spiral_watch.py \
  --queue-root queue \
  --count 5 \
  --seed 123 \
  --timeout-sec 7200 \
  --poll-interval-sec 10
```

4. Interpret the final summary:

- `done` means HFSS finished and metrics were parsed.
- `failed` means the worker returned a parseable failure result.
- `missing` means the server timed out before receiving a result JSON.

Fixed-batch fallback:

```bash
cd /home/zhanghf/projects/vector
PYTHONPATH=src:. python scripts/server/submit_spiral_batch.py \
  --batch scripts/server/examples/spiral_batch.example.json \
  --queue-root queue
```

## Troubleshooting

- Missing SSH key: verify `--ssh-key` points to a readable private key on the Windows worker.
- Missing known hosts: create a `known_hosts` file containing the `10.15.89.240:22031` host key.
- No claimed jobs: confirm the server has files under `queue/pending` and the worker `--remote-queue-root` points to `/home/zhanghf/projects/vector/queue`.
- Stuck running jobs: run the worker loop with `--once`; stale local running jobs are converted into failed results after `--stale-running-sec`.
- Missing artifacts: inspect the uploaded result JSON `artifacts` paths and check matching files under `queue/results`.
- `pytest` unavailable: install test dependencies in the active environment or use `python -m py_compile` for syntax checks.
