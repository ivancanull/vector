# Scripts

## Server: Submit Spiral Batch

Submit one batch file and emit one per-case job JSON into `queue/pending`.

```powershell
.\.venv\Scripts\python.exe scripts\server\submit_spiral_batch.py --batch scripts\server\examples\spiral_batch.example.json
```

Optional queue root override:

```powershell
.\.venv\Scripts\python.exe scripts\server\submit_spiral_batch.py --batch <path-to-batch.json> --queue-root queue
```

## Server: Submit Five Random Cases And Watch

Generate five random spiral cases, submit them to `queue/pending`, then watch for the worker-uploaded results.

```bash
cd /home/zhanghf/projects/vector
PYTHONPATH=src:. python scripts/server/submit_random_spiral_watch.py \
  --queue-root queue \
  --count 5 \
  --seed 123 \
  --timeout-sec 7200 \
  --poll-interval-sec 10
```

See `docs/server_worker_coherent_test.md` for the paired Windows worker command and troubleshooting notes.

## Worker: SFTP Queue Loop

Continuously claim remote server jobs over SFTP, run each job locally, upload the result bundle, and archive uploaded local result JSONs.

```powershell
.\.venv\Scripts\python.exe scripts\worker\run_queue_loop_sftp.py `
  --worker-config configs\worker.spiral_local.yaml `
  --server-host 10.15.89.240 `
  --server-port 22031 `
  --username zhanghf `
  --remote-queue-root /home/zhanghf/projects/vector/queue
```

Default SSH files are `%USERPROFILE%\.ssh\id_ed25519` and `%USERPROFILE%\.ssh\known_hosts` on Windows. Add `--ssh-key` and `--known-hosts` only when using non-default paths.

Use `--once` to process at most one stale local job, one unuploaded local result, or one newly claimed remote job.

## Worker: Run One Fetched Job

Run one explicit local `queue/running/*.json` job. This is the supported local debugging path after the legacy local `pending -> running` worker was removed.

```powershell
.\.venv\Scripts\python.exe scripts\server\submit_spiral_batch.py --batch scripts\server\examples\spiral_batch.localrun.json --queue-root queue
Move-Item queue\pending\spiral_demo_local_c01.json queue\running\spiral_demo_local_c01.json -Force
.\.venv\Scripts\python.exe scripts\worker\run_fetched_spiral_job.py --job-file queue\running\spiral_demo_local_c01.json --worker-config configs\worker.spiral_local.yaml
```

Expected outputs:
- `queue/done/<job_id>.result.json` on success
- `queue/failed/<job_id>.result.json` on failure
- artifacts/logs under `queue/results/`

## Queue Outputs

- Submitted job files: `queue/pending/*.json`
- Claimed job files: `queue/running/*.json`
- Success results: `queue/done/*.result.json`
- Failure results: `queue/failed/*.result.json`
- Artifacts/logs: `queue/results/*`
