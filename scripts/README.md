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

## Worker: Run One Queued Job

Claim exactly one job (`pending -> running`), run HFSS, export touchstone, write result JSON, then exit.

```powershell
.\.venv\Scripts\python.exe scripts\worker\spiral_inductor_run_one.py --worker-config configs\worker.yaml
```

## Queue Outputs

- Submitted job files: `queue/pending/*.json`
- Claimed job files: `queue/running/*.json`
- Success results: `queue/done/*.result.json`
- Failure results: `queue/failed/*.result.json`
- Artifacts/logs: `queue/results/*`
