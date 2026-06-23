# SSH/SFTP-Based Cross-Network File Exchange for Windows HFSS + Linux Orchestrator

For the current coherent test workflow, use
`server_worker_coherent_test.md`. It includes the long-running worker SFTP
loop command and the server-side five-random-case submit/watch command.

For implementation status and next planned tasks, see
`../development_log.md`.

## Goal

Use **SSH/SFTP** instead of Samba for cross-network file exchange between:

- **Linux server**: scheduler / optimizer / training side
- **Windows PC**: HFSS worker side

This design is recommended when the system should:

- avoid exposing SMB port **445**
- work reliably **across networks**
- provide a **safer** transport layer
- reduce SMB compatibility and Windows UNC path issues

---

## Why Prefer SSH/SFTP Over Samba

### Advantages

1. **Simpler firewall model**
   - usually only one port is needed
   - custom SSH port is supported

2. **Safer transport**
   - encrypted connection
   - key-based authentication
   - easier access control

3. **Better fit for task-based workflows**
   - Linux generates tasks
   - Windows pulls tasks
   - Windows runs HFSS locally
   - Windows pushes results back

4. **Avoid SMB-specific issues**
   - no dependency on port `445`
   - no Windows UNC path issues like `\\server\share`
   - fewer credential/cache issues on Windows

---

## Custom Server Information

Use the following server settings for this project:

- Host Name: `10.15.89.240`
- User: `zhanghf`
- Port: `22031`
- Remote directory: `/home/zhanghf/projects/vector/queue`
- Local directory: `.../queue`

Recommended local example for this repo:

- Local directory: `D:/vector/queue`

---

## Queue Workflow Mapping

1. Server writes jobs to remote queue directory:
   - `/home/zhanghf/projects/vector/queue/pending/*.json`
2. Worker fetches/syncs jobs to local queue directory:
   - `.../queue/pending/*.json`
3. Worker runs simulation and exports touchstone/csv locally:
   - `.../queue/results/*`
4. Worker writes status:
   - `.../queue/done/*.result.json` or `.../queue/failed/*.result.json`

Remote lifecycle for one-shot worker pipeline:

1. Worker claims one remote pending job by moving it to `running/`.
2. Worker downloads that claimed JSON to local `queue/running/`.
3. Worker runs simulation locally and writes local done/failed result.
4. Worker uploads only that job bundle back to remote queue root.
5. Server reads the uploaded result and archives the remote `running/<job>.json`.

---

## Upload Script (Worker -> Server)

Use the worker-side upload script to send one completed job bundle to the server over SFTP.

For normal testing, prefer `scripts/worker/run_queue_loop_sftp.py`; it claims
jobs, runs HFSS, uploads result bundles, and archives completed local result
JSONs in one loop. Remote running job JSONs are left for the server to archive.

### Required setup

- SSH private key file on worker machine (for user `zhanghf`), defaulting to `%USERPROFILE%\.ssh\id_ed25519`
- `known_hosts` entry containing `10.15.89.240:22031` host key, defaulting to `%USERPROFILE%\.ssh\known_hosts`

### Command

Upload latest result bundle from local queue:

```powershell
.\.venv\Scripts\python.exe scripts\worker\upload_latest_result_sftp.py `
  --queue-root D:\vector\queue `
  --server-host 10.15.89.240 `
  --server-port 22031 `
  --username zhanghf `
  --remote-queue-root /home/zhanghf/projects/vector/queue
```

Upload a specific result JSON:

```powershell
.\.venv\Scripts\python.exe scripts\worker\upload_latest_result_sftp.py `
  --result-json D:\vector\queue\done\<job_id>.result.json
```

### Uploaded file set per job

- result JSON (`done/*.result.json` or `failed/*.result.json`)
- referenced artifacts from result payload when present:
  - `results/*.s2p` (or other `.sNp`)
  - `results/*.csv`
  - `results/*.log`

All files are uploaded by mirroring local queue relative paths under:

- `/home/zhanghf/projects/vector/queue`

---

## Remote Pipeline Command

Continuously fetch remote pending jobs, run them, and upload result bundles. The server archives each remote running claim after reading its result:

```powershell
.\.venv\Scripts\python.exe scripts\worker\fetch_run_submit_spiral_job.py `
  --worker-config configs\worker.yaml `
  --server-host 10.15.89.240 `
  --server-port 22031 `
  --username zhanghf `
  --ssh-key C:\Users\Huifan\.ssh\id_ed25519_vector `
  --known-hosts C:\Users\Huifan\.ssh\known_hosts `
  --remote-queue-root /home/zhanghf/projects/vector/queue
```

Add `--once` for a one-job smoke test.
