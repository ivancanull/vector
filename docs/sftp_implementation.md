# SSH/SFTP-Based Cross-Network File Exchange for Windows HFSS + Linux Orchestrator

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

---

## Upload Script (Worker -> Server)

Use the worker-side upload script to send one completed job bundle to the server over SFTP.

### Required setup

- SSH private key file on worker machine (for user `zhanghf`)
- `known_hosts` entry containing `10.15.89.240:22031` host key

### Command

Upload latest result bundle from local queue:

```powershell
.\.venv\Scripts\python.exe scripts\worker\upload_latest_result_sftp.py `
  --queue-root D:\vector\queue `
  --server-host 10.15.89.240 `
  --server-port 22031 `
  --username zhanghf `
  --remote-queue-root /home/zhanghf/projects/vector/queue `
  --ssh-key <path-to-private-key> `
  --known-hosts <path-to-known_hosts>
```

Upload a specific result JSON:

```powershell
.\.venv\Scripts\python.exe scripts\worker\upload_latest_result_sftp.py `
  --result-json D:\vector\queue\done\<job_id>.result.json `
  --ssh-key <path-to-private-key> `
  --known-hosts <path-to-known_hosts>
```

### Uploaded file set per job

- result JSON (`done/*.result.json` or `failed/*.result.json`)
- referenced artifacts from result payload when present:
  - `results/*.s2p` (or other `.sNp`)
  - `results/*.csv`
  - `results/*.log`

All files are uploaded by mirroring local queue relative paths under:

- `/home/zhanghf/projects/vector/queue`
