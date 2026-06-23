# Server-Worker Job Contract

For an end-to-end server-worker smoke workflow, including the SFTP worker loop
and five-random-case submit/watch script, see
`server_worker_coherent_test.md`.

## Canonical Worker Job JSON

Each file in `queue/pending/*.json` must represent exactly one worker case.

Required fields:
- `job_id` (string)
- `template_name` (string): experimental object id
- `variables` (object of string -> string with explicit units)
- `output_filename` (string, Touchstone style such as `.s2p`)
- `submit_time` (ISO8601 string)
- `metrics_to_extract` (non-empty string array)

Optional compatibility fields:
- `design_name`
- `setup_name`
- `sweep_name`
- `frequency_points_ghz`: optional explicit GHz points for this job. When
  provided, the worker sorts and deduplicates the list, rejects empty,
  non-finite, or non-positive values, and replaces the template sweep with a
  PyAEDT single-point sweep over exactly those points.
- `artifact_options`
- `retry_count`

Example:

```json
{
  "job_id": "batch_20260422_c01",
  "template_name": "si_template_v1",
  "variables": {
    "Din": "55um",
    "H": "2um",
    "S": "1.5um",
    "W": "15um"
  },
  "output_filename": "spiral_Din55_S1d5_W15.s2p",
  "submit_time": "2026-04-22T08:00:00Z",
  "metrics_to_extract": ["max_s11_db", "min_s21_db"],
  "design_name": "HFSSDesign1",
  "setup_name": "SI",
  "sweep_name": "Sweep1",
  "frequency_points_ghz": [3.5, 4.2, 5.0],
  "artifact_options": {"export_touchstone": true, "export_csv": true},
  "retry_count": 0
}
```

## Batch Submission (Server Input)

Server-side producers can submit one batch object and let `QueueManager.enqueue_batch(...)` emit one worker file per case.

```json
{
  "batch_id": "batch_20260422",
  "template_name": "si_template_v1",
  "metrics_to_extract": ["max_s11_db", "min_s21_db"],
  "design_name": "HFSSDesign1",
  "setup_name": "SI",
  "sweep_name": "Sweep1",
  "cases": [
    {
      "case_id": "c01",
      "variables": {"Din": "55um", "H": "2um", "S": "1.5um", "W": "15um"},
      "output_filename": "spiral_c01.s2p"
    },
    {
      "case_id": "c02",
      "variables": {"Din": "60um", "H": "2um", "S": "1.2um", "W": "14um"},
      "output_filename": "spiral_c02.s2p"
    }
  ]
}
```

Queue emission from this example:
- `queue/pending/batch_20260422_c01.json`
- `queue/pending/batch_20260422_c02.json`

## Template Files

Ready-to-copy JSON templates are provided in `templates/`:

- `templates/queue_job_template.json`
- `templates/queue_batch_submit_template.json`
- `templates/queue_result_done_template.json`
- `templates/queue_result_failed_template.json`

Usage notes:

- Copy the queue job template into `queue/pending/<job_id>.json` for single-case runs.
- Use the batch template with `scripts/server/submit_spiral_batch.py` to emit per-case pending jobs.
- `queue_result_done_template.json` and `queue_result_failed_template.json` document the expected result payloads under `queue/done` and `queue/failed`.

## Naming and Unit Rules

- `variables` values must include explicit units (`55um`, `1.0mm`, `2.4GHz`).
- `output_filename` must be a Touchstone filename (`.sNp` style; `.s2p` recommended).
- Worker uses `output_filename` as the primary touchstone artifact name.
- If `frequency_points_ghz` is omitted, the template's configured HFSS sweep
  remains the source of truth.

## Frequency Points Contract Check

Run the local non-live contract check from the repo root:

```powershell
.\.venv\Scripts\python.exe scripts\worker\check_frequency_points_contract.py
```

This check imports the installed AEDT Python package, verifies the PyAEDT API
signatures used by the worker, validates job normalization, verifies queue JSON
emission, and checks the worker setup-designer call shape. It does not open AEDT
or mutate `.aedt` projects.

## UAW/AFS 20-Point Timing Smoke

The UAW/AFS frequency-point smoke notebook is
`notebooks/uaw_afs_frequency_points_contract.ipynb`. It samples 20 deterministic
frequency points from an existing BCLF Touchstone grid and submits a worker job
with `frequency_points_ghz`.

Measured on the BCLF template:

| Run | Job | Frequency rows | HFSS solve time | Worker total time |
| --- | --- | ---: | ---: | ---: |
| Full sweep, same geometry | `turbo_online_general_seed0_it0067` | template sweep | 395 s (6m35s) | about 395 s |
| UAW/AFS explicit points | `uaw_afs_Bandpass_Coupled_Line_Filter_export_Bandpass_Cou_fcaa3c35_seed0_n20` | 20 | 221 s (3m41s) | 222.6 s (3m42.6s) |

For this same-geometry comparison, the 20-point job saved 174 s, reducing solve
time by about 44.1 percent and giving about 1.79x speedup. Across 1129 historical
BCLF full-sweep logs without `frequency_points_ghz`, the full-sweep median solve
time was 397 s (6m37s), with mean 419.7 s and interquartile range 351-482 s.
The single-case vector-fitting reconstruction record is in
[`studies/uaw_afs/reports/uaw_afs_single_case_README.md`](../../../studies/uaw_afs/reports/uaw_afs_single_case_README.md).
