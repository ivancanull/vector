# Server-Worker Job Contract

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
