from pathlib import Path

from src.common.io_utils import read_json

from .db import ExperimentDB


class ResultIngestor:
    def __init__(self, db: ExperimentDB, result_dir: str | Path = "queue/results"):
        self.db = db
        self.result_dir = Path(result_dir)

    def ingest_one(self, result_path: str | Path) -> None:
        payload = read_json(result_path)
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO simulation_runs (job_id, design_type, status, metrics_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    metrics_json=excluded.metrics_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    payload["job_id"],
                    payload.get("design_type", "unknown"),
                    payload.get("status", "done"),
                    str(payload.get("metrics", {})),
                ),
            )
