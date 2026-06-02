from pathlib import Path

import pytest

from common.io_utils import read_json
from common.job_types import BatchCase, BatchSimulationJob, SimulationJob
from server.queue_manager import QueueManager


def test_enqueue_writes_canonical_worker_job_json(tmp_path: Path) -> None:
    queue = QueueManager(tmp_path / "queue")
    job = SimulationJob(
        job_id="job_001",
        template_name="si_template_v1",
        variables={"Din": "55um", "S": "1.5um", "W": "15um"},
        output_filename="spiral_Din55_S1d5_W15.s2p",
        metrics_to_extract=["max_s11_db"],
        submit_time="2026-04-22T08:00:00Z",
        design_name="HFSSDesign1",
        frequency_points_ghz=[3.5, 4.2, 5.0],
    )

    path = queue.enqueue(job)
    payload = read_json(path)
    assert payload["job_id"] == "job_001"
    assert payload["template_name"] == "si_template_v1"
    assert payload["variables"]["Din"] == "55um"
    assert payload["output_filename"] == "spiral_Din55_S1d5_W15.s2p"
    assert payload["metrics_to_extract"] == ["max_s11_db"]
    assert payload["frequency_points_ghz"] == [3.5, 4.2, 5.0]
    loaded = queue.load_job(path)
    assert loaded.frequency_points_ghz == [3.5, 4.2, 5.0]


def test_enqueue_batch_splits_cases_into_pending_files(tmp_path: Path) -> None:
    queue = QueueManager(tmp_path / "queue")
    batch = BatchSimulationJob(
        batch_id="batch_20260422",
        template_name="si_template_v1",
        metrics_to_extract=["max_s11_db", "min_s21_db"],
        frequency_points_ghz=[3.5, 4.2, 5.0],
        cases=[
            BatchCase(
                case_id="c01",
                variables={"Din": "55um", "S": "1.5um", "W": "15um"},
                output_filename="spiral_c01.s2p",
            ),
            BatchCase(
                case_id="c02",
                variables={"Din": "60um", "S": "1.2um", "W": "14um"},
                output_filename="spiral_c02.s2p",
            ),
        ],
    )

    emitted = queue.enqueue_batch(batch)
    assert [p.name for p in emitted] == ["batch_20260422_c01.json", "batch_20260422_c02.json"]
    payload0 = read_json(emitted[0])
    payload1 = read_json(emitted[1])
    assert payload0["job_id"] == "batch_20260422_c01"
    assert payload0["output_filename"] == "spiral_c01.s2p"
    assert payload0["frequency_points_ghz"] == [3.5, 4.2, 5.0]
    assert payload1["frequency_points_ghz"] == [3.5, 4.2, 5.0]
    assert payload1["job_id"] == "batch_20260422_c02"
    assert payload1["variables"]["Din"] == "60um"


def test_enqueue_batch_validates_required_fields(tmp_path: Path) -> None:
    queue = QueueManager(tmp_path / "queue")
    batch = BatchSimulationJob(
        batch_id="",
        template_name="si_template_v1",
        metrics_to_extract=["max_s11_db"],
        cases=[
            BatchCase(
                case_id="c01",
                variables={"Din": "55um"},
                output_filename="spiral_c01.s2p",
            )
        ],
    )
    with pytest.raises(ValueError):
        queue.enqueue_batch(batch)
