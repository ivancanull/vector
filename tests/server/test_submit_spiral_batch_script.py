from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.io_utils import read_json
from scripts.server.submit_spiral_batch import parse_batch_payload, submit_batch



def _valid_batch_payload() -> dict:
    return {
        "batch_id": "spiral_batch_01",
        "template_name": "si_template_v1",
        "design_name": "HFSSDesign1",
        "setup_name": "Setup1",
        "sweep_name": "Sweep",
        "metrics_to_extract": ["max_s11_db", "min_s21_db"],
        "cases": [
            {
                "case_id": "c01",
                "variables": {
                    "Din": "55um",
                    "H": "2um",
                    "S": "1.5um",
                    "W": "15um",
                },
                "output_filename": "spiral_c01.s2p",
            },
            {
                "case_id": "c02",
                "variables": {
                    "Din": "60um",
                    "H": "2um",
                    "S": "1.3um",
                    "W": "14um",
                },
                "output_filename": "spiral_c02.s2p",
            },
        ],
    }



def test_parse_batch_payload_missing_required_field_fails() -> None:
    payload = _valid_batch_payload()
    del payload["template_name"]
    with pytest.raises(ValueError):
        parse_batch_payload(payload)



def test_parse_batch_payload_invalid_units_fails() -> None:
    payload = _valid_batch_payload()
    payload["cases"][0]["variables"]["Din"] = "55"
    with pytest.raises(ValueError):
        parse_batch_payload(payload)



def test_parse_batch_payload_invalid_output_filename_fails() -> None:
    payload = _valid_batch_payload()
    payload["cases"][0]["output_filename"] = "spiral_c01.txt"
    with pytest.raises(ValueError):
        parse_batch_payload(payload)



def test_submit_batch_emits_one_job_per_case(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps(_valid_batch_payload()), encoding="utf-8")

    emitted = submit_batch(batch_path, queue_root)
    assert [p.name for p in emitted] == ["spiral_batch_01_c01.json", "spiral_batch_01_c02.json"]

    first_payload = read_json(emitted[0])
    second_payload = read_json(emitted[1])

    assert first_payload["job_id"] == "spiral_batch_01_c01"
    assert first_payload["output_filename"] == "spiral_c01.s2p"
    assert second_payload["job_id"] == "spiral_batch_01_c02"
    assert second_payload["variables"]["Din"] == "60um"
    assert second_payload["template_name"] == "si_template_v1"
    assert second_payload["metrics_to_extract"] == ["max_s11_db", "min_s21_db"]
