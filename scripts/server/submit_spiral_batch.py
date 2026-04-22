from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.job_types import BatchCase, BatchSimulationJob  # noqa: E402
from common.units import has_explicit_unit  # noqa: E402
from server.queue_manager import QueueManager  # noqa: E402


def _is_touchstone_name(value: str) -> bool:
    suffix = Path(value).suffix.lower()
    return suffix.startswith(".s") and suffix.endswith("p")


def _require_non_empty_string(payload: dict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _parse_metrics(payload: dict) -> list[str]:
    metrics = payload.get("metrics_to_extract")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("metrics_to_extract must be a non-empty list")
    parsed: list[str] = []
    for metric in metrics:
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("metrics_to_extract entries must be non-empty strings")
        parsed.append(metric.strip())
    return parsed


def _parse_case(case_payload: dict, index: int) -> BatchCase:
    if not isinstance(case_payload, dict):
        raise ValueError(f"cases[{index}] must be an object")

    case_id_raw = case_payload.get("case_id")
    case_id = str(index)
    if isinstance(case_id_raw, str) and case_id_raw.strip():
        case_id = case_id_raw.strip()

    variables = case_payload.get("variables")
    if not isinstance(variables, dict) or not variables:
        raise ValueError(f"cases[{index}].variables must be a non-empty object")

    parsed_variables: dict[str, str] = {}
    for key, value in variables.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"cases[{index}].variables keys must be non-empty strings")
        if not isinstance(value, str) or not has_explicit_unit(value):
            raise ValueError(f"cases[{index}].variables[{key!r}] must include explicit units")
        parsed_variables[key] = value

    output_filename = _require_non_empty_string(case_payload, "output_filename")
    if not _is_touchstone_name(output_filename):
        raise ValueError(f"cases[{index}].output_filename must be a Touchstone file name")

    return BatchCase(case_id=case_id, variables=parsed_variables, output_filename=output_filename)


def parse_batch_payload(payload: dict) -> BatchSimulationJob:
    if not isinstance(payload, dict):
        raise ValueError("Batch payload must be a JSON object")

    batch_id = _require_non_empty_string(payload, "batch_id")
    template_name = _require_non_empty_string(payload, "template_name")
    metrics_to_extract = _parse_metrics(payload)

    cases_raw = payload.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("cases must be a non-empty list")

    cases = [_parse_case(case, idx) for idx, case in enumerate(cases_raw)]

    artifact_options = payload.get("artifact_options", {})
    if artifact_options is None:
        artifact_options = {}
    if not isinstance(artifact_options, dict):
        raise ValueError("artifact_options must be an object when provided")

    retry_count = payload.get("retry_count", 0)
    if not isinstance(retry_count, int) or retry_count < 0:
        raise ValueError("retry_count must be a non-negative integer")

    design_name = payload.get("design_name")
    setup_name = payload.get("setup_name")
    sweep_name = payload.get("sweep_name")

    for field_name, field_value in {
        "design_name": design_name,
        "setup_name": setup_name,
        "sweep_name": sweep_name,
    }.items():
        if field_value is not None and (not isinstance(field_value, str) or not field_value.strip()):
            raise ValueError(f"{field_name} must be a non-empty string when provided")

    return BatchSimulationJob(
        batch_id=batch_id,
        template_name=template_name,
        cases=cases,
        metrics_to_extract=metrics_to_extract,
        design_name=design_name,
        setup_name=setup_name,
        sweep_name=sweep_name,
        artifact_options=artifact_options,
        retry_count=retry_count,
    )


def submit_batch(batch_path: Path, queue_root: Path) -> list[Path]:
    with batch_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    batch = parse_batch_payload(payload)
    manager = QueueManager(queue_root)
    return manager.enqueue_batch(batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a spiral inductor batch JSON to the queue as per-case jobs."
    )
    parser.add_argument("--batch", type=Path, required=True, help="Path to batch JSON file.")
    parser.add_argument(
        "--queue-root",
        type=Path,
        default=REPO_ROOT / "queue",
        help="Queue root directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    emitted = submit_batch(args.batch, args.queue_root)
    for path in emitted:
        print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
