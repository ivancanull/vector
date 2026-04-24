from __future__ import annotations

from worker.utils import WorkerError

IMPLEMENTED_METRICS = {"max_s11_db", "min_s21_db"}


def _max_s11_db(network, np) -> float:
    s11_db = 20.0 * np.log10(np.abs(network.s[:, 0, 0]))
    return float(np.max(s11_db))


def _min_s21_db(network, np) -> float:
    if network.number_of_ports < 2:
        raise WorkerError(
            "metric_extraction_failure", "min_s21_db requires at least 2 ports"
        )
    s21_db = 20.0 * np.log10(np.abs(network.s[:, 1, 0]))
    return float(np.min(s21_db))


def compute_metrics(touchstone_path, metrics: list[str], job_context: dict | None = None) -> dict[str, float]:
    try:
        import numpy as np
        import skrf as rf

        network = rf.Network(str(touchstone_path))
    except Exception as exc:
        raise WorkerError(
            "metric_extraction_failure", f"Failed to parse touchstone: {exc}"
        ) from exc

    output: dict[str, float] = {}
    for metric in metrics:
        if metric not in IMPLEMENTED_METRICS:
            raise WorkerError(
                "metric_extraction_failure", f"Metric not implemented yet: {metric}"
            )
        if metric == "max_s11_db":
            output[metric] = _max_s11_db(network, np)
        elif metric == "min_s21_db":
            output[metric] = _min_s21_db(network, np)

    return output
