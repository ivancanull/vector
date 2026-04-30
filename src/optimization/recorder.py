from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class RecorderOutput:
    json_path: Path
    csv_path: Path


class ExperimentRecorder:
    def __init__(self, output_dir: str | Path = "experiments/bo_replay") -> None:
        self.output_dir = Path(output_dir)
        self._config: dict[str, Any] = {}
        self._trials: list[dict[str, Any]] = []

    def start(self, config: dict[str, Any]) -> None:
        self._config = dict(config)
        self._trials = []

    def record_trial(self, trial: dict[str, Any]) -> None:
        self._trials.append(dict(trial))

    def finalize(self, run_id: str, summary: dict[str, Any]) -> RecorderOutput:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "run_id": run_id,
            "config": _to_jsonable(self._config),
            "summary": _to_jsonable(summary),
            "trials": _to_jsonable(self._trials),
        }

        json_path = self.output_dir / f"{run_id}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

        csv_path = self.output_dir / f"{run_id}.csv"
        self._write_trials_csv(csv_path, payload["trials"])

        return RecorderOutput(json_path=json_path, csv_path=csv_path)

    def _write_trials_csv(self, path: Path, trials: list[dict[str, Any]]) -> None:
        base_fields = [
            "iteration",
            "step",
            "phase",
            "dataset_index",
            "requested_geometry",
            "geometry",
            "geometry_normalized",
            "case_id",
            "objective",
            "objective_array",
            "constraints_array",
            "predicted_mean",
            "predicted_uncertainty",
            "acquisition_score",
            "feasible",
            "constraint_margin_min",
            "best_so_far",
            "cache_hit",
            "job_id",
            "simulation_status",
            "simulation_error",
            "t_engine_sec",
            "t_simulator_sec",
            "t_iteration_sec",
            "t_engine_cum_sec",
            "t_simulator_cum_sec",
            "optimizer_backend",
            "selection_debug",
        ]
        metric_names: list[str] = []
        objective_component_names: list[str] = []
        constraint_metric_names: list[str] = []
        constraint_violation_names: list[str] = []
        for row in trials:
            for name in row.get("metrics", {}).keys():
                if name not in metric_names:
                    metric_names.append(name)
            for name in row.get("objective_components", {}).keys():
                if name not in objective_component_names:
                    objective_component_names.append(name)
            for name in row.get("constraint_metrics", {}).keys():
                if name not in constraint_metric_names:
                    constraint_metric_names.append(name)
            for name in row.get("constraint_violations", {}).keys():
                if name not in constraint_violation_names:
                    constraint_violation_names.append(name)

        headers = (
            base_fields
            + [f"objective_component_{name}" for name in objective_component_names]
            + [f"constraint_metric_{name}" for name in constraint_metric_names]
            + [f"constraint_violation_{name}" for name in constraint_violation_names]
            + [f"metric_{name}" for name in metric_names]
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in trials:
                output = {name: row.get(name) for name in base_fields}
                metrics = row.get("metrics", {})
                objective_components = row.get("objective_components", {})
                constraint_metrics = row.get("constraint_metrics", {})
                constraint_violations = row.get("constraint_violations", {})
                for name in objective_component_names:
                    output[f"objective_component_{name}"] = objective_components.get(name)
                for name in constraint_metric_names:
                    output[f"constraint_metric_{name}"] = constraint_metrics.get(name)
                for name in constraint_violation_names:
                    output[f"constraint_violation_{name}"] = constraint_violations.get(name)
                for name in metric_names:
                    output[f"metric_{name}"] = metrics.get(name)
                writer.writerow(output)
