#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any, Iterator, Sequence, TextIO

import numpy as np
import pandas as pd
import yaml


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate
    raise RuntimeError(f"Cannot locate repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization import (  # noqa: E402
    BCLFOptimizationSpec,
    BCLFDataset,
    BNNExpectedImprovementBackend,
    ContinuousCandidateSpace,
    DiscreteCandidateSpace,
    ExperimentRecorder,
    OWAFSBNNOptimizer,
    OWAFSServerQueueSimulator,
    OptimizerBoTorch,
    OptimizerLHSRandom,
    ReplaySimulator,
    ServerQueueSimulator,
)
from afs import ObjectiveWeightedAFSConfig, VFFitConfig  # noqa: E402
from simulation.core import SimulationInput  # noqa: E402
from simulation.result_store import BandpassFilterNamingStrategy, FilesystemResultStore  # noqa: E402


METHOD_CHOICES = ("bo", "lhs", "turbo", "bnn", "owafs_bnn")
WARMUP_METHODS = {"bnn", "owafs_bnn"}
DEFAULT_EXPERIMENT_NAME = "online_lhsdb_warmup"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / DEFAULT_EXPERIMENT_NAME
RUNNER_LOG_TAG = "OnlineLHSDBWarmup"
INITIAL_DESIGN_SOURCE_CHOICES = ("random", "previous-lhs")
METHOD_LABELS = {
    "bo": "BO",
    "lhs": "LHS",
    "turbo": "Turbo",
    "bnn": "BNN",
    "owafs_bnn": "Objective-Weighted + Uncertainty-Aware AFS",
}


class TeeStream:
    """Write text to both the original stream and a log file."""

    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, text: str) -> int:
        self.primary.write(text)
        self.secondary.write(text)
        return len(text)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()


@contextlib.contextmanager
def tee_stdout_stderr(log_file: Path | None) -> Iterator[None]:
    if log_file is None:
        yield
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8", buffering=1) as fh:
        stdout = TeeStream(sys.stdout, fh)
        stderr = TeeStream(sys.stderr, fh)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield


def resolve_path(path: Path, *, repo_root: Path = REPO_ROOT) -> Path:
    return path if path.is_absolute() else repo_root / path


@dataclass(frozen=True)
class SharedInitialDesign:
    seed: int
    path: Path
    payload: dict[str, Any]

    @property
    def selected_rows(self) -> list[dict[str, Any]]:
        rows = self.payload.get("selected", [])
        if not isinstance(rows, list):
            raise ValueError(f"Invalid selected rows in {self.path}")
        return [dict(row) for row in rows]

    @property
    def geometries(self) -> np.ndarray:
        return np.asarray([row["geometry"] for row in self.selected_rows], dtype=float)

    @property
    def metadata(self) -> list[dict[str, Any]]:
        metadata: list[dict[str, Any]] = []
        mode = str(self.payload.get("mode", "unknown"))
        for row in self.selected_rows:
            row_metadata = {
                "initial_design_seed": int(self.seed),
                "initial_design_manifest": str(self.path),
                "initial_design_mode": mode,
            }
            for key in (
                "lhs_target_normalized",
                "nearest_distance_normalized",
                "cache_filename",
                "cache_path",
                "source_lhs_seed",
                "source_row_index",
                "source_lhs_iteration",
                "source_case_id",
                "source_csv",
                "source_objective",
                "source_best_so_far",
            ):
                if key in row:
                    row_metadata[key] = row.get(key)
            metadata.append(row_metadata)
        return metadata


class SharedInitialDesignBuilder:
    """Base class for seed-specific cached warmup initial-design manifests."""

    mode = "base_initial_design"
    compatible_modes: tuple[str, ...] = ()
    cache_required = True

    def __init__(
        self,
        *,
        variable_order: Sequence[str],
        variable_bounds: dict[str, tuple[float, float]],
        cache_dir: Path,
        manifest_dir: Path,
        naming_strategy: BandpassFilterNamingStrategy | None = None,
        port_num: int = 2,
        seed_offset: int = 0,
    ) -> None:
        self.variable_order = list(variable_order)
        self.variable_bounds = {k: (float(v[0]), float(v[1])) for k, v in variable_bounds.items()}
        self.cache_dir = Path(cache_dir)
        self.manifest_dir = Path(manifest_dir)
        self.naming_strategy = naming_strategy or BandpassFilterNamingStrategy()
        self.port_num = int(port_num)
        self.seed_offset = int(seed_offset)

    def rng_seed(self, seed: int) -> int:
        return int(seed) + int(self.seed_offset)

    def build_or_load(self, *, seed: int, n_init: int, rebuild: bool = False, write: bool = True) -> SharedInitialDesign:
        path = self.manifest_path(seed)
        if path.exists() and not rebuild:
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = self.build_payload(seed=seed, n_init=n_init)
            if write:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        design = SharedInitialDesign(seed=int(seed), path=path, payload=payload)
        self.validate(design=design, n_init=n_init)
        return design

    def manifest_path(self, seed: int) -> Path:
        return self.manifest_dir / f"seed{int(seed)}.json"

    def build_payload(self, *, seed: int, n_init: int) -> dict[str, Any]:
        raise NotImplementedError("Initial-design builders must implement build_payload().")

    def accepted_modes(self) -> set[str]:
        return {self.mode, *self.compatible_modes}

    def validate(self, *, design: SharedInitialDesign, n_init: int) -> None:
        payload = design.payload
        if int(payload.get("seed")) != int(design.seed):
            raise ValueError(f"Initial design seed mismatch in {design.path}")
        if int(payload.get("n_init")) != int(n_init):
            raise ValueError(f"Initial design n_init mismatch in {design.path}")
        if str(payload.get("mode", "")) not in self.accepted_modes():
            raise ValueError(
                f"Initial design mode mismatch in {design.path}: "
                f"got {payload.get('mode')!r}, expected one of {sorted(self.accepted_modes())!r}"
            )
        if list(payload.get("variable_order", [])) != self.variable_order:
            raise ValueError(f"Initial design variable_order mismatch in {design.path}")

        geometries = design.geometries
        if geometries.shape != (int(n_init), len(self.variable_order)):
            raise ValueError(f"Initial design geometry shape mismatch in {design.path}: {geometries.shape}")
        unique = {tuple(np.round(row.astype(float), 8)) for row in geometries}
        if len(unique) != int(n_init):
            raise ValueError(f"Initial design contains duplicate geometries: {design.path}")
        if not np.all(np.isfinite(geometries)):
            raise ValueError(f"Initial design contains non-finite values: {design.path}")
        lows = np.asarray([self.variable_bounds[name][0] for name in self.variable_order], dtype=float)
        highs = np.asarray([self.variable_bounds[name][1] for name in self.variable_order], dtype=float)
        if np.any(geometries < lows.reshape(1, -1) - 1e-9) or np.any(geometries > highs.reshape(1, -1) + 1e-9):
            raise ValueError(f"Initial design contains out-of-bounds geometries: {design.path}")

        store: FilesystemResultStore | None = None
        for idx, row in enumerate(design.selected_rows):
            row_cache_required = bool(row.get("cache_required", self.cache_required))
            if not row_cache_required:
                continue
            if store is None:
                store = FilesystemResultStore(
                    root_dir=self.cache_dir,
                    naming_strategy=self.naming_strategy,
                    port_num=self.port_num,
                )
            variables = {name: float(row["geometry"][pos]) for pos, name in enumerate(self.variable_order)}
            request = SimulationInput(
                case_id=f"shared_init_seed{design.seed}_it{idx:04d}",
                variable_order=list(self.variable_order),
                variables=variables,
            )
            if store.lookup(request) is None:
                raise FileNotFoundError(
                    f"Initial design cache lookup failed for seed={design.seed} index={idx}: {variables}"
                )


class RandomInitialDesignBuilder(SharedInitialDesignBuilder):
    """Build fresh random warmup geometries that will be simulated online."""

    mode = "random"
    cache_required = False

    def __init__(
        self,
        *,
        variable_order: Sequence[str],
        variable_bounds: dict[str, tuple[float, float]],
        cache_dir: Path,
        manifest_dir: Path,
        naming_strategy: BandpassFilterNamingStrategy | None = None,
        port_num: int = 2,
        resolution: float = 0.01,
        seed_offset: int = 0,
    ) -> None:
        super().__init__(
            variable_order=variable_order,
            variable_bounds=variable_bounds,
            cache_dir=cache_dir,
            manifest_dir=manifest_dir,
            naming_strategy=naming_strategy,
            port_num=port_num,
            seed_offset=seed_offset,
        )
        self.resolution = float(resolution)
        if self.resolution <= 0:
            raise ValueError("resolution must be > 0")

    def build_payload(self, *, seed: int, n_init: int) -> dict[str, Any]:
        selected = self._sample_unique_geometries(seed=seed, n_init=n_init)
        return {
            "schema_version": 1,
            "seed": int(seed),
            "n_init": int(n_init),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "variable_order": list(self.variable_order),
            "bounds": {name: list(self.variable_bounds[name]) for name in self.variable_order},
            "cache_dir": str(self.cache_dir),
            "port_num": int(self.port_num),
            "cache_required": False,
            "sample_policy": "np.random.default_rng(seed + initial_design_seed_offset).uniform(bounds), snapped to resolution",
            "rng_seed": int(self.rng_seed(seed)),
            "seed_offset": int(self.seed_offset),
            "resolution": float(self.resolution),
            "selected": selected,
        }

    def _sample_unique_geometries(self, *, seed: int, n_init: int) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.rng_seed(seed))
        lows = np.asarray([self.variable_bounds[name][0] for name in self.variable_order], dtype=float)
        highs = np.asarray([self.variable_bounds[name][1] for name in self.variable_order], dtype=float)
        selected: list[dict[str, Any]] = []
        seen: set[tuple[float, ...]] = set()
        max_attempts = max(1000, int(n_init) * 1000)
        attempts = 0
        while len(selected) < int(n_init) and attempts < max_attempts:
            attempts += 1
            raw = rng.uniform(lows, highs)
            snapped = np.round(raw / self.resolution) * self.resolution
            geometry = np.clip(snapped, lows, highs)
            geometry = np.asarray([round(float(value), 8) for value in geometry], dtype=float)
            key = tuple(float(value) for value in geometry)
            if key in seen:
                continue
            seen.add(key)
            variables = {name: float(geometry[pos]) for pos, name in enumerate(self.variable_order)}
            normalized = self._normalize_geometry(geometry)
            selected.append(
                {
                    "mode": self.mode,
                    "initial_index": int(len(selected)),
                    "geometry": [float(value) for value in geometry],
                    "geometry_normalized": normalized,
                    "variables": variables,
                    "cache_required": False,
                }
            )
        if len(selected) != int(n_init):
            raise RuntimeError(
                f"Could not sample {int(n_init)} unique random geometries after {attempts} attempts"
            )
        return selected

    def _normalize_geometry(self, geometry: Sequence[float]) -> list[float]:
        normalized: list[float] = []
        for pos, name in enumerate(self.variable_order):
            low, high = self.variable_bounds[name]
            span = high - low
            normalized.append(0.5 if abs(span) < 1e-12 else (float(geometry[pos]) - low) / span)
        return normalized


class PreviousLHSRowsDatasetBuilder:
    """Export and validate previous LHS campaign rows for warmup reuse."""

    def __init__(
        self,
        *,
        previous_lhs_dir: Path,
        dataset_dir: Path,
        variable_order: Sequence[str],
        variable_bounds: dict[str, tuple[float, float]],
        cache_dir: Path,
        naming_strategy: BandpassFilterNamingStrategy | None = None,
        port_num: int = 2,
    ) -> None:
        self.previous_lhs_dir = Path(previous_lhs_dir)
        self.dataset_dir = Path(dataset_dir)
        self.variable_order = list(variable_order)
        self.variable_bounds = {k: (float(v[0]), float(v[1])) for k, v in variable_bounds.items()}
        self.cache_dir = Path(cache_dir)
        self.naming_strategy = naming_strategy or BandpassFilterNamingStrategy()
        self.port_num = int(port_num)

    def build_or_load_rows(self, *, rebuild: bool = False) -> pd.DataFrame:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        rows_path = self.rows_path
        if rows_path.exists() and not rebuild:
            rows = pd.read_csv(rows_path)
        else:
            rows = self._export_rows()
            rows.to_csv(rows_path, index=False)
        self._normalize_loaded_rows(rows)
        self._write_rows_manifest(rows=rows, rows_path=rows_path)
        return rows

    @property
    def rows_path(self) -> Path:
        return self.dataset_dir / "lhs_online_general_rows.csv"

    @property
    def manifest_path(self) -> Path:
        return self.dataset_dir / "lhs_online_general_manifest.json"

    def _export_rows(self) -> pd.DataFrame:
        paths = self._source_csv_paths()
        rows: list[dict[str, Any]] = []
        for source_path in paths:
            source_seed = self._source_seed_from_path(source_path)
            source_df = pd.read_csv(source_path)
            for source_row_index, source_row in source_df.iterrows():
                geometry = self._parse_vector(source_row.get("geometry"), "geometry", source_path)
                normalized = self._parse_vector_or_none(source_row.get("geometry_normalized"))
                if normalized is None:
                    normalized = self._normalize_geometry(geometry)
                variables = {name: float(geometry[pos]) for pos, name in enumerate(self.variable_order)}
                selection_debug = self._parse_mapping(source_row.get("selection_debug"))
                cache_filename, cache_path, cache_resolves = self._cache_reference(variables, source_row)
                objective = self._numeric_or_nan(source_row.get("objective"))
                best_so_far = self._numeric_or_nan(source_row.get("best_so_far"))
                source_iteration = int(self._numeric_or_default(source_row.get("iteration"), source_row_index))
                row: dict[str, Any] = {
                    "source_lhs_seed": int(source_seed),
                    "source_row_index": int(source_row_index),
                    "source_lhs_iteration": source_iteration,
                    "source_step": int(self._numeric_or_default(source_row.get("step"), source_iteration)),
                    "source_phase": str(source_row.get("phase", "")),
                    "source_case_id": str(source_row.get("case_id", "")),
                    "source_csv": str(source_path),
                    "geometry": json.dumps([float(value) for value in geometry], ensure_ascii=True),
                    "geometry_normalized": json.dumps([float(value) for value in normalized], ensure_ascii=True),
                    "geometry_key": self._geometry_key(geometry),
                    "objective": objective,
                    "best_so_far": best_so_far,
                    "cache_hit_original": self._bool_value(source_row.get("cache_hit")),
                    "original_nearest_distance_normalized": self._numeric_or_nan(
                        selection_debug.get("nearest_distance_normalized")
                    ),
                    "original_candidate_pool_size": self._numeric_or_nan(selection_debug.get("candidate_pool_size")),
                    "original_selected_pos": self._numeric_or_nan(selection_debug.get("selected_pos")),
                    "resolved_cache_filename": cache_filename,
                    "resolved_cache_path": str(cache_path) if cache_path is not None else "",
                    "cache_resolves": bool(cache_resolves),
                    "within_bounds": self._variables_within_bounds(variables),
                }
                for name, value in variables.items():
                    row[name] = float(value)
                rows.append(row)

        if not rows:
            raise RuntimeError(f"No previous LHS rows exported from {self.previous_lhs_dir}")
        df = pd.DataFrame(rows).sort_values(["source_lhs_seed", "source_lhs_iteration"]).reset_index(drop=True)
        expected = df.groupby("source_lhs_seed").size()
        if (expected < 80).any():
            raise RuntimeError(f"Previous LHS source has fewer than 80 rows: {expected.to_dict()}")
        return df

    def _source_csv_paths(self) -> list[Path]:
        if not self.previous_lhs_dir.exists():
            raise FileNotFoundError(f"previous_lhs_dir does not exist: {self.previous_lhs_dir}")
        paths = sorted(self.previous_lhs_dir.glob("lhs_online_general_seed*.csv"))
        if not paths:
            raise FileNotFoundError(f"No lhs_online_general_seed*.csv files in {self.previous_lhs_dir}")
        return paths

    @staticmethod
    def _source_seed_from_path(path: Path) -> int:
        token = path.stem.rsplit("seed", 1)[-1]
        return int(token)

    def _write_rows_manifest(self, *, rows: pd.DataFrame, rows_path: Path) -> None:
        missing = rows[~self._bool_series(rows["cache_resolves"])].copy()
        missing_columns = ["source_lhs_seed", "source_lhs_iteration", "source_case_id", *self.variable_order]
        missing_records = missing[missing_columns].where(pd.notna(missing[missing_columns]), None)
        manifest = {
            "schema_version": 2,
            "mode": "previous_lhs_rows_dataset",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_lhs_dir": str(self.previous_lhs_dir),
            "cache_dir": str(self.cache_dir),
            "variable_order": list(self.variable_order),
            "bounds": {name: list(self.variable_bounds[name]) for name in self.variable_order},
            "rows_csv": str(rows_path),
            "row_count": int(len(rows)),
            "source_seed_counts": {str(k): int(v) for k, v in rows.groupby("source_lhs_seed").size().items()},
            "cache_resolves_count": int(self._bool_series(rows["cache_resolves"]).sum()),
            "cache_missing_count": int((~self._bool_series(rows["cache_resolves"])).sum()),
            "missing_cache_rows": missing_records.to_dict("records"),
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    def _cache_reference(self, variables: dict[str, float], source_row: pd.Series) -> tuple[str, Path | None, bool]:
        request = SimulationInput(
            case_id=str(source_row.get("case_id", "previous_lhs")),
            variable_order=list(self.variable_order),
            variables=variables,
        )
        filename = self.naming_strategy.build_filename(request, port_num=self.port_num)
        path = self.cache_dir / filename
        return filename, path, path.exists()

    def _normalize_loaded_rows(self, rows: pd.DataFrame) -> None:
        for column in ("cache_resolves", "within_bounds", "cache_hit_original"):
            if column in rows:
                rows[column] = self._bool_series(rows[column])

    @staticmethod
    def _parse_mapping(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if not isinstance(value, str) or not value.strip():
            return {}
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}

    @staticmethod
    def _parse_vector(value: Any, name: str, source_path: Path) -> list[float]:
        parsed = PreviousLHSRowsDatasetBuilder._parse_vector_or_none(value)
        if parsed is None:
            raise ValueError(f"Missing {name} vector in {source_path}")
        return parsed

    @staticmethod
    def _parse_vector_or_none(value: Any) -> list[float] | None:
        if isinstance(value, (list, tuple, np.ndarray)):
            return [float(item) for item in value]
        if not isinstance(value, str) or not value.strip():
            return None
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, (list, tuple)):
            return None
        return [float(item) for item in parsed]

    @staticmethod
    def _numeric_or_nan(value: Any) -> float:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return float(numeric) if pd.notna(numeric) else float("nan")

    @staticmethod
    def _numeric_or_default(value: Any, default: int | float) -> float:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return float(numeric) if pd.notna(numeric) else float(default)

    @staticmethod
    def _bool_value(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value) if pd.notna(value) else False

    @staticmethod
    def _bool_series(series: pd.Series) -> pd.Series:
        if series.dtype == bool:
            return series.fillna(False).astype(bool)
        return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})

    def _normalize_geometry(self, geometry: Sequence[float]) -> list[float]:
        normalized: list[float] = []
        for pos, name in enumerate(self.variable_order):
            low, high = self.variable_bounds[name]
            span = high - low
            normalized.append(0.5 if abs(span) < 1e-12 else (float(geometry[pos]) - low) / span)
        return normalized

    def _variables_within_bounds(self, variables: dict[str, float]) -> bool:
        for name in self.variable_order:
            low, high = self.variable_bounds[name]
            value = float(variables[name])
            if value < low - 1e-9 or value > high + 1e-9:
                return False
        return True

    @staticmethod
    def _geometry_key(geometry: Sequence[float]) -> str:
        return "|".join(f"{float(value):.8f}" for value in geometry)


class PreviousLHSRandomInitialDesignBuilder(SharedInitialDesignBuilder):
    """Build warmup manifests by sampling rows from the reusable previous-LHS metadata DB."""

    mode = "previous_lhs_random"
    compatible_modes = ("lhs_database_random",)
    cache_required = True

    def __init__(
        self,
        *,
        variable_order: Sequence[str],
        variable_bounds: dict[str, tuple[float, float]],
        cache_dir: Path,
        manifest_dir: Path,
        previous_lhs_dir: Path,
        lhs_dataset_dir: Path,
        rebuild_lhs_dataset: bool = False,
        naming_strategy: BandpassFilterNamingStrategy | None = None,
        port_num: int = 2,
        seed_offset: int = 0,
    ) -> None:
        super().__init__(
            variable_order=variable_order,
            variable_bounds=variable_bounds,
            cache_dir=cache_dir,
            manifest_dir=manifest_dir,
            naming_strategy=naming_strategy,
            port_num=port_num,
            seed_offset=seed_offset,
        )
        self.previous_lhs_dir = Path(previous_lhs_dir)
        self.lhs_dataset_dir = Path(lhs_dataset_dir)
        self.rebuild_lhs_dataset = bool(rebuild_lhs_dataset)
        self.dataset_builder = PreviousLHSRowsDatasetBuilder(
            previous_lhs_dir=self.previous_lhs_dir,
            dataset_dir=self.lhs_dataset_dir,
            variable_order=self.variable_order,
            variable_bounds=self.variable_bounds,
            cache_dir=self.cache_dir,
            naming_strategy=self.naming_strategy,
            port_num=self.port_num,
        )
        self._rows_cache: pd.DataFrame | None = None

    @property
    def rows_path(self) -> Path:
        return self.lhs_dataset_dir / "lhs_online_general_rows.csv"

    def build_payload(self, *, seed: int, n_init: int) -> dict[str, Any]:
        rows = self._valid_unique_rows(n_init=n_init)
        if len(rows) < int(n_init):
            raise RuntimeError(
                f"Need {int(n_init)} valid unique LHS database rows, found {len(rows)} in {self.rows_path}"
            )
        rng = np.random.default_rng(self.rng_seed(seed))
        positions = rng.choice(np.arange(len(rows)), size=int(n_init), replace=False)
        selected_rows = [
            self._manifest_row_from_lhs_database(initial_index=idx, row=rows.iloc[int(pos)])
            for idx, pos in enumerate(positions)
        ]
        return {
            "schema_version": 3,
            "seed": int(seed),
            "n_init": int(n_init),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "variable_order": list(self.variable_order),
            "bounds": {name: list(self.variable_bounds[name]) for name in self.variable_order},
            "cache_dir": str(self.cache_dir),
            "port_num": int(self.port_num),
            "previous_lhs_dir": str(self.previous_lhs_dir),
            "lhs_dataset_dir": str(self.lhs_dataset_dir),
            "lhs_dataset_rows_csv": str(self.rows_path),
            "sample_policy": (
                "np.random.default_rng(seed + initial_design_seed_offset)"
                ".choice(valid_unique_previous_lhs_rows, replace=False)"
            ),
            "rng_seed": int(self.rng_seed(seed)),
            "seed_offset": int(self.seed_offset),
            "valid_unique_row_count": int(len(rows)),
            "selected": selected_rows,
        }

    def _valid_unique_rows(self, *, n_init: int) -> pd.DataFrame:
        rows = self._load_rows(n_init=n_init)
        records: list[dict[str, Any]] = []
        for _, row in rows.iterrows():
            record = self._valid_record_or_none(row)
            if record is not None:
                records.append(record)
        if not records:
            raise RuntimeError(f"No valid rows are available in {self.rows_path}")
        df = pd.DataFrame(records).sort_values(["source_lhs_seed", "source_lhs_iteration"]).reset_index(drop=True)
        return df.drop_duplicates("_geometry_key", keep="first").reset_index(drop=True)

    def _load_rows(self, *, n_init: int) -> pd.DataFrame:
        if self._rows_cache is not None:
            return self._rows_cache.copy()
        if not self.rows_path.exists() or self.rebuild_lhs_dataset:
            rows = self.dataset_builder.build_or_load_rows(rebuild=True)
        else:
            rows = pd.read_csv(self.rows_path)
            self.dataset_builder._normalize_loaded_rows(rows)
        self._rows_cache = rows.copy()
        return rows

    def _valid_record_or_none(self, row: pd.Series) -> dict[str, Any] | None:
        if not self._truthy(row.get("cache_resolves")) or not self._truthy(row.get("within_bounds")):
            return None
        objective = pd.to_numeric(pd.Series([row.get("objective")]), errors="coerce").iloc[0]
        if not np.isfinite(float(objective)):
            return None
        geometry = PreviousLHSRowsDatasetBuilder._parse_vector_or_none(row.get("geometry"))
        normalized = PreviousLHSRowsDatasetBuilder._parse_vector_or_none(row.get("geometry_normalized"))
        if geometry is None or len(geometry) != len(self.variable_order) or not np.all(np.isfinite(geometry)):
            return None
        if normalized is None or len(normalized) != len(self.variable_order) or not np.all(np.isfinite(normalized)):
            normalized = self.dataset_builder._normalize_geometry(geometry)
        variables = {name: float(geometry[pos]) for pos, name in enumerate(self.variable_order)}
        if not self.dataset_builder._variables_within_bounds(variables):
            return None
        cache_path_text = str(row.get("resolved_cache_path", "")).strip()
        if not cache_path_text:
            return None
        cache_path = Path(cache_path_text)
        if not cache_path.exists():
            return None
        record = row.to_dict()
        record["_geometry"] = [float(value) for value in geometry]
        record["_geometry_normalized"] = [float(value) for value in normalized]
        record["_variables"] = variables
        record["_geometry_key"] = str(row.get("geometry_key") or PreviousLHSRowsDatasetBuilder._geometry_key(geometry))
        return record

    def _manifest_row_from_lhs_database(self, *, initial_index: int, row: pd.Series) -> dict[str, Any]:
        geometry = [float(value) for value in row["_geometry"]]
        normalized = [float(value) for value in row["_geometry_normalized"]]
        variables = dict(row["_variables"])
        return {
            "mode": self.mode,
            "initial_index": int(initial_index),
            "geometry": geometry,
            "geometry_normalized": normalized,
            "variables": variables,
            "cache_filename": str(row["resolved_cache_filename"]),
            "cache_path": str(row["resolved_cache_path"]),
            "nearest_distance_normalized": self._optional_float(row.get("original_nearest_distance_normalized")),
            "source_lhs_seed": int(row["source_lhs_seed"]),
            "source_row_index": int(row.get("source_row_index", -1)),
            "source_lhs_iteration": int(row["source_lhs_iteration"]),
            "source_case_id": str(row.get("source_case_id", "")),
            "source_csv": str(row.get("source_csv", "")),
            "source_objective": float(row["objective"]),
            "source_best_so_far": self._optional_float(row.get("best_so_far")),
            "cache_required": True,
        }

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value) if pd.notna(value) else False

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return float(numeric) if pd.notna(numeric) and np.isfinite(float(numeric)) else None


@dataclass(frozen=True)
class SharedOnlineExperimentConfig:
    repo_root: Path = REPO_ROOT
    backend: str = "server"
    methods: list[str] = field(default_factory=lambda: list(METHOD_CHOICES))
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    total_trials: int = 80
    warmup_size: int = 15
    n_init: int | None = None
    budget: int | None = None
    pool_size: int = 5000
    retrain_every: int = 1
    initial_epochs: int = 800
    update_epochs: int = 20
    device: str = "cuda"
    mc_samples: int = 64
    config_path: Path = REPO_ROOT / "configs" / "filter_bnn_online_uaw_uaw.yaml"
    replay_dataset_path: Path = REPO_ROOT / "data" / "filter_bnn_online_uaw_uaw_dataset.pkl"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    bnn_checkpoint_dir: Path | None = None
    cache_dir: Path = REPO_ROOT / "data" / "bandpass_coupled_line_filter_export" / "ext"
    initial_design_source: str = "previous-lhs"
    initial_design_seed_offset: int = 0
    previous_lhs_dir: Path = REPO_ROOT / "experiments" / "lhs_online_general"
    lhs_dataset_dir: Path = REPO_ROOT / "data" / "bandpass_coupled_line_filter_export" / "ext" / "lhs"
    rebuild_lhs_dataset: bool = False
    archive_on_policy_change: bool = True
    afs_root: Path | None = None
    owafs_sample_count: int = 20
    queue_root: Path = REPO_ROOT / "queue"
    server_job_timeout_sec: float = 0.0
    server_wait_on_failed: bool = True
    template_name: str = "bclf_template_v1"
    design_name: str | None = None
    setup_name: str | None = None
    sweep_name: str | None = None
    resume: bool = True
    reset_checkpoint: bool = False
    clear_queue_artifacts_on_reset: bool = True
    rebuild_initial_designs: bool = False
    generate_initial_design_only: bool = False
    debug_verbose: bool = True
    debug_print_every: int = 1
    dry_run: bool = False
    log_file: Path | None = None

    @property
    def target_trials(self) -> int:
        return int(self.total_trials)

    @property
    def initial_design_dir(self) -> Path:
        return self.output_dir / "initial_designs"

    @property
    def resolved_bnn_checkpoint_dir(self) -> Path:
        if self.bnn_checkpoint_dir is not None:
            return self.bnn_checkpoint_dir
        return self.output_dir / "bnn_checkpoints"

    @property
    def resolved_afs_root(self) -> Path:
        if self.afs_root is not None:
            return self.afs_root
        return self.cache_dir / "afs"

    @property
    def unit_map(self) -> dict[str, str]:
        return {
            "L": "mm",
            "L1": "mm",
            "L2": "mm",
            "W": "mm",
            "W1": "mm",
            "W2": "mm",
            "S1": "mm",
            "S2": "mm",
        }

    @property
    def bnn_backend_config(self) -> dict[str, Any]:
        return {
            "hidden_dims": [256, 256, 256],
            "epochs": int(self.initial_epochs),
            "update_epochs": int(self.update_epochs),
            "update_epoch_schedule": "constant",
            "update_epoch_multiplier": int(self.update_epochs),
            "warm_start": True,
            "initial_train_size": int(self.warmup_size),
            "batch_size": 1024,
            "lr": 1e-3,
            "bayesian_loss_weight": 1e-4,
            "mc_samples": int(self.mc_samples),
            "acquisition": "qlogei",
            "surrogate_variant": "fourier",
            "fourier_bands": [1.0, 2.0, 4.0, 8.0, 16.0],
            "include_raw_frequency": True,
        }


class SharedOnlineExperimentRunner:
    def __init__(self, config: SharedOnlineExperimentConfig) -> None:
        self.config = config
        self.cfg = self._load_yaml_config(config.config_path)
        self.variable_order = list(BCLFDataset.DEFAULT_FILTER_VARIABLE_ORDER)
        self.variable_bounds = self._load_variable_bounds()
        self.spec = BCLFOptimizationSpec(
            passband_ghz=(4.8, 5.4),
            lower_stopband_ghz=(3.6, 4.4),
            upper_stopband_ghz=(5.8, 8.7),
            min_passband_s21_db=-5.0,
            max_passband_s11_db=-7.0,
            max_stopband_s21_db=-15.0,
            s11_weight=0.5,
            stopband_weight=0.5,
        )
        self.initial_design_builder = self._build_initial_design_builder()

    def _build_initial_design_builder(self) -> SharedInitialDesignBuilder:
        source = str(self.config.initial_design_source)
        common_kwargs = {
            "variable_order": self.variable_order,
            "variable_bounds": self.variable_bounds,
            "cache_dir": self.config.cache_dir,
            "manifest_dir": self.config.initial_design_dir,
            "port_num": 2,
            "seed_offset": int(self.config.initial_design_seed_offset),
        }
        if source == "random":
            return RandomInitialDesignBuilder(**common_kwargs)
        if source == "previous-lhs":
            return PreviousLHSRandomInitialDesignBuilder(
                **common_kwargs,
                previous_lhs_dir=self.config.previous_lhs_dir,
                lhs_dataset_dir=self.config.lhs_dataset_dir,
                rebuild_lhs_dataset=self.config.rebuild_lhs_dataset,
            )
        raise ValueError(f"Unsupported initial_design_source={source!r}")

    def method_n_init(self, method: str) -> int:
        return int(self.config.warmup_size) if str(method).lower() in WARMUP_METHODS else 0

    def method_budget(self, method: str) -> int:
        budget = int(self.config.total_trials) - self.method_n_init(method)
        if budget < 0:
            raise ValueError(
                f"total_trials={self.config.total_trials} is smaller than "
                f"{method} warmup size={self.method_n_init(method)}"
            )
        return budget

    def method_needs_initial_design(self, method: str) -> bool:
        return self.method_n_init(method) > 0

    def run(self) -> int:
        self._configure_process_io()
        self._archive_initial_design_policy_change()
        self._ensure_output_dirs()
        self._log_header()

        initial_designs = self._build_warmup_initial_designs()
        self._write_and_log_initial_design_preview(initial_designs)

        if self.config.generate_initial_design_only or self.config.dry_run:
            self._log_planned_jobs(initial_designs)
            if self.config.dry_run:
                self._log("dry_run: true; no optimizer will be started.")
            return 0

        for method in self.config.methods:
            campaign_summary_df = self.load_campaign_summary(method)
            latest_history_df = pd.DataFrame()
            self._log(f"[Campaign] method={method} seeds={self.config.seeds}")
            for seed in self.config.seeds:
                seed = int(seed)
                checkpoint_path = self.checkpoint_for_method_seed(method, seed)
                checkpoint_history = self._checkpoint_history(method, seed)
                checkpoint_df = self._history_frame(seed, checkpoint_history)
                if self.is_seed_completed(method, seed):
                    self._log(
                        f"[Campaign] method={method} seed={seed} completed; skipping "
                        f"({len(checkpoint_df)}/{self.config.target_trials} trials)."
                    )
                    campaign_summary_df = self.upsert_campaign_row(
                        method,
                        self._summary_row_for_seed(method, seed, "completed", checkpoint_df),
                    )
                    latest_history_df = checkpoint_df
                    continue

                status_label = "resuming" if checkpoint_path.exists() and len(checkpoint_df) else "starting"
                optimizer_seed = self.effective_seed_for_rerun(method, seed)
                initial_design = initial_designs.get(seed) if self.method_needs_initial_design(method) else None
                initial_design_label = "<none>" if initial_design is None else str(initial_design.path)
                self._log(
                    f"[Campaign] method={method} seed={seed} optimizer_seed={optimizer_seed} {status_label}; "
                    f"checkpoint={checkpoint_path} initial_design={initial_design_label}"
                )
                optimizer = self.build_optimizer(method, seed, initial_design)
                run_result = optimizer.run()
                latest_history_df = pd.DataFrame(run_result.history).sort_values("iteration").reset_index(drop=True)
                latest_history_df = self._drop_in_memory_response_columns(latest_history_df)
                latest_history_df["seed"] = seed
                final_status = (
                    "completed"
                    if self._budgeted_history_count(latest_history_df) >= self.config.target_trials
                    else "in_progress"
                )
                campaign_summary_df = self.upsert_campaign_row(
                    method,
                    self._summary_row_for_seed(method, seed, final_status, latest_history_df),
                )
                self._log(f"run_id: {run_result.run_id}")
                self._log(f"best_objective: {run_result.state.best_objective}")
                self._log(f"json: {run_result.recorder_output.json_path}")
                self._log(f"csv: {run_result.recorder_output.csv_path}")

            self._log(f"{METHOD_LABELS[method]} campaign summary:")
            self._log(campaign_summary_df.to_string(index=False) if not campaign_summary_df.empty else "<empty>")
            if not latest_history_df.empty:
                self._log("Latest seed history tail:")
                self._log(latest_history_df.tail(10).to_string(index=False))
        return 0

    def _build_warmup_initial_designs(self) -> dict[int, SharedInitialDesign]:
        if not any(self.method_needs_initial_design(method) for method in self.config.methods):
            return {}
        return {
            int(seed): self.initial_design_builder.build_or_load(
                seed=int(seed),
                n_init=int(self.config.warmup_size),
                rebuild=self.config.rebuild_initial_designs,
                write=True,
            )
            for seed in self.config.seeds
        }

    def _write_and_log_initial_design_preview(self, initial_designs: dict[int, SharedInitialDesign]) -> None:
        if not initial_designs:
            return
        points_df = self._initial_design_points_frame(initial_designs)
        if points_df.empty:
            return
        summary_df = self._initial_design_summary_frame(points_df)
        points_path = self.config.initial_design_dir / "warmup_initial_points.csv"
        summary_path = self.config.initial_design_dir / "warmup_initial_summary.csv"
        points_df.to_csv(points_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        self._log(f"[InitialDesignPreview] summary_csv={summary_path}")
        self._log(f"[InitialDesignPreview] points_csv={points_path}")
        display_cols = [
            "target_seed",
            "n",
            "min_objective",
            "median_objective",
            "mean_objective",
            "std_objective",
            "best_init_idx",
            "best_source_lhs_seed",
            "best_source_lhs_iteration",
            "best_source_case_id",
        ]
        available = [col for col in display_cols if col in summary_df.columns]
        self._log("[InitialDesignPreview] seed summary:")
        self._log(summary_df[available].to_string(index=False))

    def _initial_design_points_frame(self, initial_designs: dict[int, SharedInitialDesign]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for seed, design in sorted(initial_designs.items()):
            mode = str(design.payload.get("mode", ""))
            for row in design.selected_rows:
                rows.append(
                    {
                        "target_seed": int(seed),
                        "initial_design_mode": mode,
                        "init_idx": int(row.get("initial_index", len(rows))),
                        "source_objective": row.get("source_objective"),
                        "source_best_so_far": row.get("source_best_so_far"),
                        "source_lhs_seed": row.get("source_lhs_seed"),
                        "source_lhs_iteration": row.get("source_lhs_iteration"),
                        "source_case_id": row.get("source_case_id"),
                        "nearest_distance_normalized": row.get("nearest_distance_normalized"),
                        "cache_required": row.get("cache_required", getattr(self.initial_design_builder, "cache_required", True)),
                        "cache_path": row.get("cache_path"),
                        "geometry": json.dumps(row.get("geometry", []), ensure_ascii=True),
                    }
                )
        return pd.DataFrame(rows)

    def _initial_design_summary_frame(self, points_df: pd.DataFrame) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for seed, group in points_df.groupby("target_seed", sort=True):
            objective = pd.to_numeric(group["source_objective"], errors="coerce")
            finite_mask = np.isfinite(objective.to_numpy(dtype=float))
            record: dict[str, Any] = {
                "target_seed": int(seed),
                "n": int(len(group)),
                "finite_objective_count": int(finite_mask.sum()),
            }
            if finite_mask.any():
                finite_objective = objective[finite_mask]
                best_row = group.loc[finite_objective.idxmin()]
                record.update(
                    {
                        "min_objective": float(finite_objective.min()),
                        "median_objective": float(finite_objective.median()),
                        "mean_objective": float(finite_objective.mean()),
                        "std_objective": float(finite_objective.std(ddof=1)),
                        "best_init_idx": int(best_row["init_idx"]),
                        "best_source_lhs_seed": self._optional_int(best_row.get("source_lhs_seed")),
                        "best_source_lhs_iteration": self._optional_int(best_row.get("source_lhs_iteration")),
                        "best_source_case_id": str(best_row.get("source_case_id", "")),
                        "best_geometry": str(best_row.get("geometry", "")),
                    }
                )
            else:
                record.update(
                    {
                        "min_objective": float("nan"),
                        "median_objective": float("nan"),
                        "mean_objective": float("nan"),
                        "std_objective": float("nan"),
                        "best_init_idx": None,
                        "best_source_lhs_seed": None,
                        "best_source_lhs_iteration": None,
                        "best_source_case_id": "",
                        "best_geometry": "",
                    }
                )
            if "nearest_distance_normalized" in group:
                nearest = pd.to_numeric(group["nearest_distance_normalized"], errors="coerce")
                finite_nearest = nearest[np.isfinite(nearest.to_numpy(dtype=float))]
                record["median_nearest_distance_normalized"] = (
                    float(finite_nearest.median()) if len(finite_nearest) else float("nan")
                )
            records.append(record)
        return pd.DataFrame(records)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return int(numeric) if pd.notna(numeric) and np.isfinite(float(numeric)) else None

    def method_output_dir(self, method: str) -> Path:
        return self.config.output_dir / method

    def campaign_summary_path(self, method: str) -> Path:
        return self.method_output_dir(method) / "campaign_summary.csv"

    def run_id_for_method_seed(self, method: str, seed: int) -> str:
        return f"{method}_{self.config.experiment_name}_seed{int(seed)}"

    def checkpoint_for_method_seed(self, method: str, seed: int) -> Path:
        run_id = self.run_id_for_method_seed(method, seed)
        return self.method_output_dir(method) / f"{run_id}.checkpoint.json"

    def bnn_checkpoint_for_seed(self, seed: int, method: str = "bnn") -> Path:
        method = str(method).lower()
        stem = "bnn" if method == "bnn" else method
        return self.config.resolved_bnn_checkpoint_dir / f"{stem}_{self.config.experiment_name}_seed{int(seed)}.bnn.pt"

    def bnn_backend_config_for_seed(self, seed: int, method: str = "bnn") -> dict[str, Any]:
        payload = dict(self.config.bnn_backend_config)
        payload.update(
            {
                "checkpoint_path": str(self.bnn_checkpoint_for_seed(seed, method=method)),
                "force_retrain": True,
                "require_checkpoint": False,
                "save_checkpoint": True,
            }
        )
        return payload

    def load_campaign_summary(self, method: str, path: Path | None = None) -> pd.DataFrame:
        path = self.campaign_summary_path(method) if path is None else Path(path)
        if path.exists():
            df = pd.read_csv(path)
            if "seed" in df.columns:
                df = df.sort_values("seed").reset_index(drop=True)
            return df
        return pd.DataFrame(
            columns=[
                "seed",
                "run_id",
                "method",
                "status",
                "num_trials",
                "num_budgeted_trials",
                "num_validations",
                "best_objective",
                "first_feasible_iteration",
                "cache_hit_ratio",
                "runtime_engine_sec",
                "runtime_sim_sec",
                "json",
                "csv",
                "checkpoint",
                "initial_design_manifest",
                "updated_at",
            ]
        )

    def upsert_campaign_row(self, method: str, row: dict[str, Any], path: Path | None = None) -> pd.DataFrame:
        path = self.campaign_summary_path(method) if path is None else Path(path)
        df = self.load_campaign_summary(method, path)
        seed = int(row["seed"])
        if "seed" in df.columns and (df["seed"] == seed).any():
            for key, value in row.items():
                df.loc[df["seed"] == seed, key] = value
        elif df.empty:
            df = pd.DataFrame([row])
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = df.sort_values("seed").reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df

    def build_optimizer(self, method: str, seed: int, initial_design: SharedInitialDesign | None = None):
        method = method.lower()
        run_id = self.run_id_for_method_seed(method, seed)
        checkpoint_path = self.checkpoint_for_method_seed(method, seed)
        optimizer_seed = self.effective_seed_for_rerun(method, seed)
        method_n_init = self.method_n_init(method)
        method_budget = self.method_budget(method)
        if method_n_init > 0 and initial_design is None:
            raise RuntimeError(f"method={method} requires a warmup initial design for seed={seed}")

        if self.config.reset_checkpoint:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            if method in {"bnn", "owafs_bnn"}:
                self.bnn_checkpoint_for_seed(seed, method=method).unlink(missing_ok=True)
            if self.config.clear_queue_artifacts_on_reset:
                removed = self._cleanup_queue_artifacts_for_run(self.config.queue_root, run_id)
                self._log(f"[ResetCleanup] run_id={run_id} {removed}")

        candidate_space, simulator = self._build_candidate_space_and_simulator(
            method=method,
            seed=optimizer_seed,
            run_id=run_id,
            cache_required_iterations=self._cache_required_iterations(method, method_n_init),
        )
        common_kwargs = {
            "candidate_space": candidate_space,
            "simulator": simulator,
            "optimization_spec": self.spec,
            "recorder": ExperimentRecorder(output_dir=self.method_output_dir(method)),
            "budget": method_budget,
            "n_init": method_n_init,
            "seed": optimizer_seed,
            "run_id": run_id,
            "checkpoint_path": checkpoint_path,
            "resume": self.config.resume and checkpoint_path.exists(),
            "progress_callback": self._progress_debug_callback,
        }
        if method_n_init > 0 and initial_design is not None:
            common_kwargs.update(
                {
                    "initial_geometries": initial_design.geometries,
                    "initial_geometry_metadata": initial_design.metadata,
                }
            )

        if method == "lhs":
            optimizer = OptimizerLHSRandom(**common_kwargs)
        elif method == "turbo":
            optimizer = OptimizerBoTorch(
                **common_kwargs,
                retrain_every=self.config.retrain_every,
                use_trust_region=True,
                not_ready_selection_policy="random",
            )
        elif method == "bnn":
            optimizer = OptimizerBoTorch(
                **common_kwargs,
                retrain_every=1,
                acquisition_backend=BNNExpectedImprovementBackend(
                    self.bnn_backend_config_for_seed(seed, method=method),
                    device=self.config.device,
                ),
            )
        elif method == "owafs_bnn":
            optimizer = OWAFSBNNOptimizer(
                **common_kwargs,
                retrain_every=1,
                acquisition_backend=BNNExpectedImprovementBackend(
                    self.bnn_backend_config_for_seed(seed, method=method),
                    device=self.config.device,
                ),
            )
        elif method == "bo":
            optimizer = OptimizerBoTorch(
                **common_kwargs,
                retrain_every=self.config.retrain_every,
                not_ready_selection_policy="random",
            )
        else:
            raise ValueError(f"Unsupported method={method!r}")

        if method in {"bnn", "owafs_bnn"} and self.config.resume and checkpoint_path.exists() and len(optimizer._history):
            self._rehydrate_history_responses(optimizer, simulator)
        return optimizer

    def _cache_required_iterations(self, method: str, n_init: int) -> int:
        if int(n_init) <= 0 or not self.method_needs_initial_design(method):
            return 0
        return int(n_init) if bool(getattr(self.initial_design_builder, "cache_required", True)) else 0

    def _build_candidate_space_and_simulator(
        self,
        *,
        method: str,
        seed: int,
        run_id: str,
        cache_required_iterations: int,
    ):
        if self.config.backend == "server":
            candidate_space = ContinuousCandidateSpace(
                variable_order=self.variable_order,
                variable_bounds=self.variable_bounds,
                pool_size=self.config.pool_size,
                seed=seed,
                resolution=0.01,
            )
            result_store = FilesystemResultStore(
                root_dir=self.config.cache_dir,
                naming_strategy=BandpassFilterNamingStrategy(),
                port_num=2,
            )
            simulator_cls = OWAFSServerQueueSimulator if method == "owafs_bnn" else ServerQueueSimulator
            simulator_kwargs: dict[str, Any] = {}
            if method == "owafs_bnn":
                simulator_kwargs.update(
                    {
                        "afs_root": self.config.resolved_afs_root / "owafs_bnn",
                        "owafs_config": ObjectiveWeightedAFSConfig(
                            sample_count=int(self.config.owafs_sample_count),
                        ),
                        "vf_config": VFFitConfig(
                            n_poles_real=4,
                            n_poles_complex=8,
                            max_iter=30,
                            tol=1e-7,
                            order_policy="fixed",
                        ),
                    }
                )
            simulator = simulator_cls(
                queue_root=str(self.config.queue_root),
                variable_order=self.variable_order,
                unit_map=self.config.unit_map,
                template_name=self.config.template_name,
                port_num=2,
                result_store=result_store,
                design_name=self.config.design_name,
                setup_name=self.config.setup_name,
                sweep_name=self.config.sweep_name,
                metrics_to_extract=["max_s11_db", "min_s21_db"],
                run_id=run_id,
                mm_resolution=0.01,
                cache_required_iterations=int(cache_required_iterations),
                job_timeout_sec=self.config.server_job_timeout_sec,
                wait_on_failed=self.config.server_wait_on_failed,
                verbose=self.config.debug_verbose,
                **simulator_kwargs,
            )
            return candidate_space, simulator
        if self.config.backend == "replay":
            if method == "owafs_bnn":
                raise ValueError("owafs_bnn requires --backend server because it submits explicit AFS frequency-point jobs")
            freq_cfg = self.cfg["data"]["freqs"]
            freqs = np.arange(freq_cfg["min"], freq_cfg["max"], freq_cfg["step"], dtype=float)
            replay_dataset = BCLFDataset.from_pickle(self.config.replay_dataset_path, freqs=freqs)
            candidate_space = DiscreteCandidateSpace(
                variable_order=self.variable_order,
                candidate_geometries=replay_dataset.X_pool,
                match_atol=replay_dataset.match_atol,
                match_rtol=replay_dataset.match_rtol,
            )
            simulator = ReplaySimulator(dataset=replay_dataset)
            return candidate_space, simulator
        raise ValueError(f"Unsupported backend={self.config.backend!r}. Use server or replay.")

    def effective_seed_for_rerun(self, method: str, seed: int) -> int:
        payload = self._load_checkpoint_payload(method, seed)
        if payload is None or self.config.reset_checkpoint:
            return int(seed)
        checkpoint_seed = payload.get("seed")
        return int(seed) if checkpoint_seed is None else int(checkpoint_seed)

    def is_seed_completed(self, method: str, seed: int) -> bool:
        if self.config.reset_checkpoint:
            return False
        return self._history_budgeted_count(self._checkpoint_history(method, seed)) >= self.config.target_trials

    def _history_budgeted_count(self, history: list[dict[str, Any]]) -> int:
        return sum(1 for row in history if bool(dict(row).get("counts_toward_budget", True)))

    def _load_checkpoint_payload(self, method: str, seed: int) -> dict[str, Any] | None:
        checkpoint_path = self.checkpoint_for_method_seed(method, seed)
        if not checkpoint_path.exists():
            return None
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))

    def _checkpoint_history(self, method: str, seed: int) -> list[dict[str, Any]]:
        payload = self._load_checkpoint_payload(method, seed)
        if payload is None:
            return []
        history = payload.get("history", [])
        return history if isinstance(history, list) else []

    def _history_frame(self, seed: int, history: list[dict[str, Any]]) -> pd.DataFrame:
        hist = pd.DataFrame(history)
        if hist.empty:
            return hist
        hist = hist.copy()
        hist["seed"] = int(seed)
        if "iteration" in hist:
            hist["iteration"] = pd.to_numeric(hist["iteration"], errors="coerce")
            hist = hist[np.isfinite(hist["iteration"])].copy()
            hist["iteration"] = hist["iteration"].astype(int)
            hist = hist.sort_values("iteration").reset_index(drop=True)
        return hist

    @staticmethod
    def _drop_in_memory_response_columns(history_df: pd.DataFrame) -> pd.DataFrame:
        if history_df.empty:
            return history_df
        drop_cols = [col for col in history_df.columns if str(col).startswith("_response_")]
        return history_df.drop(columns=drop_cols) if drop_cols else history_df

    def _summary_row_for_seed(self, method: str, seed: int, status: str, history_df: pd.DataFrame) -> dict[str, Any]:
        run_id = self.run_id_for_method_seed(method, seed)
        return {
            "seed": int(seed),
            "run_id": run_id,
            "method": METHOD_LABELS[method],
            "status": status,
            "num_trials": int(len(history_df)),
            "num_budgeted_trials": self._budgeted_history_count(history_df),
            "num_validations": self._validation_history_count(history_df),
            "best_objective": self._best_objective_from_history(history_df),
            "first_feasible_iteration": self._first_feasible_iteration(history_df),
            "cache_hit_ratio": self._cache_hit_ratio(history_df),
            "runtime_engine_sec": self._runtime_value(history_df, "t_engine_cum_sec"),
            "runtime_sim_sec": self._runtime_value(history_df, "t_simulator_cum_sec"),
            "json": str(self.method_output_dir(method) / f"{run_id}.json"),
            "csv": str(self.method_output_dir(method) / f"{run_id}.csv"),
            "checkpoint": str(self.checkpoint_for_method_seed(method, seed)),
            "initial_design_manifest": (
                str(self.initial_design_builder.manifest_path(seed))
                if self.method_needs_initial_design(method)
                else ""
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _first_feasible_iteration(self, history_df: pd.DataFrame) -> int | None:
        if history_df.empty or "feasible" not in history_df or "iteration" not in history_df:
            return None
        feasible_idx = history_df.index[history_df["feasible"] == True].tolist()  # noqa: E712
        return int(history_df.loc[feasible_idx[0], "iteration"]) if feasible_idx else None

    def _best_objective_from_history(self, history_df: pd.DataFrame) -> float:
        if history_df.empty:
            return float("nan")
        if "best_so_far" in history_df:
            best = pd.to_numeric(history_df["best_so_far"], errors="coerce")
            if np.isfinite(best).any():
                return float(np.nanmin(best))
        if "objective" in history_df:
            objective = pd.to_numeric(history_df["objective"], errors="coerce")
            if np.isfinite(objective).any():
                return float(np.nanmin(objective))
        return float("nan")

    def _budgeted_history_count(self, history_df: pd.DataFrame) -> int:
        if history_df.empty:
            return 0
        if "counts_toward_budget" not in history_df:
            return int(len(history_df))
        return int(history_df["counts_toward_budget"].fillna(True).astype(bool).sum())

    def _validation_history_count(self, history_df: pd.DataFrame) -> int:
        if history_df.empty or "counts_toward_budget" not in history_df:
            return 0
        return int((~history_df["counts_toward_budget"].fillna(True).astype(bool)).sum())

    def _cache_hit_ratio(self, history_df: pd.DataFrame) -> float:
        if history_df.empty or "cache_hit" not in history_df:
            return float("nan")
        values = history_df["cache_hit"].map(self._truthy_series_value)
        return float(values.mean())

    @staticmethod
    def _truthy_series_value(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value) if pd.notna(value) else False

    def _runtime_value(self, history_df: pd.DataFrame, column: str) -> float:
        if history_df.empty or column not in history_df:
            return float("nan")
        values = pd.to_numeric(history_df[column], errors="coerce")
        finite = values[np.isfinite(values)]
        return float(finite.iloc[-1]) if len(finite) else float("nan")

    def _progress_debug_callback(self, trial: dict[str, Any], history: list[dict[str, Any]]) -> None:
        if not self.config.debug_verbose:
            return
        iteration = int(trial.get("iteration", -1))
        if self.config.debug_print_every > 1 and iteration % int(self.config.debug_print_every) != 0:
            return
        self._log(
            "[Optimizer] "
            f"run={trial.get('optimizer_backend')} "
            f"iter={iteration:04d} "
            f"phase={trial.get('phase')} "
            f"obj={trial.get('objective')} "
            f"best={trial.get('best_so_far')} "
            f"status={trial.get('simulation_status')} "
            f"cache_hit={trial.get('cache_hit')} "
            f"job_id={trial.get('job_id')} "
            f"history={len(history)}"
        )

    def _server_cached_response(
        self,
        simulator: ServerQueueSimulator,
        geometry: np.ndarray,
        *,
        case_id: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
        values = np.asarray(geometry, dtype=float).reshape(-1)
        variables = {name: float(values[i]) for i, name in enumerate(simulator.variable_order)}
        quantized_variables = simulator._quantize_variables(variables)
        request = SimulationInput(
            case_id=case_id,
            variable_order=list(simulator.variable_order),
            variables=quantized_variables,
        )
        cached = simulator.result_store.lookup(request)
        if cached is None:
            return None
        freqs, s_params = simulator._load_touchstone(cached.touchstone_path)
        actual_geometry = np.array([quantized_variables[name] for name in simulator.variable_order], dtype=float)
        return freqs, s_params, actual_geometry, str(cached.touchstone_path)

    def _rehydrate_history_responses(
        self,
        optimizer: OptimizerBoTorch,
        simulator: ServerQueueSimulator | ReplaySimulator,
    ) -> dict[str, int]:
        # Checkpoints skip response arrays; BNN resume needs them in memory for training.
        stats = {"existing": 0, "rehydrated": 0, "missing": 0, "skipped": 0}
        for pos, row in enumerate(optimizer._history):
            if row.get("simulation_status") != "done":
                stats["skipped"] += 1
                continue
            if row.get("_response_freqs") is not None and row.get("_response_s_params") is not None:
                stats["existing"] += 1
                continue
            geometry = row.get("geometry") or row.get("requested_geometry")
            if geometry is None:
                stats["missing"] += 1
                continue
            geometry_arr = np.asarray(geometry, dtype=float).reshape(-1)
            metadata = dict(row.get("simulation_metadata") or {})
            direct_touchstone = self._direct_rehydrate_touchstone(metadata)
            if direct_touchstone is not None:
                try:
                    freqs, s_params = simulator._load_touchstone(direct_touchstone)  # type: ignore[attr-defined]
                except Exception:
                    stats["missing"] += 1
                    continue
                actual_geometry = geometry_arr
                touchstone_path = str(direct_touchstone)
                row["_response_freqs"] = np.asarray(freqs, dtype=float)
                row["_response_s_params"] = np.asarray(s_params, dtype=complex)
                row["geometry"] = np.asarray(actual_geometry, dtype=float).reshape(-1).tolist()
                row["geometry_normalized"] = np.asarray(
                    optimizer.candidate_space.normalize(np.asarray(actual_geometry, dtype=float)),
                    dtype=float,
                ).reshape(-1).tolist()
                metadata["touchstone_path"] = touchstone_path
                row["simulation_metadata"] = metadata
                stats["rehydrated"] += 1
                continue

            if isinstance(simulator, ReplaySimulator):
                sim_result = simulator.simulate(geometry_arr)
                if sim_result.status != "done":
                    stats["missing"] += 1
                    continue
                freqs = sim_result.freqs
                s_params = sim_result.s_params
                actual_geometry = sim_result.actual_geometry if sim_result.actual_geometry is not None else geometry_arr
                touchstone_path = None
            elif isinstance(simulator, ServerQueueSimulator):
                case_id = str(row.get("case_id") or f"rehydrate_{pos:04d}")
                cached = self._server_cached_response(simulator, geometry_arr, case_id=case_id)
                if cached is None:
                    stats["missing"] += 1
                    continue
                freqs, s_params, actual_geometry, touchstone_path = cached
            else:
                stats["missing"] += 1
                continue

            row["_response_freqs"] = np.asarray(freqs, dtype=float)
            row["_response_s_params"] = np.asarray(s_params, dtype=complex)
            row["geometry"] = np.asarray(actual_geometry, dtype=float).reshape(-1).tolist()
            row["geometry_normalized"] = np.asarray(
                optimizer.candidate_space.normalize(np.asarray(actual_geometry, dtype=float)),
                dtype=float,
            ).reshape(-1).tolist()
            if touchstone_path is not None:
                metadata = dict(row.get("simulation_metadata") or {})
                metadata["touchstone_path"] = touchstone_path
                row["simulation_metadata"] = metadata
            stats["rehydrated"] += 1
        self._log(f"[HistoryRehydrate] {stats}")
        if stats["missing"]:
            self._log(
                "[HistoryRehydrate] Missing responses will make BNN fitting fall back "
                "until new in-memory responses are available."
            )
        return stats

    def _direct_rehydrate_touchstone(self, metadata: dict[str, Any]) -> Path | None:
        for key in ("vf_reconstructed_touchstone_path", "touchstone_path"):
            value = metadata.get(key)
            if not isinstance(value, str) or not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = self.config.repo_root / path
            if path.exists():
                return path
        return None

    def _cleanup_queue_artifacts_for_run(self, queue_root: Path, run_id: str) -> dict[str, int]:
        removed = {"pending": 0, "running": 0, "done": 0, "failed": 0}
        for state in ("pending", "running", "done", "failed"):
            state_dir = queue_root / state
            if not state_dir.exists():
                continue
            suffix = ".result.json" if state in ("done", "failed") else ".json"
            for artifact_path in state_dir.glob(f"{run_id}_it*{suffix}"):
                artifact_path.unlink(missing_ok=True)
                removed[state] += 1
        return removed

    def _load_yaml_config(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"config_path does not exist: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _load_variable_bounds(self) -> dict[str, tuple[float, float]]:
        reg_min = list(self.cfg["data"]["regularization"]["min"])
        reg_max = list(self.cfg["data"]["regularization"]["max"])
        if len(reg_min) == len(self.variable_order) + 1:
            reg_min = reg_min[: len(self.variable_order)]
            reg_max = reg_max[: len(self.variable_order)]
        if len(reg_min) != len(self.variable_order):
            raise ValueError(
                f"Regularization dimension mismatch: len(min)={len(reg_min)} "
                f"variable_order={len(self.variable_order)}"
            )
        return {name: (float(reg_min[i]), float(reg_max[i])) for i, name in enumerate(self.variable_order)}

    def _configure_process_io(self) -> None:
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(line_buffering=True)

    def _archive_initial_design_policy_change(self) -> None:
        if not self.config.archive_on_policy_change:
            return
        existing_modes = self._existing_initial_design_modes()
        accepted_modes = self.initial_design_builder.accepted_modes()
        if not existing_modes:
            return
        if existing_modes.issubset(accepted_modes):
            if existing_modes != {self.initial_design_builder.mode}:
                self._log(
                    "[InitialDesignCompatibility] accepting existing manifest modes "
                    f"{sorted(existing_modes)} as {self.initial_design_builder.mode}; no archive or rebuild."
                )
            return

        archive_root = self._next_archive_root()
        archive_root.mkdir(parents=True, exist_ok=False)
        moved: list[tuple[Path, Path]] = []
        for path in self._active_artifact_paths_for_archive():
            if not self._path_has_content(path):
                continue
            destination = archive_root / path.name
            shutil.move(str(path), str(destination))
            moved.append((path, destination))

        if moved:
            self._log(
                "[Archive] initial design policy changed "
                f"{sorted(existing_modes)} -> {self.initial_design_builder.mode}; "
                f"moved {len(moved)} artifact dirs to {archive_root}"
            )
            for source, destination in moved:
                self._log(f"[Archive] {source} -> {destination}")
        else:
            archive_root.rmdir()

    def _existing_initial_design_modes(self) -> set[str]:
        modes: set[str] = set()
        initial_dir = self.config.initial_design_dir
        if not initial_dir.exists():
            return modes
        for path in sorted(initial_dir.glob("seed*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                modes.add("unreadable")
                continue
            modes.add(str(payload.get("mode", "unknown")))
        return modes

    def _next_archive_root(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        base = self.config.output_dir / "archive" / f"{timestamp}_pre_{self.initial_design_builder.mode}"
        if not base.exists():
            return base
        suffix = 1
        while True:
            candidate = base.with_name(f"{base.name}_{suffix}")
            if not candidate.exists():
                return candidate
            suffix += 1

    def _active_artifact_paths_for_archive(self) -> list[Path]:
        paths = [self.config.initial_design_dir]
        paths.extend(self.method_output_dir(method) for method in METHOD_CHOICES)
        checkpoint_dir = self.config.resolved_bnn_checkpoint_dir
        try:
            checkpoint_dir.relative_to(self.config.output_dir)
        except ValueError:
            pass
        else:
            paths.append(checkpoint_dir)
        unique: list[Path] = []
        seen: set[Path] = set()
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(path)
        return unique

    @staticmethod
    def _path_has_content(path: Path) -> bool:
        if not path.exists():
            return False
        if path.is_file():
            return True
        return any(path.iterdir())

    def _ensure_output_dirs(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.initial_design_dir.mkdir(parents=True, exist_ok=True)
        self.config.resolved_bnn_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if "owafs_bnn" in self.config.methods:
            self.config.resolved_afs_root.mkdir(parents=True, exist_ok=True)
        for method in self.config.methods:
            self.method_output_dir(method).mkdir(parents=True, exist_ok=True)

    def _log_header(self) -> None:
        self._log(f"[{RUNNER_LOG_TAG}] started: {datetime.now(timezone.utc).isoformat()}")
        self._log(f"repo_root: {self.config.repo_root}")
        self._log(f"config: {self.config.config_path} exists={self.config.config_path.exists()}")
        self._log(f"experiment_name: {self.config.experiment_name}")
        self._log(f"output: {self.config.output_dir}")
        self._log(f"methods: {self.config.methods}")
        self._log(f"seeds: {self.config.seeds}")
        self._log(f"total_trials: {self.config.total_trials}")
        self._log(f"warmup_size: {self.config.warmup_size}")
        self._log(f"legacy_budget_arg: {self.config.budget}")
        self._log(f"target_trials: {self.config.target_trials}")
        self._log(f"initial_design_seed_offset: {self.config.initial_design_seed_offset}")
        self._log(
            "method_trial_plan: "
            + json.dumps(
                {
                    method: {
                        "n_init": self.method_n_init(method),
                        "budget": self.method_budget(method),
                        "target_trials": int(self.config.total_trials),
                        "cache_required_init": self._cache_required_iterations(method, self.method_n_init(method)),
                    }
                    for method in self.config.methods
                },
                sort_keys=True,
            )
        )
        self._log(f"initial_design_source: {self.config.initial_design_source}")
        self._log(f"initial_design_mode: {self.initial_design_builder.mode}")
        if self.config.initial_design_source == "previous-lhs":
            self._log(f"previous_lhs_dir: {self.config.previous_lhs_dir}")
            self._log(f"lhs_dataset_dir: {self.config.lhs_dataset_dir}")
            self._log(f"rebuild_lhs_dataset: {self.config.rebuild_lhs_dataset}")
        self._log(f"cache_dir: {self.config.cache_dir}")
        if "owafs_bnn" in self.config.methods:
            self._log(f"afs_root: {self.config.resolved_afs_root}")
        self._log(f"initial_design_dir: {self.config.initial_design_dir}")
        self._log(f"bnn_checkpoint_dir: {self.config.resolved_bnn_checkpoint_dir}")
        self._log(f"backend: {self.config.backend}")
        self._log_bnn_config()
        self._log_owafs_bnn_config()

    def _log_bnn_config(self) -> None:
        if not any(method in self.config.methods for method in ("bnn", "owafs_bnn")):
            return
        summary = self._bnn_log_summary()
        network = summary["network"]
        training = summary["training"]
        acquisition = summary["acquisition"]
        frequency = summary["frequency_encoding"]
        checkpoint = summary["checkpoint_policy"]
        self._log(
            "[BNNConfig] type="
            f"{summary['surrogate_variant']} response_surrogate=true target_mode={summary['target_mode']}"
        )
        self._log(
            "[BNNConfig] input_dim="
            f"{summary['input_dim']} geometry_dim={summary['geometry_dim']} "
            f"frequency_feature_dim={summary['frequency_feature_dim']} "
            f"frequency_encoding={frequency}"
        )
        self._log(
            "[BNNConfig] network/training="
            f"hidden_dims={network['hidden_dims']} prior_sigma={network['prior_sigma']} "
            f"initial_epochs={training['initial_epochs']} update_epochs={training['update_epochs']} "
            f"update_epoch_schedule={training['update_epoch_schedule']} "
            f"update_epoch_multiplier={training['update_epoch_multiplier']} "
            f"warm_start={training['warm_start']} initial_train_size={training['initial_train_size']} "
            f"batch_size={training['batch_size']} lr={training['lr']} "
            f"bayesian_loss_weight={training['bayesian_loss_weight']}"
        )
        self._log(
            "[BNNConfig] acquisition="
            f"{acquisition['name']} mc_samples={acquisition['mc_samples']} beta={acquisition['beta']}"
        )
        self._log(
            "[BNNConfig] checkpoint_policy="
            f"dir={checkpoint['dir']} pattern={checkpoint['per_seed_pattern']} "
            f"force_retrain={checkpoint['force_retrain']} "
            f"require_checkpoint={checkpoint['require_checkpoint']} "
            f"save_checkpoint={checkpoint['save_checkpoint']}"
        )
        self._log(f"[BNNConfigJSON] {json.dumps(summary, sort_keys=True)}")

    def _log_owafs_bnn_config(self) -> None:
        if "owafs_bnn" not in self.config.methods:
            return
        summary = {
            "method": "owafs_bnn",
            "display_name": METHOD_LABELS["owafs_bnn"],
            "afs_root": str(self.config.resolved_afs_root / "owafs_bnn"),
            "sample_count": int(self.config.owafs_sample_count),
            "initial_anchor_count": 9,
            "boundary_ghz": [3.6, 4.4, 4.8, 5.4, 5.8],
            "score_formula": "band_weight * 0.5 * (rank01(vf_residual_db) + rank01(mc_uncertainty_db))",
            "band_weights": {"boundary": 2.0, "passband": 1.5, "stopband": 1.25, "else": 1.0},
            "vf": {
                "n_poles_real": 4,
                "n_poles_complex": 8,
                "max_iter": 30,
                "tol": 1e-7,
                "order_policy": "fixed",
            },
            "validation_policy": {
                "trigger": "afs_vf_objective_improves_current_best",
                "recording": "append_full_validation_trial",
                "counts_toward_budget": False,
                "paired_afs_row_excluded_from_training": True,
            },
        }
        self._log(
            "[ObjectiveWeightedUncertaintyAwareAFSConfig] "
            f"display_name={summary['display_name']} "
            f"sample_count={summary['sample_count']} afs_root={summary['afs_root']} "
            f"vf={summary['vf']} validation={summary['validation_policy']}"
        )
        self._log(f"[ObjectiveWeightedUncertaintyAwareAFSConfigJSON] {json.dumps(summary, sort_keys=True)}")

    def _bnn_log_summary(self) -> dict[str, Any]:
        config = dict(self.config.bnn_backend_config)
        variant = str(config.get("surrogate_variant", "raw")).strip().lower()
        geometry_dim = len(self.variable_order)
        if variant == "fourier":
            bands = [float(v) for v in config.get("fourier_bands", [])]
            include_raw_frequency = bool(config.get("include_raw_frequency", True))
            frequency_feature_dim = (1 if include_raw_frequency else 0) + 2 * len(bands)
            frequency_encoding: dict[str, Any] = {
                "kind": "fourier",
                "bands": bands,
                "include_raw_frequency": include_raw_frequency,
                "raw_frequency_scale": "normalized_0_1" if include_raw_frequency else None,
                "sin_cos_feature_dim": 2 * len(bands),
                "frequency_bounds_source": "fit_dataset_min_max",
            }
        else:
            frequency_feature_dim = 1
            frequency_encoding = {
                "kind": "raw",
                "raw_frequency_scale": "hz",
                "frequency_bounds_source": None,
            }
        return {
            "surrogate_variant": variant,
            "target_mode": "db",
            "geometry_dim": int(geometry_dim),
            "frequency_feature_dim": int(frequency_feature_dim),
            "input_dim": int(geometry_dim + frequency_feature_dim),
            "frequency_encoding": frequency_encoding,
            "network": {
                "hidden_dims": list(config.get("hidden_dims", [])),
                "prior_sigma": float(config.get("prior_sigma", 0.1)),
            },
            "training": {
                "initial_epochs": int(config.get("epochs", 0)),
                "update_epochs": int(config.get("update_epochs", 0)),
                "update_epoch_schedule": str(config.get("update_epoch_schedule", "constant")),
                "update_epoch_multiplier": int(config.get("update_epoch_multiplier", self.config.update_epochs)),
                "warm_start": bool(config.get("warm_start", True)),
                "initial_train_size": int(config.get("initial_train_size", self.config.warmup_size)),
                "batch_size": int(config.get("batch_size", 0)),
                "lr": float(config.get("lr", 0.0)),
                "bayesian_loss_weight": float(config.get("bayesian_loss_weight", 0.0)),
            },
            "acquisition": {
                "name": str(config.get("acquisition", "qlogei")),
                "mc_samples": int(config.get("mc_samples", self.config.mc_samples)),
                "beta": float(config.get("beta", 2.0)),
            },
            "checkpoint_policy": {
                "dir": str(self.config.resolved_bnn_checkpoint_dir),
                "per_seed_pattern": f"{{method}}_{self.config.experiment_name}_seed{{seed}}.bnn.pt",
                "force_retrain": True,
                "require_checkpoint": False,
                "save_checkpoint": True,
            },
        }

    def _log_planned_jobs(self, initial_designs: dict[int, SharedInitialDesign]) -> None:
        rows = []
        for method in self.config.methods:
            for seed in self.config.seeds:
                seed = int(seed)
                method_n_init = self.method_n_init(method)
                initial_design = initial_designs.get(seed) if self.method_needs_initial_design(method) else None
                bnn_checkpoint = str(self.bnn_checkpoint_for_seed(seed, method=method)) if method in {"bnn", "owafs_bnn"} else ""
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "run_id": self.run_id_for_method_seed(method, seed),
                        "n_init": method_n_init,
                        "budget": self.method_budget(method),
                        "target_trials": int(self.config.total_trials),
                        "cache_required_init": self._cache_required_iterations(method, method_n_init),
                        "checkpoint": str(self.checkpoint_for_method_seed(method, seed)),
                        "bnn_checkpoint": bnn_checkpoint,
                        "afs_root": str(self.config.resolved_afs_root / "owafs_bnn") if method == "owafs_bnn" else "",
                        "initial_design": "" if initial_design is None else str(initial_design.path),
                        "initial_design_mode": "" if initial_design is None else str(initial_design.payload.get("mode", "")),
                        "initial_points": 0 if initial_design is None else len(initial_design.selected_rows),
                    }
                )
        self._log(pd.DataFrame(rows).to_string(index=False))

    def _log(self, message: Any) -> None:
        print(str(message), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BO/LHS/Turbo/BNN online BCLF experiments with LHS-database BNN warmup.",
    )
    parser.add_argument("--backend", choices=["server", "replay"], default="server")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_CHOICES),
        help=(
            "Methods to run. Choose from bo, lhs, turbo, bnn, owafs_bnn. "
            "Accepts space-separated values or comma-separated groups. "
            "Default: all methods."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--total-trials", type=int, default=None)
    parser.add_argument("--warmup-size", type=int, default=None)
    parser.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="Deprecated alias for --warmup-size for BNN/OWAFS-BNN warmup.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Deprecated compatibility option. If --total-trials is omitted, total_trials = warmup_size + budget.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=5000,
        help="Per-step continuous resampled candidate batch size for BO/BNN-style selection.",
    )
    parser.add_argument("--retrain-every", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments") / DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--config-path", type=Path, default=Path("configs/filter_bnn_online_uaw_uaw.yaml"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/bandpass_coupled_line_filter_export/ext"))
    parser.add_argument(
        "--initial-design-source",
        choices=INITIAL_DESIGN_SOURCE_CHOICES,
        default="previous-lhs",
        help=(
            "Warmup source for BNN/OWAFS-BNN. Use random for fresh random geometries that will be "
            "simulated online, or previous-lhs for random rows from the reusable previous-LHS metadata database."
        ),
    )
    parser.add_argument(
        "--initial-design-seed-offset",
        type=int,
        default=0,
        help="Added to each target seed before sampling warmup initial designs; change it to get a new draw.",
    )
    parser.add_argument("--previous-lhs-dir", type=Path, default=Path("experiments/lhs_online_general"))
    parser.add_argument(
        "--lhs-dataset-dir",
        type=Path,
        default=Path("data/bandpass_coupled_line_filter_export/ext/lhs"),
    )
    parser.add_argument("--rebuild-lhs-dataset", action="store_true")
    parser.add_argument(
        "--no-archive-on-policy-change",
        action="store_false",
        dest="archive_on_policy_change",
        default=True,
        help="Disable automatic archive when existing manifests use a different initial-design policy.",
    )
    parser.add_argument("--afs-root", type=Path, default=None)
    parser.add_argument("--owafs-sample-count", type=int, default=20)
    parser.add_argument("--queue-root", type=Path, default=Path("queue"))
    parser.add_argument("--server-job-timeout-sec", type=float, default=0.0)
    parser.add_argument("--server-wait-on-failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replay-dataset-path", type=Path, default=Path("data/filter_bnn_online_uaw_uaw_dataset.pkl"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initial-epochs", type=int, default=800)
    parser.add_argument("--update-epochs", type=int, default=20)
    parser.add_argument("--mc-samples", type=int, default=64)
    parser.add_argument("--bnn-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument(
        "--no-clear-queue-artifacts",
        action="store_false",
        dest="clear_queue_artifacts_on_reset",
        default=True,
    )
    parser.add_argument("--rebuild-initial-designs", action="store_true")
    parser.add_argument("--generate-initial-design-only", action="store_true")
    parser.add_argument("--debug-print-every", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def normalize_methods(values: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for raw in values:
        for item in str(raw).split(","):
            method = item.strip().lower()
            if not method:
                continue
            if method not in METHOD_CHOICES:
                raise ValueError(f"Unsupported method={method!r}; choose from {METHOD_CHOICES}")
            if method not in normalized:
                normalized.append(method)
    if not normalized:
        raise ValueError("--methods must contain at least one method")
    return normalized


def config_from_args(args: argparse.Namespace) -> SharedOnlineExperimentConfig:
    warmup_size = int(args.warmup_size if args.warmup_size is not None else (args.n_init if args.n_init is not None else 15))
    if args.total_trials is not None:
        total_trials = int(args.total_trials)
    elif args.budget is not None:
        total_trials = warmup_size + int(args.budget)
    else:
        total_trials = 80
    return SharedOnlineExperimentConfig(
        repo_root=REPO_ROOT,
        backend=args.backend,
        methods=normalize_methods(args.methods),
        seeds=[int(seed) for seed in args.seeds],
        experiment_name=str(args.experiment_name),
        total_trials=int(total_trials),
        warmup_size=int(warmup_size),
        n_init=None if args.n_init is None else int(args.n_init),
        budget=None if args.budget is None else int(args.budget),
        pool_size=int(args.pool_size),
        retrain_every=int(args.retrain_every),
        initial_epochs=int(args.initial_epochs),
        update_epochs=int(args.update_epochs),
        device=str(args.device),
        mc_samples=int(args.mc_samples),
        config_path=resolve_path(args.config_path),
        replay_dataset_path=resolve_path(args.replay_dataset_path),
        output_dir=resolve_path(args.output_dir),
        bnn_checkpoint_dir=None if args.bnn_checkpoint_dir is None else resolve_path(args.bnn_checkpoint_dir),
        cache_dir=resolve_path(args.cache_dir),
        initial_design_source=str(args.initial_design_source),
        initial_design_seed_offset=int(args.initial_design_seed_offset),
        previous_lhs_dir=resolve_path(args.previous_lhs_dir),
        lhs_dataset_dir=resolve_path(args.lhs_dataset_dir),
        rebuild_lhs_dataset=bool(args.rebuild_lhs_dataset),
        archive_on_policy_change=bool(args.archive_on_policy_change),
        afs_root=None if args.afs_root is None else resolve_path(args.afs_root),
        owafs_sample_count=int(args.owafs_sample_count),
        queue_root=resolve_path(args.queue_root),
        server_job_timeout_sec=float(args.server_job_timeout_sec),
        server_wait_on_failed=bool(args.server_wait_on_failed),
        resume=bool(args.resume),
        reset_checkpoint=bool(args.reset_checkpoint),
        clear_queue_artifacts_on_reset=bool(args.clear_queue_artifacts_on_reset),
        rebuild_initial_designs=bool(args.rebuild_initial_designs),
        generate_initial_design_only=bool(args.generate_initial_design_only),
        debug_verbose=not bool(args.quiet),
        debug_print_every=int(args.debug_print_every),
        dry_run=bool(args.dry_run),
        log_file=None if args.log_file is None else resolve_path(args.log_file),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    for name in ("pool_size", "retrain_every", "initial_epochs", "update_epochs", "mc_samples", "debug_print_every", "owafs_sample_count"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    for name in ("n_init", "warmup_size", "budget"):
        value = getattr(args, name)
        if value is not None and int(value) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.total_trials is not None and int(args.total_trials) <= 0:
        parser.error("--total-trials must be > 0")
    if not args.seeds:
        parser.error("--seeds must contain at least one seed")
    try:
        config = config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not str(config.experiment_name).strip():
        parser.error("--experiment-name must be non-empty")
    if int(config.total_trials) <= 0:
        parser.error("--total-trials must be > 0")
    if int(config.warmup_size) < 0:
        parser.error("--warmup-size must be >= 0")
    for method in config.methods:
        if int(config.total_trials) < (int(config.warmup_size) if method in WARMUP_METHODS else 0):
            parser.error(f"--total-trials must be >= warmup size for method {method}")
    with tee_stdout_stderr(config.log_file):
        try:
            return SharedOnlineExperimentRunner(config).run()
        except KeyboardInterrupt:
            print(f"[{RUNNER_LOG_TAG}] interrupted by KeyboardInterrupt", flush=True)
            return 130
        except Exception:
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
