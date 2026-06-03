#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
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
    OptimizerBoTorch,
    OptimizerLHSRandom,
    ReplaySimulator,
    ServerQueueSimulator,
)
from simulation.core import SimulationInput  # noqa: E402
from simulation.result_store import BandpassFilterNamingStrategy, FilesystemResultStore  # noqa: E402


METHOD_CHOICES = ("bo", "lhs", "turbo", "bnn")
METHOD_LABELS = {"bo": "BO", "lhs": "LHS", "turbo": "Turbo", "bnn": "BNN"}


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
        for row in self.selected_rows:
            metadata.append(
                {
                    "initial_design_seed": int(self.seed),
                    "initial_design_manifest": str(self.path),
                    "initial_design_mode": "lhs_nearest_cached_geometry",
                    "lhs_target_normalized": row.get("lhs_target_normalized"),
                    "nearest_distance_normalized": row.get("nearest_distance_normalized"),
                    "cache_filename": row.get("cache_filename"),
                }
            )
        return metadata


class SharedInitialDesignBuilder:
    """Build seed-specific cached LHS initial designs for all online methods."""

    def __init__(
        self,
        *,
        variable_order: Sequence[str],
        variable_bounds: dict[str, tuple[float, float]],
        cache_dir: Path,
        manifest_dir: Path,
        naming_strategy: BandpassFilterNamingStrategy | None = None,
        port_num: int = 2,
    ) -> None:
        self.variable_order = list(variable_order)
        self.variable_bounds = {k: (float(v[0]), float(v[1])) for k, v in variable_bounds.items()}
        self.cache_dir = Path(cache_dir)
        self.manifest_dir = Path(manifest_dir)
        self.naming_strategy = naming_strategy or BandpassFilterNamingStrategy()
        self.port_num = int(port_num)
        self._inventory_cache: list[dict[str, Any]] | None = None

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
        inventory = self.cache_inventory()
        if len(inventory) < n_init:
            raise RuntimeError(
                f"Need at least {n_init} cached geometries within bounds, found {len(inventory)}"
            )

        targets = self._lhs_targets(seed=seed, n_init=n_init)
        selected = self._select_nearest_unique(targets, inventory)
        return {
            "schema_version": 1,
            "seed": int(seed),
            "n_init": int(n_init),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "lhs_nearest_cached_geometry",
            "variable_order": list(self.variable_order),
            "bounds": {name: list(self.variable_bounds[name]) for name in self.variable_order},
            "cache_dir": str(self.cache_dir),
            "cache_count_within_bounds": int(len(inventory)),
            "port_num": int(self.port_num),
            "selected": selected,
        }

    def validate(self, *, design: SharedInitialDesign, n_init: int) -> None:
        payload = design.payload
        if int(payload.get("seed")) != int(design.seed):
            raise ValueError(f"Initial design seed mismatch in {design.path}")
        if int(payload.get("n_init")) != int(n_init):
            raise ValueError(f"Initial design n_init mismatch in {design.path}")
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

        store = FilesystemResultStore(
            root_dir=self.cache_dir,
            naming_strategy=self.naming_strategy,
            port_num=self.port_num,
        )
        for idx, row in enumerate(design.selected_rows):
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

    def cache_inventory(self) -> list[dict[str, Any]]:
        if self._inventory_cache is None:
            self._inventory_cache = self._load_cache_inventory()
        return list(self._inventory_cache)

    def _load_cache_inventory(self) -> list[dict[str, Any]]:
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"cache_dir does not exist: {self.cache_dir}")
        seen: set[tuple[float, ...]] = set()
        rows: list[dict[str, Any]] = []
        for path in sorted(self.cache_dir.glob("*.s2p")):
            variables = self._parse_touchstone_filename(path.name)
            if variables is None or not self._variables_within_bounds(variables):
                continue
            geometry = [float(variables[name]) for name in self.variable_order]
            key = tuple(round(value, 8) for value in geometry)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "geometry": geometry,
                    "variables": {name: float(variables[name]) for name in self.variable_order},
                    "cache_filename": path.name,
                }
            )
        if not rows:
            raise RuntimeError(f"No cached touchstones within bounds in {self.cache_dir}")
        return rows

    def _parse_touchstone_filename(self, filename: str) -> dict[str, float] | None:
        suffix = ".s2p"
        if not filename.endswith(suffix):
            return None
        stem = filename[: -len(suffix)]
        prefix = self.naming_strategy.prefix
        if not stem.startswith(prefix + "_"):
            return None
        parts = stem[len(prefix) + 1 :].split("_")
        if len(parts) % 2 != 0:
            return None
        parsed: dict[str, float] = {}
        for pos in range(0, len(parts), 2):
            name = parts[pos]
            value_token = parts[pos + 1]
            if name in self.variable_order:
                parsed[name] = self._decode_touchstone_value(value_token)
        if any(name not in parsed for name in self.variable_order):
            return None
        return parsed

    @staticmethod
    def _decode_touchstone_value(token: str) -> float:
        sign = -1.0 if token.startswith("m") else 1.0
        text = token[1:] if token.startswith("m") else token
        return float(sign * float(text.replace("d", ".")))

    def _variables_within_bounds(self, variables: dict[str, float]) -> bool:
        for name in self.variable_order:
            value = float(variables[name])
            low, high = self.variable_bounds[name]
            if value < low - 1e-9 or value > high + 1e-9:
                return False
        return True

    def _lhs_targets(self, *, seed: int, n_init: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        dim = len(self.variable_order)
        targets = np.empty((int(n_init), dim), dtype=float)
        for col in range(dim):
            strata = rng.permutation(int(n_init))
            jitter = rng.random(int(n_init))
            targets[:, col] = (strata + jitter) / float(n_init)
        return targets

    def _select_nearest_unique(self, targets: np.ndarray, inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
        geometries = np.asarray([row["geometry"] for row in inventory], dtype=float)
        lows = np.asarray([self.variable_bounds[name][0] for name in self.variable_order], dtype=float)
        highs = np.asarray([self.variable_bounds[name][1] for name in self.variable_order], dtype=float)
        span = np.where(np.abs(highs - lows) < 1e-12, 1.0, highs - lows)
        normalized = np.clip((geometries - lows.reshape(1, -1)) / span.reshape(1, -1), 0.0, 1.0)

        used: set[int] = set()
        selected: list[dict[str, Any]] = []
        for initial_index, target in enumerate(np.asarray(targets, dtype=float)):
            distances = np.linalg.norm(normalized - target.reshape(1, -1), axis=1)
            order = np.argsort(distances)
            selected_pos = next((int(pos) for pos in order if int(pos) not in used), None)
            if selected_pos is None:
                raise RuntimeError("Could not select a unique cached geometry for every LHS target")
            used.add(selected_pos)
            inventory_row = inventory[selected_pos]
            selected.append(
                {
                    "initial_index": int(initial_index),
                    "lhs_target_normalized": target.tolist(),
                    "geometry": [float(value) for value in inventory_row["geometry"]],
                    "variables": dict(inventory_row["variables"]),
                    "cache_filename": str(inventory_row["cache_filename"]),
                    "nearest_distance_normalized": float(distances[selected_pos]),
                }
            )
        return selected


@dataclass(frozen=True)
class SharedOnlineExperimentConfig:
    repo_root: Path = REPO_ROOT
    backend: str = "server"
    methods: list[str] = field(default_factory=lambda: list(METHOD_CHOICES))
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_init: int = 15
    budget: int = 65
    pool_size: int = 1500
    retrain_every: int = 1
    initial_epochs: int = 800
    update_epochs: int = 20
    device: str = "cuda"
    mc_samples: int = 64
    config_path: Path = REPO_ROOT / "configs" / "filter_bnn_online_uaw_uaw.yaml"
    replay_dataset_path: Path = REPO_ROOT / "data" / "filter_bnn_online_uaw_uaw_dataset.pkl"
    output_dir: Path = REPO_ROOT / "experiments" / "online_lhs15_shared_init"
    bnn_checkpoint_dir: Path | None = None
    cache_dir: Path = REPO_ROOT / "data" / "bandpass_coupled_line_filter_export" / "ext"
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
        return int(self.n_init + self.budget)

    @property
    def initial_design_dir(self) -> Path:
        return self.output_dir / "initial_designs"

    @property
    def resolved_bnn_checkpoint_dir(self) -> Path:
        if self.bnn_checkpoint_dir is not None:
            return self.bnn_checkpoint_dir
        return self.output_dir / "bnn_checkpoints"

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
            "warm_start": True,
            "initial_train_size": int(self.n_init),
            "batch_size": 1024,
            "lr": 1e-3,
            "bayesian_loss_weight": 1e-4,
            "mc_samples": int(self.mc_samples),
            "acquisition": "qlogei",
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
        self.initial_design_builder = SharedInitialDesignBuilder(
            variable_order=self.variable_order,
            variable_bounds=self.variable_bounds,
            cache_dir=self.config.cache_dir,
            manifest_dir=self.config.initial_design_dir,
            port_num=2,
        )

    def run(self) -> int:
        self._configure_process_io()
        self._ensure_output_dirs()
        self._log_header()

        initial_designs = {
            int(seed): self.initial_design_builder.build_or_load(
                seed=int(seed),
                n_init=self.config.n_init,
                rebuild=self.config.rebuild_initial_designs,
                write=True,
            )
            for seed in self.config.seeds
        }

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
                self._log(
                    f"[Campaign] method={method} seed={seed} optimizer_seed={optimizer_seed} {status_label}; "
                    f"checkpoint={checkpoint_path} initial_design={initial_designs[seed].path}"
                )
                optimizer = self.build_optimizer(method, seed, initial_designs[seed])
                run_result = optimizer.run()
                latest_history_df = pd.DataFrame(run_result.history).sort_values("iteration").reset_index(drop=True)
                latest_history_df["seed"] = seed
                final_status = "completed" if len(latest_history_df) >= self.config.target_trials else "in_progress"
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

    def method_output_dir(self, method: str) -> Path:
        return self.config.output_dir / method

    def campaign_summary_path(self, method: str) -> Path:
        return self.method_output_dir(method) / "campaign_summary.csv"

    def run_id_for_method_seed(self, method: str, seed: int) -> str:
        return f"{method}_lhs15_shared_seed{int(seed)}"

    def checkpoint_for_method_seed(self, method: str, seed: int) -> Path:
        run_id = self.run_id_for_method_seed(method, seed)
        return self.method_output_dir(method) / f"{run_id}.checkpoint.json"

    def bnn_checkpoint_for_seed(self, seed: int) -> Path:
        return self.config.resolved_bnn_checkpoint_dir / f"bnn_lhs15_shared_seed{int(seed)}.bnn.pt"

    def bnn_backend_config_for_seed(self, seed: int) -> dict[str, Any]:
        payload = dict(self.config.bnn_backend_config)
        payload.update(
            {
                "checkpoint_path": str(self.bnn_checkpoint_for_seed(seed)),
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
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = df.sort_values("seed").reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df

    def build_optimizer(self, method: str, seed: int, initial_design: SharedInitialDesign):
        method = method.lower()
        run_id = self.run_id_for_method_seed(method, seed)
        checkpoint_path = self.checkpoint_for_method_seed(method, seed)
        optimizer_seed = self.effective_seed_for_rerun(method, seed)

        if self.config.reset_checkpoint:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            if method == "bnn":
                self.bnn_checkpoint_for_seed(seed).unlink(missing_ok=True)
            if self.config.clear_queue_artifacts_on_reset:
                removed = self._cleanup_queue_artifacts_for_run(self.config.queue_root, run_id)
                self._log(f"[ResetCleanup] run_id={run_id} {removed}")

        candidate_space, simulator = self._build_candidate_space_and_simulator(
            method=method,
            seed=optimizer_seed,
            run_id=run_id,
        )
        common_kwargs = {
            "candidate_space": candidate_space,
            "simulator": simulator,
            "optimization_spec": self.spec,
            "recorder": ExperimentRecorder(output_dir=self.method_output_dir(method)),
            "budget": self.config.budget,
            "n_init": self.config.n_init,
            "seed": optimizer_seed,
            "run_id": run_id,
            "checkpoint_path": checkpoint_path,
            "resume": self.config.resume and checkpoint_path.exists(),
            "initial_geometries": initial_design.geometries,
            "initial_geometry_metadata": initial_design.metadata,
            "progress_callback": self._progress_debug_callback,
        }

        if method == "lhs":
            optimizer = OptimizerLHSRandom(**common_kwargs)
        elif method == "turbo":
            optimizer = OptimizerBoTorch(
                **common_kwargs,
                retrain_every=self.config.retrain_every,
                use_trust_region=True,
            )
        elif method == "bnn":
            optimizer = OptimizerBoTorch(
                **common_kwargs,
                retrain_every=self.config.retrain_every,
                acquisition_backend=BNNExpectedImprovementBackend(
                    self.bnn_backend_config_for_seed(seed),
                    device=self.config.device,
                ),
            )
        elif method == "bo":
            optimizer = OptimizerBoTorch(**common_kwargs, retrain_every=self.config.retrain_every)
        else:
            raise ValueError(f"Unsupported method={method!r}")

        if method == "bnn" and self.config.resume and checkpoint_path.exists() and len(optimizer._history):
            self._rehydrate_history_responses(optimizer, simulator)
        return optimizer

    def _build_candidate_space_and_simulator(self, *, method: str, seed: int, run_id: str):
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
            simulator = ServerQueueSimulator(
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
                cache_required_iterations=self.config.n_init,
                job_timeout_sec=self.config.server_job_timeout_sec,
                wait_on_failed=self.config.server_wait_on_failed,
                verbose=self.config.debug_verbose,
            )
            return candidate_space, simulator
        if self.config.backend == "replay":
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
        return len(self._checkpoint_history(method, seed)) >= self.config.target_trials

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

    def _summary_row_for_seed(self, method: str, seed: int, status: str, history_df: pd.DataFrame) -> dict[str, Any]:
        run_id = self.run_id_for_method_seed(method, seed)
        return {
            "seed": int(seed),
            "run_id": run_id,
            "method": METHOD_LABELS[method],
            "status": status,
            "num_trials": int(len(history_df)),
            "best_objective": self._best_objective_from_history(history_df),
            "first_feasible_iteration": self._first_feasible_iteration(history_df),
            "cache_hit_ratio": self._cache_hit_ratio(history_df),
            "runtime_engine_sec": self._runtime_value(history_df, "t_engine_cum_sec"),
            "runtime_sim_sec": self._runtime_value(history_df, "t_simulator_cum_sec"),
            "json": str(self.method_output_dir(method) / f"{run_id}.json"),
            "csv": str(self.method_output_dir(method) / f"{run_id}.csv"),
            "checkpoint": str(self.checkpoint_for_method_seed(method, seed)),
            "initial_design_manifest": str(self.initial_design_builder.manifest_path(seed)),
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

    def _cache_hit_ratio(self, history_df: pd.DataFrame) -> float:
        if history_df.empty or "cache_hit" not in history_df:
            return float("nan")
        return float(history_df["cache_hit"].fillna(False).astype(bool).mean())

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

    def _ensure_output_dirs(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.initial_design_dir.mkdir(parents=True, exist_ok=True)
        self.config.resolved_bnn_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for method in self.config.methods:
            self.method_output_dir(method).mkdir(parents=True, exist_ok=True)

    def _log_header(self) -> None:
        self._log(f"[LHS15SharedOnline] started: {datetime.now(timezone.utc).isoformat()}")
        self._log(f"repo_root: {self.config.repo_root}")
        self._log(f"config: {self.config.config_path} exists={self.config.config_path.exists()}")
        self._log(f"output: {self.config.output_dir}")
        self._log(f"methods: {self.config.methods}")
        self._log(f"seeds: {self.config.seeds}")
        self._log(f"n_init: {self.config.n_init}")
        self._log(f"budget: {self.config.budget}")
        self._log(f"target_trials: {self.config.target_trials}")
        self._log(f"cache_dir: {self.config.cache_dir}")
        self._log(f"initial_design_dir: {self.config.initial_design_dir}")
        self._log(f"bnn_checkpoint_dir: {self.config.resolved_bnn_checkpoint_dir}")
        self._log(f"backend: {self.config.backend}")

    def _log_planned_jobs(self, initial_designs: dict[int, SharedInitialDesign]) -> None:
        rows = []
        for method in self.config.methods:
            for seed in self.config.seeds:
                seed = int(seed)
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "run_id": self.run_id_for_method_seed(method, seed),
                        "checkpoint": str(self.checkpoint_for_method_seed(method, seed)),
                        "initial_design": str(initial_designs[seed].path),
                        "initial_points": len(initial_designs[seed].selected_rows),
                    }
                )
        self._log(pd.DataFrame(rows).to_string(index=False))

    def _log(self, message: Any) -> None:
        print(str(message), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BO/LHS/Turbo/BNN online BCLF experiments with shared LHS15 cached initial designs.",
    )
    parser.add_argument("--backend", choices=["server", "replay"], default="server")
    parser.add_argument("--methods", nargs="+", default=list(METHOD_CHOICES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-init", type=int, default=15)
    parser.add_argument("--budget", type=int, default=65)
    parser.add_argument("--pool-size", type=int, default=1500)
    parser.add_argument("--retrain-every", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/online_lhs15_shared_init"))
    parser.add_argument("--config-path", type=Path, default=Path("configs/filter_bnn_online_uaw_uaw.yaml"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/bandpass_coupled_line_filter_export/ext"))
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
    return SharedOnlineExperimentConfig(
        repo_root=REPO_ROOT,
        backend=args.backend,
        methods=normalize_methods(args.methods),
        seeds=[int(seed) for seed in args.seeds],
        n_init=int(args.n_init),
        budget=int(args.budget),
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
    for name in ("n_init", "budget", "pool_size", "retrain_every", "initial_epochs", "update_epochs", "mc_samples", "debug_print_every"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if not args.seeds:
        parser.error("--seeds must contain at least one seed")
    try:
        config = config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    with tee_stdout_stderr(config.log_file):
        try:
            return SharedOnlineExperimentRunner(config).run()
        except KeyboardInterrupt:
            print("[LHS15SharedOnline] interrupted by KeyboardInterrupt", flush=True)
            return 130
        except Exception:
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
