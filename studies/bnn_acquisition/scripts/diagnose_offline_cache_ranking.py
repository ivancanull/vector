#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def find_repo_root(start: Path) -> Path:
    for path in [start.resolve(), *start.resolve().parents]:
        if (path / ".git").exists():
            return path
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from complex_func import FunctionFromTouchstone  # noqa: E402
from modeling import BCLFResponseDataset  # noqa: E402
from optimization import BCLFOptimizationSpec, BNNExpectedImprovementBackend, GPqLogEIAcquisitionBackend  # noqa: E402
from optimization.datasets import BCLFDataset  # noqa: E402


@dataclass(frozen=True)
class CacheData:
    dataset: BCLFResponseDataset
    true_objective: np.ndarray
    skipped_bad_files: int = 0
    skipped_frequency_mismatch: int = 0


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def geometry_key(row: np.ndarray, *, decimals: int = 8) -> tuple[float, ...]:
    return tuple(float(v) for v in np.round(np.asarray(row, dtype=float).reshape(-1), decimals))


def load_variable_bounds(config_path: Path, variable_order: list[str]) -> dict[str, tuple[float, float]]:
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    reg_min = list(cfg["data"]["regularization"]["min"])
    reg_max = list(cfg["data"]["regularization"]["max"])
    if len(reg_min) == len(variable_order) + 1:
        reg_min = reg_min[: len(variable_order)]
        reg_max = reg_max[: len(variable_order)]
    if len(reg_min) != len(variable_order):
        raise ValueError(f"Regularization dimension mismatch for variables={variable_order}")
    return {name: (float(reg_min[i]), float(reg_max[i])) for i, name in enumerate(variable_order)}


def normalize(X: np.ndarray, bounds: dict[str, tuple[float, float]], variable_order: list[str]) -> np.ndarray:
    lo = np.asarray([bounds[name][0] for name in variable_order], dtype=float)
    hi = np.asarray([bounds[name][1] for name in variable_order], dtype=float)
    span = hi - lo
    if np.any(span <= 0):
        raise ValueError("Variable bounds must have positive span")
    return (np.asarray(X, dtype=float) - lo.reshape(1, -1)) / span.reshape(1, -1)


def filter_bounds_mask(X: np.ndarray, bounds: dict[str, tuple[float, float]], variable_order: list[str]) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    mask = np.ones(arr.shape[0], dtype=bool)
    for idx, name in enumerate(variable_order):
        low, high = bounds[name]
        mask &= arr[:, idx] >= float(low) - 1e-9
        mask &= arr[:, idx] <= float(high) + 1e-9
    return mask


def subset_dataset(dataset: BCLFResponseDataset, mask: np.ndarray) -> BCLFResponseDataset:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    return BCLFResponseDataset(
        variable_order=list(dataset.variable_order),
        freqs=dataset.freqs,
        X=dataset.X[idx],
        s_params=dataset.s_params[idx],
        filenames=[dataset.filenames[i] for i in idx],
        keys=[dataset.keys[i] for i in idx],
        case_ids=[dataset.case_ids[i] for i in idx],
    )


def make_spec() -> BCLFOptimizationSpec:
    return BCLFOptimizationSpec(
        passband_ghz=(4.8, 5.4),
        lower_stopband_ghz=(3.6, 4.4),
        upper_stopband_ghz=(5.8, 8.7),
        min_passband_s21_db=-5.0,
        max_passband_s11_db=-7.0,
        max_stopband_s21_db=-15.0,
        s11_weight=0.5,
        stopband_weight=0.5,
    )


def _freq_scale(unit: str) -> float:
    unit_l = unit.strip().lower()
    if unit_l == "khz":
        return 1e3
    if unit_l == "mhz":
        return 1e6
    if unit_l == "ghz":
        return 1e9
    return 1.0


def _pairs_to_complex(values: list[float], fmt: str) -> list[complex]:
    out: list[complex] = []
    fmt_l = fmt.lower()
    for idx in range(0, len(values), 2):
        a = float(values[idx])
        b = float(values[idx + 1])
        if fmt_l == "ma":
            out.append(a * np.exp(1j * np.deg2rad(b)))
        elif fmt_l == "db":
            out.append((10.0 ** (a / 20.0)) * np.exp(1j * np.deg2rad(b)))
        elif fmt_l == "ri":
            out.append(a + 1j * b)
        else:
            raise ValueError(f"Unsupported Touchstone format: {fmt}")
    return out


def _fast_parse_s2p(path: Path) -> tuple[np.ndarray, np.ndarray]:
    freq_unit = "hz"
    data_format = "db"
    freqs: list[float] = []
    values: list[list[complex]] = []
    current_freq: float | None = None
    current_values: list[float] = []
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            tokens = line[1:].lower().split()
            for token in tokens:
                if token in {"hz", "khz", "mhz", "ghz"}:
                    freq_unit = token
                elif token in {"db", "ma", "ri"}:
                    data_format = token
            continue
        numbers = [float(token) for token in line.split()]
        if current_freq is None:
            current_freq = numbers[0] * _freq_scale(freq_unit)
            current_values = numbers[1:]
        else:
            current_values.extend(numbers)
        if len(current_values) >= 8:
            freqs.append(float(current_freq))
            values.append(_pairs_to_complex(current_values[:8], data_format))
            current_freq = None
            current_values = []
    if not freqs:
        raise ValueError(f"No frequency points in {path}")
    arr = np.asarray(values, dtype=complex)
    s_params = np.zeros((2, 2, len(freqs)), dtype=complex)
    s_params[0, 0, :] = arr[:, 0]
    s_params[0, 1, :] = arr[:, 1]
    s_params[1, 0, :] = arr[:, 2]
    s_params[1, 1, :] = arr[:, 3]
    return np.asarray(freqs, dtype=float), s_params


def parse_touchstone(path: Path, parser: FunctionFromTouchstone) -> tuple[np.ndarray, np.ndarray]:
    try:
        return _fast_parse_s2p(path)
    except Exception:
        parser.parse(str(path))
        freqs = np.asarray(parser.freq, dtype=float).reshape(-1)
        if freqs.size == 0:
            raise ValueError(f"No frequency points in {path}")
        return freqs, np.asarray(parser.evaluate_all(), dtype=complex)


def same_frequency_grid(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    return lhs.shape == rhs.shape and bool(np.allclose(lhs, rhs, rtol=1e-9, atol=1e-3))


def load_index_records(cache_dir: Path) -> list[dict[str, Any]]:
    index_path = cache_dir / "index.json"
    payload = json.loads(index_path.read_text())
    records = []
    for key, record in dict(payload.get("records", {})).items():
        if not isinstance(record, dict):
            continue
        variables = record.get("variables", {})
        filename = record.get("filename")
        if not isinstance(variables, dict) or not filename:
            continue
        records.append({"key": key, "variables": variables, "filename": str(filename), "case_id": record.get("case_id")})
    return records


def load_cache(
    cache_dir: Path,
    bounds: dict[str, tuple[float, float]],
    variable_order: list[str],
    *,
    reference_geometries: np.ndarray,
) -> CacheData:
    records = load_index_records(cache_dir)
    record_by_geometry = {}
    for record in records:
        try:
            row = np.asarray([float(record["variables"][name]) for name in variable_order], dtype=float)
        except KeyError:
            continue
        record_by_geometry[geometry_key(row)] = record
    parser = FunctionFromTouchstone(port_num=2)
    reference_freqs: np.ndarray | None = None
    for row in np.asarray(reference_geometries, dtype=float):
        record = record_by_geometry.get(geometry_key(row))
        if record is None:
            raise FileNotFoundError(f"Checkpoint training geometry missing from cache index: {row.tolist()}")
        freqs, _s_params = parse_touchstone(cache_dir / record["filename"], parser)
        if reference_freqs is None:
            reference_freqs = freqs
        elif not same_frequency_grid(reference_freqs, freqs):
            raise ValueError("Checkpoint training geometries span multiple frequency grids; cannot train one response BNN")
    if reference_freqs is None:
        raise ValueError("reference_geometries must be non-empty")

    X_rows: list[list[float]] = []
    s_rows: list[np.ndarray] = []
    filenames: list[str] = []
    keys: list[str] = []
    case_ids: list[str] = []
    skipped_bad = 0
    skipped_grid = 0
    for record in records:
        try:
            geometry = [float(record["variables"][name]) for name in variable_order]
        except KeyError:
            skipped_bad += 1
            continue
        arr = np.asarray(geometry, dtype=float).reshape(1, -1)
        if not filter_bounds_mask(arr, bounds, variable_order)[0]:
            continue
        path = cache_dir / record["filename"]
        try:
            freqs, s_params = parse_touchstone(path, parser)
        except Exception:
            skipped_bad += 1
            continue
        if not same_frequency_grid(reference_freqs, freqs):
            skipped_grid += 1
            continue
        X_rows.append(geometry)
        s_rows.append(s_params)
        filenames.append(str(record["filename"]))
        keys.append(str(record["key"]))
        case_ids.append(str(record.get("case_id") or record["key"]))

    if not X_rows:
        raise ValueError("No cache records remain after bounds and frequency-grid filtering")
    dataset = BCLFResponseDataset(
        variable_order=list(variable_order),
        freqs=reference_freqs,
        X=np.asarray(X_rows, dtype=float),
        s_params=np.asarray(s_rows, dtype=complex),
        filenames=filenames,
        keys=keys,
        case_ids=case_ids,
    )
    spec = make_spec()
    objectives = [
        spec.evaluate(dataset.freqs, dataset.s_params[idx]).scalar_objective()
        for idx in range(dataset.n_samples)
    ]
    return CacheData(
        dataset=dataset,
        true_objective=np.asarray(objectives, dtype=float),
        skipped_bad_files=int(skipped_bad),
        skipped_frequency_mismatch=int(skipped_grid),
    )


def finite_history_from_checkpoint(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = []
    for row in payload.get("history", []):
        objective = row.get("objective")
        geometry = row.get("geometry")
        if geometry is None or objective is None:
            continue
        value = float(objective)
        if not math.isfinite(value):
            continue
        rows.append(dict(row, objective=value, geometry=[float(v) for v in geometry]))
    if not rows:
        raise ValueError(f"No finite objective history rows found in {path}")
    return rows


def cache_index(dataset: BCLFResponseDataset) -> dict[tuple[float, ...], int]:
    return {geometry_key(row): idx for idx, row in enumerate(dataset.X)}


def rehydrate_history(history: list[dict[str, Any]], cache: CacheData) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    index = cache_index(cache.dataset)
    rows: list[dict[str, Any]] = []
    X: list[np.ndarray] = []
    y: list[float] = []
    missing: list[list[float]] = []
    for row in history:
        geom = np.asarray(row["geometry"], dtype=float).reshape(-1)
        idx = index.get(geometry_key(geom))
        if idx is None:
            missing.append(geom.tolist())
            continue
        hydrated = dict(row)
        hydrated["_response_freqs"] = np.asarray(cache.dataset.freqs, dtype=float)
        hydrated["_response_s_params"] = np.asarray(cache.dataset.s_params[idx], dtype=complex)
        hydrated["case_id"] = cache.dataset.case_ids[idx]
        rows.append(hydrated)
        X.append(geom)
        y.append(float(row["objective"]))
    if missing:
        raise FileNotFoundError(f"{len(missing)} checkpoint geometries are missing from cache; first={missing[0]}")
    return rows, np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1, 1)


def rankdata(values: np.ndarray, *, descending: bool = False) -> np.ndarray:
    arr = -np.asarray(values, dtype=float) if descending else np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    sorted_values = arr[order]
    start = 0
    while start < len(arr):
        end = start + 1
        while end < len(arr) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 1.0 + 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def spearman_from_ranks(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    corr = np.corrcoef(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0, 1]
    return float(corr)


def score_bnn_in_batches(
    backend: BNNExpectedImprovementBackend,
    X_actual: np.ndarray,
    X_normalized: np.ndarray,
    history: list[dict[str, Any]],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for start in range(0, len(X_actual), int(batch_size)):
        stop = min(start + int(batch_size), len(X_actual))
        result = backend.score_candidates(
            candidates_actual=X_actual[start:stop],
            candidates_normalized=X_normalized[start:stop],
            history=history,
            tr_debug={"candidate_source": "offline_cache_ranking"},
        )
        means.append(result.predicted_mean)
        uncertainties.append(result.predicted_uncertainty)
        scores.append(result.acquisition_score)
    return np.concatenate(means), np.concatenate(uncertainties), np.concatenate(scores)


def summarize_rankings(df: pd.DataFrame) -> dict[str, Any]:
    true_rank = df["true_rank"].to_numpy(dtype=float)
    true_best_objective = float(df["true_objective"].min())
    true_best_idx = int(df["true_objective"].idxmin())
    methods = {
        "bnn_predicted": ("bnn_predicted_rank", False),
        "bnn_ei": ("bnn_ei_rank", True),
        "gp_predicted": ("gp_predicted_rank", False),
        "gp_ei": ("gp_ei_rank", True),
    }
    summary: dict[str, Any] = {
        "num_candidates": int(len(df)),
        "true_best_objective": true_best_objective,
        "true_best_geometry": {
            name: float(df.loc[true_best_idx, name])
            for name in ["L", "L1", "L2", "W", "W1", "W2", "S1", "S2"]
            if name in df.columns
        },
        "methods": {},
    }
    true_top = {k: set(df.nsmallest(k, "true_objective").index.tolist()) for k in (1, 5, 10, 20)}
    for name, (rank_col, _higher_is_better) in methods.items():
        ranks = df[rank_col].to_numpy(dtype=float)
        top1_idx = int(df[rank_col].idxmin())
        method = {
            "spearman_rank_correlation": spearman_from_ranks(true_rank, ranks),
            "true_best_rank": float(df.loc[true_best_idx, rank_col]),
            "top1_true_objective": float(df.loc[top1_idx, "true_objective"]),
            "top1_regret": float(df.loc[top1_idx, "true_objective"] - true_best_objective),
            "top1_geometry": {
                key: float(df.loc[top1_idx, key])
                for key in ["L", "L1", "L2", "W", "W1", "W2", "S1", "S2"]
                if key in df.columns
            },
            "topk_overlap": {},
        }
        for k, true_indices in true_top.items():
            method_indices = set(df.nsmallest(k, rank_col).index.tolist())
            method["topk_overlap"][str(k)] = {
                "count": int(len(true_indices & method_indices)),
                "fraction": float(len(true_indices & method_indices) / k),
            }
        summary["methods"][name] = method
    return summary


def write_report(summary: dict[str, Any], report_path: Path, ranking_path: Path, figure_path: Path | None) -> None:
    lines = [
        "# Offline Cache Ranking Diagnostic",
        "",
        f"- Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Candidates: `{summary['num_candidates']}`",
        f"- True best objective: `{summary['true_best_objective']:.6g}`",
        f"- Ranking CSV: `{ranking_path}`",
    ]
    if figure_path is not None:
        lines.append(f"- Figure: `{figure_path}`")
    lines += [
        "",
        "| Method | Spearman | True-best rank | Top-1 true objective | Top-1 regret | Top-10 overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in summary["methods"].items():
        lines.append(
            "| "
            f"{name} | "
            f"{metrics['spearman_rank_correlation']:.3f} | "
            f"{metrics['true_best_rank']:.0f} | "
            f"{metrics['top1_true_objective']:.6g} | "
            f"{metrics['top1_regret']:.6g} | "
            f"{metrics['topk_overlap']['10']['count']}/10 |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(summary: dict[str, Any], figure_path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    methods = list(summary["methods"].keys())
    regrets = [summary["methods"][name]["top1_regret"] for name in methods]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(methods, regrets, color=["#4C78A8", "#72B7B2", "#F58518", "#54A24B"])
    ax.set_ylabel("Top-1 true-objective regret")
    ax.set_title("Offline cache ranking diagnostic")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history-checkpoint",
        type=Path,
        default=Path("experiments/bnn_online_resampled_ninit10_seed0/bnn_online_resampled_ninit10_seed0.checkpoint.json"),
    )
    parser.add_argument("--config-path", type=Path, default=Path("configs/filter_bnn_online_uaw_uaw.yaml"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/bandpass_coupled_line_filter_export/ext"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/bnn_acquisition_offline_cache_ranking/bnn_resampled_ninit10_seed0"),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("studies/bnn_acquisition/reports"))
    parser.add_argument("--figure-dir", type=Path, default=Path("studies/bnn_acquisition/figures"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bnn-epochs", type=int, default=800)
    parser.add_argument("--bnn-update-epochs", type=int, default=20)
    parser.add_argument("--mc-samples", type=int, default=64)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force-retrain-bnn", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_path = resolve_path(args.history_checkpoint)
    cache_dir = resolve_path(args.cache_dir)
    config_path = resolve_path(args.config_path)
    output_dir = resolve_path(args.output_dir)
    report_dir = resolve_path(args.report_dir)
    figure_dir = resolve_path(args.figure_dir)

    variable_order = list(BCLFDataset.DEFAULT_FILTER_VARIABLE_ORDER)
    bounds = load_variable_bounds(config_path, variable_order)
    history = finite_history_from_checkpoint(checkpoint_path)
    history_geometries = np.asarray([row["geometry"] for row in history], dtype=float)
    cache = load_cache(cache_dir, bounds, variable_order, reference_geometries=history_geometries)
    hydrated_history, X_train_actual, y_train = rehydrate_history(history, cache)
    train_keys = {geometry_key(row) for row in X_train_actual}
    heldout_mask = np.asarray([geometry_key(row) not in train_keys for row in cache.dataset.X], dtype=bool)
    if not np.any(heldout_mask):
        raise ValueError("No held-out cache candidates remain after excluding checkpoint training geometries")

    heldout = subset_dataset(cache.dataset, heldout_mask)
    true_objective = cache.true_objective[heldout_mask]
    print(f"checkpoint: {checkpoint_path}")
    print(f"train_rows: {len(hydrated_history)}")
    print(f"cache_candidates_in_bounds: {cache.dataset.n_samples}")
    print(f"skipped_bad_files: {cache.skipped_bad_files}")
    print(f"skipped_frequency_mismatch: {cache.skipped_frequency_mismatch}")
    print(f"heldout_candidates: {heldout.n_samples}")
    if args.dry_run:
        print("dry_run: true; no BNN/GP model will be trained and no outputs will be written.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    X_train_normalized = normalize(X_train_actual, bounds, variable_order)
    X_heldout_normalized = normalize(heldout.X, bounds, variable_order)
    spec = make_spec()

    bnn_checkpoint = output_dir / "offline_ranking_bnn.pt"
    bnn = BNNExpectedImprovementBackend(
        {
            "hidden_dims": [256, 256, 256],
            "epochs": int(args.bnn_epochs),
            "update_epochs": int(args.bnn_update_epochs),
            "update_epoch_schedule": "constant",
            "initial_train_size": len(hydrated_history),
            "batch_size": 1024,
            "lr": 1e-3,
            "bayesian_loss_weight": 1e-4,
            "mc_samples": int(args.mc_samples),
            "acquisition": "qlogei",
            "checkpoint_path": bnn_checkpoint,
            "force_retrain": bool(args.force_retrain_bnn),
            "save_checkpoint": True,
        },
        device=str(args.device),
    )
    bnn_fit = bnn.fit(
        X_train_normalized,
        y_train,
        seed=int(args.seed),
        X_actual=X_train_actual,
        history=hydrated_history,
        optimization_spec=spec,
        variable_order=variable_order,
    )
    if not bnn_fit.ready:
        raise RuntimeError(f"BNN fit failed: {bnn_fit.error}")

    gp = GPqLogEIAcquisitionBackend()
    gp_fit = gp.fit(X_train_normalized, y_train, seed=int(args.seed))
    if not gp_fit.ready:
        raise RuntimeError(f"GP fit failed: {gp_fit.error}")

    bnn_mean, bnn_uncertainty, bnn_ei = score_bnn_in_batches(
        bnn,
        heldout.X,
        X_heldout_normalized,
        hydrated_history,
        batch_size=int(args.score_batch_size),
    )
    gp_scores = gp.score_candidates(
        candidates_actual=heldout.X,
        candidates_normalized=X_heldout_normalized,
        history=hydrated_history,
        tr_debug={"candidate_source": "offline_cache_ranking"},
    )

    ranking = pd.DataFrame({name: heldout.X[:, idx] for idx, name in enumerate(variable_order)})
    ranking["cache_filename"] = heldout.filenames
    ranking["true_objective"] = true_objective
    ranking["bnn_predicted_objective"] = bnn_mean
    ranking["bnn_predicted_uncertainty"] = bnn_uncertainty
    ranking["bnn_ei_score"] = bnn_ei
    ranking["gp_predicted_objective"] = gp_scores.predicted_mean
    ranking["gp_predicted_uncertainty"] = gp_scores.predicted_uncertainty
    ranking["gp_ei_score"] = gp_scores.acquisition_score
    ranking["true_rank"] = rankdata(ranking["true_objective"].to_numpy())
    ranking["bnn_predicted_rank"] = rankdata(ranking["bnn_predicted_objective"].to_numpy())
    ranking["bnn_ei_rank"] = rankdata(ranking["bnn_ei_score"].to_numpy(), descending=True)
    ranking["gp_predicted_rank"] = rankdata(ranking["gp_predicted_objective"].to_numpy())
    ranking["gp_ei_rank"] = rankdata(ranking["gp_ei_score"].to_numpy(), descending=True)
    ranking = ranking.sort_values("true_rank").reset_index(drop=True)

    summary = summarize_rankings(ranking)
    summary.update(
        {
            "history_checkpoint": str(checkpoint_path),
            "cache_dir": str(cache_dir),
            "cache_candidates_after_filter": int(cache.dataset.n_samples),
            "skipped_bad_files": int(cache.skipped_bad_files),
            "skipped_frequency_mismatch": int(cache.skipped_frequency_mismatch),
            "bnn_checkpoint": str(bnn_checkpoint),
            "bnn_fit_mode": bnn._last_fit_mode,
            "bnn_epochs": int(args.bnn_epochs),
            "bnn_update_epochs": int(args.bnn_update_epochs),
            "mc_samples": int(args.mc_samples),
            "score_batch_size": int(args.score_batch_size),
        }
    )

    ranking_path = output_dir / "ranking.csv"
    summary_path = output_dir / "summary.json"
    ranking.to_csv(ranking_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    figure_path = write_figure(summary, figure_dir / f"{output_dir.name}_top1_regret.png")
    report_path = report_dir / f"{output_dir.name}.md"
    write_report(summary, report_path, ranking_path, figure_path)
    print(f"ranking_csv: {ranking_path}")
    print(f"summary_json: {summary_path}")
    print(f"report_md: {report_path}")
    if figure_path is not None:
        print(f"figure_png: {figure_path}")
    print(pd.DataFrame(summary["methods"]).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
