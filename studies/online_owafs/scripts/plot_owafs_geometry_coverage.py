from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate
    raise RuntimeError(f"Cannot locate repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_WARMUP_COUNT = 15
DEFAULT_EXPECTED_BUDGETED_TRIALS = 80
DEFAULT_FIGURE_RELATIVE_DIR = Path("studies/online_owafs/figures/online_lhsdb_warmup")
DEFAULT_SUMMARY_RELATIVE_DIR = Path("experiments/online_lhsdb_warmup/owafs_bnn/diagnostics")
VARIABLE_ORDER = ("L", "L1", "L2", "W", "W1", "W2", "S1", "S2")
GEOMETRY_MIN = np.asarray([7.7, 8.0, 7.8, 0.9, 0.7, 0.9, 0.13, 0.55], dtype=float)
GEOMETRY_MAX = np.asarray([7.9, 8.2, 8.0, 1.1, 0.9, 1.1, 0.16, 0.6], dtype=float)


@dataclass(frozen=True)
class PCAProjection:
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.mean.shape[0]:
            raise ValueError("PCA input matrix has incompatible shape")
        return (matrix - self.mean) @ self.components.T


@dataclass(frozen=True)
class SeedCoverageData:
    seed: int
    warmup_df: pd.DataFrame
    budgeted_df: pd.DataFrame
    all_df: pd.DataFrame
    warmup_normalized: np.ndarray
    budgeted_normalized: np.ndarray
    distance_to_lhs: np.ndarray
    best_objective: float
    best_iteration: int
    best_phase: str
    best_case_id: str
    best_normalized: np.ndarray

    @property
    def online_mask(self) -> np.ndarray:
        return self.budgeted_df["iteration"].to_numpy(dtype=int) >= DEFAULT_WARMUP_COUNT


def parse_vector(value: object, *, expected_length: int = 8, field_name: str = "vector") -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError(f"{field_name} is missing")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} is empty")
        parsed = ast.literal_eval(text)
    else:
        parsed = value
    array = np.asarray(parsed, dtype=float).reshape(-1)
    if array.shape != (expected_length,):
        raise ValueError(f"{field_name} must contain {expected_length} values; got {array.shape[0]}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} contains non-finite values")
    return array


def normalize_geometry(geometry: np.ndarray, geometry_min: np.ndarray, geometry_max: np.ndarray) -> np.ndarray:
    geometry = np.asarray(geometry, dtype=float).reshape(-1)
    geometry_min = np.asarray(geometry_min, dtype=float).reshape(-1)
    geometry_max = np.asarray(geometry_max, dtype=float).reshape(-1)
    span = geometry_max - geometry_min
    if geometry.shape != geometry_min.shape or geometry.shape != geometry_max.shape:
        raise ValueError("geometry and bounds dimensions do not match")
    if np.any(span <= 0):
        raise ValueError("geometry bounds must have positive span")
    normalized = (geometry - geometry_min) / span
    if not np.all(np.isfinite(normalized)):
        raise ValueError("normalized geometry contains non-finite values")
    return normalized


def fit_pca(values: np.ndarray) -> PCAProjection:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("PCA input must be a matrix")
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("PCA input needs at least two rows and two columns")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    if components.shape[0] < 2:
        pad = np.zeros((2 - components.shape[0], matrix.shape[1]), dtype=float)
        components = np.vstack([components, pad])
    variances = singular_values**2
    total_variance = float(variances.sum())
    ratios = np.zeros(2, dtype=float)
    if total_variance > 0:
        ratios[: min(2, variances.shape[0])] = variances[:2] / total_variance
    return PCAProjection(mean=mean, components=components, explained_variance_ratio=ratios)


def nearest_distances(points: np.ndarray, references: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    references = np.asarray(references, dtype=float)
    if points.ndim != 2 or references.ndim != 2 or points.shape[1] != references.shape[1]:
        raise ValueError("point and reference matrices have incompatible shapes")
    if references.shape[0] == 0:
        raise ValueError("at least one reference point is required")
    diffs = points[:, None, :] - references[None, :, :]
    return np.linalg.norm(diffs, axis=2).min(axis=1)


def parse_budget_mask(values: pd.Series) -> pd.Series:
    """Parse optional budget flags without treating string "False" as truthy."""
    if values.empty:
        return pd.Series(dtype=bool, index=values.index)

    def parse_value(value: object) -> bool:
        if pd.isna(value):
            return True
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        raise ValueError(f"Cannot parse counts_toward_budget value: {value!r}")

    return values.map(parse_value).astype(bool)


class OWAFSGeometryCoveragePlotter:
    def __init__(
        self,
        *,
        repo_root: Path = REPO_ROOT,
        seeds: Sequence[int] = DEFAULT_SEEDS,
        expected_budgeted_trials: int = DEFAULT_EXPECTED_BUDGETED_TRIALS,
        warmup_count: int = DEFAULT_WARMUP_COUNT,
        geometry_min: np.ndarray = GEOMETRY_MIN,
        geometry_max: np.ndarray = GEOMETRY_MAX,
        output_dir: Path | None = None,
        figure_dir: Path | None = None,
        summary_dir: Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.seeds = tuple(int(seed) for seed in seeds)
        self.expected_budgeted_trials = int(expected_budgeted_trials)
        self.warmup_count = int(warmup_count)
        self.geometry_min = np.asarray(geometry_min, dtype=float)
        self.geometry_max = np.asarray(geometry_max, dtype=float)
        self.initial_points_path = (
            self.repo_root / "experiments" / "online_lhsdb_warmup" / "initial_designs" / "warmup_initial_points.csv"
        )
        self.owafs_dir = self.repo_root / "experiments" / "online_lhsdb_warmup" / "owafs_bnn"
        # output_dir is a compatibility alias for callers that expect figures and
        # CSV diagnostics to be co-located.
        self.figure_dir = Path(figure_dir) if figure_dir is not None else (
            Path(output_dir) if output_dir is not None else self.repo_root / DEFAULT_FIGURE_RELATIVE_DIR
        )
        self.summary_dir = Path(summary_dir) if summary_dir is not None else (
            Path(output_dir) if output_dir is not None else self.repo_root / DEFAULT_SUMMARY_RELATIVE_DIR
        )

    def run(self) -> list[Path]:
        seed_data = self.load_data()
        pca = self._fit_shared_pca(seed_data)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        outputs = [
            *self._plot_pca_coverage(seed_data, pca),
            *self._plot_distance_curves(seed_data),
            self._write_summary(seed_data, pca),
        ]
        return outputs

    def _seed_suffix(self) -> str:
        if not self.seeds:
            raise RuntimeError("At least one seed is required")
        seeds = tuple(sorted(self.seeds))
        if seeds == tuple(range(seeds[0], seeds[-1] + 1)):
            return f"seed{seeds[0]}_{seeds[-1]}"
        return "seeds_" + "_".join(str(seed) for seed in seeds)

    def load_data(self) -> list[SeedCoverageData]:
        if not self.initial_points_path.exists():
            raise RuntimeError(f"Missing warmup initial points CSV: {self.initial_points_path}")
        warmup_df = pd.read_csv(self.initial_points_path)
        seed_data: list[SeedCoverageData] = []
        for seed in self.seeds:
            seed_data.append(self._load_seed(seed, warmup_df))
        return seed_data

    def _load_seed(self, seed: int, warmup_df: pd.DataFrame) -> SeedCoverageData:
        seed_csv = self.owafs_dir / f"owafs_bnn_online_lhsdb_warmup_seed{seed}.csv"
        if not seed_csv.exists():
            raise RuntimeError(
                f"Seed {seed} final CSV is not available yet: {seed_csv}. "
                "Wait for the campaign to finish before generating final plots."
            )

        seed_warmup = warmup_df[warmup_df["target_seed"].astype(int) == int(seed)].copy()
        if len(seed_warmup) != self.warmup_count:
            raise RuntimeError(
                f"Seed {seed} expected {self.warmup_count} warmup LHS points, found {len(seed_warmup)}"
            )
        seed_warmup = seed_warmup.sort_values("init_idx").reset_index(drop=True)
        warmup_normalized = np.vstack(
            [
                normalize_geometry(
                    parse_vector(value, expected_length=len(VARIABLE_ORDER), field_name=f"seed {seed} warmup geometry"),
                    self.geometry_min,
                    self.geometry_max,
                )
                for value in seed_warmup["geometry"].tolist()
            ]
        )

        all_rows = pd.read_csv(seed_csv)
        budgeted = self._budgeted_rows(all_rows).copy().reset_index(drop=True)
        if len(budgeted) != self.expected_budgeted_trials:
            raise RuntimeError(
                f"Seed {seed} is not complete: found {len(budgeted)}/{self.expected_budgeted_trials} "
                "budgeted rows. No final plots were written."
            )
        budgeted_normalized = np.vstack(
            [
                self._row_normalized_geometry(row, f"seed {seed} iteration {row.get('iteration')}")
                for _, row in budgeted.iterrows()
            ]
        )
        distances = nearest_distances(budgeted_normalized, warmup_normalized)
        best_row = self._best_row(all_rows, seed)
        best_normalized = self._row_normalized_geometry(best_row, f"seed {seed} best row")

        return SeedCoverageData(
            seed=seed,
            warmup_df=seed_warmup,
            budgeted_df=budgeted,
            all_df=all_rows,
            warmup_normalized=warmup_normalized,
            budgeted_normalized=budgeted_normalized,
            distance_to_lhs=distances,
            best_objective=float(best_row["objective"]),
            best_iteration=int(best_row["iteration"]),
            best_phase=str(best_row.get("phase", "")),
            best_case_id=str(best_row.get("case_id", "")),
            best_normalized=best_normalized,
        )

    def _budgeted_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        if "counts_toward_budget" in rows.columns:
            mask = parse_budget_mask(rows["counts_toward_budget"])
        else:
            phase = rows.get("phase", pd.Series("", index=rows.index)).fillna("").astype(str).str.lower()
            mask = phase != "validation"
        return rows[mask].copy()

    def _row_normalized_geometry(self, row: pd.Series, label: str) -> np.ndarray:
        value = row.get("geometry_normalized", "")
        if isinstance(value, str) and value.strip():
            return parse_vector(value, expected_length=len(VARIABLE_ORDER), field_name=f"{label} geometry_normalized")
        geometry = parse_vector(row.get("geometry"), expected_length=len(VARIABLE_ORDER), field_name=f"{label} geometry")
        return normalize_geometry(geometry, self.geometry_min, self.geometry_max)

    def _best_row(self, rows: pd.DataFrame, seed: int) -> pd.Series:
        objective = pd.to_numeric(rows["objective"], errors="coerce")
        finite = rows[np.isfinite(objective.to_numpy(dtype=float))].copy()
        if finite.empty:
            raise RuntimeError(f"Seed {seed} contains no finite objective values")
        objective = pd.to_numeric(finite["objective"], errors="coerce")
        return finite.loc[objective.idxmin()]

    def _fit_shared_pca(self, seed_data: Sequence[SeedCoverageData]) -> PCAProjection:
        matrix = np.vstack(
            [item.warmup_normalized for item in seed_data]
            + [item.budgeted_normalized for item in seed_data]
        )
        return fit_pca(matrix)

    def _plot_pca_coverage(self, seed_data: Sequence[SeedCoverageData], pca: PCAProjection) -> list[Path]:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        with plt.rc_context(self._plot_style()):
            fig, axes = plt.subplots(
                1,
                len(seed_data),
                figsize=(max(11.0, 2.4 * len(seed_data)), 3.4),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            if len(seed_data) == 1:
                axes = [axes]

            online_iterations = np.concatenate(
                [
                    item.budgeted_df["iteration"].to_numpy(dtype=int)[item.budgeted_df["iteration"].to_numpy(dtype=int) >= self.warmup_count]
                    for item in seed_data
                ]
            )
            norm = plt.Normalize(vmin=float(online_iterations.min()), vmax=float(online_iterations.max()))
            cmap = plt.get_cmap("viridis")
            scatter_for_colorbar = None

            for ax, item in zip(axes, seed_data):
                warmup_xy = pca.transform(item.warmup_normalized)
                budgeted_xy = pca.transform(item.budgeted_normalized)
                iterations = item.budgeted_df["iteration"].to_numpy(dtype=int)
                online_mask = iterations >= self.warmup_count
                online_xy = budgeted_xy[online_mask]
                online_iter = iterations[online_mask]
                best_xy = pca.transform(item.best_normalized.reshape(1, -1))[0]

                ax.scatter(
                    warmup_xy[:, 0],
                    warmup_xy[:, 1],
                    s=34,
                    facecolors="none",
                    edgecolors="0.65",
                    linewidths=1.0,
                    label="LHS warmup" if item.seed == seed_data[0].seed else None,
                )
                scatter_for_colorbar = ax.scatter(
                    online_xy[:, 0],
                    online_xy[:, 1],
                    s=22,
                    c=online_iter,
                    cmap=cmap,
                    norm=norm,
                    alpha=0.82,
                    linewidths=0,
                    label="OWAFS online" if item.seed == seed_data[0].seed else None,
                )
                ax.scatter(
                    [best_xy[0]],
                    [best_xy[1]],
                    marker="*",
                    s=170,
                    c="#c51b29",
                    edgecolors="black",
                    linewidths=0.5,
                    label="Best" if item.seed == seed_data[0].seed else None,
                    zorder=5,
                )
                ax.annotate(
                    f"best {item.best_objective:.2f}\nit {item.best_iteration}",
                    xy=(best_xy[0], best_xy[1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7,
                    color="#7f0000",
                )
                ax.set_title(f"seed {item.seed}")
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio[0] * 100:.1f}%)")
                ax.grid(True, color="0.9", linewidth=0.6)
            axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio[1] * 100:.1f}%)")
            axes[0].legend(frameon=False, loc="best")
            if scatter_for_colorbar is not None:
                colorbar = fig.colorbar(scatter_for_colorbar, ax=axes, fraction=0.025, pad=0.02)
                colorbar.set_label("budgeted iteration")
            fig.suptitle("OWAFS-BNN geometry coverage relative to original LHS warmup")
            return self._save_figure(fig, f"owafs_geometry_coverage_pca_{self._seed_suffix()}")

    def _plot_distance_curves(self, seed_data: Sequence[SeedCoverageData]) -> list[Path]:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        with plt.rc_context(self._plot_style()):
            fig, ax = plt.subplots(figsize=(7.2, 3.4))
            colors = plt.get_cmap("tab10").colors
            for idx, item in enumerate(seed_data):
                iterations = item.budgeted_df["iteration"].to_numpy(dtype=int)
                ax.plot(
                    iterations,
                    item.distance_to_lhs,
                    color=colors[idx % len(colors)],
                    linewidth=1.8,
                    label=f"seed {item.seed}",
                )
            ax.axvline(
                self.warmup_count - 0.5,
                color="0.35",
                linestyle="--",
                linewidth=1.0,
                label="warmup boundary",
            )
            ax.set_xlabel("budgeted iteration")
            ax.set_ylabel("nearest distance to original LHS warmup")
            ax.set_title("Geometry novelty relative to original LHS warmup")
            ax.grid(True, color="0.9", linewidth=0.6)
            ax.legend(frameon=False, ncol=2)
            fig.tight_layout()
            return self._save_figure(fig, f"owafs_geometry_distance_to_lhs_{self._seed_suffix()}")

    def _write_summary(self, seed_data: Sequence[SeedCoverageData], pca: PCAProjection) -> Path:
        rows = []
        for item in seed_data:
            warmup_objectives = pd.to_numeric(item.warmup_df["source_objective"], errors="coerce")
            warmup_best_idx = int(warmup_objectives.idxmin())
            online_mask = item.budgeted_df["iteration"].to_numpy(dtype=int) >= self.warmup_count
            online_distances = item.distance_to_lhs[online_mask]
            rows.append(
                {
                    "seed": item.seed,
                    "warmup_points": len(item.warmup_df),
                    "budgeted_trials": len(item.budgeted_df),
                    "validation_rows": int((item.all_df.get("phase", pd.Series("", index=item.all_df.index)) == "validation").sum()),
                    "warmup_best_objective": float(warmup_objectives.min()),
                    "warmup_best_init_idx": int(item.warmup_df.loc[warmup_best_idx, "init_idx"]),
                    "owafs_best_objective": item.best_objective,
                    "owafs_best_iteration": item.best_iteration,
                    "owafs_best_phase": item.best_phase,
                    "owafs_best_case_id": item.best_case_id,
                    "online_median_distance_to_lhs": float(np.median(online_distances)),
                    "online_max_distance_to_lhs": float(np.max(online_distances)),
                    "budgeted_median_distance_to_lhs": float(np.median(item.distance_to_lhs)),
                    "budgeted_max_distance_to_lhs": float(np.max(item.distance_to_lhs)),
                    "pca_explained_variance_pc1": float(pca.explained_variance_ratio[0]),
                    "pca_explained_variance_pc2": float(pca.explained_variance_ratio[1]),
                }
            )
        path = self.summary_dir / f"owafs_geometry_coverage_summary_{self._seed_suffix()}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _save_figure(self, fig, stem: str) -> list[Path]:
        import matplotlib.pyplot as plt

        outputs = []
        for suffix in ("pdf", "png"):
            path = self.figure_dir / f"{stem}.{suffix}"
            fig.savefig(path, format=suffix, dpi=300, bbox_inches="tight", pad_inches=0.05)
            outputs.append(path)
        plt.close(fig)
        return outputs

    def _plot_style(self) -> dict[str, object]:
        return {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot OWAFS-BNN geometry coverage diagnostics.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None, help="Compatibility alias for both figure and summary output directories.")
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--expected-budgeted-trials", type=int, default=DEFAULT_EXPECTED_BUDGETED_TRIALS)
    parser.add_argument("--warmup-count", type=int, default=DEFAULT_WARMUP_COUNT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plotter = OWAFSGeometryCoveragePlotter(
        repo_root=args.repo_root,
        seeds=args.seeds,
        expected_budgeted_trials=args.expected_budgeted_trials,
        warmup_count=args.warmup_count,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        summary_dir=args.summary_dir,
    )
    try:
        outputs = plotter.run()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
