from __future__ import annotations

import argparse
import sys
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
DEFAULT_COMPARISON_RELATIVE_DIR = Path("experiments/online_lhsdb_warmup/owafs_bnn/comparison")
DEFAULT_FIGURE_RELATIVE_DIR = Path("studies/online_comparison/figures/online_lhsdb_warmup")
DEFAULT_METHOD_ORDER = ("OWAFS", "BNN", "BO", "LHS")
METHOD_COLORS = {
    "OWAFS": "#c51b29",
    "BNN": "#1f77b4",
    "LHS": "#2ca02c",
    "BO": "#7f7f7f",
}


class OWAFSObjectivePlotter:
    def __init__(
        self,
        *,
        repo_root: Path = REPO_ROOT,
        data_dir: Path | None = None,
        output_dir: Path | None = None,
        method_order: Sequence[str] = DEFAULT_METHOD_ORDER,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.data_dir = Path(data_dir) if data_dir is not None else self.repo_root / DEFAULT_COMPARISON_RELATIVE_DIR
        self.output_dir = Path(output_dir) if output_dir is not None else self.repo_root / DEFAULT_FIGURE_RELATIVE_DIR
        self.method_order = tuple(method_order)
        self.curve_path = self.data_dir / "owafs_vs_bo_lhs_bnn_best_so_far_by_iteration.csv"
        self.per_run_path = self.data_dir / "owafs_vs_bo_lhs_bnn_per_run.csv"

    def run(self) -> list[Path]:
        curves = self._read_curves()
        per_run = self._read_per_run()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        outputs = [
            *self._plot_best_so_far(curves),
            *self._plot_final_best(per_run),
        ]
        return outputs

    def _read_curves(self) -> pd.DataFrame:
        if not self.curve_path.exists():
            raise RuntimeError(f"Missing curve summary CSV: {self.curve_path}")
        curves = pd.read_csv(self.curve_path)
        required = {"method", "trial", "n", "median", "q25", "q75"}
        missing = required.difference(curves.columns)
        if missing:
            raise RuntimeError(f"Curve summary CSV is missing columns: {sorted(missing)}")
        return curves

    def _read_per_run(self) -> pd.DataFrame:
        if not self.per_run_path.exists():
            raise RuntimeError(f"Missing per-run comparison CSV: {self.per_run_path}")
        per_run = pd.read_csv(self.per_run_path)
        required = {"method", "seed", "final_best_objective", "budgeted_rows", "source_csv"}
        missing = required.difference(per_run.columns)
        if missing:
            raise RuntimeError(f"Per-run comparison CSV is missing columns: {sorted(missing)}")
        self._validate_bnn_sources(per_run)
        return per_run

    def _validate_bnn_sources(self, per_run: pd.DataFrame) -> None:
        bnn_sources = per_run.loc[per_run["method"].eq("BNN"), "source_csv"].astype(str)
        forbidden = bnn_sources.str.contains(
            "online_lhs15_shared|lhs15_shared|previous_lhs|previous-lhs",
            case=False,
            regex=True,
        )
        if bool(forbidden.any()):
            bad = bnn_sources[forbidden].iloc[0]
            raise RuntimeError(f"Forbidden previous-LHS BNN source in per-run CSV: {bad}")
        if not bnn_sources.str.contains("experiments/bnn_online_general", regex=False).all():
            raise RuntimeError("BNN sources must come from experiments/bnn_online_general")

    def _plot_best_so_far(self, curves: pd.DataFrame) -> list[Path]:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        with plt.rc_context(self._plot_style()):
            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            for method in self.method_order:
                rows = curves[curves["method"].eq(method)].sort_values("trial")
                if rows.empty:
                    continue
                x = rows["trial"].to_numpy(dtype=float)
                median = rows["median"].to_numpy(dtype=float)
                q25 = rows["q25"].to_numpy(dtype=float)
                q75 = rows["q75"].to_numpy(dtype=float)
                n = int(rows["n"].iloc[0])
                color = METHOD_COLORS.get(method)
                ax.plot(x, median, color=color, linewidth=2.0, label=f"{method} median (n={n})")
                ax.fill_between(x, q25, q75, color=color, alpha=0.14, linewidth=0)

            ax.axvline(15, color="0.45", linestyle="--", linewidth=1.0, label="trial 15")
            ax.set_xlabel("budgeted trial")
            ax.set_ylabel("best objective so far (lower is better)")
            ax.set_title("Objective trajectory, pooled complete runs")
            ax.grid(True, color="0.9", linewidth=0.6)
            ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
            fig.tight_layout()
            return self._save_figure(fig, "owafs_vs_bo_lhs_bnn_best_so_far_objective")

    def _plot_final_best(self, per_run: pd.DataFrame) -> list[Path]:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        with plt.rc_context(self._plot_style()):
            fig, ax = plt.subplots(figsize=(5.6, 3.5))
            positions = np.arange(len(self.method_order), dtype=float)
            rng = np.random.default_rng(0)
            for pos, method in zip(positions, self.method_order):
                rows = per_run[per_run["method"].eq(method)].sort_values("seed")
                if rows.empty:
                    continue
                y = rows["final_best_objective"].to_numpy(dtype=float)
                jitter = rng.uniform(-0.08, 0.08, size=len(rows))
                color = METHOD_COLORS.get(method)
                ax.scatter(
                    np.full(len(rows), pos) + jitter,
                    y,
                    s=42,
                    color=color,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                )
                median = float(np.median(y))
                ax.hlines(median, pos - 0.22, pos + 0.22, color="black", linewidth=1.8)
                ax.text(pos, median, f"{median:.2f}", ha="center", va="bottom", fontsize=7, color="black")

            ax.set_xticks(positions)
            ax.set_xticklabels(self.method_order)
            ax.set_ylabel("final best objective (lower is better)")
            ax.set_title("Final objective across complete runs")
            ax.grid(True, axis="y", color="0.9", linewidth=0.6)
            fig.tight_layout()
            return self._save_figure(fig, "owafs_vs_bo_lhs_bnn_final_best_objective")

    def _save_figure(self, fig, stem: str) -> list[Path]:
        import matplotlib.pyplot as plt

        outputs = []
        for suffix in ("pdf", "png"):
            path = self.output_dir / f"{stem}.{suffix}"
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
    parser = argparse.ArgumentParser(description="Plot objective comparison for OWAFS vs BO/LHS/BNN.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plotter = OWAFSObjectivePlotter(repo_root=args.repo_root, data_dir=args.data_dir, output_dir=args.output_dir)
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
