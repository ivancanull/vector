from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate
    raise RuntimeError(f"Cannot locate repo root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
DEFAULT_OWAFS_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_REPORT_RELATIVE_PATH = Path("studies/online_owafs/reports/OWAFS_EXPERIMENT_STATUS_AND_NEXT_STEPS.md")
DEFAULT_DIAGNOSTICS_RELATIVE_DIR = Path("experiments/online_lhsdb_warmup/owafs_bnn/diagnostics")
DEFAULT_COMPARISON_RELATIVE_DIR = Path("experiments/online_lhsdb_warmup/owafs_bnn/comparison")
DEFAULT_OWAFS_FIGURE_RELATIVE_DIR = Path("studies/online_owafs/figures/online_lhsdb_warmup")
DEFAULT_COMPARISON_FIGURE_RELATIVE_DIR = Path("studies/online_comparison/figures/online_lhsdb_warmup")
FORBIDDEN_BNN_PATH_TOKENS = ("online_lhs15_shared", "lhs15_shared", "previous_lhs", "previous-lhs")


@dataclass(frozen=True)
class ReportInputs:
    campaign_summary: Path
    geometry_summary: Path
    comparison_summary: Path
    per_run_comparison: Path


class OWAFSExperimentStatusReporter:
    def __init__(self, *, repo_root: Path = REPO_ROOT, output_path: Path | None = None) -> None:
        self.repo_root = Path(repo_root)
        self.output_path = (
            Path(output_path)
            if output_path is not None
            else self.repo_root / DEFAULT_REPORT_RELATIVE_PATH
        )
        diagnostics_dir = self.repo_root / DEFAULT_DIAGNOSTICS_RELATIVE_DIR
        comparison_dir = self.repo_root / DEFAULT_COMPARISON_RELATIVE_DIR
        self.inputs = ReportInputs(
            campaign_summary=self.repo_root
            / "experiments"
            / "online_lhsdb_warmup"
            / "owafs_bnn"
            / "campaign_summary.csv",
            geometry_summary=diagnostics_dir / f"owafs_geometry_coverage_summary_{self._seed_suffix(DEFAULT_OWAFS_SEEDS)}.csv",
            comparison_summary=comparison_dir / "owafs_vs_bo_lhs_bnn_summary.csv",
            per_run_comparison=comparison_dir / "owafs_vs_bo_lhs_bnn_per_run.csv",
        )

    def run(self) -> Path:
        data = self._load_data()
        report = self._render_report(data)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(report, encoding="utf-8")
        return self.output_path

    def _load_data(self) -> dict[str, pd.DataFrame]:
        campaign = self._read_csv(
            self.inputs.campaign_summary,
            required_columns={"seed", "status", "num_budgeted_trials", "num_validations", "best_objective"},
        )
        geometry = self._read_csv(
            self.inputs.geometry_summary,
            required_columns={
                "seed",
                "warmup_best_objective",
                "owafs_best_objective",
                "owafs_best_iteration",
                "online_median_distance_to_lhs",
                "online_max_distance_to_lhs",
            },
        )
        comparison = self._read_csv(
            self.inputs.comparison_summary,
            required_columns={
                "method",
                "n_runs",
                "seeds",
                "median_final_best_objective",
                "mean_final_best_objective",
                "best_min_objective",
                "worst_max_objective",
                "median_improvement_15_to_80",
            },
        )
        per_run = self._read_csv(
            self.inputs.per_run_comparison,
            required_columns={"method", "seed", "source_csv", "budgeted_rows", "final_best_objective"},
        )
        self._validate_bnn_sources(per_run)
        return {
            "campaign": campaign,
            "geometry": geometry,
            "comparison": comparison,
            "per_run": per_run,
        }

    def _read_csv(self, path: Path, *, required_columns: set[str]) -> pd.DataFrame:
        if not path.exists():
            raise RuntimeError(f"Missing required CSV: {path}")
        df = pd.read_csv(path)
        missing = required_columns.difference(df.columns)
        if missing:
            raise RuntimeError(f"{path} is missing required columns: {sorted(missing)}")
        return df

    def _validate_bnn_sources(self, per_run: pd.DataFrame) -> None:
        bnn_sources = per_run.loc[per_run["method"].eq("BNN"), "source_csv"].astype(str)
        if bnn_sources.empty:
            raise RuntimeError("No BNN rows found in per-run comparison CSV")
        forbidden = bnn_sources.str.contains("|".join(FORBIDDEN_BNN_PATH_TOKENS), case=False, regex=True)
        if bool(forbidden.any()):
            bad = bnn_sources[forbidden].iloc[0]
            raise RuntimeError(f"Forbidden previous-LHS BNN source in comparison CSV: {bad}")
        if not bnn_sources.str.contains("experiments/bnn_online_general", regex=False).all():
            raise RuntimeError("BNN baseline must use experiments/bnn_online_general sources only")

    def _render_report(self, data: dict[str, pd.DataFrame]) -> str:
        campaign = data["campaign"].sort_values("seed")
        geometry = data["geometry"].sort_values("seed")
        comparison = data["comparison"].copy()

        owafs_row = self._comparison_row(comparison, "OWAFS")
        bnn_row = self._comparison_row(comparison, "BNN")
        bo_row = self._comparison_row(comparison, "BO")
        lhs_row = self._comparison_row(comparison, "LHS")
        owafs_label = self._seed_label_from_csv(owafs_row["seeds"])
        bnn_label = self._seed_label_from_csv(bnn_row["seeds"])
        bo_label = self._seed_label_from_csv(bo_row["seeds"])
        lhs_label = self._seed_label_from_csv(lhs_row["seeds"])
        geometry_label = self._seed_label(geometry["seed"].astype(int).tolist())
        improvements = (
            pd.to_numeric(geometry["warmup_best_objective"], errors="coerce")
            - pd.to_numeric(geometry["owafs_best_objective"], errors="coerce")
        )
        improved_count = int((improvements > 0).sum())
        weakest_idx = pd.to_numeric(geometry["owafs_best_objective"], errors="coerce").idxmax()
        weakest_seed = int(geometry.loc[weakest_idx, "seed"])
        distance_median_min = float(pd.to_numeric(geometry["online_median_distance_to_lhs"], errors="coerce").min())
        distance_median_max = float(pd.to_numeric(geometry["online_median_distance_to_lhs"], errors="coerce").max())
        owafs_minus_bnn = float(owafs_row["median_final_best_objective"] - bnn_row["median_final_best_objective"])
        owafs_minus_lhs = float(owafs_row["median_final_best_objective"] - lhs_row["median_final_best_objective"])
        owafs_minus_bo = float(owafs_row["median_final_best_objective"] - bo_row["median_final_best_objective"])

        lines: list[str] = [
            "# OWAFS 实验进度与后续实验计划",
            "",
            "## 数据来源与口径",
            "",
            "- 本报告用于内部决策；所有数值均从下方 CSV 自动读取。",
            f"- OWAFS campaign: `{self._rel(self.inputs.campaign_summary)}`。",
            f"- 几何诊断: `{self._rel(self.inputs.geometry_summary)}`。",
            f"- baseline summary: `{self._rel(self.inputs.comparison_summary)}`。",
            f"- per-run comparison: `{self._rel(self.inputs.per_run_comparison)}`。",
            "- BNN baseline **只使用** `experiments/bnn_online_general`；明确不使用 previous-LHS / `online_lhs15_shared` 那批 BNN。",
            f"- 当前比较是描述性 pooled evidence：OWAFS {owafs_label} 为 {int(owafs_row['n_runs'])} seeds，"
            f"BNN online_general {bnn_label} 为 {int(bnn_row['n_runs'])} seeds，"
            f"BO {bo_label} / LHS {lhs_label} 各为 {int(bo_row['n_runs'])} / {int(lhs_row['n_runs'])} seeds；不是严格 paired statistical test。",
            "",
            "## 当前实验状态",
            "",
            f"- OWAFS-BNN {geometry_label} 已完成；每个 seed 为 80 个 budgeted trials，validation row 数见表。",
            f"- {improved_count}/{len(geometry)} 个 seed 改善了自己的 15 点 previous-LHS warmup；seed{weakest_seed} 仍是最弱 run。",
            "",
            "| Seed | Budgeted trials | Validation rows | Warmup best | OWAFS best | Best iter | Improvement | Median dist-to-LHS | Max dist-to-LHS |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, row in geometry.iterrows():
            seed = int(row["seed"])
            campaign_row = campaign[campaign["seed"].astype(int).eq(seed)].iloc[0]
            improvement = float(row["warmup_best_objective"] - row["owafs_best_objective"])
            lines.append(
                "| "
                f"{seed} | "
                f"{int(campaign_row['num_budgeted_trials'])} | "
                f"{int(campaign_row['num_validations'])} | "
                f"{self._fmt(row['warmup_best_objective'])} | "
                f"{self._fmt(row['owafs_best_objective'])} | "
                f"{int(row['owafs_best_iteration'])} | "
                f"{self._fmt(improvement)} | "
                f"{self._fmt(row['online_median_distance_to_lhs'])} | "
                f"{self._fmt(row['online_max_distance_to_lhs'])} |"
            )

        lines.extend(
            [
                "",
                "## Baseline 对比",
                "",
                f"- 主要比较：OWAFS {owafs_label} 对 `bnn_online_general` {bnn_label}；BO/LHS online_general 作为 5-seed pooled reference。",
                "- LHS 主要解释为覆盖型参考，不是同 warmup 的严格 baseline。",
                "",
                "| Method | Runs | Seeds | Median final best | Mean final best | Best run | Worst run | Median gain 15->80 |",
                "|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in [bnn_row, lhs_row, owafs_row, bo_row]:
            lines.append(
                "| "
                f"{row['method']} | "
                f"{int(row['n_runs'])} | "
                f"{row['seeds']} | "
                f"{self._fmt(row['median_final_best_objective'])} | "
                f"{self._fmt(row['mean_final_best_objective'])} | "
                f"{self._fmt(row['best_min_objective'])} | "
                f"{self._fmt(row['worst_max_objective'])} | "
                f"{self._fmt(row['median_improvement_15_to_80'])} |"
            )

        lines.extend(
            [
                "",
                "关键观察：",
                f"- BNN online_general {bnn_label} median final best 为 `{self._fmt(bnn_row['median_final_best_objective'])}`，mean 为 `{self._fmt(bnn_row['mean_final_best_objective'])}`。",
                f"- OWAFS {owafs_label} median final best 为 `{self._fmt(owafs_row['median_final_best_objective'])}`，mean 为 `{self._fmt(owafs_row['mean_final_best_objective'])}`。",
                f"- BO online_general {bo_label} median 为 `{self._fmt(bo_row['median_final_best_objective'])}`；LHS online_general {lhs_label} median 为 `{self._fmt(lhs_row['median_final_best_objective'])}`。",
                f"- OWAFS median minus BO/BNN/LHS 分别为 `{self._fmt(owafs_minus_bo)}` / `{self._fmt(owafs_minus_bnn)}` / `{self._fmt(owafs_minus_lhs)}`；负数表示 OWAFS 更好。",
                "",
                "## 当前比较方法说明",
                "",
                "| 方法 | 当前用途 | 数据源 | 解释边界 |",
                "|---|---|---|---|",
                f"| OWAFS-BNN | 当前研究方法；用 previous-LHS 15 点 warmup 后继续做 objective-weighted adaptive sampling。 | `experiments/online_lhsdb_warmup/owafs_bnn` {owafs_label} | 可判断 OWAFS 是否改善自己的 warmup；当前还不能证明优于通用 BNN/LHS。 |",
                f"| BNN online_general | 主要神经 surrogate baseline；只使用完整 80-run 的 online_general BNN。 | `experiments/bnn_online_general` {bnn_label} | 明确不使用 previous-LHS / `online_lhs15_shared` 那批 BNN；初始化口径仍不同于 OWAFS。 |",
                f"| BO online_general | 传统 Bayesian optimization 参考；用于看 OWAFS 是否至少接近/超过常规 BO。 | `experiments/bo_online_general` {bo_label} | 5 seeds 参考集，不是同 warmup paired baseline。 |",
                f"| LHS online_general | 覆盖型随机设计参考；用于判断纯几何覆盖本身能达到什么水平。 | `experiments/lhs_online_general` {lhs_label} | 只作为覆盖型参考，不作为同 warmup 的严格 baseline。 |",
                "",
                "## 几何诊断与研究判断",
                "",
                f"- OWAFS online 点到原 15 个 LHS warmup 点的 median distance 约为 `{self._fmt(distance_median_min)}-{self._fmt(distance_median_max)}`，说明它不是完全贴着 warmup 点采样。",
                "- 但 nearest-distance 曲线没有随 iteration 持续抬升，best 点通常出现在中段 iteration，而不是后期强探索发现。",
                "- 当前更合理的解释是：objective weighting 带来一定局部改良能力，但未知区域扩张不足，容易在部分 seed 上过早 exploitation。",
                "",
                "## 当前 Claim 状态",
                "",
                "| Claim | 当前状态 | 证据/限制 |",
                "|---|---|---|",
                f"| OWAFS 可以改善 previous-LHS warmup 初始点 | 支持 | {improved_count}/{len(geometry)} seeds 从 warmup best 获得改善。 |",
                f"| OWAFS 比 BO 更好或接近 | 部分支持 | Median final best `{self._fmt(owafs_row['median_final_best_objective'])}` 优于 BO `{self._fmt(bo_row['median_final_best_objective'])}`，但不是同 warmup paired baseline。 |",
                f"| OWAFS 优于 BNN 或 LHS | 不支持 | BNN/LHS median 分别为 `{self._fmt(bnn_row['median_final_best_objective'])}` / `{self._fmt(lhs_row['median_final_best_objective'])}`，优于 OWAFS `{self._fmt(owafs_row['median_final_best_objective'])}`。 |",
                "| OWAFS 具备更强全局探索覆盖 | 不支持 | 几何距离没有持续扩张，best 点不是后期探索发现。 |",
                "",
                "## 后续实验方法设计",
                "",
                "| 方法 | 目标 | 具体做法 | 主要观察指标 |",
                "|---|---|---|---|",
                "| Same-warmup BNN baseline | 消除 BNN 与 OWAFS 初始化口径不一致的问题。 | 在 `online_lhsdb_warmup/bnn` 下跑 seed0-4，每个 seed 使用与 OWAFS 相同的 15 个 previous-LHS warmup 点，再完成 80 个 budgeted trials。 | final best、best-so-far 曲线、15->80 improvement、与 OWAFS 的同 seed 差值。 |",
                "| Exploration ablation | 检查当前算法是否探索扩张不足。 | 比较 current、more-exploratory、coverage-regularized 三档 OWAFS 策略；提高未知区域或 coverage 项权重。 | final best、best-so-far 曲线、nearest-distance-to-LHS 曲线、best iteration 分布。 |",
                "| Objective-weighting ablation | 判断 seed2 collapse 是否来自过早 exploitation。 | 关闭或弱化 objective-weighted sampling，只保留 uncertainty-aware / geometry-aware 分量，和 current OWAFS 对比。 | seed2 恢复程度、早期/中期/后期 improvement、候选点几何分散度。 |",
                "| Warmup fairness ablation | 检查 OWAFS 对 previous-LHS warmup 初始质量的敏感性。 | 换不同 previous-LHS warmup draw，例如 offset 100/200，并对 OWAFS 与 same-warmup BNN 同步运行。 | warmup best 与 final best 的相关性、跨 warmup draw 的方差、失败 seed 比例。 |",
                "| Validation consistency | 统一 reporting 口径，避免 validation row 混入 budgeted best 解释。 | 对 best candidate 做 full validation；报告中固定同时列出 budgeted best 与 validation best，并显式过滤 validation row 计算轨迹。 | budgeted/validation 差异、best candidate 稳定性、是否存在 validation 改写结论的 seed。 |",
                "",
                "## 后续实验优先级",
                "",
                "1. **P0: Same-warmup BNN baseline**：跑 `online_lhsdb_warmup/bnn` seed0-4，使用与 OWAFS 完全相同的 15 previous-LHS warmup。这是最关键的公平性缺口。",
                "2. **P1: Exploration ablation**：在 OWAFS 上比较 current、more-exploratory、coverage-regularized 三档策略；指标包括 final best、best-so-far 曲线、nearest-distance 曲线。",
                "3. **P1: Objective-weighting ablation**：关闭或弱化 objective weighting，只保留 uncertainty-aware / geometry-aware 分量，检查 seed2 collapse 是否来自过早 exploitation。",
                "4. **P2: Warmup fairness ablation**：用不同 previous-LHS warmup draw，例如 offset 100/200，检查 OWAFS 对初始 warmup 质量的敏感性。",
                "5. **P2: Validation consistency**：对 best candidate 做 full validation，并统一 reporting 使用 budgeted best 或 validation best，避免 validation row 改变解释口径。",
                "",
                "## 已有关键图表",
                "",
                f"- Geometry PCA: `{self._rel(self.repo_root / DEFAULT_OWAFS_FIGURE_RELATIVE_DIR / f'owafs_geometry_coverage_pca_{self._seed_suffix(DEFAULT_OWAFS_SEEDS)}.pdf')}`。",
                f"- Distance-to-LHS: `{self._rel(self.repo_root / DEFAULT_OWAFS_FIGURE_RELATIVE_DIR / f'owafs_geometry_distance_to_lhs_{self._seed_suffix(DEFAULT_OWAFS_SEEDS)}.pdf')}`。",
                f"- Objective trajectory: `{self._rel(self.repo_root / DEFAULT_COMPARISON_FIGURE_RELATIVE_DIR / 'owafs_vs_bo_lhs_bnn_best_so_far_objective.pdf')}`。",
                f"- Final objective: `{self._rel(self.repo_root / DEFAULT_COMPARISON_FIGURE_RELATIVE_DIR / 'owafs_vs_bo_lhs_bnn_final_best_objective.pdf')}`。",
                "",
            ]
        )
        return "\n".join(lines)

    def _comparison_row(self, comparison: pd.DataFrame, method: str) -> pd.Series:
        rows = comparison[comparison["method"].eq(method)]
        if rows.empty:
            raise RuntimeError(f"Missing {method} row in comparison summary")
        return rows.iloc[0]

    def _seed_label_from_csv(self, value: object) -> str:
        seeds = [int(seed.strip()) for seed in str(value).split(",") if seed.strip()]
        return self._seed_label(seeds)

    def _seed_label(self, seeds: Sequence[int]) -> str:
        return self._seed_suffix(seeds).replace("_", "-")

    def _seed_suffix(self, seeds: Sequence[int]) -> str:
        sorted_seeds = tuple(sorted(int(seed) for seed in seeds))
        if not sorted_seeds:
            raise RuntimeError("At least one seed is required")
        if sorted_seeds == tuple(range(sorted_seeds[0], sorted_seeds[-1] + 1)):
            return f"seed{sorted_seeds[0]}_{sorted_seeds[-1]}"
        return "seeds_" + "_".join(str(seed) for seed in sorted_seeds)

    def _rel(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.repo_root.resolve()))
        except ValueError:
            return str(path)

    def _fmt(self, value: object) -> str:
        return f"{float(value):.2f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate OWAFS internal experiment status report.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reporter = OWAFSExperimentStatusReporter(repo_root=args.repo_root, output_path=args.output)
    try:
        output = reporter.run()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
