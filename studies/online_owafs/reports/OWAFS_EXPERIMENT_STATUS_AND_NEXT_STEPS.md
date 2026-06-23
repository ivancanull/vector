# OWAFS 实验进度与后续实验计划

## 数据来源与口径

- 本报告用于内部决策；所有数值均从下方 CSV 自动读取。
- OWAFS campaign: `experiments/online_lhsdb_warmup/owafs_bnn/campaign_summary.csv`。
- 几何诊断: `experiments/online_lhsdb_warmup/owafs_bnn/diagnostics/owafs_geometry_coverage_summary_seed0_4.csv`。
- baseline summary: `experiments/online_lhsdb_warmup/owafs_bnn/comparison/owafs_vs_bo_lhs_bnn_summary.csv`。
- per-run comparison: `experiments/online_lhsdb_warmup/owafs_bnn/comparison/owafs_vs_bo_lhs_bnn_per_run.csv`。
- BNN baseline **只使用** `experiments/bnn_online_general`；明确不使用 previous-LHS / `online_lhs15_shared` 那批 BNN。
- 当前比较是描述性 pooled evidence：OWAFS seed0-4 为 5 seeds，BNN online_general seed0-2 为 3 seeds，BO seed0-4 / LHS seed0-4 各为 5 / 5 seeds；不是严格 paired statistical test。

## 当前实验状态

- OWAFS-BNN seed0-4 已完成；每个 seed 为 80 个 budgeted trials，validation row 数见表。
- 5/5 个 seed 改善了自己的 15 点 previous-LHS warmup；seed2 仍是最弱 run。

| Seed | Budgeted trials | Validation rows | Warmup best | OWAFS best | Best iter | Improvement | Median dist-to-LHS | Max dist-to-LHS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 80 | 1 | -16.43 | -22.08 | 44 | 5.64 | 0.69 | 1.04 |
| 1 | 80 | 1 | -11.47 | -17.41 | 32 | 5.95 | 0.66 | 0.97 |
| 2 | 80 | 1 | 0.02 | -6.80 | 41 | 6.82 | 0.67 | 1.03 |
| 3 | 80 | 1 | -2.06 | -23.82 | 18 | 21.76 | 0.73 | 1.09 |
| 4 | 80 | 2 | 6.64 | -16.80 | 26 | 23.44 | 0.71 | 1.11 |

## Baseline 对比

- 主要比较：OWAFS seed0-4 对 `bnn_online_general` seed0-2；BO/LHS online_general 作为 5-seed pooled reference。
- LHS 主要解释为覆盖型参考，不是同 warmup 的严格 baseline。

| Method | Runs | Seeds | Median final best | Mean final best | Best run | Worst run | Median gain 15->80 |
|---|---:|---|---:|---:|---:|---:|---:|
| BNN | 3 | 0,1,2 | -19.16 | -20.50 | -25.00 | -17.34 | 3.53 |
| LHS | 5 | 0,1,2,3,4 | -19.16 | -18.81 | -25.14 | -9.85 | 17.27 |
| OWAFS | 5 | 0,1,2,3,4 | -17.41 | -17.38 | -23.82 | -6.80 | 6.82 |
| BO | 5 | 0,1,2,3,4 | -14.66 | -15.65 | -23.81 | -8.05 | 5.95 |

关键观察：
- BNN online_general seed0-2 median final best 为 `-19.16`，mean 为 `-20.50`。
- OWAFS seed0-4 median final best 为 `-17.41`，mean 为 `-17.38`。
- BO online_general seed0-4 median 为 `-14.66`；LHS online_general seed0-4 median 为 `-19.16`。
- OWAFS median minus BO/BNN/LHS 分别为 `-2.76` / `1.75` / `1.75`；负数表示 OWAFS 更好。

## 当前比较方法说明

| 方法 | 当前用途 | 数据源 | 解释边界 |
|---|---|---|---|
| OWAFS-BNN | 当前研究方法；用 previous-LHS 15 点 warmup 后继续做 objective-weighted adaptive sampling。 | `experiments/online_lhsdb_warmup/owafs_bnn` seed0-4 | 可判断 OWAFS 是否改善自己的 warmup；当前还不能证明优于通用 BNN/LHS。 |
| BNN online_general | 主要神经 surrogate baseline；只使用完整 80-run 的 online_general BNN。 | `experiments/bnn_online_general` seed0-2 | 明确不使用 previous-LHS / `online_lhs15_shared` 那批 BNN；初始化口径仍不同于 OWAFS。 |
| BO online_general | 传统 Bayesian optimization 参考；用于看 OWAFS 是否至少接近/超过常规 BO。 | `experiments/bo_online_general` seed0-4 | 5 seeds 参考集，不是同 warmup paired baseline。 |
| LHS online_general | 覆盖型随机设计参考；用于判断纯几何覆盖本身能达到什么水平。 | `experiments/lhs_online_general` seed0-4 | 只作为覆盖型参考，不作为同 warmup 的严格 baseline。 |

## 几何诊断与研究判断

- OWAFS online 点到原 15 个 LHS warmup 点的 median distance 约为 `0.66-0.73`，说明它不是完全贴着 warmup 点采样。
- 但 nearest-distance 曲线没有随 iteration 持续抬升，best 点通常出现在中段 iteration，而不是后期强探索发现。
- 当前更合理的解释是：objective weighting 带来一定局部改良能力，但未知区域扩张不足，容易在部分 seed 上过早 exploitation。

## 当前 Claim 状态

| Claim | 当前状态 | 证据/限制 |
|---|---|---|
| OWAFS 可以改善 previous-LHS warmup 初始点 | 支持 | 5/5 seeds 从 warmup best 获得改善。 |
| OWAFS 比 BO 更好或接近 | 部分支持 | Median final best `-17.41` 优于 BO `-14.66`，但不是同 warmup paired baseline。 |
| OWAFS 优于 BNN 或 LHS | 不支持 | BNN/LHS median 分别为 `-19.16` / `-19.16`，优于 OWAFS `-17.41`。 |
| OWAFS 具备更强全局探索覆盖 | 不支持 | 几何距离没有持续扩张，best 点不是后期探索发现。 |

## 后续实验方法设计

| 方法 | 目标 | 具体做法 | 主要观察指标 |
|---|---|---|---|
| Same-warmup BNN baseline | 消除 BNN 与 OWAFS 初始化口径不一致的问题。 | 在 `online_lhsdb_warmup/bnn` 下跑 seed0-4，每个 seed 使用与 OWAFS 相同的 15 个 previous-LHS warmup 点，再完成 80 个 budgeted trials。 | final best、best-so-far 曲线、15->80 improvement、与 OWAFS 的同 seed 差值。 |
| Exploration ablation | 检查当前算法是否探索扩张不足。 | 比较 current、more-exploratory、coverage-regularized 三档 OWAFS 策略；提高未知区域或 coverage 项权重。 | final best、best-so-far 曲线、nearest-distance-to-LHS 曲线、best iteration 分布。 |
| Objective-weighting ablation | 判断 seed2 collapse 是否来自过早 exploitation。 | 关闭或弱化 objective-weighted sampling，只保留 uncertainty-aware / geometry-aware 分量，和 current OWAFS 对比。 | seed2 恢复程度、早期/中期/后期 improvement、候选点几何分散度。 |
| Warmup fairness ablation | 检查 OWAFS 对 previous-LHS warmup 初始质量的敏感性。 | 换不同 previous-LHS warmup draw，例如 offset 100/200，并对 OWAFS 与 same-warmup BNN 同步运行。 | warmup best 与 final best 的相关性、跨 warmup draw 的方差、失败 seed 比例。 |
| Validation consistency | 统一 reporting 口径，避免 validation row 混入 budgeted best 解释。 | 对 best candidate 做 full validation；报告中固定同时列出 budgeted best 与 validation best，并显式过滤 validation row 计算轨迹。 | budgeted/validation 差异、best candidate 稳定性、是否存在 validation 改写结论的 seed。 |

## 后续实验优先级

1. **P0: Same-warmup BNN baseline**：跑 `online_lhsdb_warmup/bnn` seed0-4，使用与 OWAFS 完全相同的 15 previous-LHS warmup。这是最关键的公平性缺口。
2. **P1: Exploration ablation**：在 OWAFS 上比较 current、more-exploratory、coverage-regularized 三档策略；指标包括 final best、best-so-far 曲线、nearest-distance 曲线。
3. **P1: Objective-weighting ablation**：关闭或弱化 objective weighting，只保留 uncertainty-aware / geometry-aware 分量，检查 seed2 collapse 是否来自过早 exploitation。
4. **P2: Warmup fairness ablation**：用不同 previous-LHS warmup draw，例如 offset 100/200，检查 OWAFS 对初始 warmup 质量的敏感性。
5. **P2: Validation consistency**：对 best candidate 做 full validation，并统一 reporting 使用 budgeted best 或 validation best，避免 validation row 改变解释口径。

## 已有关键图表

- Geometry PCA: `studies/online_owafs/figures/online_lhsdb_warmup/owafs_geometry_coverage_pca_seed0_4.pdf`。
- Distance-to-LHS: `studies/online_owafs/figures/online_lhsdb_warmup/owafs_geometry_distance_to_lhs_seed0_4.pdf`。
- Objective trajectory: `studies/online_comparison/figures/online_lhsdb_warmup/owafs_vs_bo_lhs_bnn_best_so_far_objective.pdf`。
- Final objective: `studies/online_comparison/figures/online_lhsdb_warmup/owafs_vs_bo_lhs_bnn_final_best_objective.pdf`。
