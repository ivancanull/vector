# Project Handoff：BCLF 在线优化与 HFSS 协同仿真项目

本文是给接手同学的入口文档。它解释项目结构、当前 BCLF 实验状态、如何用 Codex 继续推进实验、如何操作 server/worker 协同仿真，以及如何扩展到新的器件结构和优化目标。

Snapshot：2026-06-10。当前仍有部分 BNN online 实验在 queue 中继续运行，本文中的实验进度是写作时的状态。

## 1. Quick Start For New Collaborator

先从这几个文件开始读：

- `docs/README.md`：文档总入口。
- `docs/engineering/optimization/optimization_experiment_backbone.md`：优化实验框架。
- `docs/engineering/server_worker/server_worker_submit_receive_flow.md`：server/worker 队列流程。
- `docs/academic/experiments/bclf_optimization_target.md`：BCLF 当前目标、变量和实验设置。
- `studies/README.md`：实验脚本、报告和图的归档结构。
- `studies/bnn_acquisition/reports/bnn_resampled_ninit10_seed0.md`：最近一次 BNN offline ranking 诊断。

常用只读检查命令：

```bash
git status --short
find queue/pending queue/running -maxdepth 1 -type f | sort | tail -40
tail -f logs/<latest-log>.log
tmux ls
```

原则：

- `experiments/` 放运行产物和数据，不放解释性报告。
- `studies/` 放实验脚本、报告、图。
- `docs/engineering/` 放工程运行和接口文档。
- `docs/academic/` 放研究计划、实验解释、paper 证据链。
- notebook 主要用于展示和分析；新的优化实验优先写成 Python runner。

## 2. Repository Map

主要目录：

- `src/optimization/`：优化主干，包括 `OptimizerBase`、`OptimizerBoTorch`、LHS/随机、候选空间、objective/spec、recorder、BNN/GP acquisition。
- `src/simulation/`：仿真适配层，把优化器和 replay/server/HFSS backend 隔开。
- `src/server/`：Linux server 侧 queue 管理、job dispatcher、result ingestor。
- `src/worker/`：Windows HFSS worker 侧 job claim、PyAEDT/HFSS 执行、result persistence/upload。
- `src/modeling/`：BCLF response dataset、BNN response surrogate、Touchstone response 处理。
- `configs/`：当前 BCLF 配置，尤其是 `configs/filter_bnn_online_uaw_uaw.yaml`。
- `data/`：BCLF Touchstone cache 和离线训练数据；体积大。
- `queue/`：server/worker 协同状态目录。
- `experiments/`：CSV/JSON/checkpoint/model/cache/manifest 等运行产物。
- `studies/`：按实验主题组织的 scripts/reports/figures。
- `docs/`：工程和学术文档入口。

当前实验相关 study：

- `studies/online_bo/`：BO online runner。
- `studies/online_bnn/`：BNN online runner 和 tmux wrapper。
- `studies/online_owafs/`：OWAFS/LHSDB warmup。
- `studies/online_comparison/`：跨方法对比图和表。
- `studies/bnn_acquisition/`：BNN/GP acquisition 和 offline ranking 诊断。
- `studies/bnn_surrogate/`：BNN response surrogate 离线/few-shot 实验。

## 3. Current BCLF Optimization Setup

当前器件是 bandpass coupled-line filter，设计变量顺序固定为：

```text
[L, L1, L2, W, W1, W2, S1, S2]
```

变量范围来自 `configs/filter_bnn_online_uaw_uaw.yaml` 的 `data.regularization.min/max` 前 8 项：

| Variable | Min | Max | Unit |
|---|---:|---:|---|
| `L` | 7.7 | 7.9 | mm |
| `L1` | 8.0 | 8.2 | mm |
| `L2` | 7.8 | 8.0 | mm |
| `W` | 0.9 | 1.1 | mm |
| `W1` | 0.7 | 0.9 | mm |
| `W2` | 0.9 | 1.1 | mm |
| `S1` | 0.13 | 0.16 | mm |
| `S2` | 0.55 | 0.60 | mm |

在线优化使用 `BCLFOptimizationSpec`，优化器最小化 scalar `objective`。当前目标设置：

```python
BCLFOptimizationSpec(
    passband_ghz=(4.8, 5.4),
    lower_stopband_ghz=(3.6, 4.4),
    upper_stopband_ghz=(5.8, 8.7),
    min_passband_s21_db=-5.0,
    max_passband_s11_db=-7.0,
    max_stopband_s21_db=-15.0,
    s11_weight=0.5,
    stopband_weight=0.5,
)
```

解释方式：

- 人的目标：passband `S21` 高，passband `S11` 低，stopband `S21` 低。
- 优化器约定：`objective` 越低越好。
- 对比收敛：看 `best_so_far` 曲线，不只看每次 raw objective。
- 在线提交几何会 snap 到 `0.01 mm`。

更完整的目标说明见 `docs/academic/experiments/bclf_optimization_target.md`。

## 4. Current Experiment Status And Findings

当前重要结果快照：

| Experiment | Status | Best objective | Best iteration | Notes |
|---|---:|---:|---:|---|
| BO continuous `n_init=10`, seed 0 | 80/80 done | `-16.759` | 29 | 使用 continuous `optimize_acqf` 后 snap。 |
| BNN resampled `n_init=10`, seed 0 | 67 trials | `-12.743` | 65 | 仍有 pending job；早期表现弱，后续有所改善。 |
| BNN soft qLogEI `n_init=30`, seed 0 | 39 trials | `-22.076` | 12 | 正在运行；当前 pending `it0039`。 |
| old BNN general seed 0 | 79 trials | `-24.999` | 55 | 旧配置结果，不能和新 rerun 完全等价。 |
| old five-seed comparison | done | LHS/Turbo median about `-19.16`, BO median about `-14.66` | - | 当前证据不支持 BO/BNN/OWAFS 明显优于 LHS/Turbo。 |

最近的 BNN offline cache ranking 诊断：

- 输入 checkpoint：`experiments/bnn_online_resampled_ninit10_seed0/bnn_online_resampled_ninit10_seed0.checkpoint.json`
- 训练点：67 个 finite trials。
- held-out cache candidates：1602 个。
- true best objective：`-25.141`。

诊断结果：

| Ranking | Spearman | True-best rank | Top-1 true objective | Top-10 overlap |
|---|---:|---:|---:|---:|
| BNN predicted | 0.331 | 206 | 23.948 | 0/10 |
| BNN EI | 0.322 | 161 | 23.948 | 0/10 |
| GP predicted | 0.306 | 313 | 5.286 | 0/10 |
| GP EI | 0.144 | 896 | -0.092 | 0/10 |

当前观察：

- BNN response surrogate 在 held-out cache 上存在明显 false positive：它把真实很差的点排到最前。
- 这说明风险主要在 response surrogate / uncertainty calibration，而不只是 qLogEI 数值坍缩。
- BNN soft qLogEI `n_init=30` 当前表现更好，但仍需要等完成后跑同样 offline ranking 诊断。
- 当前 objective 对 response surrogate 误差较敏感，因为它包含 worst/hinge/threshold 项。后续可以考虑给 acquisition 使用 smoother proxy objective，同时最终报告仍使用原始 RF objective。

相关报告：

- `studies/bnn_acquisition/reports/bnn_resampled_ninit10_seed0.md`
- `experiments/bnn_acquisition_offline_cache_ranking/bnn_resampled_ninit10_seed0/summary.json`
- `experiments/bnn_acquisition_offline_cache_ranking/bnn_resampled_ninit10_seed0/ranking.csv`

## 5. Running Experiments With Codex

推荐把 Codex 当作实验协作者，而不是只让它写代码。一个好的流程是：

1. 让 Codex 先读相关 docs/studies/checkpoints。
2. 让它给出 dry-run 或计划。
3. 确认输出目录、run id、queue prefix 不会覆盖旧实验。
4. 用 tmux/current-window wrapper 跑长实验。
5. 跑完后让 Codex 读 CSV/JSON/checkpoint，总结并写到对应 study 的 reports 目录或 `docs/academic/`。

常用 prompt 模板：

```text
检查 BNN soft qLogEI seed0 当前进度，只读 checkpoint 和 queue，不要改文件。
```

```text
对某个 experiments 目录的结果重新画 best_so_far 图，输出到对应 study 的 figures 目录，不要重跑实验。
```

```text
给这个新 online runner 做 dry-run，确认 run_id、target_trials、output_dir、queue case_id 前缀。
```

```text
把这个 notebook 改成 Python script，notebook 只保留展示和读取结果。
```

```text
新增一个实验 study，脚本放 studies/<name>/scripts，数据写 experiments/<name>，报告图写 studies/<name>/reports 和 figures。
```

长实验启动前先 dry-run。例如 BNN online：

```bash
python studies/online_bnn/scripts/run_bnn_online_experiment.py \
  --dry-run \
  --backend server \
  --seeds 0 \
  --n-init 30 \
  --budget 50 \
  --output-dir experiments/bnn_online_soft_qlogei_ninit30_seed0 \
  --run-id-prefix bnn_online_soft_qlogei_ninit30 \
  --force-retrain-bnn \
  --pool-size 5000
```

在已经创建好的 tmux window 里运行：

```bash
studies/online_bnn/scripts/run_bnn_online_in_tmux_window.sh \
  --backend server \
  --seeds 0 \
  --n-init 30 \
  --budget 50 \
  --output-dir experiments/bnn_online_soft_qlogei_ninit30_seed0 \
  --run-id-prefix bnn_online_soft_qlogei_ninit30 \
  --force-retrain-bnn \
  --pool-size 5000
```

不要随便使用 `--reset-checkpoint`，除非你明确想覆盖同一 run prefix 的 checkpoint/queue artifacts。

## 6. Server/Worker HFSS Workflow

在线实验的 server/backend 流程：

```text
optimizer chooses geometry
  -> ServerQueueSimulator checks local cache
  -> cache hit: load .s2p and score
  -> cache miss: write queue/pending/<case_id>.json
  -> worker claims job and runs HFSS
  -> worker writes queue/done/<case_id>.result.json or queue/failed/<case_id>.result.json
  -> optimizer reads result, loads Touchstone, scores objective, saves checkpoint
```

Queue lifecycle：

```text
queue/pending/<job>.json
  -> queue/running/<job>.json
  -> queue/done/<job>.result.json
  -> queue/results/<artifacts>
```

失败时：

```text
queue/failed/<job>.result.json
```

常用检查：

```bash
find queue/pending queue/running -maxdepth 1 -type f | sort | tail -40
find queue/done -maxdepth 1 -type f -name '<prefix>*.result.json' | sort | tail -20
tail -f logs/<latest-run>.log
tmux ls
tmux attach -t <session>
```

如果某个 job 卡住，先看：

- `queue/pending/`：worker 是否还没 claim。
- `queue/running/`：worker 是否正在跑。
- `queue/done/`：result 是否已经回来但 runner 没读到。
- `queue/failed/`：HFSS/worker 是否失败。
- `logs/`：runner 是否正在等某个 case。

关键文档：

- `docs/engineering/server_worker/server_worker_submit_receive_flow.md`
- `docs/engineering/server_worker/server_worker_job_contract.md`
- `docs/engineering/server_worker/server_side.md`
- `docs/engineering/hfss/hfss_worker_side.md`
- `docs/engineering/simulation/simulation_interface.md`

## 7. Adding A New Device, Objective, Or Method

新增一个结构，例如 spiral inductor，推荐顺序：

1. 定义 HFSS template。
2. 定义 worker job JSON contract：变量名、单位、`template_name`、`design_name/setup_name/sweep_name`、输出 `.sNp`。
3. 定义 cache naming 和 result lookup 规则。
4. 定义变量 bounds 和 resolution。
5. 写一个最小 single-case smoke test，确认 worker 可以出 `.s2p`。
6. 做 cache/replay diagnostic，再跑 server-backed online optimization。

新增 objective：

1. 实现或配置新的 `OptimizationSpec`。
2. 明确 objective sign convention：优化器永远最小化。
3. 明确 metrics：例如 BCLF 的 passband/stopband，或 inductor 的 `L(5GHz)`、`Q(5GHz)`、SRF。
4. 先用已有 `.s2p` 或少量 cache 样本做 offline score sanity check。
5. 再接入 online runner。

新增方法：

1. 尽量复用 `OptimizerBase + CandidateSpace + Simulator + ExperimentRecorder`。
2. 方法逻辑放 optimizer/acquisition backend，不要直接在方法里调用 HFSS/queue。
3. 数据写入 `experiments/<experiment_name>/`。
4. 脚本/报告/图写入 `studies/<study_name>/`。
5. 每个 seed 一个独立 run id/checkpoint。
6. 先支持 `--dry-run`、`--resume`，谨慎支持 `--reset-checkpoint`。

推荐先看：

- `docs/engineering/optimization/optimization_experiment_backbone.md`
- `studies/online_bo/scripts/run_bo_online_experiment.py`
- `studies/online_bnn/scripts/run_bnn_online_experiment.py`
- `studies/bnn_acquisition/scripts/diagnose_offline_cache_ranking.py`

## 8. Common Commands And Safety Notes

查看 BNN soft qLogEI 当前进度：

```bash
python - <<'PY'
import json, math
from pathlib import Path
p = Path('experiments/bnn_online_soft_qlogei_ninit30_seed0/bnn_online_soft_qlogei_ninit30_seed0.checkpoint.json')
h = json.loads(p.read_text()).get('history', []) if p.exists() else []
vals = [r for r in h if r.get('objective') is not None and math.isfinite(float(r['objective']))]
print('num_trials', len(vals))
if vals:
    best = min(vals, key=lambda r: float(r['objective']))
    print('best_iter', best.get('iteration'))
    print('best_objective', best.get('objective'))
    print('best_geometry', best.get('geometry'))
    print('last_iter', vals[-1].get('iteration'))
    print('last_objective', vals[-1].get('objective'))
PY
```

查看某个 prefix 的 queue：

```bash
find queue/pending queue/running queue/done queue/failed -maxdepth 1 \
  -type f -name 'bnn_online_soft_qlogei_ninit30_seed0_it*' \
  -printf '%p\n' | sort | tail -80
```

跑 BNN/GP offline cache ranking diagnostic：

```bash
python studies/bnn_acquisition/scripts/diagnose_offline_cache_ranking.py \
  --history-checkpoint experiments/bnn_online_resampled_ninit10_seed0/bnn_online_resampled_ninit10_seed0.checkpoint.json \
  --cache-dir data/bandpass_coupled_line_filter_export/ext \
  --output-dir experiments/bnn_acquisition_offline_cache_ranking/bnn_resampled_ninit10_seed0 \
  --report-dir studies/bnn_acquisition/reports \
  --figure-dir studies/bnn_acquisition/figures \
  --force-retrain-bnn
```

测试常用命令：

```bash
python -m py_compile studies/online_bnn/scripts/run_bnn_online_experiment.py
pytest tests/optimization/test_bnn_botorch_optimizer.py -q
pytest tests/optimization/test_offline_cache_ranking_diagnostic.py -q
```

安全提醒：

- 不要删除或重置 `queue/`，除非确认没有 worker/runner 依赖这些 jobs。
- 不要把新实验写进旧 output dir，除非明确要 resume/overwrite。
- 不要把 notebook 当作唯一实验入口；长期实验要脚本化。
- 不要把 presentation figures 写到 `experiments/`；图和报告放 `studies/`。
- 不要把旧 `n_init=30` BNN checkpoint 复用到新 `n_init=10/15/30` 逻辑混杂的实验里。

## 9. Pointers To Existing Docs

工程：

- `docs/engineering/README.md`
- `docs/engineering/optimization/optimization_experiment_backbone.md`
- `docs/engineering/server_worker/server_worker_submit_receive_flow.md`
- `docs/engineering/server_worker/server_worker_job_contract.md`
- `docs/engineering/hfss/hfss_worker_side.md`
- `docs/engineering/simulation/simulation_interface.md`

学术/实验：

- `docs/academic/README.md`
- `docs/academic/experiments/bclf_optimization_target.md`
- `docs/academic/experiments/bnn_surrogate_experiments.md`
- `docs/academic/experiments/bclf_response_shape_bnn_experiments.md`
- `docs/academic/paper/results_index.md`

Studies：

- `studies/README.md`
- `studies/online_bo/README.md`
- `studies/online_bnn/README.md`
- `studies/online_owafs/README.md`
- `studies/online_comparison/README.md`
- `studies/bnn_acquisition/README.md`
- `studies/bnn_acquisition/reports/bnn_resampled_ninit10_seed0.md`
