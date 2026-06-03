# Notebook Scripts

Experiment-specific notebook runners live here. Keep notebooks as the editing,
plotting, and analysis surface; use these scripts for long background runs.

## Layout

- `bnn_online_experiment/run_bnn_online_experiment.py`: run-only Python version
  of `notebooks/bnn_online_experiment.ipynb`.
- `bnn_online_experiment/run_bnn_online_tmux.sh`: detached tmux launcher for the
  BNN online runner.
- `online_lhs15_shared/{bo,lhs,turbo,bnn}/run_*_lhs15_shared.py`: method-specific
  LHS15 shared-init online runners.
- `online_lhs15_shared/{bo,lhs,turbo,bnn}/run_*_in_tmux_window.sh`: method-specific
  launchers intended to be run inside an existing tmux window. They do not call
  `tmux new-session`.

## BNN Online: Run In Tmux

Recommended command:

```bash
cd /home/zhanghf/projects/vector
notebooks/scripts/bnn_online_experiment/run_bnn_online_tmux.sh
```

The launcher starts a detached tmux session named like
`bnn_online_<timestamp>`, writes continuous logs under `logs/`, writes a final
status file, and preserves the existing `experiments/bnn_online_general/`
checkpoint/resume artifacts.

BNN model checkpoints are loaded by default from
`experiments/bnn_online_general/bnn_checkpoints/`. Add `--force-retrain-bnn`
only when intentionally rebuilding those `.pt` checkpoints from the current
optimizer history.

Common follow-up commands:

```bash
# list active sessions
tmux ls

# reconnect to the printed session
tmux attach -t bnn_online_<timestamp>

# watch progress without attaching
tail -f /home/zhanghf/projects/vector/logs/bnn_online_<timestamp>.log

# check final exit status after completion; 0 means success
cat /home/zhanghf/projects/vector/logs/bnn_online_<timestamp>.status

# stop politely if needed
tmux send-keys -t bnn_online_<timestamp> C-c
```

Pass runner options through the tmux launcher:

```bash
notebooks/scripts/bnn_online_experiment/run_bnn_online_tmux.sh --seeds 0 --backend server
notebooks/scripts/bnn_online_experiment/run_bnn_online_tmux.sh --dry-run --backend server
notebooks/scripts/bnn_online_experiment/run_bnn_online_tmux.sh --no-resume --reset-checkpoint --force-retrain-bnn --seeds 0
```

Run directly without tmux:

```bash
python notebooks/scripts/bnn_online_experiment/run_bnn_online_experiment.py --backend server --log-file logs/bnn_online_manual.log
```

Inspect all CLI options:

```bash
python notebooks/scripts/bnn_online_experiment/run_bnn_online_experiment.py --help
```

## LHS15 Shared-Init Online: Run Inside Existing Tmux Windows

Create tmux windows yourself, then run one method per window. These scripts write
logs/status files under `logs/`, but they do not create tmux sessions or tmux
windows.

```bash
cd /home/zhanghf/projects/vector

# tmux window 1
notebooks/scripts/online_lhs15_shared/bo/run_bo_in_tmux_window.sh

# tmux window 2
notebooks/scripts/online_lhs15_shared/lhs/run_lhs_in_tmux_window.sh

# tmux window 3
notebooks/scripts/online_lhs15_shared/turbo/run_turbo_in_tmux_window.sh

# tmux window 4
notebooks/scripts/online_lhs15_shared/bnn/run_bnn_in_tmux_window.sh
```

Useful dry-runs:

```bash
notebooks/scripts/online_lhs15_shared/bo/run_bo_in_tmux_window.sh --dry-run --seeds 0
notebooks/scripts/online_lhs15_shared/bnn/run_bnn_in_tmux_window.sh --dry-run --seeds 0
```

If you want to run methods sequentially in the current tmux window:

```bash
notebooks/scripts/online_lhs15_shared/run_selected_methods_in_tmux_window.sh

# subset only
ONLINE_LHS15_METHODS="bo turbo" \
  notebooks/scripts/online_lhs15_shared/run_selected_methods_in_tmux_window.sh --seeds 0
```

The direct Python runners are also available:

```bash
python notebooks/scripts/online_lhs15_shared/bo/run_bo_lhs15_shared.py --help
python notebooks/scripts/online_lhs15_shared/lhs/run_lhs_lhs15_shared.py --help
python notebooks/scripts/online_lhs15_shared/turbo/run_turbo_lhs15_shared.py --help
python notebooks/scripts/online_lhs15_shared/bnn/run_bnn_lhs15_shared.py --help
```

## Generic Notebook Fallback

There is no generic helper script in this directory. For notebooks that do not
yet have a dedicated Python runner, run `jupyter nbconvert` manually inside a
named tmux session:

```bash
cd /home/zhanghf/projects/vector
mkdir -p logs executed_notebooks
tmux new -s turbo-nb
```

Inside tmux:

```bash
jupyter nbconvert \
  --to notebook \
  --execute notebooks/turbo_online_experiment.ipynb \
  --output-dir executed_notebooks \
  --output turbo_online_experiment_executed.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  2>&1 | tee logs/turbo_online_nbconvert.log
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t turbo-nb`, and watch the
log with:

```bash
tail -f logs/turbo_online_nbconvert.log
```
