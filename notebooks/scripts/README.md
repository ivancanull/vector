# Notebook Scripts

Experiment-specific notebook runners live here. Keep notebooks as the editing,
plotting, and analysis surface; use these scripts for long background runs.

## Layout

- `bnn_online_experiment/run_bnn_online_experiment.py`: run-only Python version
  of `notebooks/bnn_online_experiment.ipynb`.
- `bnn_online_experiment/run_bnn_online_tmux.sh`: detached tmux launcher for the
  BNN online runner.

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
notebooks/scripts/bnn_online_experiment/run_bnn_online_tmux.sh --no-resume --reset-checkpoint --seeds 0
```

Run directly without tmux:

```bash
python notebooks/scripts/bnn_online_experiment/run_bnn_online_experiment.py --backend server --log-file logs/bnn_online_manual.log
```

Inspect all CLI options:

```bash
python notebooks/scripts/bnn_online_experiment/run_bnn_online_experiment.py --help
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
