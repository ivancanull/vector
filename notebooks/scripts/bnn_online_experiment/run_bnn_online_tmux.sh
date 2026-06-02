#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  notebooks/scripts/bnn_online_experiment/run_bnn_online_tmux.sh [runner options]

Starts the BNN online Python runner in a detached tmux session with timestamped
log and status files. Extra arguments are passed directly to
notebooks/scripts/bnn_online_experiment/run_bnn_online_experiment.py.

Examples:
  notebooks/scripts/bnn_online_experiment/run_bnn_online_tmux.sh
  notebooks/scripts/bnn_online_experiment/run_bnn_online_tmux.sh --seeds 0 --backend server
  notebooks/scripts/bnn_online_experiment/run_bnn_online_tmux.sh --dry-run --backend server

Useful runner help:
  python notebooks/scripts/bnn_online_experiment/run_bnn_online_experiment.py --help

Environment overrides:
  VECTOR_NOTEBOOK_PYTHON   Python executable to use
                           default: /home/zhanghf/anaconda3/envs/vector/bin/python
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is not available on PATH." >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || realpath "$script_dir/../../..")"
python_bin="${VECTOR_NOTEBOOK_PYTHON:-/home/zhanghf/anaconda3/envs/vector/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Error: Python executable is not runnable: $python_bin" >&2
  echo "Set VECTOR_NOTEBOOK_PYTHON=/path/to/python and try again." >&2
  exit 1
fi

mkdir -p "$repo_root/logs"

timestamp="$(date +%Y%m%d_%H%M%S)"
session_name="bnn_online_${timestamp}"
log_file="$repo_root/logs/${session_name}.log"
status_file="$repo_root/logs/${session_name}.status"
run_script="$repo_root/logs/${session_name}.run.sh"

extra_args=""
if [[ $# -gt 0 ]]; then
  printf -v extra_args '%q ' "$@"
fi

cat >"$run_script" <<EOF
#!/usr/bin/env bash
set -o pipefail
cd "$repo_root"
export PYTHONUNBUFFERED=1
echo "[bnn-online-tmux] started: \$(date)" | tee -a "$log_file"
echo "[bnn-online-tmux] repo: $repo_root" | tee -a "$log_file"
echo "[bnn-online-tmux] session: $session_name" | tee -a "$log_file"
echo "[bnn-online-tmux] log: $log_file" | tee -a "$log_file"
echo "[bnn-online-tmux] status: $status_file" | tee -a "$log_file"
echo "[bnn-online-tmux] command: $python_bin notebooks/scripts/bnn_online_experiment/run_bnn_online_experiment.py --log-file $log_file $extra_args" | tee -a "$log_file"
echo | tee -a "$log_file"
"$python_bin" notebooks/scripts/bnn_online_experiment/run_bnn_online_experiment.py --log-file "$log_file" $extra_args
status=\$?
echo "\$status" > "$status_file"
echo | tee -a "$log_file"
echo "[bnn-online-tmux] finished: \$(date)" | tee -a "$log_file"
echo "[bnn-online-tmux] exit status: \$status" | tee -a "$log_file"
echo "[bnn-online-tmux] attach command: tmux attach -t $session_name" | tee -a "$log_file"
echo "This tmux shell is kept open for inspection. Press Ctrl-d to close it."
exec bash
EOF
chmod +x "$run_script"

tmux new-session -d -s "$session_name" "$run_script"

cat <<EOF
Started BNN online runner in tmux.

Session:
  $session_name

Log:
  $log_file

Status file:
  $status_file

Attach:
  tmux attach -t $session_name

Watch log:
  tail -f "$log_file"

Stop politely:
  tmux send-keys -t $session_name C-c
EOF
