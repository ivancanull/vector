#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  studies/online_bnn/scripts/run_bnn_online_in_tmux_window.sh [runner options]

Runs the BNN online Python runner in the current shell/tmux window.
This script does not create a tmux session or tmux window. Extra arguments are
passed directly to studies/online_bnn/scripts/run_bnn_online_experiment.py.

Examples:
  studies/online_bnn/scripts/run_bnn_online_in_tmux_window.sh
  studies/online_bnn/scripts/run_bnn_online_in_tmux_window.sh --seeds 0 --backend server
  studies/online_bnn/scripts/run_bnn_online_in_tmux_window.sh --dry-run --backend server

Environment overrides:
  VECTOR_NOTEBOOK_PYTHON   Python executable to use
                           default: /home/zhanghf/anaconda3/envs/vector/bin/python
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
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
run_name="bnn_online_${timestamp}"
log_file="$repo_root/logs/${run_name}.log"
status_file="$repo_root/logs/${run_name}.status"

cd "$repo_root"
export PYTHONUNBUFFERED=1
echo "[bnn-online] started: $(date)" | tee -a "$log_file"
echo "[bnn-online] repo: $repo_root" | tee -a "$log_file"
echo "[bnn-online] log: $log_file" | tee -a "$log_file"
echo "[bnn-online] status: $status_file" | tee -a "$log_file"
echo "[bnn-online] command: $python_bin studies/online_bnn/scripts/run_bnn_online_experiment.py --log-file $log_file $*" | tee -a "$log_file"
echo | tee -a "$log_file"

set +e
"$python_bin" studies/online_bnn/scripts/run_bnn_online_experiment.py --log-file "$log_file" "$@"
status=$?
set -e

echo "$status" > "$status_file"
echo | tee -a "$log_file"
echo "[bnn-online] finished: $(date)" | tee -a "$log_file"
echo "[bnn-online] exit status: $status" | tee -a "$log_file"
exit "$status"
