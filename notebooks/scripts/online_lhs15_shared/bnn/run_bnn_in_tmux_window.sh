#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  notebooks/scripts/online_lhs15_shared/bnn/run_bnn_in_tmux_window.sh [runner options]

Runs the BNN LHS15 shared-init online experiment in the current shell/tmux window.
This script does not create a tmux session or window. Extra arguments are passed
to run_bnn_lhs15_shared.py.

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
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || realpath "$script_dir/../../../..")"
python_bin="${VECTOR_NOTEBOOK_PYTHON:-/home/zhanghf/anaconda3/envs/vector/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Error: Python executable is not runnable: $python_bin" >&2
  echo "Set VECTOR_NOTEBOOK_PYTHON=/path/to/python and try again." >&2
  exit 1
fi

mkdir -p "$repo_root/logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="bnn_lhs15_shared_${timestamp}"
log_file="$repo_root/logs/${run_name}.log"
status_file="$repo_root/logs/${run_name}.status"

cd "$repo_root"
export PYTHONUNBUFFERED=1
echo "[bnn-lhs15-shared] started: $(date)" | tee -a "$log_file"
echo "[bnn-lhs15-shared] repo: $repo_root" | tee -a "$log_file"
echo "[bnn-lhs15-shared] log: $log_file" | tee -a "$log_file"
echo "[bnn-lhs15-shared] status: $status_file" | tee -a "$log_file"
echo "[bnn-lhs15-shared] command: $python_bin notebooks/scripts/online_lhs15_shared/bnn/run_bnn_lhs15_shared.py --log-file $log_file $*" | tee -a "$log_file"
echo | tee -a "$log_file"

set +e
"$python_bin" notebooks/scripts/online_lhs15_shared/bnn/run_bnn_lhs15_shared.py --log-file "$log_file" "$@"
status=$?
set -e

echo "$status" > "$status_file"
echo | tee -a "$log_file"
echo "[bnn-lhs15-shared] finished: $(date)" | tee -a "$log_file"
echo "[bnn-lhs15-shared] exit status: $status" | tee -a "$log_file"
exit "$status"
