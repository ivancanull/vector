#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  notebooks/scripts/online_lhs15_shared/run_selected_methods_in_tmux_window.sh [runner options]

Runs selected LHS15 shared-init methods sequentially in the current shell/tmux
window. This script does not create a tmux session or window.

By default it runs all methods: bo lhs turbo bnn owafs_bnn.
Set ONLINE_LHS15_METHODS to choose a subset, for example:
  ONLINE_LHS15_METHODS="bo turbo" notebooks/scripts/online_lhs15_shared/run_selected_methods_in_tmux_window.sh --seeds 0

For parallel runs, create separate tmux windows yourself and run one method
script in each window.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || realpath "$script_dir/../../..")"
cd "$repo_root"

methods="${ONLINE_LHS15_METHODS:-bo lhs turbo bnn owafs_bnn}"
for method in $methods; do
  case "$method" in
    bo)
      notebooks/scripts/online_lhs15_shared/bo/run_bo_in_tmux_window.sh "$@"
      ;;
    lhs)
      notebooks/scripts/online_lhs15_shared/lhs/run_lhs_in_tmux_window.sh "$@"
      ;;
    turbo)
      notebooks/scripts/online_lhs15_shared/turbo/run_turbo_in_tmux_window.sh "$@"
      ;;
    bnn)
      notebooks/scripts/online_lhs15_shared/bnn/run_bnn_in_tmux_window.sh "$@"
      ;;
    owafs_bnn)
      notebooks/scripts/online_lhs15_shared/owafs_bnn/run_owafs_bnn_in_tmux_window.sh "$@"
      ;;
    *)
      echo "Unsupported method in ONLINE_LHS15_METHODS: $method" >&2
      exit 2
      ;;
  esac
done
