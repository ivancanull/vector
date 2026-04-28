from __future__ import annotations

# Fixed-size server convenience command:
#
#   cd /home/zhanghf/projects/vector
#   PYTHONPATH=src:. python scripts/server/run_bclf_5_watch.py \
#     --queue-root queue \
#     --seed 123 \
#     --timeout-sec 7200 \
#     --poll-interval-sec 10
#
# This script always submits 5 random BCLF experiment cases, then waits for results.

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.server import submit_random_bclf_watch as bclf_watch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit exactly 5 random BCLF experiments and wait for worker results."
    )
    parser.add_argument("--queue-root", type=Path, default=REPO_ROOT / "queue")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=REPO_ROOT / "data" / "bandpass_coupled_line_filter_export" / "ext",
    )
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--poll-interval-sec", type=int, default=10)

    parser.add_argument("--template-name", default="bclf_template_v1")
    parser.add_argument("--design-name", default="HFSSDesign1")
    parser.add_argument("--setup-name", default="Setup1")
    parser.add_argument("--sweep-name", default="Sweep")

    parser.add_argument("--unit", default="mm")
    parser.add_argument("--l-min", type=float, default=7.7)
    parser.add_argument("--l-max", type=float, default=7.9)
    parser.add_argument("--l1-min", type=float, default=8.0)
    parser.add_argument("--l1-max", type=float, default=8.2)
    parser.add_argument("--l2-min", type=float, default=7.8)
    parser.add_argument("--l2-max", type=float, default=8.0)
    parser.add_argument("--w-min", type=float, default=0.9)
    parser.add_argument("--w-max", type=float, default=1.1)
    parser.add_argument("--w1-min", type=float, default=0.7)
    parser.add_argument("--w1-max", type=float, default=0.9)
    parser.add_argument("--w2-min", type=float, default=0.9)
    parser.add_argument("--w2-max", type=float, default=1.1)
    parser.add_argument("--s1-min", type=float, default=0.13)
    parser.add_argument("--s1-max", type=float, default=0.16)
    parser.add_argument("--s2-min", type=float, default=0.55)
    parser.add_argument("--s2-max", type=float, default=0.6)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    delegated_args = argparse.Namespace(
        queue_root=args.queue_root,
        export_dir=args.export_dir,
        batch_id=args.batch_id,
        count=5,
        seed=args.seed,
        timeout_sec=args.timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
        template_name=args.template_name,
        design_name=args.design_name,
        setup_name=args.setup_name,
        sweep_name=args.sweep_name,
        unit=args.unit,
        l_min=args.l_min,
        l_max=args.l_max,
        l1_min=args.l1_min,
        l1_max=args.l1_max,
        l2_min=args.l2_min,
        l2_max=args.l2_max,
        w_min=args.w_min,
        w_max=args.w_max,
        w1_min=args.w1_min,
        w1_max=args.w1_max,
        w2_min=args.w2_min,
        w2_max=args.w2_max,
        s1_min=args.s1_min,
        s1_max=args.s1_max,
        s2_min=args.s2_min,
        s2_max=args.s2_max,
    )
    return bclf_watch.run(delegated_args)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
