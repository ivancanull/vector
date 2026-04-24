from __future__ import annotations

# Fixed-size server convenience command:
#
#   cd /home/zhanghf/projects/vector
#   PYTHONPATH=src:. python scripts/server/run_spiral_5_watch.py \
#     --queue-root queue \
#     --seed 123 \
#     --timeout-sec 7200 \
#     --poll-interval-sec 10
#
# This script always submits 5 spiral experiment cases, then waits for results.

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.server import submit_random_spiral_watch as spiral_watch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit exactly 5 random spiral experiments and wait for worker results."
    )
    parser.add_argument("--queue-root", type=Path, default=REPO_ROOT / "queue")
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--poll-interval-sec", type=int, default=10)
    parser.add_argument("--template-name", default="si_template_v1")
    parser.add_argument("--design-name", default="HFSSDesign1")
    parser.add_argument("--setup-name", default="Setup1")
    parser.add_argument("--sweep-name", default="Sweep")
    parser.add_argument("--din-min-um", type=float, default=50.0)
    parser.add_argument("--din-max-um", type=float, default=70.0)
    parser.add_argument("--s-min-um", type=float, default=1.0)
    parser.add_argument("--s-max-um", type=float, default=2.0)
    parser.add_argument("--w-min-um", type=float, default=12.0)
    parser.add_argument("--w-max-um", type=float, default=18.0)
    parser.add_argument("--h-um", type=float, default=2.0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    delegated_args = argparse.Namespace(
        queue_root=args.queue_root,
        batch_id=args.batch_id,
        count=5,
        seed=args.seed,
        timeout_sec=args.timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
        template_name=args.template_name,
        design_name=args.design_name,
        setup_name=args.setup_name,
        sweep_name=args.sweep_name,
        din_min_um=args.din_min_um,
        din_max_um=args.din_max_um,
        s_min_um=args.s_min_um,
        s_max_um=args.s_max_um,
        w_min_um=args.w_min_um,
        w_max_um=args.w_max_um,
        h_um=args.h_um,
    )
    return spiral_watch.run(delegated_args)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
