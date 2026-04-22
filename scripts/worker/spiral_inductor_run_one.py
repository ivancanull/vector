from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.paths import ensure_runtime_dirs  # noqa: E402
from worker.config import load_templates_config, load_worker_config  # noqa: E402
from worker.logger import setup_global_logger  # noqa: E402
from worker.queue_watcher import try_claim_one_job  # noqa: E402
from worker.template_registry import TemplateRegistry  # noqa: E402
from worker.worker_main import process_claimed_job  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Claim one queued spiral job, run simulation, and export touchstone."
    )
    parser.add_argument(
        "--worker-config",
        type=Path,
        default=REPO_ROOT / "configs" / "worker.yaml",
        help="Path to worker config YAML.",
    )
    return parser.parse_args()


def run_one(worker_config_path: Path) -> int:
    cfg = load_worker_config(worker_config_path)
    templates_cfg = load_templates_config(cfg.templates_yaml)
    ensure_runtime_dirs(cfg.queue_root, cfg.logs_dir)

    template_registry = TemplateRegistry(templates_cfg)
    global_logger = setup_global_logger(cfg.logs_dir)

    claimed = try_claim_one_job(cfg.queue_root)
    if claimed is None:
        print("No pending job found.")
        return 0

    process_claimed_job(claimed, cfg, template_registry, global_logger)
    return 0


def main() -> int:
    args = parse_args()
    return run_one(args.worker_config)


if __name__ == "__main__":
    raise SystemExit(main())
