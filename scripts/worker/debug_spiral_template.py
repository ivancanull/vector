from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from worker.spiral_template_builder import (  # noqa: E402
    SpiralTemplateDebugger,
    SpiralTemplateDeviceDefinition,
)
from worker.utils import WorkerError  # noqa: E402


DEFAULT_DEFINITION = REPO_ROOT / "configs" / "spiral_inductor_template_device_v1.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one nominal HFSS debug simulation for a generated spiral inductor template."
    )
    parser.add_argument("--device-definition", type=Path, default=DEFAULT_DEFINITION)
    parser.add_argument("--aedt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output Touchstone path, for example .s2p.")
    parser.add_argument("--target-ghz", type=float, default=5.0)
    parser.add_argument("--graphical", action="store_true", help="Run AEDT in graphical mode.")
    parser.add_argument("--new-desktop", action="store_true", help="Start a new AEDT desktop session.")
    parser.add_argument("--close-desktop", action="store_true", help="Close AEDT desktop after debug run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    definition = SpiralTemplateDeviceDefinition.load(args.device_definition)
    try:
        result = SpiralTemplateDebugger().run(
            definition,
            aedt_path=args.aedt,
            output_touchstone=args.output,
            target_ghz=args.target_ghz,
            non_graphical=not args.graphical,
            new_desktop=args.new_desktop,
            close_desktop=args.close_desktop,
        )
    except (WorkerError, FileNotFoundError) as exc:
        if isinstance(exc, WorkerError):
            print(f"{exc.failure_class}: {exc}", file=sys.stderr)
        else:
            print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
