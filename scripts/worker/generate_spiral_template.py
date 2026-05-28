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
    PyAEDTSpiralTemplateGenerator,
    SpiralTemplateBuilder,
    SpiralTemplateDeviceDefinition,
)
from worker.utils import WorkerError  # noqa: E402


DEFAULT_DEFINITION = REPO_ROOT / "configs" / "spiral_inductor_template_device_v1.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a parameterized square spiral inductor AEDT template on an HFSS worker."
    )
    parser.add_argument("--device-definition", type=Path, default=DEFAULT_DEFINITION)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the YAML and print the generated geometry plan without importing PyAEDT.",
    )
    parser.add_argument("--graphical", action="store_true", help="Run AEDT in graphical mode.")
    parser.add_argument("--new-desktop", action="store_true", help="Start a new AEDT desktop session.")
    parser.add_argument("--close-desktop", action="store_true", help="Close AEDT desktop after generation.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing AEDT template.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    definition = SpiralTemplateDeviceDefinition.load(args.device_definition)
    if args.dry_run:
        summary = SpiralTemplateBuilder(definition).dry_run_summary()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    try:
        output_path = PyAEDTSpiralTemplateGenerator().generate(
            definition,
            non_graphical=not args.graphical,
            new_desktop=args.new_desktop,
            close_desktop=args.close_desktop,
            overwrite=args.overwrite,
        )
    except WorkerError as exc:
        print(f"{exc.failure_class}: {exc}", file=sys.stderr)
        return 1
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"generated_aedt={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
