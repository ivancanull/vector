#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


SHARED_SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SHARED_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPT_DIR))

from run_online_lhs15_shared import main as shared_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return shared_main([*args, "--methods", "bnn"])


if __name__ == "__main__":
    raise SystemExit(main())
