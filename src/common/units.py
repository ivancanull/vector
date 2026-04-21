from __future__ import annotations

import re

UNITED_VALUE_PATTERN = re.compile(
    r"^\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*[a-zA-Z][a-zA-Z0-9/_-]*\s*$"
)


def has_explicit_unit(value: str) -> bool:
    return bool(UNITED_VALUE_PATTERN.match(value))
