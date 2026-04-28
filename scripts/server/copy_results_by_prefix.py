from __future__ import annotations

"""Copy result artifacts by filename prefix into a target export directory."""

from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO_ROOT / "queue" / "results"
DEFAULT_DEST_DIR = REPO_ROOT / "data" / "bandpass_coupled_line_filter_export" / "ext"
FIXED_PREFIX = "Bandpass_Coupled_Line_Filter_export_Bandpass_Coupled_Line_Filter"


def _iter_matched_files(source_dir: Path, prefix: str) -> list[Path]:
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"source-dir not found or not a directory: {source_dir}")
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file()
        and path.name.startswith(prefix)
        and path.suffix.lower().startswith(".s")
        and path.suffix.lower().endswith("p")
    )


def run() -> int:
    matched = _iter_matched_files(DEFAULT_SOURCE_DIR, FIXED_PREFIX)
    if not matched:
        print(
            f"matched_count=0 prefix={FIXED_PREFIX} source_dir={DEFAULT_SOURCE_DIR}"
        )
        return 0

    print(
        f"matched_count={len(matched)} prefix={FIXED_PREFIX} source_dir={DEFAULT_SOURCE_DIR}"
    )
    DEFAULT_DEST_DIR.mkdir(parents=True, exist_ok=True)
    for src in matched:
        dst = DEFAULT_DEST_DIR / src.name
        # Required behavior: replace destination when it already exists.
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
