#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _load_index(index_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        return {"records": {}}
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid index format: {index_path}")
    records = payload.get("records")
    if not isinstance(records, dict):
        payload["records"] = {}
    return payload


def _write_index(index_path: Path, payload: dict[str, Any]) -> None:
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(index_path)


def _copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def merge_cache(
    *,
    src_dir: Path,
    dst_dir: Path,
    prefer: str,
    dry_run: bool,
) -> dict[str, int]:
    if prefer not in {"dst", "src"}:
        raise ValueError("prefer must be one of: dst, src")

    src_index_path = src_dir / "index.json"
    dst_index_path = dst_dir / "index.json"

    src_index = _load_index(src_index_path)
    dst_index = _load_index(dst_index_path)
    src_records: dict[str, Any] = dict(src_index.get("records", {}))
    dst_records: dict[str, Any] = dict(dst_index.get("records", {}))

    stats = {
        "src_records": len(src_records),
        "dst_records_before": len(dst_records),
        "added_records": 0,
        "updated_records": 0,
        "skipped_records": 0,
        "copied_files": 0,
        "missing_source_files": 0,
    }

    for key, src_record in src_records.items():
        if not isinstance(src_record, dict):
            stats["skipped_records"] += 1
            continue

        filename = src_record.get("filename")
        if not isinstance(filename, str) or not filename:
            stats["skipped_records"] += 1
            continue

        src_file = src_dir / filename
        dst_file = dst_dir / filename
        if not src_file.exists():
            stats["missing_source_files"] += 1
            continue

        should_write_record = False
        if key not in dst_records:
            should_write_record = True
            stats["added_records"] += 1
        elif prefer == "src":
            should_write_record = True
            stats["updated_records"] += 1
        else:
            stats["skipped_records"] += 1

        if should_write_record:
            _copy_file(src_file, dst_file, dry_run=dry_run)
            if not dst_file.exists() or prefer == "src":
                stats["copied_files"] += 1
            dst_records[key] = dict(src_record)

    stats["dst_records_after"] = len(dst_records)

    if not dry_run:
        dst_index["records"] = dst_records
        dst_dir.mkdir(parents=True, exist_ok=True)
        _write_index(dst_index_path, dst_index)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge touchstone cache from source directory into destination directory, "
            "including index.json records."
        )
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/home/zhanghf/projects/vector/data/cache/filter_touchstone_cache"),
        help="Source cache directory (contains .s2p files and index.json).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/home/zhanghf/projects/vector/data/bandpass_coupled_line_filter_export/ext"),
        help="Destination cache directory (contains .s2p files and index.json).",
    )
    parser.add_argument(
        "--prefer",
        choices=["dst", "src"],
        default="dst",
        help="Conflict policy when a key already exists in destination.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing files.",
    )
    args = parser.parse_args()

    stats = merge_cache(
        src_dir=args.src.resolve(),
        dst_dir=args.dst.resolve(),
        prefer=args.prefer,
        dry_run=bool(args.dry_run),
    )
    print("merge_cache summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
