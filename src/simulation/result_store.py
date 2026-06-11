from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .core import SimulationInput


@dataclass(frozen=True)
class CachedArtifact:
    touchstone_path: Path
    filename: str
    key: str
    metadata: dict[str, Any]


class TouchstoneNamingStrategy(Protocol):
    def build_filename(self, request: SimulationInput, *, port_num: int) -> str:
        ...


class BandpassFilterNamingStrategy:
    """Naming strategy matching existing filter export style."""

    def __init__(
        self,
        *,
        prefix: str = "Bandpass_Coupled_Line_Filter_export_Bandpass_Coupled_Line_Filter",
    ) -> None:
        self.prefix = prefix

    def build_filename(self, request: SimulationInput, *, port_num: int) -> str:
        _ = port_num
        tokens: list[str] = [self.prefix]
        for name in request.variable_order:
            if name not in request.variables:
                raise ValueError(f"Missing simulation variable for naming: {name}")
            value = float(request.variables[name])
            tokens.append(f"{name}_{_to_touchstone_value(value)}")
        return "_".join(tokens) + ".s2p"


class VariableTouchstoneNamingStrategy:
    """Generic variable-token Touchstone naming for simulator result caches."""

    def __init__(self, *, prefix: str) -> None:
        if not prefix:
            raise ValueError("prefix must be non-empty")
        self.prefix = str(prefix)

    def build_filename(self, request: SimulationInput, *, port_num: int) -> str:
        tokens: list[str] = [self.prefix]
        for name in request.variable_order:
            if name not in request.variables:
                raise ValueError(f"Missing simulation variable for naming: {name}")
            value = float(request.variables[name])
            tokens.append(f"{name}_{_to_touchstone_value(value)}")
        return "_".join(tokens) + f".s{int(port_num)}p"


class SimulationResultStore(Protocol):
    def lookup(self, request: SimulationInput) -> CachedArtifact | None:
        ...

    def save(
        self,
        request: SimulationInput,
        source_touchstone: Path,
        metadata: dict[str, Any] | None = None,
    ) -> CachedArtifact:
        ...


class FilesystemResultStore:
    """Filesystem-backed simulation result database with JSON index."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        naming_strategy: TouchstoneNamingStrategy,
        index_filename: str = "index.json",
        key_decimals: int = 12,
        port_num: int = 2,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.naming_strategy = naming_strategy
        self.index_path = self.root_dir / index_filename
        self.key_decimals = int(key_decimals)
        self.port_num = int(port_num)
        if self.key_decimals < 0:
            raise ValueError("key_decimals must be >= 0")

    def lookup(self, request: SimulationInput) -> CachedArtifact | None:
        key = self._build_key(request)
        expected_filename = self.naming_strategy.build_filename(request, port_num=self.port_num)

        index = self._load_index()
        records: dict[str, Any] = dict(index.get("records", {}))

        record = records.get(key)
        if isinstance(record, dict):
            filename = str(record.get("filename", ""))
            if filename:
                path = self.root_dir / filename
                if path.exists():
                    return CachedArtifact(
                        touchstone_path=path,
                        filename=filename,
                        key=key,
                        metadata=dict(record),
                    )
                # stale index entry
                records.pop(key, None)
                index["records"] = records
                self._write_index(index)

        expected_path = self.root_dir / expected_filename
        if expected_path.exists():
            repaired = {
                "case_id": request.case_id,
                "variable_order": list(request.variable_order),
                "variables": {k: float(v) for k, v in request.variables.items()},
                "filename": expected_filename,
            }
            records[key] = repaired
            index["records"] = records
            self._write_index(index)
            return CachedArtifact(
                touchstone_path=expected_path,
                filename=expected_filename,
                key=key,
                metadata=repaired,
            )

        return None

    def save(
        self,
        request: SimulationInput,
        source_touchstone: Path,
        metadata: dict[str, Any] | None = None,
    ) -> CachedArtifact:
        src = Path(source_touchstone)
        if not src.exists():
            raise FileNotFoundError(f"Source touchstone not found: {src}")

        key = self._build_key(request)
        filename = self.naming_strategy.build_filename(request, port_num=self.port_num)
        dst = self.root_dir / filename
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

        index = self._load_index()
        records: dict[str, Any] = dict(index.get("records", {}))
        record = {
            "case_id": request.case_id,
            "variable_order": list(request.variable_order),
            "variables": {k: float(v) for k, v in request.variables.items()},
            "filename": filename,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            record["metadata"] = dict(metadata)
        records[key] = record
        index["records"] = records
        self._write_index(index)

        return CachedArtifact(
            touchstone_path=dst,
            filename=filename,
            key=key,
            metadata=record,
        )

    def _build_key(self, request: SimulationInput) -> str:
        parts: list[str] = []
        for name in request.variable_order:
            if name not in request.variables:
                raise ValueError(f"Missing simulation variable for key: {name}")
            value = float(request.variables[name])
            parts.append(f"{name}={_canonical_decimal(value, self.key_decimals)}")
        return "|".join(parts)

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"records": {}}
        with self.index_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid index format in {self.index_path}")
        if "records" not in loaded or not isinstance(loaded["records"], dict):
            loaded["records"] = {}
        return loaded

    def _write_index(self, index: dict[str, Any]) -> None:
        tmp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=True, indent=2)
        tmp_path.replace(self.index_path)


def _to_touchstone_value(value: float) -> str:
    text = f"{value:.12f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = text + ".0"
    return text.replace("-", "m").replace(".", "d")


def _canonical_decimal(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = text + ".0"
    return text
