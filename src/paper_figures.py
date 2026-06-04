from __future__ import annotations

import csv
import shutil
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

IEEE_SINGLE_COLUMN_WIDTH_IN = 3.5
IEEE_DOUBLE_COLUMN_WIDTH_IN = 7.16
DEFAULT_SINGLE_HEIGHT_IN = 2.45
DEFAULT_DOUBLE_HEIGHT_IN = 3.1
DEFAULT_DPI = 300

Layout = str
FigureBuilder = Callable[[float, float, Layout], Any]


def _relative(path: Path, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


@contextmanager
def paper_figure_style():
    import matplotlib as mpl

    overrides = {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": DEFAULT_DPI,
        "savefig.dpi": DEFAULT_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "stix",
    }
    with mpl.rc_context(overrides):
        yield


def choose_ieee_layout(
    *,
    layout: str = "auto",
    num_panels: int = 1,
    num_curves: int = 1,
    dense_ticks: bool = False,
    legend_items: int = 0,
    has_large_legend: bool = False,
    has_heatmap: bool = False,
    has_response_markers_or_bands: bool = False,
) -> tuple[str, str]:
    if layout in {"single", "double"}:
        return layout, "explicit"
    if layout != "auto":
        raise ValueError("layout must be 'auto', 'single', or 'double'")

    reasons: list[str] = []
    if int(num_panels) > 1:
        reasons.append("multi_panel")
    if int(num_curves) >= 4:
        reasons.append("many_curves")
    if dense_ticks:
        reasons.append("dense_ticks")
    if int(legend_items) >= 4 or has_large_legend:
        reasons.append("large_legend")
    if has_heatmap:
        reasons.append("heatmap")
    if has_response_markers_or_bands:
        reasons.append("response_markers_or_bands")

    if reasons:
        return "double", "_".join(reasons)
    return "single", "simple_single_panel"


class PaperFigureExporter:
    def __init__(
        self,
        output_dir: Path,
        *,
        asset_dir: Path | None = None,
        repo_root: Path | None = None,
        dpi: int = DEFAULT_DPI,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.asset_dir = Path(asset_dir) if asset_dir is not None else None
        self.repo_root = Path(repo_root) if repo_root is not None else None
        self.dpi = int(dpi)

    def export(
        self,
        *,
        figure_id: str,
        build_figure: FigureBuilder,
        layout: str | Iterable[str] = "auto",
        height_in: float | None = None,
        preview_png: bool = True,
        reason: str | None = None,
        num_panels: int = 1,
        num_curves: int = 1,
        dense_ticks: bool = False,
        legend_items: int = 0,
        has_large_legend: bool = False,
        has_heatmap: bool = False,
        has_response_markers_or_bands: bool = False,
    ) -> list[dict[str, str | float | int]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.asset_dir is not None:
            self.asset_dir.mkdir(parents=True, exist_ok=True)

        selected_layouts = self._resolve_layouts(
            layout=layout,
            num_panels=num_panels,
            num_curves=num_curves,
            dense_ticks=dense_ticks,
            legend_items=legend_items,
            has_large_legend=has_large_legend,
            has_heatmap=has_heatmap,
            has_response_markers_or_bands=has_response_markers_or_bands,
        )
        self._remove_unselected_layouts(figure_id, [item[0] for item in selected_layouts])

        rows: list[dict[str, str | float | int]] = []
        for chosen_layout, auto_reason in selected_layouts:
            width = self._width_for_layout(chosen_layout)
            height = float(height_in) if height_in is not None else self._default_height_for_layout(chosen_layout)
            export_reason = reason or auto_reason
            rows.extend(
                self._export_one(
                    figure_id=figure_id,
                    build_figure=build_figure,
                    layout=chosen_layout,
                    reason=export_reason,
                    width_in=width,
                    height_in=height,
                    preview_png=preview_png,
                )
            )

        self._write_manifest(rows, figure_id)
        return rows

    def _resolve_layouts(
        self,
        *,
        layout: str | Iterable[str],
        num_panels: int,
        num_curves: int,
        dense_ticks: bool,
        legend_items: int,
        has_large_legend: bool,
        has_heatmap: bool,
        has_response_markers_or_bands: bool,
    ) -> list[tuple[str, str]]:
        if isinstance(layout, str):
            layout_items = [layout]
        else:
            layout_items = list(layout)
        if not layout_items:
            raise ValueError("at least one layout must be requested")

        resolved: list[tuple[str, str]] = []
        for item in layout_items:
            resolved.append(
                choose_ieee_layout(
                    layout=str(item),
                    num_panels=num_panels,
                    num_curves=num_curves,
                    dense_ticks=dense_ticks,
                    legend_items=legend_items,
                    has_large_legend=has_large_legend,
                    has_heatmap=has_heatmap,
                    has_response_markers_or_bands=has_response_markers_or_bands,
                )
            )
        return list(dict.fromkeys(resolved))

    def _export_one(
        self,
        *,
        figure_id: str,
        build_figure: FigureBuilder,
        layout: str,
        reason: str,
        width_in: float,
        height_in: float,
        preview_png: bool,
    ) -> list[dict[str, str | float | int]]:
        import matplotlib.pyplot as plt

        rows: list[dict[str, str | float | int]] = []
        with paper_figure_style():
            fig = build_figure(width_in, height_in, layout)
            fig.set_size_inches(width_in, height_in, forward=True)
            pdf_path = self.output_dir / f"{figure_id}_ieee_{layout}.pdf"
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
            rows.append(self._manifest_row(figure_id, layout, reason, "pdf", width_in, height_in, pdf_path))

            if preview_png:
                png_path = self.output_dir / f"{figure_id}_ieee_{layout}.png"
                fig.savefig(png_path, format="png", dpi=self.dpi, bbox_inches="tight", pad_inches=0.03)
                rows.append(self._manifest_row(figure_id, layout, reason, "png", width_in, height_in, png_path))

            plt.close(fig)
        return rows

    def _manifest_row(
        self,
        figure_id: str,
        layout: str,
        reason: str,
        file_format: str,
        width_in: float,
        height_in: float,
        path: Path,
    ) -> dict[str, str | float | int]:
        mirrored_path = ""
        if self.asset_dir is not None:
            destination = self.asset_dir / path.name
            shutil.copy2(path, destination)
            mirrored_path = _relative(destination, self.repo_root)

        return {
            "figure_id": figure_id,
            "layout": layout,
            "reason": reason,
            "format": file_format,
            "width_in": round(float(width_in), 4),
            "height_in": round(float(height_in), 4),
            "dpi": self.dpi,
            "path": _relative(path, self.repo_root),
            "mirrored_path": mirrored_path,
        }

    def _write_manifest(self, rows: list[dict[str, str | float | int]], figure_id: str) -> None:
        if not rows:
            return
        manifest_path = self.output_dir / "paper_figure_manifest.csv"
        existing = self._read_manifest(manifest_path)
        kept = [row for row in existing if row.get("figure_id") != figure_id]
        all_rows = kept + rows
        self._write_manifest_file(manifest_path, all_rows)
        if self.asset_dir is not None:
            self._write_manifest_file(self.asset_dir / "paper_figure_manifest.csv", all_rows)

    def _read_manifest(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _write_manifest_file(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "figure_id",
            "layout",
            "reason",
            "format",
            "width_in",
            "height_in",
            "dpi",
            "path",
            "mirrored_path",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _remove_unselected_layouts(self, figure_id: str, selected_layouts: list[str]) -> None:
        all_layouts = {"single", "double"}
        selected = set(selected_layouts)
        for layout in all_layouts - selected:
            for suffix in ("pdf", "png"):
                path = self.output_dir / f"{figure_id}_ieee_{layout}.{suffix}"
                if path.exists():
                    path.unlink()
                if self.asset_dir is not None:
                    asset_path = self.asset_dir / f"{figure_id}_ieee_{layout}.{suffix}"
                    if asset_path.exists():
                        asset_path.unlink()

    def _width_for_layout(self, layout: str) -> float:
        if layout == "single":
            return IEEE_SINGLE_COLUMN_WIDTH_IN
        if layout == "double":
            return IEEE_DOUBLE_COLUMN_WIDTH_IN
        raise ValueError(f"unknown resolved layout: {layout}")

    def _default_height_for_layout(self, layout: str) -> float:
        if layout == "single":
            return DEFAULT_SINGLE_HEIGHT_IN
        if layout == "double":
            return DEFAULT_DOUBLE_HEIGHT_IN
        raise ValueError(f"unknown resolved layout: {layout}")
