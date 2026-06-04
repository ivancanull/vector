from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from paper_figures import PaperFigureExporter, choose_ieee_layout


def test_auto_layout_uses_single_for_simple_one_panel_figure() -> None:
    layout, reason = choose_ieee_layout(num_panels=1, num_curves=3, legend_items=3)

    assert layout == "single"
    assert reason == "simple_single_panel"


def test_auto_layout_uses_double_for_dense_multi_method_figure() -> None:
    layout, reason = choose_ieee_layout(num_panels=2, num_curves=4, dense_ticks=True, legend_items=4)

    assert layout == "double"
    assert "multi_panel" in reason
    assert "many_curves" in reason


def test_exporter_writes_only_selected_auto_layout(tmp_path) -> None:
    output_dir = tmp_path / "paper_figures"
    asset_dir = tmp_path / "assets"

    # Pretend an old single-column artifact exists; the auto double export should remove it.
    output_dir.mkdir()
    stale_single = output_dir / "fig_ieee_single.pdf"
    stale_single.write_bytes(b"%PDF-1.4 stale")

    def build(width_in: float, height_in: float, layout: str):
        fig, ax = plt.subplots(figsize=(width_in, height_in))
        ax.plot([0, 1], [0, 1], label=layout)
        ax.legend()
        return fig

    exporter = PaperFigureExporter(output_dir, asset_dir=asset_dir, repo_root=tmp_path)
    rows = exporter.export(
        figure_id="fig",
        build_figure=build,
        layout="auto",
        num_panels=2,
        num_curves=4,
        dense_ticks=True,
        legend_items=4,
        reason="multi_panel_dense_methods",
    )

    assert not stale_single.exists()
    assert (output_dir / "fig_ieee_double.pdf").exists()
    assert (output_dir / "fig_ieee_double.png").exists()
    assert not (output_dir / "fig_ieee_single.pdf").exists()
    assert (asset_dir / "fig_ieee_double.pdf").exists()
    assert (asset_dir / "paper_figure_manifest.csv").exists()
    assert {row["layout"] for row in rows} == {"double"}
    assert {row["reason"] for row in rows} == {"multi_panel_dense_methods"}
