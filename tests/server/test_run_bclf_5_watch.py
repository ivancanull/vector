from __future__ import annotations

import argparse
from pathlib import Path

from scripts.server import run_bclf_5_watch as script


def test_run_forces_count_to_five(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(args: argparse.Namespace) -> int:
        captured["count"] = args.count
        captured["queue_root"] = args.queue_root
        captured["export_dir"] = args.export_dir
        captured["setup_name"] = args.setup_name
        captured["sweep_name"] = args.sweep_name
        return 0

    monkeypatch.setattr(script.bclf_watch, "run", _fake_run)
    args = argparse.Namespace(
        queue_root=tmp_path / "queue",
        export_dir=tmp_path / "data" / "bandpass_coupled_line_filter_export" / "ext",
        batch_id="batch",
        seed=123,
        timeout_sec=100,
        poll_interval_sec=1,
        template_name="bclf_template_v1",
        design_name="HFSSDesign1",
        setup_name="Setup1",
        sweep_name="Sweep",
        unit="mm",
        l_min=7.7,
        l_max=7.9,
        l1_min=8.0,
        l1_max=8.2,
        l2_min=7.8,
        l2_max=8.0,
        w_min=0.9,
        w_max=1.1,
        w1_min=0.7,
        w1_max=0.9,
        w2_min=0.9,
        w2_max=1.1,
        s1_min=0.13,
        s1_max=0.16,
        s2_min=0.55,
        s2_max=0.6,
    )
    rc = script.run(args)
    assert rc == 0
    assert captured["count"] == 5
    assert captured["queue_root"] == tmp_path / "queue"
    assert captured["setup_name"] == "Setup1"
    assert captured["sweep_name"] == "Sweep"
    assert captured["export_dir"] == tmp_path / "data" / "bandpass_coupled_line_filter_export" / "ext"
