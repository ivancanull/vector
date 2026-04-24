from __future__ import annotations

import argparse
from pathlib import Path

from scripts.server import run_spiral_5_watch as script


def test_run_forces_count_to_five(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(args: argparse.Namespace) -> int:
        captured["count"] = args.count
        captured["queue_root"] = args.queue_root
        return 0

    monkeypatch.setattr(script.spiral_watch, "run", _fake_run)
    args = argparse.Namespace(
        queue_root=tmp_path / "queue",
        batch_id="batch",
        seed=123,
        timeout_sec=100,
        poll_interval_sec=1,
        template_name="si_template_v1",
        design_name="HFSSDesign1",
        setup_name="Setup1",
        sweep_name="Sweep",
        din_min_um=50.0,
        din_max_um=70.0,
        s_min_um=1.0,
        s_max_um=2.0,
        w_min_um=12.0,
        w_max_um=18.0,
        h_um=2.0,
    )
    rc = script.run(args)
    assert rc == 0
    assert captured["count"] == 5
    assert captured["queue_root"] == tmp_path / "queue"
