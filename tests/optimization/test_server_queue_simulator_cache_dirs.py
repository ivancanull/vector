from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optimization.simulators import ServerQueueSimulator
from simulation.result_store import BandpassFilterNamingStrategy, FilesystemResultStore


PARAM_SET = {
    "variable_order": ["L", "W"],
    "geometry": np.array([7.8, 1.0]),
    "unit_map": {"L": "mm", "W": "mm"},
    "template_name": "bclf_template_v1",
    "port_num": 2,
}


def _make_simulator(queue_root: Path, cache_root: Path) -> ServerQueueSimulator:
    store = FilesystemResultStore(
        root_dir=cache_root,
        naming_strategy=BandpassFilterNamingStrategy(),
        port_num=PARAM_SET["port_num"],
    )
    return ServerQueueSimulator(
        queue_root=str(queue_root),
        variable_order=PARAM_SET["variable_order"],
        unit_map=PARAM_SET["unit_map"],
        template_name=PARAM_SET["template_name"],
        port_num=PARAM_SET["port_num"],
        result_store=store,
        run_id="cache_dir_test",
    )


def _prepare_existing_touchstone(cache_root: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    filename = "Bandpass_Coupled_Line_Filter_export_Bandpass_Coupled_Line_Filter_L_7d8_W_1d0.s2p"
    path = cache_root / filename
    path.write_text("dummy", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "subdir",
    [
        Path("data/bandpass_coupled_line_filter_export/ext"),
        Path("data/bandpass_coupled_line_filter_export"),
    ],
)
def test_server_queue_simulator_reads_existing_snp_from_cache_dirs(
    tmp_path: Path, monkeypatch, subdir: Path
) -> None:
    queue_root = tmp_path / "queue"
    (queue_root / "pending").mkdir(parents=True, exist_ok=True)

    cache_dir = tmp_path / subdir
    _prepare_existing_touchstone(cache_dir)

    simulator = _make_simulator(queue_root=queue_root, cache_root=cache_dir)
    monkeypatch.setattr(
        simulator,
        "_load_touchstone",
        lambda _path: (np.array([1.0, 2.0]), np.zeros((2, 2, 2), dtype=complex)),
    )

    result = simulator.simulate(PARAM_SET["geometry"])
    assert result.status == "done"
    assert result.metadata is not None
    assert result.metadata["cache_hit"] is True
    assert list((queue_root / "pending").glob("*.json")) == []
