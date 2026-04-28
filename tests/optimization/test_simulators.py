from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from optimization.datasets import PassiveDataset
from optimization.simulators import HfssSimulator, ReplaySimulator


def _write_payload(path: Path) -> None:
    s = np.zeros((2, 2, 3), dtype=complex)
    s[0, 0, :] = 10 ** (-10.0 / 20.0)
    payload = {
        "dataset": [
            {"caseid": "c01", "geo_params": [55.0, 1.5, 15.0], "s_params": s},
        ],
        "history_idx": [],
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def test_replay_simulator_hit_and_miss(tmp_path: Path) -> None:
    path = tmp_path / "dataset.pkl"
    _write_payload(path)

    dataset = PassiveDataset.from_pickle(
        path,
        variable_order=["Din", "S", "W"],
        freqs=np.array([1.0, 2.0, 3.0]),
    )
    simulator = ReplaySimulator(dataset=dataset)

    ok = simulator.simulate(np.array([55.0, 1.5, 15.0]))
    assert ok.status == "done"
    assert ok.freqs is not None and ok.s_params is not None
    assert ok.metadata is not None and ok.metadata["case_id"] == "c01"

    fail = simulator.simulate(np.array([99.0, 1.5, 15.0]))
    assert fail.status == "failed"
    assert fail.error is not None

    assert simulator.num_calls == 2
    assert simulator.num_success == 1
    assert simulator.num_failures == 1


def test_hfss_simulator_adapter_success_and_failure() -> None:
    class FakeOutput:
        def __init__(self, case_id: str, geo: np.ndarray) -> None:
            self.case_id = case_id
            self.freqs = np.array([1.0, 2.0])
            self.s_params = np.zeros((2, 2, 2), dtype=complex)
            self.geo_params = geo
            self.artifacts = None

    class FakeBackend:
        def simulate(self, request):  # type: ignore[no-untyped-def]
            if request.variables["Din"] < 0:
                raise ValueError("Din must be positive")
            return FakeOutput(request.case_id, request.geo_params())

    sim = HfssSimulator(backend=FakeBackend(), variable_order=["Din", "S", "W"])

    ok = sim.simulate(np.array([55.0, 1.5, 15.0]))
    assert ok.status == "done"
    assert ok.s_params is not None and ok.s_params.shape == (2, 2, 2)

    fail = sim.simulate(np.array([-1.0, 1.5, 15.0]))
    assert fail.status == "failed"
    assert fail.error is not None

    assert sim.num_calls == 2
    assert sim.num_success == 1
    assert sim.num_failures == 1
