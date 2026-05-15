from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from optimization.datasets import PassiveDataset
from optimization.simulators import HfssSimulator, ReplaySimulator, ServerQueueSimulator


class _FakeCached:
    def __init__(self, path: Path) -> None:
        self.touchstone_path = path
        self.filename = path.name
        self.key = "k"
        self.metadata = {}


class _FakeStore:
    def __init__(self, *, cached_path: Path | None = None) -> None:
        self.cached_path = cached_path
        self.lookup_calls = 0
        self.save_calls = 0

    def lookup(self, request):  # type: ignore[no-untyped-def]
        self.lookup_calls += 1
        if self.cached_path is None:
            return None
        return _FakeCached(self.cached_path)

    def save(self, request, source_touchstone, metadata=None):  # type: ignore[no-untyped-def]
        self.save_calls += 1
        return _FakeCached(Path(source_touchstone))


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


def test_replay_simulator_repeat_same_vars_same_result(tmp_path: Path) -> None:
    path = tmp_path / "dataset.pkl"
    _write_payload(path)

    dataset = PassiveDataset.from_pickle(
        path,
        variable_order=["Din", "S", "W"],
        freqs=np.array([1.0, 2.0, 3.0]),
    )
    simulator = ReplaySimulator(dataset=dataset)

    geometry = np.array([55.0, 1.5, 15.0], dtype=float)
    first = simulator.simulate(geometry)
    second = simulator.simulate(geometry)

    assert first.status == "done"
    assert second.status == "done"
    assert first.freqs is not None and second.freqs is not None
    assert first.s_params is not None and second.s_params is not None
    assert first.metadata is not None and second.metadata is not None

    # Re-running the same variable setting in the fake replay simulator
    # should produce the same sampled response.
    assert np.array_equal(first.freqs, second.freqs)
    assert np.array_equal(first.s_params, second.s_params)
    assert first.metadata["case_id"] == second.metadata["case_id"] == "c01"
    assert first.metadata["dataset_index"] == second.metadata["dataset_index"] == 0

    assert simulator.num_calls == 2
    assert simulator.num_success == 2
    assert simulator.num_failures == 0


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


def test_server_queue_simulator_cache_hit_skips_enqueue(tmp_path: Path, monkeypatch) -> None:
    queue_root = tmp_path / "queue"
    (queue_root / "pending").mkdir(parents=True, exist_ok=True)
    touchstone = tmp_path / "cached.s2p"
    touchstone.write_text("dummy", encoding="utf-8")

    store = _FakeStore(cached_path=touchstone)
    simulator = ServerQueueSimulator(
        queue_root=str(queue_root),
        variable_order=["L", "W"],
        unit_map={"L": "mm", "W": "mm"},
        template_name="bclf_template_v1",
        port_num=2,
        result_store=store,
        run_id="run_cache",
    )
    monkeypatch.setattr(
        simulator,
        "_load_touchstone",
        lambda _path: (np.array([1.0, 2.0]), np.zeros((2, 2, 2), dtype=complex)),
    )

    out = simulator.simulate(np.array([7.8, 1.0]))
    assert out.status == "done"
    assert out.metadata is not None and out.metadata["cache_hit"] is True
    assert store.lookup_calls == 1
    assert store.save_calls == 0
    assert list((queue_root / "pending").glob("*.json")) == []


def test_server_queue_simulator_cache_miss_done_and_save(tmp_path: Path, monkeypatch) -> None:
    queue_root = tmp_path / "queue"
    for name in ("pending", "running", "done", "failed", "results"):
        (queue_root / name).mkdir(parents=True, exist_ok=True)

    store = _FakeStore(cached_path=None)
    simulator = ServerQueueSimulator(
        queue_root=str(queue_root),
        variable_order=["L", "W"],
        unit_map={"L": "mm", "W": "mm"},
        template_name="bclf_template_v1",
        port_num=2,
        result_store=store,
        run_id="run_miss",
        poll_interval_sec=0.01,
        job_timeout_sec=0.3,
    )
    simulator.set_context(run_id="run_miss", iteration=0)
    job_id = "run_miss_it0000"
    (queue_root / "running" / f"{job_id}.json").write_text("{}", encoding="utf-8")
    touchstone = queue_root / "results" / f"{job_id}.s2p"
    touchstone.write_text("dummy", encoding="utf-8")
    (queue_root / "done" / f"{job_id}.result.json").write_text(
        (
            '{\n'
            f'  "job_id": "{job_id}",\n'
            '  "status": "done",\n'
            f'  "artifacts": {{"touchstone": "results/{job_id}.s2p"}}\n'
            '}\n'
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        simulator,
        "_load_touchstone",
        lambda _path: (np.array([1.0, 2.0]), np.zeros((2, 2, 2), dtype=complex)),
    )

    out = simulator.simulate(np.array([7.8, 1.0]))
    assert out.status == "done"
    assert out.metadata is not None and out.metadata["cache_hit"] is False
    assert store.lookup_calls == 1
    assert store.save_calls == 1
    # enqueue happened before done-read
    assert (queue_root / "pending" / f"{job_id}.json").exists()
    assert not (queue_root / "running" / f"{job_id}.json").exists()
    assert (queue_root / "archive" / "running" / f"{job_id}.json").exists()


def test_server_queue_simulator_failed_and_timeout(tmp_path: Path, monkeypatch) -> None:
    queue_root = tmp_path / "queue"
    for name in ("pending", "done", "failed", "results"):
        (queue_root / name).mkdir(parents=True, exist_ok=True)

    store = _FakeStore(cached_path=None)
    simulator = ServerQueueSimulator(
        queue_root=str(queue_root),
        variable_order=["L", "W"],
        unit_map={"L": "mm", "W": "mm"},
        template_name="bclf_template_v1",
        port_num=2,
        result_store=store,
        run_id="run_fail",
        poll_interval_sec=0.01,
        job_timeout_sec=0.05,
    )
    monkeypatch.setattr(
        simulator,
        "_load_touchstone",
        lambda _path: (np.array([1.0]), np.zeros((2, 2, 1), dtype=complex)),
    )

    simulator.set_context(run_id="run_fail", iteration=1)
    fail_id = "run_fail_it0001"
    (queue_root / "failed" / f"{fail_id}.result.json").write_text(
        (
            '{\n'
            f'  "job_id": "{fail_id}",\n'
            '  "status": "failed",\n'
            '  "error_message": "mesh_failure",\n'
            '  "artifacts": {"log": "results/fail.log"}\n'
            '}\n'
        ),
        encoding="utf-8",
    )
    failed = simulator.simulate(np.array([7.8, 1.0]))
    assert failed.status == "failed"
    assert failed.error == "mesh_failure"

    # Timeout path.
    simulator.set_context(run_id="run_fail", iteration=2)
    timed = simulator.simulate(np.array([7.7, 0.9]))
    assert timed.status == "failed"
    assert timed.error == "job_timeout"
