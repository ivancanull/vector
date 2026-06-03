from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np

from simulation.core import SimulationInput
from simulation.result_store import BandpassFilterNamingStrategy, FilesystemResultStore


def _load_runner_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "notebooks" / "scripts" / "online_lhs15_shared" / "run_online_lhs15_shared.py"
    spec = importlib.util.spec_from_file_location("run_online_lhs15_shared", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_cached_touchstones(
    cache_dir: Path,
    *,
    variable_order: list[str],
    geometries: np.ndarray,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    strategy = BandpassFilterNamingStrategy()
    for idx, geometry in enumerate(np.asarray(geometries, dtype=float)):
        variables = {name: float(geometry[pos]) for pos, name in enumerate(variable_order)}
        request = SimulationInput(
            case_id=f"cached_{idx:03d}",
            variable_order=list(variable_order),
            variables=variables,
        )
        filename = strategy.build_filename(request, port_num=2)
        (cache_dir / filename).write_text("dummy touchstone placeholder", encoding="utf-8")


def test_shared_initial_design_builder_selects_deterministic_cached_lhs_points(tmp_path: Path) -> None:
    module = _load_runner_module()
    variable_order = ["L", "W"]
    variable_bounds = {"L": (0.0, 1.0), "W": (0.0, 1.0)}
    grid = np.asarray(
        [[round(0.05 + 0.1 * i, 2), round(0.05 + 0.1 * j, 2)] for i in range(10) for j in range(10)],
        dtype=float,
    )
    cache_dir = tmp_path / "cache"
    _write_cached_touchstones(cache_dir, variable_order=variable_order, geometries=grid)

    builder = module.SharedInitialDesignBuilder(
        variable_order=variable_order,
        variable_bounds=variable_bounds,
        cache_dir=cache_dir,
        manifest_dir=tmp_path / "initial_designs",
    )

    seed0_first = builder.build_or_load(seed=0, n_init=15, rebuild=True)
    seed0_second = builder.build_or_load(seed=0, n_init=15, rebuild=True)
    seed1 = builder.build_or_load(seed=1, n_init=15, rebuild=True)

    assert seed0_first.path.exists()
    assert seed0_first.geometries.shape == (15, 2)
    assert len({tuple(row) for row in seed0_first.geometries.tolist()}) == 15
    assert np.all(seed0_first.geometries >= 0.0)
    assert np.all(seed0_first.geometries <= 1.0)
    assert seed0_first.geometries.tolist() == seed0_second.geometries.tolist()
    assert seed0_first.geometries.tolist() != seed1.geometries.tolist()

    store = FilesystemResultStore(
        root_dir=cache_dir,
        naming_strategy=BandpassFilterNamingStrategy(),
        port_num=2,
    )
    for idx, geometry in enumerate(seed0_first.geometries):
        request = SimulationInput(
            case_id=f"verify_{idx:03d}",
            variable_order=variable_order,
            variables={name: float(geometry[pos]) for pos, name in enumerate(variable_order)},
        )
        assert store.lookup(request) is not None
