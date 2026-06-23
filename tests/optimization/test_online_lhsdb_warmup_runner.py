from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd

from simulation.core import SimulationInput
from simulation.result_store import BandpassFilterNamingStrategy, FilesystemResultStore


def _load_runner_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "studies" / "online_owafs" / "scripts" / "run_online_lhsdb_warmup.py"
    spec = importlib.util.spec_from_file_location("run_online_lhsdb_warmup", script_path)
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
) -> dict[tuple[float, ...], Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    strategy = BandpassFilterNamingStrategy()
    paths: dict[tuple[float, ...], Path] = {}
    for idx, geometry in enumerate(np.asarray(geometries, dtype=float)):
        variables = {name: float(geometry[pos]) for pos, name in enumerate(variable_order)}
        request = SimulationInput(
            case_id=f"cached_{idx:03d}",
            variable_order=list(variable_order),
            variables=variables,
        )
        filename = strategy.build_filename(request, port_num=2)
        path = cache_dir / filename
        path.write_text("dummy touchstone placeholder", encoding="utf-8")
        paths[tuple(round(float(value), 8) for value in geometry)] = path
    return paths


def _write_lhs_database_rows(
    lhs_dataset_dir: Path,
    *,
    variable_order: list[str],
    geometries_by_seed: dict[int, np.ndarray],
    cache_paths: dict[tuple[float, ...], Path],
) -> None:
    lhs_dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_seed, geometries in sorted(geometries_by_seed.items()):
        best_so_far = float("inf")
        for idx, geometry in enumerate(np.asarray(geometries, dtype=float)):
            objective = float(source_seed * 100 + idx)
            best_so_far = min(best_so_far, objective)
            key = tuple(round(float(value), 8) for value in geometry)
            cache_path = cache_paths[key]
            variables = {name: float(geometry[pos]) for pos, name in enumerate(variable_order)}
            row = {
                "source_lhs_seed": int(source_seed),
                "source_row_index": int(idx),
                "source_lhs_iteration": int(idx),
                "source_step": int(idx),
                "source_phase": "lhs",
                "source_case_id": f"lhs_online_general_seed{source_seed}_it{idx:04d}",
                "source_csv": "",
                "geometry": json.dumps([float(value) for value in geometry]),
                "geometry_normalized": json.dumps([float(value) for value in geometry]),
                "geometry_key": "|".join(f"{float(value):.8f}" for value in geometry),
                "objective": objective,
                "best_so_far": best_so_far,
                "cache_hit_original": True,
                "original_nearest_distance_normalized": 0.1,
                "original_candidate_pool_size": 20,
                "original_selected_pos": idx,
                "resolved_cache_filename": cache_path.name,
                "resolved_cache_path": str(cache_path),
                "cache_resolves": True,
                "within_bounds": True,
            }
            row.update(variables)
            rows.append(row)
    pd.DataFrame(rows).to_csv(lhs_dataset_dir / "lhs_online_general_rows.csv", index=False)


def test_retained_initial_design_sources_are_deterministic_and_distinct(tmp_path: Path) -> None:
    module = _load_runner_module()
    variable_order = ["L", "W"]
    variable_bounds = {"L": (0.0, 1.0), "W": (0.0, 1.0)}
    geometries_by_seed = {
        0: np.asarray([[round(0.05 + 0.03 * i, 2), round(0.10 + 0.02 * i, 2)] for i in range(20)]),
        1: np.asarray([[round(0.10 + 0.02 * i, 2), round(0.55 + 0.01 * i, 2)] for i in range(20)]),
    }
    all_geometries = np.vstack(list(geometries_by_seed.values()))
    cache_dir = tmp_path / "cache"
    cache_paths = _write_cached_touchstones(cache_dir, variable_order=variable_order, geometries=all_geometries)
    lhs_dataset_dir = tmp_path / "lhs"
    _write_lhs_database_rows(
        lhs_dataset_dir,
        variable_order=variable_order,
        geometries_by_seed=geometries_by_seed,
        cache_paths=cache_paths,
    )

    random_builder = module.RandomInitialDesignBuilder(
        variable_order=variable_order,
        variable_bounds=variable_bounds,
        cache_dir=cache_dir,
        manifest_dir=tmp_path / "random_initial_designs",
    )

    seed0_first = random_builder.build_or_load(seed=0, n_init=5, rebuild=True)
    seed0_second = random_builder.build_or_load(seed=0, n_init=5, rebuild=True)
    seed1 = random_builder.build_or_load(seed=1, n_init=5, rebuild=True)

    assert seed0_first.path.exists()
    assert seed0_first.payload["mode"] == "random"
    assert seed0_first.payload["cache_required"] is False
    assert seed0_first.geometries.shape == (5, 2)
    assert len({tuple(row) for row in seed0_first.geometries.tolist()}) == 5
    assert np.all(seed0_first.geometries >= 0.0)
    assert np.all(seed0_first.geometries <= 1.0)
    assert seed0_first.geometries.tolist() == seed0_second.geometries.tolist()
    assert seed0_first.geometries.tolist() != seed1.geometries.tolist()
    assert all(row["cache_required"] is False for row in seed0_first.selected_rows)

    previous_builder = module.PreviousLHSRandomInitialDesignBuilder(
        variable_order=variable_order,
        variable_bounds=variable_bounds,
        cache_dir=cache_dir,
        manifest_dir=tmp_path / "previous_initial_designs",
        previous_lhs_dir=tmp_path / "previous_lhs",
        lhs_dataset_dir=lhs_dataset_dir,
    )
    previous_seed1 = previous_builder.build_or_load(seed=1, n_init=5, rebuild=True)
    previous_seed1_repeat = previous_builder.build_or_load(seed=1, n_init=5, rebuild=True)
    assert previous_seed1.payload["mode"] == "previous_lhs_random"
    assert "lhs_database_random" in previous_builder.accepted_modes()
    assert previous_seed1.geometries.tolist() == previous_seed1_repeat.geometries.tolist()
    assert all(row["cache_required"] is True for row in previous_seed1.selected_rows)
    selected_keys = {tuple(row) for row in previous_seed1.geometries.tolist()}
    source_keys = {tuple(row) for row in all_geometries.tolist()}
    assert selected_keys.issubset(source_keys)
    assert previous_seed1.geometries.tolist() != geometries_by_seed[1][:5].tolist()

    legacy_payload = dict(previous_seed1.payload)
    legacy_payload["mode"] = "lhs_database_random"
    previous_seed1.path.write_text(json.dumps(legacy_payload, indent=2), encoding="utf-8")
    previous_seed1_legacy = previous_builder.build_or_load(seed=1, n_init=5, rebuild=False)
    assert previous_seed1_legacy.payload["mode"] == "lhs_database_random"
    assert previous_seed1_legacy.geometries.tolist() == previous_seed1.geometries.tolist()

    store = FilesystemResultStore(
        root_dir=cache_dir,
        naming_strategy=BandpassFilterNamingStrategy(),
        port_num=2,
    )
    for idx, geometry in enumerate(previous_seed1.geometries):
        request = SimulationInput(
            case_id=f"verify_{idx:03d}",
            variable_order=variable_order,
            variables={name: float(geometry[pos]) for pos, name in enumerate(variable_order)},
        )
        assert store.lookup(request) is not None
