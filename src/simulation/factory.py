from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from worker.schemas import WorkerConfig
from worker.template_registry import ResolvedTemplate

from .core import Simulator
from .result_store import SimulationResultStore

if TYPE_CHECKING:
    from .dataset_simulator import DatasetSimulator
    from .hfss_simulator import HfssExecutor, RealHfssSimulator


class SimulatorFactory:
    """Create optimization-facing simulators from experiment configuration."""

    @staticmethod
    def dataset_from_config(
        config: Any,
        *,
        variable_order: list[str],
        match_atol: float = 1e-9,
        match_rtol: float = 1e-9,
        dataset_path: str | Path | None = None,
    ) -> DatasetSimulator:
        from .dataset_simulator import DatasetSimulator

        data_cfg = _get(config, "data")
        target_cfg = _get(config, "target")
        freqs_cfg = _get(data_cfg, "freqs")
        freqs = np.arange(
            _get(freqs_cfg, "min"),
            _get(freqs_cfg, "max"),
            _get(freqs_cfg, "step"),
        )
        return DatasetSimulator(
            dataset_path=dataset_path or _get(data_cfg, "path"),
            freqs=freqs,
            port_num=_get(target_cfg, "port_num"),
            param_start_idx=_get(data_cfg, "start_idx"),
            variable_order=variable_order,
            match_atol=match_atol,
            match_rtol=match_rtol,
        )

    @staticmethod
    def real_hfss(
        *,
        template_cfg: ResolvedTemplate,
        worker_cfg: WorkerConfig,
        port_num: int,
        unit_map: dict[str, str],
        variable_order: list[str],
        metrics_to_extract: list[str] | None = None,
        executor: HfssExecutor | None = None,
        result_store: SimulationResultStore | None = None,
    ) -> RealHfssSimulator:
        from .hfss_simulator import RealHfssSimulator

        return RealHfssSimulator(
            template_cfg=template_cfg,
            worker_cfg=worker_cfg,
            port_num=port_num,
            unit_map=unit_map,
            variable_order=variable_order,
            metrics_to_extract=metrics_to_extract,
            executor=executor,
            result_store=result_store,
        )

    @staticmethod
    def create(kind: str, **kwargs) -> Simulator:
        if kind == "dataset":
            return SimulatorFactory.dataset_from_config(**kwargs)
        if kind == "hfss":
            return SimulatorFactory.real_hfss(**kwargs)
        raise ValueError(f"Unsupported simulator kind: {kind}")


def _get(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config[key]
    return getattr(config, key)
