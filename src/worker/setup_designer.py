from __future__ import annotations

from dataclasses import dataclass

from worker.schemas import HfssSweepConfig
from worker.utils import WorkerError


@dataclass(frozen=True)
class ResolvedHfssSetup:
    setup_name: str
    sweep_name: str


class HfssSetupDesigner:
    def ensure_setup(
        self,
        hfss,
        *,
        setup_name: str,
        sweep_name: str,
        sweep_config: HfssSweepConfig,
    ) -> ResolvedHfssSetup:
        try:
            setup = self._get_setup(hfss, setup_name)
            if setup is None:
                setup = hfss.create_setup(setup_name)
            self._ensure_sweep(setup, sweep_name, sweep_config)
            return ResolvedHfssSetup(setup_name=setup_name, sweep_name=sweep_name)
        except WorkerError:
            raise
        except Exception as exc:
            raise WorkerError(
                "setup_definition_failure",
                f"Failed to ensure HFSS setup {setup_name}/{sweep_name}: {exc}",
            ) from exc

    def _get_setup(self, hfss, setup_name: str):
        for setup in getattr(hfss, "setups", []):
            if getattr(setup, "name", None) == setup_name:
                return setup
        existing_names = getattr(hfss, "existing_analysis_setups", None)
        if isinstance(existing_names, list) and setup_name in existing_names:
            for setup in getattr(hfss, "setups", []):
                if getattr(setup, "name", None) == setup_name:
                    return setup
        return None

    def _ensure_sweep(self, setup, sweep_name: str, config: HfssSweepConfig):
        existing = setup.get_sweep(sweep_name)
        if existing:
            deleted = setup.delete_sweep(sweep_name)
            if deleted is False:
                raise WorkerError(
                    "setup_definition_failure",
                    f"Failed to replace existing sweep {sweep_name} in setup {setup.name}",
                )

        sweep_type = self._normalize_sweep_type(config.sweep_type)
        if config.sweep_data is not None:
            range_type, start, stop, count_or_step = self._parse_sweep_data(config.sweep_data)
            if range_type == "LINC":
                sweep = setup.create_frequency_sweep(
                    start_frequency=start,
                    stop_frequency=stop,
                    num_of_freq_points=int(count_or_step),
                    name=sweep_name,
                    sweep_type=sweep_type,
                )
            elif range_type == "LIN":
                sweep = setup.create_linear_step_sweep(
                    start_frequency=start,
                    stop_frequency=stop,
                    step_size=count_or_step,
                    name=sweep_name,
                    sweep_type=sweep_type,
                )
            else:
                raise WorkerError(
                    "setup_definition_failure",
                    f"Unsupported sweep_data format: {config.sweep_data}",
                )
        else:
            sweep = setup.create_frequency_sweep(
                start_frequency=config.start_freq,
                stop_frequency=config.stop_freq,
                num_of_freq_points=config.point_count,
                name=sweep_name,
                sweep_type=sweep_type,
            )

        if not sweep:
            raise WorkerError(
                "setup_definition_failure",
                f"Failed to create sweep {sweep_name} in setup {setup.name}",
            )
        return sweep

    def _normalize_sweep_type(self, sweep_type: str) -> str:
        mapping = {
            "interpolating": "Interpolating",
            "discrete": "Discrete",
            "fast": "Fast",
        }
        return mapping[sweep_type]

    def _parse_sweep_data(self, sweep_data: str) -> tuple[str, str, str, str]:
        parts = sweep_data.split()
        if len(parts) != 4:
            raise WorkerError(
                "setup_definition_failure",
                f"Unsupported sweep_data format: {sweep_data}",
            )
        range_type, start, stop, count_or_step = parts
        return range_type.upper(), start, stop, count_or_step

