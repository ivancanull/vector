from __future__ import annotations

from dataclasses import dataclass

from worker.schemas import HfssSetupConfig, HfssSweepConfig
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
        setup_config: HfssSetupConfig,
        sweep_config: HfssSweepConfig,
    ) -> ResolvedHfssSetup:
        try:
            setup = self._get_setup(hfss, setup_name)
            if setup is None:
                setup = hfss.create_setup(setup_name)
            self._apply_setup_config(setup, setup_config)
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

    def _apply_setup_config(self, setup, config: HfssSetupConfig) -> None:
        ok = setup.enable_adaptive_setup_single(
            freq=config.solution_frequency,
            max_passes=config.maximum_passes,
            max_delta_s=config.max_delta_s,
        )
        if ok is False:
            raise WorkerError(
                "setup_definition_failure",
                f"Failed to configure adaptive HFSS setup {setup.name}",
            )

        props = setup.props
        if config.minimum_passes is not None:
            props["MinimumPasses"] = config.minimum_passes
        if config.minimum_converged_passes is not None:
            props["MinimumConvergedPasses"] = config.minimum_converged_passes
        if config.percent_refinement is not None:
            props["PercentRefinement"] = config.percent_refinement
        if config.basis_order is not None:
            props["BasisOrder"] = config.basis_order
        self._commit(setup)

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

    def _commit(self, obj) -> None:
        update = getattr(obj, "update", None)
        if callable(update):
            update()
