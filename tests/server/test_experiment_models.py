from __future__ import annotations

import pytest

from server.experiment_adapter import to_batch_simulation_job
from server.experiment_models import (
    CaseVariant,
    DesignSpace,
    Experiment,
    ExperimentObjective,
    SpiralInductorExperiment,
    SpiralVariableRanges,
    VariableSpec,
)


def test_design_space_rejects_missing_or_invalid_units() -> None:
    design_space = DesignSpace(
        variable_specs={
            "Din": VariableSpec(
                name="Din",
                unit="um",
                constraint_type="range",
                min_value=50.0,
                max_value=70.0,
            )
        }
    )
    with pytest.raises(ValueError, match="explicit units"):
        design_space.validate_case_variables({"Din": "55"})

    with pytest.raises(ValueError, match="must use unit"):
        design_space.validate_case_variables({"Din": "55mm"})


def test_design_space_rejects_out_of_range_and_invalid_enum() -> None:
    design_space = DesignSpace(
        variable_specs={
            "Din": VariableSpec(
                name="Din",
                unit="um",
                constraint_type="range",
                min_value=50.0,
                max_value=70.0,
            ),
            "H": VariableSpec(
                name="H",
                unit="um",
                constraint_type="enum",
                options=[2.0],
            ),
        }
    )
    with pytest.raises(ValueError, match="outside range"):
        design_space.validate_case_variables({"Din": "75um", "H": "2.0um"})

    with pytest.raises(ValueError, match="not in enum options"):
        design_space.validate_case_variables({"Din": "55um", "H": "3.0um"})


def test_spiral_inductor_experiment_has_defaults_and_required_variables() -> None:
    exp = SpiralInductorExperiment.from_random(
        experiment_id="spiral_exp",
        count=2,
        seed=123,
        ranges=SpiralVariableRanges(),
    )

    assert exp.template_name == "si_template_v1"
    assert exp.design_name == "HFSSDesign1"
    assert exp.setup_name == "Setup1"
    assert exp.sweep_name == "Sweep"
    for case in exp.cases:
        assert set(case.variables) == {"Din", "H", "S", "W"}
        assert case.output_filename.endswith(".s2p")


def test_adapter_conversion_to_batch_job_is_stable() -> None:
    first = SpiralInductorExperiment.from_random(
        experiment_id="spiral_opt",
        count=3,
        seed=321,
        ranges=SpiralVariableRanges(),
    )
    second = SpiralInductorExperiment.from_random(
        experiment_id="spiral_opt",
        count=3,
        seed=321,
        ranges=SpiralVariableRanges(),
    )

    first_batch = to_batch_simulation_job(first)
    second_batch = to_batch_simulation_job(second)

    assert [c.case_id for c in first_batch.cases] == [c.case_id for c in second_batch.cases]
    assert [c.output_filename for c in first_batch.cases] == [
        c.output_filename for c in second_batch.cases
    ]
    assert first_batch.batch_id == "spiral_opt"
    assert first_batch.metrics_to_extract == ["max_s11_db", "min_s21_db"]


def test_adapter_metrics_follow_objectives_order_dedup() -> None:
    design_space = DesignSpace(
        variable_specs={
            "Din": VariableSpec(
                name="Din",
                unit="um",
                constraint_type="range",
                min_value=50.0,
                max_value=70.0,
            )
        }
    )
    exp = Experiment(
        experiment_id="exp_manual",
        template_name="si_template_v1",
        design_space=design_space,
        objectives=[
            ExperimentObjective(metric_name="max_s11_db", direction="minimize"),
            ExperimentObjective(metric_name="max_s11_db", direction="minimize"),
            ExperimentObjective(metric_name="min_s21_db", direction="maximize"),
        ],
        cases=[
            CaseVariant(
                case_id="c01",
                variables={"Din": "55um"},
                output_filename="exp_manual_c01.s2p",
            )
        ],
    )

    batch = to_batch_simulation_job(exp)
    assert batch.metrics_to_extract == ["max_s11_db", "min_s21_db"]
