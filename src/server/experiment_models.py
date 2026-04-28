from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import random
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from common.units import has_explicit_unit

_UNIT_VALUE_PATTERN = re.compile(
    r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-zA-Z][a-zA-Z0-9/_-]*)\s*$"
)

ConstraintType = Literal["range", "enum"]
ObjectiveDirection = Literal["minimize", "maximize"]
CrossFieldValidator = Callable[[dict[str, float]], None]


def _parse_unit_value(value: str) -> tuple[float, str]:
    match = _UNIT_VALUE_PATTERN.match(value)
    if not match:
        raise ValueError(f"Invalid unit value: {value!r}")
    return float(match.group(1)), match.group(2)


class VariableSpec(BaseModel):
    name: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    constraint_type: ConstraintType
    min_value: float | None = None
    max_value: float | None = None
    options: list[float] | None = None

    @model_validator(mode="after")
    def validate_constraint(self) -> "VariableSpec":
        if self.constraint_type == "range":
            if self.min_value is None or self.max_value is None:
                raise ValueError("range VariableSpec requires min_value and max_value")
            if self.min_value > self.max_value:
                raise ValueError("min_value must be <= max_value")
            if self.options:
                raise ValueError("range VariableSpec must not set options")
        elif self.constraint_type == "enum":
            if not self.options:
                raise ValueError("enum VariableSpec requires non-empty options")
            if self.min_value is not None or self.max_value is not None:
                raise ValueError("enum VariableSpec must not set min_value/max_value")
        return self

    def format_value(self, numeric_value: float, precision: int = 2) -> str:
        return f"{numeric_value:.{precision}f}{self.unit}"

    def validate_unit_value(self, value: str) -> float:
        if not has_explicit_unit(value):
            raise ValueError(f"{self.name} must include explicit units")
        numeric, unit = _parse_unit_value(value)
        if unit != self.unit:
            raise ValueError(
                f"{self.name} must use unit '{self.unit}', got '{unit}' in {value!r}"
            )

        if self.constraint_type == "range":
            assert self.min_value is not None and self.max_value is not None
            if numeric < self.min_value or numeric > self.max_value:
                raise ValueError(
                    f"{self.name}={numeric}{self.unit} is outside range "
                    f"[{self.min_value}, {self.max_value}]"
                )
        else:
            assert self.options is not None
            if not any(abs(numeric - option) <= 1e-12 for option in self.options):
                raise ValueError(
                    f"{self.name}={numeric}{self.unit} is not in enum options {self.options}"
                )

        return numeric


class DesignSpace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    variable_specs: dict[str, VariableSpec]
    cross_field_validators: list[CrossFieldValidator] = Field(default_factory=list)

    @field_validator("variable_specs")
    @classmethod
    def validate_variable_specs(cls, value: dict[str, VariableSpec]) -> dict[str, VariableSpec]:
        if not value:
            raise ValueError("variable_specs must be non-empty")
        for name, spec in value.items():
            if spec.name != name:
                raise ValueError(f"VariableSpec key/name mismatch: {name!r} != {spec.name!r}")
        return value

    def validate_case_variables(self, variables: dict[str, str]) -> dict[str, float]:
        missing = set(self.variable_specs) - set(variables)
        extra = set(variables) - set(self.variable_specs)
        if missing:
            raise ValueError(f"Missing variables: {sorted(missing)}")
        if extra:
            raise ValueError(f"Unexpected variables: {sorted(extra)}")

        parsed: dict[str, float] = {}
        for name, spec in self.variable_specs.items():
            parsed[name] = spec.validate_unit_value(variables[name])

        for validate in self.cross_field_validators:
            validate(parsed)
        return parsed


class CaseVariant(BaseModel):
    case_id: str = Field(min_length=1)
    variables: dict[str, str]
    output_filename: str = Field(min_length=1)

    @field_validator("variables")
    @classmethod
    def validate_variables_have_units(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("variables must be non-empty")
        for name, item in value.items():
            if not isinstance(item, str) or not has_explicit_unit(item):
                raise ValueError(f"Case variable {name!r} must include explicit units")
        return value

    @field_validator("output_filename")
    @classmethod
    def validate_output_filename(cls, value: str) -> str:
        suffix = value.lower().rsplit(".", 1)[-1] if "." in value else ""
        if not (suffix.startswith("s") and suffix.endswith("p")):
            raise ValueError("output_filename must be Touchstone style (.sNp)")
        return value


class ExperimentObjective(BaseModel):
    metric_name: str = Field(min_length=1)
    direction: ObjectiveDirection
    target: float | None = None
    weight: float = Field(default=1.0, gt=0)


class Experiment(BaseModel):
    experiment_id: str = Field(min_length=1)
    template_name: str = Field(min_length=1)
    design_name: str | None = None
    setup_name: str | None = None
    sweep_name: str | None = None
    design_space: DesignSpace
    objectives: list[ExperimentObjective]
    cases: list[CaseVariant]
    artifact_options: dict[str, bool | None] = Field(default_factory=dict)
    retry_count: int = Field(default=0, ge=0)

    @field_validator("objectives")
    @classmethod
    def validate_objectives(cls, value: list[ExperimentObjective]) -> list[ExperimentObjective]:
        if not value:
            raise ValueError("objectives must be non-empty")
        return value

    @field_validator("cases")
    @classmethod
    def validate_cases_non_empty(cls, value: list[CaseVariant]) -> list[CaseVariant]:
        if not value:
            raise ValueError("cases must be non-empty")
        return value

    def validate_cases_against_design_space(self) -> None:
        seen_case_ids: set[str] = set()
        seen_outputs: set[str] = set()
        for case in self.cases:
            if case.case_id in seen_case_ids:
                raise ValueError(f"Duplicate case_id: {case.case_id}")
            if case.output_filename in seen_outputs:
                raise ValueError(f"Duplicate output_filename: {case.output_filename}")
            seen_case_ids.add(case.case_id)
            seen_outputs.add(case.output_filename)
            self.design_space.validate_case_variables(case.variables)

    def metrics_to_extract(self) -> list[str]:
        # Preserve declaration order while removing duplicates.
        return list(dict.fromkeys(obj.metric_name for obj in self.objectives))


@dataclass(frozen=True)
class SpiralVariableRanges:
    din_min_um: float = 50.0
    din_max_um: float = 70.0
    s_min_um: float = 1.0
    s_max_um: float = 2.0
    w_min_um: float = 12.0
    w_max_um: float = 18.0
    h_um: float = 2.0

    def validate(self) -> None:
        if self.din_min_um > self.din_max_um:
            raise ValueError("din_min_um must be <= din_max_um")
        if self.s_min_um > self.s_max_um:
            raise ValueError("s_min_um must be <= s_max_um")
        if self.w_min_um > self.w_max_um:
            raise ValueError("w_min_um must be <= w_max_um")
        if self.h_um <= 0:
            raise ValueError("h_um must be > 0")


class SpiralInductorExperiment(Experiment):
    @staticmethod
    def _spiral_cross_rule(parsed: dict[str, float]) -> None:
        # Keep a physically sane geometry margin for Din relative to width/spacing.
        # Din > 2 * (W + S) ensures inner opening remains positive.
        if parsed["Din"] <= 2.0 * (parsed["W"] + parsed["S"]):
            raise ValueError("Spiral geometry invalid: Din must be > 2*(W+S)")

    @staticmethod
    def _filename_number(value: float) -> str:
        return f"{value:.2f}".replace(".", "d")

    @classmethod
    def default_design_space(cls, ranges: SpiralVariableRanges) -> DesignSpace:
        ranges.validate()
        return DesignSpace(
            variable_specs={
                "Din": VariableSpec(
                    name="Din",
                    unit="um",
                    constraint_type="range",
                    min_value=ranges.din_min_um,
                    max_value=ranges.din_max_um,
                ),
                "H": VariableSpec(
                    name="H",
                    unit="um",
                    constraint_type="enum",
                    options=[ranges.h_um],
                ),
                "S": VariableSpec(
                    name="S",
                    unit="um",
                    constraint_type="range",
                    min_value=ranges.s_min_um,
                    max_value=ranges.s_max_um,
                ),
                "W": VariableSpec(
                    name="W",
                    unit="um",
                    constraint_type="range",
                    min_value=ranges.w_min_um,
                    max_value=ranges.w_max_um,
                ),
            },
            cross_field_validators=[cls._spiral_cross_rule],
        )

    @classmethod
    def from_random(
        cls,
        *,
        experiment_id: str,
        count: int,
        seed: int | None,
        ranges: SpiralVariableRanges,
        template_name: str = "si_template_v1",
        design_name: str = "HFSSDesign1",
        setup_name: str = "Setup1",
        sweep_name: str = "Sweep",
    ) -> "SpiralInductorExperiment":
        if count <= 0:
            raise ValueError("count must be positive")
        design_space = cls.default_design_space(ranges)
        rng = random.Random(seed)
        cases: list[CaseVariant] = []
        for index in range(1, count + 1):
            din = rng.uniform(ranges.din_min_um, ranges.din_max_um)
            spacing = rng.uniform(ranges.s_min_um, ranges.s_max_um)
            width = rng.uniform(ranges.w_min_um, ranges.w_max_um)
            case_id = f"c{index:02d}"
            output_filename = (
                f"{experiment_id}_{case_id}_"
                f"Din{cls._filename_number(din)}_"
                f"S{cls._filename_number(spacing)}_"
                f"W{cls._filename_number(width)}.s2p"
            )
            cases.append(
                CaseVariant(
                    case_id=case_id,
                    variables={
                        "Din": f"{din:.2f}um",
                        "H": f"{ranges.h_um:.2f}um",
                        "S": f"{spacing:.2f}um",
                        "W": f"{width:.2f}um",
                    },
                    output_filename=output_filename,
                )
            )

        experiment = cls(
            experiment_id=experiment_id,
            template_name=template_name,
            design_name=design_name,
            setup_name=setup_name,
            sweep_name=sweep_name,
            design_space=design_space,
            objectives=[
                ExperimentObjective(metric_name="max_s11_db", direction="minimize"),
                ExperimentObjective(metric_name="min_s21_db", direction="maximize"),
            ],
            cases=cases,
            artifact_options={"export_touchstone": True, "export_csv": False},
        )
        experiment.validate_cases_against_design_space()
        return experiment
