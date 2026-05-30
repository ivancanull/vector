from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml

from worker.hfss_api import HfssSessionFactory
from worker.schemas import HfssSweepConfig
from worker.setup_designer import HfssSetupDesigner
from worker.template_registry import ResolvedTemplate
from worker.utils import WorkerError


REQUIRED_SPIRAL_VARIABLES = ("Din", "W", "S", "H")
FEED_EXTENSION_WIDTHS = 4
INNER_BRIDGE_CLEARANCE_UM = 2.0
SPIRAL_PORT_STRATEGY = "driven_modal_gap_ports_extended_feed"


@dataclass(frozen=True)
class SpiralVariable:
    name: str
    unit: str
    default: float
    min_value: float
    max_value: float
    optimize: bool

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any]) -> "SpiralVariable":
        for key in ("unit", "default", "min", "max"):
            if key not in data:
                raise ValueError(f"Variable {name} missing required field: {key}")
        default = float(data["default"])
        min_value = float(data["min"])
        max_value = float(data["max"])
        if min_value > max_value:
            raise ValueError(f"Variable {name} has invalid range: min > max")
        if not min_value <= default <= max_value:
            raise ValueError(f"Variable {name} default is outside its range")
        return cls(
            name=name,
            unit=str(data["unit"]),
            default=default,
            min_value=min_value,
            max_value=max_value,
            optimize=bool(data.get("optimize", True)),
        )

    def default_quantity(self) -> str:
        return f"{_fmt_number(self.default)}{self.unit}"


@dataclass(frozen=True)
class SpiralTemplateOutput:
    project_path: Path
    template_name: str
    design_name: str
    setup_name: str
    sweep_name: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SpiralTemplateOutput":
        for key in ("project_path", "template_name", "design_name", "setup_name", "sweep_name"):
            if key not in data:
                raise ValueError(f"template_output missing required field: {key}")
        return cls(
            project_path=Path(str(data["project_path"])),
            template_name=str(data["template_name"]),
            design_name=str(data["design_name"]),
            setup_name=str(data["setup_name"]),
            sweep_name=str(data["sweep_name"]),
        )


@dataclass(frozen=True)
class SpiralPdk:
    length_unit: str
    materials: dict[str, dict[str, Any]]
    stackup: dict[str, dict[str, Any]]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SpiralPdk":
        for key in ("length_unit", "materials", "stackup"):
            if key not in data:
                raise ValueError(f"pdk missing required field: {key}")
        return cls(
            length_unit=str(data["length_unit"]),
            materials=_as_mapping(data["materials"], "pdk.materials"),
            stackup=_as_mapping(data["stackup"], "pdk.stackup"),
        )


@dataclass(frozen=True)
class SpiralDevice:
    topology: str
    turns: int
    port_count: int
    port_style: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SpiralDevice":
        for key in ("topology", "turns", "port_count", "port_style"):
            if key not in data:
                raise ValueError(f"device missing required field: {key}")
        topology = str(data["topology"])
        if topology != "square_spiral":
            raise ValueError(f"Unsupported spiral topology: {topology}")
        turns = int(data["turns"])
        if turns <= 0:
            raise ValueError("device.turns must be positive")
        port_count = int(data["port_count"])
        if port_count != 2:
            raise ValueError("Only two-port spiral templates are supported")
        return cls(
            topology=topology,
            turns=turns,
            port_count=port_count,
            port_style=str(data["port_style"]),
        )


@dataclass(frozen=True)
class SpiralHfssDefinition:
    solution_type: str
    model_units: str
    solution_frequency: str
    air_box_padding_um: float
    radiation_boundary: bool
    sweep_type: str
    sweep_data: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SpiralHfssDefinition":
        for key in ("solution_type", "model_units", "solution_frequency", "air_box_padding_um", "sweep"):
            if key not in data:
                raise ValueError(f"hfss missing required field: {key}")
        sweep = _as_mapping(data["sweep"], "hfss.sweep")
        for key in ("sweep_type", "sweep_data"):
            if key not in sweep:
                raise ValueError(f"hfss.sweep missing required field: {key}")
        return cls(
            solution_type=str(data["solution_type"]),
            model_units=str(data["model_units"]),
            solution_frequency=str(data["solution_frequency"]),
            air_box_padding_um=float(data["air_box_padding_um"]),
            radiation_boundary=bool(data.get("radiation_boundary", True)),
            sweep_type=str(sweep["sweep_type"]),
            sweep_data=str(sweep["sweep_data"]),
        )

    def sweep_config(self) -> HfssSweepConfig:
        return HfssSweepConfig(sweep_type=self.sweep_type, sweep_data=self.sweep_data)


@dataclass(frozen=True)
class SquareSpiralEstimate:
    din_um: float
    w_um: float
    s_um: float
    turns: int
    f0_ghz: float
    dout_um: float
    davg_um: float
    fill_ratio: float
    inductance_h: float
    reactance_ohm_at_f0: float

    @property
    def inductance_nh(self) -> float:
        return self.inductance_h * 1e9

    def to_dict(self) -> dict[str, float | int]:
        return {
            "din_um": self.din_um,
            "w_um": self.w_um,
            "s_um": self.s_um,
            "turns": self.turns,
            "f0_ghz": self.f0_ghz,
            "dout_um": self.dout_um,
            "davg_um": self.davg_um,
            "fill_ratio": self.fill_ratio,
            "inductance_h": self.inductance_h,
            "inductance_nh": self.inductance_nh,
            "reactance_ohm_at_f0": self.reactance_ohm_at_f0,
        }


@dataclass(frozen=True)
class SpiralTemplateDeviceDefinition:
    schema_version: int
    template_output: SpiralTemplateOutput
    pdk: SpiralPdk
    device: SpiralDevice
    variables: dict[str, SpiralVariable]
    hfss: SpiralHfssDefinition

    @classmethod
    def load(cls, path: Path) -> "SpiralTemplateDeviceDefinition":
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_mapping(_as_mapping(data, str(path)))

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SpiralTemplateDeviceDefinition":
        for key in ("schema_version", "template_output", "pdk", "device", "variables", "hfss"):
            if key not in data:
                raise ValueError(f"Spiral device definition missing required key: {key}")
        variables_mapping = _as_mapping(data["variables"], "variables")
        variables = {
            name: SpiralVariable.from_mapping(name, _as_mapping(variables_mapping[name], f"variables.{name}"))
            for name in variables_mapping
        }
        missing = [name for name in REQUIRED_SPIRAL_VARIABLES if name not in variables]
        if missing:
            raise ValueError(f"Spiral definition missing required variables: {missing}")
        return cls(
            schema_version=int(data["schema_version"]),
            template_output=SpiralTemplateOutput.from_mapping(
                _as_mapping(data["template_output"], "template_output")
            ),
            pdk=SpiralPdk.from_mapping(_as_mapping(data["pdk"], "pdk")),
            device=SpiralDevice.from_mapping(_as_mapping(data["device"], "device")),
            variables=variables,
            hfss=SpiralHfssDefinition.from_mapping(_as_mapping(data["hfss"], "hfss")),
        )

    def nominal_variables(self) -> dict[str, str]:
        return {name: self.variables[name].default_quantity() for name in REQUIRED_SPIRAL_VARIABLES}

    def nominal_estimate(self, *, f0_ghz: float = 5.0) -> SquareSpiralEstimate:
        return estimate_square_spiral(
            din_um=self.variables["Din"].default,
            w_um=self.variables["W"].default,
            s_um=self.variables["S"].default,
            turns=self.device.turns,
            f0_ghz=f0_ghz,
        )

    def max_outer_diameter_um(self) -> float:
        return square_spiral_outer_diameter_um(
            din_um=self.variables["Din"].max_value,
            w_um=self.variables["W"].max_value,
            s_um=self.variables["S"].max_value,
            turns=self.device.turns,
        )

    def sweep_freq_range_ghz(self) -> tuple[float, float]:
        range_type, start, stop, _count = self.hfss.sweep_data.split()
        if range_type.upper() != "LINC":
            raise ValueError(f"Unsupported sweep_data range type: {range_type}")
        return _frequency_to_ghz(start), _frequency_to_ghz(stop)


def square_spiral_outer_diameter_um(*, din_um: float, w_um: float, s_um: float, turns: int) -> float:
    return din_um + 2.0 * (turns * w_um + (turns - 1) * s_um)


def estimate_square_spiral(
    *,
    din_um: float,
    w_um: float,
    s_um: float,
    turns: int,
    f0_ghz: float,
) -> SquareSpiralEstimate:
    dout_um = square_spiral_outer_diameter_um(din_um=din_um, w_um=w_um, s_um=s_um, turns=turns)
    din_m = din_um * 1e-6
    dout_m = dout_um * 1e-6
    davg_m = (dout_m + din_m) / 2.0
    rho = (dout_m - din_m) / (dout_m + din_m)
    mu0 = 4.0 * math.pi * 1e-7
    c1, c2, c3, c4 = 1.27, 2.07, 0.18, 0.13
    inductance_h = (
        mu0
        * turns**2
        * davg_m
        * c1
        / 2.0
        * (math.log(c2 / rho) + c3 * rho + c4 * rho**2)
    )
    f0_hz = f0_ghz * 1e9
    return SquareSpiralEstimate(
        din_um=din_um,
        w_um=w_um,
        s_um=s_um,
        turns=turns,
        f0_ghz=f0_ghz,
        dout_um=dout_um,
        davg_um=(dout_um + din_um) / 2.0,
        fill_ratio=rho,
        inductance_h=inductance_h,
        reactance_ohm_at_f0=2.0 * math.pi * f0_hz * inductance_h,
    )


@dataclass(frozen=True)
class SpiralGeometry:
    definition: SpiralTemplateDeviceDefinition

    def centerline_points(self) -> list[list[str]]:
        z = self.top_metal_center_z_expr()
        radii = [self.radius_expr(i) for i in reversed(range(self.definition.device.turns))]
        outer = radii[0]
        points: list[list[str]] = [
            [f"{outer} + {FEED_EXTENSION_WIDTHS}*W", _neg_expr(outer), z],
            [outer, _neg_expr(outer), z],
        ]

        for idx, radius in enumerate(radii):
            points.extend(
                [
                    [_neg_expr(radius), _neg_expr(radius), z],
                    [_neg_expr(radius), radius, z],
                    [radius, radius, z],
                ]
            )
            if idx + 1 < len(radii):
                next_radius = radii[idx + 1]
                points.append([radius, _neg_expr(next_radius), z])
        return points

    def radius_expr(self, turn_index_from_inner: int) -> str:
        if turn_index_from_inner == 0:
            return "(Din/2 + W/2)"
        return f"(Din/2 + W/2 + {turn_index_from_inner}*(W + S))"

    def top_metal_start_z_expr(self) -> str:
        top_metal = _as_mapping(self.definition.pdk.stackup["top_metal"], "pdk.stackup.top_metal")
        return f"{_fmt_number(float(top_metal['z_start_um']))}um"

    def top_metal_center_z_expr(self) -> str:
        return f"{self.top_metal_start_z_expr()} + H/2"


class SpiralTemplateBuilder:
    """Create a parameterized square spiral in a live PyAEDT HFSS project."""

    def __init__(
        self,
        definition: SpiralTemplateDeviceDefinition,
        *,
        setup_designer: HfssSetupDesigner | None = None,
    ) -> None:
        self.definition = definition
        self.geometry = SpiralGeometry(definition)
        self.setup_designer = setup_designer or HfssSetupDesigner()

    def dry_run_summary(self) -> dict[str, Any]:
        nominal = self.definition.nominal_estimate()
        return {
            "template": self.definition.template_output.template_name,
            "project_path": str(self.definition.template_output.project_path),
            "design_name": self.definition.template_output.design_name,
            "variables": self.definition.nominal_variables(),
            "turns": self.definition.device.turns,
            "max_outer_diameter_um": self.definition.max_outer_diameter_um(),
            "nominal_estimate": nominal.to_dict(),
            "centerline_points": self.geometry.centerline_points(),
        }

    def build(self, hfss) -> None:
        self._set_model_units(hfss)
        self._assign_design_variables(hfss)
        self._ensure_materials(hfss)
        self._create_stackup(hfss)
        self._create_spiral_trace(hfss)
        self._create_terminal_pads(hfss)
        self._create_inner_feed_bridge(hfss)
        self._unite_spiral_conductor(hfss)
        self._create_ports(hfss)
        self._create_air_box_and_boundary(hfss)
        self._ensure_setup_and_sweep(hfss)

    def _set_model_units(self, hfss) -> None:
        modeler = getattr(hfss, "modeler", None)
        if modeler is not None:
            modeler.model_units = self.definition.hfss.model_units

    def _assign_design_variables(self, hfss) -> None:
        for name in REQUIRED_SPIRAL_VARIABLES:
            hfss[name] = self.definition.variables[name].default_quantity()

    def _ensure_materials(self, hfss) -> None:
        materials = getattr(hfss, "materials", None)
        if materials is None:
            return
        for material in self.definition.pdk.materials.values():
            name = str(material["name"])
            mat_obj = self._get_or_create_material(materials, name)
            if mat_obj is None:
                continue
            if "relative_permittivity" in material:
                _set_optional_attr(mat_obj, "permittivity", float(material["relative_permittivity"]))
            if "loss_tangent" in material:
                _set_optional_attr(mat_obj, "dielectric_loss_tangent", float(material["loss_tangent"]))
            if "conductivity_s_per_m" in material:
                _set_optional_attr(mat_obj, "conductivity", float(material["conductivity_s_per_m"]))

    def _get_or_create_material(self, materials, name: str):
        exists = False
        checker = getattr(materials, "checkifmaterialexists", None)
        if callable(checker):
            exists = bool(checker(name))
        if exists and hasattr(materials, "__getitem__"):
            return materials[name]
        add_material = getattr(materials, "add_material", None)
        if callable(add_material) and not exists:
            return add_material(name)
        return None

    def _create_stackup(self, hfss) -> None:
        modeler = hfss.modeler
        lateral_um = self._model_lateral_size_um()
        half_um = lateral_um / 2.0
        for layer_name in ("substrate", "oxide"):
            layer = _as_mapping(self.definition.pdk.stackup[layer_name], f"pdk.stackup.{layer_name}")
            thickness_um = float(layer["thickness_um"])
            z_start_um = float(layer["z_start_um"])
            material_name = self._material_name(str(layer["material"]))
            modeler.create_box(
                [_um(-half_um), _um(-half_um), _um(z_start_um)],
                [_um(lateral_um), _um(lateral_um), _um(thickness_um)],
                name=_camel(layer_name),
                material=material_name,
            )

    def _create_ground_reference(self, hfss) -> None:
        modeler = hfss.modeler
        lateral_um = self._model_lateral_size_um()
        half_um = lateral_um / 2.0
        substrate = _as_mapping(self.definition.pdk.stackup["substrate"], "pdk.stackup.substrate")
        ground = modeler.create_rectangle(
            "XY",
            [_um(-half_um), _um(-half_um), _um(float(substrate["z_start_um"]))],
            [_um(lateral_um), _um(lateral_um)],
            name="Ground",
        )
        assign_perfect_e = getattr(hfss, "assign_perfecte_to_sheets", None)
        if callable(assign_perfect_e):
            assign_perfect_e(ground, name="PerfE_Ground")

    def _create_spiral_trace(self, hfss) -> None:
        modeler = hfss.modeler
        material = self._top_metal_material_name()
        z = self.geometry.top_metal_start_z_expr()
        points = self.geometry.centerline_points()
        for idx, (start, stop) in enumerate(zip(points, points[1:]), start=1):
            name = f"SpiralTraceSegment{idx}"
            if start[1] == stop[1]:
                x_min, x_max = self._ordered_expr_pair(start[0], stop[0])
                modeler.create_box(
                    [x_min, f"({start[1]}) - W/2", z],
                    [f"({x_max}) - ({x_min})", "W", "H"],
                    name=name,
                    material=material,
                )
            elif start[0] == stop[0]:
                y_min, y_max = self._ordered_expr_pair(start[1], stop[1])
                modeler.create_box(
                    [f"({start[0]}) - W/2", y_min, z],
                    ["W", f"({y_max}) - ({y_min})", "H"],
                    name=name,
                    material=material,
                )
            else:
                raise WorkerError("geometry_failure", f"Spiral segment is not axis-aligned: {start} -> {stop}")

    def _create_terminal_pads(self, hfss) -> None:
        modeler = hfss.modeler
        outer = self.geometry.radius_expr(self.definition.device.turns - 1)
        inner = self.geometry.radius_expr(0)
        z = self.geometry.top_metal_start_z_expr()
        material = self._top_metal_material_name()
        modeler.create_box(
            [outer, f"-({outer}) - W/2", z],
            [f"{FEED_EXTENSION_WIDTHS}*W", "W", "H"],
            name="OuterTerminalPad",
            material=material,
        )
        modeler.create_box([f"{inner} - W/2", f"{inner} - W/2", z], ["W", "W", "H"], name="InnerTerminalPad", material=material)

    def _create_inner_feed_bridge(self, hfss) -> None:
        modeler = hfss.modeler
        outer = self.geometry.radius_expr(self.definition.device.turns - 1)
        inner = self.geometry.radius_expr(0)
        z0 = self.geometry.top_metal_start_z_expr()
        bridge_z = f"{z0} + H + {_um(INNER_BRIDGE_CLEARANCE_UM)}"
        material = self._top_metal_material_name()
        modeler.create_box(
            [f"{inner} - W/2", f"{inner} - W/2", z0],
            ["W", "W", f"H + {_um(INNER_BRIDGE_CLEARANCE_UM)}"],
            name="InnerBridgeVia",
            material=material,
        )
        modeler.create_box(
            [inner, f"{inner} - W/2", bridge_z],
            [f"({outer} + {FEED_EXTENSION_WIDTHS}*W) - ({inner})", "W", "H"],
            name="InnerBridgeTrace",
            material=material,
        )

    def _unite_spiral_conductor(self, hfss) -> None:
        unite = getattr(hfss.modeler, "unite", None)
        if callable(unite):
            unite(
                [
                    *(f"SpiralTraceSegment{idx}" for idx in range(1, len(self.geometry.centerline_points()))),
                    "OuterTerminalPad",
                    "InnerTerminalPad",
                    "InnerBridgeVia",
                    "InnerBridgeTrace",
                ]
            )

    def _ordered_expr_pair(self, first: str, second: str) -> tuple[str, str]:
        first_value = self._nominal_expr_value(first)
        second_value = self._nominal_expr_value(second)
        return (first, second) if first_value <= second_value else (second, first)

    def _nominal_expr_value(self, expr: str) -> float:
        text = expr.replace("um", "")
        for name, variable in self.definition.variables.items():
            text = text.replace(name, str(variable.default))
        return float(eval(text, {"__builtins__": {}}, {}))

    def _create_ports(self, hfss) -> None:
        modeler = hfss.modeler
        outer = self.geometry.radius_expr(self.definition.device.turns - 1)
        inner = self.geometry.radius_expr(0)
        z = self.geometry.top_metal_start_z_expr()
        bridge_z = f"{z} + H + {_um(INNER_BRIDGE_CLEARANCE_UM)}"
        zc = self.geometry.top_metal_center_z_expr()
        bridge_zc = f"{bridge_z} + H/2"
        port1_line, port2_line = self._nominal_port_integration_lines()
        port1 = modeler.create_rectangle(
            "XY",
            [f"{outer} + {FEED_EXTENSION_WIDTHS}*W", f"{_neg_expr(outer)} - W/2", zc],
            ["W", "W"],
            name="Port1Sheet",
            material="vacuum",
        )
        modeler.create_box(
            [f"{outer} + {FEED_EXTENSION_WIDTHS + 1}*W", f"{_neg_expr(outer)} - W/2", z],
            [
                "W",
                f"({inner} + W/2) - ({_neg_expr(outer)} - W/2)",
                f"H + {_um(INNER_BRIDGE_CLEARANCE_UM)} + H",
            ],
            name="PortReferenceRail",
            material="pec",
        )
        port2 = modeler.create_rectangle(
            "XY",
            [f"{outer} + {FEED_EXTENSION_WIDTHS}*W", f"{inner} - W/2", bridge_zc],
            ["W", "W"],
            name="Port2Sheet",
            material="vacuum",
        )
        self._assign_lumped_port(
            hfss,
            name="Port1",
            sheet=port1,
            integration_line=port1_line,
        )
        self._assign_lumped_port(
            hfss,
            name="Port2",
            sheet=port2,
            integration_line=port2_line,
        )

    def _nominal_port_integration_lines(self) -> tuple[list[list[str]], list[list[str]]]:
        din = self.definition.variables["Din"].default
        width = self.definition.variables["W"].default
        spacing = self.definition.variables["S"].default
        height = self.definition.variables["H"].default
        turns = self.definition.device.turns
        outer = din / 2.0 + width / 2.0 + (turns - 1) * (width + spacing)
        inner = din / 2.0 + width / 2.0
        z0 = float(_as_mapping(self.definition.pdk.stackup["top_metal"], "pdk.stackup.top_metal")["z_start_um"])
        zc = z0 + height / 2.0
        bridge_zc = z0 + height + INNER_BRIDGE_CLEARANCE_UM + height / 2.0
        return (
            [
                [_um(outer + FEED_EXTENSION_WIDTHS * width), _um(-outer), _um(zc)],
                [_um(outer + (FEED_EXTENSION_WIDTHS + 1) * width), _um(-outer), _um(zc)],
            ],
            [
                [_um(outer + FEED_EXTENSION_WIDTHS * width), _um(inner), _um(bridge_zc)],
                [_um(outer + (FEED_EXTENSION_WIDTHS + 1) * width), _um(inner), _um(bridge_zc)],
            ],
        )

    def _assign_lumped_port(
        self,
        hfss,
        *,
        name: str,
        sheet,
        integration_line: list[list[str]],
        reference: str | None = None,
    ) -> None:
        solution_type = str(getattr(hfss, "solution_type", self.definition.hfss.solution_type))
        if "Terminal" in solution_type:
            lumped_port = getattr(hfss, "lumped_port", None)
            if not callable(lumped_port):
                raise WorkerError("geometry_failure", "PyAEDT HFSS object has no lumped_port method")
            lumped_port(
                assignment=sheet,
                reference=[reference] if reference else None,
                create_port_sheet=False,
                impedance=50,
                name=name,
            )
            return
        create_boundary = getattr(hfss, "_create_boundary", None)
        if callable(create_boundary):
            sheet_name = getattr(sheet, "name", sheet)
            create_boundary(
                name,
                {
                    "Objects": [str(sheet_name)],
                    "DoDeembed": False,
                    "RenormalizeAllTerminals": True,
                    "Modes": {
                        "Mode1": {
                            "ModeNum": 1,
                            "UseIntLine": True,
                            "IntLine": {"Start": integration_line[0], "End": integration_line[1]},
                            "AlignmentGroup": 0,
                            "CharImp": "Zpi",
                            "RenormImp": "50ohm",
                        }
                    },
                    "ShowReporterFilter": False,
                    "ReporterFilter": [True],
                    "Impedance": "50ohm",
                },
                "Lumped Port",
            )
            return
        lumped_port = getattr(hfss, "lumped_port", None)
        if not callable(lumped_port):
            raise WorkerError("geometry_failure", "PyAEDT HFSS object has no lumped_port method")
        lumped_port(
            assignment=sheet,
            create_port_sheet=False,
            port_on_plane=True,
            integration_line=integration_line,
            impedance=50,
            name=name,
        )

    def _create_air_box_and_boundary(self, hfss) -> None:
        modeler = hfss.modeler
        padding_um = self.definition.hfss.air_box_padding_um
        lateral_um = self._model_lateral_size_um()
        half_um = lateral_um / 2.0
        substrate = _as_mapping(self.definition.pdk.stackup["substrate"], "pdk.stackup.substrate")
        top_metal = _as_mapping(self.definition.pdk.stackup["top_metal"], "pdk.stackup.top_metal")
        z_min_um = float(substrate["z_start_um"]) - padding_um
        z_max_um = (
            float(top_metal["z_start_um"])
            + self.definition.variables["H"].max_value
            + padding_um
        )
        air_box = modeler.create_box(
            [_um(-half_um), _um(-half_um), _um(z_min_um)],
            [_um(lateral_um), _um(lateral_um), _um(z_max_um - z_min_um)],
            name="AirBox",
            material="air",
        )
        if self.definition.hfss.radiation_boundary:
            assign = getattr(hfss, "assign_radiation_boundary_to_objects", None)
            if callable(assign):
                assign(air_box, name="Rad1")

    def _ensure_setup_and_sweep(self, hfss) -> None:
        self.setup_designer.ensure_setup(
            hfss,
            setup_name=self.definition.template_output.setup_name,
            sweep_name=self.definition.template_output.sweep_name,
            sweep_config=self.definition.hfss.sweep_config(),
        )

    def _model_lateral_size_um(self) -> float:
        return self.definition.max_outer_diameter_um() + 2.0 * self.definition.hfss.air_box_padding_um

    def _material_name(self, stackup_material: str) -> str:
        for material in self.definition.pdk.materials.values():
            if str(material["name"]) == stackup_material:
                return stackup_material
        return stackup_material

    def _top_metal_material_name(self) -> str:
        top_metal = _as_mapping(self.definition.pdk.stackup["top_metal"], "pdk.stackup.top_metal")
        return self._material_name(str(top_metal["material"]))


class PyAEDTSpiralTemplateGenerator:
    def __init__(
        self,
        *,
        hfss_factory: "PyAEDTHfssFactory | None" = None,
        builder_factory: type[SpiralTemplateBuilder] = SpiralTemplateBuilder,
    ) -> None:
        self.hfss_factory = hfss_factory or PyAEDTHfssFactory()
        self.builder_factory = builder_factory

    def generate(
        self,
        definition: SpiralTemplateDeviceDefinition,
        *,
        non_graphical: bool = True,
        new_desktop: bool = False,
        close_desktop: bool = False,
        overwrite: bool = False,
    ) -> Path:
        project_path = definition.template_output.project_path
        if project_path.exists() and not overwrite:
            raise FileExistsError(f"Template already exists, pass --overwrite to replace: {project_path}")
        if overwrite:
            self._remove_existing_project_artifacts(project_path)
        import_check = getattr(self.hfss_factory, "import_hfss_class", None)
        if callable(import_check):
            import_check()
        project_path.parent.mkdir(parents=True, exist_ok=True)
        hfss = self.hfss_factory.open(
            definition,
            non_graphical=non_graphical,
            new_desktop=new_desktop,
        )
        try:
            self.builder_factory(definition).build(hfss)
            save_project = getattr(hfss, "save_project", None)
            if not callable(save_project):
                raise WorkerError("geometry_failure", "PyAEDT HFSS object has no save_project method")
            try:
                save_project(str(project_path))
            except TypeError:
                save_project()
            return project_path
        finally:
            release = getattr(hfss, "release_desktop", None)
            if callable(release):
                release(close_projects=False, close_desktop=close_desktop)

    def _remove_existing_project_artifacts(self, project_path: Path) -> None:
        for path in (
            project_path,
            project_path.with_name(project_path.name + ".auto"),
            project_path.with_name(project_path.name + ".lock"),
            project_path.with_name(project_path.name + "results"),
        ):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()


class PyAEDTHfssFactory:
    def import_hfss_class(self):
        try:
            return import_module("pyaedt").Hfss  # type: ignore[attr-defined]
        except ModuleNotFoundError:
            try:
                return import_module("ansys.aedt.core").Hfss  # type: ignore[attr-defined]
            except ModuleNotFoundError as exc:
                raise WorkerError("hfss_start_failure", f"PyAEDT import failure: {exc}") from exc

    def open(
        self,
        definition: SpiralTemplateDeviceDefinition,
        *,
        non_graphical: bool,
        new_desktop: bool,
    ):
        Hfss = self.import_hfss_class()
        kwargs = {
            "project": str(definition.template_output.project_path),
            "design": definition.template_output.design_name,
            "solution_type": definition.hfss.solution_type,
            "non_graphical": non_graphical,
        }
        try:
            return Hfss(**kwargs, new_desktop_session=new_desktop)
        except TypeError:
            return Hfss(**kwargs, new_desktop=new_desktop)


@dataclass(frozen=True)
class SpiralDebugMetrics:
    freq_range_ghz: tuple[float, float]
    l_at_target_nh: float | None
    q_at_target: float | None
    srf_ghz: float | None
    s21_at_target_db: float | None

    def to_json_dict(self, *, target_ghz: float) -> dict[str, float | None | list[float]]:
        return {
            "freq_range_ghz": [self.freq_range_ghz[0], self.freq_range_ghz[1]],
            f"l_at_{_fmt_number(target_ghz)}ghz_nh": self.l_at_target_nh,
            f"q_at_{_fmt_number(target_ghz)}ghz": self.q_at_target,
            f"s21_at_{_fmt_number(target_ghz)}ghz_db": self.s21_at_target_db,
            "srf_ghz": self.srf_ghz,
        }


class SpiralTouchstoneMetricsExtractor:
    def extract(self, touchstone_path: Path, *, target_ghz: float = 5.0) -> SpiralDebugMetrics:
        try:
            import numpy as np
            import skrf as rf
        except ModuleNotFoundError as exc:
            raise WorkerError("metric_extraction_failure", f"Missing Touchstone dependency: {exc}") from exc

        try:
            network = rf.Network(str(touchstone_path))
        except Exception as exc:
            raise WorkerError("metric_extraction_failure", f"Failed to parse Touchstone: {exc}") from exc
        if network.number_of_ports != 2:
            raise WorkerError("metric_extraction_failure", "Spiral debug extraction requires a 2-port Touchstone")

        freqs_hz = np.asarray(network.f, dtype=float)
        s_params = np.asarray(network.s, dtype=complex)
        z0 = _network_reference_impedance(network)
        eye = np.eye(2, dtype=complex)
        z_params = np.empty_like(s_params)
        try:
            for idx, s_matrix in enumerate(s_params):
                z_params[idx] = z0 * (eye + s_matrix) @ np.linalg.inv(eye - s_matrix)
        except np.linalg.LinAlgError as exc:
            raise WorkerError("metric_extraction_failure", f"Singular S-to-Z conversion: {exc}") from exc
        zdiff = z_params[:, 0, 0] + z_params[:, 1, 1] - z_params[:, 0, 1] - z_params[:, 1, 0]
        inductance_h = np.imag(zdiff) / (2.0 * np.pi * freqs_hz)
        q_values = np.imag(zdiff) / np.real(zdiff)
        target_idx = int(np.argmin(np.abs(freqs_hz - target_ghz * 1e9)))
        s21_mag = float(abs(s_params[target_idx, 1, 0]))
        s21_db = None if s21_mag <= 0.0 else 20.0 * math.log10(s21_mag)
        return SpiralDebugMetrics(
            freq_range_ghz=(float(freqs_hz[0] / 1e9), float(freqs_hz[-1] / 1e9)),
            l_at_target_nh=_finite_or_none(float(inductance_h[target_idx] * 1e9)),
            q_at_target=_finite_or_none(float(q_values[target_idx])),
            srf_ghz=_first_positive_to_negative_crossing_ghz(freqs_hz, np.imag(zdiff)),
            s21_at_target_db=_finite_or_none(s21_db) if s21_db is not None else None,
        )


class SpiralTemplateDebugger:
    def __init__(
        self,
        *,
        session_factory: HfssSessionFactory | None = None,
        setup_designer: HfssSetupDesigner | None = None,
        extractor: SpiralTouchstoneMetricsExtractor | None = None,
    ) -> None:
        self.session_factory = session_factory or HfssSessionFactory()
        self.setup_designer = setup_designer or HfssSetupDesigner()
        self.extractor = extractor or SpiralTouchstoneMetricsExtractor()

    def run(
        self,
        definition: SpiralTemplateDeviceDefinition,
        *,
        aedt_path: Path,
        output_touchstone: Path,
        target_ghz: float = 5.0,
        non_graphical: bool = True,
        new_desktop: bool = False,
        close_desktop: bool = False,
    ) -> dict[str, Any]:
        if not aedt_path.exists():
            raise FileNotFoundError(f"Generated AEDT template not found: {aedt_path}")
        output_touchstone.parent.mkdir(parents=True, exist_ok=True)
        session = None
        try:
            template = ResolvedTemplate(
                template_name=definition.template_output.template_name,
                project_path=aedt_path,
                design_name=definition.template_output.design_name,
                setup_name=definition.template_output.setup_name,
                sweep_name=definition.template_output.sweep_name,
                hfss_sweep=definition.hfss.sweep_config(),
            )
            cfg = type("Cfg", (), {"non_graphical": non_graphical, "new_desktop": new_desktop})()
            session = self.session_factory.open(template, cfg)
            session.apply_variables(definition.nominal_variables())
            resolved_setup = self.setup_designer.ensure_setup(
                session.hfss,
                setup_name=definition.template_output.setup_name,
                sweep_name=definition.template_output.sweep_name,
                sweep_config=definition.hfss.sweep_config(),
            )
            session.analyze(resolved_setup.setup_name)
            exported = session.export_touchstone(resolved_setup.setup_name, resolved_setup.sweep_name, output_touchstone)
            metrics = self.extractor.extract(exported, target_ghz=target_ghz)
            nominal = definition.nominal_estimate(f0_ghz=target_ghz)
            l_error_pct = None
            if metrics.l_at_target_nh is not None and nominal.inductance_nh != 0.0:
                l_error_pct = abs(metrics.l_at_target_nh - nominal.inductance_nh) / nominal.inductance_nh * 100.0
            result = {
                "status": "done",
                "template": definition.template_output.template_name,
                "aedt": str(aedt_path),
                "variables": definition.nominal_variables(),
                **metrics.to_json_dict(target_ghz=target_ghz),
                "l_error_pct": l_error_pct,
                "mohan_estimate_nh": nominal.inductance_nh,
                "port_strategy": SPIRAL_PORT_STRATEGY,
                "q_positive": metrics.q_at_target is not None and metrics.q_at_target > 0.0,
            }
            debug_json = output_touchstone.with_suffix(".json")
            debug_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            return result
        finally:
            if session is not None:
                session.close(close_desktop=close_desktop)


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _fmt_number(value: float) -> str:
    return f"{value:g}"


def _um(value: float) -> str:
    return f"{_fmt_number(value)}um"


def _neg_expr(value: str) -> str:
    return f"(-1*{value})"


def _camel(value: str) -> str:
    return "".join(part.capitalize() for part in value.split("_"))


def _set_optional_attr(obj, name: str, value: float) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _frequency_to_ghz(value: str) -> float:
    text = value.strip()
    lower = text.lower()
    if lower.endswith("ghz"):
        return float(text[:-3])
    if lower.endswith("mhz"):
        return float(text[:-3]) / 1000.0
    if lower.endswith("khz"):
        return float(text[:-3]) / 1e6
    if lower.endswith("hz"):
        return float(text[:-2]) / 1e9
    raise ValueError(f"Frequency value must include units: {value}")


def _network_reference_impedance(network) -> float:
    try:
        import numpy as np

        z0 = np.asarray(network.z0)
        return float(np.real(z0.reshape(-1)[0]))
    except Exception:
        return 50.0


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _first_positive_to_negative_crossing_ghz(freqs_hz, values) -> float | None:
    for idx in range(len(values) - 1):
        y0 = float(values[idx])
        y1 = float(values[idx + 1])
        if y0 > 0.0 and y1 <= 0.0:
            f0 = float(freqs_hz[idx])
            f1 = float(freqs_hz[idx + 1])
            if y1 == y0:
                return f0 / 1e9
            frac = y0 / (y0 - y1)
            return (f0 + frac * (f1 - f0)) / 1e9
    return None
