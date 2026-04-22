from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:  # Allow direct script usage without pytest installed.
    pytest = None

from worker.hfss_api import (
    apply_variables,
    close_hfss_session,
    export_touchstone,
    open_hfss_session,
    run_analysis,
)
from worker.param_strings import build_spiral_inductor_export_path


DEFAULT_PROJECT = Path(r"D:\ansys\spiral_inductor_example.aedt")
DEFAULT_DESIGN = "HFSSDesign1"
DEFAULT_SETUP = "Setup1"
DEFAULT_SWEEP = "Sweep"
DEFAULT_OUTPUT_DIR = Path(r"D:\ansys\spiral_inductor_s_parameters_export")
DEFAULT_DIN = 55
DEFAULT_H = 2
DEFAULT_S = 1.5
DEFAULT_W = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test an existing HFSS .aedt project by solving and exporting Touchstone."
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--design", default=DEFAULT_DESIGN)
    parser.add_argument("--setup", default=DEFAULT_SETUP)
    parser.add_argument("--sweep", default=DEFAULT_SWEEP)
    parser.add_argument("--Din", type=float, default=DEFAULT_DIN)
    parser.add_argument("--H", type=float, default=DEFAULT_H)
    parser.add_argument("--S", type=float, default=DEFAULT_S)
    parser.add_argument("--W", type=float, default=DEFAULT_W)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--graphical", action="store_true")
    parser.add_argument("--new-desktop", action="store_true")
    parser.add_argument("--close-desktop", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra HFSS diagnostics around solve/export failures.",
    )
    return parser.parse_args()


def build_variables(Din: int | float, H: int | float, S: int | float, W: int | float) -> dict[str, str]:
    return {
        "Din": f"{Din}um",
        "H": f"{H}um",
        "S": f"{S}um",
        "W": f"{W}um",
    }


def _best_effort_hfss_debug_dump(hfss, setup_name: str, sweep_name: str) -> None:
    print("[debug] ---- HFSS diagnostics ----")
    print(f"[debug] requested setup: {setup_name}")
    print(f"[debug] requested sweep: {sweep_name}")
    try:
        active_design_name = getattr(hfss, "design_name", None)
        print(f"[debug] active design: {active_design_name}")
    except Exception as exc:
        print(f"[debug] failed to read active design: {exc}")
    try:
        existing_setups = getattr(hfss, "existing_analysis_setups", None)
        if existing_setups is not None:
            print(f"[debug] existing analysis setups: {existing_setups}")
    except Exception as exc:
        print(f"[debug] failed to read existing setups: {exc}")
    try:
        setup_names = [s.name for s in getattr(hfss, "setups", []) if hasattr(s, "name")]
        if setup_names:
            print(f"[debug] setup objects: {setup_names}")
    except Exception as exc:
        print(f"[debug] failed to read setup objects: {exc}")
    try:
        messages = hfss.odesktop.GetMessages("", "", 2)
        if messages:
            print("[debug] recent AEDT messages (severity>=warning):")
            for line in messages[-20:]:
                print(f"[debug]   {line}")
        else:
            print("[debug] no warning/error messages returned by AEDT")
    except Exception as exc:
        print(f"[debug] failed to read AEDT messages: {exc}")
    print("[debug] --------------------------")


def run_existing_aedt_test(
    project_path: Path,
    design_name: str,
    setup_name: str,
    sweep_name: str,
    output_path: Path,
    *,
    Din: int | float,
    H: int | float,
    S: int | float,
    W: int | float,
    debug: bool = False,
    graphical: bool = False,
    new_desktop: bool = False,
    close_desktop: bool = False,
) -> Path:
    project_path = project_path.resolve()
    output_path = output_path.resolve()
    if not project_path.exists():
        raise FileNotFoundError(f"Project file not found: {project_path}")

    hfss = None
    try:
        print(f"Opening project: {project_path}")
        print(f"Design: {design_name}")
        print(f"Setup: {setup_name}")
        print(f"Sweep: {sweep_name}")
        print(f"Single-case parameters: Din={Din}um H={H}um S={S}um W={W}um")
        hfss = open_hfss_session(
            project_path=str(project_path),
            design_name=design_name,
            non_graphical=not graphical,
            new_desktop=new_desktop,
        )
        variables = build_variables(Din=Din, H=H, S=S, W=W)
        print(f"Applying variables: {variables}")
        apply_variables(hfss, variables)
        try:
            run_analysis(hfss, setup_name)
        except Exception as exc:
            if debug:
                print(f"[debug] solve raised: {exc}")
                print("[debug] python traceback:")
                print(traceback.format_exc())
                _best_effort_hfss_debug_dump(hfss, setup_name, sweep_name)
            raise
        try:
            exported_path = Path(export_touchstone(hfss, setup_name, sweep_name, output_path))
        except Exception as exc:
            if debug:
                print(f"[debug] export raised: {exc}")
                print("[debug] python traceback:")
                print(traceback.format_exc())
                _best_effort_hfss_debug_dump(hfss, setup_name, sweep_name)
            raise
        print(f"Export complete: {exported_path}")
        return exported_path
    finally:
        if hfss is not None:
            close_hfss_session(hfss, close_desktop=close_desktop)


def test_existing_aedt_smoke() -> None:
    if pytest is None:
        raise RuntimeError("pytest is required to run this file as a pytest test")
    if os.environ.get("RUN_HFSS_SMOKE_TEST") != "1":
        pytest.skip("Set RUN_HFSS_SMOKE_TEST=1 to run the live HFSS smoke test.")
    exported = run_existing_aedt_test(
        project_path=DEFAULT_PROJECT,
        design_name=DEFAULT_DESIGN,
        setup_name=DEFAULT_SETUP,
        sweep_name=DEFAULT_SWEEP,
        output_path=Path(
            build_spiral_inductor_export_path(
                DEFAULT_DIN,
                DEFAULT_S,
                DEFAULT_W,
                output_dir=str(DEFAULT_OUTPUT_DIR),
            )
        ),
        Din=DEFAULT_DIN,
        H=DEFAULT_H,
        S=DEFAULT_S,
        W=DEFAULT_W,
        debug=True,
    )
    assert exported.exists(), f"Expected Touchstone export to exist: {exported}"


if __name__ == "__main__":
    args = parse_args()
    output_path = (
        args.output
        if args.output is not None
        else Path(
            build_spiral_inductor_export_path(
                args.Din,
                args.S,
                args.W,
                output_dir=str(DEFAULT_OUTPUT_DIR),
            )
        )
    )
    exported = run_existing_aedt_test(
        project_path=args.project,
        design_name=args.design,
        setup_name=args.setup,
        sweep_name=args.sweep,
        output_path=output_path,
        Din=args.Din,
        H=args.H,
        S=args.S,
        W=args.W,
        debug=args.debug,
        graphical=args.graphical,
        new_desktop=args.new_desktop,
        close_desktop=args.close_desktop,
    )
    print(exported)
