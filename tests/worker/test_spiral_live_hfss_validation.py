from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import pytest

from worker.spiral_template_builder import (
    PyAEDTSpiralTemplateGenerator,
    SpiralTemplateDebugger,
    SpiralTemplateDeviceDefinition,
)


DEVICE_DEFINITION = Path("configs/spiral_inductor_template_device_v1.yaml")


def test_live_spiral_template_generate_solve_export_and_metrics(tmp_path: Path) -> None:
    if os.environ.get("RUN_AEDT_LIVE_TESTS") != "1":
        pytest.skip("Set RUN_AEDT_LIVE_TESTS=1 to run live AEDT/HFSS validation.")

    pytest.importorskip("ansys.aedt.core")
    pytest.importorskip("skrf")

    project_path = tmp_path / "spiral_live_validation.aedt"
    touchstone_path = tmp_path / "spiral_live_validation.s2p"
    definition = SpiralTemplateDeviceDefinition.load(DEVICE_DEFINITION)
    definition = replace(
        definition,
        template_output=replace(definition.template_output, project_path=project_path),
    )

    generated = PyAEDTSpiralTemplateGenerator().generate(
        definition,
        overwrite=True,
        close_desktop=False,
    )
    assert generated == project_path

    result = SpiralTemplateDebugger().run(
        definition,
        aedt_path=project_path,
        output_touchstone=touchstone_path,
        target_ghz=5.0,
        close_desktop=False,
    )

    assert result["status"] == "done"
    assert touchstone_path.exists()
    assert result["freq_range_ghz"][0] <= 5.0 <= result["freq_range_ghz"][1]
    assert result["l_at_5ghz_nh"] is not None
    assert result["q_positive"] in (True, False)
