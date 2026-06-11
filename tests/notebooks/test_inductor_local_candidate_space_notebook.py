from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/inductor_local_candidate_space_run.ipynb")


def _notebook_source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources: list[str] = []
    for cell in payload["cells"]:
        sources.extend(cell.get("source", []))
    return "".join(sources)


def test_inductor_notebook_uses_local_queue_worker_not_direct_hfss() -> None:
    source = _notebook_source()

    assert "DiscreteCandidateSpace" in source
    assert "run_local_queue_worker.py" in source
    assert "D:/ansys/spiral_inductor_s_parameters_export/ext" in source
    assert "build_spiral_inductor_export_filename" in source
    assert "spiral_inductor_t3_template_v1" in source
    assert "Hfss(" not in source
    assert "HfssJobExecutor" not in source
    assert "ansys.aedt" not in source


def test_inductor_notebook_defaults_to_dry_worker_mode() -> None:
    source = _notebook_source()

    assert "RUN_WORKER = False" in source
    assert "'copy_template_per_job': True" in source
    assert "'results_dir': str(EXPORT_DIR)" in source


def test_inductor_notebook_uses_shared_runtime_queue() -> None:
    source = _notebook_source()

    assert "RUNTIME_ROOT = REPO_ROOT / '.runtime'" in source
    assert "QUEUE_ROOT = RUNTIME_ROOT / 'queue'" in source
    assert "worker_configs" in source
    assert "QUEUE_ROOT / 'results'" in source


def test_inductor_notebook_uses_snapped_resolution() -> None:
    source = _notebook_source()

    assert "DEFAULT_INDUCTOR_RESOLUTION_UM" in source
    assert "normalize_inductor_resolution" in source
    assert "resolution_um=RESOLUTION_UM" in source
    assert "candidate_df['resolution_um']" in source
    assert "np.allclose(candidate_df['Din'] / RESOLUTION_UM['Din']" in source


def test_inductor_notebook_does_not_import_optimization_inductor() -> None:
    source = _notebook_source()

    assert ("optimization" + ".inductor") not in source
