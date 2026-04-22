# PyAEDT Basic Tasks (HFSS)

This is a compact task checklist based on the official PyAEDT user guide and API docs, aligned with this repo's `worker` flow.

## 1) Open AEDT and attach to a project/design

Use `Hfss(...)` with project/design names and session options.

```python
from ansys.aedt.core import Hfss

hfss = Hfss(
    project=r"D:\ansys\spiral_inductor_example.aedt",
    design="HFSSDesign1",
    non_graphical=True,
    new_desktop=False,
    close_on_exit=False,
)
```

Repo mapping:
- `open_hfss_session(...)` in `src/worker/hfss_api.py`

## 2) Set design variables

For one explicit case (no parametric loop), apply `Din/H/S/W` before solving.

```python
hfss["Din"] = "55um"
hfss["H"] = "2um"
hfss["S"] = "1.5um"
hfss["W"] = "15um"
```

Repo mapping:
- `apply_variables(...)` in `src/worker/hfss_api.py`
- single-case variable defaults in `tests/worker/test_existing_aedt.py`

## 3) Solve one setup

Run one setup directly (for example `Setup1` or `Setup2`).

```python
ok = hfss.analyze_setup("Setup1")
```

Repo mapping:
- `run_analysis(...)` in `src/worker/hfss_api.py`

## 4) Export Touchstone

Export solved results from a specific `setup + sweep` pair.

```python
touchstone_path = hfss.export_touchstone(
    setup="Setup1",
    sweep="Sweep",
    output_file=r"D:\ansys\spiral_inductor_s_parameters_export\single_case.s2p",
)
```

Repo mapping:
- `export_touchstone(...)` in `src/worker/hfss_api.py`
- single-case output path generation in `src/worker/param_strings.py`

## 5) Close/release AEDT session

```python
hfss.release_desktop(close_projects=False, close_desktop=False)
```

Repo mapping:
- `close_hfss_session(...)` in `src/worker/hfss_api.py`

## 6) Debug run guidance for this repo

Run the single-case script with diagnostics:

```powershell
cd D:\vector
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe tests/worker/test_existing_aedt.py --debug
```

To target a setup/sweep explicitly:

```powershell
.\.venv\Scripts\python.exe tests/worker/test_existing_aedt.py --setup Setup1 --sweep Sweep --debug
```

## Reminder (Current Team Rule)

- Run one explicit parameter combination per execution.
- Do not run parametric sweep loops in the worker test script.
- Use `SetupX + SweepY` for export target naming, but execute a single case.

## KISS Rule (Keep It Simple and Stupid)

For this repo, prefer the simplest workflow that can validate HFSS end-to-end:

- Start with one project, one design, one setup, one sweep, one parameter combination.
- Use direct PyAEDT calls in scripts (`Hfss`, variable assignment, `setup.analyze`, `export_touchstone`).
- Avoid extra abstraction layers in quick-run scripts unless they reduce repeated code immediately.
- Keep CLI surface minimal: only include flags needed for daily runs.
- Fail fast on solve/export errors and print actionable AEDT messages.
- Only add loops or advanced orchestration after the single-case path is stable.

## Official Sources

- User guide index: https://aedt.docs.pyansys.com/version/stable/User_guide/index.html
- HFSS class (`Hfss`): https://aedt.docs.pyansys.com/version/stable/API/_autosummary/ansys.aedt.core.hfss.Hfss.html
- `export_touchstone`: https://aedt.docs.pyansys.com/version/stable/API/_autosummary/ansys.aedt.core.hfss.Hfss.export_touchstone.html
- CLI guide: https://aedt.docs.pyansys.com/version/stable/Getting_started/cli.html
- Installation: https://aedt.docs.pyansys.com/version/stable/Getting_started/Installation.html
