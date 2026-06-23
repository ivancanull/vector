# Real/Fake Simulation Interface

## Purpose

The BNN online optimization loop needs one stable way to request a simulation:

1. propose geometry parameters,
2. call a simulator,
3. receive in-memory `geo_params`, `freqs`, and complex `s_params`,
4. append the result to the online dataset and continue training.

The optimization loop should not know whether the sample came from HFSS or from already-generated data in `data/`.

## Core API

The optimization-facing API lives in `src/simulation/`.

```python
from simulation import SimulationInput, Simulator

output = simulator.simulate(
    SimulationInput(
        case_id="candidate_001",
        variables={"Din": 55.0, "S": 1.5, "W": 15.0},
        variable_order=["Din", "S", "W"],
    )
)
```

### `SimulationInput`

- `case_id`: stable identifier for the candidate.
- `variables`: unitless numeric design variables.
- `variable_order`: explicit ordering used to convert variables into `geo_params`.

### `SimulationOutput`

- `case_id`: copied from the request.
- `geo_params`: numeric vector in `variable_order`.
- `freqs`: frequency vector in Hz.
- `s_params`: complex ndarray with shape `(port_num, port_num, freq_num)`.
- `artifacts`: optional artifact paths, such as exported Touchstone files.

Use `output.to_dataset_item()` to convert a simulation result into an `SPOnlineDataset`-compatible item:

```python
{
    "caseid": output.case_id,
    "geo_params": output.geo_params,
    "s_params": output.s_params,
    "sampled_indices": None,
}
```

## Fake Simulator

`DatasetSimulator` is the virtual simulator. It reads existing generated data and returns exact matching samples.

```python
from simulation import DatasetSimulator

simulator = DatasetSimulator(
    dataset_path="data/inductor_bnn_online_uaw_uaw_dataset.pkl",
    freqs=freqs,
    port_num=2,
    param_start_idx=4,
    variable_order=["Din", "S", "W"],
)
```

Behavior:

- loads either an online dataset pickle or a directory of Touchstone files,
- converts `SimulationInput.variables` into `geo_params`,
- finds an existing generated sample using `np.allclose`,
- returns the sample as `SimulationOutput`,
- raises `SimulationError(failure_class="sample_not_found")` if no generated sample matches.

It intentionally does not interpolate or extrapolate. This keeps fake simulation honest: it only returns data that already exists.

The demo notebook is:

```text
notebooks/fake_simulation_demo.ipynb
```

It supports:

- `filter`
- `inductor`
- `line`

and plots S-parameter magnitude and phase from the fake simulator.

## Real HFSS Simulator

`RealHfssSimulator` is the real simulator adapter. It wraps the existing worker-side HFSS executor, but exposes the same optimization-level `simulate()` API.

Responsibilities:

- convert unitless numeric variables into HFSS strings using `unit_map`,
- build a single `JobRequest`,
- synchronously call the injected HFSS executor,
- parse the returned Touchstone artifact,
- return `SimulationOutput`.

Example construction shape:

```python
from simulation import RealHfssSimulator

simulator = RealHfssSimulator(
    template_cfg=resolved_template,
    worker_cfg=worker_config,
    port_num=2,
    unit_map={"Din": "um", "S": "um", "W": "um"},
    variable_order=["Din", "S", "W"],
)
```

The real simulator reuses:

- `worker.hfss_api.HfssJobExecutor`
- `worker.schemas.JobRequest`
- `worker.template_registry.ResolvedTemplate`
- `complex_func.FunctionFromTouchstone`

The BNN optimizer does not call these worker/HFSS classes directly.

## Design Boundary

There are two levels in the system:

- Optimization level: `Simulator`, `DatasetSimulator`, `RealHfssSimulator`.
- Worker level: queue jobs, HFSS sessions, artifact export, metrics, result JSONs.

The optimization level is synchronous and simple:

```text
BNN optimizer -> Simulator.simulate() -> SimulationOutput
```

The worker level can still be asynchronous and queue-based:

```text
server queue -> Windows worker -> HFSS -> Touchstone/result JSON
```

`RealHfssSimulator` is the adapter between these levels.

## Current Tests

Simulator tests are in:

```text
tests/simulation/test_simulators.py
```

Covered scenarios:

- fake simulator returns a matching generated sample,
- fake simulator raises a clear error when no sample matches,
- real simulator formats HFSS variables with units,
- real simulator builds the expected job request,
- real simulator parses a Touchstone result through a fake executor.

Run them with:

```bash
PYTHONPATH=src python -m pytest -q tests/simulation/test_simulators.py
```

