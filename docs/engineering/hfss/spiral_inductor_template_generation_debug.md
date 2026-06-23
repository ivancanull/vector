# Spiral Inductor Template Generation and Debug

## Purpose

This document defines the worker-side flow for generating a parameterized
spiral inductor AEDT template and validating it with one nominal HFSS
simulation before using it in online optimization.

The intended workflow is:

```text
device definition YAML
  -> worker-side PyAEDT template generator
  -> generated AEDT template
  -> one nominal debug simulation
  -> template registry entry
  -> BO/LHS/BNN online optimization
```

The optimizer must not call PyAEDT directly. Template generation and debug
simulation belong on the worker side.

## Inputs

Primary device definition:

```text
configs/spiral_inductor_template_device_v1.yaml
```

This file defines:

- generic silicon PDK and stackup
- square spiral topology
- fixed `turns = 3`
- template output path and design/setup/sweep names
- worker-compatible variables: `Din`, `W`, `S`, `H`
- HFSS solution frequency and sweep: `5GHz`, `LINC 0.1GHz 30GHz 601`

The generic PDK values are for pipeline validation only. They are not a
foundry-qualified process.

## Step 1: Generate the AEDT Template

Future implementation target:

```text
src/worker/spiral_template_builder.py
scripts/worker/generate_spiral_template.py
```

The generator should run on the Windows HFSS worker machine:

```powershell
python scripts\worker\generate_spiral_template.py `
  --device-definition configs\spiral_inductor_template_device_v1.yaml
```

Expected output:

```text
D:/ansys/spiral_inductor_template_v1.aedt
```

The generated AEDT must be a parameterized template, not a one-case geometry.
It must expose these HFSS design variables:

```text
Din
W
S
H
```

The first version should fix:

```text
topology = square_spiral
turns = 3
port_style = two_port_lumped
```

The generator must create:

- substrate, oxide, and top-metal layers from the YAML stackup
- square spiral top metal using `Din`, `W`, `S`, and fixed turns
- two ports suitable for `.s2p` export
- air box and radiation boundary
- `Setup1`
- `Sweep`
- Touchstone export compatibility for 2 ports

## Step 2: Template Build Checks

After generation, inspect the AEDT with PyAEDT and fail fast if any required
object is missing.

Required checks:

- AEDT file exists at `template_output.project_path`
- design exists: `SpiralInductorT3`
- variables exist: `Din`, `W`, `S`, `H`
- setup exists: `Setup1`
- sweep exists: `Sweep`
- port count is 2
- model has an air/radiation boundary
- model units are `um`

Do not register the template for online optimization until these checks pass.

## Step 3: Run One Nominal Debug Simulation

Future implementation target:

```text
scripts/worker/debug_spiral_template.py
```

Run one nominal simulation on the worker:

```powershell
python scripts\worker\debug_spiral_template.py `
  --device-definition configs\spiral_inductor_template_device_v1.yaml `
  --aedt D:/ansys/spiral_inductor_template_v1.aedt `
  --output D:/vector/queue/results/spiral_template_nominal_debug.s2p
```

Nominal variables:

```text
Din = 60um
W   = 15um
S   = 1.5um
H   = 2um
```

The debug script should:

1. Open the generated AEDT.
2. Apply nominal variables.
3. Analyze `Setup1`.
4. Export `Sweep` to `.s2p`.
5. Parse the Touchstone file.
6. Extract `L(5GHz)`, `Q(5GHz)`, and SRF.
7. Compare `L(5GHz)` against a Mohan/current-sheet estimate.
8. Write a debug JSON next to the `.s2p`.

Expected debug artifacts:

```text
D:/vector/queue/results/spiral_template_nominal_debug.s2p
D:/vector/queue/results/spiral_template_nominal_debug.json
```

Recommended debug JSON shape:

```json
{
  "status": "done",
  "template": "spiral_inductor_t3_template_v1",
  "aedt": "D:/ansys/spiral_inductor_template_v1.aedt",
  "variables": {
    "Din": "60um",
    "W": "15um",
    "S": "1.5um",
    "H": "2um"
  },
  "freq_range_ghz": [0.1, 30.0],
  "l_at_5ghz_nh": 1.1,
  "q_at_5ghz": 8.4,
  "srf_ghz": 18.7,
  "mohan_estimate_nh": 1.27
}
```

## Step 4: Metric Sanity Criteria

Use the bundled Codex skill `spiral-inductor-estimator` or the same Mohan
formula to compute the nominal estimate:

```text
Din = 60um, W = 15um, S = 1.5um, turns = 3
Mohan estimate: L ~= 1.27 nH
```

Acceptance criteria for the first debug simulation:

- `.s2p` exists and parses successfully
- Touchstone has 2 ports
- frequency range covers `0.1GHz` to `30GHz`
- no NaN/Inf in parsed S-parameters
- `L(5GHz)` is finite and physically plausible
- `L(5GHz)` is in the same order as the Mohan estimate, for example `0.5nH` to `3nH`
- `Q(5GHz)` is finite
- SRF extraction either returns a finite value inside sweep or reports that no crossing was found

If `L(5GHz)` is negative, extremely large, or many Q values are invalid, debug
the port definition, reference direction, metal connectivity, and substrate
boundary before using the template online.

## Step 5: Register the Template

Only after the nominal debug simulation passes, add the generated template to
the worker template registry, for example:

```yaml
templates:
  spiral_inductor_t3_template_v1:
    project_path: "D:/ansys/spiral_inductor_template_v1.aedt"
    design_name: "SpiralInductorT3"
    setup_name: "Setup1"
    sweep_name: "Sweep"
    hfss_sweep:
      sweep_type: "interpolating"
      sweep_data: "LINC 0.1GHz 30GHz 601"
```

For local worker runs this belongs in:

```text
configs/templates.spiral_local.yaml
```

After registration, the online optimizer can use the same queue-backed flow as
BCLF:

```text
optimizer -> queue job -> worker opens generated AEDT template
          -> set Din/W/S/H -> solve/export .s2p -> objective
```

## Failure Policy

Do not proceed to BO/LHS/BNN online optimization if any of these fail:

- template generation fails
- required variables/setup/sweep/ports are missing
- nominal debug simulation fails to solve
- Touchstone export or parse fails
- extracted nominal metrics are non-physical

Fix the generated template first; online campaigns amplify template errors.
