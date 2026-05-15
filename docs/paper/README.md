# Paper Docs Handbook

This section is the paper-facing documentation layer for this repository.

Audience:
- Internal collaborators writing methods, experiments, and results sections.

Principles:
- Keep operational/runbook docs in `docs/` unchanged.
- Keep paper narrative, experiment metadata, and claim traceability in `docs/paper/`.
- Treat notebooks and files under `experiments/` as data-generation source of truth.

## Navigation

- Methods: [methods.md](methods.md)
- Optimization experiment backbone: [../optimization_experiment_backbone.md](../optimization_experiment_backbone.md)
- Experiment design and protocol: [experiments.md](experiments.md)
- Claim-to-evidence map (figures/tables): [results_index.md](results_index.md)
- Reproducibility policy: [reproducibility.md](reproducibility.md)
- Canonical experiment-card template: [templates/experiment_card.md](templates/experiment_card.md)

## Minimum Maintenance Rules

- Any new reported result must update at least one experiment card in the same PR/commit.
- Any new plotted claim must be added to `results_index.md` with exact artifact paths.
- Every experiment card must carry one status:
  - `draft`
  - `paper-candidate`
  - `camera-ready`
