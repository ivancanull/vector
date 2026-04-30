# Experiment Card Template

Use this template for each experiment family that may appear in the paper.

## Metadata

- experiment_id: `stable_id_here`
- status: `draft | paper-candidate | camera-ready`
- owner: `name_or_alias`
- last_updated_utc: `YYYY-MM-DD`

## Research Question / Hypothesis

- What specific question this experiment answers.

## Method Variant

- Optimizer variant and key algorithm settings.
- Any ablation/treatment/control definition.

## Dataset / Simulator Source

- Source type (queue-backed HFSS, replay dataset, synthetic).
- Relevant source references.

## Config References

- Notebook path + section/cell description.
- Config files (`json/yaml`) used by this run family.

## Seed Policy and Seeds Used

- Seed policy (single seed / fixed list / sweep).
- Actual seeds used in reported result.

## Artifact Links

- Summary metrics files (`csv/json`).
- Per-run history files (`csv/json`) where needed.
- Plot files used in manuscript.

## Inclusion Rule for Paper Figures/Tables

- Exact rule that determines which runs are included in final figures/tables.
- Any exclusion rules.

## Known Caveats

- Known data quality risks, pending reruns, or interpretation limits.
