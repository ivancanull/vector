# TCAD Journal TODO

Last updated UTC: 2026-06-03

## Build Readiness

- Keep the home-local compile helper available at `~/tcad-latex-env/bin/compile_tcad.sh`.
- Re-run the full build chain after any manuscript or figure change.
- Inspect non-blocking LaTeX warnings before submission:
  - unused `lettersize` class option;
  - PDF-version inclusion warnings from imported figures;
  - modest overfull hbox in the parameter table.
- If a future Regular Paper draft exceeds the TCAD page limit, reduce layout overhead only after preserving the faithful first-draft source.

## TCAD Compliance

- Confirm the intended manuscript type: Regular Paper unless the scope is deliberately narrowed.
- Check the TCAD Regular Paper page budget of 14 pages in IEEE two-column format.
- Keep the abstract within 100 to 250 words and keywords within 4 to 8 items.
- Add ORCID information for all authors if required by the submission system.
- Prepare the prior-conference disclosure and cover-letter statement.
- Prepare the AI-use acknowledgement if any manuscript text, figures, or editing support must be disclosed under the active TCAD policy.
- Add final acknowledgments only after confirming funding and collaborator wording.

## Required Journal Expansion

- Add a clear statement of how the TCAD manuscript extends the conference version.
- Target at least 30 percent substantive new material before submission.
- Strengthen related work for CAD, microwave/RF modeling, adaptive sampling, vector fitting, and Bayesian surrogate modeling.
- Expand the method discussion around:
  - why uncertainty needs to cover both geometry and frequency domains;
  - how the reconfigurable BNN heads map to point and vector modes;
  - how online learning avoids repeated full retraining.
- Add a limitations paragraph that separates supported results from future optimization claims.

## Experiment And Evidence TODO

- Decide which journal extensions are actually supported by current artifacts before writing claims.
- Recompute every number from artifact files before moving it into the manuscript.
- Add or promote experiment cards in `docs/academic/paper/experiments.md` only when the underlying artifacts are complete.
- Update `docs/academic/paper/results_index.md` for every new claim, figure, or table.
- Do not claim BNN online superiority under the current completed campaign; the five-seed result is complete but does not beat LHS/Turbo by median final best.

## Submission Package TODO

- Prepare a cover letter that states:
  - the target venue is TCAD;
  - the manuscript is extended from the conference draft;
  - the main new contributions relative to the conference version;
  - any prior publication or submission status.
- Prepare source archive with `.tex`, `.bib` or `.bbl`, figures, `IEEEtran.cls`, and required style files.
- Check all figure fonts, line widths, and captions after the TCAD PDF builds.
- Run a final citation audit before submission.
