# TCAD Journal Idea Log

Last updated UTC: 2026-06-03

## Core Idea

The paper proposes an uncertainty-aware Bayesian online learning framework for RF passive component modeling. The central thesis is that expensive EM simulations can be reduced by using a Bayesian neural network to quantify uncertainty across both geometric parameters and frequency points, then using that uncertainty to guide adaptive sampling.

## Current Conference Narrative

- Problem: conventional ML-based RF passive modeling still requires large EM-generated datasets across geometric and frequency spaces.
- Method:
  - reconfigurable BNN with shared backbone and point/vector heads;
  - Bayesian online learning loop;
  - uncertainty-aware geometric sampling;
  - uncertainty-aware adaptive frequency sampling for vector fitting.
- Evidence:
  - three RF passive components: BCLF, SI, and MTL;
  - Uncertainty-Aware AFS compared with uniform sampling;
  - final comparison against conventional ML-based modeling;
  - reported 2.86 percent EM simulation time and 35x speedup.
- Current claim boundary:
  - these are conference-draft claims and should stay unchanged in the faithful TCAD first draft;
  - journal submission should verify every number against artifacts before expanding claims.

## TCAD Positioning

TCAD fit should emphasize automated design methods and tools for integrated circuits and systems, especially analog, mixed-signal, RF, microwave, and simulation-efficient modeling workflows.

The strongest TCAD framing is:

- RF passive model generation is a CAD bottleneck because EM simulation dominates the design loop.
- The proposed BNN sampler is a data-efficient modeling engine, not just a neural predictor.
- Joint geometry-frequency sampling turns model construction into an active CAD data acquisition problem.
- Vector fitting connects the learning method to established RF/CAD modeling practice.

## Candidate Journal Extensions

- Add a conference-to-journal delta section in the introduction or cover letter.
- Expand related work around:
  - RF/microwave surrogate modeling;
  - active learning and Bayesian optimization in EDA/CAD;
  - adaptive frequency sampling and vector fitting;
  - uncertainty quantification in neural surrogate models.
- Add deeper ablations if artifacts support them:
  - geometry-only uncertainty sampling;
  - frequency-only Uncertainty-Aware AFS;
  - joint geometry-frequency uncertainty-aware sampling;
  - random and uniform sampling controls;
  - BNN versus deterministic uncertainty proxies.
- Add sensitivity analysis:
  - sample budget;
  - number of BNN Monte Carlo samples;
  - online batch size;
  - prior variance or learning-rate schedule.
- Add practical CAD discussion:
  - HFSS integration;
  - simulator-call accounting;
  - how a designer would use the generated surrogate;
  - limitations for high-dimensional layouts and discontinuous responses.

## Claims To Avoid Until Supported

- Do not claim TCAD-ready 30 percent new content until it is actually written and traceable.
- Do not claim BNN online optimization superiority unless the completed queue-backed BNN online campaign supports it.
- Do not claim general dominance of Uncertainty-Aware AFS for every RF reconstruction setting without full-sweep evidence.
- Do not claim a submission-ready page count until the PDF builds in a complete TeX environment.

## Source Notes

- TCAD publication page: https://ieee-ceda.org/publications/tcad
- TCAD paper submission page: https://ieee-ceda.org/publications/tcad/tcad-paper-submissions
