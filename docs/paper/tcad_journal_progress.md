# TCAD Journal Progress

Last updated UTC: 2026-06-03

## Target

- Venue: IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD)
- Source conference draft: `submissions/bnn-vector-fitting-journal/conference/aspdac_2026.tex`
- TCAD first draft: `submissions/bnn-vector-fitting-journal/journal/bnn_vector_fitting_tcad.tex`
- Status: faithful TCAD-oriented LaTeX port created and compiled successfully; not yet a submission-ready TCAD manuscript.

## Completed

- Created a self-contained `journal/` draft directory beside `conference/` and `templates/`.
- Ported the paper to IEEE journal class with `\documentclass[lettersize,journal]{IEEEtran}`.
- Added the TCAD running header:
  `IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems`.
- Preserved the conference title, authors, affiliation, funding note, abstract, keywords, sections, figures, tables, algorithm, equations, citations, and bibliography database.
- Copied local build resources:
  - `journal/IEEEtran.cls`
  - `journal/aspdac_2026.bib`
  - `journal/Figures/`

## Faithfulness Checks

- Body diff from `\begin{abstract}` through the text before `\bibliographystyle{IEEEtran}`: passed with no differences.
- Bibliography database comparison against the conference source: passed.
- Figure path existence check for all `\includegraphics` entries: passed.
- IEEE sample-template residue check: passed.

## Build Check

Minimal home-local LaTeX overlay:

```bash
~/texmf/tex/latex/algorithms/algorithm.sty
~/texmf/tex/latex/algorithms/algorithmic.sty
~/texmf/bibtex/bst/IEEEtran/IEEEtran.bst
~/tcad-latex-env/bin/compile_tcad.sh
```

Build command from `submissions/bnn-vector-fitting-journal/journal/`:

```bash
~/tcad-latex-env/bin/compile_tcad.sh
```

Current result: success. `latexmk` exits with code `0` and produces:

```text
bnn_vector_fitting_tcad.pdf
```

Final undefined reference/citation check: none.

Known non-blocking LaTeX warnings:
- `lettersize` is reported as an unused global option by the local IEEEtran setup.
- Some included figure PDFs are newer than the pdfTeX output target and trigger PDF-version inclusion warnings.
- One table block has a modest overfull hbox warning.


## Page Count

Current generated PDF:

```text
Pages: 7
File size: 2197814 bytes
PDF version: 1.5
```

## Notes

- The current draft intentionally does not add TCAD-specific new content, disclosure text, biographies, acknowledgments, or expanded experiments.
- This is a faithful first journal shell, not a final TCAD submission package.
