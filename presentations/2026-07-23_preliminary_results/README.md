# Preliminary results presentation — 2026-07-23

This directory contains five self-contained LaTeX Beamer presentations for the
scientific advisory meeting. The result figures are generated directly from
the repository's resting-state analysis TSV files; no report screenshots or
publication figures are copied.

- `presentation_v1.tex` / `presentation_v1.pdf`: archived original deck.
- `presentation_v2.tex` / `presentation_v2.pdf`: recommended plain-language revision.
- `presentation_v3.tex` / `presentation_v3.pdf`: minimal, decision-focused revision.
- `v4/presentation_v4.tex` / `v4/presentation_v4.pdf`: presentation-ready revision with explicit task names and no citations.
- `v5/presentation_v5.tex` / `v5/presentation_v5.pdf`: v4 plus a side-by-side comparison of Tozzi and FOR2107 scan conditions.
- `presentation.tex` / `presentation.pdf`: original filenames retained for compatibility; identical in content to v1.

## Compile

Compile the recommended v5 from this directory:

```bash
latexmk -lualatex -interaction=nonstopmode -halt-on-error -outdir=v5 v5/presentation_v5.tex
```

The command regenerates both result-figure versions automatically and
produces `v5/presentation_v5.pdf`. Python dependencies are taken from the project
virtual environment when available; otherwise the script uses `python3`.

To remove LaTeX build products while preserving the generated figure and PDF:

```bash
latexmk -c -outdir=v5 v5/presentation_v5.tex
```

## Data dependencies

`figures/generate_figures.py` reads:

- `derivatives/for2107_tozzi_external_rs/analysis/solution_evaluation.tsv`
- `derivatives/for2107_tozzi_external_rs/analysis/cluster_assignments.tsv`

All decks deliberately label all current grouping results as exploratory.
The v2 deck uses plain-language descriptions and avoids specialist terminology
where it is not required for the scientific decision.
