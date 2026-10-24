# Burridge–Knopoff model ↔ sandpile mapping

> Work in progress. This README will be expanded into a full project overview when
> the results are in.

Can the earthquake statistics of the one-dimensional Burridge–Knopoff (BK) spring–block
model be understood by mapping it onto a sandpile model? This repository holds a rewrite
of a bachelor project (Sharif University of Technology, supervised by
Dr. Saman Moghimi Araghi). Every piece of simulation code is tested against exact
results, and the theory and results are written up as a thesis report.

## Layout

| Path | Contents |
|---|---|
| `bksoc/` | Python package (NumPy + Numba): sandpile models, exact theory, plotting style |
| `tests/` | `pytest` suite: conservation, Abelian property, exact 1D/2D results |
| `scripts/thesis/` | scripts that regenerate the data and figures of the thesis |
| `thesis/` | LaTeX thesis report |
| `papers/` | reference papers |
| `data/` | simulation output (not tracked; regenerable) |

## Quick start

```bash
pip install -e ".[test]"
pytest                                   # ~10 s
python scripts/thesis/soc_figures.py     # ~15 min on 8 cores
cd thesis && latexmk -pdf -outdir=build main.tex
```
