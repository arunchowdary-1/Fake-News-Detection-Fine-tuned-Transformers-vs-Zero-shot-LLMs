# Report in the NLE format

`main.tex` is the final report in the CUP Natural Language Engineering (NLE) style (`\documentclass{nle}`),
with the same content and numbers as `docs/REPORT.md` and `report/main.tex`.

## Build (Overleaf, pdfLaTeX, compile twice)
1. Get `nle.cls` / `NLE.sty` (the class files are not in this repo; use the copy from the course page).
2. Upload to Overleaf: `main.tex`, `tables.tex`, the NLE class files, and the figures.
   `main.tex` loads the figures from `../figures/`; if you upload the files into one flat folder, change the
   `\graphicspath` line to `{{./}}`.

## Files
- `main.tex`: the report.
- `tables.tex`: the four results tables, **generated** from `results/` (do not edit by hand).
- `make_tables.py`: rebuilds `tables.tex`. From the repository root: `python report/nle/make_tables.py`.
  It also prints the McNemar and fake-vs-real numbers quoted in the text.

## Before you submit
- Replace the `[Surname]`, `[Department]`, `[University]`, `[City]` and `[Country]` placeholders in `main.tex`.
- This version has **not been compiled** (no LaTeX and no NLE class file on the machine that wrote it).
  Compile it on Overleaf and check: the citations (`\cite` is assumed to print its own parentheses in this class;
  if you see double brackets, remove the extra ones), the table widths, and the page count.
