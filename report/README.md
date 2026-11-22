# Report

`main.tex` is the project report (LaTeX, pdfLaTeX). `figures/` holds the charts it uses.

## Build on Overleaf
1. Zip this `report/` folder, or upload `main.tex` and the `figures/` folder.
2. In Overleaf choose **New Project > Upload Project**, and set the compiler to **pdfLaTeX**.
3. Compile twice (the title-page banner needs two passes).

## Rebuild the figures
From the repository root, after the result files in `results/` change:

```bash
python report/make_figures.py
```

Every chart is drawn from `results/*.json` and `results/*_preds.csv`, so the figures always match the saved results.

## Before you submit
- Replace the first names on the title page with full names (see the TODO in `main.tex`).
- Check the reference list.
