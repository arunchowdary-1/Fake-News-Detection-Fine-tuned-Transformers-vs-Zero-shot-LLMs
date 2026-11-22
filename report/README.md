# Report

The project report is a short, 7-page paper in journal-article layout. It exists in three forms with the same content:

| File | What it is |
|---|---|
| `docs/REPORT.md` | The source text. Readable on GitHub and in the VS Code preview. Edit this one. |
| `docs/REPORT.pdf` | The finished PDF, built from `REPORT.md`. |
| `report/main.tex` | The same report as LaTeX, for Overleaf. |

`report/figures/` holds the charts. The report uses `fig_data`, `fig_main_results`, `fig_cost_recall`, `fig_confusion` and `fig_llm_bias`. The other PNGs are the same charts as single images.

## Rebuild the PDF
From the repository root, after editing `docs/REPORT.md`:

```bash
python report/build_pdf.py
```

It needs the `markdown` Python package and Google Chrome (or Chromium). Chrome prints the PDF, so no LaTeX is needed.

## Rebuild the figures
After the result files in `results/` change:

```bash
python report/make_figures.py
```

Every chart is drawn from `results/*.json` and `results/*_preds.csv`, so the figures always match the saved results. Rebuild the PDF afterwards.

## Build the LaTeX version on Overleaf
1. Zip this `report/` folder, or upload `main.tex` and the `figures/` folder.
2. In Overleaf choose **New Project > Upload Project** and set the compiler to **pdfLaTeX**.
3. Compile. `main.tex` has not been compiled on the machine that wrote it, so check that it renders.

## Before you submit
- Replace the department, university, city and country placeholders on the title page (in `REPORT.md` and `main.tex`), then rebuild the PDF.
- Check the reference list, which was written from memory.
