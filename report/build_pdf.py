"""Build docs/REPORT.pdf from docs/REPORT.md in a journal-article layout.

Run from the repository root:
    python report/build_pdf.py

Needs: the `markdown` Python package, and Google Chrome (or Chromium) installed.
Chrome does the final HTML -> PDF step, so no LaTeX is needed.

Layout: centred title block, one-paragraph abstract, numbered sections, run-in bold
paragraph headings, table captions above and figure captions below, running headers that
alternate between the authors and the short title, and page numbers at the bottom.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "docs" / "REPORT.md"
PDF = ROOT / "docs" / "REPORT.pdf"

RUNNING_LEFT = "Dhakad and others"
RUNNING_RIGHT = "Fake news detection on LIAR"

CSS = """
@page { size: A4; margin: 22mm 22mm 24mm 22mm; }
@page :left  { @top-left  { content: "@@LEFT@@";  font: italic 9pt 'Liberation Serif', serif; color: #444; }
               @bottom-center { content: counter(page); font: 9pt 'Liberation Serif', serif; color: #444; } }
@page :right { @top-right { content: "@@RIGHT@@"; font: italic 9pt 'Liberation Serif', serif; color: #444; }
               @bottom-center { content: counter(page); font: 9pt 'Liberation Serif', serif; color: #444; } }
@page :first { @top-left { content: none; } @top-right { content: none; }
               @bottom-center { content: counter(page); font: 9pt 'Liberation Serif', serif; color: #444; } }
* { box-sizing: border-box; }
html { font-family: 'Liberation Serif', 'DejaVu Serif', serif; font-size: 10.4pt; line-height: 1.36; color: #111;
       text-align: justify; hyphens: auto; }
body { margin: 0; }
h1, h2, h3, h4 { font-family: 'Liberation Serif', 'DejaVu Serif', serif; color: #000; break-after: avoid; text-align: left; }
h2 { font-size: 12.5pt; margin: 15pt 0 5pt; }
h3 { font-size: 10.8pt; margin: 11pt 0 4pt; }
p { margin: 0 0 5.5pt; orphans: 2; widows: 2; }
a { color: #1b4f9c; text-decoration: none; }
code { font-family: 'Liberation Mono', 'DejaVu Sans Mono', monospace; font-size: 8.4pt; background: #f2f2f2; padding: 0 2pt; }
pre { background: #f4f4f4; padding: 3pt 6pt; margin: 2pt 0 7pt; white-space: pre-wrap; text-align: left; break-inside: avoid; }
pre code { background: none; font-size: 8pt; padding: 0; }
ul, ol { margin: 0 0 6pt; padding-left: 16pt; }
li { margin: 0 0 2.5pt; }

/* title block and abstract */
.titleblock { text-align: center; margin-bottom: 8pt; }
.titleblock h1 { font-size: 17pt; text-align: center; margin: 4pt 0 4pt; line-height: 1.2; }
.titleblock h3 { font-size: 12.5pt; text-align: center; margin: 0 0 12pt; font-weight: bold; }
.titleblock p { text-align: center; margin: 0 0 4pt; }
.abstract h2 { text-align: center; font-size: 10.5pt; margin: 10pt 0 4pt; }
.abstract p { font-size: 9.4pt; line-height: 1.33; margin: 0 9mm 8pt 9mm; }

/* tables: academic style, caption above */
table { width: 100%; border-collapse: collapse; margin: 2pt 0 8pt; font-size: 8.5pt; line-height: 1.28; text-align: left; hyphens: manual;
        border-top: 2.2pt double #000; border-bottom: 2.2pt double #000; }
thead th { text-align: left; font-weight: normal; padding: 2.5pt 5pt; border-bottom: 0.7pt solid #000; }
tbody td { padding: 1.8pt 5pt; vertical-align: top; }
tr { break-inside: avoid; }
table { break-inside: avoid; }
p.tabcap { text-align: center; font-size: 8.8pt; margin: 8pt 0 2pt; break-after: avoid; }
p.tabcap strong, p.figcap strong { font-weight: bold; }

/* figures: caption below */
figure { margin: 6pt 0 2pt; text-align: center; break-inside: avoid; }
figure img { width: 100%; height: auto; }
p.figcap { text-align: center; font-size: 8.8pt; margin: 2pt 6mm 9pt; }

/* references */
.refs p { text-align: left; font-size: 9pt; line-height: 1.28; padding-left: 12pt; text-indent: -12pt; margin: 0 0 3.5pt; }
""".replace("@@LEFT@@", RUNNING_LEFT).replace("@@RIGHT@@", RUNNING_RIGHT)


def convert():
    text = MD.read_text(encoding="utf-8")

    # Python-Markdown needs a blank line before a list; GitHub does not. Insert it for the PDF only.
    text = re.sub(r"(?m)^(?![ \t]*(?:[-*+]|\d+\.)[ \t])(?![ \t]*(?:\||```))([^\n]*\S)\n([ \t]*(?:[-*+]|\d+\.)[ \t])",
                  r"\1\n\n\2", text)

    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "toc"],
                             extension_configs={"toc": {"permalink": False}})

    # Images -> <figure>, absolute file paths so Chrome finds them.
    def fig(m):
        alt, src = m.group(1), m.group(2)
        path = (MD.parent / src).resolve()
        return f'<figure><img src="file://{path}" alt="{alt}"></figure>'
    html = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)"\s*/?></p>', fig, html)

    # Captions: tables are captioned above, figures below.
    html = re.sub(r"<p>(<strong>Table \d+\.</strong>)", r'<p class="tabcap">\1', html)
    html = re.sub(r"<p>(<strong>Fig\. \d+\.</strong>)", r'<p class="figcap">\1', html)

    # Title block = everything before the abstract heading; abstract = heading + its paragraph.
    m = re.search(r'<h2 id="abstract">', html)
    if not m:
        sys.exit("REPORT.md must contain a '## Abstract' heading")
    title, rest = html[:m.start()], html[m.start():]
    m2 = re.search(r'<h2 id="1-introduction">', rest)
    if not m2:
        sys.exit("REPORT.md must contain a '## 1 Introduction' heading")
    abstract, body = rest[:m2.start()], rest[m2.start():]

    # References: hanging indent.
    m3 = re.search(r'<h2 id="references">', body)
    if m3:
        body = body[:m3.start()] + '<div class="refs">' + body[m3.start():] + "</div>"

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>LIAR Truth Detector: report</title><style>{CSS}</style></head><body>
<div class="titleblock">{title}</div><div class="abstract">{abstract}</div>{body}</body></html>"""
    return page


def find_chrome():
    for name in ("google-chrome", "chromium-browser", "chromium", "chrome"):
        path = shutil.which(name)
        if path:
            return path
    sys.exit("Chrome or Chromium is needed to print the PDF, but none was found on PATH.")


def main():
    html = convert()
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "report.html"
        html_path.write_text(html, encoding="utf-8")
        out_tmp = Path(tmp) / "report.pdf"
        cmd = [find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
               "--virtual-time-budget=8000", f"--user-data-dir={tmp}/profile",
               f"--print-to-pdf={out_tmp}", f"file://{html_path}"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if not out_tmp.exists() or out_tmp.stat().st_size < 10_000:
            sys.exit(f"Chrome did not produce a PDF.\n{res.stderr[-800:]}")
        shutil.copyfile(out_tmp, PDF)
    print(f"Wrote {PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
