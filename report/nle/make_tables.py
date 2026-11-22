"""Write the LaTeX tables for the NLE-format report (report/nle/tables.tex).

Every number comes from results/*.json and results/*_preds.csv, so the tables cannot drift from the
saved results. Decimals use the NLE style macro \\hpt (centred decimal point), tables use the NLE
double rules at top and bottom and no vertical rules.

Usage (from the repository root):  python report/nle/make_tables.py
"""
import csv
import json
import os
from math import comb

RESULTS = "results"
OUT = os.path.join("report", "nle", "tables.tex")

# (file stem, name in the report). Order = order of the main results table.
MODELS = [
    ("roberta_meta", "RoBERTa + speaker info"),
    ("bert_meta", "BERT + speaker info"),
    ("bert_text", "BERT, text only"),
    ("roberta_text", "RoBERTa, text only"),
    ("llm_fewshot", "Qwen 7B few-shot"),
    ("lstm", "BiLSTM"),
    ("llm_zeroshot", "Qwen 7B zero-shot"),
]


def load(stem):
    with open(os.path.join(RESULTS, f"{stem}.json"), encoding="utf-8") as f:
        return json.load(f)


def preds(stem):
    with open(os.path.join(RESULTS, f"{stem}_preds.csv"), newline="", encoding="utf-8") as f:
        return {int(r["id"]): (int(r["y_true"]), int(r["y_pred"])) for r in csv.DictReader(f)}


def hpt(x, nd):
    """Format x with nd decimals using the NLE centred decimal point."""
    s = f"{x:.{nd}f}"
    return s.replace(".", "\\hpt ") if "." in s else s


def two_way(stem):
    """Accuracy of the fake-vs-real grouping (0,1,2 = false-ish; 3,4,5 = true-ish), in percent."""
    p = preds(stem)
    # An invalid LLM answer is stored as -1 and counts as wrong here too (never as "false-ish").
    ok = sum(q >= 0 and (t <= 2) == (q <= 2) for t, q in p.values())
    return 100 * ok / len(p)


def mcnemar(a, b):
    """(only a right, only b right, exact two-sided p) on the shared rows."""
    pa, pb = preds(a), preds(b)
    only_a = only_b = 0
    for i in pa:
        ra, rb = pa[i][0] == pa[i][1], pb[i][0] == pb[i][1]
        only_a += ra and not rb
        only_b += rb and not ra
    n = only_a + only_b
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)
    return only_a, only_b, p


def fmt_p(p):
    return "$<$0\\hpt 0001" if p < 0.0001 else (hpt(p, 4) if p < 0.001 else hpt(p, 3) if p < 0.01 else hpt(p, 2))


def table(caption, label, colspec, head, rows, note=None):
    out = ["\\begin{table}", f"\\caption{{{caption}}}", "\\begin{minipage}{\\textwidth}",
           f"\\begin{{tabular}}{{{colspec}}}", "\\hline\\hline", head + " \\\\", "\\hline"]
    out += [r + " \\\\" for r in rows]
    out += ["\\hline\\hline", "\\end{tabular}"]
    if note:
        out += [f"\\par\\smallskip{{\\small {note}}}"]
    out += ["\\vspace{-1.5\\baselineskip}", "\\end{minipage}", f"\\label{{{label}}}", "\\end{table}", ""]
    return "\n".join(out)


def main():
    R = {s: load(s) for s, _ in MODELS}
    parts = []

    # Table: main results on the 300-row subset
    rows = []
    for s, name in MODELS:
        r = R[s]
        rows.append(f"{name} & {hpt(r['accuracy'], 3)} & {hpt(r['macro_f1'], 3)} & {hpt(two_way(s), 1)}\\% & "
                    f"{r['invalid_outputs']} / {r['n_eval']}")
    parts.append(table(
        "Main results on the shared 300-statement test subset. The fake-vs-real column is our own grouping "
        "of the saved predictions. The BiLSTM was timed on a CPU.",
        "tab:main", "lcccc",
        "Model & Accuracy & Macro-F1 & Fake vs real & Invalid answers", rows,
        "Random guessing: 16\\hpt 7\\% (six levels). Always guessing one side: 56\\hpt 3\\% (fake vs real)."))

    # Table: full test set
    rows = []
    for s, name in [m for m in MODELS if m[0] in ("roberta_meta", "bert_meta", "roberta_text", "bert_text", "lstm")]:
        ft = R[s]["full_test"]
        rows.append(f"{name} & {hpt(ft['accuracy'], 3)} & {hpt(ft['macro_f1'], 3)}")
    # sort by accuracy as in the report
    rows.sort(key=lambda r: -float(r.split("&")[1].replace("\\hpt ", ".")))
    parts.append(table("Results on the full test set (1,283 statements).", "tab:full", "lcc",
                       "Model & Accuracy & Macro-F1", rows))

    # Table: McNemar
    pairs = [("llm_fewshot", "llm_zeroshot", "Qwen few-shot", "Qwen zero-shot"),
             ("bert_text", "llm_fewshot", "BERT text only", "Qwen few-shot"),
             ("bert_text", "llm_zeroshot", "BERT text only", "Qwen zero-shot"),
             ("bert_meta", "bert_text", "BERT + speaker info", "BERT text only"),
             ("roberta_text", "bert_text", "RoBERTa text only", "BERT text only"),
             ("roberta_meta", "bert_meta", "RoBERTa + speaker info", "BERT + speaker info")]
    rows, mc = [], []
    for a, b, na, nb in pairs:
        oa, ob, p = mcnemar(a, b)
        mc.append((na, nb, oa, ob, p))
        rows.append(f"{na} & {nb} & {oa} & {ob} & {fmt_p(p)}")
    parts.append(table("McNemar exact tests on the 300 statements. ``Only A right'' counts the statements "
                       "that A got right and B got wrong.", "tab:sig", "llccc",
                       "Model A & Model B & Only A right & Only B right & $p$-value", rows))

    # Table: cost and speed
    cost_rows = []
    order = ["lstm", "bert_text", "bert_meta", "roberta_text", "roberta_meta", "llm_zeroshot", "llm_fewshot"]
    names = dict(MODELS)
    names["lstm"] = "BiLSTM (CPU)"
    for s in order:
        r = R[s]
        mem = r.get("gpu_peak_mem_mb") or r.get("gpu_mem_inference_mb")
        sec = r.get("infer_seconds_per_1000") or r.get("inference_sec_per_1000")
        train = "none" if r["train_minutes"] == 0 else hpt(r["train_minutes"], 1)
        sec_s = hpt(sec, 1) if sec else hpt(r["avg_latency_ms"], 1)
        cost_rows.append(f"{names[s]} & {r['params'] / 1e6:,.0f}M & {train} & {sec_s} & "
                         f"{hpt(r['est_cost_per_1k'], 4)} & {f'{mem:,.0f}' if mem else 'n/a'}")
    parts.append(table("Size, time and cost, priced at a T4 GPU rate of \\$0.35 per hour plus 10\\% overhead, "
                       "batch size 1.", "tab:cost", "lrrrrr",
                       "Model & Params & Train (min) & Sec per 1,000 & \\$ per 1,000 & GPU (MB)", cost_rows))

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("% Generated by report/nle/make_tables.py from results/. Do not edit by hand.\n\n")
        f.write("\n".join(parts))

    # Print the numbers the text quotes, so they can be checked against the prose.
    print("fake-vs-real:", {s: round(two_way(s), 1) for s, _ in MODELS})
    for na, nb, oa, ob, p in mc:
        print(f"McNemar {na} vs {nb}: {oa} / {ob}  p={p:.4g}")
    fs, bt = R["llm_fewshot"], R["bert_text"]
    print("ratios BERT text vs few-shot: params", round(fs["params"] / bt["params"], 1),
          "speed", round((fs["infer_seconds_per_1000"]) / bt["inference_sec_per_1000"], 1),
          "cost", round(fs["est_cost_per_1k"] / bt["est_cost_per_1k"], 1),
          "macroF1 gain", round(bt["macro_f1"] - fs["macro_f1"], 3))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
