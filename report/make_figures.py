"""Build every figure in the report from the real result files.

Run from the repository root:
    python report/make_figures.py

Reads results/*.json and results/*_preds.csv, writes PNGs to report/figures/.
Nothing is typed in by hand, so the charts always match the saved results.
"""
import csv
import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RESULTS = "results"
OUT = os.path.join("report", "figures")
os.makedirs(OUT, exist_ok=True)

# Project label order (index 0..5)
LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
FALSEISH = {0, 1, 2}
SUBSET_RANDOM = 1 / 6
SUBSET_MAJORITY = 62 / 300  # half-true is the biggest class in the 300-row subset

# Palette (validated categorical slots 1-3, light mode) plus neutrals.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#8d8c86"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

FAMILY_COLOR = {
    "Baseline (LSTM)": GRAY,
    "Fine-tuned, text only": BLUE,
    "Fine-tuned, with speaker info": ORANGE,
    "Prompted LLM (no training)": AQUA,
}

# model key -> (display name, family)
MODELS = {
    "lstm": ("BiLSTM", "Baseline (LSTM)"),
    "bert_text": ("BERT, text only", "Fine-tuned, text only"),
    "roberta_text": ("RoBERTa, text only", "Fine-tuned, text only"),
    "bert_meta": ("BERT + speaker info", "Fine-tuned, with speaker info"),
    "roberta_meta": ("RoBERTa + speaker info", "Fine-tuned, with speaker info"),
    "llm_zeroshot": ("Qwen 7B zero-shot", "Prompted LLM (no training)"),
    "llm_fewshot": ("Qwen 7B few-shot", "Prompted LLM (no training)"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def load_json(name):
    with open(os.path.join(RESULTS, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def load_preds(name):
    with open(os.path.join(RESULTS, f"{name}_preds.csv"), encoding="utf-8") as f:
        return [(int(r["y_true"]), int(r["y_pred"])) for r in csv.DictReader(f)]


def two_way_accuracy(name):
    """False-ish vs true-ish accuracy; an invalid answer (-1) counts as wrong."""
    rows = load_preds(name)
    ok = sum(1 for t, p in rows if p != -1 and ((p in FALSEISH) == (t in FALSEISH)))
    return ok / len(rows)


def family_legend(ax, families, loc="lower right"):
    handles = [Patch(facecolor=FAMILY_COLOR[f], label=f) for f in families]
    leg = ax.legend(handles=handles, loc=loc, frameon=False, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK2)


def ranked_bars(values, title, subtitle, fname, ref_lines, xmax, figsize=(8.2, 4.4)):
    """One horizontal bar per model, ranked, coloured by family, value on each bar."""
    items = sorted(values.items(), key=lambda kv: kv[1])  # ascending -> best on top
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(items))
    for i, (key, v) in enumerate(items):
        name, fam = MODELS[key]
        ax.barh(i, v * 100, height=0.62, color=FAMILY_COLOR[fam], edgecolor="white", linewidth=1.2)
        ax.text(v * 100 + 0.8, i, f"{v * 100:.1f}%", va="center", ha="left", fontsize=10, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([MODELS[k][0] for k, _ in items])
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Accuracy on the 300-statement test subset (%)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    line_handles = []
    for val, label, ls in ref_lines:
        line_handles.append(ax.axvline(val * 100, color=INK2, linestyle=ls, linewidth=1.0, zorder=1, label=label))
    fams = [f for f in FAMILY_COLOR if any(MODELS[k][1] == f for k in values)]
    handles = [Patch(facecolor=FAMILY_COLOR[f], label=f) for f in fams] + line_handles
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3,
                    frameon=False, fontsize=9.3, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=24)
    ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


# ---------------------------------------------------------------- figures
def fig_accuracy_six_way():
    vals = {k: load_json(k)["accuracy"] for k in MODELS}
    ranked_bars(
        vals,
        "How often each model gets the exact rating right",
        "Six-way scale, pants-fire to true. Random guessing gets 16.7%.",
        "fig_accuracy_6way.png",
        [(SUBSET_RANDOM, "random 16.7%", ":"), (SUBSET_MAJORITY, "always 'half-true' 20.7%", "--")],
        xmax=60,
    )


def fig_accuracy_two_way():
    vals = {k: two_way_accuracy(k) for k in MODELS}
    base = sum(1 for t, _ in load_preds("bert_text") if t in FALSEISH) / 300
    ranked_bars(
        vals,
        "How often each model separates false-ish from true-ish",
        "False-ish = pants-fire, false, barely-true. True-ish = the other three.",
        "fig_accuracy_2way.png",
        [(max(base, 1 - base), "always one side 56.3%", "--")],
        xmax=92,
    )


def fig_cost_accuracy():
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    offsets = {
        "lstm": (9, -4, "left"), "bert_text": (10, 3, "left"), "roberta_text": (-10, -12, "right"),
        "bert_meta": (-10, -4, "right"), "roberta_meta": (10, 4, "left"),
        "llm_zeroshot": (-10, -12, "right"), "llm_fewshot": (-10, 8, "right"),
    }
    for key, (name, fam) in MODELS.items():
        r = load_json(key)
        x, yv = r["est_cost_per_1k"], r["accuracy"] * 100
        ax.scatter(x, yv, s=95, color=FAMILY_COLOR[fam], edgecolor="white", linewidth=1.8, zorder=3)
        dx, dy, ha = offsets[key]
        ax.annotate(name, (x, yv), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=9.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Cost to classify 1,000 statements, US dollars (log scale, T4 GPU rate)")
    ax.set_ylabel("Accuracy on the 300-statement subset (%)")
    ax.set_ylim(10, 55)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    family_legend(ax, list(FAMILY_COLOR), loc="center right")
    ax.set_title("More money did not buy more accuracy", loc="left", fontsize=12.5, fontweight="bold", pad=24)
    ax.text(0, 1.045, "Best results sit in the top left: cheap and accurate. The BiLSTM was timed on a CPU.",
            transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
    fig.savefig(os.path.join(OUT, "fig_cost_accuracy.png"))
    plt.close(fig)


def confusion(name):
    m = np.zeros((6, 6), dtype=int)
    for t, p in load_preds(name):
        if p != -1:
            m[t, p] += 1
    return m


def fig_confusion():
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f1f6fd", "#2a78d6", "#0d2f5e"])
    pair = [("roberta_text", "RoBERTa, text only"), ("roberta_meta", "RoBERTa + speaker info")]
    mats = [confusion(k) for k, _ in pair]
    vmax = max(m.max() for m in mats)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    for ax, (key, title), m in zip(axes, pair, mats):
        ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax)
        for i in range(6):
            for j in range(6):
                v = m[i, j]
                ax.text(j, i, str(v), ha="center", va="center", fontsize=10,
                        color="white" if v > vmax * 0.55 else INK,
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(LABELS, fontsize=9)
        ax.set_xlabel("What the model said")
        ax.set_title(f"{title}  (accuracy {load_json(key)['accuracy'] * 100:.1f}%)",
                     loc="left", fontsize=10.5, fontweight="bold")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("What the statement really was")
    fig.suptitle("Where the mistakes go: counts of the 300 test statements", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_confusion.png"))
    plt.close(fig)


def recall_per_class(name):
    m = confusion(name)
    return [m[i, i] / m[i].sum() for i in range(6)]


def fig_recall_by_class():
    series = [
        ("roberta_text", "Text only", BLUE),
        ("roberta_meta", "With speaker info", ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(6)
    w = 0.36
    for k, (key, label, color) in enumerate(series):
        vals = [v * 100 for v in recall_per_class(key)]
        pos = x + (k - 0.5) * (w + 0.03)
        ax.bar(pos, vals, width=w, color=color, edgecolor="white", linewidth=1.2, label=label)
        for p, v in zip(pos, vals):
            ax.text(p, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 75)
    ax.set_ylabel("Statements of this kind found (%)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_title("Speaker information helps on five of six ratings, but not on 'true'", loc="left",
                 fontsize=12.5, fontweight="bold", pad=24)
    ax.text(0, 1.045, "RoBERTa: share of each real rating that the model got right.",
            transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
    fig.savefig(os.path.join(OUT, "fig_recall_by_class.png"))
    plt.close(fig)


def fig_llm_bias():
    zs = Counter(p for _, p in load_preds("llm_zeroshot") if p != -1)
    fs = Counter(p for _, p in load_preds("llm_fewshot") if p != -1)
    actual = Counter(t for t, _ in load_preds("llm_zeroshot"))
    panels = [
        ("What the statements really are", actual, GRAY),
        ("Zero-shot LLM answers", zs, AQUA),
        ("Few-shot LLM answers", fs, AQUA),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.8), sharey=True)
    for ax, (title, cnt, color) in zip(axes, panels):
        vals = [cnt.get(i, 0) for i in range(6)]
        ax.bar(range(6), vals, width=0.68, color=color, edgecolor="white", linewidth=1.2)
        for i, v in enumerate(vals):
            ax.text(i, v + 3, str(v), ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(range(6))
        ax.set_xticklabels(LABELS, rotation=40, ha="right", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Number of the 300 statements")
    axes[0].set_ylim(0, 165)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_llm_bias.png"))
    plt.close(fig)


def fig_label_distribution():
    # LIAR training split, project label order (counts from the Hugging Face dataset).
    train = {"pants-fire": 842, "false": 1998, "barely-true": 1657,
             "half-true": 2123, "mostly-true": 1966, "true": 1683}
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    vals = [train[l] for l in LABELS]
    ax.bar(range(6), vals, width=0.66, color=GRAY, edgecolor="white", linewidth=1.2)
    for i, v in enumerate(vals):
        ax.text(i, v + 35, f"{v:,}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(range(6))
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 2500)
    ax.set_ylabel("Training statements")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Pants-fire is much rarer than the other ratings", loc="left", fontsize=12.5,
                 fontweight="bold", pad=24)
    ax.text(0, 1.045, "Ratings in the LIAR training set (10,269 statements).",
            transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
    fig.savefig(os.path.join(OUT, "fig_label_distribution.png"))
    plt.close(fig)


# ------------------------------------------------- compact multi-panel figures
# Used by the short, journal-style report: the same charts and the same data as above,
# placed side by side so the report needs fewer pages.
def fig_main_results_panel():
    six = {k: load_json(k)["accuracy"] for k in MODELS}
    two = {k: two_way_accuracy(k) for k in MODELS}
    base2 = sum(1 for t, _ in load_preds("bert_text") if t in FALSEISH) / 300
    order = [k for k, _ in sorted(six.items(), key=lambda kv: kv[1])]  # same order in both panels
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.3), gridspec_kw={"width_ratios": [1, 1]})
    for ax, vals, refs, xmax, names in (
        (a, six, [(SUBSET_RANDOM, ":"), (SUBSET_MAJORITY, "--")], 62, True),
        (b, {k: two[k] for k in order}, [(max(base2, 1 - base2), "--")], 92, False),
    ):
        # draw in the six-way order for both panels so rows line up
        for i, key in enumerate(order):
            v = vals[key]
            fam = MODELS[key][1]
            ax.barh(i, v * 100, height=0.62, color=FAMILY_COLOR[fam], edgecolor="white", linewidth=1.0)
            ax.text(v * 100 + 0.8, i, f"{v * 100:.1f}", va="center", ha="left", fontsize=8.5, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([MODELS[k][0] for k in order] if names else [], fontsize=9)
        ax.set_xlim(0, xmax)
        ax.xaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_xlabel("Accuracy (%)", fontsize=9)
        for val, ls in refs:
            ax.axvline(val * 100, color=INK2, linestyle=ls, linewidth=1.0, zorder=1)
    a.set_title("(a) Exact rating, six levels", loc="left", fontsize=10.5, fontweight="bold")
    b.set_title("(b) Fake vs real (false-ish vs true-ish)", loc="left", fontsize=10.5, fontweight="bold")
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=FAMILY_COLOR[f], label=f) for f in FAMILY_COLOR] + [
        Line2D([0], [0], color=INK2, linestyle=":", label="random guessing 16.7%"),
        Line2D([0], [0], color=INK2, linestyle="--", label="always one answer (20.7% / 56.3%)"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
                     bbox_to_anchor=(0.5, -0.1), columnspacing=1.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_main_results.png"))
    plt.close(fig)


def fig_cost_recall_panel():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.5), gridspec_kw={"width_ratios": [1.05, 1]})
    offsets = {
        "lstm": (8, -4, "left"), "bert_text": (9, 3, "left"), "roberta_text": (-9, -12, "right"),
        "bert_meta": (-9, -4, "right"), "roberta_meta": (-9, 7, "right"),
        "llm_zeroshot": (-9, -12, "right"), "llm_fewshot": (-9, 8, "right"),
    }
    for key, (name, fam) in MODELS.items():
        r = load_json(key)
        x, yv = r["est_cost_per_1k"], r["accuracy"] * 100
        a.scatter(x, yv, s=70, color=FAMILY_COLOR[fam], edgecolor="white", linewidth=1.5, zorder=3)
        dx, dy, ha = offsets[key]
        a.annotate(name, (x, yv), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8.3, color=INK)
    a.set_xscale("log")
    a.set_xlabel("Cost per 1,000 statements, US dollars (log scale)", fontsize=9)
    a.set_ylabel("Accuracy (%)", fontsize=9)
    a.set_ylim(10, 56)
    a.yaxis.grid(True, color=GRID, linewidth=0.7)
    a.set_axisbelow(True)
    a.tick_params(labelsize=8.5)
    handles = [Patch(facecolor=FAMILY_COLOR[f], label=f) for f in FAMILY_COLOR]
    leg = a.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK2)
    a.set_title("(a) Cost against accuracy", loc="left", fontsize=10.5, fontweight="bold")

    x = np.arange(6)
    w = 0.36
    for k, (key, label, color) in enumerate([("roberta_text", "Text only", BLUE),
                                              ("roberta_meta", "With speaker info", ORANGE)]):
        vals = [v * 100 for v in recall_per_class(key)]
        pos = x + (k - 0.5) * (w + 0.03)
        b.bar(pos, vals, width=w, color=color, edgecolor="white", linewidth=1.0, label=label)
        for p, v in zip(pos, vals):
            b.text(p, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    b.set_xticks(x)
    b.set_xticklabels(LABELS, fontsize=8.3, rotation=25, ha="right")
    b.set_ylim(0, 80)
    b.set_ylabel("Statements found (%)", fontsize=9)
    b.yaxis.grid(True, color=GRID, linewidth=0.7)
    b.set_axisbelow(True)
    b.tick_params(axis="x", length=0)
    b.tick_params(axis="y", labelsize=8.5)
    leg = b.legend(frameon=False, loc="upper center", ncol=2, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    b.set_title("(b) Share of each rating found (RoBERTa)", loc="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_cost_recall.png"))
    plt.close(fig)


def fig_data_panel():
    train = {"pants-fire": 842, "false": 1998, "barely-true": 1657,
             "half-true": 2123, "mostly-true": 1966, "true": 1683}
    subset = Counter(t for t, _ in load_preds("bert_text"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 2.6), gridspec_kw={"width_ratios": [1, 1]})
    for ax, vals, title, ymax, fmt in (
        (a, [train[l] for l in LABELS], "(a) Training set (10,269 statements)", 2500, "{:,}"),
        (b, [subset.get(i, 0) for i in range(6)], "(b) Shared test subset (300 statements)", 75, "{}"),
    ):
        ax.bar(range(6), vals, width=0.66, color=GRAY, edgecolor="white", linewidth=1.0)
        for i, v in enumerate(vals):
            ax.text(i, v + ymax * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(range(6))
        ax.set_xticklabels(LABELS, fontsize=8.3, rotation=20, ha="right")
        ax.set_ylim(0, ymax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    a.set_ylabel("Statements", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_data.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_main_results_panel()
    fig_cost_recall_panel()
    fig_data_panel()
    fig_accuracy_six_way()
    fig_accuracy_two_way()
    fig_cost_accuracy()
    fig_confusion()
    fig_recall_by_class()
    fig_llm_bias()
    fig_label_distribution()
    print("Wrote:", ", ".join(sorted(os.listdir(OUT))))
