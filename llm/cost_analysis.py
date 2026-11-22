"""Cost-vs-accuracy comparison: fine-tuned small models vs prompted 7B LLM.

Reads the per-run JSON files in results/ (skips any that do not exist yet) and writes
    results/cost_comparison.json   all numbers
    results/cost_comparison.md     tables for the report
    results/cost_vs_accuracy.csv   one row per model, for the cost-vs-accuracy plot

Cost model (same for every model): T4 at $0.35/hour + 10% overhead.
  * one-off cost  = training minutes priced at that rate (0 for prompted LLMs)
  * running cost  = est_cost_per_1k from each results file (batch size 1)
  * break-even    = predictions after which a fine-tuned model's one-off training
                    cost is paid back by its cheaper inference vs the LLM

Usage:  python -m llm.cost_analysis
"""
import csv
import json
import os

from common.config import RESULTS_DIR

# Same rate and overhead as llm/evaluate_llm.py, so fine-tuned and prompted models are priced
# on one scale. Keep the two files in sync if the assumption changes.
T4_PRICE_PER_HOUR = 0.35
OVERHEAD = 1.10

# (results file stem, display name, kind). "kind" decides which rows are compared against
# which: every fine-tuned row is compared with the best prompted (LLM) row.
RUNS = [
    ("lstm", "BiLSTM", "fine-tuned"),
    ("bert_text", "BERT text-only", "fine-tuned"),
    ("bert_meta", "BERT + metadata", "fine-tuned"),
    ("roberta_text", "RoBERTa text-only", "fine-tuned"),
    ("roberta_meta", "RoBERTa + metadata", "fine-tuned"),
    ("llm_zeroshot", "Qwen2.5-7B zero-shot", "prompted"),
    ("llm_fewshot", "Qwen2.5-7B few-shot", "prompted"),
]


def train_cost(train_minutes: float) -> float:
    """One-off training cost in USD at the shared T4 rate."""
    return train_minutes / 60 * T4_PRICE_PER_HOUR * OVERHEAD


def break_even_predictions(ft_train_cost: float, ft_per_1k: float, llm_per_1k: float) -> float | None:
    """Predictions after which fine-tuning is cheaper in total than prompting the LLM.

    None if fine-tuned inference is not cheaper per prediction (never pays back).
    Total cost after n predictions: ft = train_cost + n * ft_per_1k/1000 and
    llm = n * llm_per_1k/1000 (the LLM has no training cost), so they are equal at
    n = train_cost / saving_per_prediction.
    """
    saving_per_pred = (llm_per_1k - ft_per_1k) / 1000
    if saving_per_pred <= 0:
        return None
    return ft_train_cost / saving_per_pred


def load_rows(results_dir: str = RESULTS_DIR) -> list[dict]:
    """Read each existing results JSON in RUNS order into one flat row per model.

    Result files come from different scripts, so a few fields have more than one name; the
    `or` chains below accept either spelling and fall back to avg_latency_ms.
    """
    rows = []
    for stem, name, kind in RUNS:
        path = os.path.join(results_dir, f"{stem}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        rows.append({
            "run": stem, "name": name, "kind": kind,
            "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
            "params": r["params"], "train_minutes": r["train_minutes"],
            "train_cost_usd": round(train_cost(r["train_minutes"]), 4),
            "ms_per_statement": r["avg_latency_ms"],
            # LLM files use infer_seconds_per_1000, BERT files inference_sec_per_1000. At
            # batch size 1, ms per statement equals seconds per 1,000 statements, hence the
            # last fallback.
            "seconds_per_1000": r.get("infer_seconds_per_1000") or r.get("inference_sec_per_1000") or r["avg_latency_ms"],
            "cost_per_1k_usd": r["est_cost_per_1k"],
            # Peak memory is named differently in LLM (gpu_peak_mem_mb) and BERT
            # (gpu_mem_inference_mb) files; older/other runs may have neither (-> None).
            "gpu": r["gpu"], "gpu_peak_mem_mb": r.get("gpu_peak_mem_mb") or r.get("gpu_mem_inference_mb"),
            "invalid_outputs": r.get("invalid_outputs", 0), "n_eval": r["n_eval"],
        })
    return rows


def _ratio(a: float, b: float) -> float | None:
    """a / b rounded, or None when b is 0 (a cost that rounds to zero)."""
    return round(a / b, 1) if b else None


def add_comparisons(rows: list[dict]) -> list[dict]:
    """Per fine-tuned row: ratios and break-even vs the best-macro-F1 LLM run.

    The reference is the strongest LLM (zero-shot or few-shot) by macro-F1, so the
    comparison is against the LLM's best case, not a weaker prompt. Returns [] if no LLM
    results exist yet.
    """
    llms = [r for r in rows if r["kind"] == "prompted"]
    if not llms:
        return []
    ref = max(llms, key=lambda r: r["macro_f1"])
    out = []
    for r in rows:
        if r["kind"] != "fine-tuned":
            continue
        out.append({
            "finetuned": r["name"], "vs_llm": ref["name"],
            "params_ratio": round(ref["params"] / r["params"], 1) if r["params"] else None,
            "speedup": _ratio(ref["seconds_per_1000"], r["seconds_per_1000"]),
            "cost_ratio": _ratio(ref["cost_per_1k_usd"], r["cost_per_1k_usd"]),
            "macro_f1_gain": round(r["macro_f1"] - ref["macro_f1"], 4),
            "break_even_predictions": break_even_predictions(
                r["train_cost_usd"], r["cost_per_1k_usd"], ref["cost_per_1k_usd"]),
        })
    return out


def to_markdown(rows: list[dict], comps: list[dict]) -> str:
    """Render the per-model table and the fine-tuned-vs-LLM table pasted into the report."""
    def fmt(v, f="{}"):
        return "n/a" if v is None else f.format(v)

    # Models timed on CPU are still priced at the T4 rate (same formula for everyone), so their
    # speed and cost ratios are not like-for-like with the GPU models. Mark them with a dagger.
    cpu_names = {r["name"] for r in rows if "cpu" in str(r["gpu"]).lower()}

    def label(name):
        return name + " †" if name in cpu_names else name

    lines = ["| Model | Acc | Macro-F1 | Params | Train (min) | s / 1,000 stmts | $ / 1,000 | GPU peak MB | Invalid |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {label(r['name'])} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['params']/1e6:,.0f}M | "
                     f"{r['train_minutes']:.1f} | {r['seconds_per_1000']:.1f} | {r['cost_per_1k_usd']:.4f} | "
                     f"{fmt(r['gpu_peak_mem_mb'], '{:,.0f}')} | {r['invalid_outputs']}/{r['n_eval']} |")
    if comps:
        lines += ["", f"Fine-tuned vs {comps[0]['vs_llm']} (the better LLM run by macro-F1):", "",
                  "| Fine-tuned model | Params smaller | Faster | Cheaper / prediction | Macro-F1 gain | Break-even (predictions) |",
                  "|---|---|---|---|---|---|"]
        for c in comps:
            be = "never" if c["break_even_predictions"] is None else f"{c['break_even_predictions']:,.0f}"
            lines.append(f"| {label(c['finetuned'])} | {fmt(c['params_ratio'], '{}x')} | {fmt(c['speedup'], '{}x')} | "
                         f"{fmt(c['cost_ratio'], '{}x')} | {c['macro_f1_gain']:+.3f} | {be} |")
    if cpu_names:
        lines += ["", "† Timed on CPU but priced at the T4 rate ($0.35/h + 10% overhead), so its speed and "
                      "cost ratios are not like-for-like with the GPU models."]
    return "\n".join(lines) + "\n"


def main(results_dir: str = RESULTS_DIR) -> None:
    """Write the JSON, Markdown and CSV outputs (results_dir is a parameter for tests)."""
    rows = load_rows(results_dir)
    if not rows:
        raise SystemExit(f"No result files found in {results_dir}")
    comps = add_comparisons(rows)
    with open(os.path.join(results_dir, "cost_comparison.json"), "w", encoding="utf-8") as f:
        json.dump({"cost_assumption": f"T4 ${T4_PRICE_PER_HOUR}/h + {round((OVERHEAD - 1) * 100)}% overhead",
                   "models": rows, "finetuned_vs_llm": comps}, f, indent=2)
    md = to_markdown(rows, comps)
    with open(os.path.join(results_dir, "cost_comparison.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(results_dir, "cost_vs_accuracy.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "kind", "params", "cost_per_1k_usd", "seconds_per_1000", "accuracy", "macro_f1"])
        for r in rows:
            w.writerow([r["name"], r["kind"], r["params"], r["cost_per_1k_usd"],
                        r["seconds_per_1000"], r["accuracy"], r["macro_f1"]])
    print(md)


if __name__ == "__main__":
    main()
