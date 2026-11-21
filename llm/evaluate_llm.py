"""Full zero-shot LLM evaluation (issue #8): writes the shared results files.

Outputs (formats from plan.md section 4):
    results/llm_preds.csv   id,y_true,y_pred,latency_ms
    results/llm.json        accuracy, macro_f1, latency, cost, invalid count...

Example (T4 GPU, 4-bit):
    python -m llm.evaluate_llm --data path/to/test.tsv --4bit

Rows come from Person A's fixed test subset if --indices is given (one TSV row
position per line, e.g. common/test_indices.txt). Otherwise a seeded stratified
sample of --n rows is used so the run is still repeatable.

Auth: set HF_TOKEN in the environment if the model needs it; never hardcode it.
"""
import argparse
import csv
import json
import os
import time

from llm.inference import classify
from llm.parser import INVALID
from llm.prompt import DEFAULT_MODEL_ID, LABELS
from llm.smoke_test import load_liar_tsv, make_generate_fn, stratified_sample

LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
INVALID_IDX = -1  # an invalid output is never a correct prediction

# Cost assumption from plan.md: T4 at $0.35/hour plus 10% overhead.
T4_USD_PER_HOUR = 0.35
OVERHEAD = 1.10


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    """Macro-averaged F1 over all 6 labels (invalid predictions count as wrong)."""
    scores = []
    for c in range(len(LABELS)):
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return sum(scores) / len(scores)


def run_eval(rows: list[dict], generate_fn) -> list[dict]:
    """Classify every row, timing each one. Returns one record per row."""
    records = []
    for r in rows:
        start = time.perf_counter()
        pred = classify(r["statement"], generate_fn)
        latency_ms = (time.perf_counter() - start) * 1000
        records.append({
            "id": r["id"],
            "y_true": LABEL_TO_IDX[r["label"]],
            "y_pred": LABEL_TO_IDX.get(pred, INVALID_IDX),
            "latency_ms": round(latency_ms, 1),
        })
    return records


def summarize(records: list[dict], model_id: str, gpu: str, params: int | None) -> dict:
    """Build the results JSON from per-row records."""
    n = len(records)
    y_true = [r["y_true"] for r in records]
    y_pred = [r["y_pred"] for r in records]
    avg_ms = sum(r["latency_ms"] for r in records) / n
    # Cost per 1000 predictions = GPU time for 1000 samples * hourly rate.
    cost_per_1k = (avg_ms * 1000 / 3_600_000) * T4_USD_PER_HOUR * OVERHEAD
    return {
        "model": "llm",
        "accuracy": round(sum(t == p for t, p in zip(y_true, y_pred)) / n, 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "params": params,
        "train_minutes": 0,
        "gpu": gpu,
        "avg_latency_ms": round(avg_ms, 1),
        "est_cost_per_1k": round(cost_per_1k, 4),
        "cost_assumption": "T4: $0.35/hour, includes 10% overhead",
        "n_eval": n,
        "invalid_outputs": sum(p == INVALID_IDX for p in y_pred),
        "notes": f"Zero-shot, greedy decoding, model={model_id}",
    }


def write_results(records: list[dict], summary: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "llm_preds.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "y_true", "y_pred", "latency_ms"])
        w.writeheader()
        w.writerows(records)
    with open(os.path.join(out_dir, "llm.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def load_indices(path: str) -> list[int]:
    with open(path, encoding="utf-8") as f:
        return [int(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to LIAR test.tsv")
    ap.add_argument("--indices", help="fixed test subset: one TSV row position per line")
    ap.add_argument("--n", type=int, default=300, help="sample size if no --indices")
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--4bit", dest="use_4bit", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    all_rows = load_liar_tsv(args.data)
    if args.indices:
        rows = [all_rows[i] for i in load_indices(args.indices)]
    else:
        rows = stratified_sample(all_rows, args.n)

    import torch  # only needed for the GPU name and parameter count

    gen = make_generate_fn(args.model, args.use_4bit)
    records = run_eval(rows, gen)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    summary = summarize(records, args.model, gpu, params=None)
    write_results(records, summary, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
