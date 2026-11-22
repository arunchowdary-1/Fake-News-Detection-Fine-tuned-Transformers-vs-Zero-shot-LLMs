"""Full zero-shot LLM evaluation (issue #8) on the shared test subset.

Uses the same pieces as the LSTM and BERT scripts so the results are directly
comparable: the Hugging Face LIAR loader, the fixed test indices in
common/test_indices.txt, and the shared evaluate() / results writers.

Outputs (plan.md section 4):
    results/llm_<mode>_preds.csv   id,y_true,y_pred,latency_ms   (invalid output = -1)
    results/llm_<mode>.json        shared results schema + invalid_outputs, GPU memory,
                                   seconds per 1,000 statements

Example (T4 GPU, 4-bit):
    python -m llm.evaluate_llm --4bit --mode zeroshot   # -> results/llm_zeroshot.*
    python -m llm.evaluate_llm --4bit --mode fewshot    # -> results/llm_fewshot.*

Auth: set HF_TOKEN in the environment if the model needs it; never hardcode it.
"""
import argparse
import json
import os
import time

from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)

from common.config import IDX_TO_LABEL, RESULTS_DIR
from common.evaluate import build_results_dict, save_predictions_csv, save_results_json
from llm import inference
from llm.inference import classify
from llm.parser import INVALID
from llm.prompt import DEFAULT_MODEL_ID, LABELS

LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
# Stored in the preds CSV; never equal to a real label (0-5), so an invalid output can only
# count as a miss. It is deliberately not mapped to any class, to avoid inventing predictions.
INVALID_IDX = -1

# Cost assumption from plan.md: T4 at $0.35/hour plus 10% overhead (same as the other models).
T4_PRICE_PER_HOUR = 0.35
OVERHEAD = 1.10
TEST_INDICES_PATH = "common/test_indices.txt"


def load_shared_rows(indices_path: str = TEST_INDICES_PATH) -> tuple[list[int], list[dict]]:
    """Return (indices, rows) for the shared test subset, in the project label order."""
    # Imported here so the pure helpers below can be tested without `datasets`.
    from common.data_loader import get_test_indices, label_to_idx, load_liar, load_test_indices

    test = load_liar()[2]
    if os.path.exists(indices_path):
        indices = load_test_indices(indices_path)
    else:
        # Same seed and function as the other models, so the rows still match.
        print(f"WARNING: {indices_path} not found; regenerating the subset with the shared function.")
        indices = get_test_indices(test)
    rows = [
        {"statement": test[i]["statement"], "y_true": label_to_idx(test[i]["label"])}
        for i in indices
    ]
    return indices, rows


def load_few_shot_examples() -> list[dict]:
    """One TRAIN-split example per class (seed 42), shared by every few-shot run.

    Drawn from train so no example can also be a test statement; fixed seed so every run
    (and every test statement within a run) sees the same prompt.
    """
    from common.data_loader import label_to_idx, load_liar
    from llm.fewshot import select_examples

    train = load_liar()[0]
    return select_examples(train["statement"], [label_to_idx(l) for l in train["label"]])


def run_eval(rows: list[dict], generate_fn, examples: list[dict] | None = None) -> tuple[list[int], list[float]]:
    """Classify every row. Returns (predictions, latencies in ms); invalid -> -1."""
    preds, latencies = [], []
    for r in rows:
        start = time.perf_counter()
        # No retry: decoding is greedy, so a second try would give the same answer.
        label = classify(r["statement"], generate_fn, max_retries=0, examples=examples)
        latencies.append((time.perf_counter() - start) * 1000)
        # Invalid outputs stay -1 and are scored as wrong later; they are never guessed.
        preds.append(LABEL_TO_IDX.get(label, INVALID_IDX) if label != INVALID else INVALID_IDX)
    return preds, latencies


def score(y_true: list[int], preds: list[int]) -> dict:
    """Metrics with invalid outputs counted as wrong, without inventing predictions.

    An invalid output (-1) never equals a real label, so it is a miss for its true
    class (lower recall/accuracy) but adds no false positive to any class. Metrics are
    restricted to the 6 real labels, so -1 is not treated as an extra class.
    `confusion_matrix` is 6x6 over valid predictions only (rows sum to support minus
    invalid); `confusion_matrix_with_invalid` is 6x7, last column = invalid outputs.
    Same keys as the shared evaluate().
    """
    if len(y_true) != len(preds) or not y_true:
        raise ValueError("y_true and preds must be non-empty and the same length")
    # Passing labels= explicitly keeps the report 6-wide even if a class is absent from
    # y_true/preds, and keeps -1 out of the metrics.
    labels = list(range(len(LABELS)))
    cm =confusion_matrix(y_true, preds, labels=labels)
    invalid_by_class = [sum(1 for t, p in zip(y_true, preds) if t == c and p == INVALID_IDX) for c in labels]
    report = classification_report(
        y_true, preds, labels=labels, target_names=[IDX_TO_LABEL[i] for i in labels],
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, preds, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, preds, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_with_invalid": [row + [inv] for row, inv in zip(cm.tolist(), invalid_by_class)],
        "per_class_metrics": report,
        "n_samples": len(y_true),
    }


def reset_peak_memory(torch) -> None:
    """Reset peak-memory counters on every visible GPU (the model may be split across several).

    Falls back to the current device if per-device calls are rejected by this torch build.
    """
    try:
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(torch.device("cuda", i))
    except Exception:
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(torch) -> float:
    """Peak allocated GPU memory in MB, summed over visible GPUs (current device on fallback).

    Summed because with device_map="auto" the 7B model may be sharded over several GPUs
    (e.g. 2x T4 on Kaggle); the current device alone would under-report the footprint.
    """
    try:
        total = sum(torch.cuda.max_memory_allocated(torch.device("cuda", i))
                    for i in range(torch.cuda.device_count()))
    except Exception:
        total = torch.cuda.max_memory_allocated()
    return round(total / 2**20, 1)


def cost_per_1k(avg_latency_ms: float) -> float:
    """GPU cost of 1000 predictions at batch size 1.

    ms/prediction -> hours for 1000 predictions, times the T4 hourly rate with overhead.
    Batch size 1 matches how the latency was measured; batching would lower the real cost.
    """
    return avg_latency_ms / 1000 / 3600 * 1000 * T4_PRICE_PER_HOUR * OVERHEAD


def main() -> None:
    """CLI: run one mode on the shared test subset and write the preds CSV and results JSON."""
    ap =argparse.ArgumentParser()
    ap.add_argument("--indices", default=TEST_INDICES_PATH)
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--mode", choices=["zeroshot", "fewshot"], default="zeroshot")
    ap.add_argument("--4bit", dest="use_4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (quick checks)")
    ap.add_argument("--out", default=RESULTS_DIR)
    args = ap.parse_args()

    import torch  # GPU name + memory

    from llm.smoke_test import make_generate_fn

    run_name = f"llm_{args.mode}"
    indices, rows = load_shared_rows(args.indices)
    if args.limit:
        indices, rows = indices[: args.limit], rows[: args.limit]
        run_name += "_limit"  # never overwrite the real result with a partial run
    y_true = [r["y_true"] for r in rows]
    examples = load_few_shot_examples() if args.mode == "fewshot" else None
    if examples:
        print("Few-shot examples:", [(e["label"], e["statement"][:50]) for e in examples])

    cuda = torch.cuda.is_available()
    if cuda:
        # Reset before loading the model so the peak covers load + inference of this run only.
        reset_peak_memory(torch)
    t0 = time.perf_counter()
    gen = make_generate_fn(args.model, args.use_4bit)
    load_s = time.perf_counter() - t0

    preds, latencies = run_eval(rows, gen, examples)

    metrics = score(y_true, preds)
    avg_ms = sum(latencies) / len(latencies)
    gpu = torch.cuda.get_device_name(0) if cuda else "CPU"
    results = build_results_dict(
        model_name=run_name,
        metrics=metrics,
        params=getattr(gen, "n_params", 0),
        train_minutes=0,
        gpu=gpu,
        avg_latency_ms=round(avg_ms, 3),
        est_cost_per_1k=round(cost_per_1k(avg_ms), 6),
        cost_assumption="T4: $0.35/hour incl. 10% overhead; batch size 1",
        invalid_outputs=sum(p == INVALID_IDX for p in preds),
        notes=(f"{args.mode}, greedy decoding, no retry, model={args.model}, 4bit={args.use_4bit}"
               + (f", {len(examples)} few-shot examples (1/class, train split, seed 42)" if examples else "")),
    )
    # Extra fields (the shared schema has no slot for these), used by llm/cost_analysis.py
    # and the report:
    #   mode / llm_model_id / quantization   which configuration produced the numbers
    #   infer_seconds_per_1000 / total_infer_seconds / load_seconds   speed and cost inputs
    #   gpu_peak_mem_mb                      memory footprint (None on CPU)
    #   few_shot_examples                    the exact prompt examples, for reproducibility
    #   runtime_errors                       crashes in generate(), told apart from bad outputs
    #   confusion_matrix_with_invalid        6x7 matrix, last column = invalid outputs
    results.update({
        "mode": args.mode,
        "llm_model_id": args.model,
        "quantization": "4bit-fp4" if args.use_4bit else "none",
        "infer_seconds_per_1000": round(avg_ms, 3),  # ms/sample == s per 1,000 samples
        "total_infer_seconds": round(sum(latencies) / 1000, 2),
        "load_seconds": round(load_s, 1),
        "gpu_peak_mem_mb": peak_memory_mb(torch) if cuda else None,
        "few_shot_examples": examples,
        "runtime_errors": inference.RUNTIME_ERRORS,  # failed generate calls (not parse failures)
        "confusion_matrix_with_invalid": metrics["confusion_matrix_with_invalid"],
    })
    # JSON after every run, as the team rules require.
    save_results_json(results, run_name, args.out)
    save_predictions_csv(indices, y_true, preds, latencies, run_name, args.out)
    print(json.dumps({k: results[k] for k in ("accuracy", "macro_f1", "invalid_outputs", "n_eval",
                                                "infer_seconds_per_1000", "gpu_peak_mem_mb")}, indent=2))


if __name__ == "__main__":
    main()
