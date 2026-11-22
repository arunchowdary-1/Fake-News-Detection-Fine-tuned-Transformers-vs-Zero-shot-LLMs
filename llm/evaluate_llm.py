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

# =====================================================================================
# FILE GUIDE (plain-English summary, added for understanding and viva preparation)
#
# WHAT THIS FILE DOES (the whole LLM experiment in one script)
#   1. Load the SAME 300 test statements every other model used.
#   2. Load the LLM once.
#   3. For each statement: ask the LLM, parse the answer, time it.
#   4. Score the answers (accuracy, macro-F1, confusion matrix) and compute speed, memory, cost.
#   5. Save two files: results/llm_<mode>.json (the numbers) and llm_<mode>_preds.csv (each answer).
#
# THE FLOW IN ONE LINE
#   load rows -> load model -> run_eval() -> score() -> build results -> save JSON + CSV
#
# HOW AN INVALID ANSWER IS HANDLED (important for the exam)
#   The parser returns "invalid" -> stored as -1 -> never equals a real label (0-5) -> counts as a
#   miss for its true class, but is NOT turned into a made-up prediction. So the LLM is never
#   given credit it did not earn.
#
# VIVA QUESTIONS FOR THIS FILE (with short answers)
#   Q: Why evaluate on the same 300 rows as BERT and the LSTM?
#   A: A fair comparison needs identical test data. Different rows could be easier or harder.
#   Q: Why only 300 and not all 1,283 test rows?
#   A: The 7B LLM is slow (about 0.4 to 0.7 seconds per statement here), so the full set would take much longer.
#   Q: What is macro-F1 and why use it?
#   A: The average of the F1 score of each of the 6 labels. It punishes a model that favours a few labels.
#   Q: What is a confusion matrix?
#   A: A table: rows = the real label, columns = what the model said. The diagonal is the correct answers.
#   Q: How is cost per 1,000 predictions computed?
#   A: Seconds per prediction x 1,000, converted to hours, times the GPU price ($0.35/h) plus 10% overhead.
# =====================================================================================

# argparse = command-line options; json = save results as JSON; os = check if a file exists;
# time = measure speed.
import argparse
import json
import os
import time

# scikit-learn's ready-made scoring functions (we reuse them instead of writing our own).
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)

# IDX_TO_LABEL: 0 -> "pants-fire" ... 5 -> "true". RESULTS_DIR: the "results/" folder.
from common.config import IDX_TO_LABEL, RESULTS_DIR
# Shared helpers: build the standard results dictionary and write the CSV / JSON files.
from common.evaluate import build_results_dict, save_predictions_csv, save_results_json
# "inference" (the module) is imported so we can read inference.RUNTIME_ERRORS at the end.
from llm import inference
from llm.inference import classify
from llm.parser import INVALID
from llm.prompt import DEFAULT_MODEL_ID, LABELS

# Turns a label name into its number: {"pants-fire": 0, "false": 1, ..., "true": 5}.
# enumerate() counts 0, 1, 2... alongside each label.
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
# Stored in the preds CSV; never equal to a real label (0-5), so an invalid output can only
# count as a miss. It is deliberately not mapped to any class, to avoid inventing predictions.
INVALID_IDX = -1

# Cost assumption from plan.md: T4 at $0.35/hour plus 10% overhead (same as the other models).
T4_PRICE_PER_HOUR = 0.35
# 1.10 = the price plus 10% extra.
OVERHEAD = 1.10
# Where the fixed list of the 300 test-row numbers is saved.
TEST_INDICES_PATH = "common/test_indices.txt"


# ---- load_shared_rows -----------------------------------------------------------
# WHAT: get the 300 shared test statements and their true labels (as numbers 0-5).
# RETURNS: (indices, rows). Example row: {"statement": "...", "y_true": 3}
def load_shared_rows(indices_path: str = TEST_INDICES_PATH) -> tuple[list[int], list[dict]]:
    """Return (indices, rows) for the shared test subset, in the project label order."""
    # Imported here so the pure helpers below can be tested without `datasets`.
    from common.data_loader import get_test_indices, label_to_idx, load_liar, load_test_indices

    # load_liar() returns (train, validation, test); [2] picks the test split.
    test = load_liar()[2]
    # If the shared indices file exists, use it so EVERY model sees exactly the same rows.
    if os.path.exists(indices_path):
        indices = load_test_indices(indices_path)
    else:
        # Same seed and function as the other models, so the rows still match.
        print(f"WARNING: {indices_path} not found; regenerating the subset with the shared function.")
        indices = get_test_indices(test)
    # For each chosen row number i, keep the statement text and the label converted to 0-5.
    rows = [
        {"statement": test[i]["statement"], "y_true": label_to_idx(test[i]["label"])}
        for i in indices
    ]
    return indices, rows


# ---- load_few_shot_examples -----------------------------------------------------
# WHAT: pick the 6 solved examples (one per label) shown to the LLM in few-shot mode.
def load_few_shot_examples() -> list[dict]:
    """One TRAIN-split example per class (seed 42), shared by every few-shot run.

    Drawn from train so no example can also be a test statement; fixed seed so every run
    (and every test statement within a run) sees the same prompt.
    """
    from common.data_loader import label_to_idx, load_liar
    from llm.fewshot import select_examples

    # [0] picks the TRAIN split (never the test split).
    train = load_liar()[0]
    # select_examples wants statements and label numbers; label_to_idx converts each label.
    return select_examples(train["statement"], [label_to_idx(l) for l in train["label"]])


# ---- run_eval -------------------------------------------------------------------
# WHAT: the main loop. Ask the model about every row and time each answer.
# RETURNS: predictions (numbers 0-5, or -1 for invalid) and the time per row in milliseconds.
def run_eval(rows: list[dict], generate_fn, examples: list[dict] | None = None) -> tuple[list[int], list[float]]:
    """Classify every row. Returns (predictions, latencies in ms); invalid -> -1."""
    preds, latencies = [], []
    for r in rows:
        # perf_counter is a precise stopwatch. We time one full classify() call.
        start = time.perf_counter()
        # No retry: decoding is greedy, so a second try would give the same answer.
        label = classify(r["statement"], generate_fn, max_retries=0, examples=examples)
        # seconds x 1000 = milliseconds.
        latencies.append((time.perf_counter() - start) * 1000)
        # Invalid outputs stay -1 and are scored as wrong later; they are never guessed.
        # .get(label, INVALID_IDX) = "look up the number, or use -1 if the label is unknown".
        preds.append(LABEL_TO_IDX.get(label, INVALID_IDX) if label != INVALID else INVALID_IDX)
    return preds, latencies


# ---- score ----------------------------------------------------------------------
# WHAT: turn lists of real labels and predictions into all the metrics.
# EXAMPLE: score([5, 1, 3], [5, 1, -1]) -> accuracy 2/3 (the -1 is simply a miss).
def score(y_true: list[int], preds: list[int]) -> dict:
    """Metrics with invalid outputs counted as wrong, without inventing predictions.

    An invalid output (-1) never equals a real label, so it is a miss for its true
    class (lower recall/accuracy) but adds no false positive to any class. Metrics are
    restricted to the 6 real labels, so -1 is not treated as an extra class.
    `confusion_matrix` is 6x6 over valid predictions only (rows sum to support minus
    invalid); `confusion_matrix_with_invalid` is 6x7, last column = invalid outputs.
    Same keys as the shared evaluate().
    """
    # Safety check: scoring needs two lists of the same, non-zero length.
    if len(y_true) != len(preds) or not y_true:
        raise ValueError("y_true and preds must be non-empty and the same length")
    # Passing labels= explicitly keeps the report 6-wide even if a class is absent from
    # y_true/preds, and keeps -1 out of the metrics.
    labels = list(range(len(LABELS)))
    # Confusion matrix: rows = real label, columns = predicted label.
    cm =confusion_matrix(y_true, preds, labels=labels)
    # For each class c, count the statements of class c that got an invalid answer.
    invalid_by_class = [sum(1 for t, p in zip(y_true, preds) if t == c and p == INVALID_IDX) for c in labels]
    # A per-class table (precision, recall, F1). zero_division=0 avoids errors for empty classes.
    report = classification_report(
        y_true, preds, labels=labels, target_names=[IDX_TO_LABEL[i] for i in labels],
        output_dict=True, zero_division=0,
    )
    # float(...) converts numpy numbers into plain Python numbers so JSON can save them.
    # average="macro" = simple mean over the 6 classes; "weighted" = mean weighted by class size.
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, preds, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, preds, labels=labels, average="macro", zero_division=0)),
        # .tolist() turns the numpy matrix into normal lists (JSON cannot store numpy arrays).
        "confusion_matrix": cm.tolist(),
        # Add one extra column per row (the number of invalid answers) to make a 6x7 table.
        "confusion_matrix_with_invalid": [row + [inv] for row, inv in zip(cm.tolist(), invalid_by_class)],
        "per_class_metrics": report,
        "n_samples": len(y_true),
    }


# ---- reset_peak_memory / peak_memory_mb -----------------------------------------
# WHAT: measure the most GPU memory used during the run (a number we report in the cost table).
# HOW : reset the counter before loading the model, read it after the last prediction.
def reset_peak_memory(torch) -> None:
    """Reset peak-memory counters on every visible GPU (the model may be split across several).

    Falls back to the current device if per-device calls are rejected by this torch build.
    """
    # try/except = "try this; if it fails, do the backup plan instead".
    try:
        # A big model may be split over several GPUs, so reset each one.
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
        # Add up the peak of every GPU.
        total = sum(torch.cuda.max_memory_allocated(torch.device("cuda", i))
                    for i in range(torch.cuda.device_count()))
    except Exception:
        total = torch.cuda.max_memory_allocated()
    # 2**20 = 1,048,576 bytes = 1 MB. round(..., 1) keeps one decimal place.
    return round(total / 2**20, 1)


# ---- cost_per_1k ----------------------------------------------------------------
# WHAT: dollars to classify 1,000 statements on a T4 GPU.
# WORKED EXAMPLE: 661 ms per statement (the few-shot run)
#   661 / 1000      = 0.661 seconds per statement   (milliseconds -> seconds)
#   0.661 / 3600    = 0.000184 hours per statement  (seconds -> hours)
#   x 1000          = 0.184 hours for 1,000 statements
#   x $0.35 x 1.10  = about $0.0707      (matches the cost table in the report)
def cost_per_1k(avg_latency_ms: float) -> float:
    """GPU cost of 1000 predictions at batch size 1.

    ms/prediction -> hours for 1000 predictions, times the T4 hourly rate with overhead.
    Batch size 1 matches how the latency was measured; batching would lower the real cost.
    """
    return avg_latency_ms / 1000 / 3600 * 1000 * T4_PRICE_PER_HOUR * OVERHEAD


# ---- main (command-line entry) --------------------------------------------------
# WHAT: run one full experiment (zero-shot or few-shot) and save the results.
def main() -> None:
    """CLI: run one mode on the shared test subset and write the preds CSV and results JSON."""
    # Command-line options. Example: python -m llm.evaluate_llm --4bit --mode fewshot
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

    # Name of this run, e.g. "llm_fewshot". It becomes the output file name.
    run_name = f"llm_{args.mode}"
    indices, rows = load_shared_rows(args.indices)
    if args.limit:
        # [: N] keeps only the first N items. Used for quick checks on a few rows.
        indices, rows = indices[: args.limit], rows[: args.limit]
        run_name += "_limit"  # never overwrite the real result with a partial run
    # The real answers, as a list of numbers 0-5.
    y_true = [r["y_true"] for r in rows]
    # Few-shot mode loads the 6 examples; zero-shot passes None (no examples).
    examples = load_few_shot_examples() if args.mode == "fewshot" else None
    if examples:
        # Print only the first 50 characters of each example, to keep the log short.
        print("Few-shot examples:", [(e["label"], e["statement"][:50]) for e in examples])

    # Is a GPU available? CPU runs still work, but we skip the GPU-only measurements.
    cuda = torch.cuda.is_available()
    if cuda:
        # Reset before loading the model so the peak covers load + inference of this run only.
        reset_peak_memory(torch)
    # Time how long loading the model takes (reported separately from the per-statement speed).
    t0 = time.perf_counter()
    gen = make_generate_fn(args.model, args.use_4bit)
    load_s = time.perf_counter() - t0

    # The main loop: answers and timings for all rows.
    preds, latencies = run_eval(rows, gen, examples)

    # Metrics, average time per statement, and which GPU was used.
    metrics = score(y_true, preds)
    avg_ms = sum(latencies) / len(latencies)
    gpu = torch.cuda.get_device_name(0) if cuda else "CPU"
    # Build the standard results dictionary (same fields as BERT, RoBERTa and the LSTM).
    # getattr(gen, "n_params", 0) reads the parameter count we attached to the function.
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
    # Print a short summary on screen: the full details are in the saved JSON.
    print(json.dumps({k: results[k] for k in ("accuracy", "macro_f1", "invalid_outputs", "n_eval",
                                                "infer_seconds_per_1000", "gpu_peak_mem_mb")}, indent=2))


# Run main() only when this file is started directly.
if __name__ == "__main__":
    main()

# EXTRA VIVA QUESTIONS (whole file)
#   Q: Why is the few-shot run slower than zero-shot?
#   A: Every request carries six extra examples, so the prompt is longer (661 s vs 393 s per 1,000).
#   Q: Why is greedy decoding used?
#   A: It is deterministic, so the same run can be repeated and compared fairly.
#   Q: What does "invalid_outputs" tell us?
#   A: How many answers could not be turned into exactly one label. Zero-shot had 1 of 300, few-shot 0.
#   Q: Why does a partial --limit run get a different file name?
#   A: So a quick check can never overwrite the real result.
#   Q: What is the difference between runtime_errors and invalid_outputs?
#   A: runtime_errors = the program or GPU crashed on a call. invalid_outputs = the model answered
#      but not with one clean label.
#   Q: Why report GPU memory?
#   A: It shows what hardware is needed. The 7B model needs about 5.7 GB even in 4-bit, BERT about 2 GB.
