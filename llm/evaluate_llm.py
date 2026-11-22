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

from common.config import IDX_TO_LABEL, RESULTS_DIR
from common.evaluate import build_results_dict, evaluate, save_predictions_csv, save_results_json
from llm.inference import classify
from llm.parser import INVALID
from llm.prompt import DEFAULT_MODEL_ID, LABELS

LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
INVALID_IDX = -1  # stored in the preds CSV; never equal to a real label

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
    """One TRAIN-split example per class (seed 42), shared by every few-shot run."""
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
        preds.append(LABEL_TO_IDX.get(label, INVALID_IDX) if label != INVALID else INVALID_IDX)
    return preds, latencies


def score(y_true: list[int], preds: list[int]) -> dict:
    """Shared evaluate(), with invalid outputs counted as wrong.

    sklearn would treat -1 as an extra class and drag macro-F1 down, so each
    invalid prediction is swapped for a label that is guaranteed to be wrong.
    The real outputs (with -1) are still what gets written to the CSV.
    """
    scored = [p if p != INVALID_IDX else (t + 1) % len(LABELS) for t, p in zip(y_true, preds)]
    return evaluate(y_true, scored)


def cost_per_1k(avg_latency_ms: float) -> float:
    """GPU cost of 1000 predictions at batch size 1."""
    return avg_latency_ms / 1000 / 3600 * 1000 * T4_PRICE_PER_HOUR * OVERHEAD


def main() -> None:
    ap = argparse.ArgumentParser()
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
        for i in range(torch.cuda.device_count()):  # device_map="auto" may spread the model over GPUs
            torch.cuda.reset_peak_memory_stats(i)
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
    # Extra cost fields (the shared schema has no slot for these).
    results.update({
        "mode": args.mode,
        "llm_model_id": args.model,
        "quantization": "4bit-nf4" if args.use_4bit else "none",
        "infer_seconds_per_1000": round(avg_ms, 3),  # ms/sample == s per 1,000 samples
        "total_infer_seconds": round(sum(latencies) / 1000, 2),
        "load_seconds": round(load_s, 1),
        "gpu_peak_mem_mb": round(sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**20, 1) if cuda else None,
        "few_shot_examples": examples,
    })
    # JSON after every run, as the team rules require.
    save_results_json(results, run_name, args.out)
    save_predictions_csv(indices, y_true, preds, latencies, run_name, args.out)
    print(json.dumps({k: results[k] for k in ("accuracy", "macro_f1", "invalid_outputs", "n_eval",
                                                "infer_seconds_per_1000", "gpu_peak_mem_mb")}, indent=2))


if __name__ == "__main__":
    main()
