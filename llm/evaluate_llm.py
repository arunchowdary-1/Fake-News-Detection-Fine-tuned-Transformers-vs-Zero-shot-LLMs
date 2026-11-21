"""Full zero-shot LLM evaluation (issue #8) on the shared test subset.

Uses the same pieces as the LSTM and BERT scripts so the results are directly
comparable: the Hugging Face LIAR loader, the fixed test indices in
common/test_indices.txt, and the shared evaluate() / results writers.

Outputs (plan.md section 4):
    results/llm_preds.csv   id,y_true,y_pred,latency_ms   (invalid output = -1)
    results/llm.json        shared results schema + invalid_outputs count

Example (T4 GPU, 4-bit):
    python -m llm.evaluate_llm --4bit

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


def run_eval(rows: list[dict], generate_fn) -> tuple[list[int], list[float]]:
    """Classify every row. Returns (predictions, latencies in ms); invalid -> -1."""
    preds, latencies = [], []
    for r in rows:
        start = time.perf_counter()
        # No retry: decoding is greedy, so a second try would give the same answer.
        label = classify(r["statement"], generate_fn, max_retries=0)
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
    ap.add_argument("--4bit", dest="use_4bit", action="store_true")
    ap.add_argument("--out", default=RESULTS_DIR)
    args = ap.parse_args()

    import torch  # only needed for the GPU name

    from llm.smoke_test import make_generate_fn

    indices, rows = load_shared_rows(args.indices)
    y_true = [r["y_true"] for r in rows]

    gen = make_generate_fn(args.model, args.use_4bit)
    preds, latencies = run_eval(rows, gen)

    metrics = score(y_true, preds)
    avg_ms = sum(latencies) / len(latencies)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    results = build_results_dict(
        model_name="llm",
        metrics=metrics,
        params=getattr(gen, "n_params", 0),
        train_minutes=0,
        gpu=gpu,
        avg_latency_ms=round(avg_ms, 3),
        est_cost_per_1k=round(cost_per_1k(avg_ms), 6),
        cost_assumption="T4: $0.35/hour incl. 10% overhead; batch size 1",
        invalid_outputs=sum(p == INVALID_IDX for p in preds),
        notes=f"Zero-shot, greedy decoding, no retry, model={args.model}, 4bit={args.use_4bit}",
    )
    save_results_json(results, "llm", args.out)
    save_predictions_csv(indices, y_true, preds, latencies, "llm", args.out)
    print(json.dumps({k: results[k] for k in ("accuracy", "macro_f1", "invalid_outputs", "n_eval")}, indent=2))


if __name__ == "__main__":
    main()
