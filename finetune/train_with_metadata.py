"""
finetune/train_with_metadata.py  --  Issue #6 "[Model] BERT fine-tune - with metadata"
=======================================================================================

Same model, data split, hyperparameters and scoring as finetune/train_text_only.py,
but every statement is prefixed with speaker / party / subject / credit-history text
(built by the shared common.data_loader.build_metadata_text helper).

Because the ONLY difference from the text-only run is the input text, comparing
results/bert_meta.json with results/bert_text.json is a clean ablation of how much
the metadata helps.

LEAKAGE WARNING (credit history)
    LIAR's per-speaker credit-history counts (barely-true, false, half-true,
    mostly-true, pants-on-fire) are tallies of the speaker's past rulings, and the
    dataset's counts can include the very statement being classified. The model may
    therefore partly "read the answer" from the history. Treat the gain over
    text-only as an upper bound on what metadata really adds. Use --no-history to
    drop the history field and measure speaker/party/subject alone.

HOW TO RUN (GPU needed):
    python -m finetune.train_with_metadata
    python -m finetune.train_with_metadata --no-history            # leakage check
    python -m finetune.train_with_metadata --epochs 1 --limit 200  # smoke test

OUTPUT
    results/bert_meta.json (or bert_meta_nohistory.json), plus matching *_preds.csv
    checkpoints/bert_meta_checkpoint.pt   (git-ignored, never commit weights)
"""

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.config import LABEL_TO_IDX, RANDOM_SEED, RESULTS_DIR, CHECKPOINTS_DIR
from common.data_loader import load_liar, build_metadata_text
from common.evaluate import evaluate, build_results_dict, save_results_json, save_predictions_csv
from finetune.train_text_only import (
    set_seed, to_project_labels, shared_test_indices, predict, predict_with_latency,
    tokenize_batch, T4_PRICE_PER_HOUR, OVERHEAD,
)

def build_texts(split, include_history: bool = True) -> List[str]:
    """Metadata-prefixed text for every row of a split (optionally without credit history)."""
    texts = []
    for row in split:
        text = build_metadata_text(row)
        if not include_history:
            # build_metadata_text always prints a history field; strip it when asked not to.
            head, _, statement = text.partition(" [SEP] ")
            head = " | ".join(p for p in head.split(" | ") if not p.startswith("history:"))
            text = f"{head} [SEP] {statement}"
        texts.append(text)
    return texts


def text_only_baseline(results_dir: str) -> Dict:
    """Load the text-only result (if present) to record the ablation delta."""
    path = os.path.join(results_dir, "bert_text.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        r = json.load(f)
    return {"text_only_accuracy": r.get("accuracy"), "text_only_macro_f1": r.get("macro_f1")}


def main():
    ap = argparse.ArgumentParser(description="Fine-tune BERT with metadata prefix on LIAR")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-history", action="store_true", help="drop credit-history (leakage check)")
    ap.add_argument("--limit", type=int, default=0, help="use only N training rows (smoke test)")
    args = ap.parse_args()

    name = "bert_meta_nohistory" if args.no_history else "bert_meta"
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    use_fp16 = device.type == "cuda" and not args.no_fp16
    print(f"Device: {gpu_name} | fp16: {use_fp16} | model: {args.model} | run: {name}")

    # ---------- data ----------
    train, val, test = load_liar()
    y_train, y_val, y_test = to_project_labels(train), to_project_labels(val), to_project_labels(test)
    hist = not args.no_history
    x_train, x_val, x_test = (build_texts(s, hist) for s in (train, val, test))
    print("Example input:", x_train[0][:200])
    if args.limit:
        x_train, y_train = x_train[: args.limit], y_train[: args.limit]

    # ---------- model ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(LABEL_TO_IDX))
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    counts = np.bincount(y_train, minlength=len(LABEL_TO_IDX)).clip(min=1)
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # ---------- train (same loop as the text-only run) ----------
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{name}_checkpoint.pt")
    best_f1, best_epoch, bad = -1.0, 0, 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(x_train))
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i : i + args.batch_size]
            enc = tokenize_batch(tokenizer, [x_train[j] for j in idx], args.max_len, device)
            yb = torch.tensor([y_train[j] for j in idx], dtype=torch.long, device=device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                loss = loss_fn(model(**enc).logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(idx)
        val_m = evaluate(y_val, predict(model, tokenizer, x_val, args.max_len, device))
        print(f"epoch {epoch} | loss {total / len(x_train):.4f} | "
              f"val acc {val_m['accuracy']:.4f} macro-F1 {val_m['macro_f1']:.4f}")
        if val_m["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = val_m["macro_f1"], epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping (best epoch {best_epoch})")
                break
    train_minutes = (time.time() - t_start) / 60
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ---------- evaluate on the full test set and the shared subset ----------
    full_m = evaluate(y_test, predict(model, tokenizer, x_test, args.max_len, device))
    idx = shared_test_indices(test)
    sub_x, sub_y = [x_test[i] for i in idx], [y_test[i] for i in idx]
    sub_pred, lat = predict_with_latency(model, tokenizer, sub_x, args.max_len, device)
    sub_m = evaluate(sub_y, sub_pred)

    avg_lat = float(np.mean(lat))
    cost_per_1k = avg_lat / 1000 / 3600 * 1000 * T4_PRICE_PER_HOUR * OVERHEAD

    leak_note = "" if args.no_history else " Leakage risk: credit-history counts may include the labelled statement."
    results = build_results_dict(
        model_name=name,
        metrics=sub_m,
        params=n_params,
        train_minutes=round(train_minutes, 2),
        gpu=gpu_name,
        avg_latency_ms=round(avg_lat, 3),
        est_cost_per_1k=round(cost_per_1k, 6),
        cost_assumption="T4: $0.35/hour incl. 10% overhead; batch size 1",
        notes=(f"{args.model}, metadata prefix (speaker/party/subject"
               f"{'' if args.no_history else '/credit-history'}), max_len {args.max_len}, "
               f"best epoch {best_epoch}, fp16={use_fp16}.{leak_note}"),
    )
    results["full_test"] = {"accuracy": full_m["accuracy"], "macro_f1": full_m["macro_f1"], "n": len(y_test)}
    results["val_best_macro_f1"] = best_f1
    results["hyperparams"] = vars(args)
    base = text_only_baseline(RESULTS_DIR)
    if base:
        results["ablation_vs_text_only"] = {
            **base,
            "delta_accuracy": sub_m["accuracy"] - base["text_only_accuracy"],
            "delta_macro_f1": sub_m["macro_f1"] - base["text_only_macro_f1"],
        }

    save_results_json(results, name, RESULTS_DIR)
    save_predictions_csv(idx, sub_y, sub_pred, lat, name, RESULTS_DIR)
    print(f"Subset ({len(idx)}): acc {sub_m['accuracy']:.4f} macro-F1 {sub_m['macro_f1']:.4f} | "
          f"full test: acc {full_m['accuracy']:.4f} macro-F1 {full_m['macro_f1']:.4f} | "
          f"latency {avg_lat:.1f} ms | train {train_minutes:.2f} min")
    if base:
        print("Ablation vs text-only:", results["ablation_vs_text_only"])


if __name__ == "__main__":
    main()
