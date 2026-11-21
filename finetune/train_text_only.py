"""
finetune/train_text_only.py  --  Issue #5 "[Model] BERT fine-tune - text only"
==============================================================================

PURPOSE OF THIS FILE
    Fine-tune a pre-trained BERT on the LIAR statements (TEXT ONLY, no speaker /
    party / history) and score it the same way as every other model.
    "Text only" is the control experiment: issue #6 later adds metadata, and the
    difference between the two tells us how much the metadata really helps.

WHY BERT (and why it is different from the LSTM baseline in issue #4)
    BERT was already trained on huge amounts of text, so it "knows" English
    before it ever sees LIAR. We only adjust it ("fine-tune") with a new
    6-class output layer. The LSTM starts from zero knowledge. The project's
    central question is whether this 110M-parameter model beats the cheap
    LSTM and the 7B zero-shot LLM, considering both accuracy and cost.

ISSUE #5 CHECKLIST -> WHERE IT IS DONE
    [x] bert-base-uncased, max_len 128, 3 epochs, fp16, 6-class head -> CLI defaults + build_model()
    [x] Fallback DistilBERT / max_len 64 if the GPU is slow          -> --model and --max-len flags
    [x] Evaluate with shared evaluate(); checkpoint outside git       -> main(): evaluate(), checkpoints/
    [x] Record params, train minutes, GPU, latency                    -> build_results_dict(...) in main()

HOW TO RUN (needs a GPU; use Google Colab if your laptop has none):
    python -m finetune.train_text_only
    python -m finetune.train_text_only --model distilbert-base-uncased --max-len 64   # fallback
    python -m finetune.train_text_only --epochs 1 --limit 200                         # smoke test

OUTPUT
    results/bert_text.json, results/bert_text_preds.csv   (shared 300-row test subset)
    checkpoints/bert_text_checkpoint.pt                   (git-ignored, never commit weights)
"""

import argparse   # command-line options
import os         # files and folders
import random     # Python RNG (seeded)
import time       # timing for training minutes and latency
from typing import List

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset   # tiny label-name dataset for the shared subset helper
# AutoTokenizer / AutoModel... pick the right classes from the model NAME, so the
# same script works for bert-base-uncased AND the distilbert fallback.
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Shared project pieces (single source of truth for labels, seed, folders, metrics).
from common.config import LABEL_TO_IDX, RANDOM_SEED, RESULTS_DIR, CHECKPOINTS_DIR
from common.data_loader import load_liar, get_test_indices, load_test_indices, save_test_indices
from common.evaluate import evaluate, build_results_dict, save_results_json, save_predictions_csv

# Where the shared fixed test-subset indices live (same file every model uses).
TEST_INDICES_PATH = "common/test_indices.txt"
# Cost assumption from plan.md: T4 GPU at $0.35/hour + 10% overhead (same for all models).
T4_PRICE_PER_HOUR = 0.35
OVERHEAD = 1.10


def set_seed(seed: int) -> None:
    """
    Fix all random generators so the run is reproducible (same seed -> same
    shuffling, same dropout, same head initialisation). A result nobody can
    reproduce cannot be trusted in the final comparison.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)   # also seed every GPU


def to_project_labels(split) -> List[int]:
    """
    Convert Hugging Face LIAR label ids to the PROJECT's fixed label order.

    WHY: HF numbers the labels 0=false, 1=half-true, 2=mostly-true, 3=true,
    4=barely-true, 5=pants-fire, but plan.md fixes 0=pants-fire ... 5=true.
    Using the HF ids directly would scramble every metric and make BERT
    impossible to compare with the LSTM and the LLM. We map via the label NAME.
    """
    feature = split.features["label"]
    if hasattr(feature, "int2str"):
        return [LABEL_TO_IDX[feature.int2str(int(l))] for l in split["label"]]
    return [LABEL_TO_IDX[str(l).lower()] for l in split["label"]]


def shared_test_indices(test_split) -> List[int]:
    """
    Return the fixed ~300 test rows that every model is scored on (fair comparison).
    Load the file if it exists; otherwise generate it with the shared function.
    """
    if os.path.exists(TEST_INDICES_PATH):
        return load_test_indices(TEST_INDICES_PATH)
    names = test_split.features["label"].names
    # Build a minimal dataset whose "label" column holds the label NAMES (strings).
    # (Using .map() would NOT work: the column keeps its integer ClassLabel type.)
    named = Dataset.from_dict({"label": [names[i] for i in test_split["label"]]})
    indices = get_test_indices(named)
    save_test_indices(indices, TEST_INDICES_PATH)
    return indices


def tokenize_batch(tokenizer, texts: List[str], max_len: int, device):
    """
    Turn raw sentences into the tensors BERT expects and move them to the GPU.

    - truncation=True : cut statements longer than max_len tokens (BERT's cost grows with length)
    - padding=True    : pad the batch to its longest sentence so it is rectangular
    - the result has input_ids (word-piece ids) and attention_mask (1 = real token, 0 = padding)
    """
    enc = tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()   # no gradients needed when only predicting: faster, less memory
def predict(model, tokenizer, texts, max_len, device, batch_size=64) -> List[int]:
    """Predict the class of many statements in batches (validation / full test)."""
    model.eval()   # dropout OFF so predictions are deterministic
    preds = []
    for i in range(0, len(texts), batch_size):
        enc = tokenize_batch(tokenizer, texts[i : i + batch_size], max_len, device)
        preds.extend(model(**enc).logits.argmax(dim=-1).tolist())   # highest score = predicted class
    return preds


@torch.no_grad()
def predict_with_latency(model, tokenizer, texts, max_len, device):
    """
    Predict ONE statement at a time and time each call.

    WHY batch size 1: the demo app (issues #9/#10) answers one statement per
    request, so this is the realistic latency for the cost-vs-accuracy table (#11).
    We synchronize the GPU before reading the clock, otherwise we would measure
    only the launch of the work, not its completion.
    """
    model.eval()
    preds, lat = [], []
    for t in texts:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        enc = tokenize_batch(tokenizer, [t], max_len, device)
        preds.append(int(model(**enc).logits.argmax(dim=-1)))
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)   # seconds -> milliseconds
    return preds, lat


def main():
    """Full pipeline: load data -> fine-tune -> keep best epoch -> evaluate -> save."""

    # ---------- 1. Options (nothing important is hard-coded) ----------
    ap = argparse.ArgumentParser(description="Fine-tune BERT (text only) on LIAR")
    ap.add_argument("--model", default="bert-base-uncased")   # use distilbert-base-uncased as the fallback
    ap.add_argument("--max-len", type=int, default=128)       # issue #5: 128 (64 for the fallback)
    ap.add_argument("--epochs", type=int, default=3)          # issue #5: 3 epochs
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)         # standard BERT fine-tuning learning rate
    ap.add_argument("--patience", type=int, default=2, help="early stopping patience (val macro-F1)")
    ap.add_argument("--no-fp16", action="store_true", help="disable mixed precision")
    ap.add_argument("--limit", type=int, default=0, help="use only N training rows (smoke test)")
    args = ap.parse_args()

    set_seed(RANDOM_SEED)
    # Use the GPU if there is one; a CPU run works but is far too slow for full BERT.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    # fp16 (half precision) = about 2x faster and half the memory on a GPU, with almost no
    # accuracy loss. It only works on CUDA, so it is switched off automatically on CPU.
    use_fp16 = device.type == "cuda" and not args.no_fp16
    print(f"Device: {gpu_name} | fp16: {use_fp16} | model: {args.model}")

    # ---------- 2. Data (TEXT ONLY: only the 'statement' column is used) ----------
    train, val, test = load_liar()
    y_train, y_val, y_test = to_project_labels(train), to_project_labels(val), to_project_labels(test)
    x_train, x_val, x_test = list(train["statement"]), list(val["statement"]), list(test["statement"])
    if args.limit:   # tiny run to check that everything works before a long GPU run
        x_train, y_train = x_train[: args.limit], y_train[: args.limit]

    # ---------- 3. Model: pre-trained BERT + a fresh 6-class output layer ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # num_labels=6 adds a new classification head on top of BERT (randomly initialised, then trained).
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(LABEL_TO_IDX))
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)   # required in results JSON
    print(f"Trainable params: {n_params:,}")

    # AdamW = Adam with weight decay, the standard optimiser for BERT.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # GradScaler prevents tiny fp16 gradients from rounding to zero; disabled when fp16 is off.
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    # Class-weighted loss because LIAR classes are unbalanced (helps macro-F1, the headline metric).
    counts = np.bincount(y_train, minlength=len(LABEL_TO_IDX)).clip(min=1)
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # ---------- 4. Training loop with early stopping ----------
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "bert_text_checkpoint.pt")   # git-ignored by *.pt
    best_f1, best_epoch, bad = -1.0, 0, 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()                                   # dropout ON while learning
        order = np.random.permutation(len(x_train))     # new shuffle every epoch
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i : i + args.batch_size]
            enc = tokenize_batch(tokenizer, [x_train[j] for j in idx], args.max_len, device)
            yb = torch.tensor([y_train[j] for j in idx], dtype=torch.long, device=device)
            opt.zero_grad()
            # autocast runs the forward pass in fp16 where it is safe (no-op when disabled).
            with torch.cuda.amp.autocast(enabled=use_fp16):
                loss = loss_fn(model(**enc).logits, yb)
            scaler.scale(loss).backward()               # backprop on the scaled loss
            scaler.unscale_(opt)                        # restore real gradient size before clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # avoid exploding gradients
            scaler.step(opt)                            # update weights
            scaler.update()                             # adapt the fp16 scale factor
            total += loss.item() * len(idx)
        # Score the validation set (never trained on) to choose the best epoch.
        val_m = evaluate(y_val, predict(model, tokenizer, x_val, args.max_len, device))
        print(f"epoch {epoch} | loss {total / len(x_train):.4f} | "
              f"val acc {val_m['accuracy']:.4f} macro-F1 {val_m['macro_f1']:.4f}")
        if val_m["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = val_m["macro_f1"], epoch, 0
            torch.save(model.state_dict(), ckpt_path)   # keep only the best weights
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping (best epoch {best_epoch})")
                break
    train_minutes = (time.time() - t_start) / 60

    # Reload the best epoch (not necessarily the last).
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ---------- 5. Final evaluation ----------
    # Full test set: more rows, more reliable. Shared 300-row subset: apples-to-apples with LSTM/LLM.
    full_m = evaluate(y_test, predict(model, tokenizer, x_test, args.max_len, device))
    idx = shared_test_indices(test)
    sub_x, sub_y = [x_test[i] for i in idx], [y_test[i] for i in idx]
    sub_pred, lat = predict_with_latency(model, tokenizer, sub_x, args.max_len, device)
    sub_m = evaluate(sub_y, sub_pred)

    # ---------- 6. Cost estimate (same formula as the other models) ----------
    avg_lat = float(np.mean(lat))
    cost_per_1k = avg_lat / 1000 / 3600 * 1000 * T4_PRICE_PER_HOUR * OVERHEAD

    # ---------- 7. Save results in the shared schema ----------
    results = build_results_dict(
        model_name="bert_text",
        metrics=sub_m,                          # headline = shared subset
        params=n_params,
        train_minutes=round(train_minutes, 2),
        gpu=gpu_name,
        avg_latency_ms=round(avg_lat, 3),
        est_cost_per_1k=round(cost_per_1k, 6),
        cost_assumption="T4: $0.35/hour incl. 10% overhead; batch size 1",
        notes=f"{args.model}, text only, max_len {args.max_len}, best epoch {best_epoch}, fp16={use_fp16}",
    )
    results["full_test"] = {"accuracy": full_m["accuracy"], "macro_f1": full_m["macro_f1"], "n": len(y_test)}
    results["val_best_macro_f1"] = best_f1
    results["hyperparams"] = vars(args)

    save_results_json(results, "bert_text", RESULTS_DIR)
    save_predictions_csv(idx, sub_y, sub_pred, lat, "bert_text", RESULTS_DIR)
    print(f"Subset ({len(idx)}): acc {sub_m['accuracy']:.4f} macro-F1 {sub_m['macro_f1']:.4f} | "
          f"full test: acc {full_m['accuracy']:.4f} macro-F1 {full_m['macro_f1']:.4f} | "
          f"latency {avg_lat:.1f} ms | train {train_minutes:.2f} min")


# Run only when executed directly (python -m finetune.train_text_only), not when imported by tests.
if __name__ == "__main__":
    main()
