"""
finetune/train_bert.py  --  BERT fine-tuning on LIAR (Run 1: text, Run 2: meta)
================================================================================

HOW TO RUN (from the repository root, ideally on a Colab/Kaggle T4):
    python -m finetune.train_bert --mode text     # Run 1: statement only
    python -m finetune.train_bert --mode meta     # Run 2: metadata + statement
    python -m finetune.train_bert --mode text --epochs 1 --max-train 200   # smoke test

    # RoBERTa (Sandya): same script, different model + result name
    python -m finetune.train_bert --mode text --model-name roberta-base --run-name roberta_text

OUTPUT FILES (same schema as the LSTM, plus GPU-memory / per-1k-time extras):
    results/<run-name>.json         default run names: bert_text, bert_meta
    results/<run-name>_preds.csv    id,y_true,y_pred,latency_ms on the shared 300-row subset
    checkpoints/<run-name>.pt       best weights (git-ignored)
"""

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Learning-rate schedule: warm up, then decay.
from transformers import get_linear_schedule_with_warmup

# Reuse the LSTM team's code: same seed function, same label mapping, same 300-row subset => fair comparison.
from baselines.train import set_seed, to_project_labels, shared_test_indices
from common.config import (
    BATCH_SIZE_EVAL, BATCH_SIZE_TRAIN, CHECKPOINTS_DIR, LABEL_TO_IDX,
    LEARNING_RATE_DEFAULT, MAX_LENGTH_TEXT, NUM_EPOCHS_DEFAULT, RANDOM_SEED, RESULTS_DIR,
)
from common.data_loader import load_liar
from common.evaluate import build_results_dict, evaluate, save_predictions_csv, save_results_json
from finetune.bert_model import (
    DEFAULT_MODEL_NAME, EncodedDataset, build_texts, count_params,
    load_model_and_tokenizer, make_collate,
)

T4_PRICE_PER_HOUR = 0.35   # same cost assumption as every other model
OVERHEAD = 1.10            # +10% on top of the hourly price (same as the other models)


def _sync(device: torch.device) -> None:
    """Wait for queued GPU work so timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def predict(model, loader, device, use_amp: bool) -> List[int]:
    """Batched argmax predictions (project label ids); used for validation and full test."""
    model.eval()
    # Collect predicted class ids.
    preds: List[int] = []
    # Go through the data one batch at a time.
    for batch in loader:
        # Send tensors to the GPU; leave the labels out because we only predict here.
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        # Run the forward pass in fp16 where safe (faster, less memory).
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            # Raw scores, one per class, for every example in the batch.
            logits = model(**batch).logits
        # Pick the class with the highest score for each example.
        preds.extend(logits.argmax(dim=-1).tolist())
    return preds


@torch.no_grad()
def predict_with_latency(model, texts, tokenizer, max_len, device, use_amp) -> Tuple[List[int], List[float]]:
    """One statement at a time (batch size 1), like the demo app, timed per statement."""
    model.eval()
    preds, lat = [], []
    # Warm-up so the first timed call doesn't include CUDA start-up cost (context creation,
    # kernel loading), which would inflate the average latency.
    warm = tokenizer(texts[0], truncation=True, max_length=max_len, return_tensors="pt").to(device)
    model(**warm)
    # Time each statement separately.
    for t in texts:
        # CUDA calls are asynchronous: sync before starting the clock so earlier queued work
        # isn't charged to this statement, and again before stopping it so the GPU work
        # is finished. The timed span includes tokenisation, as in real use.
        _sync(device)
        # Start the clock (after syncing the GPU).
        t0 = time.perf_counter()
        # Tokenisation is included in the timing, as in real use.
        enc = tokenizer(t, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(**enc).logits
        # Store the predicted class.
        preds.append(int(logits.argmax(dim=-1)))
        _sync(device)
        # Store this statement's latency in milliseconds.
        lat.append((time.perf_counter() - t0) * 1000)
    return preds, lat


def main():
    """Train with early stopping on val macro-F1, then evaluate and write results files."""
    ap = argparse.ArgumentParser(description="Fine-tune an encoder on LIAR")
    # REQUIRED: text = statement only, meta = metadata + statement.
    ap.add_argument("--mode", choices=["text", "meta"], required=True)
    # Any Hugging Face encoder, e.g. roberta-base.
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    # Name of the output files (default bert_<mode>).
    ap.add_argument("--run-name", default=None, help="results file stem (default bert_<mode>)")
    # Maximum passes over the training data.
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS_DEFAULT)
    # Stop if validation macro-F1 does not improve for this many epochs.
    ap.add_argument("--patience", type=int, default=2, help="early stopping on val macro-F1")
    # Examples per training step.
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_TRAIN)
    # Learning rate: step size of the optimiser.
    ap.add_argument("--lr", type=float, default=LEARNING_RATE_DEFAULT)
    # Maximum tokens per input; longer text is cut.
    ap.add_argument("--max-len", type=int, default=None,
                    help=f"default {MAX_LENGTH_TEXT} for text, 2x for meta")
    # Fraction of steps spent slowly raising the learning rate.
    ap.add_argument("--warmup", type=float, default=0.1, help="warm-up fraction of total steps")
    # Regularisation strength (shrinks weights slightly each step).
    ap.add_argument("--weight-decay", type=float, default=0.01)
    # Switch OFF the class-imbalance correction.
    ap.add_argument("--no-class-weights", action="store_true")
    # Switch OFF mixed precision.
    ap.add_argument("--no-fp16", action="store_true")
    # Use only N training rows: quick smoke test.
    ap.add_argument("--max-train", type=int, default=None, help="use only N train rows (smoke test)")
    # Read what the user typed.
    args = ap.parse_args()

    # Output file stem.
    run_name = args.run_name or f"bert_{args.mode}"
    # Meta mode gets twice the length because the metadata prefix is prepended to the statement.
    max_len = args.max_len or (MAX_LENGTH_TEXT if args.mode == "text" else MAX_LENGTH_TEXT * 2)

    set_seed(RANDOM_SEED)   # same seed as every other model for comparable runs
    # Use the GPU if there is one, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fp16 autocast speeds up training and cuts memory on a T4; CPU runs stay in fp32.
    use_amp = device.type == "cuda" and not args.no_fp16
    # For the results file.
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    # Show the setup so you can see what is running.
    print(f"Device: {gpu_name} | fp16: {use_amp} | mode: {args.mode} | model: {args.model_name}")

    # ---------- data (same split, same label order as every other model) ----------
    try:
        # Load LIAR: 10,269 train / 1,284 val / 1,283 test.
        train, val, test = load_liar()
    except Exception as e:   # script-based "liar" is unsupported on recent `datasets`
        print(f"load_liar() failed ({e}); falling back to the official LIAR zip")
        # Imported only when needed.
        from finetune.liar_fallback import load_liar_from_zip
        train, val, test = load_liar_from_zip()
    print(f"Train {len(train)} | val {len(val)} | test {len(test)}")
    # Go through the project label order, not the raw HF ids, so results are comparable.
    y_train, y_val, y_test = to_project_labels(train), to_project_labels(val), to_project_labels(test)
    # Smoke-test shortcut: keep only the first N training rows.
    if args.max_train:
        train = train.select(range(args.max_train))
        y_train = y_train[: args.max_train]

    # Build BERT (or RoBERTa) with a 6-class head.
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    # Use the model's own separator (BERT "[SEP]", RoBERTa "</s>") between metadata and statement.
    sep = tokenizer.sep_token or "[SEP]"
    # Build the input strings for train/val/test in the chosen mode.
    texts = {n: build_texts(s, args.mode, sep) for n, s in (("train", train), ("val", val), ("test", test))}
    # Eyeball one real input to confirm the format.
    print("Example input:", texts["train"][0][:200])

    # Batch padding function.
    collate = make_collate(tokenizer)
    # Small helper that builds a DataLoader for val/test.
    mk = lambda name, ys, bs, shuffle: DataLoader(
        EncodedDataset(texts[name], ys, tokenizer, max_len),
        batch_size=bs, shuffle=shuffle, collate_fn=collate)
    # Seeded generator makes the training shuffle order reproducible; eval loaders don't shuffle.
    g = torch.Generator().manual_seed(RANDOM_SEED)
    # Training batches, shuffled each epoch.
    train_loader = DataLoader(EncodedDataset(texts["train"], y_train, tokenizer, max_len),
                              batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=g)
    # Validation batches (no shuffle): used to choose the best epoch.
    val_loader = mk("val", y_val, BATCH_SIZE_EVAL, False)
    # Test batches (no shuffle): used once at the end.
    test_loader = mk("test", y_test, BATCH_SIZE_EVAL, False)

    # Move the model onto the GPU.
    model.to(device)
    # For the results table.
    n_params = count_params(model)
    print(f"Parameters: {n_params:,}")

    # ---------- loss / optimiser / schedule ----------
    # LIAR classes are imbalanced; inverse-frequency weights (n / (k * count)) stop the model
    # ignoring rare classes, which matters because macro-F1 weighs all classes equally.
    weights = None
    # LIAR classes are imbalanced; weight rare classes up.
    if not args.no_class_weights:
        # How many training examples per class.
        counts = np.bincount(y_train, minlength=len(LABEL_TO_IDX))
        # Weight = n / (k * count): rare class => big weight.
        weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float, device=device)
    # Standard multi-class loss, with optional class weights.
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    # Standard BERT recipe: no weight decay on biases and LayerNorm weights.
    no_decay = ("bias", "LayerNorm.weight")
    # Two parameter groups: with weight decay / without.
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    # AdamW = Adam with proper weight decay; the standard choice for BERT.
    opt = torch.optim.AdamW(groups, lr=args.lr)
    # Linear warm-up then linear decay; warm-up avoids large early updates to pretrained weights.
    total_steps = len(train_loader) * args.epochs
    # Warm up for 10% of steps, then decrease linearly to 0.
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    # Loss scaling keeps small fp16 gradients from underflowing; no-op when fp16 is off.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---------- training with early stopping on val macro-F1 ----------
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    # Where the best weights are saved.
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{run_name}.pt")
    # Reset so the reported training memory covers training only.
    if device.type == "cuda":
        # Start measuring GPU memory from zero.
        torch.cuda.reset_peak_memory_stats()
    # Track the best validation score, which epoch, and how many epochs without improvement.
    best_f1, best_epoch, bad = -1.0, 0, 0
    # Start the training clock.
    t_start = time.time()
    # One epoch = one pass over all training data.
    for epoch in range(1, args.epochs + 1):
        # Training mode: dropout ON.
        model.train()
        # Sum of the loss, for the epoch average.
        running = 0.0
        # One training step per batch.
        for batch in train_loader:
            # Move the batch to the GPU.
            batch = {k: v.to(device) for k, v in batch.items()}
            # Separate the answers from the inputs.
            labels = batch.pop("labels")
            # Erase gradients from the previous step.
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)   # loss in fp32 for numerical stability
            # Backpropagation: compute gradients (on the scaled loss).
            scaler.scale(loss).backward()
            scaler.unscale_(opt)   # unscale first so the clip threshold applies to true gradients
            # Cap the gradient size at 1.0 to prevent unstable jumps.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update the weights.
            scaler.step(opt)
            # Adjust the loss scale for the next step.
            scaler.update()
            # Move the learning-rate schedule one step.
            sched.step()
            # Accumulate loss weighted by batch size.
            running += loss.item() * len(labels)
        # Model selection uses the validation set only; the test set is never looked at
        # until training is over, so the final scores are not tuned to it.
        val_m = evaluate(y_val, predict(model, val_loader, device, use_amp))
        # Progress line: loss, val accuracy, val macro-F1.
        print(f"epoch {epoch} | loss {running / len(train_loader.dataset):.4f} | "
              f"val acc {val_m['accuracy']:.4f} macro-F1 {val_m['macro_f1']:.4f}")
        # Macro-F1 (not accuracy) is the selection metric since it counts every class equally.
        # Only the best epoch's weights are kept on disk; patience counts epochs without gain.
        if val_m["macro_f1"] > best_f1:
            # Remember the new best and reset the patience counter.
            best_f1, best_epoch, bad = val_m["macro_f1"], epoch, 0
            # Keep only the best weights on disk.
            torch.save({"state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
        else:
            # No improvement this epoch.
            bad += 1
            # Too many epochs without improvement => stop (early stopping).
            if bad >= args.patience:
                # Tell the user why training ended.
                print(f"Early stopping (best epoch {best_epoch})")
                break
    _sync(device)   # make sure all GPU work is done before stopping the training clock
    # Total training time in minutes.
    train_minutes = (time.time() - t_start) / 60
    # Peak GPU memory used while training.
    train_mem_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0

    # ---------- final evaluation with the best epoch ----------
    # Reload the best-epoch weights (the model in memory is from the last epoch run).
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    # Full test set (1,283 rows) is scored in batches for a more stable accuracy estimate.
    full_m = evaluate(y_test, predict(model, test_loader, device, use_amp))

    # The shared ~300-row subset is what every model (incl. the LLMs) is compared on, and the
    # only part timed at batch size 1.
    idx = shared_test_indices(test)
    # Their texts and true labels.
    sub_texts, sub_y = [texts["test"][i] for i in idx], [y_test[i] for i in idx]
    # Reset again so inference memory is measured separately from training memory.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Predict those 300 one by one and record latency.
    sub_pred, lat = predict_with_latency(model, sub_texts, tokenizer, max_len, device, use_amp)
    # Peak GPU memory during inference.
    infer_mem_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    # Headline metrics (accuracy, macro-F1, confusion matrix) on the 300 rows.
    sub_m = evaluate(sub_y, sub_pred)

    # Average milliseconds per statement.
    avg_lat = float(np.mean(lat))
    # ms -> hours per statement, x1000 statements, x hourly price, x overhead.
    cost_per_1k = avg_lat / 1000 / 3600 * 1000 * T4_PRICE_PER_HOUR * OVERHEAD

    # Standard results dictionary shared by all models.
    results = build_results_dict(
        model_name=run_name,
        metrics=sub_m,
        params=n_params,
        train_minutes=round(train_minutes, 2),
        gpu=gpu_name,
        avg_latency_ms=round(avg_lat, 3),
        est_cost_per_1k=round(cost_per_1k, 6),
        cost_assumption="Priced at T4 rate $0.35/hour incl. 10% overhead; batch size 1 inference",
        notes=f"{args.model_name}, mode={args.mode}, best epoch {best_epoch}, "
              f"class-weighted loss={not args.no_class_weights}, fp16={use_amp}, max_len={max_len}",
    )
    # Extra fields requested in the project brief (additive, schema stays compatible):
    results["inference_sec_per_1000"] = round(avg_lat, 3)   # ms/statement == seconds per 1,000
    results["infer_seconds_per_1000"] = round(avg_lat, 3)   # same key the LLM results / cost_analysis use
    # Peak memory is the headline GPU figure; llm/cost_analysis reads gpu_peak_mem_mb, and
    # falls back to gpu_mem_inference_mb if it is absent.
    results["gpu_peak_mem_mb"] = round(max(train_mem_mb, infer_mem_mb), 1)
    results["gpu_mem_train_mb"] = round(train_mem_mb, 1)
    results["gpu_mem_inference_mb"] = round(infer_mem_mb, 1)
    # Headline metrics above are on the 300-row subset; full_test gives the same scores on
    # all test rows (with the confusion matrix) as a lower-variance cross-check.
    results["full_test"] = {"accuracy": full_m["accuracy"], "macro_f1": full_m["macro_f1"],
                            "n": len(y_test), "confusion_matrix": full_m["confusion_matrix"]}
    results["val_best_macro_f1"] = best_f1       # validation score of the checkpoint that was kept
    results["hyperparams"] = vars(args)          # lets anyone reproduce this exact run

    # Write results/<run>.json.
    save_results_json(results, run_name, RESULTS_DIR)
    # Write results/<run>_preds.csv (one row per test statement).
    save_predictions_csv(idx, sub_y, sub_pred, lat, run_name, RESULTS_DIR)
    # Final summary line.
    print(f"Subset ({len(idx)}): acc {sub_m['accuracy']:.4f} macro-F1 {sub_m['macro_f1']:.4f} | "
          f"full test: acc {full_m['accuracy']:.4f} macro-F1 {full_m['macro_f1']:.4f} | "
          f"latency {avg_lat:.1f} ms | train {train_minutes:.1f} min | "
          f"GPU mem train {train_mem_mb:.0f} MB / infer {infer_mem_mb:.0f} MB")


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    main()
