"""
baselines/train.py  --  Issue #4 "[Model] LSTM baseline"
=========================================================

PURPOSE OF THIS FILE
    Train the BiLSTM from baselines/lstm_model.py on LIAR, evaluate it, and save
    the results in the formats every teammate agreed on in plan.md, so that
    the analysis (issue #11) can compare all models fairly.

HOW THIS MAPS TO THE ISSUE #4 CHECKLIST
    [x] BiLSTM + trainable embeddings, text only  -> lstm_model.py + main()
    [x] Train/val, early stopping, fixed seed     -> set_seed(), the epoch loop in main()
    [x] Evaluate on full test + the shared subset -> full_m / sub_m in main()
    [x] Record params, train time, latency        -> build_results_dict(...) in main()

HOW TO RUN (from the repository root):
    python -m baselines.train             # full run
    python -m baselines.train --epochs 1  # quick smoke test

OUTPUT FILES
    results/lstm.json          metrics + params + time + latency + cost (shared schema)
    results/lstm_preds.csv     id,y_true,y_pred,latency_ms for the shared 300 rows
    checkpoints/lstm_checkpoint.pt   best weights (git-ignored: never commit weights)
"""

import argparse   # reads command-line options such as --epochs
import os         # file/folder handling
import random     # Python's random generator (seeded for reproducibility)
import time       # measures training time and per-prediction latency
from typing import List

import numpy as np   # numeric helpers (shuffling, averaging, class counts)
import torch
import torch.nn as nn

# Shared constants written by teammate Person A in common/config.py.
# We IMPORT them instead of copying so everyone always uses the same label order,
# seed and folders (a single source of truth).
from common.config import LABEL_TO_IDX, RANDOM_SEED, RESULTS_DIR, CHECKPOINTS_DIR
# Shared data helpers: load LIAR + the fixed stratified test subset.
from common.data_loader import load_liar, get_test_indices, load_test_indices, save_test_indices
# Shared metrics + file writers, so our output has the SAME schema as BERT's and the LLM's.
from common.evaluate import evaluate, build_results_dict, save_results_json, save_predictions_csv
# Our own model pieces.
from baselines.lstm_model import BiLSTMClassifier, Vocab, pad_batch, count_params

# Where the shared test-subset indices are stored (generated once, reused by all models).
TEST_INDICES_PATH = "common/test_indices.txt"
# Cost assumptions from plan.md: a T4 GPU costs $0.35/hour, plus 10% overhead.
# We use the same price for every model so the cost column in issue #11 is comparable.
T4_PRICE_PER_HOUR = 0.35
OVERHEAD = 1.10


def set_seed(seed: int) -> None:
    """
    Fix every random number generator.

    WHY: issue #4 asks for a "fixed seed". Same seed => same shuffling and same
    initial weights => anyone re-running the script gets (nearly) the same
    numbers. Without it our reported accuracy can't be reproduced or trusted.
    """
    random.seed(seed)          # Python's own random module
    np.random.seed(seed)       # NumPy (used for batch shuffling)
    torch.manual_seed(seed)    # PyTorch (weight initialisation, dropout)


def to_project_labels(split) -> List[int]:
    """
    Convert the Hugging Face label ids into OUR fixed label ids.

    WHY THIS EXISTS (important bug-avoidance): the HF copy of LIAR numbers its
    labels as 0=false, 1=half-true, 2=mostly-true, 3=true, 4=barely-true,
    5=pants-fire. The team agreed on a DIFFERENT order in plan.md:
    0=pants-fire, 1=false, 2=barely-true, 3=half-true, 4=mostly-true, 5=true.
    If we used the HF numbers directly, our confusion matrix would be
    scrambled and could not be compared with the other models. So we go
    through the label NAME and look it up in LABEL_TO_IDX.
    """
    # The "feature" object describes the label column (including the id->name list).
    feature = split.features["label"]
    if hasattr(feature, "int2str"):
        # Normal case: id -> name (e.g. 3 -> "true") -> project id (5).
        return [LABEL_TO_IDX[feature.int2str(int(l))] for l in split["label"]]
    # Fallback if the labels are already stored as text.
    return [LABEL_TO_IDX[str(l).lower()] for l in split["label"]]


def shared_test_indices(test_split) -> List[int]:
    """
    Return the ~300 fixed test row numbers that EVERY model must be scored on.

    WHY: plan.md requires the same stratified subset for all models, otherwise
    one model might get an easier sample and the comparison would be unfair.
    If the file already exists (created by Person A / another teammate) we just
    load it; only if it is missing do we generate and save it.
    """
    if os.path.exists(TEST_INDICES_PATH):
        return load_test_indices(TEST_INDICES_PATH)
    # The shared get_test_indices() expects label NAMES (strings), but the HF
    # dataset stores ids, so we convert them to names first.
    names = test_split.features["label"].names
    named = test_split.map(lambda r: {"label": names[r["label"]]}, features=None)
    indices = get_test_indices(named)           # stratified, seed 42
    save_test_indices(indices, TEST_INDICES_PATH)  # save so everyone reuses it
    return indices


def batches(xs, ys, batch_size, shuffle):
    """
    Yield the data in small chunks ("mini-batches").

    WHY: training on all 10k sentences at once would use too much memory and
    learn slowly; one-at-a-time is noisy and slow. Mini-batches are the standard
    compromise. We shuffle every epoch so the model doesn't memorise the order.
    """
    # Random order for training, natural order for evaluation.
    order = np.random.permutation(len(xs)) if shuffle else np.arange(len(xs))
    for i in range(0, len(xs), batch_size):
        idx = order[i : i + batch_size]                       # row numbers of this batch
        # Return the sentences and their labels as (list, tensor).
        yield [xs[j] for j in idx], torch.tensor([ys[j] for j in idx], dtype=torch.long)


@torch.no_grad()   # switches off gradient tracking: faster and uses less memory when only predicting
def predict(model, xs, pad_idx, batch_size=256) -> List[int]:
    """
    Predict the class of many sentences quickly (used for validation / full test).
    """
    model.eval()   # turns dropout OFF so predictions are deterministic
    preds = []
    for i in range(0, len(xs), batch_size):
        ids, lengths = pad_batch(xs[i : i + batch_size], pad_idx)
        # argmax over the 6 scores = the class the model believes in most.
        preds.extend(model(ids, lengths).argmax(dim=1).tolist())
    return preds


@torch.no_grad()
def predict_with_latency(model, xs, pad_idx):
    """
    Predict ONE statement at a time and time each prediction.

    WHY one at a time: the demo app (issue #9/#10) answers one user statement
    per request, so the realistic latency is for batch size 1. This number goes
    into results JSON ("avg_latency_ms") and the cost-vs-accuracy table (#11).
    Returns (predictions, latencies in milliseconds).
    """
    model.eval()
    preds, lat = [], []
    for x in xs:
        t0 = time.perf_counter()                  # start the stopwatch (high resolution)
        ids, lengths = pad_batch([x], pad_idx)    # a "batch" of one sentence
        preds.append(int(model(ids, lengths).argmax(dim=1)))
        lat.append((time.perf_counter() - t0) * 1000)   # seconds -> milliseconds
    return preds, lat


def main():
    """The full pipeline: load data -> train -> pick best epoch -> evaluate -> save."""

    # ---------- 1. Command-line options (so hyper-parameters aren't hard-coded) ----------
    ap = argparse.ArgumentParser(description="Train the BiLSTM baseline on LIAR")
    ap.add_argument("--epochs", type=int, default=15)         # max passes over the training data
    ap.add_argument("--patience", type=int, default=3,
                    help="early-stopping patience on val macro-F1")  # stop if no improvement for 3 epochs
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)         # learning rate of the Adam optimiser
    ap.add_argument("--max-len", type=int, default=64)        # truncate statements to 64 words
    ap.add_argument("--embed-dim", type=int, default=100)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--no-class-weights", action="store_true",
                    help="disable inverse-frequency loss weights")
    args = ap.parse_args()

    # ---------- 2. Reproducibility ----------
    set_seed(RANDOM_SEED)                                     # seed 42 from the shared config
    torch.set_num_threads(max(1, os.cpu_count() or 1))        # use all CPU cores for speed

    # ---------- 3. Data ----------
    train, val, test = load_liar()                            # shared loader (Hugging Face LIAR)
    # Convert the HF label ids to our fixed label order (see to_project_labels).
    y_train, y_val, y_test = to_project_labels(train), to_project_labels(val), to_project_labels(test)

    # Vocabulary from TRAIN ONLY (no test leakage, see Vocab docstring).
    vocab = Vocab.build(train["statement"])
    # Helper that turns every statement of a split into a list of word ids.
    enc = lambda split: [vocab.encode(t, args.max_len) for t in split["statement"]]
    x_train, x_val, x_test = enc(train), enc(val), enc(test)
    print(f"Vocab size: {len(vocab)} | train {len(x_train)} val {len(x_val)} test {len(x_test)}")

    # ---------- 4. Model ----------
    model = BiLSTMClassifier(
        len(vocab), len(LABEL_TO_IDX), args.embed_dim, args.hidden_dim,
        dropout=args.dropout, pad_idx=vocab.pad_idx,
    )
    n_params = count_params(model)                            # needed for results JSON
    print(f"Trainable params: {n_params:,}")

    # ---------- 5. Loss function and optimiser ----------
    weights = None
    if not args.no_class_weights:
        # LIAR classes are unbalanced (e.g. only ~800 "pants-fire" vs ~2000 "false").
        # Without weights the model would just guess the big classes and macro-F1
        # (which treats all 6 classes equally) would suffer. Weight = inverse frequency,
        # so rare classes count more in the loss.
        counts = np.bincount(y_train, minlength=len(LABEL_TO_IDX))
        weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float)
    loss_fn = nn.CrossEntropyLoss(weight=weights)             # standard loss for multi-class classification
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)    # Adam: robust default optimiser

    # ---------- 6. Training loop with early stopping ----------
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)               # folder for weights (git-ignored)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "lstm_checkpoint.pt")
    best_f1, best_epoch, bad = -1.0, 0, 0                     # best validation score so far
    t_start = time.time()                                     # for the "train_minutes" result
    for epoch in range(1, args.epochs + 1):
        model.train()                                         # dropout ON while learning
        total = 0.0                                           # running sum of the loss
        for xb, yb in batches(x_train, y_train, args.batch_size, shuffle=True):
            ids, lengths = pad_batch(xb, vocab.pad_idx)
            opt.zero_grad()                                   # clear the previous step's gradients
            loss = loss_fn(model(ids, lengths), yb)           # how wrong are we on this batch?
            loss.backward()                                   # compute gradients (backpropagation)
            # Cap gradient size: LSTMs can have "exploding gradients" that wreck training.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()                                        # update the weights
            total += loss.item() * len(yb)
        # After each epoch, score the VALIDATION set (data the model never trains on).
        val_m = evaluate(y_val, predict(model, x_val, vocab.pad_idx))
        print(f"epoch {epoch:2d} | loss {total / len(x_train):.4f} | "
              f"val acc {val_m['accuracy']:.4f} macro-F1 {val_m['macro_f1']:.4f}")
        # We pick the best epoch by macro-F1 (the project's headline metric).
        if val_m["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = val_m["macro_f1"], epoch, 0
            # Save the best weights + vocab, so we can reload exactly this model later.
            torch.save({"state_dict": model.state_dict(), "stoi": vocab.stoi, "args": vars(args)}, ckpt_path)
        else:
            bad += 1
            if bad >= args.patience:
                # EARLY STOPPING: validation hasn't improved for `patience` epochs;
                # further training would only overfit, so stop and save time.
                print(f"Early stopping (best epoch {best_epoch})")
                break
    train_minutes = (time.time() - t_start) / 60

    # Reload the BEST epoch (not the last one, which may be slightly overfit).
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])

    # ---------- 7. Final evaluation ----------
    # (a) The FULL test set: more rows => more reliable numbers.
    full_m = evaluate(y_test, predict(model, x_test, vocab.pad_idx))
    # (b) The SHARED 300-row subset: the apples-to-apples number used to compare
    #     against BERT and the LLM (the LLM is too slow to run on all 1,283 rows).
    idx = shared_test_indices(test)
    sub_x, sub_y = [x_test[i] for i in idx], [y_test[i] for i in idx]
    sub_pred, lat = predict_with_latency(model, sub_x, vocab.pad_idx)
    sub_m = evaluate(sub_y, sub_pred)

    # ---------- 8. Cost estimate ----------
    avg_lat = float(np.mean(lat))   # average milliseconds per prediction
    # ms -> hours, times 1000 predictions, times $/hour, plus overhead.
    cost_per_1k = avg_lat / 1000 / 3600 * 1000 * T4_PRICE_PER_HOUR * OVERHEAD

    # ---------- 9. Save results in the SHARED schema ----------
    results = build_results_dict(
        model_name="lstm",
        metrics=sub_m,                       # headline numbers = shared subset
        params=n_params,
        train_minutes=round(train_minutes, 2),
        gpu="CPU",                           # honest: trained/measured on CPU, not a T4
        avg_latency_ms=round(avg_lat, 3),
        est_cost_per_1k=round(cost_per_1k, 6),
        cost_assumption="Priced at T4 rate $0.35/hour incl. 10% overhead; batch size 1, CPU inference",
        notes=f"Text-only BiLSTM, trainable embeddings, best epoch {best_epoch}, "
              f"class-weighted loss={not args.no_class_weights}",
    )
    # Extra fields (additive, they don't break the shared schema):
    results["full_test"] = {"accuracy": full_m["accuracy"], "macro_f1": full_m["macro_f1"], "n": len(y_test)}
    results["val_best_macro_f1"] = best_f1
    results["hyperparams"] = vars(args)      # lets anyone reproduce this exact run

    save_results_json(results, "lstm", RESULTS_DIR)                       # results/lstm.json
    save_predictions_csv(idx, sub_y, sub_pred, lat, "lstm", RESULTS_DIR)  # results/lstm_preds.csv
    print(f"Subset ({len(idx)}): acc {sub_m['accuracy']:.4f} macro-F1 {sub_m['macro_f1']:.4f} | "
          f"full test: acc {full_m['accuracy']:.4f} macro-F1 {full_m['macro_f1']:.4f} | "
          f"latency {avg_lat:.2f} ms | train {train_minutes:.2f} min")


# Run main() only when this file is executed directly (python -m baselines.train),
# not when another file merely imports from it (e.g. the unit tests).
if __name__ == "__main__":
    main()
