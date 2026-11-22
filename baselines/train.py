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
from typing import List  # type hints (List[int] jaise); sirf padhne me madad, code ka behaviour nahi badalta

import numpy as np   # numeric helpers (shuffling, averaging, class counts)
import torch  # PyTorch: tensors aur neural network library
import torch.nn as nn  # nn = layers (Linear, LSTM...) aur loss functions
from datasets import Dataset   # used to build a tiny label-name dataset for the shared subset helper

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
# VIVA Q: Cost per 1,000 predictions kaise nikalte hain?
#     A: Latency (ms) ko ghante me badlo, 1,000 se guna karo, T4 ke rate $0.35/ghanta se guna karo, aur
#        10% overhead jodo: ms / 1000 / 3600 x 1000 x 0.35 x 1.10. LSTM ke 2.08 ms par ye $0.000222 per
#        1,000 aata hai.
# VIVA Q: LSTM CPU par chala, phir T4 ka rate kyun?
#     A: Taaki saare models ek hi formula se price hon. Par ye like-for-like nahi hai (CPU time, T4
#        price), isliye cost table me LSTM par dagger (footnote) lagta hai: Timed on CPU but priced at
#        the T4 rate.
T4_PRICE_PER_HOUR = 0.35
OVERHEAD = 1.10  # +10% overhead, har model ke liye same (fair cost)


# VIVA Q: Seed kya hai, aur 42 hi kyun?
#     A: Weights ki shuruaat, data shuffle aur dropout random hote hain; seed fix karne par wo har baar
#        ek hi order me aate hain, toh result dobara aata hai. 42 sirf convention hai; zaroori ye hai ki
#        SAARE models me same seed ho (fair comparison).
# VIVA Q: Sirf ek seed ka kya nuksan hai?
#     A: Run-to-run variation naapa nahi gaya. Kai seeds chalake mean +/- std dena behtar hota; yeh
#        report ki limitation me likha hai.
def set_seed(seed: int) -> None:  # saare random generators fix karta hai taaki dobara chalane par wahi result aaye
    """
    Fix every random number generator.

    WHY: issue #4 asks for a "fixed seed". Same seed => same shuffling and same
    initial weights => anyone re-running the script gets (nearly) the same
    numbers. Without it our reported accuracy can't be reproduced or trusted.
    """
    random.seed(seed)          # Python's own random module
    np.random.seed(seed)       # NumPy (used for batch shuffling)
    torch.manual_seed(seed)    # PyTorch (weight initialisation, dropout)


# VIVA Q: Label remap kyun karna pada?
#     A: Hugging Face ka LIAR ids ko 0=false, 1=half-true, 2=mostly-true, 3=true, 4=barely-true, 5=pants-
#        fire batata hai, jabki project ka order 0=pants-fire se 5=true tak hai. Seedha use karte toh
#        confusion matrix aur har metric galat nikalte. Isliye label ke NAAM se project id nikalte hain.
def to_project_labels(split) -> List[int]:  # HF ke label ids ko project ke fixed order (0=pants-fire ... 5=true) me badalta hai
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
    if hasattr(feature, "int2str"):  # label agar ClassLabel hai (id -> naam ka table), toh naam se project id nikalenge
        # Normal case: id -> name (e.g. 3 -> "true") -> project id (5).
        return [LABEL_TO_IDX[feature.int2str(int(l))] for l in split["label"]]
    # Fallback if the labels are already stored as text.
    return [LABEL_TO_IDX[str(l).lower()] for l in split["label"]]


# VIVA Q: 300-row shared subset kyun?
#     A: Qwen 7B LLM ko saari 1,283 test rows par chalana bahut slow hai, aur har model ko EK HI rows par
#        score karna zaroori hai warna comparison unfair. Subset stratified hai: class mix asli test
#        jaisa (22 / 59 / 50 / 62 / 58 / 49).
# VIVA Q: Is subset ki limit kya hai?
#     A: 300 rows par har accuracy lagbhag +/-5 points uncertain hoti hai; isliye full test (1,283 rows)
#        ke numbers bhi report karte hain.
def shared_test_indices(test_split) -> List[int]:  # wahi 300 test rows jo har model par score hoti hain
    """
    Return the ~300 fixed test row numbers that EVERY model must be scored on.

    WHY: plan.md requires the same stratified subset for all models, otherwise
    one model might get an easier sample and the comparison would be unfair.
    If the file already exists (created by Person A / another teammate) we just
    load it; only if it is missing do we generate and save it.
    """
    if os.path.exists(TEST_INDICES_PATH):  # file pehle se hai toh wahi use karo (sabko same rows milengi)
        return load_test_indices(TEST_INDICES_PATH)  # saved row numbers padh ke wapas
    # The shared get_test_indices() expects label NAMES (strings), but the HF
    # dataset stores ids, so we convert them to names first.
    names = test_split.features["label"].names
    # Build a minimal dataset whose "label" column holds the label NAMES (strings).
    # (Using .map() would NOT work: the column keeps its integer ClassLabel type.)
    named = Dataset.from_dict({"label": [names[i] for i in test_split["label"]]})
    indices = get_test_indices(named)           # stratified, seed 42
    save_test_indices(indices, TEST_INDICES_PATH)  # save so everyone reuses it
    return indices  # 300 row numbers wapas


# VIVA Q: Mini-batch aur shuffle kyun?
#     A: Poora data ek saath daalna memory me bhaari aur seekhne me dheema; ek-ek sample noisy aur slow.
#        Mini-batch (64) beech ka raasta hai. Har epoch shuffle karte hain taaki model data ke order ko
#        na ratte.
def batches(xs, ys, batch_size, shuffle):  # data ko chhote tukdon (mini-batches) me deta hai
    """
    Yield the data in small chunks ("mini-batches").

    WHY: training on all 10k sentences at once would use too much memory and
    learn slowly; one-at-a-time is noisy and slow. Mini-batches are the standard
    compromise. We shuffle every epoch so the model doesn't memorise the order.
    """
    # Random order for training, natural order for evaluation.
    order = np.random.permutation(len(xs)) if shuffle else np.arange(len(xs))
    for i in range(0, len(xs), batch_size):  # batch_size ke step me aage badho
        idx = order[i : i + batch_size]                       # row numbers of this batch
        # Return the sentences and their labels as (list, tensor).
        yield [xs[j] for j in idx], torch.tensor([ys[j] for j in idx], dtype=torch.long)


# VIVA Q: predict me model.eval() aur no_grad kyun?
#     A: eval() dropout band karta hai (predictions stable), aur no_grad gradient nahi banata (tez aur
#        kam memory). Training me ye nahi lagate.
@torch.no_grad()   # switches off gradient tracking: faster and uses less memory when only predicting
def predict(model, xs, pad_idx, batch_size=256) -> List[int]:  # bahut sentences ek saath predict (validation aur full test ke liye)
    """
    Predict the class of many sentences quickly (used for validation / full test).
    """
    model.eval()   # turns dropout OFF so predictions are deterministic
    preds = []  # yahan predicted class ids jama honge
    for i in range(0, len(xs), batch_size):  # 256 sentences ke batch me
        ids, lengths = pad_batch(xs[i : i + batch_size], pad_idx)  # batch ko rectangle (padded matrix) me badlo
        # argmax over the 6 scores = the class the model believes in most.
        preds.extend(model(ids, lengths).argmax(dim=1).tolist())
    return preds  # saari predictions ki list


# VIVA Q: Latency batch size 1 par kyun naapte hain?
#     A: Demo app ek user statement par ek jawab deta hai, toh asli latency ek-ek statement ki hai. Batch
#        me per-statement time kam dikhta, jo asli use jaisa nahi hota.
@torch.no_grad()  # gradient tracking band: predict ke waqt tez aur kam memory
def predict_with_latency(model, xs, pad_idx):  # ek-ek statement predict karke har ek ka time naapta hai (batch size 1)
    """
    Predict ONE statement at a time and time each prediction.

    WHY one at a time: the demo app (issue #9/#10) answers one user statement
    per request, so the realistic latency is for batch size 1. This number goes
    into results JSON ("avg_latency_ms") and the cost-vs-accuracy table (#11).
    Returns (predictions, latencies in milliseconds).
    """
    model.eval()  # dropout band, taaki predictions stable rahein
    preds, lat = [], []  # predictions aur har ek ka latency (ms)
    for x in xs:  # ek statement at a time, jaise demo app me hota hai
        t0 = time.perf_counter()                  # start the stopwatch (high resolution)
        ids, lengths = pad_batch([x], pad_idx)    # a "batch" of one sentence
        preds.append(int(model(ids, lengths).argmax(dim=1)))  # argmax = sabse zyada score wali class
        lat.append((time.perf_counter() - t0) * 1000)   # seconds -> milliseconds
    return preds, lat  # (predictions, latency ki list)


def main():  # poora pipeline: data -> train -> best epoch -> evaluate -> save
    """The full pipeline: load data -> train -> pick best epoch -> evaluate -> save."""

    # ---------- 1. Command-line options (so hyper-parameters aren't hard-coded) ----------
    ap = argparse.ArgumentParser(description="Train the BiLSTM baseline on LIAR")
    ap.add_argument("--epochs", type=int, default=15)         # max passes over the training data
    # VIVA Q: Early stopping kya hai?
    #     A: Har epoch ke baad validation macro-F1 dekhte hain; patience=3 epochs tak sudhar na ho toh
    #        training rok dete hain, kyunki aage model train data ratta (overfit). Sirf best epoch ke
    #        weights rakhe jaate hain (LSTM me best epoch 6).
    ap.add_argument("--patience", type=int, default=3,  # early stopping: itne epochs tak sudhar na ho toh ruk jao
                    help="early-stopping patience on val macro-F1")  # stop if no improvement for 3 epochs
    ap.add_argument("--batch-size", type=int, default=64)  # ek baar me 64 sentences
    ap.add_argument("--lr", type=float, default=1e-3)         # learning rate of the Adam optimiser
    ap.add_argument("--max-len", type=int, default=64)        # truncate statements to 64 words
    ap.add_argument("--embed-dim", type=int, default=100)  # word vector ka size (100)
    ap.add_argument("--hidden-dim", type=int, default=128)  # LSTM memory ka size (128)
    ap.add_argument("--dropout", type=float, default=0.5)  # dropout probability (0.5)
    ap.add_argument("--no-class-weights", action="store_true",  # ye flag dene par class weights band
                    help="disable inverse-frequency loss weights")  # (upar wale flag ka help text)
    args = ap.parse_args()  # command line ke options padh ke args me rakhta hai

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
    x_train, x_val, x_test = enc(train), enc(val), enc(test)  # teeno splits ke statements ko numbers ki lists me badla
    print(f"Vocab size: {len(vocab)} | train {len(x_train)} val {len(x_val)} test {len(x_test)}")  # vocabulary aur data ka size dikhao (sanity check)

    # ---------- 4. Model ----------
    model = BiLSTMClassifier(
        len(vocab), len(LABEL_TO_IDX), args.embed_dim, args.hidden_dim,  # model ke sizes: vocab, 6 classes, embedding dim, hidden dim
        dropout=args.dropout, pad_idx=vocab.pad_idx,  # dropout aur pad id
    )
    n_params = count_params(model)                            # needed for results JSON
    print(f"Trainable params: {n_params:,}")  # parameter count print (932,998)

    # ---------- 5. Loss function and optimiser ----------
    weights = None
    if not args.no_class_weights:  # default me class weights ON
        # LIAR classes are unbalanced (e.g. only ~800 "pants-fire" vs ~2000 "false").
        # Without weights the model would just guess the big classes and macro-F1
        # (which treats all 6 classes equally) would suffer. Weight = inverse frequency,
        # so rare classes count more in the loss.
        counts = np.bincount(y_train, minlength=len(LABEL_TO_IDX))
        # VIVA Q: Class weights kyun, aur formula kya hai?
        #     A: LIAR ke classes unbalanced hain (pants-fire kam, false zyada). Bina weight ke model bade
        #        classes ko hi guess karta aur macro-F1 girta. Weight = n / (k x count_c): n total rows,
        #        k = 6 classes; kam rows wali class ko zyada weight, toh uski galti loss me zyada
        #        mehengi.
        weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float)  # weight = n / (k x count): kam rows wali class ko zyada weight
    # VIVA Q: CrossEntropyLoss kyun?
    #     A: Multi-class classification ka standard loss hai: softmax + negative log-likelihood. Sahi
    #        class ki probability badhne par loss ghatta hai.
    loss_fn = nn.CrossEntropyLoss(weight=weights)             # standard loss for multi-class classification
    # VIVA Q: Adam optimiser aur learning rate 0.001 kyun?
    #     A: Adam har weight ki learning rate khud adjust karta hai, isliye default setting achha chalti
    #        hai. 1e-3 chhote networks ke liye standard shuruaati value hai.
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)    # Adam: robust default optimiser

    # ---------- 6. Training loop with early stopping ----------
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)               # folder for weights (git-ignored)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "lstm_checkpoint.pt")  # best model ka save path (checkpoints/ git me nahi jaata)
    best_f1, best_epoch, bad = -1.0, 0, 0                     # best validation score so far
    t_start = time.time()                                     # for the "train_minutes" result
    for epoch in range(1, args.epochs + 1):  # har epoch = poore training data par ek pass
        # VIVA Q: model.train() aur model.eval() me kya fark hai?
        #     A: train() me dropout ON hota hai (overfitting rokta hai); eval() me OFF (stable
        #        predictions). Isliye training loop me train(), predict me eval().
        model.train()                                         # dropout ON while learning
        total = 0.0                                           # running sum of the loss
        for xb, yb in batches(x_train, y_train, args.batch_size, shuffle=True):  # ek mini-batch: statements xb aur sahi labels yb
            ids, lengths = pad_batch(xb, vocab.pad_idx)  # batch ko padded matrix me badlo
            # VIVA Q: zero_grad() kyun?
            #     A: PyTorch gradients ko jodta (accumulate) jaata hai. Har batch se pehle purane
            #        gradients saaf na karein toh pichle batches ka gradient bhi mil jaata hai.
            opt.zero_grad()                                   # clear the previous step's gradients
            loss = loss_fn(model(ids, lengths), yb)           # how wrong are we on this batch?
            # VIVA Q: Backpropagation kya hai?
            #     A: Loss se peeche ki taraf chalke har weight ka gradient nikalna, yaani loss ghatane ke
            #        liye har weight kis taraf aur kitna badle.
            loss.backward()                                   # compute gradients (backpropagation)
            # Cap gradient size: LSTMs can have "exploding gradients" that wreck training.
            # VIVA Q: Gradient clipping kyun?
            #     A: LSTM me kabhi gradients bahut bade (exploding) ho jaate hain aur weights ek jhatke
            #        me kharab ho jaate hain. Unka size 1.0 par kaat dete hain.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()                                        # update the weights
            total += loss.item() * len(yb)  # epoch ka total loss jodte jao (baad me average)
        # After each epoch, score the VALIDATION set (data the model never trains on).
        # VIVA Q: Best epoch validation se kyun chunte hain, test se kyun nahi?
        #     A: Test set ko training ya selection me kabhi nahi dekhte, warna test score tuned (jhootha
        #        achha) ho jaata. Epoch validation se chunte hain; test sirf aakhir me ek baar.
        val_m = evaluate(y_val, predict(model, x_val, vocab.pad_idx))
        print(f"epoch {epoch:2d} | loss {total / len(x_train):.4f} | "  # har epoch ka loss aur validation score print
              f"val acc {val_m['accuracy']:.4f} macro-F1 {val_m['macro_f1']:.4f}")  # validation accuracy aur macro-F1
        # We pick the best epoch by macro-F1 (the project's headline metric).
        # VIVA Q: Epoch chunne ka metric macro-F1 kyun, accuracy kyun nahi?
        #     A: Macro-F1 har class ko barabar tolta hai, jabki accuracy bade classes ki taraf jhuk sakti
        #        hai. Project ka headline metric yahi hai.
        if val_m["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = val_m["macro_f1"], epoch, 0  # naya best mila: score yaad rakho aur sabr (bad) zero karo
            # Save the best weights + vocab, so we can reload exactly this model later.
            torch.save({"state_dict": model.state_dict(), "stoi": vocab.stoi, "args": vars(args)}, ckpt_path)
        else:
            bad += 1  # is epoch me sudhar nahi hua, sabr ka counter badhao
            if bad >= args.patience:  # patience khatam hui?
                # EARLY STOPPING: validation hasn't improved for `patience` epochs;
                # further training would only overfit, so stop and save time.
                print(f"Early stopping (best epoch {best_epoch})")
                break  # training rok do (early stopping)
    train_minutes = (time.time() - t_start) / 60  # training me lage minutes (results JSON ke liye)

    # Reload the BEST epoch (not the last one, which may be slightly overfit).
    # VIVA Q: Best epoch dobara load kyun karte hain?
    #     A: Memory me jo model hai wo aakhri epoch ka hai (thoda overfit ho sakta hai). Best validation
    #        wale epoch ke saved weights wapas load karte hain.
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])

    # ---------- 7. Final evaluation ----------
    # (a) The FULL test set: more rows => more reliable numbers.
    # VIVA Q: Full test aur 300-row subset dono kyun?
    #     A: Subset sab models ka fair headline hai (LLM bhi shamil). Full test me zyada rows hain toh
    #        zyada bharosemand. LSTM me subset 0.190 aur full 0.212, fark noise ke andar hai.
    full_m = evaluate(y_test, predict(model, x_test, vocab.pad_idx))
    # (b) The SHARED 300-row subset: the apples-to-apples number used to compare
    #     against BERT and the LLM (the LLM is too slow to run on all 1,283 rows).
    idx = shared_test_indices(test)
    sub_x, sub_y = [x_test[i] for i in idx], [y_test[i] for i in idx]  # test ki sirf shared 300 rows ke statements aur labels
    sub_pred, lat = predict_with_latency(model, sub_x, vocab.pad_idx)  # inki ek-ek predict karke latency bhi naapo
    sub_m = evaluate(sub_y, sub_pred)  # 300 rows par accuracy, macro-F1, confusion matrix (headline numbers)

    # ---------- 8. Cost estimate ----------
    avg_lat = float(np.mean(lat))   # average milliseconds per prediction
    # ms -> hours, times 1000 predictions, times $/hour, plus overhead.
    # VIVA Q: est_cost_per_1k kaise bana?
    #     A: avg latency (ms) ko 1000 se baanto (seconds), 3600 se baanto (ghante), 1000 predictions se
    #        guna karo, phir 0.35 x 1.10 se. Formula upar OVERHEAD ke paas samjhaya hai.
    cost_per_1k = avg_lat / 1000 / 3600 * 1000 * T4_PRICE_PER_HOUR * OVERHEAD

    # ---------- 9. Save results in the SHARED schema ----------
    # VIVA Q: Results JSON aur preds CSV me kya hota hai?
    #     A: JSON: accuracy, macro_f1 (300-row subset par), params, train_minutes, gpu, latency, cost,
    #        confusion_matrix, aur full_test me poore test ke numbers. Preds CSV: id, y_true, y_pred,
    #        latency_ms; error analysis aur McNemar test isi se chalte hain.
    results = build_results_dict(
        model_name="lstm",  # results file ka naam (results/lstm.json)
        metrics=sub_m,                       # headline numbers = shared subset
        params=n_params,  # parameter count
        train_minutes=round(train_minutes, 2),  # training time (minutes)
        gpu="CPU",                           # honest: trained/measured on CPU, not a T4
        avg_latency_ms=round(avg_lat, 3),  # average latency (ms per statement)
        est_cost_per_1k=round(cost_per_1k, 6),  # 1,000 predictions ka andaza kharch (USD)
        cost_assumption="Priced at T4 rate $0.35/hour incl. 10% overhead; batch size 1, CPU inference",  # kharch ki shart: T4 ka rate, par LSTM CPU par naapa gaya
        notes=f"Text-only BiLSTM, trainable embeddings, best epoch {best_epoch}, "  # free-text note (best epoch aur class weights)
              f"class-weighted loss={not args.no_class_weights}",  # (note ka doosra hissa)
    )
    # Extra fields (additive, they don't break the shared schema):
    results["full_test"] = {"accuracy": full_m["accuracy"], "macro_f1": full_m["macro_f1"], "n": len(y_test)}
    results["val_best_macro_f1"] = best_f1  # best epoch ka validation macro-F1 (selection ka saboot)
    results["hyperparams"] = vars(args)      # lets anyone reproduce this exact run

    save_results_json(results, "lstm", RESULTS_DIR)                       # results/lstm.json
    save_predictions_csv(idx, sub_y, sub_pred, lat, "lstm", RESULTS_DIR)  # results/lstm_preds.csv
    # VIVA Q: LSTM ka score itna kam (19%) kyun?
    #     A: Random guessing 16.7%, sabse common class (half-true) hamesha bolne par 20.7%; LSTM subset
    #        par 19.0% (full test 21.2%), yaani chance ke aas-paas. Wajah: sirf ~10k sentences, koi pre-
    #        training nahi, aur LIAR khud mushkil hai (text-only models 25-30% par rukte hain).
    #        BERT/RoBERTa pre-training ki wajah se ~28-29% tak jaate hain.
    print(f"Subset ({len(idx)}): acc {sub_m['accuracy']:.4f} macro-F1 {sub_m['macro_f1']:.4f} | "  # aakhri line: subset aur full test ke numbers
          f"full test: acc {full_m['accuracy']:.4f} macro-F1 {full_m['macro_f1']:.4f} | "  # full test (1,283 rows) ke numbers
          f"latency {avg_lat:.2f} ms | train {train_minutes:.2f} min")  # latency aur training time


# Run main() only when this file is executed directly (python -m baselines.train),
# not when another file merely imports from it (e.g. the unit tests).
if __name__ == "__main__":
    main()  # script seedha chalane par pipeline shuru
