# BERT fine-tuning (report section)

## Method

We fine-tune **bert-base-uncased** (109.5M parameters) with a 6-class classification head on LIAR,
using the shared train/validation/test split, the shared label order and seed 42. Two runs differ only in input:

- **Run 1, statement only:** the statement text, max 128 tokens.
- **Run 2, metadata + statement:** `speaker: X | party: Y | subject: Z | history: a,b,c,d,e [SEP] statement`,
  max 256 tokens. `history` is the speaker's five credit-history counts supplied with LIAR.

Training setup (identical for both runs): AdamW, learning rate 2e-5, batch size 16, 3 epochs, 10% linear warm-up,
weight decay 0.01, fp16 mixed precision, gradient clipping 1.0, inverse-frequency class weights in the loss
(LIAR is imbalanced and macro-F1 weights every class equally). The best epoch is chosen by validation
macro-F1 (it was epoch 3 in both runs). Hardware: Colab Tesla T4.

Evaluation: the shared 300-row stratified test subset (headline numbers, comparable with the LSTM and LLM)
and the full 1,283-row test set (more reliable).

## Results

| Run | Acc (300) | Macro-F1 (300) | Acc (full 1,283) | Macro-F1 (full) | Train (min) | s / 1,000 stmts | GPU mem train / infer (MB) | $ / 1,000 |
|---|---|---|---|---|---|---|---|---|
| BERT, statement only | 0.293 | 0.289 | 0.273 | 0.277 | 3.2 | 14.7 | 2,144 / 1,898 | 0.0016 |
| BERT, metadata + statement | 0.457 | 0.470 | 0.449 | 0.457 | 3.8 | 13.9 | 2,301 / 1,907 | 0.0015 |

Inference time is batch size 1 on the T4. Cost uses the shared assumption of $0.35/hour plus 10% overhead.
Parameter count is 109,486,854 for both runs.

**Metadata helps a lot.** Adding speaker, party, subject and credit history raises accuracy by about 16 points
on the subset (29.3% to 45.7%) and 17.6 points on the full test set (27.3% to 44.9%), at almost no extra cost
(+0.5 min training, same inference speed). With 300 rows the 95% interval is roughly +/-5 points, but the gap
is well outside that, and it holds on the full test set.

**Where it helps.** Statement-only BERT is close to guessing on the middle classes (F1: half-true 0.19,
barely-true 0.26) and does best on the extremes (true 0.37, false 0.34). With metadata every class improves:
pants-fire 0.27 to 0.65, mostly-true 0.31 to 0.55, false 0.34 to 0.43, barely-true 0.26 to 0.43,
true 0.37 to 0.40, half-true 0.19 to 0.36. Half-true remains the hardest class; in the metadata confusion matrix it is
mostly confused with mostly-true (18 of 62 rows) and false/barely-true (9 each).

## Caveat to state in the report

The five credit-history counts are the speaker's record of past statements. In the LIAR release these counts
include the current statement's own label, so a model can partly read the answer from the speaker's history.
The metadata run is therefore not a pure "does the model understand the claim" result, and its gain should be
presented as the value of speaker information, not of better language understanding. The statement-only run
is the fair comparison with the zero-shot LLM, which only sees the statement. A run without the history field
(speaker, party and subject only) would separate the two effects if time allows.

## Notes and limitations

- Best epoch was the last one in both runs, so validation macro-F1 was still rising; more epochs may help a little.
- One run per setting (seed 42), so run-to-run variance is not measured.
- Statement-only BERT (27.3% full-test accuracy) is in the range usually reported for text-only LIAR models,
  so the low absolute number is expected for this task, not a sign of a bug.
