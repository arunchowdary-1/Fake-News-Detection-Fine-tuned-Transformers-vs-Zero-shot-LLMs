# Zero-shot and few-shot LLM prompting (report section)

## Method

We prompt **Qwen2.5-7B-Instruct** (7.6B parameters), loaded in **4-bit NF4** with bitsandbytes, with no
fine-tuning. Each statement is classified into the six LIAR labels (pants-fire, false, barely-true,
half-true, mostly-true, true) by a chat prompt whose system message defines the task and requires a
label-only answer. Decoding is greedy (`max_new_tokens=12`), so runs are deterministic.

- **Zero-shot:** system message plus the statement.
- **Few-shot:** the same prompt preceded by six worked examples, one per class, as user/assistant turns.
  Examples are drawn from the *training* split with seed 42 (statements of 60-200 characters), so they
  cannot overlap the test subset. Their order is shuffled, not sorted by label, so that the last example's
  label is not favoured.
- **Strict output parsing:** an answer counts only if it contains exactly one valid label (spelling variants
  such as "pants on fire" are normalised; negations and multiple labels are rejected). Anything else is
  recorded as *invalid* and counted as wrong, so parsing never inflates accuracy. Invalid outputs are never
  turned into predictions in the confusion matrix.
- **Input:** the statement text only, no speaker or credit-history metadata, the same input as the
  statement-only BERT.
- **Evaluation:** the shared 300-row stratified test subset (seed 42), shared label order and the shared
  `evaluate` metrics. Hardware: one Tesla T4 (Kaggle), batch size 1, model weights cached on disk.
- **No prompt tuning:** each prompt was written once and run once. A 20-statement smoke test checked only that the output format parses; no prompt was changed after seeing results.

## Results (300-row test subset)

| Model | Accuracy | Macro-F1 | Invalid outputs |
|---|---|---|---|
| Qwen2.5-7B zero-shot | 0.167 | 0.126 | 1 / 300 |
| Qwen2.5-7B few-shot (6 examples) | 0.250 | 0.183 | 0 / 300 |
| BERT, statement only (for reference) | 0.293 | 0.289 | 0 / 300 |

For scale: guessing uniformly gives 16.7% accuracy and always answering the most common class (half-true) gives
20.7%. With 300 rows the 95% interval on an accuracy is roughly +/-5 points.

**Few-shot helps, and the gain is real.** Accuracy rises from 16.7% to 25.0% (50 to 75 correct). On the same
300 statements, few-shot is right where zero-shot is wrong 48 times and the reverse 23 times (exact McNemar
p = 0.004). Zero-shot is at chance level.

**The LLM is not significantly different from statement-only BERT in accuracy, but is clearly worse on
macro-F1.** BERT is right where few-shot is wrong 66 times and the reverse 53 times (p = 0.27, so the accuracy
gap of 4 points is within noise), but macro-F1 is 0.289 against 0.183. Zero-shot is significantly worse than
BERT (p = 0.0004).

**Where the errors come from: strong label bias.** The model does not spread its answers over the six classes:

- Zero-shot answers *pants-fire* for 136 of 300 statements (45%), *half-true* for 90, and almost never
  *barely-true* (5) or *true* (4). It rarely says "true", although 49 statements are true.
- Few-shot moves the bias to *false* (146 of 300, 49%), and *never* answers *mostly-true* (0 of 300, F1 = 0.00).
  Its best classes are *false* (F1 0.34) and *half-true* (0.36).

The six-way LIAR scale is subtle (barely-true vs half-true vs mostly-true), and an instruction-tuned model
without task data seems to map "sounds dubious" to false or pants-fire and has no calibrated sense of the middle
classes. Six examples shift the bias but do not remove it.

## Cost analysis: fine-tuned 110M vs prompted 7B

Cost model for every model: Tesla T4 at $0.35/hour plus 10% overhead, batch size 1. Training cost is one-off;
inference cost is per 1,000 statements.

| Model | Acc | Macro-F1 | Params | Train (min) | s / 1,000 stmts | $ / 1,000 | GPU peak MB |
|---|---|---|---|---|---|---|---|
| BERT, statement only | 0.293 | 0.289 | 109M | 3.2 | 14.7 | 0.0016 | 2,144 |
| Qwen2.5-7B zero-shot | 0.167 | 0.126 | 7,616M | 0 | 393.5 | 0.0421 | 5,743 |
| Qwen2.5-7B few-shot | 0.250 | 0.183 | 7,616M | 0 | 661.3 | 0.0707 | 5,789 |

(`results/cost_comparison.md` has the full table, including the other models. The BERT-with-metadata row is
left out here because its credit-history input partly contains the answer; see the BERT section.)

1. **Size and memory.** The LLM has about 70x the parameters (7.62B against 109.5M) and needs 5.7-5.8 GB of GPU
   memory even in 4-bit, against about 2 GB for BERT. BERT fits on almost any GPU; the 4-bit 7B model needs
   an 8 GB-class card or better.
2. **Speed and price.** Statement-only BERT classifies 1,000 statements in 14.7 s for $0.0016. Zero-shot takes
   393 s (about 27x slower and costlier) and few-shot 661 s for $0.071 (about 45x). Few-shot is slower than
   zero-shot because every request carries six extra examples.
3. **Training pays back almost immediately.** The LLM has no training cost, but BERT's one-off 3.2 minutes of
   T4 time costs about $0.02, which equals the LLM's inference cost on roughly 300 predictions. After about 300
   statements the fine-tuned model is the cheaper option overall. (This ignores the human effort of labelled data
   and engineering, which the LLM avoids.)
4. **Accuracy.** The fine-tuned 110M model matches or beats the prompted 7B model on this task, with a
   macro-F1 of 0.289 against 0.183 for the better LLM run, at about 1/45 of the cost and 1/70 of the size.
   The accuracy difference alone (0.293 against 0.250) is not statistically significant on 300 rows.

## Limitations

- Absolute accuracy is low for every model: LIAR is hard and noisy, and statement-only models usually score
  in the 25-30% range, so this gap should be read as a ranking, not a verdict on the models.
- One prompt, one model, one seed, 300 statements. Other prompts, more examples per class, or a larger model
  may close part of the gap. A different 7B model (for example Mistral-7B) was not tested.
- The LLM sees only the statement. Giving it the speaker and party would be a fairer comparison with the
  metadata-based BERT run, but that was not run.
- Latency and cost use batch size 1 on a T4. Batching or an inference server such as vLLM would reduce the
  LLM's cost per statement considerably, so the cost ratios here are an upper bound on the real gap.
- Retrieved few-shot examples (similar training statements per query) and fine-tuning the LLM (QLoRA) are
  natural next steps and were out of scope.
