# Zero-shot and Few-shot LLM (report section, draft)

> DRAFT: every `TBD` is filled from `results/llm_zeroshot.json`, `results/llm_fewshot.json`
> and `results/cost_comparison.md` after the Colab runs. Do not submit with TBD left in.

## Method

We prompt **Qwen2.5-7B-Instruct** (7.6B parameters) loaded in **4-bit NF4** with bitsandbytes, with no
fine-tuning. Each statement is classified into the six LIAR labels (pants-fire, false, barely-true,
half-true, mostly-true, true) by a chat prompt whose system message defines the task and requires a
label-only answer. Decoding is greedy (`max_new_tokens=12`), so runs are deterministic.

- **Zero-shot:** system message plus the statement.
- **Few-shot:** the same prompt preceded by six worked examples, one per class, as user/assistant turns.
  Examples are drawn from the *training* split with seed 42 (statements of 60-200 characters), so
  they cannot overlap the test subset. Their order is shuffled, not sorted by label, to avoid biasing
  the model toward the last example's label.
- **Output parsing is strict:** an answer counts only if it contains exactly one valid label (spelling
  variants such as "pants on fire" are normalised; negations and multiple labels are rejected). Anything
  else is recorded as *invalid* and scored as wrong, so parsing never inflates accuracy. Invalid rate is
  reported.
- **Metadata is not given to the LLM**, only the statement text, the same input as the text-only models.
  (Adding speaker and history would be a fair extension; it is not what was run.)
- **Evaluation:** the same 300-row stratified test subset (`common/test_indices.txt`), label order and
  seed 42 as every other model; same `evaluate()` function.

## Results

TBD: paste the LLM rows of `results/cost_comparison.md`.

| Model | Acc | Macro-F1 | Invalid |
|---|---|---|---|
| Qwen2.5-7B zero-shot | TBD | TBD | TBD |
| Qwen2.5-7B few-shot | TBD | TBD | TBD |

TBD (2-3 sentences): did the six examples help? Which classes does the model confuse? (see its confusion
matrix; compare with Sandya's error analysis).

## Cost analysis: fine-tuned 110M vs prompted 7B

Cost model for every model: T4 at $0.35/hour plus 10% overhead, batch size 1. Training cost is one-off;
inference cost is per 1,000 statements.

TBD: paste the full table and the "fine-tuned vs LLM" table from `results/cost_comparison.md`
(`python -m llm.cost_analysis`). The plot data is `results/cost_vs_accuracy.csv` (for Prachi).

Points to make once the numbers are in:

1. **Size:** BERT-base has ~110M parameters, Qwen2.5-7B ~7.6B (about 70x more). In 4-bit the LLM still needs
   ~TBD MB of GPU memory against ~TBD MB for BERT inference.
2. **Speed and price:** LLM inference takes TBD s per 1,000 statements against TBD s for BERT, so it costs
   TBD x more per prediction. Few-shot is slower again because the prompt carries six extra examples.
3. **Training vs inference:** the LLM has no training cost, BERT paid ~TBD min of T4 time once. The break-even
   column gives the number of predictions after which fine-tuning is cheaper overall.
4. **Accuracy:** TBD, state plainly which is better and by how much. With 300 test rows, one accuracy point is
   3 statements and the 95% interval is roughly +/-5 points, so small gaps are not significant.
5. **Caveat:** LIAR is hard and noisy (best published text-only accuracy is around 25-30% on 6 classes), so all models are
   close to each other; the cost gap is large, the accuracy gap is small. Latency is batch-size-1 on a shared
   Colab T4; batching or vLLM would cut the LLM's per-statement cost substantially.
