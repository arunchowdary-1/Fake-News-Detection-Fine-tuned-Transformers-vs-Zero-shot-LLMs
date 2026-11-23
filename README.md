# LIAR Truth Detector: fine-tuned transformers vs zero-shot LLMs

Rates how true a short political statement is on the six-level LIAR scale (pants-fire, false, barely-true,
half-true, mostly-true, true), and compares three ways of doing it: a BiLSTM trained from scratch, BERT and
RoBERTa fine-tuned on LIAR, and the Qwen2.5-7B-Instruct LLM used with a prompt only (zero-shot and few-shot).
All models are scored the same way: same data split, labels, seed (42), 300-statement test subset, metrics and
cost model.

**Full write-up:** [docs/REPORT.pdf](docs/REPORT.pdf) (source: [docs/REPORT.md](docs/REPORT.md)).

## Results (shared 300-statement test subset)

| Model | Accuracy | Macro-F1 | Fake vs real | $ per 1,000 statements |
|---|---|---|---|---|
| RoBERTa + speaker info | 0.473 | 0.483 | 75.7% | 0.0018 |
| BERT + speaker info | 0.457 | 0.470 | 75.3% | 0.0015 |
| BERT, text only | 0.293 | 0.289 | 64.0% | 0.0016 |
| RoBERTa, text only | 0.283 | 0.282 | 63.0% | 0.0015 |
| Qwen2.5-7B few-shot | 0.250 | 0.183 | 59.0% | 0.0707 |
| BiLSTM | 0.190 | 0.173 | 56.3% | 0.0002 (CPU) |
| Qwen2.5-7B zero-shot | 0.167 | 0.126 | 56.7% | 0.0421 |

Random guessing scores 16.7% on six levels. "Fake vs real" groups pants-fire, false and barely-true against the
other three. With 300 statements each accuracy is uncertain by about 5 points.

- A fine-tuned 110M-parameter BERT matches or beats the prompted 7.6B model in accuracy and is clearly better in
  macro-F1, while being about 70 times smaller and 45 times cheaper to run.
- **Read the speaker-info rows with care.** LIAR's credit-history counts include the rating of the statement itself,
  so part of that gain is not understanding of the claim. The fair comparison with the LLM (which sees only the
  statement) is the text-only rows.

## Repository layout

| Folder | Contents |
|---|---|
| `common/` | Shared labels and config, data loader, test-subset selection and scoring used by every model |
| `baselines/` | BiLSTM model and training script |
| `finetune/` | BERT / RoBERTa fine-tuning (`--mode text` or `meta`) |
| `llm/` | Qwen prompting: prompts, few-shot selection, strict answer parser, evaluation and `cost_analysis.py` |
| `analysis/` | Error analysis from the saved confusion matrices and predictions |
| `results/` | Per-model result JSON and prediction CSV, plus the cost comparison |
| `liar_data/` | The LIAR `train`, `valid` and `test` TSV files |
| `notebooks/` | Colab / Kaggle runners for the BERT, RoBERTa, LSTM and LLM runs |
| `report/`, `docs/` | Report source, figures, and the per-model write-ups |
| `tests/` | 99 automated tests |

## Run it

```bash
pip install -r requirements.txt          # Colab and Kaggle already have torch; see the notebooks
python -m pytest tests/ -q               # all 99 tests pass
```

If the Hugging Face download of LIAR fails (it does on some networks), read the TSV files in `liar_data/` instead:
`export LIAR_TSV_DIR=liar_data` (PowerShell: `$env:LIAR_TSV_DIR = "liar_data"`).

```bash
python -m baselines.train                                    # BiLSTM, a CPU is enough (about 4 min)
python -m finetune.train_bert --mode text                    # BERT, statement only
python -m finetune.train_bert --mode meta                    # BERT + speaker, party, subject, history
python -m finetune.train_bert --mode text --model-name roberta-base --run-name roberta_text   # RoBERTa
python -m llm.evaluate_llm --4bit --mode zeroshot            # Qwen zero-shot (needs a GPU)
python -m llm.evaluate_llm --4bit --mode fewshot             # Qwen few-shot
python -m llm.cost_analysis                                  # cost vs accuracy tables
python -m analysis.error_analysis --run roberta_text         # error analysis for one run
python report/make_figures.py                                # rebuild the report figures
```

BERT, RoBERTa and the LLM runs need a GPU (a free Colab or Kaggle T4 is enough). The notebooks run the same
commands step by step: `notebooks/run_bert_colab.ipynb`, `run_roberta_colab.ipynb`, `run_lstm_colab.ipynb` and
`run_llm_colab.ipynb`. More detail: [colab/RUN_COLAB.md](colab/RUN_COLAB.md), [llm/RUN_COLAB.md](llm/RUN_COLAB.md)
and [report/README.md](report/README.md).

Every run writes `results/<name>.json` (metrics, parameters, training and inference time, GPU memory, cost) and
`results/<name>_preds.csv`.

## Notes and limitations

- **Cost model:** every model is priced as a Tesla T4 at $0.35 per hour plus 10% overhead, one statement at a
  time. The BiLSTM was timed on a CPU but priced at the same rate, so its speed and cost are not like-for-like.
- **LLM:** Qwen2.5-7B-Instruct is used without any training, loaded in 4-bit FP4. Each prompt was written once and
  run once. Invalid answers count as wrong.
- **Scope:** one seed, 300 test statements, one LLM and one prompt, and US political statements only.
- **Not built:** the demo web app. `app/` is empty, and there is no CI workflow.

## Team

Prachi Dhakad, Sandhya Kaushal, Shilpi Shaw and Arun Kumar Nallamothu, B.Tech Computer Science, Sitare University.

Dataset: Wang, W. Y. (2017) "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection, ACL.
