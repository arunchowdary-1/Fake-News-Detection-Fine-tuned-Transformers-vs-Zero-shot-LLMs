# Running the BERT / RoBERTa fine-tuning (Colab T4 / Kaggle 2x T4)

Runtime > Change runtime type > T4 GPU. Each block is one Colab cell. About 4 min per run.

```python
!git clone https://github.com/arunchowdary-1/Fake-News-Detection-Fine-tuned-Transformers-vs-Zero-shot-LLMs.git repo
%cd repo
!pip install -q -U transformers scikit-learn accelerate "datasets<4"
```

```python
# quick smoke test, then the real runs
!python -m finetune.train_bert --mode text --epochs 1 --max-train 200
!python -m finetune.train_bert --mode text    # -> results/bert_text.json
!python -m finetune.train_bert --mode meta    # -> results/bert_meta.json
```

RoBERTa: same script, `--model-name roberta-base --run-name roberta_text` (and `--mode meta --run-name roberta_meta`).

Data: `liar_data/*.tsv` from the repo is used automatically (set `LIAR_TSV_DIR` to change the folder).
If the files are missing, the loader falls back to downloading the official zip.

The notebook for the RoBERTa runs, saved with the outputs of the actual Colab T4 run, is `notebooks/roberta_colab.ipynb`.
