# Running the LLM experiments (Colab T4 / Kaggle 2x T4)

The same steps are in `notebooks/run_llm_colab.ipynb` (outputs cleared). The saved results were produced with a Kaggle run of the same commands.

Each cell below is one Colab cell. Runtime > Change runtime type > T4 GPU.
Total: about 2 x (3 min model load + 300 statements x ~0.5-1.5 s) = roughly 25 min.

```python
# 1. Code + deps (the repo pins torch 2.1; Colab's own torch is fine, only add these)
!git clone https://github.com/arunchowdary-1/Fake-News-Detection-Fine-tuned-Transformers-vs-Zero-shot-LLMs.git repo
%cd repo
!pip install -q -U bitsandbytes accelerate transformers 'datasets<4' scikit-learn
```

```python
# 2. Smoke test on 20 statements first (catches parse problems before the long run)
!python -m llm.evaluate_llm --4bit --mode zeroshot --limit 20
```

```python
# 3. Full runs on the shared 300-row test subset (seed 42, common/test_indices.txt)
!python -m llm.evaluate_llm --4bit --mode zeroshot   # -> results/llm_zeroshot.json
!python -m llm.evaluate_llm --4bit --mode fewshot    # -> results/llm_fewshot.json
```

```python
# 4. Cost table, then copy results to the shared Drive folder
!python -m llm.cost_analysis
from google.colab import drive; drive.mount('/content/drive')
!cp results/llm_*.json results/llm_*_preds.csv results/cost_* "/content/drive/MyDrive/<shared folder>/"
```

Notes
- LIAR is a script dataset, so `datasets` must be `<4` (pinned above). If its download host is blocked (it returns 403 on Kaggle),
  put the original `train/valid/test.tsv` in a folder (they are in `liar_data/` once PR #26 is merged) and run with
  `LIAR_TSV_DIR=<folder>`; same splits (10,269 / 1,284 / 1,283).
- If Colab is out of GPU, use Kaggle (2x T4): same commands; the 4-bit model fits on one T4 (~5.5 GB).
- Too slow? Fallback is `--model Qwen/Qwen2.5-3B-Instruct`; say so in the report (the JSON records the model id).
- Only `--limit` runs write `llm_*_limit.json`, so a quick check never overwrites a real result.
- Set `HF_TOKEN` in the environment if the model download asks for it. Never paste it into a notebook cell that gets committed.
