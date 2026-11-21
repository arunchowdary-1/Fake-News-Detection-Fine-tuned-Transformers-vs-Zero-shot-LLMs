# Fake News Detection: Project Plan
## Fine-tuned Transformers vs Zero-shot LLMs

**Timeline:** 60 minutes  
**Team:** 4 people (A: Data, B: Fine-tuning, C: LLM, D: App & Analysis)  
**Goal:** Build and compare LSTM → BERT fine-tuning vs zero-shot LLM prompting on LIAR dataset (6-class truthfulness classification)

---

## 1. Project Overview

### Objective
Compare three approaches to fake news classification:
1. **LSTM baseline** (small, trainable embeddings)
2. **BERT fine-tuning** (text-only and with metadata fusion)
3. **Zero-shot LLM prompting** (7B model, 4-bit quantization)

**Primary Question:** Is a 110M fine-tuned model better than a 7B prompted LLM? (Cost + accuracy)

### Dataset: LIAR
- **Source:** Hugging Face `datasets.load_dataset("liar")`
- **Train:** ~10.2k | **Val:** ~1.3k | **Test:** ~1.3k
- **Classes:** 6 (pants-fire, false, barely-true, half-true, mostly-true, true)
- **Fields:** statement, label, subject, speaker, job_title, state_info, party_affiliation, credit_history (5 counts), context
- **Split for all models:** Fixed stratified ~300-row test subset (same indices for fair comparison)

---

## 2. Tech Stack & GPU Requirements

| Component | Technology | Notes |
|---|---|---|
| **Data Loading** | Hugging Face `datasets` | Shared loader in `common/` |
| **LSTM Baseline** | PyTorch + trainable embeddings | ~5M params, trains in ~2 min |
| **BERT Fine-tuning** | `transformers` (bert-base-uncased) | 110M params; fallback: DistilBERT |
| **LLM Inference** | `bitsandbytes` (4-bit) or `vLLM` | Qwen2.5-7B-Instruct or Mistral-7B-Instruct |
| **Backend API** | FastAPI + Pydantic | Validation, error handling, local only |
| **Frontend UI** | Streamlit (recommended) or Gradio | Input: statement, model selector, output: label + confidence |
| **Evaluation** | scikit-learn (accuracy, macro-F1, confusion matrix) | Shared `evaluate()` function |
| **Experiment Tracking** | JSON files in `results/` | No third-party MLOps required |

### GPU Access (Critical)
- **Required:** 1× T4 GPU (minimum) or equivalent
- **Source:** Google Colab (free, recommended) or Kaggle Notebooks or approved org GPU
- **Check availability NOW** before starting
- **Fallback:** DistilBERT + 1.5B LLM + reduced samples (see risks section)

---

## 3. Repository Structure

```
Fake-News-Detection-Fine-tuned-Transformers-vs-Zero-shot-LLMs/
├── common/                      # Shared by all (Person A drives)
│   ├── __init__.py
│   ├── data_loader.py           # LIAR loader, label_to_idx, metadata builder
│   ├── evaluate.py              # Shared evaluate(y_true, y_pred) function
│   └── config.py                # Shared constants, label order, paths
├── baselines/                   # LSTM baseline (Person A)
│   ├── __init__.py
│   ├── lstm_model.py
│   ├── train.py
│   ├── results.json             # Generated during training
│   └── checkpoint_lstm.pt       # .gitignore this
├── finetune/                    # BERT fine-tuning (Person B)
│   ├── __init__.py
│   ├── bert_model.py
│   ├── train_text_only.py
│   ├── train_with_metadata.py
│   ├── results_bert_text.json   # Generated
│   ├── results_bert_meta.json   # Generated
│   └── checkpoint_bert_*.pt     # .gitignore
├── llm/                         # Zero-shot LLM prompting (Person C)
│   ├── __init__.py
│   ├── prompt.py                # Prompt template + system message
│   ├── parser.py                # Output parser (strict label matching)
│   ├── inference.py             # Load model, run inference with retry logic
│   ├── evaluate_llm.py           # Main evaluation script
│   ├── results_llm.json         # Generated
│   └── test_parser.py           # Unit tests
├── app/                         # FastAPI + Streamlit (Person D)
│   ├── backend.py               # FastAPI service
│   ├── frontend.py              # Streamlit UI
│   ├── schemas.py               # Pydantic models
│   └── test_api.py              # Unit tests
├── docs/                        # Documentation (Person D)
│   ├── README.md                # Setup, training, evaluation, demo instructions
│   ├── ARCHITECTURE.md          # Design decisions, ablation rationale
│   └── references.md            # Papers, links, citations
├── tests/                       # Shared tests
│   ├── test_common.py           # Data loading, evaluate()
│   └── test_parser.py           # LLM parser
├── results/                     # Experiment outputs (auto-generated, in .gitignore)
│   ├── lstm_preds.csv
│   ├── lstm.json
│   ├── bert_text_preds.csv
│   ├── bert_text.json
│   ├── bert_meta_preds.csv
│   ├── bert_meta.json
│   ├── llm_preds.csv
│   └── llm.json
├── .gitignore                   # Checkpoints, .env, data cache
├── requirements.txt             # Frozen versions
└── plan.md                      # This file
```

**Key Rule:** Each person owns a folder. Only **Person A** commits to `common/`. Merge conflicts = feature branches + PRs.

---

## 4. Shared Contracts (Agree on these in minute 0–5)

### Label Order (Fixed)
```python
LABEL_TO_IDX = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}
```

### Prediction CSV Format
**File:** `results/<model>_preds.csv`
```
id,y_true,y_pred,latency_ms
0,2,2,45.2
1,4,4,43.1
...
```

### Results JSON Format
**File:** `results/<model>.json`
```json
{
  "model": "lstm",
  "accuracy": 0.287,
  "macro_f1": 0.24,
  "params": 5000000,
  "train_minutes": 2.3,
  "gpu": "CPU (or T4 for BERT/LLM)",
  "avg_latency_ms": 45.2,
  "est_cost_per_1k": 0.08,
  "cost_assumption": "T4: $0.35/hour, includes 10% overhead",
  "n_eval": 300,
  "invalid_outputs": 0,
  "notes": "Text-only baseline"
}
```

### Test Subset
- **Fixed indices:** Person A generates a stratified random split (seed=42), saves indices to `common/test_indices.txt`
- **All models use the same subset** for fair comparison
- **Size:** ~300 samples (stratified by label)

---

## 5. Timeline & Milestones

| Minute | Milestone | Checkpoint |
|---|---|---|
| **0–10** | Setup complete, `common/` pushed | Branch strategy locked, GPU tested, Person A's stub merged |
| **10–25** | Baselines running | LSTM + BERT text-only training in progress |
| **25–45** | Fine-tuning + LLM runs | B runs BERT+meta; C runs full LLM on 300 samples |
| **45–55** | Results collected & analysis | D builds comparison table; A analyzes confusion matrices |
| **55–60** | Docs, PRs merged, demo rehearsal | README done, all branches → main, final checks |

### Parallelization Strategy
1. **Min 0–10:** Everyone waits for Person A's `common/` push. Person D scaffolds the API stub.
2. **Min 10–25:** A runs LSTM; B runs BERT text-only; C tests parser on 20 samples; D wires UI to stub.
3. **Min 25–45:** B runs BERT+metadata (sequential, same GPU). C runs full LLM (can run in parallel on a different GPU or after B finishes). A starts error analysis.
4. **Min 45–55:** All results are JSON files. D reads them and builds tables. Everyone reviews PRs.
5. **Min 55–60:** Final merges, docs, rehearsal.

**GPU Contention:** If one T4, schedule: A (2 min) → B text-only (5 min) → B metadata (5 min) → C LLM (10 min). Total ~22 min.

---

## 6. Person-by-Person Responsibilities

### Person A: Data Loader & Baseline
**Deliverables:**
- `common/data_loader.py`: Load LIAR, label_to_idx, test_indices
- `common/evaluate.py`: Shared metrics function
- `baselines/lstm_model.py` + `train.py`: LSTM model and training loop
- Confusion matrix analysis in write-up
- **Push `common/` by minute 10** (blocks everyone else)

**Pseudo-code:**
```python
# common/data_loader.py
from datasets import load_dataset
def load_liar():
    ds = load_dataset("liar")
    return ds["train"], ds["validation"], ds["test"]

def build_metadata_text(row):
    return f"speaker: {row['speaker']} | party: {row['party']} | subject: {row['subject']} | {row['statement']}"

# common/evaluate.py
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, output_dict=True)
    }

# baselines/train.py
# Load data, train BiLSTM, save results/<model>.json and preds CSV
```

---

### Person B: BERT Fine-tuning
**Deliverables:**
- `finetune/bert_model.py`: BERT wrapper for 6-class classification
- `finetune/train_text_only.py`: Text-only fine-tuning (3 epochs, fp16, early stopping)
- `finetune/train_with_metadata.py`: Metadata-prepended fine-tuning (ablation)
- Two results JSON files, two checkpoint files (outside git)
- Checkpoint metadata (params, epochs, learning rate) in results JSON

**Fallback:** If GPU is slow, use DistilBERT (66M params) or reduce max_len to 64.

**Pseudo-code:**
```python
# finetune/bert_model.py
from transformers import BertForSequenceClassification, BertTokenizer
class BertClassifier:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    def predict(self, texts):
        # Tokenize, run inference, return logits
        pass

# finetune/train_text_only.py
# Load data via common.data_loader
# Train on (statement, label) for 3 epochs
# Evaluate on shared test subset
# Save checkpoint + results JSON
```

---

### Person C: Zero-shot LLM Prompting
**Deliverables:**
- `llm/prompt.py`: Prompt template + system message
- `llm/parser.py`: Strict output parser (match only the 6 labels, retry once, mark invalid if parsing fails)
- `llm/inference.py`: Load model in 4-bit, run inference with retry + timeout
- `llm/test_parser.py`: Unit tests for the parser
- Results JSON + predictions CSV

**Model Options (team to approve):**
- Recommended: Qwen2.5-7B-Instruct (permissive license)
- Alternative: Mistral-7B-Instruct
- Fallback if slow: Llama2-7B-Chat or 1.5B variant

**Pseudo-code:**
```python
# llm/prompt.py
SYSTEM_MESSAGE = """You are a fact-checking expert. Classify the given political statement as one of these labels:
pants-fire, false, barely-true, half-true, mostly-true, true

Respond with ONLY the label, nothing else."""

PROMPT_TEMPLATE = "Statement: {statement}\nLabel:"

# llm/parser.py
import re
def parse_output(text):
    text = text.strip().lower()
    for label in ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]:
        if label in text:
            return label
    raise ValueError(f"Could not parse: {text}")

# llm/inference.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
def load_model_4bit():
    # Load Qwen2.5-7B with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="auto", load_in_4bit=True)
    return model

def predict_batch(statements, model, tokenizer):
    # Batch inference with retry logic
    preds = []
    for stmt in statements:
        try:
            output = model.generate(...) # LLM inference
            label = parse_output(output)
            preds.append(label)
        except:
            preds.append("INVALID")
    return preds
```

---

### Person D: Backend, Frontend & Analysis
**Deliverables:**
- `app/backend.py`: FastAPI service with `/predict` and `/results` endpoints
- `app/frontend.py`: Streamlit UI
- `app/schemas.py`: Pydantic validation
- `app/test_api.py`: Unit tests
- `docs/README.md`: Setup + usage instructions
- `docs/ARCHITECTURE.md`: Design decisions
- Comparison table: cost vs accuracy, confusion matrices, final write-up

**Pseudo-code:**
```python
# app/backend.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import json

app = FastAPI()

class PredictionRequest(BaseModel):
    statement: str
    model: str = "bert"
    
    @validator("statement")
    def validate_statement(cls, v):
        if not v or len(v) > 1000:
            raise ValueError("Statement must be 1-1000 chars")
        return v

@app.post("/predict")
def predict(req: PredictionRequest):
    # Load model, predict, return label + latency
    return {"label": "true", "confidence": 0.95, "latency_ms": 45.2}

@app.get("/results")
def get_results():
    # Read results/*.json, build comparison table
    with open("results/lstm.json") as f:
        lstm_res = json.load(f)
    # ... repeat for BERT, LLM
    return {"models": [lstm_res, ...]}

# app/frontend.py (Streamlit)
import streamlit as st
st.title("Fake News Detector")
statement = st.text_input("Enter a political statement:")
model = st.selectbox("Model", ["LSTM", "BERT (text)", "BERT (meta)", "LLM (Qwen)"])
if st.button("Predict"):
    response = requests.post("http://localhost:8000/predict", json={"statement": statement, "model": model})
    pred = response.json()
    st.write(f"Label: {pred['label']} (confidence: {pred['confidence']:.2f}, latency: {pred['latency_ms']:.1f}ms)")

# Results tab
if st.sidebar.button("Show Results"):
    response = requests.get("http://localhost:8000/results")
    results = response.json()
    df = pd.DataFrame(results["models"])
    st.table(df)
```

---

## 7. Development Workflow

### Git Branching
1. **Main branch:** `main` (production, stable)
2. **Feature branches:** `feat/setup`, `feat/lstm`, `feat/bert`, `feat/llm`, `feat/app`, `feat/analysis`
3. **PR workflow:**
   - Push to feature branch
   - Open PR into `main`
   - Request ≥1 approval from another team member
   - Merge after approval

### Commit Message Format
```
[Type] Short description

Detailed explanation (if needed).

Fixes: #<issue-number>
```

**Examples:**
```
[Setup] Initialize repo structure and gitignore
[Data] Add LIAR dataset loader and test subset
[LSTM] Add BiLSTM baseline, train script, results
[BERT] Fine-tune BERT text-only on LIAR
[LLM] Add Qwen 7B inference and parser
[App] Add FastAPI backend and Streamlit UI
[Analysis] Add cost-vs-accuracy comparison table
[Docs] Add README and architecture notes
```

### Code Review Checklist (Every PR)
- [ ] Code is clean and readable
- [ ] No hardcoded secrets (API keys, tokens, file paths)
- [ ] Input validation present
- [ ] Error messages are user-friendly (no stack traces in API)
- [ ] Security: no SQL injection, no arbitrary file access, no unvalidated user input
- [ ] Tests pass (unit tests for critical functions)
- [ ] `.gitignore` excludes checkpoints, `.env`, cache
- [ ] Docstrings on public functions

---

## 8. Security & Compliance (Chalo Policy)

### Required
- ✅ **Never commit secrets:** .env, API keys, tokens, model weights
- ✅ **Verify all packages:** Check `transformers`, `torch`, `datasets`, `bitsandbytes`, `fastapi`, `streamlit` against Chalo's approved software list. If not approved, raise an IT/Security request.
- ✅ **HuggingFace token:** Use `HF_TOKEN` environment variable only, never hardcode
- ✅ **Input validation:** Max length checks, label whitelisting, type validation
- ✅ **Error responses:** No stack traces in the API; return user-friendly JSON errors
- ✅ **Data privacy:** Use only public LIAR dataset; no company/customer data in demo
- ✅ **Logging:** Do not log full user statements; log only label predictions and latency
- ✅ **LLM license:** Verify Qwen/Mistral model licenses are compatible with the use case

### Model Licensing
- **LIAR dataset:** Public, cited as [Wang et al. 2017](https://arxiv.org/abs/1705.00648)
- **BERT:** Apache 2.0 (HuggingFace)
- **Qwen2.5-7B:** Qwen License Agreement (permissive for research/demos)
- **Mistral-7B:** Apache 2.0
- **Document all licenses in `docs/references.md`**

---

## 9. Testing

### Unit Tests (Minimum)
```python
# tests/test_common.py
def test_load_liar():
    train, val, test = load_liar()
    assert len(train) > 0 and len(test) > 0

def test_evaluate():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]
    metrics = evaluate(y_true, y_pred)
    assert metrics["accuracy"] == 1.0

# tests/test_parser.py
def test_parse_valid():
    assert parse_output("true") == "true"
    assert parse_output("PANTS-FIRE!!!") == "pants-fire"

def test_parse_invalid():
    with pytest.raises(ValueError):
        parse_output("unknown label")

# app/test_api.py
def test_predict_endpoint():
    response = client.post("/predict", json={"statement": "The earth is flat.", "model": "bert"})
    assert response.status_code == 200
    assert "label" in response.json()

def test_validation_too_long():
    response = client.post("/predict", json={"statement": "x" * 2000, "model": "bert"})
    assert response.status_code == 422
```

**Run tests before every merge:**
```bash
pytest tests/ -v
```

---

## 10. Expected Results (Rough Benchmarks)

| Model | Accuracy | Macro-F1 | Params | Train Time | Latency | Cost/1k |
|---|---|---|---|---|---|---|
| **LSTM** | ~25% | ~20% | 5M | 2 min | 45 ms | $0.01 |
| **BERT (text)** | ~27% | ~22% | 110M | 5 min | 50 ms | $0.08 |
| **BERT (meta)** | ~30% | ~25% | 110M | 6 min | 52 ms | $0.08 |
| **LLM (Qwen 7B)** | ~26% | ~21% | 7B | 0 min (prebuilt) | 500 ms | $2.00 |

**Caveats:**
- LIAR is a hard dataset with label noise. Published benchmarks are ~28% for text-only.
- LLM results depend heavily on the prompt; this is a first attempt.
- Cost assumes T4 at $0.35/hr and cloud LLM API (adjust to your rates).
- Subset size (300 samples) means confidence intervals are wide.

---

## 11. Known Risks & Fallbacks

| Risk | Mitigation |
|---|---|
| **GPU not available** | Use Colab (free, needs Google account) or DistilBERT (66M params, 2× faster) |
| **BERT training too slow** | Reduce `max_len` to 64, skip metadata fusion, use DistilBERT |
| **LLM inference fails** | 4-bit quantization reduces VRAM to ~6GB; fallback: 1.5B model or fewer samples (report this) |
| **LLM output won't parse** | Retry once; if still invalid, mark it and count. Report invalid rate. |
| **Merge conflicts** | Small, isolated commits; PRs for every major change; discuss before major refactors |
| **Time runs out** | Cut order: (1) few-shot LLM, (2) RoBERTa, (3) separate metadata fusion, (4) fancy charts. Keep baselines and core comparison. |

---

## 12. Minute-by-Minute Checklist

### Min 0–5: Kickoff
- [ ] All 4 people know their role (A, B, C, D)
- [ ] GPU verified (Colab/Kaggle/approved)
- [ ] Agree on LLM model (Qwen or Mistral)
- [ ] Agree on label order and file formats (this plan)

### Min 5–10: Setup
- [ ] Clone/pull repo; create feature branches
- [ ] Person A: Push `common/` with data loader + evaluate() + test_indices
- [ ] Person D: Scaffold `app/backend.py` stub with dummy model
- [ ] Others: Pull `common/`, test import

### Min 10–25: Baselines
- [ ] A: LSTM training in progress (should finish by min 12)
- [ ] B: BERT text-only training (starts min 10, finishes ~min 15)
- [ ] C: LLM parser tested on 20 samples
- [ ] D: Streamlit UI skeleton wired to stub API

### Min 25–45: Full Runs
- [ ] B: BERT + metadata training (another 5 min)
- [ ] C: LLM full 300-sample run (~10 min)
- [ ] A: Error analysis, confusion matrix
- [ ] D: Plug real model paths into API, test end-to-end

### Min 45–55: Synthesis
- [ ] All results JSON + CSV files in `results/`
- [ ] D: Build comparison table (read results/*.json, format nicely)
- [ ] A + D: Write cost analysis, interpret confusion matrices
- [ ] Everyone: Review PRs, approve

### Min 55–60: Finalization
- [ ] All branches merged to `main`
- [ ] D: README and ARCHITECTURE done
- [ ] D: Run the Streamlit demo end-to-end (dry run)
- [ ] Tag final commit, demo ready

---

## 13. References

### Papers
1. **LIAR Dataset:** Wang, William Yang, et al. "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection." *Proceedings of ACL 2017.* https://arxiv.org/abs/1705.00648
2. **BERT:** Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL 2019.* https://arxiv.org/abs/1810.04805

### Libraries & Docs
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) (4-bit quantization)

---

## 14. Contact & Notes

**Team Lead:** Prachi Dhakad  
**Repository:** https://github.com/arunchowdary-1/Fake-News-Detection-Fine-tuned-Transformers-vs-Zero-shot-LLMs

**Questions during the hour?** Check this document first. If blocked, ask in chat.

---

**Last updated:** 2026-11-21  
**Status:** Ready to execute
