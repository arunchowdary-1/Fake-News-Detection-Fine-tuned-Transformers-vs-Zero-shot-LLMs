"""
finetune/bert_model.py
======================
Model + data helpers for fine-tuning a Hugging Face encoder (bert-base-uncased by
default; roberta-base works by changing --model-name) on LIAR.

Two input modes:
    text : the statement only
    meta : "speaker | party | subject | credit history" prefix + statement
           (built with the shared common.data_loader.build_metadata_text)
"""

from typing import Dict, List

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.config import NUM_CLASSES, IDX_TO_LABEL, LABEL_TO_IDX
from common.data_loader import build_metadata_text

# Which pretrained model to use by default. Change this (or pass --model-name) to try RoBERTa/DistilBERT.
DEFAULT_MODEL_NAME = "bert-base-uncased"


def build_texts(split, mode: str, sep_token: str = "[SEP]") -> List[str]:
    """
    Turn a LIAR split into the list of input strings for the chosen mode.

    The shared metadata builder joins prefix and statement with a literal "[SEP]".
    We swap that for the tokenizer's own separator so RoBERTa ("</s>") also works.
    """
    # Text mode is the control run: it shows how much the metadata prefix adds on top of
    # the statement alone. `or ""` guards against missing statements.
    if mode == "text":
        # 's or ""' avoids a crash if a statement is None; strip() removes stray spaces.
        return [(s or "").strip() for s in split["statement"]]
    # META mode: metadata prefix + statement (speaker, party, subject, credit history).
    if mode == "meta":
        # The match is on " [SEP] " with surrounding spaces, so only the separator that
        # build_metadata_text inserts is replaced, never text inside the statement itself.
        return [build_metadata_text(row).replace(" [SEP] ", f" {sep_token} ") for row in split]
    # Fail loudly on a typo in --mode instead of silently doing the wrong thing.
    raise ValueError(f"mode must be 'text' or 'meta', got {mode!r}")


class EncodedDataset(TorchDataset):
    """Pre-tokenised texts + project-order labels, ready for a DataLoader."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        # Tokenise once up front, without padding: padding is done per batch in the
        # collate function so short batches stay short.
        self.enc = tokenizer(texts, truncation=True, max_length=max_len)
        # Labels are already in OUR order (0=pants-fire ... 5=true).
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> Dict:
        # enc has input_ids / attention_mask / token_type_ids lists; take the i-th of each.
        item = {k: v[i] for k, v in self.enc.items()}
        # Attach the correct answer for this example.
        item["labels"] = self.labels[i]
        return item


def make_collate(tokenizer):
    """Pad each batch to its longest sequence (faster than padding everything to max_len)."""

    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Labels are split off first because tokenizer.pad only knows how to pad the
        # tokenizer's own fields (input_ids, attention_mask, ...).
        labels = torch.tensor([b.pop("labels") for b in batch], dtype=torch.long)
        # Pad every example to the LONGEST in this batch (faster than padding all to 128).
        out = tokenizer.pad(batch, return_tensors="pt")
        # Put the labels back so the training loop can use them.
        out["labels"] = labels
        return out

    # Hand back the inner function.
    return collate


def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    """6-class classifier head on top of the pretrained encoder, in OUR label order."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # id2label/label2id use the project label order so the head's output index i means
    # the same class here as in every other model and in the confusion matrices.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        # 6 outputs, one score (logit) per truth class.
        num_labels=NUM_CLASSES,
        # Store names with the model so output index i means the same class as everywhere else in the project.
        id2label=dict(IDX_TO_LABEL),
        label2id=dict(LABEL_TO_IDX),
    )
    return model, tokenizer


def count_params(model: torch.nn.Module) -> int:
    """Total parameter count (trainable or not), reported in the results JSON."""
    return sum(p.numel() for p in model.parameters())
