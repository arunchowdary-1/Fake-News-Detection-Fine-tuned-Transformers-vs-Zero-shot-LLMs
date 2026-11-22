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

DEFAULT_MODEL_NAME = "bert-base-uncased"


def build_texts(split, mode: str, sep_token: str = "[SEP]") -> List[str]:
    """
    Turn a LIAR split into the list of input strings for the chosen mode.

    The shared metadata builder joins prefix and statement with a literal "[SEP]".
    We swap that for the tokenizer's own separator so RoBERTa ("</s>") also works.
    """
    if mode == "text":
        return [(s or "").strip() for s in split["statement"]]
    if mode == "meta":
        return [build_metadata_text(row).replace(" [SEP] ", f" {sep_token} ") for row in split]
    raise ValueError(f"mode must be 'text' or 'meta', got {mode!r}")


class EncodedDataset(TorchDataset):
    """Pre-tokenised texts + project-order labels, ready for a DataLoader."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.enc = tokenizer(texts, truncation=True, max_length=max_len)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> Dict:
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def make_collate(tokenizer):
    """Pad each batch to its longest sequence (faster than padding everything to max_len)."""

    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        labels = torch.tensor([b.pop("labels") for b in batch], dtype=torch.long)
        out = tokenizer.pad(batch, return_tensors="pt")
        out["labels"] = labels
        return out

    return collate


def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    """6-class classifier head on top of the pretrained encoder, in OUR label order."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        id2label=dict(IDX_TO_LABEL),
        label2id=dict(LABEL_TO_IDX),
    )
    return model, tokenizer


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
