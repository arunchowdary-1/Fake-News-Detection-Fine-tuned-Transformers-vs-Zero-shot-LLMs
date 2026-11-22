"""
finetune/liar_fallback.py
=========================
Loads LIAR straight from the official release zip when `load_dataset("liar")` fails
(it is a script-based dataset, which recent `datasets` versions no longer support).

Produces the SAME splits (train 10,269 / val 1,284 / test 1,283), the same column
names and the same integer label ids as the Hugging Face copy, so the rest of the
pipeline (to_project_labels, build_metadata_text, shared test indices) is unchanged.
"""

import io
import os
import urllib.request
import zipfile
from typing import Dict, List

from common.data_loader import HF_LIAR_LABEL_NAMES

LIAR_ZIP_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"

COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state_info",
    "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context",
]
COUNT_COLUMNS = [c for c in COLUMNS if c.endswith("_counts")]


def parse_tsv(text: str) -> Dict[str, List]:
    """Parse one LIAR .tsv file into {column: list}. Labels become HF integer ids."""
    cols: Dict[str, List] = {c: [] for c in COLUMNS}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        parts = (parts + [""] * len(COLUMNS))[: len(COLUMNS)]   # tolerate short rows
        for name, value in zip(COLUMNS, parts):
            if name == "label":
                value = HF_LIAR_LABEL_NAMES.index(value.strip().lower())
            elif name in COUNT_COLUMNS:
                value = float(value) if value.strip() else 0.0
            elif name == "id":
                value = value.replace(".json", "")
            cols[name].append(value)
    return cols


TSV_FILES = ("train.tsv", "valid.tsv", "test.tsv")


def _read_tsv_texts(cache_dir: str) -> list:
    """Return the text of train/valid/test .tsv.

    Prefers the original files committed in the repo (LIAR_TSV_DIR, default `liar_data/`);
    only downloads the official zip if they are missing.
    """
    tsv_dir = os.environ.get("LIAR_TSV_DIR", "liar_data")
    if all(os.path.exists(os.path.join(tsv_dir, f)) for f in TSV_FILES):
        print(f"Reading LIAR from {tsv_dir}/")
        return [open(os.path.join(tsv_dir, f), encoding="utf-8").read() for f in TSV_FILES]

    os.makedirs(cache_dir, exist_ok=True)
    zpath = os.path.join(cache_dir, "liar_dataset.zip")
    if not os.path.exists(zpath):
        print(f"Downloading LIAR from {LIAR_ZIP_URL} ...")
        urllib.request.urlretrieve(LIAR_ZIP_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        return [io.TextIOWrapper(io.BytesIO(z.read(f)), encoding="utf-8").read() for f in TSV_FILES]


def load_liar_from_zip(cache_dir: str = "data/"):
    """Return (train, val, test) as `datasets.Dataset` objects."""
    from datasets import ClassLabel, Dataset, Features, Value

    feats = Features({
        c: (ClassLabel(names=HF_LIAR_LABEL_NAMES) if c == "label"
            else Value("float32") if c in COUNT_COLUMNS else Value("string"))
        for c in COLUMNS
    })
    return tuple(Dataset.from_dict(parse_tsv(t), features=feats) for t in _read_tsv_texts(cache_dir))
