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

# Official LIAR download (Wang, ACL 2017).
LIAR_ZIP_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"

# The official .tsv files have no header row; this is their fixed column order, and the
# names match the Hugging Face copy so downstream code works on either source.
COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state_info",
    "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context",
]
# The five credit-history count columns (end with _counts); these must be numbers.
COUNT_COLUMNS = [c for c in COLUMNS if c.endswith("_counts")]


def parse_tsv(text: str) -> Dict[str, List]:
    """Parse one LIAR .tsv file into {column: list}. Labels become HF integer ids."""
    cols: Dict[str, List] = {c: [] for c in COLUMNS}
    # Each line is one statement.
    for line in text.splitlines():
        # Skip blank lines.
        if not line.strip():
            continue
        # Columns are separated by TAB characters.
        parts = line.split("\t")
        parts = (parts + [""] * len(COLUMNS))[: len(COLUMNS)]   # tolerate short rows
        # Pair each column name with its value for this row.
        for name, value in zip(COLUMNS, parts):
            # Labels use HF integer ids (not project ids) on purpose: to_project_labels
            # converts both sources the same way, via the label name.
            if name == "label":
                # Position in the HF name list = HF id. to_project_labels later maps it to our order.
                value = HF_LIAR_LABEL_NAMES.index(value.strip().lower())
            # Counts come as text; make them floats, 0.0 if empty.
            elif name in COUNT_COLUMNS:
                value = float(value) if value.strip() else 0.0
            # Ids look like '2635.json'.
            elif name == "id":
                # Raw ids carry a ".json" suffix; drop it to keep just the statement id.
                value = value.replace(".json", "")
            # Store the cleaned value.
            cols[name].append(value)
    # Dict of columns, ready for datasets.Dataset.from_dict.
    return cols


# The three splits: train / validation / test.
TSV_FILES = ("train.tsv", "valid.tsv", "test.tsv")


def _read_tsv_texts(cache_dir: str) -> list:
    """Return the text of train/valid/test .tsv.

    Prefers the original files committed in the repo (LIAR_TSV_DIR, default `liar_data/`);
    only downloads the official zip if they are missing.
    """
    # Local files first so no network is needed and the exact committed data is used.
    tsv_dir = os.environ.get("LIAR_TSV_DIR", "liar_data")
    # Use the committed files if all three exist: no internet needed, exact same data for everyone.
    if all(os.path.exists(os.path.join(tsv_dir, f)) for f in TSV_FILES):
        print(f"Reading LIAR from {tsv_dir}/")
        # Read each file as UTF-8 text.
        return [open(os.path.join(tsv_dir, f), encoding="utf-8").read() for f in TSV_FILES]

    # Otherwise prepare a cache folder for the zip.
    os.makedirs(cache_dir, exist_ok=True)
    # Where the zip is saved.
    zpath = os.path.join(cache_dir, "liar_dataset.zip")
    # Download only once.
    if not os.path.exists(zpath):
        print(f"Downloading LIAR from {LIAR_ZIP_URL} ...")
        # Fetch the official zip.
        urllib.request.urlretrieve(LIAR_ZIP_URL, zpath)
    # Read the three tsv files out of the zip without extracting to disk.
    with zipfile.ZipFile(zpath) as z:
        return [io.TextIOWrapper(io.BytesIO(z.read(f)), encoding="utf-8").read() for f in TSV_FILES]


def load_liar_from_zip(cache_dir: str = "data/"):
    """Return (train, val, test) as `datasets.Dataset` objects."""
    # Imported here so the module can be imported (e.g. for parse_tsv) without `datasets`.
    from datasets import ClassLabel, Dataset, Features, Value

    # Explicit features keep the label column a ClassLabel, which to_project_labels needs
    # (it calls int2str), and make counts float32 like the HF copy.
    feats = Features({
        # Label stays a ClassLabel because to_project_labels calls int2str on it; counts are float32 like the HF copy.
        c: (ClassLabel(names=HF_LIAR_LABEL_NAMES) if c == "label"
            else Value("float32") if c in COUNT_COLUMNS else Value("string"))
        for c in COLUMNS
    })
    # Build one Dataset per split from the parsed columns.
    return tuple(Dataset.from_dict(parse_tsv(t), features=feats) for t in _read_tsv_texts(cache_dir))
