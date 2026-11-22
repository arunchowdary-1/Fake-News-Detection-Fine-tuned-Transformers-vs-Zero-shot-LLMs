"""
Data loader for the LIAR dataset with metadata handling and test subset generation.
"""

import os
from typing import Tuple, List, Dict, Optional
import numpy as np
from datasets import load_dataset, Dataset
from common.config import LABEL_TO_IDX, TEST_SUBSET_SIZE, RANDOM_SEED, DATA_DIR

# Class order of the Hugging Face "liar" ClassLabel feature (differs from LABEL_TO_IDX)
HF_LIAR_LABEL_NAMES = ["false", "half-true", "mostly-true", "true", "barely-true", "pants-fire"]


def _label_name(label) -> Optional[str]:
    """Normalize a label (string or Hugging Face integer id) to a lowercase label name."""
    if isinstance(label, bool):
        return None
    if isinstance(label, (int, np.integer)):
        if 0 <= label < len(HF_LIAR_LABEL_NAMES):
            return HF_LIAR_LABEL_NAMES[label]
        return None
    if isinstance(label, str):
        return label.lower()
    return None


def load_liar() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the LIAR dataset from Hugging Face.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    try:
        print("Loading LIAR dataset from Hugging Face...")
        # Newer datasets/huggingface_hub need the full repo id and trust_remote_code
        # (liar is a script dataset); the pinned older versions accept the plain call.
        # datasets >= 4 removed script datasets entirely, so pin datasets<4 there.
        attempts = [
            lambda: load_dataset("liar"),
            lambda: load_dataset("liar", trust_remote_code=True),
            lambda: load_dataset("ucsbai/liar", trust_remote_code=True),
        ]
        last_err = None
        for attempt in attempts:
            try:
                ds = attempt()
                break
            except Exception as err:
                last_err = err
        else:
            raise last_err
        return ds["train"], ds["validation"], ds["test"]
    except Exception as e:
        raise RuntimeError(f"Failed to load LIAR dataset: {e}")


def validate_statement(statement: str, max_length: int = 1000) -> bool:
    """
    Validate a political statement.

    Args:
        statement: Input text
        max_length: Maximum allowed length

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(statement, str):
        return False
    if len(statement.strip()) == 0 or len(statement) > max_length:
        return False
    return True


def validate_label(label) -> bool:
    """
    Validate that a label is in the allowed set.

    Args:
        label: Label string, or Hugging Face integer label id

    Returns:
        True if valid, False otherwise
    """
    return _label_name(label) in LABEL_TO_IDX


def label_to_idx(label) -> int:
    """
    Convert a label to the project index (see LABEL_TO_IDX).

    Args:
        label: Label string (case-insensitive) or Hugging Face integer label id

    Returns:
        Index 0-5

    Raises:
        ValueError if label is invalid
    """
    label_lower = _label_name(label)
    if label_lower not in LABEL_TO_IDX:
        raise ValueError(f"Invalid label '{label}'. Must be one of: {list(LABEL_TO_IDX.keys())}")
    return LABEL_TO_IDX[label_lower]


# Speaker credit-history columns in the Hugging Face "liar" dataset, in output order.
CREDIT_HISTORY_COLUMNS = [
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
]


def _clean_field(value, default: str = "UNKNOWN") -> str:
    """Return a stripped string, or `default` if the value is missing/None/blank."""
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _credit_history(row: Dict) -> List[int]:
    """
    Extract the 5 credit-history counts (barely-true, false, half-true, mostly-true,
    pants-on-fire) from a LIAR row. Supports the real dataset columns, and a
    pre-combined `credit_history` list as a fallback. Missing values become 0.
    """
    if any(col in row for col in CREDIT_HISTORY_COLUMNS):
        raw = [row.get(col) for col in CREDIT_HISTORY_COLUMNS]
    else:
        raw = list(row.get("credit_history") or [])
        raw = (raw + [0] * len(CREDIT_HISTORY_COLUMNS))[: len(CREDIT_HISTORY_COLUMNS)]
    return [int(c) if c is not None else 0 for c in raw]


def build_metadata_text(row: Dict) -> str:
    """
    Build metadata-prefixed text from a LIAR row.
    Format: speaker: X | party: Y | subject: Z | history: A,B,C,D,E | statement

    Args:
        row: LIAR dataset row (dict)

    Returns:
        Prefixed text string
    """
    speaker = _clean_field(row.get("speaker"))
    party = _clean_field(row.get("party_affiliation"))
    subject = _clean_field(row.get("subject"))

    history_str = ",".join(str(c) for c in _credit_history(row))

    statement = (row.get("statement") or "").strip()

    # Combine metadata with statement
    metadata_prefix = f"speaker: {speaker} | party: {party} | subject: {subject} | history: {history_str}"
    return f"{metadata_prefix} [SEP] {statement}"


def get_test_indices(test_dataset: Dataset, subset_size: int = TEST_SUBSET_SIZE) -> List[int]:
    """
    Generate a fixed stratified test subset for fair model comparison.
    Uses stratification by label to maintain class distribution.

    Args:
        test_dataset: Full test dataset
        subset_size: Number of samples to select

    Returns:
        List of indices for the test subset
    """
    if subset_size <= 0:
        raise ValueError(f"subset_size must be positive, got {subset_size}")

    labels = np.array([label_to_idx(label) for label in test_dataset["label"]])
    total = len(labels)
    if subset_size >= total:
        return list(range(total))

    rng = np.random.RandomState(RANDOM_SEED)  # local RNG: no global state mutation
    n_classes = len(LABEL_TO_IDX)
    class_members = [np.where(labels == c)[0] for c in range(n_classes)]

    # Proportional allocation (largest-remainder) so the subset mirrors the test label distribution.
    exact = np.array([len(m) * subset_size / total for m in class_members])
    alloc = np.floor(exact).astype(int)
    leftover = subset_size - int(alloc.sum())
    for c in np.argsort(-(exact - alloc), kind="stable")[:leftover]:
        alloc[c] += 1

    indices: List[int] = []
    for members, n in zip(class_members, alloc):
        if n > 0:
            indices.extend(rng.choice(members, size=int(n), replace=False).tolist())
    indices = sorted(indices)

    print(f"Generated stratified test subset: {len(indices)} samples out of {total}")
    print("Class distribution in subset (subset/full):")
    for c in range(n_classes):
        print(f"  Class {c}: {int(alloc[c])}/{len(class_members[c])}")

    return indices


def save_test_indices(indices: List[int], filepath: str = "common/test_indices.txt") -> None:
    """
    Save test indices to file for reproducibility.

    Args:
        indices: List of indices
        filepath: Where to save
    """
    with open(filepath, "w") as f:
        f.write(",".join(str(i) for i in indices))
    print(f"Saved test indices to {filepath}")


def load_test_indices(filepath: str = "common/test_indices.txt") -> List[int]:
    """
    Load test indices from file.

    Args:
        filepath: Where to load from

    Returns:
        List of indices
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test indices file not found: {filepath}")

    with open(filepath, "r") as f:
        indices = [int(x) for x in f.read().strip().split(",")]

    print(f"Loaded {len(indices)} test indices from {filepath}")
    return indices


def get_test_subset(test_dataset: Dataset, indices: Optional[List[int]] = None) -> Dataset:
    """
    Get a subset of the test dataset using fixed indices.

    Args:
        test_dataset: Full test dataset
        indices: Indices to select. If None, generate new ones.

    Returns:
        Subset dataset
    """
    if indices is None:
        indices = get_test_indices(test_dataset)
        save_test_indices(indices)

    return test_dataset.select(indices)


if __name__ == "__main__":
    # Test the loader
    print("Testing LIAR data loader...")
    train, val, test = load_liar()
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Test metadata building
    sample_row = train[0]
    print(f"\nSample row:")
    print(f"  Statement: {sample_row['statement'][:100]}...")
    print(f"  Label: {sample_row['label']}")
    print(f"  Metadata text: {build_metadata_text(sample_row)[:150]}...")

    # Test test subset generation
    indices = get_test_indices(test)
    save_test_indices(indices)
    subset = get_test_subset(test, indices)
    print(f"\nTest subset size: {len(subset)}")
