"""
Data loader for the LIAR dataset with metadata handling and test subset generation.
"""

import os
from typing import Tuple, List, Dict, Optional
import numpy as np
from datasets import load_dataset, Dataset
from common.config import LABEL_TO_IDX, TEST_SUBSET_SIZE, RANDOM_SEED, DATA_DIR


def load_liar() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the LIAR dataset from Hugging Face.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    try:
        print("Loading LIAR dataset from Hugging Face...")
        ds = load_dataset("liar")
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


def validate_label(label: str) -> bool:
    """
    Validate that a label is in the allowed set.

    Args:
        label: Label string

    Returns:
        True if valid, False otherwise
    """
    return label.lower() in LABEL_TO_IDX


def label_to_idx(label: str) -> int:
    """
    Convert label string to index.

    Args:
        label: Label string (case-insensitive)

    Returns:
        Index 0-5

    Raises:
        ValueError if label is invalid
    """
    label_lower = label.lower()
    if label_lower not in LABEL_TO_IDX:
        raise ValueError(f"Invalid label '{label}'. Must be one of: {list(LABEL_TO_IDX.keys())}")
    return LABEL_TO_IDX[label_lower]


def build_metadata_text(row: Dict) -> str:
    """
    Build metadata-prefixed text from a LIAR row.
    Format: speaker: X | party: Y | subject: Z | history: A,B,C,D,E | statement

    Args:
        row: LIAR dataset row (dict)

    Returns:
        Prefixed text string
    """
    speaker = row.get("speaker", "UNKNOWN").strip()
    party = row.get("party_affiliation", "UNKNOWN").strip()
    subject = row.get("subject", "UNKNOWN").strip()

    # Credit history: (barely-true, false, half-true, mostly-true, pants-fire)
    history_counts = row.get("credit_history", [0, 0, 0, 0, 0])
    history_str = ",".join(str(c) for c in history_counts)

    statement = row.get("statement", "").strip()

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
    np.random.seed(RANDOM_SEED)

    # Get labels
    labels = np.array([label_to_idx(row["label"]) for row in test_dataset])
    n_classes = len(np.unique(labels))

    # Stratified sampling
    indices = []
    samples_per_class = subset_size // n_classes

    for class_idx in range(n_classes):
        class_indices = np.where(labels == class_idx)[0]
        selected = np.random.choice(
            class_indices,
            size=min(samples_per_class, len(class_indices)),
            replace=False
        )
        indices.extend(selected.tolist())

    # If we don't have exactly subset_size, adjust
    if len(indices) < subset_size:
        all_indices = np.arange(len(test_dataset))
        remaining = subset_size - len(indices)
        additional = np.random.choice(all_indices, size=remaining, replace=False)
        indices.extend(additional.tolist())

    indices = sorted(list(set(indices))[:subset_size])  # Ensure unique and correct size

    print(f"Generated stratified test subset: {len(indices)} samples out of {len(test_dataset)}")
    print(f"Class distribution in subset:")
    for class_idx in range(n_classes):
        count = sum(1 for i in indices if labels[i] == class_idx)
        print(f"  Class {class_idx}: {count} samples")

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
