"""Unit tests for Issue #6 (BERT with metadata): input-text construction only, no model download."""

from finetune.train_with_metadata import build_texts

ROWS = [{
    "statement": "Taxes went up.",
    "speaker": "jane-doe",
    "party_affiliation": "democrat",
    "subject": "taxes",
    "barely_true_counts": 1, "false_counts": 2, "half_true_counts": 3,
    "mostly_true_counts": 4, "pants_on_fire_counts": 5,
}]


def test_metadata_prefix_with_history():
    text = build_texts(ROWS, include_history=True)[0]
    assert text == "speaker: jane-doe | party: democrat | subject: taxes | history: 1,2,3,4,5 [SEP] Taxes went up."


def test_no_history_drops_only_the_history_field():
    text = build_texts(ROWS, include_history=False)[0]
    assert "history" not in text
    assert text == "speaker: jane-doe | party: democrat | subject: taxes [SEP] Taxes went up."
