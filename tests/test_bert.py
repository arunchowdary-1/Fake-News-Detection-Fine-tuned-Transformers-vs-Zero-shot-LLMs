"""Unit tests for finetune/bert_model.py input building (no model download needed)."""

import pytest
from datasets import Dataset

from finetune.bert_model import build_texts


def _split():
    return Dataset.from_dict({
        "statement": ["  Taxes went up.  ", "We cut spending."],
        "speaker": ["jane-doe", None],
        "party_affiliation": ["democrat", "republican"],
        "subject": ["taxes", "budget"],
        "barely_true_counts": [1, 0],
        "false_counts": [2, 0],
        "half_true_counts": [3, 0],
        "mostly_true_counts": [4, 0],
        "pants_on_fire_counts": [5, 0],
    })


def test_text_mode_is_just_the_stripped_statement():
    assert build_texts(_split(), "text") == ["Taxes went up.", "We cut spending."]


def test_meta_mode_has_metadata_prefix_and_tokenizer_separator():
    out = build_texts(_split(), "meta", sep_token="</s>")
    assert out[0] == ("speaker: jane-doe | party: democrat | subject: taxes | "
                      "history: 1,2,3,4,5 </s> Taxes went up.")
    assert "speaker: UNKNOWN" in out[1]       # missing speaker handled
    assert "[SEP]" not in out[0]


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        build_texts(_split(), "nope")
