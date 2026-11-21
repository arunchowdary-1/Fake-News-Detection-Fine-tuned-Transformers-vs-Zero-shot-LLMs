"""
tests/test_lstm.py  --  unit tests for Issue #4 (LSTM baseline)
================================================================

PURPOSE
    plan.md's code-review checklist says every PR needs passing tests for
    critical functions. These tests check the small, easy-to-break parts of the
    LSTM code. They need NO internet and NO dataset download, so they run in
    seconds on any machine (and in CI).

RUN WITH:  pytest tests/test_lstm.py -v
"""

import torch

from common.config import LABEL_TO_IDX
from baselines.lstm_model import BiLSTMClassifier, Vocab, pad_batch, tokenize
from baselines.train import to_project_labels


def test_tokenize():
    # WHY: if tokenizing silently drops "$" or "%", we'd lose important
    # signal in political claims. Also checks lowercasing and None-safety.
    assert tokenize("Says Obama spent $3 TRILLION, 50%!") == [
        "says", "obama", "spent", "$", "3", "trillion", "50", "%"
    ]
    assert tokenize(None) == []   # bad input must not crash training


def test_vocab_build_and_encode():
    # WHY: checks the three vocabulary rules that affect results.
    # "tax" and "cuts" appear twice (kept); "work"/"fail" once (dropped by min_freq=2).
    vocab = Vocab.build(["tax cuts work", "tax cuts fail"], min_freq=2)
    assert "tax" in vocab.stoi and "work" not in vocab.stoi
    # Unknown word "policy" must become <unk>, not raise KeyError.
    ids = vocab.encode("tax policy", max_len=10)
    assert ids == [vocab.stoi["tax"], vocab.unk_idx]
    # Long statements are truncated to max_len.
    assert len(vocab.encode("tax " * 50, max_len=5)) == 5
    # An empty statement must still give 1 token (LSTM can't take length 0).
    assert vocab.encode("", max_len=5) == [vocab.unk_idx]


def test_pad_batch():
    # WHY: wrong padding would feed garbage to the LSTM. Row 2 is shorter, so it
    # must be filled with pad id 0, and the true lengths must be reported.
    ids, lengths = pad_batch([[2, 3, 4], [5]], pad_idx=0)
    assert ids.tolist() == [[2, 3, 4], [5, 0, 0]]
    assert lengths.tolist() == [3, 1]


def test_model_forward_shape():
    # WHY: guards the layer sizes (the hidden_dim*4 concat). A shape mistake
    # would crash on the very first training batch. Expect [batch=2, classes=6].
    model = BiLSTMClassifier(vocab_size=50, num_classes=6, embed_dim=8, hidden_dim=4)
    ids, lengths = pad_batch([[2, 3, 4], [5]], pad_idx=0)
    assert model(ids, lengths).shape == (2, 6)


class _FakeClassLabel:
    """Imitates the Hugging Face label column, using HF's label order."""
    names = ["false", "half-true", "mostly-true", "true", "barely-true", "pants-fire"]

    def int2str(self, i):
        return self.names[i]


class _FakeSplit:
    """A tiny stand-in for a dataset split so we can test without downloading LIAR."""
    features = {"label": _FakeClassLabel()}

    def __getitem__(self, key):
        # HF ids 0..5, one of each label.
        return [0, 1, 2, 3, 4, 5]


def test_hf_labels_are_remapped_to_project_order():
    # WHY: this is the MOST important test. HF numbers labels differently from
    # the project's agreed order. If the remap breaks, every metric and confusion
    # matrix we report would be silently wrong and not comparable to BERT/LLM.
    got = to_project_labels(_FakeSplit())
    expected = [LABEL_TO_IDX[n] for n in _FakeClassLabel.names]
    assert got == expected == [1, 3, 4, 5, 2, 0]
