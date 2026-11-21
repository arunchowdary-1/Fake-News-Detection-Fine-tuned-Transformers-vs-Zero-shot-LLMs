"""
tests/test_bert_text.py  --  unit tests for Issue #5 (BERT text-only fine-tuning)
==================================================================================

PURPOSE
    Check the parts of finetune/train_text_only.py that can silently break the
    comparison, WITHOUT downloading BERT or LIAR (so they run in seconds, anywhere).

RUN WITH:  pytest tests/test_bert_text.py -v
"""

from common.config import LABEL_TO_IDX
from finetune.train_text_only import to_project_labels


class _FakeClassLabel:
    """Imitates the Hugging Face label column, which uses HF's own label order."""
    names = ["false", "half-true", "mostly-true", "true", "barely-true", "pants-fire"]

    def int2str(self, i):
        return self.names[i]


class _FakeSplit:
    """Tiny stand-in for a dataset split so no download is needed."""
    features = {"label": _FakeClassLabel()}

    def __getitem__(self, key):
        return [0, 1, 2, 3, 4, 5]   # one row of every HF label id


def test_hf_labels_are_remapped_to_project_order():
    # WHY: the most important check. If HF ids were used directly, BERT's metrics and
    # confusion matrix would be wrong and not comparable with the LSTM and LLM.
    got = to_project_labels(_FakeSplit())
    expected = [LABEL_TO_IDX[n] for n in _FakeClassLabel.names]
    assert got == expected == [1, 3, 4, 5, 2, 0]
