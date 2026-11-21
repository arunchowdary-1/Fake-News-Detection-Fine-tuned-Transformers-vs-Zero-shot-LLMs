"""Tests for the LLM evaluation bookkeeping (no GPU, no dataset download)."""
from llm.evaluate_llm import INVALID_IDX, cost_per_1k, run_eval, score

# project label ids: true=5, false=1, half-true=3
ROWS = [
    {"statement": "a", "y_true": 5},
    {"statement": "b", "y_true": 1},
    {"statement": "c", "y_true": 3},
]


def test_run_eval_maps_labels_and_marks_invalid():
    answers = iter(["true", "false", "garbage"])  # no retry: one call per row
    preds, lat = run_eval(ROWS, lambda m: next(answers))
    assert preds == [5, 1, INVALID_IDX]
    assert len(lat) == 3 and all(x >= 0 for x in lat)


def test_score_counts_invalid_as_wrong_without_extra_class():
    m = score([5, 1, 3], [5, 1, INVALID_IDX])
    assert abs(m["accuracy"] - 2 / 3) < 1e-9
    assert m["n_samples"] == 3
    # shared schema fields other models also report
    assert len(m["confusion_matrix"]) == 6


def test_score_all_invalid_is_zero():
    m = score([0, 1], [INVALID_IDX, INVALID_IDX])
    assert m["accuracy"] == 0.0 and m["macro_f1"] == 0.0


def test_cost_per_1k_matches_other_models_formula():
    assert abs(cost_per_1k(1000.0) - (1000 / 1000 / 3600 * 1000 * 0.35 * 1.10)) < 1e-12
