"""Tests for the LLM evaluation bookkeeping (no GPU, no dataset download)."""

# =====================================================================================
# FILE GUIDE (plain-English summary, added for understanding and viva preparation)
#
# WHAT THESE TESTS CHECK
#   evaluate_llm.py does the "bookkeeping" of the LLM experiment: turning answers into
#   numbers, scoring them, and computing cost. These tests check that bookkeeping with tiny
#   made-up data, so they run in a blink with no GPU and no dataset download.
#
# THE BIG IDEA BEHIND THEM
#   An invalid answer (-1) must count as WRONG, but must not become a made-up prediction and
#   must not appear as a seventh class. Three of the four tests protect exactly that rule.
#
# VIVA QUESTIONS FOR THIS FILE (with short answers)
#   Q: Why is the cost formula tested?
#   A: So the LLM's cost and BERT's cost always use the same formula and stay comparable.
#   Q: Why use tiny fake data instead of the real dataset?
#   A: Tests must be fast and repeatable. The real run needs a GPU and takes a long time.
#   Q: What does an accuracy of 2/3 mean here?
#   A: 2 of the 3 statements were right, and the invalid one counts as a miss.
# =====================================================================================

# The pieces under test: the -1 marker, the cost formula, the main loop, and the scoring function.
from llm.evaluate_llm import INVALID_IDX, cost_per_1k, run_eval, score

# project label ids: true=5, false=1, half-true=3
# Three tiny fake rows: the text does not matter here, only the true label number.
ROWS = [
    {"statement": "a", "y_true": 5},
    {"statement": "b", "y_true": 1},
    {"statement": "c", "y_true": 3},
]


def test_run_eval_maps_labels_and_marks_invalid():
    """Valid labels map to project ids; an unparseable answer becomes -1, not a guess."""
    # The fake model answers "true" (id 5), "false" (id 1), then gibberish for the third row.
    answers = iter(["true", "false", "garbage"])  # no retry: one call per row
    preds, lat = run_eval(ROWS, lambda m: next(answers))
    # Expect [5, 1, -1]: the gibberish became the -1 marker, not a guessed label.
    assert preds == [5, 1, INVALID_IDX]
    # One timing per row, and a time can never be negative.
    assert len(lat) == 3 and all(x >= 0 for x in lat)


def test_score_counts_invalid_as_wrong_without_extra_class():
    """An invalid output lowers accuracy but does not add a 7th class to the metrics."""
    # Real labels [5, 1, 3]; the model got the first two right and the third was invalid.
    m = score([5, 1, 3], [5, 1, INVALID_IDX])
    # 2 right out of 3 = 0.666...; the tiny tolerance avoids float rounding issues.
    assert abs(m["accuracy"] - 2 / 3) < 1e-9
    assert m["n_samples"] == 3
    # shared schema fields other models also report
    # The confusion matrix must stay 6 rows, never 7 (the -1 is not a class).
    assert len(m["confusion_matrix"]) == 6


def test_score_all_invalid_is_zero():
    """A run where every output is invalid scores 0, without errors from empty predictions."""
    # Worst case: the model never gave a usable answer. Scores must be 0.0, not a crash.
    m = score([0, 1], [INVALID_IDX, INVALID_IDX])
    assert m["accuracy"] == 0.0 and m["macro_f1"] == 0.0


def test_cost_per_1k_matches_other_models_formula():
    """Guards the cost formula (T4 $0.35/h + 10%, batch 1) so LLM and BERT costs stay comparable."""
    # 1000 ms per prediction = 1 second each. 1000 predictions = 1000 s = 0.278 hours,
    # then times the price per hour and the 10% overhead. The test writes the maths out
    # long-hand and checks the function gives the same answer.
    assert abs(cost_per_1k(1000.0) - (1000 / 1000 / 3600 * 1000 * 0.35 * 1.10)) < 1e-12

# EXTRA VIVA QUESTIONS (whole file)
#   Q: What does "score" return for macro-F1 when nothing is correct?
#   A: 0.0 (the tests above check this edge case on purpose).
#   Q: What is an "edge case" and why test it?
#   A: An unusual input, such as "every answer invalid". Bugs often hide there.
