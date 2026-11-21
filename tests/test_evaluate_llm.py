"""Tests for the LLM evaluation bookkeeping (no GPU: the model is a fake)."""
import csv
import json

from llm.evaluate_llm import macro_f1, run_eval, summarize, write_results

ROWS = [
    {"id": "1", "label": "true", "statement": "a"},
    {"id": "2", "label": "false", "statement": "b"},
    {"id": "3", "label": "half-true", "statement": "c"},
]


def test_macro_f1_perfect_and_zero():
    assert macro_f1([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]) == 1.0
    assert macro_f1([0, 1], [-1, -1]) == 0.0


def test_run_eval_and_summary(tmp_path):
    answers = iter(["true", "false", "garbage", "garbage"])  # row 3 fails twice
    records = run_eval(ROWS, lambda m: next(answers))
    assert [r["y_pred"] for r in records] == [5, 1, -1]

    s = summarize(records, "fake-model", "CPU", None)
    assert s["n_eval"] == 3
    assert s["invalid_outputs"] == 1
    assert abs(s["accuracy"] - 2 / 3) < 1e-3

    write_results(records, s, str(tmp_path))
    with open(tmp_path / "llm_preds.csv", newline="") as f:
        assert next(csv.reader(f)) == ["id", "y_true", "y_pred", "latency_ms"]
    assert json.loads((tmp_path / "llm.json").read_text())["model"] == "llm"
