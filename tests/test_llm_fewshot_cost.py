"""Tests for few-shot prompting and the cost analysis (no GPU / dataset needed)."""
import json

from common.config import IDX_TO_LABEL
from llm.cost_analysis import add_comparisons, break_even_predictions, load_rows, main, train_cost
from llm.evaluate_llm import run_eval
from llm.fewshot import select_examples
from llm.inference import classify
from llm.prompt import build_messages

# 12 fake rows per class, all with a usable length
STMTS = [f"statement number {i} about topic {c} with enough words to pass the filter ok"
         for c in range(6) for i in range(12)]
LABELS = [c for c in range(6) for _ in range(12)]


def test_select_examples_one_per_class_deterministic():
    a = select_examples(STMTS, LABELS)
    assert a == select_examples(STMTS, LABELS)
    assert sorted(e["label"] for e in a) == sorted(IDX_TO_LABEL.values())
    assert a != select_examples(STMTS, LABELS, seed=7)


def test_select_examples_order_not_label_sorted():
    labels = [e["label"] for e in select_examples(STMTS, LABELS)]
    assert labels != [IDX_TO_LABEL[i] for i in range(6)]


def test_build_messages_few_shot_structure():
    ex = select_examples(STMTS, LABELS)
    m = build_messages("Is this true?", ex)
    assert m[0]["role"] == "system" and len(m) == 1 + 2 * 6 + 1
    assert [x["role"] for x in m[1:]] == ["user", "assistant"] * 6 + ["user"]
    assert m[-1]["content"].endswith("Is this true?\nLabel:")
    assert build_messages("x") == build_messages("x", None) == build_messages("x", [])


def test_classify_and_run_eval_pass_examples_through():
    ex = [{"statement": "s", "label": "true"}]
    seen = []

    def gen(messages):
        seen.append(len(messages))
        return "false"

    assert classify("q", gen, examples=ex) == "false"
    preds, _ = run_eval([{"statement": "q", "y_true": 1}], gen, ex)
    assert preds == [1] and seen == [4, 4]


def test_break_even():
    assert break_even_predictions(1.0, 0.001, 0.101) == 10000
    assert break_even_predictions(1.0, 0.2, 0.1) is None


def _res(path, name, acc, f1, params, train, ms, cost, **extra):
    d = dict(model=name, accuracy=acc, macro_f1=f1, params=params, train_minutes=train, gpu="T4",
             avg_latency_ms=ms, est_cost_per_1k=cost, n_eval=300, invalid_outputs=0, **extra)
    (path / f"{name}.json").write_text(json.dumps(d))


def test_cost_analysis_end_to_end(tmp_path):
    _res(tmp_path, "bert_text", 0.25, 0.24, 110_000_000, 3.0, 10, 0.001)
    _res(tmp_path, "llm_zeroshot", 0.20, 0.15, 7_600_000_000, 0, 500, 0.06, gpu_peak_mem_mb=5400)
    _res(tmp_path, "llm_fewshot", 0.22, 0.18, 7_600_000_000, 0, 900, 0.11, gpu_peak_mem_mb=5600)
    rows = load_rows(str(tmp_path))
    assert [r["run"] for r in rows] == ["bert_text", "llm_zeroshot", "llm_fewshot"]
    comps = add_comparisons(rows)
    assert comps[0]["vs_llm"] == "Qwen2.5-7B few-shot" and comps[0]["speedup"] == 90.0
    assert abs(train_cost(60) - 0.385) < 1e-9
    main(str(tmp_path))
    assert (tmp_path / "cost_comparison.md").exists() and (tmp_path / "cost_vs_accuracy.csv").exists()


def test_read_liar_tsv_parses_original_format(tmp_path):
    import pytest
    pytest.importorskip("datasets")  # data_loader imports it
    from common.data_loader import label_to_idx, read_liar_tsv
    row = "1.json\tpants-fire\tA claim.\tsubj\tsomeone\tGovernor\tTexas\trepublican\t1\t2\t3\t4\t5\tctx"
    short = "2.json\ttrue\tShort row\tsubj"  # missing trailing columns
    p = tmp_path / "t.tsv"
    p.write_text(row + "\n" + short + "\n", encoding="utf-8")
    cols = read_liar_tsv(str(p))
    assert cols["statement"] == ["A claim.", "Short row"]
    assert [label_to_idx(l) for l in cols["label"]] == [0, 5]
    assert cols["pants_on_fire_counts"] == [5, 0] and cols["barely_true_counts"] == [1, 0]
