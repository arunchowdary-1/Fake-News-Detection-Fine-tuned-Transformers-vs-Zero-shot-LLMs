"""
tests/test_error_analysis.py  --  analysis/error_analysis.py ke tests
======================================================================

MAQSAD: error analysis ke chhote, galti-prone hisson ko bina asli data ke test karna.
Chhoti nakli (fake) matrix aur nakli TSV banate hain, taaki test 1 second me, kahin bhi chale.

RUN:  pytest tests/test_error_analysis.py -v
"""

from analysis.error_analysis import (
    adjacent_error_share, analyse, check_alignment, load_test_tsv, per_class_recall,
    select_examples, top_confused_pairs, _shorten,
)

# 6x6 nakli matrix. Rows = asli class, columns = model ka jawab (project label order).
# Dhyan do: (half-true=3 asli) -> (mostly-true=4 predicted) = 9, sabse badi galti jaan-bujhkar rakhi hai.
CM = [
    [5, 1, 0, 0, 0, 0],
    [0, 6, 2, 0, 0, 0],
    [0, 1, 4, 3, 0, 0],
    [0, 0, 2, 5, 9, 1],
    [0, 0, 0, 2, 7, 1],
    [1, 0, 0, 0, 1, 8],
]


def test_per_class_recall_uses_diagonal_over_row_total():
    # WHY: recall galat nikla toh poori report ke "kaunsi class acchi/buri" wale statements galat honge.
    r = per_class_recall(CM)
    assert r[0] == 5 / 6
    assert r[3] == 5 / 17


def test_per_class_recall_empty_class_is_none():
    # WHY: kisi class ke 0 rows ho toh zero se divide karke crash nahi hona chahiye.
    assert per_class_recall([[0, 0], [1, 1]]) == [None, 0.5]


def test_top_confused_pairs_ranks_by_count_and_skips_diagonal():
    # WHY: sabse badi galti (3 -> 4, count 9) pehle aani chahiye, aur diagonal kabhi nahi.
    pairs = top_confused_pairs(CM, top_n=3)
    assert (pairs[0]["true"], pairs[0]["pred"], pairs[0]["count"]) == (3, 4, 9)
    assert all(p["true"] != p["pred"] for p in pairs)
    assert pairs[0]["distance"] == 1                  # padosi class (half-true -> mostly-true)
    assert abs(pairs[0]["rate"] - 9 / 17) < 1e-9      # rate = count / asli class ke total rows


def test_adjacent_error_share():
    # WHY: ye "galtiyan padosi levels me hain" wala headline number hai. Haath se gina hua expected:
    # adjacent: (0,1)=1,(1,2)=2,(2,1)=1,(2,3)=3,(3,2)=2,(3,4)=9,(4,3)=2,(4,5)=1,(5,4)=1 -> 22
    # far: (3,5)=1,(5,0)=1 -> 2  => total 24
    s = adjacent_error_share(CM)
    assert (s["adjacent"], s["far"], s["total_errors"]) == (22, 2, 24)
    assert abs(s["share"] - 22 / 24) < 1e-9


def _fake_tsv(tmp_path):
    """3 rows wali nakli test.tsv banata hai (bina header ke, tab se alag)."""
    rows = [
        ["1.json", "half-true", "Statement A about taxes", "taxes", "alice", "Gov", "TX", "democrat", "0", "0", "0", "0", "0", "ctx"],
        ["2.json", "mostly-true", "Statement B | with pipe", "jobs", "bob", "Sen", "OH", "republican", "1", "1", "1", "1", "1", "ctx"],
        ["3.json", "false", "Statement C", "health", "carol", "Rep", "NY", "none", "2", "2", "2", "2", "2"],  # last column missing
    ]
    p = tmp_path / "test.tsv"
    p.write_text("\n".join("\t".join(r) for r in rows), encoding="utf-8")
    return str(p)


def test_load_test_tsv_maps_labels_and_handles_missing_column(tmp_path):
    # WHY: label ko project number me badalna zaroori hai (preds se match karne ke liye),
    # aur ek row me last column gayab ho toh bhi script crash nahi honi chahiye.
    tsv = load_test_tsv(_fake_tsv(tmp_path))
    assert [r["label_idx"] for r in tsv] == [3, 4, 1]
    assert tsv[2]["context"] == ""
    assert tsv[0]["statement"] == "Statement A about taxes"


def test_check_alignment_detects_mismatch(tmp_path):
    # WHY: sabse zaroori safety net. id aur TSV ka label na milein toh hume pata chalna chahiye.
    tsv = load_test_tsv(_fake_tsv(tmp_path))
    good = [{"id": 0, "y_true": 3, "y_pred": 4}, {"id": 2, "y_true": 1, "y_pred": 1}]
    bad = [{"id": 0, "y_true": 5, "y_pred": 4}, {"id": 99, "y_true": 1, "y_pred": 1}]
    assert check_alignment(good, tsv) == 0
    assert check_alignment(bad, tsv) == 2             # ek galat label + ek id TSV ke bahar


def test_select_examples_picks_wrong_rows_for_the_pair_in_id_order(tmp_path):
    # WHY: example sirf usi (asli, predicted) pair ka hona chahiye, aur hamesha same order me.
    tsv = load_test_tsv(_fake_tsv(tmp_path))
    preds = [
        {"id": 1, "y_true": 4, "y_pred": 3},
        {"id": 0, "y_true": 3, "y_pred": 4},
        {"id": 2, "y_true": 1, "y_pred": 1},          # sahi prediction, example nahi banna chahiye
    ]
    pairs = [{"true": 3, "pred": 4, "count": 1, "rate": 1.0, "distance": 1}]
    ex = select_examples(preds, tsv, pairs, per_pair=2)
    assert len(ex) == 1 and ex[0]["id"] == 0 and ex[0]["statement"] == "Statement A about taxes"


def test_shorten_escapes_pipe_and_truncates():
    # WHY: '|' markdown table todta hai, aur lamba text table ko bahut chauda kar deta hai.
    assert "\\|" in _shorten("a | b")
    out = _shorten("x" * 300, limit=20)
    assert len(out) == 20 and out.endswith("...")


def test_analyse_end_to_end(tmp_path):
    # WHY: poora pipeline (JSON + CSV + TSV -> markdown) ek saath chalta hai, jaisa asli run me chalega.
    import json
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "demo.json").write_text(json.dumps({
        "accuracy": 0.5, "macro_f1": 0.4, "confusion_matrix": CM,
        "full_test": {"accuracy": 0.45, "macro_f1": 0.35, "n": 1283, "confusion_matrix": CM},
    }), encoding="utf-8")
    (results_dir / "demo_preds.csv").write_text("id,y_true,y_pred,latency_ms\n0,3,4,10.0\n1,4,3,11.0\n",
                                                encoding="utf-8")
    md, mismatches = analyse("demo", str(results_dir), _fake_tsv(tmp_path), top_pairs=2, per_pair=1)
    assert mismatches == 0
    assert "Error analysis: demo" in md and "half-true" in md and "Statement A about taxes" in md
