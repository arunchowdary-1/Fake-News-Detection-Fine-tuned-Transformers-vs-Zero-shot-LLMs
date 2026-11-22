"""
analysis/error_analysis.py  --  RoBERTa task (error analysis for the final report)
==================================================================================

KAAM KYA HAI? (what is the task)
    Team ne bola: "RoBERTa ke 2 runs (roberta_text, roberta_meta) ke baad report ke liye error
    analysis karo: confusion matrix aur preds CSV se SABSE ZYADA CONFUSE hone wale class pairs batao,
    aur kuch GALAT predictions ke examples dikhao."
    Ye script wahi kaam automatically karti hai. Ek command chalao, aur report me chipkane layak
    markdown table mil jaati hai.

KAUN SI FILES PADHTI HAI? (inputs)
    1) results/<run>.json        -> confusion matrix (6x6), accuracy, macro-F1
                                    (upar wali matrix = 300-row shared subset,
                                     "full_test" ke andar wali = poore 1,283 test rows)
    2) results/<run>_preds.csv   -> columns id,y_true,y_pred,latency_ms (300 rows)
    3) liar_data/test.tsv        -> asli statements (bina header ki file). preds ka "id" column
                                    isi file ki row number (0 se shuru) hai.

KAISE CHALAYEIN? (how to run, repo root se)
    python -m analysis.error_analysis --run roberta_text
    python -m analysis.error_analysis --run roberta_text --out docs/roberta_text_errors.md

WHY IS THIS A SEPARATE SCRIPT? (alag script kyun?)
    Training script (finetune/train_bert.py) sirf numbers banati hai. "Kaun si classes mix hoti hain
    aur kyun" ye alag sawal hai, aur ye har model (LSTM, BERT, RoBERTa) ke liye same tarah chal
    sakta hai. Isliye alag file, aur har kaam ka alag chhota function, taaki test ho sake.

LABEL ORDER (bahut zaroori)
    0 pants-fire, 1 false, 2 barely-true, 3 half-true, 4 mostly-true, 5 true.
    Ye jhooth se sach ki taraf ek "seedhi line" hai, isliye do padosi classes (jaise half-true aur
    mostly-true) ka confuse hona "kam bura" hai, aur pants-fire ko true bolna "bahut bura".
"""

import argparse   # command-line options (--run, --out ...) padhne ke liye
import csv        # preds CSV padhne ke liye
import json       # results JSON padhne ke liye
import os         # file path jodne ke liye
from typing import Dict, List, Optional, Tuple

# Project ke shared constants: label ka order ek hi jagah se aata hai, hum copy nahi karte.
from common.config import IDX_TO_LABEL, LABEL_TO_IDX, NUM_CLASSES, RESULTS_DIR

# liar_data/test.tsv me header nahi hai, isliye columns ke naam hum khud yahan likhte hain.
# Order LIAR dataset ke README wala hai.
TSV_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state",
    "party", "barely_true", "false", "half_true", "mostly_true", "pants_fire", "context",
]
DEFAULT_TSV = "liar_data/test.tsv"


# ======================================================================================
# 1. LOADING (files padhna)
# ======================================================================================

def load_results(run: str, results_dir: str = RESULTS_DIR) -> Dict:
    """
    results/<run>.json padhkar dictionary return karta hai.

    WHY: confusion matrix, accuracy aur macro-F1 isi file me hain. Agar file nahi mili (jaise
    RoBERTa ka run abhi hua hi nahi), toh saaf message ke saath rukte hain, lamba traceback nahi.
    """
    path = os.path.join(results_dir, f"{run}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run the training first (python -m finetune.train_bert ... "
            f"--run-name {run}) and make sure the run name is spelled exactly."
        )
    with open(path, encoding="utf-8") as f:   # utf-8: statements me special characters ho sakte hain
        return json.load(f)


def load_preds(run: str, results_dir: str = RESULTS_DIR) -> List[Dict]:
    """
    results/<run>_preds.csv padhta hai. Har row = ek test statement ki prediction.

    Return: list of dict, jisme id, y_true, y_pred teeno ints hain.
    WHY int me badalte hain: CSV me sab kuch text hota hai, par hume gintee/compare karni hai.
    """
    path = os.path.join(results_dir, f"{run}_preds.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):   # DictReader pehli line ko header maan leta hai
            rows.append({"id": int(r["id"]), "y_true": int(r["y_true"]), "y_pred": int(r["y_pred"])})
    return rows


def load_test_tsv(path: str = DEFAULT_TSV) -> List[Dict]:
    """
    liar_data/test.tsv padhta hai aur har row ko dictionary me badalta hai.

    List ka index (0, 1, 2 ...) wahi "id" hai jo preds CSV me hota hai, isliye
    tsv[id] se seedha statement mil jaata hai.
    Label ko text se project ke number (0-5) me badalte hain, taaki preds se match kar sakein.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found (the LIAR test file should be in the repo).")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")   # TSV = tab se alag ki hui values
            if len(parts) < len(TSV_COLUMNS):
                # Kuch rows me last column (context) khaali ho sakta hai, tab usse "" se bhar do.
                parts += [""] * (len(TSV_COLUMNS) - len(parts))
            row = dict(zip(TSV_COLUMNS, parts))
            # "pants-fire" jaise text ko 0 jaise number me badalna. Unknown label par crash ke bajay -1.
            row["label_idx"] = LABEL_TO_IDX.get(row["label"].strip().lower(), -1)
            rows.append(row)
    return rows


def check_alignment(preds: List[Dict], tsv: List[Dict]) -> int:
    """
    Check karta hai ki preds ka "id" sach me TSV ki sahi row ko point kar raha hai.

    WHY (ye sabse zaroori safety check hai): agar id aur TSV row ka mel na ho, toh hum galat
    statement ko galat example ke roop me report me daal denge, aur kisi ko pata bhi nahi chalega.
    Tareeka: preds ka y_true (sahi jawab) aur TSV ka label ek hi hona chahiye.
    Return: kitni rows me mismatch hai. Sahi hone par 0.
    """
    bad = 0
    for p in preds:
        i = p["id"]
        # id TSV ke bahar ho ya label alag ho, dono ko mismatch gino.
        if i >= len(tsv) or tsv[i]["label_idx"] != p["y_true"]:
            bad += 1
    return bad


# ======================================================================================
# 2. CONFUSION MATRIX ANALYSIS (gintee wale functions, bina file ke test ho sakte hain)
# ======================================================================================

def per_class_recall(cm: List[List[int]]) -> List[Optional[float]]:
    """
    Har class ke liye recall = (sahi pehchane gaye) / (us class ke total asli rows).

    Matrix ka rule: cm[i][j] = asli class i, model ne bola class j.
    Isliye diagonal cm[i][i] sahi pehchaane gaye rows hain, aur row i ka sum asli rows.
    Agar kisi class ke 0 rows hain toh None (zero se divide nahi karte).
    """
    out = []
    for i, row in enumerate(cm):
        total = sum(row)
        out.append(row[i] / total if total else None)
    return out


def top_confused_pairs(cm: List[List[int]], top_n: int = 5) -> List[Dict]:
    """
    Sabse zyada galti wale (asli -> predicted) pairs dhoondta hai.

    Sirf off-diagonal (i != j) dekhte hain, kyunki diagonal sahi jawab hai.
    "rate" = us galti ka hissa us asli class ke total me. Count akela kabhi dhokha deta hai
    (bade class me galti zyada dikhti hai), isliye rate bhi saath dete hain.
    Sort: pehle count bade se chhota.
    """
    pairs = []
    for i in range(len(cm)):
        row_total = sum(cm[i])
        for j in range(len(cm[i])):
            if i == j or cm[i][j] == 0:
                continue                      # diagonal (sahi) aur zero wale chhod do
            pairs.append({
                "true": i, "pred": j, "count": cm[i][j],
                "rate": cm[i][j] / row_total if row_total else 0.0,
                "distance": abs(i - j),       # truthfulness line par kitni door (1 = padosi class)
            })
    pairs.sort(key=lambda p: (-p["count"], p["true"], p["pred"]))   # tie hone par order fixed rahe
    return pairs[:top_n]


def adjacent_error_share(cm: List[List[int]]) -> Dict:
    """
    Kitni galtiyan "padosi" classes ke beech hain (|asli - predicted| == 1)?

    WHY: LIAR ke labels ek line par hain. Agar zyada galtiyan padosi classes ke beech hain, toh
    model ko "lagbhag sahi" idea hai, bas fine line nahi kheench paata. Ye report ka
    mahatvapurn insight hai ("adjacent truthfulness levels" issue #11 me bhi bola gaya tha).
    """
    adjacent = far = 0
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            if i == j:
                continue
            if abs(i - j) == 1:
                adjacent += cm[i][j]
            else:
                far += cm[i][j]
    total = adjacent + far
    return {"adjacent": adjacent, "far": far, "total_errors": total,
            "share": adjacent / total if total else 0.0}


# ======================================================================================
# 3. WRONG EXAMPLES (galat predictions ke asli statements)
# ======================================================================================

def select_examples(preds: List[Dict], tsv: List[Dict], pairs: List[Dict], per_pair: int = 2) -> List[Dict]:
    """
    Har top confused pair ke liye kuch galat examples chunta hai, aur unka statement jodta hai.

    Chunav deterministic hai (id ke order me pehle wale), random nahi, taaki script dobara
    chalane par report me wahi examples aayein. Aur har example kisi top pair se judta hai,
    isliye report me "ye galti kaisi dikhti hai" seedha samajh aata hai.
    """
    examples = []
    for pair in pairs:
        # Is pair (asli, predicted) wali saari galat rows, id ke order me.
        wrong = sorted(
            (p for p in preds if p["y_true"] == pair["true"] and p["y_pred"] == pair["pred"]),
            key=lambda p: p["id"],
        )
        for p in wrong[:per_pair]:                # sirf per_pair itne lo
            row = tsv[p["id"]]
            examples.append({
                "id": p["id"], "true": p["y_true"], "pred": p["y_pred"],
                "statement": row["statement"], "speaker": row["speaker"], "subject": row["subject"],
            })
    return examples


# ======================================================================================
# 4. REPORT BANANA (markdown text)
# ======================================================================================

def _name(idx: int) -> str:
    """Number (0-5) ko label ka naam (jaise 'half-true') banata hai."""
    return IDX_TO_LABEL[idx]


def _shorten(text: str, limit: int = 140) -> str:
    """
    Lamba statement kaat deta hai aur '|' ko '\\|' bana deta hai.

    WHY '|': markdown table me '|' column ka alag karne wala symbol hota hai. Statement me agar
    '|' aaya toh table toot jayegi, isliye usse escape karte hain.
    """
    text = text.replace("|", "\\|").strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _pairs_table(pairs: List[Dict]) -> List[str]:
    """Top confused pairs ki markdown table ki lines banata hai."""
    lines = ["| True label | Predicted as | Count | Share of true class | Levels apart |",
             "|---|---|---|---|---|"]
    for p in pairs:
        lines.append(f"| {_name(p['true'])} | {_name(p['pred'])} | {p['count']} | "
                     f"{p['rate'] * 100:.0f}% | {p['distance']} |")
    return lines


def render_markdown(run: str, results: Dict, pairs_subset: List[Dict], pairs_full: Optional[List[Dict]],
                    examples: List[Dict], adjacency: Dict, adjacency_full: Optional[Dict]) -> str:
    """
    Saare numbers ko ek saaf markdown report me jodta hai (report me seedha chipka sakte ho).
    """
    cm = results["confusion_matrix"]
    recalls = per_class_recall(cm)
    out = [f"# Error analysis: {run}", ""]
    out.append(f"Shared 300-row subset: accuracy {results['accuracy']:.3f}, macro-F1 {results['macro_f1']:.3f}."
               + (f" Full test ({results['full_test']['n']} rows): accuracy "
                  f"{results['full_test']['accuracy']:.3f}, macro-F1 {results['full_test']['macro_f1']:.3f}."
                  if "full_test" in results else ""))
    out += ["", "## Recall per class (300-row subset)", "",
            "| Class | Rows | Recall |", "|---|---|---|"]
    for i in range(NUM_CLASSES):
        rec = "n/a" if recalls[i] is None else f"{recalls[i] * 100:.0f}%"
        out.append(f"| {_name(i)} | {sum(cm[i])} | {rec} |")

    out += ["", "## Most confused pairs (300-row subset)", ""] + _pairs_table(pairs_subset)
    if pairs_full:
        out += ["", "## Most confused pairs (full test set, more reliable)", ""] + _pairs_table(pairs_full)

    out += ["", "## Do mistakes stay between neighbouring levels?", "",
            f"On the subset, {adjacency['adjacent']} of {adjacency['total_errors']} errors "
            f"({adjacency['share'] * 100:.0f}%) are between neighbouring truthfulness levels."]
    if adjacency_full:
        out.append(f"On the full test set the share is {adjacency_full['share'] * 100:.0f}% "
                   f"({adjacency_full['adjacent']} of {adjacency_full['total_errors']}).")

    out += ["", "## Example mistakes (subset)", "",
            "| Row id | True | Predicted | Speaker | Statement |", "|---|---|---|---|---|"]
    for e in examples:
        out.append(f"| {e['id']} | {_name(e['true'])} | {_name(e['pred'])} | "
                   f"{_shorten(e['speaker'], 30)} | {_shorten(e['statement'])} |")
    out.append("")
    return "\n".join(out)


# ======================================================================================
# 5. MAIN (command line se chalane wala hissa)
# ======================================================================================

def analyse(run: str, results_dir: str = RESULTS_DIR, tsv_path: str = DEFAULT_TSV,
            top_pairs: int = 5, per_pair: int = 2) -> Tuple[str, int]:
    """
    Poora pipeline: files padho -> alignment check -> pairs -> examples -> markdown.

    Return: (markdown text, alignment mismatches ki sankhya). Alag function isliye ki test
    bina command line ke ise seedha call kar sake.
    """
    results = load_results(run, results_dir)
    preds = load_preds(run, results_dir)
    tsv = load_test_tsv(tsv_path)
    mismatches = check_alignment(preds, tsv)

    cm = results["confusion_matrix"]                       # 300-row subset ki matrix
    full_cm = results.get("full_test", {}).get("confusion_matrix")   # poore test ki matrix (ho sakta hai na ho)
    pairs_subset = top_confused_pairs(cm, top_pairs)
    pairs_full = top_confused_pairs(full_cm, top_pairs) if full_cm else None
    # Examples subset ke top pairs se lete hain, kyunki preds sirf subset ke hain.
    examples = select_examples(preds, tsv, pairs_subset, per_pair)
    md = render_markdown(run, results, pairs_subset, pairs_full, examples,
                         adjacent_error_share(cm), adjacent_error_share(full_cm) if full_cm else None)
    return md, mismatches


def main() -> None:
    ap = argparse.ArgumentParser(description="Error analysis for a finished model run")
    ap.add_argument("--run", required=True, help="results file stem, e.g. roberta_text")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--tsv", default=DEFAULT_TSV, help="path to liar_data/test.tsv")
    ap.add_argument("--top-pairs", type=int, default=5)
    ap.add_argument("--examples-per-pair", type=int, default=2)
    ap.add_argument("--out", default=None, help="optional .md file to write the report to")
    args = ap.parse_args()

    md, mismatches = analyse(args.run, args.results_dir, args.tsv, args.top_pairs, args.examples_per_pair)
    if mismatches:
        # Loud warning: examples galat statements dikha sakte hain. Report me daalne se pehle check karo.
        print(f"WARNING: {mismatches} preds rows do not match liar_data/test.tsv labels; "
              "the example statements may be wrong.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Wrote {args.out}")
    else:
        print(md)


# Script ko seedha chalane par hi main() chalta hai (import karne par nahi).
if __name__ == "__main__":
    main()
