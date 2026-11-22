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
from typing import Dict, List, Optional, Tuple  # type hints (Dict, List, Optional, Tuple); sirf padhne me madad, behaviour nahi badalta

# Project ke shared constants: label ka order ek hi jagah se aata hai, hum copy nahi karte.
from common.config import IDX_TO_LABEL, LABEL_TO_IDX, NUM_CLASSES, RESULTS_DIR

# liar_data/test.tsv me header nahi hai, isliye columns ke naam hum khud yahan likhte hain.
# Order LIAR dataset ke README wala hai.
# VIVA Q: test.tsv me header nahi hai, toh columns ke naam kahan se aaye?
#     A: LIAR dataset ke README ka order hai: id, label, statement, subject, speaker, job title, state,
#        party, 5 credit-history counts, context. Hum naam khud likhte hain taaki row ko dictionary bana
#        sakein.
TSV_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state",  # test.tsv ke pehle 7 columns ke naam
    "party", "barely_true", "false", "half_true", "mostly_true", "pants_fire", "context",  # baaki columns: party, 5 credit-history counts aur context
]
DEFAULT_TSV = "liar_data/test.tsv"  # test.tsv ka default path (repo root se chalane par)


# ======================================================================================
# 1. LOADING (files padhna)
# ======================================================================================

# VIVA Q: JSON me do confusion matrix kyun hain?
#     A: Upar wali (confusion_matrix) 300-row shared subset ki hai, jo sab models ke liye same hai.
#        full_test ke andar wali poore 1,283 test rows ki hai, jo zyada bharosemand hai par LLM par nahi
#        chal paati. Report dono dikhati hai.
def load_results(run: str, results_dir: str = RESULTS_DIR) -> Dict:  # results/<run>.json padhta hai
    """
    results/<run>.json padhkar dictionary return karta hai.

    WHY: confusion matrix, accuracy aur macro-F1 isi file me hain. Agar file nahi mili (jaise
    RoBERTa ka run abhi hua hi nahi), toh saaf message ke saath rukte hain, lamba traceback nahi.
    """
    path = os.path.join(results_dir, f"{run}.json")  # file ka path jodta hai, jaise results/roberta_text.json
    if not os.path.exists(path):  # file nahi mili?
        raise FileNotFoundError(  # saaf error deke rukte hain (lamba traceback nahi)
            f"{path} not found. Run the training first (python -m finetune.train_bert ... "  # error message ka pehla hissa: kaun si file nahi mili
            f"--run-name {run}) and make sure the run name is spelled exactly."  # doosra hissa: kya karna hai aur naam bilkul sahi likhna hai
        )
    with open(path, encoding="utf-8") as f:   # utf-8: statements me special characters ho sakte hain
        return json.load(f)  # JSON ko Python dictionary banake wapas


# VIVA Q: Preds CSV me kya hota hai, aur id kya hai?
#     A: Columns: id, y_true, y_pred, latency_ms (300 rows, shared subset). id test split ki row number
#        hai (0 se shuru), isi se TSV me asli statement milta hai.
def load_preds(run: str, results_dir: str = RESULTS_DIR) -> List[Dict]:  # results/<run>_preds.csv padhta hai (har row = ek test statement ki prediction)
    """
    results/<run>_preds.csv padhta hai. Har row = ek test statement ki prediction.

    Return: list of dict, jisme id, y_true, y_pred teeno ints hain.
    WHY int me badalte hain: CSV me sab kuch text hota hai, par hume gintee/compare karni hai.
    """
    path = os.path.join(results_dir, f"{run}_preds.csv")  # preds CSV ka path
    if not os.path.exists(path):  # CSV nahi mili?
        raise FileNotFoundError(f"{path} not found.")  # file na ho toh seedha error
    rows = []  # yahan saari rows jama hongi
    with open(path, newline="", encoding="utf-8") as f:  # CSV kholo (utf-8; newline='' csv module ka rule hai)
        for r in csv.DictReader(f):   # DictReader pehli line ko header maan leta hai
            rows.append({"id": int(r["id"]), "y_true": int(r["y_true"]), "y_pred": int(r["y_pred"])})  # text values ko int banao (id, sahi label, predicted label)
    return rows  # 300 rows ki list wapas


def load_test_tsv(path: str = DEFAULT_TSV) -> List[Dict]:  # liar_data/test.tsv padhta hai taaki id se asli statement mile
    """
    liar_data/test.tsv padhta hai aur har row ko dictionary me badalta hai.

    List ka index (0, 1, 2 ...) wahi "id" hai jo preds CSV me hota hai, isliye
    tsv[id] se seedha statement mil jaata hai.
    Label ko text se project ke number (0-5) me badalte hain, taaki preds se match kar sakein.
    """
    if not os.path.exists(path):  # TSV nahi mili?
        raise FileNotFoundError(f"{path} not found (the LIAR test file should be in the repo).")  # TSV na ho toh seedha error
    rows = []  # yahan test ki saari 1,283 rows jama hongi
    with open(path, encoding="utf-8") as f:  # TSV kholo
        for line in f:  # ek line = ek statement
            parts = line.rstrip("\n").split("\t")   # TSV = tab se alag ki hui values
            if len(parts) < len(TSV_COLUMNS):  # kisi row me aakhri column (context) gayab ho toh?
                # Kuch rows me last column (context) khaali ho sakta hai, tab usse "" se bhar do.
                parts += [""] * (len(TSV_COLUMNS) - len(parts))
            row = dict(zip(TSV_COLUMNS, parts))  # column-naam aur value ko jodke dictionary banao
            # "pants-fire" jaise text ko 0 jaise number me badalna. Unknown label par crash ke bajay -1.
            row["label_idx"] = LABEL_TO_IDX.get(row["label"].strip().lower(), -1)
            rows.append(row)  # list me jodo; list ka index hi preds ka id hai
    return rows  # saari rows wapas


# VIVA Q: Alignment check kyun zaroori hai?
#     A: Agar preds ka id aur TSV ki row ka mel na ho toh hum galat statement ko galat example ke roop me
#        report me daal dete, aur kisi ko pata na chalta. Tareeka: preds ka sahi label (y_true) aur TSV
#        ka label ek hi hona chahiye. Hamare runs me mismatch 0 mila.
# VIVA Q: Alignment kaise guarantee hota hai?
#     A: 300 rows shared subset hain (seed 42, stratified); har model usi ke ids par predict karta hai,
#        aur TSV ki row order HF dataset jaisi hai.
def check_alignment(preds: List[Dict], tsv: List[Dict]) -> int:  # check: preds ka id sach me TSV ki sahi row ko point karta hai?
    """
    Check karta hai ki preds ka "id" sach me TSV ki sahi row ko point kar raha hai.

    WHY (ye sabse zaroori safety check hai): agar id aur TSV row ka mel na ho, toh hum galat
    statement ko galat example ke roop me report me daal denge, aur kisi ko pata bhi nahi chalega.
    Tareeka: preds ka y_true (sahi jawab) aur TSV ka label ek hi hona chahiye.
    Return: kitni rows me mismatch hai. Sahi hone par 0.
    """
    bad = 0  # mismatch ka counter
    for p in preds:  # har prediction row ke liye
        i = p["id"]  # ye test split ki row number hai
        # id TSV ke bahar ho ya label alag ho, dono ko mismatch gino.
        if i >= len(tsv) or tsv[i]["label_idx"] != p["y_true"]:
            bad += 1  # mismatch mila
    return bad  # kitni rows me mismatch (0 hona chahiye)


# ======================================================================================
# 2. CONFUSION MATRIX ANALYSIS (gintee wale functions, bina file ke test ho sakte hain)
# ======================================================================================

# VIVA Q: Confusion matrix kaise padhte hain?
#     A: Rows = asli class, columns = model ka jawab. cm[i][j] = asli class i ko model ne j bola.
#        Diagonal sahi jawab hain; baaki galtiyan.
# VIVA Q: Accuracy, recall, precision, F1 aur macro-F1 me kya fark hai?
#     A: Accuracy = kul sahi / kul rows. Recall (class) = us class ke kitne % asli rows pakde. Precision
#        = jo model ne us class ka bola unme kitne sahi. F1 = precision aur recall ka harmonic mean.
#        Macro-F1 = 6 classes ke F1 ka simple average, toh rare class bhi barabar tolti hai.
# VIVA Q: RoBERTa me recall kya dikhata hai?
#     A: Text-only: half-true sirf 10%, true 47%, false 22%. Speaker info ke saath half-true 42% aur
#        false 44% tak jaate hain, par true 29% par gir jaata hai (model true ko mostly-true bolne lagta
#        hai).
def per_class_recall(cm: List[List[int]]) -> List[Optional[float]]:  # har class ka recall nikalta hai
    """
    Har class ke liye recall = (sahi pehchane gaye) / (us class ke total asli rows).

    Matrix ka rule: cm[i][j] = asli class i, model ne bola class j.
    Isliye diagonal cm[i][i] sahi pehchaane gaye rows hain, aur row i ka sum asli rows.
    Agar kisi class ke 0 rows hain toh None (zero se divide nahi karte).
    """
    out = []  # 6 classes ke recall yahan
    for i, row in enumerate(cm):  # i = asli class, row = us class ke model ke jawab
        total = sum(row)  # us class ki kul rows
        out.append(row[i] / total if total else None)  # diagonal (sahi pehchani) / kul rows; 0 rows ho toh None
    return out  # recall ki list (class 0 se 5)


# VIVA Q: Count aur rate dono kyun dete hain?
#     A: Count akela dhokha deta hai: bade class me galti zyada dikhti hai. Rate = galti / us asli class
#        ki kul rows, toh class ke size ka asar hat jaata hai.
# VIVA Q: Sabse zyada confuse kaun se pairs hain?
#     A: RoBERTa text-only (300 rows): half-true -> true 20 (32%), false -> barely-true 16, mostly-true
#        -> true 16, true -> mostly-true 13. Full test: mostly-true -> true 76, half-true -> true 71.
#        Model true ki taraf jhukta hai, aur false -> true (4 level door) bhi 53 baar hota hai.
def top_confused_pairs(cm: List[List[int]], top_n: int = 5) -> List[Dict]:  # sabse zyada galti wale (asli -> predicted) pairs
    """
    Sabse zyada galti wale (asli -> predicted) pairs dhoondta hai.

    Sirf off-diagonal (i != j) dekhte hain, kyunki diagonal sahi jawab hai.
    "rate" = us galti ka hissa us asli class ke total me. Count akela kabhi dhokha deta hai
    (bade class me galti zyada dikhti hai), isliye rate bhi saath dete hain.
    Sort: pehle count bade se chhota.
    """
    pairs = []  # yahan saare galat pairs jama honge
    for i in range(len(cm)):  # i = asli class
        row_total = sum(cm[i])  # us asli class ki kul rows (rate ke liye)
        for j in range(len(cm[i])):  # j = model ka jawab
            if i == j or cm[i][j] == 0:  # diagonal (sahi) aur zero galtiyan chhod do
                continue                      # diagonal (sahi) aur zero wale chhod do
            pairs.append({  # ek galat pair ka record banao
                "true": i, "pred": j, "count": cm[i][j],  # asli class, predicted class aur kitni baar
                "rate": cm[i][j] / row_total if row_total else 0.0,  # rate = ye galti us class ki kul rows ka kitna hissa hai
                "distance": abs(i - j),       # truthfulness line par kitni door (1 = padosi class)
            })  # record khatam
    pairs.sort(key=lambda p: (-p["count"], p["true"], p["pred"]))   # tie hone par order fixed rahe
    return pairs[:top_n]  # sirf top_n sabse badi galtiyan


# VIVA Q: Padosi (adjacent) galtiyan kyun dekhte hain?
#     A: LIAR ke 6 labels ek line par hain (pants-fire se true). Half-true ko mostly-true bolna 'lagbhag
#        sahi' hai, jabki pants-fire ko true bolna bahut bura. Isliye dekhte hain ki galtiyan padosi
#        levels ke beech hain ya door.
# VIVA Q: Is analysis me kya mila?
#     A: Text-only RoBERTa: 42% (subset) aur 46% (full test) galtiyan padosi levels ke beech, yaani bahut
#        galtiyan door ki bhi hain. Metadata ke saath 54% aur 53%, toh galtiyan zyada paas wali ho jaati
#        hain.
def adjacent_error_share(cm: List[List[int]]) -> Dict:  # kitni galtiyan padosi levels ke beech hain?
    """
    Kitni galtiyan "padosi" classes ke beech hain (|asli - predicted| == 1)?

    WHY: LIAR ke labels ek line par hain. Agar zyada galtiyan padosi classes ke beech hain, toh
    model ko "lagbhag sahi" idea hai, bas fine line nahi kheench paata. Ye report ka
    mahatvapurn insight hai ("adjacent truthfulness levels" issue #11 me bhi bola gaya tha).
    """
    adjacent = far = 0  # padosi aur door ki galtiyon ke counters
    for i in range(len(cm)):  # asli class
        for j in range(len(cm[i])):  # model ka jawab
            if i == j:  # sahi jawab galti nahi hai
                continue  # agle pair par jao
            if abs(i - j) == 1:  # 1 level ka fark = padosi class
                adjacent += cm[i][j]  # padosi galti ginti me jodo
            else:
                far += cm[i][j]  # 2 ya zyada level door ki galti
    total = adjacent + far  # kul galtiyan
    return {"adjacent": adjacent, "far": far, "total_errors": total,  # padosi, door, kul galtiyan wapas
            "share": adjacent / total if total else 0.0}  # share = padosi galtiyan / kul galtiyan


# ======================================================================================
# 3. WRONG EXAMPLES (galat predictions ke asli statements)
# ======================================================================================

# VIVA Q: Examples random kyun nahi chune?
#     A: Taaki script dobara chalane par report me wahi examples aayein (reproducible). Har example kisi
#        top confused pair se judta hai, isliye dikhta hai ki wo galti kaisi lagti hai.
# VIVA Q: Examples kya dikhate hain?
#     A: Jaise half-true statement 'sirf 1 percent Americans millionaires hain...' ko model ne true bola.
#        Ye statements aadhe sach aur aadhe gumrah karne wale hote hain, toh insaan ko bhi sabut dekhna
#        padta; sirf text se model ke liye ye mushkil hai.
def select_examples(preds: List[Dict], tsv: List[Dict], pairs: List[Dict], per_pair: int = 2) -> List[Dict]:  # har top pair ke kuch galat examples chunta hai aur unka asli statement jodta hai
    """
    Har top confused pair ke liye kuch galat examples chunta hai, aur unka statement jodta hai.

    Chunav deterministic hai (id ke order me pehle wale), random nahi, taaki script dobara
    chalane par report me wahi examples aayein. Aur har example kisi top pair se judta hai,
    isliye report me "ye galti kaisi dikhti hai" seedha samajh aata hai.
    """
    examples = []  # chune hue examples yahan
    for pair in pairs:  # har top confused pair ke liye
        # Is pair (asli, predicted) wali saari galat rows, id ke order me.
        wrong = sorted(
            (p for p in preds if p["y_true"] == pair["true"] and p["y_pred"] == pair["pred"]),  # is pair (asli, predicted) wali saari galat predictions
            key=lambda p: p["id"],  # id ke order me, taaki hamesha wahi examples aayein (random nahi)
        )
        for p in wrong[:per_pair]:                # sirf per_pair itne lo
            row = tsv[p["id"]]  # preds ka id = TSV ki row number, toh seedha asli statement mil gaya
            examples.append({  # example ka record
                "id": p["id"], "true": p["y_true"], "pred": p["y_pred"],  # row id, asli label, predicted label
                "statement": row["statement"], "speaker": row["speaker"], "subject": row["subject"],  # asli statement, speaker aur subject
            })  # record khatam
    return examples  # examples ki list


# ======================================================================================
# 4. REPORT BANANA (markdown text)
# ======================================================================================

def _name(idx: int) -> str:  # number (0-5) ko label ka naam banata hai
    """Number (0-5) ko label ka naam (jaise 'half-true') banata hai."""
    return IDX_TO_LABEL[idx]  # jaise 3 -> half-true


# VIVA Q: '|' ko escape kyun karte hain?
#     A: Markdown table me '|' column ko alag karta hai. Statement me agar '|' aaya toh table toot jaati,
#        isliye '\|' likhte hain.
def _shorten(text: str, limit: int = 140) -> str:  # lamba statement chhota karta hai aur '|' ko escape karta hai
    """
    Lamba statement kaat deta hai aur '|' ko '\\|' bana deta hai.

    WHY '|': markdown table me '|' column ka alag karne wala symbol hota hai. Statement me agar
    '|' aaya toh table toot jayegi, isliye usse escape karte hain.
    """
    text = text.replace("|", "\\|").strip()  # '|' markdown table todta hai, isliye '\|'
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."  # limit se chhota ho toh waisa hi, warna kaatke '...' lagao


def _pairs_table(pairs: List[Dict]) -> List[str]:  # confused pairs ki markdown table ki lines banata hai
    """Top confused pairs ki markdown table ki lines banata hai."""
    lines = ["| True label | Predicted as | Count | Share of true class | Levels apart |",  # table ka header
             "|---|---|---|---|---|"]  # header ke neeche ki line (markdown ka rule)
    for p in pairs:  # har pair ek row
        lines.append(f"| {_name(p['true'])} | {_name(p['pred'])} | {p['count']} | "  # asli label | predicted label | count
                     f"{p['rate'] * 100:.0f}% | {p['distance']} |")  # rate (%) aur kitne level door
    return lines  # table ki lines wapas


# VIVA Q: Report me kya-kya aata hai?
#     A: Headline accuracy/macro-F1, class-wise recall, top confused pairs (subset aur full test),
#        padosi-galti ka share, aur galat examples. Ye sab seedha report me chipkaya ja sakta hai.
def render_markdown(run: str, results: Dict, pairs_subset: List[Dict], pairs_full: Optional[List[Dict]],  # saare numbers ko ek markdown report me jodta hai
                    examples: List[Dict], adjacency: Dict, adjacency_full: Optional[Dict]) -> str:  # (baaki parameters: examples aur padosi-galti ke numbers)
    """
    Saare numbers ko ek saaf markdown report me jodta hai (report me seedha chipka sakte ho).
    """
    cm = results["confusion_matrix"]  # 300-row subset ki confusion matrix
    recalls = per_class_recall(cm)  # har class ka recall
    out = [f"# Error analysis: {run}", ""]  # report ki lines: pehli heading
    out.append(f"Shared 300-row subset: accuracy {results['accuracy']:.3f}, macro-F1 {results['macro_f1']:.3f}."  # headline: subset ki accuracy aur macro-F1
               + (f" Full test ({results['full_test']['n']} rows): accuracy "  # agar full-test ka data hai toh uska bhi zikr
                  f"{results['full_test']['accuracy']:.3f}, macro-F1 {results['full_test']['macro_f1']:.3f}."  # full test ki accuracy aur macro-F1
                  if "full_test" in results else ""))  # nahi hai toh khaali string
    out += ["", "## Recall per class (300-row subset)", "",  # recall table ki heading
            "| Class | Rows | Recall |", "|---|---|---|"]  # table ka header
    for i in range(NUM_CLASSES):  # 6 classes me se har ek
        rec = "n/a" if recalls[i] is None else f"{recalls[i] * 100:.0f}%"  # recall % me, ya n/a agar us class ki 0 rows
        out.append(f"| {_name(i)} | {sum(cm[i])} | {rec} |")  # ek class ki row: naam | rows | recall

    out += ["", "## Most confused pairs (300-row subset)", ""] + _pairs_table(pairs_subset)  # subset ke top confused pairs
    if pairs_full:  # full-test ke pairs hon toh
        out += ["", "## Most confused pairs (full test set, more reliable)", ""] + _pairs_table(pairs_full)  # full test ke top pairs (zyada bharosemand)

    out += ["", "## Do mistakes stay between neighbouring levels?", "",  # padosi levels wali galtiyon ka section
            f"On the subset, {adjacency['adjacent']} of {adjacency['total_errors']} errors "  # subset: kitni galtiyan padosi levels ke beech
            f"({adjacency['share'] * 100:.0f}%) are between neighbouring truthfulness levels."]  # us ka percentage
    if adjacency_full:  # full test ka bhi ho toh
        out.append(f"On the full test set the share is {adjacency_full['share'] * 100:.0f}% "  # full test ka percentage
                   f"({adjacency_full['adjacent']} of {adjacency_full['total_errors']}).")  # (adjacent aur kul galtiyan)

    out += ["", "## Example mistakes (subset)", "",  # galat examples ka section
            "| Row id | True | Predicted | Speaker | Statement |", "|---|---|---|---|---|"]  # examples table ka header
    for e in examples:  # har example ek row
        out.append(f"| {e['id']} | {_name(e['true'])} | {_name(e['pred'])} | "  # row id | asli | predicted
                   f"{_shorten(e['speaker'], 30)} | {_shorten(e['statement'])} |")  # speaker | chhota kiya hua statement
    out.append("")  # aakhir me khaali line
    return "\n".join(out)  # saari lines ek text me jodke wapas


# ======================================================================================
# 5. MAIN (command line se chalane wala hissa)
# ======================================================================================

# VIVA Q: Examples subset ke pairs se hi kyun, full test ke pairs se kyun nahi?
#     A: Preds CSV me sirf 300 subset ki rows hain. Full-test ke pairs ke liye hamare paas rows-level
#        predictions hi nahi hain, sirf matrix hai.
# VIVA Q: Is analysis ki limitations kya hain?
#     A: 300 rows par +/-5 points ka noise; ek hi seed; LIAR ke labels insaani raay hain (subjective).
#        Metadata run me credit-history me statement ka apna label shamil hai, toh uska score upar se
#        achha dikhta hai (LLM se fair comparison text-only run hai).
def analyse(run: str, results_dir: str = RESULTS_DIR, tsv_path: str = DEFAULT_TSV,  # poora pipeline: files padho -> alignment check -> pairs -> examples -> markdown
            top_pairs: int = 5, per_pair: int = 2) -> Tuple[str, int]:  # (top_pairs = kitne pairs; per_pair = har pair ke kitne examples)
    """
    Poora pipeline: files padho -> alignment check -> pairs -> examples -> markdown.

    Return: (markdown text, alignment mismatches ki sankhya). Alag function isliye ki test
    bina command line ke ise seedha call kar sake.
    """
    results = load_results(run, results_dir)  # JSON padho
    preds = load_preds(run, results_dir)  # preds CSV padho
    tsv = load_test_tsv(tsv_path)  # asli statements padho
    mismatches = check_alignment(preds, tsv)  # id aur TSV ka mel check (0 hona chahiye)

    cm = results["confusion_matrix"]                       # 300-row subset ki matrix
    full_cm = results.get("full_test", {}).get("confusion_matrix")   # poore test ki matrix (ho sakta hai na ho)
    pairs_subset = top_confused_pairs(cm, top_pairs)  # subset ke top confused pairs
    pairs_full = top_confused_pairs(full_cm, top_pairs) if full_cm else None  # full test ke top pairs (agar data hai)
    # Examples subset ke top pairs se lete hain, kyunki preds sirf subset ke hain.
    examples = select_examples(preds, tsv, pairs_subset, per_pair)
    md = render_markdown(run, results, pairs_subset, pairs_full, examples,  # sab kuch markdown report me
                         adjacent_error_share(cm), adjacent_error_share(full_cm) if full_cm else None)  # padosi-galti ke numbers: subset aur full test
    return md, mismatches  # (report ka text, mismatch ki ginti)


# VIVA Q: Ye script kaise chalate hain?
#     A: Repo root se: python -m analysis.error_analysis --run roberta_text --out
#        docs/roberta_text_errors.md. --out na dein toh report screen par aati hai.
def main() -> None:  # command line se chalane wala hissa
    ap = argparse.ArgumentParser(description="Error analysis for a finished model run")  # options padhne wala object
    ap.add_argument("--run", required=True, help="results file stem, e.g. roberta_text")  # kaun sa run (jaise roberta_text); zaroori
    ap.add_argument("--results-dir", default=RESULTS_DIR)  # results ka folder
    ap.add_argument("--tsv", default=DEFAULT_TSV, help="path to liar_data/test.tsv")  # test.tsv ka path
    ap.add_argument("--top-pairs", type=int, default=5)  # kitne confused pairs dikhane hain (default 5)
    ap.add_argument("--examples-per-pair", type=int, default=2)  # har pair ke kitne examples (default 2)
    ap.add_argument("--out", default=None, help="optional .md file to write the report to")  # report file me likhni ho toh uska naam
    args = ap.parse_args()  # options padh liye

    md, mismatches = analyse(args.run, args.results_dir, args.tsv, args.top_pairs, args.examples_per_pair)  # pura analysis chalao
    if mismatches:  # alignment me gadbad mili?
        # Loud warning: examples galat statements dikha sakte hain. Report me daalne se pehle check karo.
        print(f"WARNING: {mismatches} preds rows do not match liar_data/test.tsv labels; "
              "the example statements may be wrong.")  # (warning ka doosra hissa)
    if args.out:  # --out diya gaya hai?
        with open(args.out, "w", encoding="utf-8") as f:  # report file kholo (utf-8)
            f.write(md)  # report likho
        print(f"Wrote {args.out}")  # batao ki file ban gayi
    else:
        print(md)  # --out nahi diya toh screen par dikhao


# Script ko seedha chalane par hi main() chalta hai (import karne par nahi).
if __name__ == "__main__":
    main()  # script seedha chalane par pipeline shuru
