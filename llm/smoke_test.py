"""Smoke test for the zero-shot prompt + parser on a few real LIAR samples.

Purpose (issue #7): check on ~20 samples that the model's raw output can be
parsed into one of the 6 labels BEFORE the full 300-sample run (#8).

Example (small model, CPU is fine):
    python -m llm.smoke_test --data path/to/test.tsv --model Qwen/Qwen2.5-1.5B-Instruct

Example (the real 7B model on a T4, 4-bit):
    python -m llm.smoke_test --data path/to/test.tsv --4bit

Auth: if the model needs a Hugging Face token, set the HF_TOKEN environment
variable. Never put a token in code or on the command line.
"""

# =====================================================================================
# FILE GUIDE (plain-English summary, added for understanding and viva preparation)
#
# WHAT THIS FILE DOES
#   1) make_generate_fn(): loads the LLM ONCE and returns a function  text-in -> text-out.
#      evaluate_llm.py reuses it, so the real run and this quick check use the same model code.
#   2) main(): a quick CHECK on about 20 statements. Question answered: "Can our parser read
#      what this model says?" It is run BEFORE the long 300-statement run, so we do not waste
#      an hour of GPU time on a broken prompt.
#
# WHY THE MODEL IS LOADED IN 4-BIT
#   A 7-billion-parameter model is about 15 GB in normal 16-bit form. In 4-bit it needs only
#   about 5-6 GB, so it fits on a free T4 GPU. We lose a little accuracy for a lot of memory.
#
# VIVA QUESTIONS FOR THIS FILE (with short answers)
#   Q: What is a smoke test?
#   A: A tiny, quick test that only checks "does it run and basically work?" (like switching
#      a device on to see if smoke comes out). It does not measure final accuracy.
#   Q: What is quantization?
#   A: Storing the model's numbers with fewer bits (4 instead of 16) to save memory.
#   Q: What is FP4?
#   A: A 4-bit floating-point format. It is the default of the bitsandbytes library. NF4 is
#      another 4-bit format, usually slightly more accurate, and we did not test it.
#   Q: What is a tokenizer?
#   A: It cuts text into small pieces (tokens) and turns them into numbers the model can read.
#   Q: What is a chat template?
#   A: Each model expects its chat in its own special format. apply_chat_template writes our
#      system/user messages in the format this model was trained on.
# =====================================================================================

# argparse reads command-line options such as --data and --n.
import argparse
# csv reads the tab-separated LIAR file.
import csv
# random with a fixed seed gives the SAME "random" picks on every run.
import random
# time measures how long the check takes.
import time
# defaultdict is a dictionary that creates an empty list the first time a key is used.
from collections import defaultdict

# classify = statement -> label (see inference.py). INVALID = the "invalid" marker.
from llm.inference import classify
from llm.parser import INVALID
# The model name and the six labels.
from llm.prompt import DEFAULT_MODEL_ID, LABELS


# ---- load_liar_tsv --------------------------------------------------------------
# WHAT: read the ORIGINAL LIAR file (test.tsv) into a list of small dictionaries.
# The columns in that file are: 0 = id, 1 = label, 2 = statement (the rest are not needed here).
def load_liar_tsv(path: str) -> list[dict]:
    """Read the original LIAR TSV (no header). Only id, label, statement are used."""
    rows = []  # (the shared loader in common/data_loader.py is used for the real run)
    # encoding="utf-8" reads special characters correctly; newline="" is what the csv module expects.
    with open(path, encoding="utf-8", newline="") as f:
        # delimiter="\t": columns are separated by a TAB, not by a comma.
        for r in csv.reader(f, delimiter="\t"):
            # Skip malformed lines and any label we don't know about.
            # r is a list of columns, so r[1] is the label and r[2] is the statement.
            if len(r) >= 3 and r[1] in LABELS:
                rows.append({"id": r[0], "label": r[1], "statement": r[2]})
    return rows


# ---- stratified_sample ----------------------------------------------------------
# WHAT: pick the same number of statements from each of the 6 labels.
# EXAMPLE: n=20 -> per_label = 20 // 6 = 3 -> 3 x 6 = 18 statements (rounded down to a multiple of 6).
def stratified_sample(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Pick ~n rows spread evenly over the 6 labels (fixed seed = repeatable).

    Stratified so rare classes (pants-fire) are exercised too: a plain random sample of 20
    could miss them and hide parse problems for those labels. n is rounded down to a
    multiple of 6.
    """
    # Our own random generator with a fixed seed: same picks every time, anywhere.
    rng = random.Random(seed)
    # Group the rows by label: {"true": [rows...], "false": [rows...], ...}
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    # "//" is integer division. max(1, ...) guarantees at least one row per label.
    per_label = max(1, n // len(LABELS))
    picked = []
    for label in LABELS:
        # rng.sample picks WITHOUT repeats; min(...) avoids asking for more rows than exist.
        picked += rng.sample(by_label[label], min(per_label, len(by_label[label])))
    return picked


# ---- make_generate_fn -----------------------------------------------------------
# WHAT: load the model and the tokenizer once, then return generate_fn(messages) -> text.
# The inner function "remembers" tok and model even after this outer function ends. This is a
# "closure" and it is why loading happens only once.
def make_generate_fn(model_id: str, use_4bit: bool):
    """Load the model once and return generate_fn(messages) -> str.

    Loading once matters: reloading a 7B model per statement would dominate the run time.
    """
    # Imported here so the parser/unit tests don't need torch installed.
    # (torch = the deep-learning library; transformers = Hugging Face's model library.)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Downloads (the first time) and loads the tokenizer that matches this model.
    tok = AutoTokenizer.from_pretrained(model_id)
    # Extra loading options. Empty for normal loading; filled in below for 4-bit.
    kwargs = {}
    if use_4bit:
        # 4-bit quantization shrinks a 7B model to ~5-6 GB of VRAM.
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            # fp4 is the bitsandbytes default; set explicitly so the label saved in the results
            # JSON ("4bit-fp4") cannot drift from what was actually used.
            # load_in_4bit=True        store the weights in 4 bits
            # bnb_4bit_quant_type      which 4-bit format (fp4)
            # bnb_4bit_compute_dtype   do the maths in float16 (weights are stored small, but
            #                          calculated at higher precision for better quality)
            load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.float16
        )
        # "auto" lets the library place the model on the available GPU(s) by itself.
        kwargs["device_map"] = "auto"
    # **kwargs unpacks the dictionary into named options for this call.
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    # eval() switches the model to prediction mode (turns off training-only behaviour like dropout).
    model.eval()
    # num_parameters() unpacks 4-bit weights (two per stored element); a plain sum would halve them.
    n_params = model.num_parameters()

    # ---- the function we hand back ----
    def generate_fn(messages: list[dict]) -> str:
        # Apply the model's own chat template so the system/user roles are right.
        # tokenize=False -> return text, not numbers yet.
        # add_generation_prompt=True -> add the marker that says "now the assistant speaks".
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Turn the text into number tensors ("pt" = PyTorch) and move them to the model's device (GPU).
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        # no_grad = do not record information for training: saves memory and time when only predicting.
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=12,  # a label is only a few tokens; cap rambling
                do_sample=False,  # greedy = deterministic, repeatable runs
                pad_token_id=tok.eos_token_id,
            )
        # Decode only the newly generated tokens, not the prompt.
        # out[0] is the whole sequence (prompt + answer). The slice [prompt_length:] keeps only the
        # answer. skip_special_tokens=True drops markers like end-of-text from the final string.
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Read by evaluate_llm for the results JSON (4-bit packs weights, so this is approximate).
    # Functions are objects in Python, so we can attach an extra value to this one.
    generate_fn.n_params = n_params
    return generate_fn


# ---- main (command-line entry) --------------------------------------------------
# WHAT: run the quick check and print the invalid rate and accuracy.
def main() -> None:
    """CLI: classify a small stratified sample and print the invalid rate and accuracy."""
    # Declare the command-line options.
    ap =argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to LIAR test.tsv")
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--n", type=int, default=20)
    # "--4bit" cannot be a Python attribute name (it starts with a digit), so dest= renames it.
    # action="store_true" makes it a simple on/off switch.
    ap.add_argument("--4bit", dest="use_4bit", action="store_true")
    args = ap.parse_args()

    # Pick the sample, then load the model once.
    sample = stratified_sample(load_liar_tsv(args.data), args.n)
    gen = make_generate_fn(args.model, args.use_4bit)

    # Keep the last raw model output per call so we can show it for bad parses.
    # A dictionary is used because the inner function below can change its contents
    # without needing the "nonlocal" keyword.
    raw = {}

    def recording_gen(messages):
        # Call the real model, remember what it said, and pass the text on unchanged.
        raw["last"] = gen(messages)
        return raw["last"]

    # Two counters; "correct = invalid = 0" sets both to zero in one line.
    correct = invalid = 0
    start = time.time()
    for r in sample:
        pred = classify(r["statement"], recording_gen)
        # (pred == INVALID) is True or False, and Python adds True as 1, False as 0.
        invalid += pred == INVALID
        correct += pred == r["label"]
        # Never print the full statement into logs; a short prefix is enough to debug.
        # Format tricks: {x:<12} = left-align in 12 spaces; !r = show with quotes; [:40] = first 40 chars.
        print(f"true={r['label']:<12} pred={pred:<12} raw={raw['last']!r:<30} {r['statement'][:40]!r}")

    n = len(sample)
    # {x:.0%} shows a fraction as a percentage with no decimals (0.05 -> 5%).
    print(f"\nn={n}  invalid={invalid} ({invalid / n:.0%})  accuracy={correct / n:.0%}  "
          f"avg_latency={(time.time() - start) / n:.1f}s/sample")


# Run main() only when this file is started directly, not when another file imports it.
if __name__ == "__main__":
    main()

# EXTRA VIVA QUESTIONS (whole file)
#   Q: Why 4-bit and not the full 16-bit model?
#   A: A T4 GPU has about 16 GB. The 7B model in 16-bit does not fit comfortably, in 4-bit it does.
#   Q: Why does the smoke test use only 20 statements?
#   A: It only checks that the output can be parsed. A full run takes much longer.
#   Q: Why is the sample "stratified"?
#   A: So rare labels such as pants-fire are included and any parsing problem for them shows up.
#   Q: What does max_new_tokens=12 do?
#   A: It stops the model after 12 new tokens. A label needs only a few, so this prevents rambling.
#   Q: Why set a random seed?
#   A: So the same statements are picked every time and results can be repeated.
