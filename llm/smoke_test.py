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
import argparse
import csv
import random
import time
from collections import defaultdict

from llm.inference import classify
from llm.parser import INVALID
from llm.prompt import DEFAULT_MODEL_ID, LABELS


def load_liar_tsv(path: str) -> list[dict]:
    """Read the original LIAR TSV (no header). Only id, label, statement are used."""
    rows = []  # (the shared loader in common/data_loader.py is used for the real run)
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.reader(f, delimiter="\t"):
            # Skip malformed lines and any label we don't know about.
            if len(r) >= 3 and r[1] in LABELS:
                rows.append({"id": r[0], "label": r[1], "statement": r[2]})
    return rows


def stratified_sample(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Pick ~n rows spread evenly over the 6 labels (fixed seed = repeatable).

    Stratified so rare classes (pants-fire) are exercised too: a plain random sample of 20
    could miss them and hide parse problems for those labels. n is rounded down to a
    multiple of 6.
    """
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    per_label = max(1, n // len(LABELS))
    picked = []
    for label in LABELS:
        picked += rng.sample(by_label[label], min(per_label, len(by_label[label])))
    return picked


def make_generate_fn(model_id: str, use_4bit: bool):
    """Load the model once and return generate_fn(messages) -> str.

    Loading once matters: reloading a 7B model per statement would dominate the run time.
    """
    # Imported here so the parser/unit tests don't need torch installed.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    kwargs = {}
    if use_4bit:
        # 4-bit quantization shrinks a 7B model to ~5-6 GB of VRAM.
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            # fp4 is the bitsandbytes default; set explicitly so the label saved in the results
            # JSON ("4bit-fp4") cannot drift from what was actually used.
            load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.float16
        )
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    # num_parameters() unpacks 4-bit weights (two per stored element); a plain sum would halve them.
    n_params = model.num_parameters()

    def generate_fn(messages: list[dict]) -> str:
        # Apply the model's own chat template so the system/user roles are right.
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=12,  # a label is only a few tokens; cap rambling
                do_sample=False,  # greedy = deterministic, repeatable runs
                pad_token_id=tok.eos_token_id,
            )
        # Decode only the newly generated tokens, not the prompt.
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Read by evaluate_llm for the results JSON (4-bit packs weights, so this is approximate).
    generate_fn.n_params = n_params
    return generate_fn


def main() -> None:
    """CLI: classify a small stratified sample and print the invalid rate and accuracy."""
    ap =argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to LIAR test.tsv")
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--4bit", dest="use_4bit", action="store_true")
    args = ap.parse_args()

    sample = stratified_sample(load_liar_tsv(args.data), args.n)
    gen = make_generate_fn(args.model, args.use_4bit)

    # Keep the last raw model output per call so we can show it for bad parses.
    raw = {}

    def recording_gen(messages):
        raw["last"] = gen(messages)
        return raw["last"]

    correct = invalid = 0
    start = time.time()
    for r in sample:
        pred = classify(r["statement"], recording_gen)
        invalid += pred == INVALID
        correct += pred == r["label"]
        # Never print the full statement into logs; a short prefix is enough to debug.
        print(f"true={r['label']:<12} pred={pred:<12} raw={raw['last']!r:<30} {r['statement'][:40]!r}")

    n = len(sample)
    print(f"\nn={n}  invalid={invalid} ({invalid / n:.0%})  accuracy={correct / n:.0%}  "
          f"avg_latency={(time.time() - start) / n:.1f}s/sample")


if __name__ == "__main__":
    main()
