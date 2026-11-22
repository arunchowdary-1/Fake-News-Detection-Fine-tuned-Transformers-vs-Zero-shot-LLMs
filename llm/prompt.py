"""Zero-shot prompt for LIAR 6-class truthfulness classification.

LABELS comes from common/config.py so the label order has one source of truth.
"""

# =====================================================================================
# FILE GUIDE (plain-English summary, added for understanding and viva preparation)
#
# WHAT THIS FILE DOES
#   An LLM does not take a "statement" directly. It takes a short chat: a SYSTEM message
#   (the rules), then USER / ASSISTANT turns. This file builds that chat for one
#   statement. That is all "prompting" means here.
#
# THE TWO MODES
#   zero-shot : rules + the statement                 (no examples)
#   few-shot  : rules + 6 solved examples + statement (examples teach the answer format)
#
# EXAMPLE OF WHAT build_messages("Taxes went up.") RETURNS
#   [ {"role": "system", "content": "You are a fact-checking expert. ..."},
#     {"role": "user",   "content": "Statement: Taxes went up.\nLabel:"} ]
#
# VIVA QUESTIONS FOR THIS FILE (with short answers)
#   Q: What is a prompt?           A: The text we give an LLM so it knows the task.
#   Q: Zero-shot vs few-shot?      A: Zero-shot gives only instructions. Few-shot also
#                                     gives a few solved examples inside the prompt.
#   Q: Is the model trained here?  A: No. We never change the model's weights. We only
#                                     change the text we send it.
#   Q: Why a "system" message?     A: It sets the model's role and rules for the whole chat.
#   Q: Why force "ONLY the label"? A: A short answer is easy to check with the parser.
# =====================================================================================

# Loads the shared label order from the common module (so every model uses the same one).
from common.config import LABEL_TO_IDX

# Single source of truth: the shared label map (dict order is index order 0..5).
# list(dict) gives the KEYS in insertion order:
#   ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
LABELS = list(LABEL_TO_IDX)

# Model is NOT pinned here: the team has to approve it and check its license
# (Qwen2.5-7B-Instruct recommended; Mistral-7B-Instruct alternative).
# This is the Hugging Face model name; "7B" means about 7 billion parameters.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# System message: defines the task and forces a label-only answer, which keeps
# the output short and easy for the strict parser to check.
# The brackets after the strings are just Python joining neighbouring strings into one.
# "\n" = a new line inside the message.
SYSTEM_MESSAGE = (
    "You are a fact-checking expert. Classify the given political statement "
    "by its truthfulness using exactly one of these labels:\n"
    "pants-fire (ridiculously false), false, barely-true, half-true, "
    "mostly-true, true\n\n"
    "Respond with ONLY the label, nothing else."
)

# User turn: just the statement, ending in "Label:" so the model completes it.
# "{statement}" is a blank that .format() fills in later.
# Example: USER_TEMPLATE.format(statement="Taxes went up.") -> "Statement: Taxes went up.\nLabel:"
USER_TEMPLATE = "Statement: {statement}\nLabel:"

# Same limit as the API's input validation (#9) so both reject the same inputs.
# 1000 characters is far more than any LIAR statement (the longest is about 400).
MAX_STATEMENT_CHARS = 1000


# ---- build_messages -------------------------------------------------------------
# WHAT: turns one statement (and optional solved examples) into the chat list for the LLM.
# WHY : the same function serves zero-shot (examples=None) and few-shot (examples=[...]).
# VIVA Q: What do the type hints mean?
#   A: "list[dict] | None" says "a list of dictionaries, or None". They are hints for
#      readers and tools; Python does not enforce them.
def build_messages(statement: str, examples: list[dict] | None = None) -> list[dict]:
    """Return chat messages (system + optional few-shot turns + user) for one statement.

    `examples` is a list of {"statement": str, "label": str} dicts (see llm/fewshot.py).
    Each one becomes a user/assistant turn in the same format as the real question,
    so the model sees exactly what answer shape we want. None/empty = zero-shot.
    """
    # Validate before building anything so bad input fails with a clear error.
    # Rejects: not text, empty, or only spaces ("   ".strip() is "", which counts as False).
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("statement must be a non-empty string")
    # Rejects very long input so a huge text cannot waste time or memory.
    if len(statement) > MAX_STATEMENT_CHARS:
        raise ValueError(f"statement exceeds {MAX_STATEMENT_CHARS} characters")
    # The chat always starts with the system message (the rules).
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    # "examples or []" means: if examples is None, loop over an empty list (zero-shot).
    for ex in examples or []:
        # A typo in an example label would teach the model a wrong answer, so stop early.
        if ex["label"] not in LABELS:
            raise ValueError(f"unknown few-shot label {ex['label']!r}")
        # Each solved example is TWO turns: the user asks, the assistant answers correctly.
        # The model copies this pattern for the real statement. This is "in-context learning".
        messages.append({"role": "user", "content": USER_TEMPLATE.format(statement=ex["statement"].strip())})
        messages.append({"role": "assistant", "content": ex["label"]})
    # Finally the real statement we want rated. The model continues after "Label:".
    # .strip() removes spaces at the ends. .format() only fills the template's blank, so
    # curly brackets inside the statement itself cannot break it.
    messages.append({"role": "user", "content": USER_TEMPLATE.format(statement=statement.strip())})
    return messages

# EXTRA VIVA QUESTIONS (whole file)
#   Q: Why are the few-shot examples taken from the TRAIN set?
#   A: So no example can also be a test statement (that would be cheating).
#   Q: Why do the examples use the same format as the real question?
#   A: The model copies the pattern, so its answer comes back in the shape we can parse.
#   Q: What happens if the statement is empty?
#   A: build_messages raises ValueError. classify() in inference.py catches it and counts
#      the sample as "invalid" instead of crashing the whole run.
