"""Strict parser for LLM label output.

The LLM is asked to answer with ONLY a label, but models often add extra
words ("The label is false.") or ramble. This parser accepts an answer only
if it contains exactly one valid label; anything else is treated as invalid
so we never silently guess and inflate the accuracy.
"""

# =====================================================================================
# FILE GUIDE (plain-English summary, added for understanding and viva preparation)
#
# THE PROBLEM
#   We ask the LLM for one label, but it may reply "The label is false." or "true or false"
#   or "I cannot determine that". The computer needs exactly one of our six labels.
#
# THE RULE (strict, on purpose)
#   Accept the answer ONLY if exactly ONE valid label is found. Otherwise mark it
#   INVALID, and invalid counts as WRONG. We never guess what the model "probably meant".
#   Why: guessing would quietly raise the accuracy and make the LLM look better than it is.
#
# EXAMPLES (what parse_output does)
#   "PANTS-FIRE!!!"        -> "pants-fire"   (capitals and punctuation are fine)
#   "Pants on fire"        -> "pants-fire"   (spelling variant is normalised first)
#   "The label is false."  -> "false"        (extra words are fine if one label is there)
#   "mostly-true"          -> "mostly-true"  (NOT "true": whole-word matching, see below)
#   "true or false"        -> ParseError     (two labels: the model hedged)
#   "not true"             -> ParseError     (negation: unclear which label was meant)
#   ""                     -> ParseError     (nothing to read)
#
# VIVA QUESTIONS FOR THIS FILE (with short answers)
#   Q: Why not just check "if 'true' in text"?
#   A: "true" is also inside "mostly-true" and "half-true". That would give wrong labels.
#   Q: Why is an invalid answer counted as wrong, not skipped?
#   A: Skipping would hide failures and inflate accuracy. Counting it as wrong is the fair way.
#   Q: What is a regular expression (regex)?
#   A: A small pattern language for finding text, like a very powerful Ctrl+F.
# =====================================================================================

# "re" is Python's regular-expression module (pattern matching on text).
import re

# The six allowed labels, in the shared order.
from llm.prompt import LABELS

# Sentinel stored as the prediction when the output can't be parsed.
# It is counted and reported as the "invalid output rate" in results JSON.
INVALID = "invalid"

# Words that flip the meaning of a label ("not true"). We reject these
# instead of guessing which label the model really meant.
# Regex reading: \b = word edge, (a|b|c) = "a or b or c". So it finds the whole words
#   not / isn't / isnt / never / no.   \b stops "no" matching inside "know" or "note".
# re.compile builds the pattern once (faster than rebuilding it for every statement).
# Limitation (good to admit in a viva): it does not list every negation, e.g. "doesn't".
_NEGATIONS = re.compile(r"\b(not|isn't|isnt|never|no)\b")


# Our own error type. It inherits from ValueError, so "except ValueError" also catches it,
# but "except ParseError" lets inference.py react only to unreadable answers.
class ParseError(ValueError):
    """Output did not contain exactly one valid label."""


# ---- _normalize -----------------------------------------------------------------
# WHAT: rewrite spelling variants into the exact label spelling.
# WHY : models write "pants on fire" or "half true", but our labels use hyphens.
# The leading underscore means "internal helper, used only inside this file".
def _normalize(text: str) -> str:
    """Lowercase and map spelling variants onto the canonical label strings."""
    # Lowercase so "TRUE", "True" and "true" are the same.
    text = text.lower()
    # "pants on fire" / "pants fire" / "pants_fire" -> "pants-fire"
    # Regex reading: pants, then [\s_-]+ (one or more spaces/underscores/hyphens), then an
    #   optional "on" plus separators, then fire. re.sub replaces the whole match.
    text = re.sub(r"pants[\s_-]+(on[\s_-]+)?fire", "pants-fire", text)
    # "half true" / "half_true" -> "half-true" (same for barely / mostly)
    # (barely|half|mostly) is saved as group 1. In the replacement, \1 means "put group 1
    # back", so "half true" becomes "half" + "-true". \b keeps whole words only.
    text = re.sub(r"\b(barely|half|mostly)[\s_-]+true\b", r"\1-true", text)
    return text


# ---- parse_output ---------------------------------------------------------------
# WHAT: the main function. Text in -> exactly one label out, or ParseError.
# STEPS: (1) reject empty  (2) reject negation  (3) normalise spelling
#        (4) find every label as a whole word  (5) accept only if exactly one was found
def parse_output(text: str) -> str:
    """Return the single label found in `text`, or raise ParseError.

    Rejects empty output, several different labels, and negated output
    ("not true"). Labels are matched as whole tokens so "true" does not
    match inside "mostly-true".
    """
    # Non-string or blank output (e.g. the model returned nothing).
    # "not isinstance(text, str)" also protects against None.
    if not isinstance(text, str) or not text.strip():
        raise ParseError("empty output")

    # Negation makes the answer ambiguous, so refuse it outright.
    # text[:60] shows only the first 60 characters in the error; !r shows it in quotes.
    if _NEGATIONS.search(text.lower()):
        raise ParseError(f"ambiguous (negation): {text[:60]!r}")

    # Spelling variants are fixed so "Pants on fire" can match the label "pants-fire".
    norm = _normalize(text)

    # Collect every label that appears as a whole token. The lookbehind and
    # lookahead forbid a word character or hyphen on either side, which is
    # what stops "true" from matching inside "mostly-true" or "half-true".
    # Regex reading:
    #   (?<![\w-])  "the character before must NOT be a letter, digit, _ or hyphen"
    #   (?![\w-])   "the character after must NOT be a letter, digit, _ or hyphen"
    #   re.escape   treats the label's hyphen as a plain character, not as regex syntax
    #   rf"..."     "r" = raw string (backslashes kept), "f" = fill in {label}
    # Example: in "mostly-true", the label "true" is preceded by "-", so it does NOT match.
    # The curly brackets build a set, so the same label found twice still counts once.
    found = {
        label
        for label in LABELS
        if re.search(rf"(?<![\w-]){re.escape(label)}(?![\w-])", norm)
    }

    # Zero labels = nonsense; two or more = the model hedged ("true or false").
    # sorted(found) just prints the labels in a stable order inside the error message.
    if len(found) != 1:
        raise ParseError(f"expected exactly one label, found {sorted(found)}: {text[:60]!r}")
    # A set has no order, so .pop() simply takes out its only item: the answer.
    return found.pop()

# EXTRA VIVA QUESTIONS (whole file)
#   Q: Why are negations checked on the ORIGINAL text, before normalising?
#   A: Normalising only fixes spelling, so it cannot create or hide a negation word.
#   Q: What is a "lookbehind" / "lookahead"?
#   A: A check of the characters just before/after a spot, without including them in the match.
#   Q: What are the limits of this parser?
#   A: It rejects some correct answers (such as "No evidence, so false") and does not cover
#      every negation word. We accept that: a false "invalid" is safer than a wrong guess.
#   Q: How is the invalid rate reported?
#   A: classify() returns INVALID, evaluate_llm counts them, and the results JSON stores
#      "invalid_outputs" (for example 1 out of 300 for zero-shot).
