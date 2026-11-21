"""Strict parser for LLM label output.

The LLM is asked to answer with ONLY a label, but models often add extra
words ("The label is false.") or ramble. This parser accepts an answer only
if it contains exactly one valid label; anything else is treated as invalid
so we never silently guess and inflate the accuracy.
"""
import re

from llm.prompt import LABELS

# Sentinel stored as the prediction when the output can't be parsed.
# It is counted and reported as the "invalid output rate" in results JSON.
INVALID = "invalid"

# Words that flip the meaning of a label ("not true"). We reject these
# instead of guessing which label the model really meant.
_NEGATIONS = re.compile(r"\b(not|isn't|isnt|never|no)\b")


class ParseError(ValueError):
    """Output did not contain exactly one valid label."""


def _normalize(text: str) -> str:
    """Lowercase and map spelling variants onto the canonical label strings."""
    text = text.lower()
    # "pants on fire" / "pants fire" / "pants_fire" -> "pants-fire"
    text = re.sub(r"pants[\s_-]+(on[\s_-]+)?fire", "pants-fire", text)
    # "half true" / "half_true" -> "half-true" (same for barely / mostly)
    text = re.sub(r"\b(barely|half|mostly)[\s_-]+true\b", r"\1-true", text)
    return text


def parse_output(text: str) -> str:
    """Return the single label found in `text`, or raise ParseError.

    Rejects empty output, several different labels, and negated output
    ("not true"). Labels are matched as whole tokens so "true" does not
    match inside "mostly-true".
    """
    # Non-string or blank output (e.g. the model returned nothing).
    if not isinstance(text, str) or not text.strip():
        raise ParseError("empty output")

    # Negation makes the answer ambiguous, so refuse it outright.
    if _NEGATIONS.search(text.lower()):
        raise ParseError(f"ambiguous (negation): {text[:60]!r}")

    norm = _normalize(text)

    # Collect every label that appears as a whole token. The lookbehind and
    # lookahead forbid a word character or hyphen on either side, which is
    # what stops "true" from matching inside "mostly-true" or "half-true".
    found = {
        label
        for label in LABELS
        if re.search(rf"(?<![\w-]){re.escape(label)}(?![\w-])", norm)
    }

    # Zero labels = nonsense; two or more = the model hedged ("true or false").
    if len(found) != 1:
        raise ParseError(f"expected exactly one label, found {sorted(found)}: {text[:60]!r}")
    return found.pop()
