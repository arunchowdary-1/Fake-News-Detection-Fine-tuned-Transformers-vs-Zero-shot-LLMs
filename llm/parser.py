"""Strict parser for LLM label output."""
import re

from llm.prompt import LABELS

INVALID = "invalid"

_NEGATIONS = re.compile(r"\b(not|isn't|isnt|never|no)\b")


class ParseError(ValueError):
    """Output did not contain exactly one valid label."""


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"pants[\s_-]+(on[\s_-]+)?fire", "pants-fire", text)
    # "half true" / "half_true" -> "half-true"
    text = re.sub(r"\b(barely|half|mostly)[\s_-]+true\b", r"\1-true", text)
    return text


def parse_output(text: str) -> str:
    """Return the single label found in `text`, or raise ParseError.

    Rejects empty output, several different labels, and negated output
    ("not true"). Labels are matched as whole tokens so "true" does not
    match inside "mostly-true".
    """
    if not isinstance(text, str) or not text.strip():
        raise ParseError("empty output")
    if _NEGATIONS.search(text.lower()):
        raise ParseError(f"ambiguous (negation): {text[:60]!r}")
    norm = _normalize(text)
    found = {
        label
        for label in LABELS
        if re.search(rf"(?<![\w-]){re.escape(label)}(?![\w-])", norm)
    }
    # "true"/"false" inside a compound label never counts on its own: the
    # lookarounds above already exclude "-true" / "true-" neighbours.
    if len(found) != 1:
        raise ParseError(f"expected exactly one label, found {sorted(found)}: {text[:60]!r}")
    return found.pop()
