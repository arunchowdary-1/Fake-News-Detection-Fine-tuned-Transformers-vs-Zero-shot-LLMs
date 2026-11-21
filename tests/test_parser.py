"""Unit tests for the LLM prompt builder, strict parser and classify() loop.

These need no GPU or model: classify() takes a plain function, so we pass in
fake "models" that return canned strings.
"""
import pytest

from llm.inference import classify
from llm.parser import INVALID, ParseError, parse_output
from llm.prompt import LABELS, build_messages


# Every canonical label must round-trip unchanged.
@pytest.mark.parametrize("label", LABELS)
def test_exact_labels(label):
    assert parse_output(label) == label


# Real models add punctuation, prefixes and spelling variants. These should
# still resolve to exactly one label.
@pytest.mark.parametrize(
    "text,expected",
    [
        ("PANTS-FIRE!!!", "pants-fire"),
        ("  True.  ", "true"),
        ("Label: mostly-true", "mostly-true"),
        ("half true", "half-true"),
        ("Pants on fire", "pants-fire"),
        ('"barely_true"', "barely-true"),
        ("The label is false.", "false"),
    ],
)
def test_messy_but_unambiguous(text, expected):
    assert parse_output(text) == expected


# Regression guard: "true" must not be picked out of a compound label.
def test_true_does_not_match_inside_compound_label():
    assert parse_output("mostly-true") == "mostly-true"
    assert parse_output("half-true") == "half-true"


# Anything with no label, several labels, or a negation must be rejected
# rather than guessed.
@pytest.mark.parametrize(
    "text",
    ["", "   ", "unknown label", "true or false", "half-true, maybe mostly-true",
     "not true", "I cannot determine that", None],
)
def test_invalid(text):
    with pytest.raises(ParseError):
        parse_output(text)


# Happy path: the first answer parses.
def test_classify_ok():
    assert classify("The earth is flat.", lambda m: "pants-fire") == "pants-fire"


# First answer is junk, the retry returns a valid label.
def test_classify_retries_once_then_succeeds():
    outputs = iter(["no idea", "false"])
    assert classify("x", lambda m: next(outputs)) == "false"


# Persistent junk -> "invalid" after exactly 2 attempts (1 try + 1 retry).
def test_classify_marks_invalid_after_retry():
    calls = []

    def gen(m):
        calls.append(1)
        return "garbage"

    assert classify("x", gen) == INVALID
    assert len(calls) == 2


# A crashing model (e.g. out of memory) must not abort the whole run.
def test_classify_survives_model_errors():
    def boom(m):
        raise RuntimeError("CUDA OOM")

    assert classify("x", boom) == INVALID


# Input validation plus the shape of the chat messages sent to the model.
def test_build_messages_validation():
    with pytest.raises(ValueError):
        build_messages("   ")  # blank
    with pytest.raises(ValueError):
        build_messages("x" * 2000)  # over the length limit
    msgs = build_messages("Taxes went up.")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "Taxes went up." in msgs[1]["content"]
