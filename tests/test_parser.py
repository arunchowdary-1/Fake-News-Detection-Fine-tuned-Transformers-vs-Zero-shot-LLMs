"""Unit tests for the LLM prompt builder, strict parser and classify() loop.

These need no GPU or model: classify() takes a plain function, so we pass in
fake "models" that return canned strings.
"""

# =====================================================================================
# FILE GUIDE (plain-English summary, added for understanding and viva preparation)
#
# WHAT A UNIT TEST IS
#   A tiny program that feeds a function a known input and checks the output is what we expect.
#   If someone changes the code later and breaks it, the test fails and warns us.
#   Run all tests with:  python -m pytest tests/ -q
#
# THE TRICK USED HERE: FAKE MODELS
#   classify() needs a model, but a real 7B model needs a GPU. So we pass a tiny function
#   instead, for example  lambda m: "pants-fire"  (a "model" that always answers pants-fire).
#   That lets us test all the logic in milliseconds on any laptop.
#
# VIVA QUESTIONS FOR THIS FILE (with short answers)
#   Q: Why do we write tests?      A: To catch bugs early and to be safe when changing code.
#   Q: What is a "fake" (mock)?    A: A simple stand-in for something heavy, such as the LLM.
#   Q: What is a regression test?  A: A test added after a bug, to make sure it never returns.
#   Q: What does parametrize do?   A: It runs one test function many times with different inputs.
# =====================================================================================

# pytest is the testing tool. "pytest.raises" checks that an error IS raised.
import pytest

from llm.inference import classify
from llm.parser import INVALID, ParseError, parse_output
from llm.prompt import LABELS, build_messages


# Every canonical label must round-trip unchanged.
# @pytest.mark.parametrize runs this test 6 times, once for each label in LABELS.
# Example: parse_output("pants-fire") must give back exactly "pants-fire".
@pytest.mark.parametrize("label", LABELS)
def test_exact_labels(label):
    assert parse_output(label) == label


# Real models add punctuation, prefixes and spelling variants. These should
# still resolve to exactly one label.
# Each pair below is (what the model said, the label we expect). One test run per pair.
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
# This is the classic bug: a naive search for "true" would also fire inside "mostly-true".
def test_true_does_not_match_inside_compound_label():
    assert parse_output("mostly-true") == "mostly-true"
    assert parse_output("half-true") == "half-true"


# Anything with no label, several labels, or a negation must be rejected
# rather than guessed.
# These inputs must all raise ParseError: empty, spaces only, no label, two labels, a negation,
# an unrelated sentence, and None.
@pytest.mark.parametrize(
    "text",
    ["", "   ", "unknown label", "true or false", "half-true, maybe mostly-true",
     "not true", "I cannot determine that", None],
)
def test_invalid(text):
    # "with pytest.raises(...)" passes only if the code inside raises that error.
    with pytest.raises(ParseError):
        parse_output(text)


# Happy path: the first answer parses.
# The fake model is a one-line function that always says "pants-fire".
def test_classify_ok():
    assert classify("The earth is flat.", lambda m: "pants-fire") == "pants-fire"


# First answer is junk, the retry returns a valid label.
# iter([...]) hands out one item per call, so the fake model says "no idea" first, then "false".
def test_classify_retries_once_then_succeeds():
    outputs = iter(["no idea", "false"])
    assert classify("x", lambda m: next(outputs), max_retries=1) == "false"


# Persistent junk -> "invalid" after exactly 2 attempts (1 try + 1 explicit retry).
# "calls" records every time the fake model is used, so we can count the attempts.
def test_classify_marks_invalid_after_retry():
    calls = []

    def gen(m):
        calls.append(1)
        return "garbage"

    assert classify("x", gen, max_retries=1) == INVALID
    assert len(calls) == 2


# Default is no retry (greedy decoding would just repeat the answer).
def test_classify_default_makes_one_attempt():
    calls = []

    def gen(m):
        calls.append(1)
        return "garbage"

    assert classify("x", gen) == INVALID
    assert len(calls) == 1


# A blank or over-long statement is one invalid sample; it must not raise.
# 2000 characters is over the 1000-character limit in prompt.py.
def test_classify_bad_statement_is_invalid_not_a_crash():
    assert classify("   ", lambda m: "true") == INVALID
    assert classify("x" * 2000, lambda m: "true") == INVALID


# A crashing model (e.g. out of memory) must not abort the whole run.
# The fake model raises an error on purpose, like a real GPU running out of memory.
def test_classify_survives_model_errors():
    def boom(m):
        raise RuntimeError("CUDA OOM")

    assert classify("x", boom) == INVALID


# Input validation plus the shape of the chat messages sent to the model.
# A zero-shot chat has exactly two messages: system (the rules) and user (the statement).
def test_build_messages_validation():
    with pytest.raises(ValueError):
        build_messages("   ")  # blank
    with pytest.raises(ValueError):
        build_messages("x" * 2000)  # over the length limit
    msgs = build_messages("Taxes went up.")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "Taxes went up." in msgs[1]["content"]

# EXTRA VIVA QUESTIONS (whole file)
#   Q: Why test that "mostly-true" is NOT read as "true"?
#   A: That mistake would silently change predictions and make the accuracy numbers wrong.
#   Q: Why must an over-long statement NOT crash classify()?
#   A: One bad row should count as one invalid sample, not stop a whole 300-statement run.
#   Q: How many tests does the whole project have?
#   A: 99 passed when we last ran them (python -m pytest tests/ -q).
