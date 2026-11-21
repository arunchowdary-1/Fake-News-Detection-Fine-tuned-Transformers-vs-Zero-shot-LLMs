import pytest

from llm.inference import classify
from llm.parser import INVALID, ParseError, parse_output
from llm.prompt import LABELS, build_messages


@pytest.mark.parametrize("label", LABELS)
def test_exact_labels(label):
    assert parse_output(label) == label


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


def test_true_does_not_match_inside_compound_label():
    assert parse_output("mostly-true") == "mostly-true"
    assert parse_output("half-true") == "half-true"


@pytest.mark.parametrize(
    "text",
    ["", "   ", "unknown label", "true or false", "half-true, maybe mostly-true",
     "not true", "I cannot determine that", None],
)
def test_invalid(text):
    with pytest.raises(ParseError):
        parse_output(text)


def test_classify_ok():
    assert classify("The earth is flat.", lambda m: "pants-fire") == "pants-fire"


def test_classify_retries_once_then_succeeds():
    outputs = iter(["no idea", "false"])
    assert classify("x", lambda m: next(outputs)) == "false"


def test_classify_marks_invalid_after_retry():
    calls = []

    def gen(m):
        calls.append(1)
        return "garbage"

    assert classify("x", gen) == INVALID
    assert len(calls) == 2


def test_classify_survives_model_errors():
    def boom(m):
        raise RuntimeError("CUDA OOM")

    assert classify("x", boom) == INVALID


def test_build_messages_validation():
    with pytest.raises(ValueError):
        build_messages("   ")
    with pytest.raises(ValueError):
        build_messages("x" * 2000)
    msgs = build_messages("Taxes went up.")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "Taxes went up." in msgs[1]["content"]
