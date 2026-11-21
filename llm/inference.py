"""Model-agnostic classify loop: strict parse, retry once, else mark invalid.

`generate_fn(messages) -> str` wraps the real model. Keeping the model behind
a plain function lets us unit-test this logic without a GPU, and lets the
smoke test and the full run (#8) share the same code path.
"""
from typing import Callable

from llm.parser import INVALID, ParseError, parse_output
from llm.prompt import build_messages


def classify(
    statement: str,
    generate_fn: Callable[[list[dict]], str],
    max_retries: int = 1,
) -> str:
    """Return a label, or "invalid" after 1 + max_retries failed attempts."""
    # Validates the statement (non-empty, max length) before touching the model.
    messages = build_messages(statement)

    # First attempt plus `max_retries` more. Sampling can make a retry succeed
    # where the first answer was unparseable.
    for _ in range(max_retries + 1):
        try:
            return parse_output(generate_fn(messages))
        except ParseError:
            # Model answered, but not with exactly one label: try again.
            continue
        except Exception:
            # Model/runtime failure (e.g. CUDA out of memory): don't crash the
            # whole evaluation run, just count this sample as a failed attempt.
            continue

    # All attempts failed: record it so the invalid rate can be reported.
    return INVALID
