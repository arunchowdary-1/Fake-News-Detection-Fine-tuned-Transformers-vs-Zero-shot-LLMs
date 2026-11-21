"""Model-agnostic classify loop: strict parse, retry once, else mark invalid.

`generate_fn(messages) -> str` wraps the real model. The 4-bit Qwen/Mistral
loader and batch evaluation run belong to issue #8 and need a GPU.
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
    messages = build_messages(statement)
    for _ in range(max_retries + 1):
        try:
            return parse_output(generate_fn(messages))
        except ParseError:
            continue
        except Exception:  # model/runtime failure: don't crash the whole run
            continue
    return INVALID
