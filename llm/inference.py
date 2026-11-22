"""Model-agnostic classify loop: strict parse, optional retry, else mark invalid.

`generate_fn(messages) -> str` wraps the real model. Keeping the model behind
a plain function lets us unit-test this logic without a GPU, and lets the
smoke test and the full run (#8) share the same code path.
"""

# =====================================================================================
# FILE GUIDE (plain-English summary, added for understanding and viva preparation)
#
# WHAT THIS FILE DOES
#   It is the "classify one statement" routine. It glues three things together:
#       prompt.py  (build the chat)  ->  the model  ->  parser.py  (read the answer)
#   and it never lets one bad sample crash a long run.
#
# THE KEY IDEA: generate_fn
#   The model is hidden behind a plain function: generate_fn(messages) -> text.
#   - In real runs it is the 7B LLM on a GPU.
#   - In tests it is a tiny fake function such as  lambda m: "true".
#   So we can test all this logic on a laptop with no GPU. (This is called "dependency
#   injection": the thing we depend on is passed in instead of being hard-wired.)
#
# WHAT classify() RETURNS
#   one of the six labels, or "invalid". It never raises for a bad statement or a bad answer.
#
# VIVA QUESTIONS FOR THIS FILE (with short answers)
#   Q: Why pass generate_fn instead of loading the model inside classify()?
#   A: Easy testing without a GPU, and the smoke test and full run share the same code.
#   Q: Why is max_retries 0 by default?
#   A: We decode greedily (always the most likely word), so a retry gives the same answer.
#      It would only double the time.
#   Q: What is the difference between ParseError and another Exception here?
#   A: ParseError = the model answered but not with one clean label.
#      Other exceptions = the program or GPU failed (for example out of memory).
#   Q: Why count runtime errors separately?
#   A: Otherwise a broken run would just look like "100% invalid answers" and hide the cause.
# =====================================================================================

# sys gives access to stderr, where we print warnings (kept apart from normal output).
import sys
# Callable is only a type hint meaning "a function".
from typing import Callable

# INVALID = the marker string "invalid"; ParseError = our error; parse_output = text -> label.
from llm.parser import INVALID, ParseError, parse_output
# build_messages turns a statement (plus optional examples) into the chat for the model.
from llm.prompt import build_messages


# Runtime (non-parse) failures are rare; print the first few so a systematic problem
# (CUDA OOM, bad generate call) is visible instead of looking like a 100% invalid rate.
# Module-level counter: it lives for the whole run, so evaluate_llm can save the total.
RUNTIME_ERRORS = 0
# Print only the first 3 errors, so a problem on every sample does not flood the screen.
_MAX_LOGGED_ERRORS = 3


# ---- classify -------------------------------------------------------------------
# WHAT: statement in -> label (or "invalid") out.
# STEPS: build the chat -> ask the model -> parse the answer -> retry only if allowed.
def classify(
    statement: str,
    generate_fn: Callable[[list[dict]], str],
    max_retries: int = 0,
    examples: list[dict] | None = None,
) -> str:
    """Return a label, or "invalid" after 1 + max_retries failed attempts.

    max_retries defaults to 0: our decoding is greedy, so a retry would repeat
    the same answer and only double the latency. Use max_retries > 0 only with
    a sampling generate_fn. `examples` switches on few-shot prompting.
    """
    try:
        # Validates the statement (non-empty, max length) before touching the model.
        messages = build_messages(statement, examples)
    except ValueError:
        # A bad statement is one invalid sample, not a reason to abort a whole run.
        return INVALID

    # First attempt plus `max_retries` more.
    # With max_retries=0, range(1) runs exactly once. "_" = a loop variable we never use.
    for _ in range(max_retries + 1):
        try:
            # generate_fn(messages) -> raw text from the model; parse_output(...) -> one label.
            # If the label is clean we return it straight away and the loop ends.
            return parse_output(generate_fn(messages))
        except ParseError:
            # Model answered, but not with exactly one label: try again.
            # "continue" jumps to the next loop turn (if there are no turns left, we fall out).
            continue
        # Order matters: ParseError is a kind of Exception, so it must be caught FIRST.
        except Exception as err:
            # Model/runtime failure (e.g. CUDA out of memory): don't crash the
            # whole evaluation run, count this sample as a failed attempt, but say why.
            # The counter is module-level so evaluate_llm can save it in the results JSON.
            # "global" is needed because we CHANGE the module-level number inside a function.
            global RUNTIME_ERRORS
            RUNTIME_ERRORS += 1
            # Print only the first few errors: type(err).__name__ is the error's class name,
            # for example RuntimeError. We print the error text, never the statement itself.
            if RUNTIME_ERRORS <= _MAX_LOGGED_ERRORS:
                print(f"classify: runtime error #{RUNTIME_ERRORS}: {type(err).__name__}: {err}", file=sys.stderr)
            continue

    # All attempts failed: record it so the invalid rate can be reported.
    # We return the marker, not a guessed label, so the score is never inflated.
    return INVALID

# EXTRA VIVA QUESTIONS (whole file)
#   Q: What does "greedy decoding" mean?
#   A: At every step the model takes the single most likely next word. No randomness, so
#      the same input always gives the same output.
#   Q: What would change if we used sampling instead?
#   A: Answers would vary between runs, so a retry could succeed and max_retries > 0 makes sense.
#   Q: Why return a marker instead of raising an error for a bad statement?
#   A: One bad row among 300 must not stop the whole evaluation.
#   Q: What is shown if the GPU runs out of memory on every sample?
#   A: The first three error messages are printed, and the count is saved in the results
#      JSON as "runtime_errors", so the problem is easy to spot.
