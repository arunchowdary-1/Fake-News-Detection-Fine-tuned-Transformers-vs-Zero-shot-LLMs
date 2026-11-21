"""Zero-shot prompt for LIAR 6-class truthfulness classification.

LABELS is duplicated here only until common/config.py lands (issue #2);
then import it from there so the label order has a single source of truth.
"""

LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]

# Model is NOT pinned here: the team has to approve it and check its license
# (Qwen2.5-7B-Instruct recommended; Mistral-7B-Instruct alternative).
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_MESSAGE = (
    "You are a fact-checking expert. Classify the given political statement "
    "by its truthfulness using exactly one of these labels:\n"
    "pants-fire (ridiculously false), false, barely-true, half-true, "
    "mostly-true, true\n\n"
    "Respond with ONLY the label, nothing else."
)

USER_TEMPLATE = "Statement: {statement}\nLabel:"

MAX_STATEMENT_CHARS = 1000


def build_messages(statement: str) -> list[dict]:
    """Return chat messages (system + user) for one statement."""
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("statement must be a non-empty string")
    if len(statement) > MAX_STATEMENT_CHARS:
        raise ValueError(f"statement exceeds {MAX_STATEMENT_CHARS} characters")
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_TEMPLATE.format(statement=statement.strip())},
    ]
