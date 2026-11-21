"""Zero-shot prompt for LIAR 6-class truthfulness classification.

LABELS comes from common/config.py so the label order has one source of truth.
"""

from common.config import LABEL_TO_IDX

# Single source of truth: the shared label map (dict order is index order 0..5).
LABELS = list(LABEL_TO_IDX)

# Model is NOT pinned here: the team has to approve it and check its license
# (Qwen2.5-7B-Instruct recommended; Mistral-7B-Instruct alternative).
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# System message: defines the task and forces a label-only answer, which keeps
# the output short and easy for the strict parser to check.
SYSTEM_MESSAGE = (
    "You are a fact-checking expert. Classify the given political statement "
    "by its truthfulness using exactly one of these labels:\n"
    "pants-fire (ridiculously false), false, barely-true, half-true, "
    "mostly-true, true\n\n"
    "Respond with ONLY the label, nothing else."
)

# User turn: just the statement, ending in "Label:" so the model completes it.
USER_TEMPLATE = "Statement: {statement}\nLabel:"

# Same limit as the API's input validation (#9) so both reject the same inputs.
MAX_STATEMENT_CHARS = 1000


def build_messages(statement: str) -> list[dict]:
    """Return chat messages (system + user) for one statement."""
    # Validate before building anything so bad input fails with a clear error.
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("statement must be a non-empty string")
    if len(statement) > MAX_STATEMENT_CHARS:
        raise ValueError(f"statement exceeds {MAX_STATEMENT_CHARS} characters")
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_TEMPLATE.format(statement=statement.strip())},
    ]
