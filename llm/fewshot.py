"""Pick the few-shot examples: one training statement per class, fixed by seed 42.

Examples come from the TRAIN split only, so they can never overlap the shared
test subset. Pure function (no torch / datasets) so it is unit-testable.
"""
import numpy as np

from common.config import IDX_TO_LABEL, RANDOM_SEED

# Short-but-informative statements keep the few-shot prompt cheap.
MIN_CHARS, MAX_CHARS = 60, 200


def select_examples(statements: list[str], label_idxs: list[int], per_class: int = 1,
                    seed: int = RANDOM_SEED) -> list[dict]:
    """Return per_class examples for each of the 6 classes as [{statement, label}, ...].

    The order is shuffled (seeded) rather than sorted by label: a fixed
    pants-fire..true order would put "true" last, which biases small prompts
    toward answering with the last example's label.
    """
    if len(statements) != len(label_idxs):
        raise ValueError("statements and label_idxs must have the same length")
    rng = np.random.RandomState(seed)
    picked = []
    for idx, label in IDX_TO_LABEL.items():
        pool = [i for i, (s, l) in enumerate(zip(statements, label_idxs))
                if l == idx and MIN_CHARS <= len(s.strip()) <= MAX_CHARS]
        if len(pool) < per_class:  # very rare class + tight length filter: relax the filter
            pool = [i for i, l in enumerate(label_idxs) if l == idx]
        if len(pool) < per_class:
            raise ValueError(f"not enough training rows for class {label!r}")
        for i in rng.choice(pool, size=per_class, replace=False):
            picked.append({"statement": statements[int(i)].strip(), "label": label})
    order = rng.permutation(len(picked))
    return [picked[i] for i in order]
