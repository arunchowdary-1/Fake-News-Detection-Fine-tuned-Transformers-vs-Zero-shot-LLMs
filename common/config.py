"""
Shared configuration and constants for the Fake News Detection project.
"""

# Label mapping (fixed order across all models)
LABEL_TO_IDX = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

NUM_CLASSES = len(LABEL_TO_IDX)

# Data paths
DATA_DIR = "data/"
RESULTS_DIR = "results/"
CHECKPOINTS_DIR = "checkpoints/"

# Evaluation settings
TEST_SUBSET_SIZE = 300
RANDOM_SEED = 42

# Model training defaults
MAX_LENGTH_TEXT = 128
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL = 32
NUM_EPOCHS_DEFAULT = 3
LEARNING_RATE_DEFAULT = 2e-5
