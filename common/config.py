"""
Shared configuration and constants for the Fake News Detection project.
"""

# Label mapping (fixed order across all models)
# Ordered from least to most truthful. This is NOT the Hugging Face "liar" class order;
# data_loader.label_to_idx() translates HF integer ids into this order so every model,
# metric and confusion matrix shares one index scheme.
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
# Every model is scored on the same fixed stratified 300-sample slice of the test set,
# which keeps the (slow, costly) LLM runs comparable to the fine-tuned models.
TEST_SUBSET_SIZE = 300
RANDOM_SEED = 42

# Model training defaults
# Token budget for the statement alone; the metadata-prefixed variant uses 2x this
# (see finetune/train_bert.py) because the speaker/party/subject/history prefix is long.
MAX_LENGTH_TEXT = 128
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL = 32
NUM_EPOCHS_DEFAULT = 3
LEARNING_RATE_DEFAULT = 2e-5  # typical BERT fine-tuning rate
