"""
baselines/lstm_model.py  --  Issue #4 "[Model] LSTM baseline"
==============================================================

PURPOSE OF THIS FILE
    The project compares three kinds of models on the LIAR fake-news dataset:
        1. a small LSTM trained from scratch   <-- THIS FILE (the "baseline")
        2. fine-tuned BERT                      (issues #5 and #6)
        3. a zero-shot 7B LLM                   (issues #7 and #8)
    The LSTM is the cheapest and simplest model. It tells us how much accuracy
    we get WITHOUT any pre-trained language knowledge. If BERT or the LLM can't
    beat it by a clear margin, their extra cost isn't justified. That is the
    whole point of a baseline.

WHAT IS IN THIS FILE (only the model pieces, no training loop):
    tokenize()        -> turns a sentence into a list of lowercase words
    Vocab             -> remembers which word has which number
    pad_batch()       -> makes sentences of different length the same length
    BiLSTMClassifier  -> the neural network itself
    count_params()    -> counts trainable weights (needed for the results JSON)
The training loop lives in baselines/train.py.
"""

# `re` = regular expressions. We use it to split text into words.
import re
# `Counter` counts how often each word appears (used to build the vocabulary).
from collections import Counter
# Type hints only: they make the code easier to read, they don't change behaviour.
from typing import Dict, Iterable, List

# PyTorch: the deep-learning library that provides tensors and neural-network layers.
import torch
# `nn` holds the building blocks (Embedding, LSTM, Linear, Dropout ...).
import torch.nn as nn

# Special token #1: used to fill empty space so all sentences in a batch have equal length.
PAD_TOKEN = "<pad>"
# Special token #2: stands in for any word that is NOT in our vocabulary.
# Without it, a new word at test time would crash the lookup.
UNK_TOKEN = "<unk>"

# A regular expression that defines what counts as a "word" for this project:
#   [a-z0-9]+(?:'[a-z]+)?  -> letters/digits, optionally followed by an apostrophe part
#                             (so "don't" stays ONE token instead of "don" + "t")
#   |[$%]                  -> OR a single "$" or "%" symbol. We keep them because
#                             political claims are full of money and percentages
#                             ("spent $3 trillion", "cut taxes by 50%").
# We compile it once here (not inside the function) so it isn't rebuilt for every sentence.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[$%]")


def tokenize(text: str) -> List[str]:
    """
    Split one statement into a list of lowercase word tokens.

    WHY: a neural network can't read raw text, it needs numbers. Step one of
    that conversion is cutting the sentence into words ("tokens").
    WHY a simple regex instead of a fancy tokenizer: the LSTM baseline is meant
    to be simple and fast; BERT will use its own tokenizer later.
    """
    # Defensive check: if the dataset ever contains None/NaN instead of a string,
    # return an empty list instead of crashing the whole training run.
    if not isinstance(text, str):
        return []
    # Lowercase first so "Tax" and "tax" become the same word, then find all matches.
    return _TOKEN_RE.findall(text.lower())


class Vocab:
    """
    Dictionary that maps every known word to an integer id.

    WHY: the embedding layer of the network looks up a vector by integer id,
    so every word needs a number. Example: {"<pad>":0, "<unk>":1, "tax":2, ...}
    IMPORTANT: it is built from the TRAINING split only. If we also counted
    words from the test split, information from the test set would leak into
    training and our accuracy would look better than it really is.
    """

    def __init__(self, stoi: Dict[str, int]):
        # "stoi" = "string to index": the word -> id dictionary.
        self.stoi = stoi
        # Remember the ids of the two special tokens, since the model needs them.
        self.pad_idx = stoi[PAD_TOKEN]
        self.unk_idx = stoi[UNK_TOKEN]

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 2, max_size: int = 20000) -> "Vocab":
        """
        Build the vocabulary from a list of training sentences.

        min_freq=2 : ignore words seen only once. Words that appear once are
                     mostly typos/rare names; the model can't learn anything
                     useful from one example and they only add noise + size.
        max_size   : keep at most 20,000 words so the embedding table (and the
                     parameter count) stays small. "Small" is the point of a baseline.
        """
        # Tokenize every training sentence and count each word's frequency.
        counts = Counter(tok for t in texts for tok in tokenize(t))
        # most_common(max_size) gives the top words by frequency; then drop rare ones.
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        # Reserve ids 0 and 1 for the special tokens first (padding must be 0 by convention).
        stoi = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        # Give every kept word the next free id (2, 3, 4, ...).
        for w in words:
            stoi[w] = len(stoi)
        # `cls(...)` creates a Vocab object from the finished dictionary.
        return cls(stoi)

    def __len__(self) -> int:
        # Lets us write len(vocab) to get the vocabulary size (needed for the embedding table).
        return len(self.stoi)

    def encode(self, text: str, max_len: int) -> List[int]:
        """
        Convert one sentence into a list of integer ids, cut to max_len words.

        max_len exists because LSTM cost grows with sentence length and LIAR
        statements are short (~18 words on average). 64 keeps almost everything.
        """
        # For each word: look up its id, or use <unk> if we never saw it in training.
        # `[:max_len]` truncates very long statements.
        ids = [self.stoi.get(tok, self.unk_idx) for tok in tokenize(text)][:max_len]
        # An empty list would crash the LSTM (sequence length 0 is illegal),
        # so for an empty statement return a single <unk> token instead.
        return ids or [self.unk_idx]


def pad_batch(seqs: List[List[int]], pad_idx: int):
    """
    Turn a list of id-lists with DIFFERENT lengths into ONE rectangular tensor.

    WHY: GPUs/CPUs process a batch as a single matrix, and a matrix must be
    rectangular. So shorter sentences are filled with the <pad> id on the right.
    We also return the true lengths so the LSTM can IGNORE the padding
    (otherwise the model would "read" dozens of meaningless pad tokens).

    Returns: (ids  [batch, longest_len],  lengths [batch])
    """
    # The real (unpadded) length of each sentence, as a tensor.
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    # Create a [batch, longest] matrix completely filled with the pad id.
    out = torch.full((len(seqs), int(lengths.max())), pad_idx, dtype=torch.long)
    # Overwrite the left part of each row with that sentence's real ids.
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out, lengths


class BiLSTMClassifier(nn.Module):
    """
    The neural network: Embedding -> Bidirectional LSTM -> Linear (6 classes).

    Data flow for one sentence:
        word ids -> embedding vectors -> BiLSTM reads left->right AND right->left
        -> combine final states + max-pooled states -> dropout -> 6 scores
    The 6 scores correspond to: pants-fire, false, barely-true, half-true,
    mostly-true, true (the fixed label order in common/config.py).
    """

    def __init__(
        self,
        vocab_size: int,           # how many different words the embedding table must hold
        num_classes: int = 6,      # LIAR has 6 truthfulness labels
        embed_dim: int = 100,      # size of the vector that represents each word
        hidden_dim: int = 128,     # size of the LSTM's memory in each direction
        num_layers: int = 1,       # stacked LSTM layers (1 keeps the baseline small)
        dropout: float = 0.5,      # fraction of values randomly switched off during training
        pad_idx: int = 0,          # id of <pad>, so its embedding stays all zeros
    ):
        # Required: initialises the parent nn.Module so PyTorch tracks our layers/weights.
        super().__init__()

        # EMBEDDING LAYER: a trainable table [vocab_size x embed_dim].
        # "Trainable embeddings" is exactly what issue #4 asks for: the word
        # vectors start random and are learned during training (no GloVe etc.).
        # padding_idx keeps the <pad> vector at zero and never updates it.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # THE LSTM: a recurrent layer that reads the sentence word by word and
        # keeps a memory of what it has seen (so word ORDER matters).
        self.lstm = nn.LSTM(
            embed_dim,                 # input size per time-step = one word vector
            hidden_dim,                # size of the memory
            num_layers=num_layers,
            batch_first=True,          # tensors are shaped [batch, time, features]
            bidirectional=True,        # "Bi": also read right->left, so every word
                                       # gets context from both before and after it
            # PyTorch only allows LSTM-internal dropout when there are 2+ layers,
            # so we pass 0 for a single layer to avoid a warning.
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # DROPOUT: randomly zeroes values while training. It fights overfitting,
        # which is a real danger here: LIAR is only ~10k sentences.
        self.dropout = nn.Dropout(dropout)

        # OUTPUT LAYER: maps the sentence summary to one score per class.
        # Input size is hidden_dim * 4 because we concatenate TWO things, each
        # of which is 2 directions * hidden_dim wide:
        #     (final forward+backward state)  +  (max-pooled states)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Run a batch of padded sentences through the network.
        Returns raw class scores ("logits") of shape [batch, num_classes].
        """
        # 1) ids -> vectors, then apply dropout to the word vectors.
        emb = self.dropout(self.embedding(ids))

        # 2) "Pack" the batch so the LSTM skips the <pad> positions entirely.
        #    lengths must live on the CPU for this PyTorch function.
        #    enforce_sorted=False lets us pass sentences in any length order.
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # 3) Run the LSTM. `out` = output at every word position,
        #    `h` = the final hidden state of each layer/direction.
        out, (h, _) = self.lstm(packed)

        # 4) Unpack back to a normal padded tensor. Padding positions get -1e4
        #    (a huge negative number) so that the max-pooling in step 6
        #    can never accidentally pick a padded position.
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, padding_value=-1e4)

        # 5) Sentence summary #1: the last forward state (h[-2]) glued to the last
        #    backward state (h[-1]). It "remembers" the whole sentence.
        last = torch.cat([h[-2], h[-1]], dim=1)

        # 6) Sentence summary #2: for each feature, the strongest value over all
        #    word positions (max-pooling). It captures the most important
        #    words wherever they occur, which helps on long statements.
        pooled = out.max(dim=1).values

        # 7) Join both summaries, apply dropout, and produce the 6 class scores.
        return self.fc(self.dropout(torch.cat([last, pooled], dim=1)))


def count_params(model: nn.Module) -> int:
    """
    Count the trainable weights of the model.

    WHY: issue #4 requires "params" in the results JSON, and issue #11 compares
    model SIZE (LSTM ~ a few million vs BERT 110M vs LLM 7B) against accuracy.
    """
    # p.numel() = number of values in one weight tensor; we add them all up.
    # Only parameters with requires_grad=True are trained, so only those count.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
