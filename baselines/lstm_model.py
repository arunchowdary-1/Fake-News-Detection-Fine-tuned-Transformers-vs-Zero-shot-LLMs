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
# VIVA Q: <pad> aur <unk> kya hain, aur inki zaroorat kyun padi?
#     A: <pad> (id 0) chhote sentences ko lamba karne wala bharti ka token hai, taaki ek batch rectangle
#        ban sake. <unk> (id 1) un words ke liye hai jo training vocabulary me nahi the; iske bina test
#        me naya word aate hi lookup crash ho jaata.
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
# VIVA Q: Tokenizer simple regex kyun, aur $ aur % ko kyun rakha?
#     A: Baseline ko simple aur tez rakhna tha (BERT apna tokenizer khud use karta hai). Political claims
#        me '$3 trillion' ya '50%' jaise numbers aur symbols important signal hote hain, isliye unhe alag
#        tokens rakha.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[$%]")


# VIVA Q: Text ko lowercase kyun karte hain?
#     A: Taaki 'Tax' aur 'tax' ek hi word maane jayein; vocabulary chhoti rehti hai aur model ko kam data
#        me zyada seekhne ko milta hai. Nuksan: 'US' aur 'us' mil jaate hain.
def tokenize(text: str) -> List[str]:  # sentence (text) -> lowercase words ki list; ye pehla step hai text ko numbers me badalne ka
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
        return []  # text string nahi hai (None/number) toh khaali list, taaki training crash na kare
    # Lowercase first so "Tax" and "tax" become the same word, then find all matches.
    return _TOKEN_RE.findall(text.lower())


# VIVA Q: Vocabulary sirf training data se kyun banti hai, test se kyun nahi?
#     A: Agar test ke words bhi dekh lein toh test ki jaankari training me leak ho jaati hai (data
#        leakage) aur score jhootha achha dikhta hai. Isliye test ke naye words <unk> ban jaate hain,
#        jaisa asli duniya me hota.
class Vocab:  # Vocab = word <-> number ki dictionary; embedding layer ko number chahiye, word nahi
    """
    Dictionary that maps every known word to an integer id.

    WHY: the embedding layer of the network looks up a vector by integer id,
    so every word needs a number. Example: {"<pad>":0, "<unk>":1, "tax":2, ...}
    IMPORTANT: it is built from the TRAINING split only. If we also counted
    words from the test split, information from the test set would leak into
    training and our accuracy would look better than it really is.
    """

    def __init__(self, stoi: Dict[str, int]):  # ready-made dictionary (stoi) se Vocab object banata hai
        # "stoi" = "string to index": the word -> id dictionary.
        self.stoi = stoi
        # Remember the ids of the two special tokens, since the model needs them.
        self.pad_idx = stoi[PAD_TOKEN]
        self.unk_idx = stoi[UNK_TOKEN]  # <unk> ka number yaad rakho, unknown words ko yahi milega

    # VIVA Q: min_freq=2 aur max_size=20000 kyun?
    #     A: Ek hi baar dikhne wale words zyadatar typos ya naam hote hain; unse kuch seekha nahi ja
    #        sakta aur embedding table bekaar bada ho jaata. Asli run me 10,269 training statements se
    #        vocabulary 6,944 words ki bani, yaani 20,000 ki limit tak pahunchi hi nahi.
    @classmethod  # classmethod: Vocab.build(...) seedha class se chalta hai, pehle object banane ki zaroorat nahi
    def build(cls, texts: Iterable[str], min_freq: int = 2, max_size: int = 20000) -> "Vocab":  # training sentences se vocabulary banata hai (min_freq=2: kam se kam 2 baar dikhe, max_size=20000)
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
            stoi[w] = len(stoi)  # agla khaali number (2, 3, 4, ...) is word ko de do
        # `cls(...)` creates a Vocab object from the finished dictionary.
        return cls(stoi)

    def __len__(self) -> int:  # len(vocab) likhne par vocabulary ka size mile (embedding table ka size isi se tay hota hai)
        # Lets us write len(vocab) to get the vocabulary size (needed for the embedding table).
        return len(self.stoi)

    def encode(self, text: str, max_len: int) -> List[int]:  # ek sentence -> numbers ki list, max_len se zyada lamba ho toh kaat do
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


# VIVA Q: Padding aur packing kyun karte hain?
#     A: CPU/GPU ek rectangle matrix par kaam karte hain, isliye chhote sentences ko <pad> se bharte
#        hain. Par LSTM ko <pad> 'padhna' nahi chahiye, isliye asli lambai (lengths) bhi saath bhejte
#        hain aur forward() me pack_padded_sequence se pad skip karwate hain.
def pad_batch(seqs: List[List[int]], pad_idx: int):  # alag-alag lambai ke sentences ko ek rectangle (tensor) banata hai, chhote ko <pad> se bharke
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
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)  # i-th sentence ke asli numbers bayen hisse me daalo, baaki jagah <pad> hi rehti hai
    return out, lengths  # (padded matrix, har sentence ki asli lambai) dono wapas; lambai LSTM ko pad ignore karne ke liye chahiye


# VIVA Q: Baseline me LSTM hi kyun?
#     A: Ye bina pre-training ka chhota model hai (CPU par ~4 minute, sirf 0.93M params). BERT, RoBERTa
#        aur LLM ko isse compare karke pata chalta hai ki pre-trained knowledge se kitna fayda hua; agar
#        wo isse ZYADA behtar nahi hote toh unka kharch jaayaz nahi.
# VIVA Q: LSTM simple RNN se behtar kyun hai?
#     A: LSTM me gates (input, forget, output) aur cell memory hoti hai, jo lambe sentence me purani baat
#        yaad rakhne deti hai aur vanishing gradient ki dikkat kam karti hai. Simple RNN kuch words baad
#        hi pichla context bhool jaata hai.
class BiLSTMClassifier(nn.Module):  # poora neural network: Embedding -> BiLSTM -> Linear(6 classes)
    """
    The neural network: Embedding -> Bidirectional LSTM -> Linear (6 classes).

    Data flow for one sentence:
        word ids -> embedding vectors -> BiLSTM reads left->right AND right->left
        -> combine final states + max-pooled states -> dropout -> 6 scores
    The 6 scores correspond to: pants-fire, false, barely-true, half-true,
    mostly-true, true (the fixed label order in common/config.py).
    """

    def __init__(  # network ki layers yahan banti hain (sirf ek baar, model banate waqt)
        self,  # network object khud
        vocab_size: int,           # how many different words the embedding table must hold
        num_classes: int = 6,      # LIAR has 6 truthfulness labels
        embed_dim: int = 100,      # size of the vector that represents each word
        hidden_dim: int = 128,     # size of the LSTM's memory in each direction
        num_layers: int = 1,       # stacked LSTM layers (1 keeps the baseline small)
        dropout: float = 0.5,      # fraction of values randomly switched off during training
        pad_idx: int = 0,          # id of <pad>, so its embedding stays all zeros
    ):  # yahan tak constructor ke parameters ki list khatam
        # Required: initialises the parent nn.Module so PyTorch tracks our layers/weights.
        super().__init__()

        # EMBEDDING LAYER: a trainable table [vocab_size x embed_dim].
        # "Trainable embeddings" is exactly what issue #4 asks for: the word
        # vectors start random and are learned during training (no GloVe etc.).
        # padding_idx keeps the <pad> vector at zero and never updates it.
        # VIVA Q: Embedding kya hai, aur 'trainable' ka matlab?
        #     A: Har word ko 100 numbers ka vector milta hai. Vectors random shuru hote hain aur training
        #        me seekhe jaate hain (GloVe jaisa pre-trained nahi liya). Embedding table = 6,944 words
        #        x 100 = 694,400 params, jo total 932,998 ka lagbhag 74% hai.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # THE LSTM: a recurrent layer that reads the sentence word by word and
        # keeps a memory of what it has seen (so word ORDER matters).
        # VIVA Q: Bidirectional ka kya fayda hai?
        #     A: Ek LSTM sentence left se right padhta hai aur doosra right se left, toh har word ko aage
        #        aur peeche dono taraf ka context milta hai. Iska kharch: LSTM ke params 2 guna ho jaate
        #        hain (235,520).
        # VIVA Q: num_layers=1 kyun?
        #     A: Baseline chhota rakhna tha, aur data sirf ~10k sentences ka hai; zyada layers overfit
        #        karte.
        self.lstm = nn.LSTM(
            embed_dim,                 # input size per time-step = one word vector
            hidden_dim,                # size of the memory
            num_layers=num_layers,  # LSTM layers ki sankhya (1 = chhota baseline)
            batch_first=True,          # tensors are shaped [batch, time, features]
            bidirectional=True,        # "Bi": also read right->left, so every word
                                       # gets context from both before and after it
            # PyTorch only allows LSTM-internal dropout when there are 2+ layers,
            # so we pass 0 for a single layer to avoid a warning.
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # DROPOUT: randomly zeroes values while training. It fights overfitting,
        # which is a real danger here: LIAR is only ~10k sentences.
        # VIVA Q: Dropout kya hai, aur 0.5 kyun?
        #     A: Training me random 50% values band kar dete hain, toh model ek-do neurons par nirbhar
        #        nahi rehta. Isse overfitting kam hoti hai, jo 10k sentences me badi dikkat hai.
        #        Evaluation (model.eval()) me dropout band rehta hai, isliye predictions ek jaise aate
        #        hain.
        self.dropout = nn.Dropout(dropout)

        # OUTPUT LAYER: maps the sentence summary to one score per class.
        # Input size is hidden_dim * 4 because we concatenate TWO things, each
        # of which is 2 directions * hidden_dim wide:
        #     (final forward+backward state)  +  (max-pooled states)
        # VIVA Q: Linear layer ka input hidden_dim*4 = 512 kyun?
        #     A: Do cheezein jodte hain: aakhri forward+backward state (2 x 128 = 256) aur max-pooled
        #        features (2 x 128 = 256), yaani 512. Output 6 = classes (pants-fire se true). Params =
        #        512 x 6 + 6 = 3,078.
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:  # ek batch ko network se guzaarta hai aur har class ka score (logit) wapas deta hai
        """
        Run a batch of padded sentences through the network.
        Returns raw class scores ("logits") of shape [batch, num_classes].
        """
        # 1) ids -> vectors, then apply dropout to the word vectors.
        emb = self.dropout(self.embedding(ids))

        # 2) "Pack" the batch so the LSTM skips the <pad> positions entirely.
        #    lengths must live on the CPU for this PyTorch function.
        #    enforce_sorted=False lets us pass sentences in any length order.
        # VIVA Q: pack_padded_sequence ka kaam kya hai?
        #     A: LSTM ko batch ke <pad> wale bekaar steps skip karwana; warna model un khaali steps se
        #        confuse hota aur chhote sentences ka summary kharab hota.
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False  # emb = word vectors; lengths CPU par chahiye; enforce_sorted=False = sentences ko lambai ke order me rakhna zaroori nahi
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
        # VIVA Q: Aakhri state aur max-pool, dono kyun?
        #     A: Aakhri state poore sentence ka summary deti hai; max-pool har feature ka sabse tez
        #        signal pakadta hai chahe wo word kahin bhi ho. Dono jodne se chhote aur lambe dono
        #        sentences par model behtar chalta hai.
        last = torch.cat([h[-2], h[-1]], dim=1)

        # 6) Sentence summary #2: for each feature, the strongest value over all
        #    word positions (max-pooling). It captures the most important
        #    words wherever they occur, which helps on long statements.
        pooled = out.max(dim=1).values

        # 7) Join both summaries, apply dropout, and produce the 6 class scores.
        # VIVA Q: Softmax kyun nahi lagaya?
        #     A: Hum raw scores (logits) wapas dete hain. CrossEntropyLoss andar softmax khud lagata hai,
        #        aur prediction ke liye argmax kaafi hai, toh alag softmax ki zaroorat nahi.
        return self.fc(self.dropout(torch.cat([last, pooled], dim=1)))


# VIVA Q: 932,998 parameters kahan se aaye?
#     A: Embedding 694,400 + LSTM 235,520 + Linear 3,078 = 932,998.
# VIVA Q: Parameter count itna kyun important hai?
#     A: Issue #11 ka sawal hai size vs accuracy vs cost. LSTM ~1M, BERT 109M, RoBERTa 125M, Qwen 7.6B
#        params; jitna bada model, utna zyada memory aur kharch.
def count_params(model: nn.Module) -> int:  # trainable weights ginta hai (results JSON me 'params' ke liye, 932,998)
    """
    Count the trainable weights of the model.

    WHY: issue #4 requires "params" in the results JSON, and issue #11 compares
    model SIZE (LSTM ~ a few million vs BERT 110M vs LLM 7B) against accuracy.
    """
    # p.numel() = number of values in one weight tensor; we add them all up.
    # Only parameters with requires_grad=True are trained, so only those count.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
