"""Named Entity Recognition (NER) experiments.

This script demonstrates three common approaches to NER:

1. BIO tagging utilities (span <-> BIO label conversion).
2. A rule-based tagger driven by simple gazetteers (lookup lists).
3. A feature-based CRF tagger (via sklearn-crfsuite).
4. A BiLSTM-CRF neural model head (via PyTorch).
"""

import torch
import torch.nn as nn
import sklearn_crfsuite


# ---------------------------------------------------------------------------
# BIO tagging utilities
# ---------------------------------------------------------------------------

def spans_to_bio(tokens, spans):
    """Convert entity spans into per-token BIO labels.

    Args:
        tokens: List of tokens in the sentence.
        spans: List of (start, end, label) tuples, where `start` and `end`
            are token indices (end-exclusive) and `label` is the entity type.

    Returns:
        List of BIO-formatted labels, one per token (e.g. "B-ORG", "I-ORG", "O").
    """
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    """Convert per-token BIO labels back into entity spans.

    Args:
        tokens: List of tokens in the sentence.
        labels: List of BIO-formatted labels, one per token.

    Returns:
        List of (start, end, label) tuples, where `start` and `end` are
        token indices (end-exclusive) and `label` is the entity type.
    """
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans


# ---------------------------------------------------------------------------
# Rule-based NER (gazetteer lookup)
# ---------------------------------------------------------------------------

ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def word_shape(word):
    """Compute the abstract "shape" of a word.

    Uppercase letters become "X", lowercase letters become "x", digits
    become "d", and all other characters are kept as-is. Useful as a
    feature for NER since it captures casing/formatting patterns (e.g.
    "Xxxx" for a capitalized word, "dd/dd/dddd" for a date).

    Args:
        word: The input word.

    Returns:
        The shape string for the word.
    """
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)


def rule_based_ner(tokens):
    """Tag tokens using simple gazetteer lookups.

    Note: this only ever emits "B-" labels (no "I-" continuation), so it
    is limited to single-token entities.

    Args:
        tokens: List of tokens in the sentence.

    Returns:
        List of BIO-formatted labels, one per token.
    """
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels


# ---------------------------------------------------------------------------
# Feature-based CRF NER
# ---------------------------------------------------------------------------

def token_features(token, prev_token, next_token):
    """Build a feature dict for a single token using its neighbors.

    Args:
        token: The current token.
        prev_token: The preceding token, or None/"" at the start of a sentence.
        next_token: The following token, or None/"" at the end of a sentence.

    Returns:
        Dict of named features describing the token in context.
    """
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def to_features(tokens):
    """Build per-token CRF feature dicts for a whole sentence.

    Args:
        tokens: List of tokens in the sentence.

    Returns:
        List of feature dicts, one per token, suitable for sklearn-crfsuite.
    """
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


def train_crf(sentences_tokenized, bio_labels_train):
    """Train a linear-chain CRF tagger for NER.

    Args:
        sentences_tokenized: List of tokenized sentences (list of token lists).
        bio_labels_train: List of BIO label sequences, one per sentence.

    Returns:
        A fitted `sklearn_crfsuite.CRF` model.
    """
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True
    )
    X_train = [to_features(s) for s in sentences_tokenized]
    crf.fit(X_train, bio_labels_train)
    return crf


# ---------------------------------------------------------------------------
# Neural NER: BiLSTM + CRF head
# ---------------------------------------------------------------------------

class BiLSTM_CRF_Head(nn.Module):
    """BiLSTM encoder with a linear layer producing per-label emission scores.

    This module outputs raw emission scores; pairing it with an actual CRF
    decoding layer (e.g. from `torchcrf`) is left to the caller.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        """
        Args:
            vocab_size: Number of distinct tokens in the vocabulary.
            embed_dim: Dimensionality of token embeddings.
            hidden_dim: Hidden size of each LSTM direction.
            n_labels: Number of output BIO label classes.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        """Compute per-token label emission scores.

        Args:
            token_ids: LongTensor of shape (batch, seq_len) of token ids.

        Returns:
            FloatTensor of shape (batch, seq_len, n_labels) of emission scores.
        """
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions


if __name__ == "__main__":
    tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
    labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
    print(bio_to_spans(tokens, labels))
    print(rule_based_ner(tokens))
