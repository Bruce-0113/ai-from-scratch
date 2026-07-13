import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    """Build a vocabulary and a distance-weighted word co-occurrence matrix.

    Args:
        docs: An iterable of tokenized documents (each a list/sequence of tokens).
        window: Context window size; tokens within this many positions of a
            center token are counted as co-occurring.

    Returns:
        A tuple ``(vocab, pair_counts)`` where ``vocab`` maps each token to a
        unique integer index, and ``pair_counts`` is a ``Counter`` mapping
        ``(center_idx, context_idx)`` pairs to their co-occurrence weight
        (weighted by ``1 / distance`` between the two tokens).
    """
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    """Train GloVe word vectors from co-occurrence counts via SGD.

    Args:
        vocab: Mapping from token to integer index (as returned by
            ``build_cooccurrence``).
        pair_counts: Mapping from ``(center_idx, context_idx)`` to
            co-occurrence weight (as returned by ``build_cooccurrence``).
        dim: Dimensionality of the word/context embeddings.
        epochs: Number of full passes over ``pair_counts``.
        lr: Learning rate for the SGD updates.
        x_max: Cutoff above which the weighting function saturates to 1.0.
        alpha: Exponent used in the weighting function for counts below
            ``x_max``.
        seed: Seed for the random number generator used to initialize
            embeddings.

    Returns:
        A ``(n_vocab, dim)`` array where row ``i`` is the final embedding for
        token ``i``, computed as the sum of its word vector and context
        vector (a common convention for using GloVe outputs).
    """
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde

def char_ngrams(word, n_min=3, n_max=6):
    """Generate the set of character n-grams FastText uses to represent a word.

    Args:
        word: The word to break into character n-grams.
        n_min: Minimum n-gram length (inclusive).
        n_max: Maximum n-gram length (inclusive).

    Returns:
        A set of substrings of ``<word>`` (word wrapped with boundary
        markers ``<`` and ``>``) of length ``n_min`` to ``n_max``, plus the
        whole wrapped word itself.
    """
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams

def fasttext_vector(word, ngram_table):
    """Compute a FastText-style word vector by summing its n-gram embeddings.

    Args:
        word: The word to embed.
        ngram_table: Mapping from character n-gram (as produced by
            ``char_ngrams``) to its embedding vector.

    Returns:
        The sum of the embeddings for all of the word's n-grams found in
        ``ngram_table``, or ``None`` if none of them are present.
    """
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)

def learn_bpe(corpus, k_merges):
    """Learn Byte-Pair Encoding merge rules from a word-frequency corpus.

    Args:
        corpus: Mapping from word (string) to its frequency count in the
            training corpus.
        k_merges: Maximum number of merge operations to learn.

    Returns:
        A list of ``(a, b)`` symbol-pair tuples, in the order they were
        merged. Learning stops early if no mergeable pairs remain.
    """
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    """Tokenize a word by greedily applying learned BPE merge rules in order.

    Args:
        word: The word to tokenize.
        merges: Ordered list of ``(a, b)`` symbol-pair merges, as returned
            by ``learn_bpe``.

    Returns:
        A list of subword tokens representing ``word`` after applying each
        merge rule in sequence, with a trailing ``"</w>"`` end-of-word
        marker.
    """
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens

