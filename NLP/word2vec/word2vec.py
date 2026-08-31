import numpy as np
from gensim.models import Word2Vec

def build_vocab(docs):
    """Build a word-to-index vocabulary from a collection of documents.

    Assigns each unique token the next available integer index, in order
    of first appearance.

    Args:
        docs: A list of documents, where each document is a list of tokens.

    Returns:
        A dict mapping each unique token to its vocabulary index.
    """
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def skipgram_pairs(docs, window=2):
    """Generate (center, context) word pairs for skip-gram training.

    For every word in every document, pairs it with each neighboring word
    within `window` positions on either side.

    Args:
        docs: A list of documents, where each document is a list of tokens.
        window: Maximum distance between a center word and a context word.

    Returns:
        A list of (center_word, context_word) tuples.
    """
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs

def init_embeddings(vocab_size, dim, seed=0):
    """Randomly initialize the two embedding matrices used by word2vec.

    Args:
        vocab_size: Number of unique words in the vocabulary.
        dim: Dimensionality of each embedding vector.
        seed: Seed for the random number generator, for reproducibility.

    Returns:
        A tuple (W, W_prime):
            W: Input/center-word embedding matrix, shape (vocab_size, dim).
            W_prime: Output/context-word embedding matrix, shape (vocab_size, dim).
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime

def sigmoid(x):
    """Compute the logistic sigmoid, clipping inputs to avoid overflow.

    Args:
        x: A scalar or numpy array of input values.

    Returns:
        The sigmoid of `x`, elementwise, in the range (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    """Run one skip-gram-with-negative-sampling update for a single pair.

    Updates `W` and `W_prime` in place: pulls the center word's embedding
    toward the true context word and pushes it away from the sampled
    negative (noise) words.

    Args:
        W: Center-word embedding matrix, shape (vocab_size, dim). Modified in place.
        W_prime: Context-word embedding matrix, shape (vocab_size, dim). Modified in place.
        center_idx: Vocabulary index of the center word.
        context_idx: Vocabulary index of the true context word.
        negative_indices: Vocabulary indices of the sampled negative words.
        lr: Learning rate for the gradient update.
    """
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center

def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    """Train skip-gram word embeddings with negative sampling.

    Builds the vocabulary and skip-gram pairs from `docs`, then repeatedly
    shuffles the pairs and applies a negative-sampling update for each one.

    Args:
        docs: A list of documents, where each document is a list of tokens.
        dim: Dimensionality of the learned embeddings.
        window: Context window size passed to `skipgram_pairs`.
        k_neg: Number of negative samples drawn per training pair.
        epochs: Number of passes over the full set of training pairs.
        lr: Learning rate for the gradient updates.
        seed: Seed for the random number generator, for reproducibility.

    Returns:
        A tuple (vocab, W):
            vocab: Dict mapping each word to its vocabulary index.
            W: Trained center-word embedding matrix, shape (vocab_size, dim).

    """
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W

def nearest(vocab, W, target_vec, topk=5, exclude=None):
    """Find the words whose embeddings are most cosine-similar to a target vector.

    Args:
        vocab: Dict mapping each word to its vocabulary index.
        W: Embedding matrix, shape (vocab_size, dim).
        target_vec: Query vector to compare against, shape (dim,).
        topk: Number of nearest words to return.
        exclude: Optional set of vocabulary indices to skip (e.g. the query
            word itself).

    Returns:
        A list of up to `topk` (word, cosine_similarity) tuples, sorted by
        descending similarity.
    """
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    """Solve a word analogy of the form "a is to b as c is to ?".

    Computes the vector b - a + c (e.g. king - man + woman ~ queen) and
    returns the words whose embeddings are closest to it.

    Args:
        vocab: Dict mapping each word to its vocabulary index.
        W: Embedding matrix, shape (vocab_size, dim).
        a: First word of the analogy.
        b: Second word of the analogy.
        c: Third word of the analogy.
        topk: Number of candidate answer words to return.

    Returns:
        A list of up to `topk` (word, cosine_similarity) tuples, sorted by
        descending similarity, excluding a, b, and c themselves.
    """
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})    

def gensim_test():
    """Train a skip-gram word2vec model with gensim and print a quick sanity check.

    Fits `gensim.models.Word2Vec` (sg=1, i.e. skip-gram, with negative
    sampling) on two toy sentences, then prints the embedding for "cat" and
    its top-3 most similar words, as a smoke test against the from-scratch
    implementation above.
    """
    sentences = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "ran", "across", "the", "room"],
    ]

    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        sg=1,
        negative=5,
        workers=4,
        epochs=30,
    )

    print(model.wv["cat"])
    print(model.wv.most_similar("cat", topn=3))
    