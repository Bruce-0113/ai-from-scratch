"""Scaled dot-product self-attention and causal masking, built from scratch
with NumPy.

Mirrors the "Self-Attention from Scratch" lesson from
rohitg00/ai-engineering-from-scratch
(phases/07-transformers-deep-dive/02-self-attention-from-scratch): every
token projects into a query, a key, and a value; attention is a soft lookup
where a query's dot product against every key becomes a softmax-weighted
blend of values, scaled by sqrt(dk) to keep softmax out of its
vanishing-gradient regime.

Splitting this single head into several parallel heads (multi-head
attention), and the Grouped-/Multi-Query variants used in production models,
is its own lesson -- see `../Multi_Head_Attention/multi_head_attention.py`.

1. softmax / scaled_dot_product_attention  -- the core formula:
                                               softmax(QK^T / sqrt(dk)) @ V
2. causal_mask                             -- upper-triangular mask that
                                               turns bidirectional attention
                                               into autoregressive
                                               (decoder-style) attention
3. SelfAttention                           -- single-head self-attention with
                                               learned Wq/Wk/Wv projections
4. print_attention_weights / ascii_heatmap -- text-table and ASCII-art views
                                               of an attention matrix
5. demo_*                                  -- self-attention on a toy
                                               sentence, and bidirectional
                                               vs. causal attention

Run directly (`python self_attention.py`) to reproduce all demos.
"""

import sys

import numpy as np


# --- 1. Softmax and scaled dot-product attention ----------------------------
def softmax(x):
    """Softmax over the last axis, shifted by the row max for numerical stability.

    Args:
        x: Array of raw scores; softmax is applied along `axis=-1`.

    Returns:
        Array the same shape as `x`, each row summing to 1.
    """
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q, K, V) = softmax(Q @ K^T / sqrt(dk)) @ V.

    Args:
        Q: Queries, shape (n_queries, dk).
        K: Keys, shape (n_keys, dk).
        V: Values, shape (n_keys, dv).
        mask: Optional boolean array broadcastable to (n_queries, n_keys).
            Positions where `mask` is True are set to -inf before softmax, so
            they receive zero attention weight -- this is how causal masking
            (see `causal_mask`) is applied.

    Returns:
        Tuple of (output, weights):
            output: shape (n_queries, dv), the attended output.
            weights: shape (n_queries, n_keys), the softmax attention matrix.
    """
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)
    weights = softmax(scores)
    output = weights @ V
    return output, weights


# --- 2. Causal masking -------------------------------------------------------
def causal_mask(n_tokens):
    """Build a causal mask that hides future positions from each query.

    Args:
        n_tokens: Sequence length.

    Returns:
        Boolean array of shape (n_tokens, n_tokens); True above the diagonal
        (position j is in the future of position i), False elsewhere. Feed
        this to `scaled_dot_product_attention`'s `mask` argument to turn
        bidirectional self-attention into autoregressive, decoder-style
        attention where token i can only attend to tokens <= i.
    """
    return np.triu(np.ones((n_tokens, n_tokens), dtype=bool), k=1)


# --- 3. Single-head self-attention -------------------------------------------
class SelfAttention:
    """Single-head self-attention with learned Q/K/V projections.

    Attributes:
        Wq: Query projection, shape (d_model, dk).
        Wk: Key projection, shape (d_model, dk).
        Wv: Value projection, shape (d_model, dv).
        dk: Query/key dimension, kept for reference.
    """

    def __init__(self, d_model, dk, dv, seed=42):
        """Initialize Wq/Wk/Wv with Xavier-like normal scaling.

        Args:
            d_model: Input embedding dimension.
            dk: Query/key projection dimension.
            dv: Value projection dimension.
            seed: Seed for the random number generator, for reproducibility.
        """
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X, mask=None):
        """Project `X` into Q/K/V and run scaled dot-product attention.

        Args:
            X: Input embeddings, shape (n_tokens, d_model).
            mask: Optional mask forwarded to `scaled_dot_product_attention`
                (see `causal_mask`).

        Returns:
            Tuple of (output, weights) -- see `scaled_dot_product_attention`.
        """
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        return output, weights


# --- 4. Displaying attention matrices ----------------------------------------
def print_attention_weights(weights, tokens, title=None):
    """Print an attention matrix as an aligned table, one row per query token.

    Args:
        weights: Attention matrix, shape (len(tokens), len(tokens)).
        tokens: Sequence of token strings labeling rows and columns.
        title: Optional heading printed above the table.
    """
    if title:
        print(title)
    print(f"{'':>6}", end="")
    for token in tokens:
        print(f"{token:>6}", end="")
    print()

    for i, token in enumerate(tokens):
        print(f"{token:>6}", end="")
        for j in range(len(tokens)):
            print(f"{weights[i][j]:6.3f}", end="")
        print()


def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    """Render an attention matrix as an ASCII-art heatmap.

    Args:
        weights: Attention matrix, shape (len(tokens), len(tokens)).
        tokens: Sequence of token strings labeling rows and columns.
        chars: Shading characters from lightest to darkest, indexed by each
            weight's fraction of `weights.max()`.
    """
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()


# --- 5. Demos -----------------------------------------------------------------
def demo_softmax():
    """Reproduce the softmax numerical-stability walkthrough on a toy logit vector."""
    print("=== 1. Softmax ===")
    logits = np.array([2.0, 1.0, 0.1])
    print(f"logits:  {logits}")
    print(f"softmax: {softmax(logits)}")
    print(f"sum:     {softmax(logits).sum():.4f}")


def make_toy_sentence(d_model=8, seed=42):
    """Build a toy sentence and random "embeddings" to drive the demos below.

    Args:
        d_model: Embedding dimension.
        seed: Seed for the random number generator, for reproducibility.

    Returns:
        Tuple of (sentence, X): `sentence` is a list of token strings, `X` is
        a (len(sentence), d_model) array standing in for a real embedding
        lookup.
    """
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (len(sentence), d_model))
    return sentence, X


def demo_self_attention(sentence, X, dk=4, dv=4):
    """Run single-head self-attention on `X` and print the resulting weights.

    Returns:
        The fitted `SelfAttention` instance, reused by `demo_causal_mask` so
        the only difference between bidirectional and causal attention is
        the mask, not the learned projections.
    """
    print("\n=== 2. Self-attention on a toy sentence ===")
    attn = SelfAttention(d_model=X.shape[-1], dk=dk, dv=dv, seed=42)
    _, weights = attn.forward(X)
    print_attention_weights(
        weights, sentence, title="Attention weights (each row: where that token looks):\n"
    )
    ascii_heatmap(weights, sentence)
    return attn


def demo_causal_mask(attn, X, sentence):
    """Compare bidirectional vs. causal-masked attention from the same `attn`.

    Reusing `attn`'s already-learned Wq/Wk/Wv isolates the effect of the
    mask: causal attention zeroes out every upper-triangular (future) entry,
    leaving each row's weight concentrated on the current and earlier
    tokens -- the shape of attention a decoder uses during autoregressive
    generation.
    """
    print("\n=== 3. Causal masking (decoder-style attention) ===")
    mask = causal_mask(len(sentence))
    _, causal_weights = attn.forward(X, mask=mask)
    print_attention_weights(
        causal_weights, sentence, title="Causal attention weights (future positions forced to 0):"
    )


def main():
    """Run every demo end to end: softmax, self-attention, and causal masking."""
    demo_softmax()
    sentence, X = make_toy_sentence()
    attn = demo_self_attention(sentence, X)
    demo_causal_mask(attn, X, sentence)


if __name__ == "__main__":
    # ascii_heatmap prints block-shading characters (░▒▓█); force UTF-8 so
    # this doesn't crash under a non-UTF-8 console codepage (e.g. Windows cp950).
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
