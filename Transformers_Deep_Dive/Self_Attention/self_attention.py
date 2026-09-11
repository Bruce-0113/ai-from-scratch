"""Scaled dot-product self-attention, causal masking, and multi-head
attention, built from scratch with NumPy and checked against PyTorch's
`nn.MultiheadAttention`.

Mirrors the "Self-Attention from Scratch" lesson from
rohitg00/ai-engineering-from-scratch
(phases/07-transformers-deep-dive/02-self-attention-from-scratch): every
token projects into a query, a key, and a value; attention is a soft lookup
where a query's dot product against every key becomes a softmax-weighted
blend of values, scaled by sqrt(dk) to keep softmax out of its
vanishing-gradient regime.

1. softmax / scaled_dot_product_attention  -- the core formula:
                                               softmax(QK^T / sqrt(dk)) @ V
2. causal_mask                             -- upper-triangular mask that
                                               turns bidirectional attention
                                               into autoregressive
                                               (decoder-style) attention
3. SelfAttention                           -- single-head self-attention with
                                               learned Wq/Wk/Wv projections
4. MultiHeadAttention                      -- splits Q/K/V into n_heads
                                               chunks, attends in parallel,
                                               concatenates, and projects
                                               through Wo
5. print_attention_weights / ascii_heatmap -- text-table and ASCII-art views
                                               of an attention matrix
6. demo_*                                  -- self-attention on a toy
                                               sentence, bidirectional vs.
                                               causal attention, multi-head
                                               attention, and a shape/behavior
                                               check against
                                               torch.nn.MultiheadAttention

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


# --- 4. Multi-head attention -------------------------------------------------
class MultiHeadAttention:
    """Multi-head attention: n_heads parallel attention functions, concatenated
    and projected back to d_model.

    Each head gets its own slice of the full Q/K/V projections (dimension
    d_model / n_heads), attends independently, and the concatenated head
    outputs are projected through `Wo`. This lets the model attend to
    different relationship types (e.g. syntax vs. coreference) at once,
    instead of averaging them into a single attention pattern.

    Attributes:
        n_heads: Number of attention heads.
        d_head: Per-head Q/K/V dimension (d_model / n_heads).
        Wq, Wk, Wv, Wo: Learned projections, each shape (d_model, d_model).
    """

    def __init__(self, d_model, n_heads, seed=42):
        """Initialize per-head projections as one (d_model, d_model) matrix
        per Q/K/V/output, so `d_model` must divide evenly by `n_heads`.

        Args:
            d_model: Input/output embedding dimension.
            n_heads: Number of attention heads.
            seed: Seed for the random number generator, for reproducibility.

        Raises:
            ValueError: If `d_model` is not divisible by `n_heads`.
        """
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.Wq = rng.normal(0, scale, (d_model, d_model))
        self.Wk = rng.normal(0, scale, (d_model, d_model))
        self.Wv = rng.normal(0, scale, (d_model, d_model))
        self.Wo = rng.normal(0, scale, (d_model, d_model))

    def _split_heads(self, M, n_tokens):
        """Reshape (n_tokens, d_model) into (n_heads, n_tokens, d_head)."""
        return M.reshape(n_tokens, self.n_heads, self.d_head).transpose(1, 0, 2)

    def forward(self, X, mask=None):
        """Run n_heads attention functions in parallel and merge their outputs.

        Args:
            X: Input embeddings, shape (n_tokens, d_model).
            mask: Optional mask forwarded to every head's
                `scaled_dot_product_attention` call (see `causal_mask`).

        Returns:
            Tuple of (output, weights):
                output: shape (n_tokens, d_model), heads concatenated and
                    projected through `Wo`.
                weights: shape (n_heads, n_tokens, n_tokens), one attention
                    matrix per head.
        """
        n_tokens = X.shape[0]
        Q = self._split_heads(X @ self.Wq, n_tokens)
        K = self._split_heads(X @ self.Wk, n_tokens)
        V = self._split_heads(X @ self.Wv, n_tokens)

        head_outputs, head_weights = [], []
        for h in range(self.n_heads):
            out_h, w_h = scaled_dot_product_attention(Q[h], K[h], V[h], mask=mask)
            head_outputs.append(out_h)
            head_weights.append(w_h)

        output = np.concatenate(head_outputs, axis=-1) @ self.Wo
        weights = np.stack(head_weights, axis=0)
        return output, weights


# --- 5. Displaying attention matrices ----------------------------------------
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


# --- 6. Demos -----------------------------------------------------------------
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


def demo_multi_head_attention(sentence, X, n_heads=2):
    """Run multi-head attention on `X` and print per-head shapes and weights."""
    print("\n=== 4. Multi-head attention ===")
    mha = MultiHeadAttention(d_model=X.shape[-1], n_heads=n_heads, seed=42)
    output, weights = mha.forward(X)
    print(f"Input shape:   {X.shape}")
    print(f"Output shape:  {output.shape}")
    print(f"Weights shape: {weights.shape}  (n_heads, n_tokens, n_tokens)")
    for h in range(n_heads):
        print_attention_weights(weights[h], sentence, title=f"\n-- Head {h} --")


def demo_pytorch_comparison(sentence, X):
    """Check shapes and mechanism against `torch.nn.MultiheadAttention`.

    torch's implementation does exactly what `MultiHeadAttention` does above
    -- split, attend per head, concatenate, project -- plus a fused output
    projection. The weight values differ because the two are initialized
    independently; what should match exactly is the shape contract.
    """
    import torch
    import torch.nn as nn

    print("\n=== 5. Comparison against torch.nn.MultiheadAttention ===")
    d_model = X.shape[-1]
    n_heads = 2
    mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
    X_torch = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    output, attn_weights = mha(X_torch, X_torch, X_torch)

    print(f"Input shape:            {tuple(X_torch.shape)}")
    print(f"Output shape:           {tuple(output.shape)}")
    print(f"Attention weight shape: {tuple(attn_weights.shape)}")
    print_attention_weights(
        attn_weights[0].detach().numpy(), sentence, title="\nAttn weights (averaged over heads):"
    )


def main():
    """Run every demo end to end: softmax, self-attention, causal masking,
    multi-head attention, and a shape/behavior check against PyTorch."""
    demo_softmax()
    sentence, X = make_toy_sentence()
    attn = demo_self_attention(sentence, X)
    demo_causal_mask(attn, X, sentence)
    demo_multi_head_attention(sentence, X)
    demo_pytorch_comparison(sentence, X)


if __name__ == "__main__":
    # ascii_heatmap prints block-shading characters (░▒▓█); force UTF-8 so
    # this doesn't crash under a non-UTF-8 console codepage (e.g. Windows cp950).
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
