"""Multi-head attention, Grouped-Query Attention (GQA), and Multi-Query
Attention (MQA), built from scratch with NumPy and checked against PyTorch's
`nn.MultiheadAttention` / `scaled_dot_product_attention`.

Mirrors the "Multi-Head Attention" lesson from
rohitg00/ai-engineering-from-scratch
(phases/07-transformers-deep-dive/03-multi-head-attention), which builds on
the single-head self-attention already implemented in
`../Self_Attention/self_attention.py`: one attention head learns one kind of
relationship at a time (subject-verb agreement, coreference, plain
adjacency, ...) and is forced to average all of them into a single softmax
distribution. Splitting Q/K/V into n_heads independent, smaller subspaces
lets each head specialize -- for the same total parameter count -- then
concatenates and mixes the results through a learned output projection Wo.

1. softmax                          -- shared softmax helper (see
                                        Self_Attention for the full
                                        numerical-stability derivation)
2. split_heads / combine_heads      -- reshape (N, d_model) <->
                                        (n_heads, N, d_head), no loop needed
3. mha_forward                      -- full multi-head attention: split,
                                        attend per head, concatenate, project
                                        through Wo
4. gqa_project / gqa_forward        -- Grouped-/Multi-Query Attention: fewer
                                        K/V groups than Q heads, repeated to
                                        match (n_kv_heads == 1 is MQA)
5. kv_cache_bytes                   -- KV-cache memory accounting, to
                                        quantify the "n_heads / n_kv_heads
                                        cache shrink" GQA/MQA are built for
6. print_attention_weights           -- aligned text-table view of an
                                        attention matrix
7. demo_*                            -- split/combine round-trip, MHA head
                                        probing, GQA vs. MQA, KV-cache
                                        savings, and a shape/behavior check
                                        against PyTorch

Run directly (`python multi_head_attention.py`) to reproduce all demos.
"""

import numpy as np


# --- 1. Softmax ---------------------------------------------------------------
def softmax(x, axis=-1):
    """Softmax along `axis`, shifted by the per-slice max for numerical stability.

    Args:
        x: Array of raw scores.
        axis: Axis softmax normalizes over. Multi-head attention scores have
            shape (n_heads, n_queries, n_keys), so this is a parameter here
            rather than hardcoded to the last axis, unlike the single-head
            version in Self_Attention.

    Returns:
        Array the same shape as `x`, summing to 1 along `axis`.
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# --- 2. Splitting and combining heads ------------------------------------------
def split_heads(X, n_heads):
    """Reshape (N, d_model) into (n_heads, N, d_head) with one reshape + transpose.

    Args:
        X: Array of shape (N, d_model) -- e.g. a Q, K, or V projection.
        n_heads: Number of heads to split the last dimension into.

    Returns:
        Array of shape (n_heads, N, d_head) where d_head = d_model / n_heads.
        This is exactly the reshape `nn.MultiheadAttention` does internally --
        no explicit per-head loop is needed, because a batched matmul over
        the leading `n_heads` axis is equivalent to running attention
        independently per head.
    """
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)


def combine_heads(H):
    """Inverse of `split_heads`: (n_heads, N, d_head) -> (N, n_heads * d_head).

    Args:
        H: Per-head array, shape (n_heads, N, d_head) -- e.g. per-head
            attention outputs.

    Returns:
        Array of shape (N, n_heads * d_head): heads concatenated along the
        feature axis, ready for the output projection `W_o`.
    """
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)


# --- 3. Multi-head attention ---------------------------------------------------
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    """Multi-head attention: split Q/K/V into heads, attend per head, merge.

    On real hardware, `Qh @ Kh.transpose(...)` below is a single batched
    matmul of shape (heads, N, d_head) x (heads, d_head, N) -> (heads, N, N)
    -- adding heads costs no extra kernel launches, only more FLOPs.

    Args:
        X: Input embeddings, shape (n, d_model).
        W_q, W_k, W_v: Q/K/V projections, each shape (d_model, d_model).
        W_o: Output projection, shape (d_model, d_model), applied after heads
            are concatenated -- the only place heads are allowed to mix.
        n_heads: Number of attention heads; must divide `d_model` evenly.

    Returns:
        Tuple of (output, weights):
            output: shape (n, d_model).
            weights: shape (n_heads, n, n), one attention matrix per head.
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights


# --- 4. Grouped-/Multi-Query Attention ------------------------------------------
def gqa_project(X, W, n_kv_heads, n_heads):
    """Project `X` into `n_kv_heads` K/V groups, then repeat to `n_heads`.

    Args:
        X: Input embeddings, shape (n, d_model).
        W: K or V projection, shape (d_model, n_kv_heads * d_head) --
            narrower than the (d_model, d_model) plain MHA uses, which is
            exactly where GQA/MQA save parameters and KV-cache memory.
        n_kv_heads: Number of distinct K/V groups (MQA: 1; GQA:
            1 < n_kv_heads < n_heads; MHA: n_kv_heads == n_heads).
        n_heads: Number of query heads each K/V group must be repeated to match.

    Returns:
        Array of shape (n_heads, n, d_head): each of the `n_kv_heads` groups
        repeated `n_heads // n_kv_heads` times, so it lines up head-for-head
        with the query heads from `split_heads(X @ W_q, n_heads)`.
    """
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)


def gqa_forward(X, W_q, W_k, W_v, W_o, n_heads, n_kv_heads):
    """Grouped-/Multi-Query attention: full Q heads, fewer K/V groups.

    Identical to `mha_forward` except K and V come from `gqa_project`
    instead of `split_heads` directly. Passing `n_kv_heads == n_heads`
    reproduces plain MHA exactly; `n_kv_heads == 1` is MQA.

    Args:
        X: Input embeddings, shape (n, d_model).
        W_q: Query projection, shape (d_model, d_model).
        W_k, W_v: K/V projections, each shape (d_model, n_kv_heads * d_head)
            where d_head = d_model / n_heads.
        W_o: Output projection, shape (d_model, d_model).
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value groups; must divide `n_heads`.

    Returns:
        Tuple of (output, weights) -- same shapes as `mha_forward`.
    """
    Qh = split_heads(X @ W_q, n_heads)
    Kh = gqa_project(X, W_k, n_kv_heads, n_heads)
    Vh = gqa_project(X, W_v, n_kv_heads, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh
    concat = combine_heads(out)
    return concat @ W_o, weights


# --- 5. KV-cache memory accounting ---------------------------------------------
def kv_cache_bytes(n_layers, seq_len, n_kv_heads, d_head, dtype_bytes=2):
    """Total KV-cache memory (bytes), across every layer, for one sequence.

    Args:
        n_layers: Number of transformer layers, each with its own K and V cache.
        seq_len: Number of cached positions per layer.
        n_kv_heads: Number of key/value heads/groups (MHA: n_heads; GQA:
            1 < n_kv_heads < n_heads; MQA: 1).
        d_head: Per-head dimension.
        dtype_bytes: Bytes per cached scalar (2 for FP16/BF16).

    Returns:
        Total bytes for K and V caches combined across all layers. This is
        exactly the quantity GQA/MQA reduce by shrinking `n_kv_heads` --
        notably, it does not depend on the number of query heads at all.
    """
    per_tensor = n_layers * seq_len * n_kv_heads * d_head * dtype_bytes
    return 2 * per_tensor  # separate K and V caches


# --- 6. Displaying attention matrices --------------------------------------------
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


# --- 7. Demos --------------------------------------------------------------------
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


def demo_split_combine_roundtrip(X, n_heads):
    """Verify `split_heads`/`combine_heads` are exact inverses of each other.

    Splitting into heads and immediately combining them back must return the
    original array unchanged -- the "one reshape and one transpose, no loop"
    claim the lesson makes about how cheap adding heads is.
    """
    print("=== 1. split_heads / combine_heads round-trip ===")
    split = split_heads(X, n_heads)
    roundtrip = combine_heads(split)
    print(f"Input shape:     {X.shape}")
    print(f"Split shape:     {split.shape}  (n_heads, n, d_head)")
    print(f"Round-trip PASS: {np.allclose(roundtrip, X)}")


def demo_mha(sentence, X, n_heads=4, seed=42):
    """Run multi-head attention and probe what each head attends to.

    Returns:
        `W_q`, the query projection, reused by `demo_gqa_vs_mqa` so the
        query side stays identical across the MHA/GQA/MQA comparison and
        only the K/V grouping changes.
    """
    print("\n=== 2. Multi-head attention ===")
    d_model = X.shape[-1]
    rng = np.random.default_rng(seed)
    scale = np.sqrt(2.0 / (d_model + d_model))
    W_q = rng.normal(0, scale, (d_model, d_model))
    W_k = rng.normal(0, scale, (d_model, d_model))
    W_v = rng.normal(0, scale, (d_model, d_model))
    W_o = rng.normal(0, scale, (d_model, d_model))

    output, weights = mha_forward(X, W_q, W_k, W_v, W_o, n_heads)
    print(f"Input shape:   {X.shape}")
    print(f"Output shape:  {output.shape}")
    print(f"Weights shape: {weights.shape}  (n_heads, n, n)")
    for h in range(n_heads):
        print_attention_weights(weights[h], sentence, title=f"\n-- Head {h} --")
    return W_q


def demo_gqa_vs_mqa(sentence, X, W_q, n_heads=4, seed=7):
    """Run GQA (n_kv_heads=2) and MQA (n_kv_heads=1) on the same input.

    Reuses `W_q` from `demo_mha` so the query-side projection is identical
    to the plain-MHA run; only how K/V are grouped changes.
    """
    d_model = X.shape[-1]
    d_head = d_model // n_heads
    rng = np.random.default_rng(seed)

    for label, n_kv_heads in (("GQA", 2), ("MQA", 1)):
        print(f"\n=== 3. {label} (n_heads={n_heads}, n_kv_heads={n_kv_heads}) ===")
        scale = np.sqrt(2.0 / (d_model + d_model))
        W_k = rng.normal(0, scale, (d_model, n_kv_heads * d_head))
        W_v = rng.normal(0, scale, (d_model, n_kv_heads * d_head))
        W_o = rng.normal(0, scale, (d_model, d_model))

        output, weights = gqa_forward(X, W_q, W_k, W_v, W_o, n_heads, n_kv_heads)
        repeat = n_heads // n_kv_heads
        print(f"W_k/W_v shape: {W_k.shape}  (vs. MHA's ({d_model}, {d_model}))")
        print(f"Output shape:  {output.shape}")
        print(f"Weights shape: {weights.shape}  (n_heads, n, n)")
        print(f"Every group of {repeat} consecutive query heads shares one K/V group")


def demo_kv_cache_savings():
    """Quantify the KV-cache memory MHA/GQA/MQA use for a Llama-3-70B-shaped
    config, reproducing the lesson's "8x cache shrink" claim with real numbers.
    """
    print("\n=== 4. KV-cache memory: MHA vs. GQA vs. MQA ===")
    n_layers, seq_len, d_head = 80, 2048, 128
    n_heads = 64
    configs = [("MHA", n_heads), ("GQA (Llama 3 70B)", 8), ("MQA", 1)]

    mha_bytes = kv_cache_bytes(n_layers, seq_len, n_heads, d_head)
    for label, n_kv_heads in configs:
        size = kv_cache_bytes(n_layers, seq_len, n_kv_heads, d_head)
        print(
            f"{label:20s} n_kv_heads={n_kv_heads:2d}  "
            f"cache={size / 2**30:6.2f} GiB  shrink={mha_bytes / size:4.1f}x"
        )


def demo_pytorch_equivalents(sentence, X):
    """Check shapes/mechanism against PyTorch's MHA and GQA-enabled SDPA.

    `nn.MultiheadAttention` is the one-line version of `mha_forward`.
    `scaled_dot_product_attention(..., enable_gqa=True)` (PyTorch 2.5+) is
    the one-line version of `gqa_forward` -- it accepts K/V with fewer heads
    than Q and repeats them internally, auto-dispatching to a fused kernel
    (e.g. Flash Attention on CUDA) instead of materializing the repeat.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("\n=== 5. Comparison against PyTorch ===")
    d_model = X.shape[-1]
    n_heads, n_kv_heads = 4, 2
    d_head = d_model // n_heads

    mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
    X_torch = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    output, attn_weights = mha(X_torch, X_torch, X_torch)
    print(
        f"nn.MultiheadAttention:  output={tuple(output.shape)}  "
        f"weights={tuple(attn_weights.shape)}"
    )

    q = torch.randn(1, n_heads, len(sentence), d_head)
    k = torch.randn(1, n_kv_heads, len(sentence), d_head)
    v = torch.randn(1, n_kv_heads, len(sentence), d_head)
    gqa_out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    print(
        f"scaled_dot_product_attention(enable_gqa=True): "
        f"q={tuple(q.shape)} k=v={tuple(k.shape)} -> out={tuple(gqa_out.shape)}"
    )


def main():
    """Run every demo end to end: split/combine round-trip, MHA head
    probing, GQA vs. MQA, KV-cache savings, and a PyTorch comparison."""
    sentence, X = make_toy_sentence()
    demo_split_combine_roundtrip(X, n_heads=4)
    W_q = demo_mha(sentence, X, n_heads=4)
    demo_gqa_vs_mqa(sentence, X, W_q, n_heads=4)
    demo_kv_cache_savings()
    demo_pytorch_equivalents(sentence, X)


if __name__ == "__main__":
    main()
