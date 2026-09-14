"""The full transformer block skeleton -- encoder + decoder -- built from
scratch in pure Python (a tiny row-major `Matrix` class, no NumPy/PyTorch
needed for the forward pass).

Mirrors the "The Full Transformer -- Encoder + Decoder" lesson from
rohitg00/ai-engineering-from-scratch
(phases/07-transformers-deep-dive/05-full-transformer), which builds on
`../Self_Attention/`, `../Multi_Head_Attention/`, and `../Positional_Encoding/`:
one attention layer is a feature extractor, not a model. Turning it into
something that trains past a handful of layers takes six pieces wired
together -- embeddings + position, self-attention, a feed-forward network,
residual connections, normalization, and (for decoders) cross-attention.
Every transformer since 2017 -- encoder-only (BERT), decoder-only (GPT),
encoder-decoder (T5) -- inherits this same skeleton; only the norm/FFN/
position choices inside it changed by 2026.

This script builds both the 2017 classic block (LayerNorm, ReLU-FFN) and the
2026 modernized block (RMSNorm, SwiGLU-FFN) behind the *same*
`encoder_block` / `decoder_block` wiring, then runs both end to end on a toy
source/target pair through a final projection to vocabulary logits.

1. Matrix / randn / matmul /         -- tiny row-major matrix type and the
   transpose / add                      linear-algebra primitives every
                                        other section builds on
2. softmax_rows                      -- shared softmax helper, with an
                                        optional mask for causal attention
3. layer_norm / rms_norm             -- LayerNorm (2017) vs. RMSNorm (2026):
                                        same job (stabilize the residual
                                        stream), RMSNorm skips mean-centering
4. silu / ffn_relu / ffn_swiglu      -- FFN (2017 ReLU) vs. FFN (2026 gated
                                        SwiGLU)
5. scaled_dot_product_attention /    -- the attention primitive from
   multi_head_attention                 Self_Attention / Multi_Head_Attention,
                                        reused as-is for self-attention
                                        (causal or not) and cross-attention
6. BlockParams /                     -- per-block weight bundles (cross-attn
   CrossAttentionParams                 weights are separate: only decoder
                                        blocks need them)
7. encoder_block / decoder_block     -- the six pieces wired into one
                                        stackable block each
8. output_projection                 -- final norm + projection to
                                        vocabulary logits
9. demo_*                            -- 2-layer encoder+decoder stack, run
                                        twice (2017 classic vs. 2026 modern
                                        config) on the same toy input

Run directly (`python transformer.py`) to reproduce all demos.
"""

import math
import random


# --- 1. Matrix and linear algebra primitives ------------------------------------
class Matrix:
    """A flat, row-major 2D array of floats -- this script's stand-in for a
    tensor, with only the operations the transformer block needs.

    Attributes:
        rows: Number of rows.
        cols: Number of columns.
        data: Flat list of `rows * cols` floats, row-major (`data[i*cols+j]`
            is element `(i, j)`).
    """

    __slots__ = ("rows", "cols", "data")

    def __init__(self, rows, cols, fill=0.0, data=None):
        """Args:
            rows: Number of rows.
            cols: Number of columns.
            fill: Value every cell starts at, when `data` is not given.
            data: Optional flat, row-major list of `rows * cols` floats to
                use directly instead of allocating a filled one.
        """
        self.rows = rows
        self.cols = cols
        self.data = data if data is not None else [fill] * (rows * cols)

    def get(self, i, j):
        """Element at row `i`, column `j`."""
        return self.data[i * self.cols + j]

    def set(self, i, j, v):
        """Set element at row `i`, column `j` to `v`."""
        self.data[i * self.cols + j] = v

    def row(self, i):
        """Row `i` as a plain list of `cols` floats."""
        return self.data[i * self.cols:(i + 1) * self.cols]

    def copy(self):
        """A copy with the same shape and an independent `data` list."""
        return Matrix(self.rows, self.cols, data=list(self.data))


def randn(rows, cols, rng, scale=None):
    """Random (rows, cols) `Matrix` for weight initialization.

    Args:
        rows: Number of rows.
        cols: Number of columns.
        rng: `random.Random` instance to draw from (kept explicit, not
            module-global, so callers can reproduce each stack's weights
            independently -- see `demo_transformer_stack`).
        scale: Standard deviation of the Gaussian draw. Defaults to
            `sqrt(2 / (rows + cols))`, a Xavier/Glorot-style scale that
            keeps activations from exploding or vanishing as more blocks
            are stacked.

    Returns:
        A new `Matrix` of independent `N(0, scale^2)` samples.
    """
    if scale is None:
        scale = math.sqrt(2.0 / (rows + cols))
    m = Matrix(rows, cols)
    for i in range(rows * cols):
        m.data[i] = rng.gauss(0.0, scale)
    return m


def matmul(A, B):
    """Matrix product `A @ B`.

    Args:
        A: Left matrix, shape (m, k).
        B: Right matrix, shape (k, n). `A.cols` must equal `B.rows`.

    Returns:
        Product, shape (m, n).
    """
    out = Matrix(A.rows, B.cols)
    for i in range(A.rows):
        for k in range(A.cols):
            aik = A.get(i, k)
            if aik == 0.0:
                continue
            base_i = i * B.cols
            base_k = k * B.cols
            for j in range(B.cols):
                out.data[base_i + j] += aik * B.data[base_k + j]
    return out


def transpose(A):
    """Transpose: shape (rows, cols) -> (cols, rows)."""
    out = Matrix(A.cols, A.rows)
    for i in range(A.rows):
        for j in range(A.cols):
            out.set(j, i, A.get(i, j))
    return out


def add(A, B):
    """Elementwise sum of two same-shaped matrices.

    This is the residual connection: `x + sublayer(x)`. Without it,
    gradients vanish past roughly six stacked layers.
    """
    assert (A.rows, A.cols) == (B.rows, B.cols)
    return Matrix(A.rows, A.cols, data=[a + b for a, b in zip(A.data, B.data)])


# --- 2. Softmax ------------------------------------------------------------------
def softmax_rows(A, mask=None):
    """Row-wise softmax, shifted by each row's max for numerical stability.

    Args:
        A: Scores, shape (n, m) -- e.g. raw attention scores `QK^T / sqrt(dk)`.
        mask: Optional (n, m) nested list of booleans; `mask[i][j]` truthy
            means position `(i, j)` is forced to 0 probability. Used for the
            decoder's causal mask (see `scaled_dot_product_attention`).

    Returns:
        Matrix the same shape as `A`, each row summing to 1.
    """
    out = Matrix(A.rows, A.cols)
    for i in range(A.rows):
        row = A.row(i)
        if mask is not None:
            row = [row[j] if not mask[i][j] else float("-inf") for j in range(A.cols)]
        m = max(v for v in row if v != float("-inf"))
        exps = [math.exp(v - m) if v != float("-inf") else 0.0 for v in row]
        s = sum(exps)
        for j, e in enumerate(exps):
            out.set(i, j, e / s if s > 0 else 0.0)
    return out


# --- 3. Normalization: LayerNorm (2017) vs. RMSNorm (2026) -----------------------
def layer_norm(X, eps=1e-5):
    """LayerNorm (2017): per-row mean-center, then divide by standard deviation.

    Args:
        X: Input, shape (n, d) -- normalized independently per row (per
            token), across the feature dimension.
        eps: Added to the variance before the square root, to avoid
            dividing by ~0 on a near-constant row.

    Returns:
        Normalized matrix, same shape as `X`; each row is zero-mean and
        unit-variance.
    """
    out = Matrix(X.rows, X.cols)
    for i in range(X.rows):
        row = X.row(i)
        mean = sum(row) / len(row)
        var = sum((v - mean) ** 2 for v in row) / len(row)
        denom = math.sqrt(var + eps)
        for j in range(X.cols):
            out.set(i, j, (row[j] - mean) / denom)
    return out


def rms_norm(X, eps=1e-6):
    """RMSNorm (2026 default): per-row divide by root-mean-square, no mean-centering.

    One fewer op than `layer_norm` (no mean to compute or subtract) and
    empirically at least as stable (Zhang & Sennrich, 2019) -- why 2026
    stacks (Llama, Qwen, Mistral, ...) use RMSNorm instead of LayerNorm.

    Args:
        X: Input, shape (n, d).
        eps: Added inside the square root, to avoid dividing by ~0.

    Returns:
        Normalized matrix, same shape as `X`; each row's RMS is scaled to 1.
    """
    out = Matrix(X.rows, X.cols)
    for i in range(X.rows):
        row = X.row(i)
        rms = math.sqrt(sum(v * v for v in row) / len(row) + eps)
        for j in range(X.cols):
            out.set(i, j, row[j] / rms)
    return out


# --- 4. Feed-forward network: ReLU (2017) vs. SwiGLU (2026) ----------------------
def silu(x):
    """SiLU/Swish activation: `x * sigmoid(x)`, the gating half of SwiGLU."""
    return x / (1.0 + math.exp(-x))


def ffn_swiglu(X, W1, W2, W3):
    """SwiGLU feed-forward (2026 default): `(SiLU(X W1) * (X W3)) W2`.

    Three matrices instead of ReLU-FFN's two: `W1`/`W3` both project up to
    the hidden dimension, one gated through SiLU and multiplied elementwise
    against the other, before `W2` projects back down. Beats ReLU/GELU FFN
    by ~0.5 ppl point in the Llama/PaLM/Qwen papers (Shazeer, 2020).

    Args:
        X: Input, shape (n, d).
        W1, W3: Up-projections, each shape (d, h).
        W2: Down-projection, shape (h, d).

    Returns:
        Output, shape (n, d).
    """
    h1 = matmul(X, W1)
    h3 = matmul(X, W3)
    gated = Matrix(h1.rows, h1.cols)
    for i in range(len(h1.data)):
        gated.data[i] = silu(h1.data[i]) * h3.data[i]
    return matmul(gated, W2)


def ffn_relu(X, W1, W2):
    """ReLU feed-forward (2017 default): `relu(X W1) W2`.

    Args:
        X: Input, shape (n, d).
        W1: Up-projection, shape (d, h).
        W2: Down-projection, shape (h, d).

    Returns:
        Output, shape (n, d).
    """
    h = matmul(X, W1)
    for i in range(len(h.data)):
        if h.data[i] < 0:
            h.data[i] = 0.0
    return matmul(h, W2)


# --- 5. Attention: shared by self-attention and cross-attention ------------------
def scaled_dot_product_attention(Q, K, V, causal=False):
    """Scaled dot-product attention for one head: `softmax(QK^T / sqrt(dk)) V`.

    Args:
        Q: Queries, shape (n_q, dk).
        K: Keys, shape (n_kv, dk).
        V: Values, shape (n_kv, dv).
        causal: If True, mask position `(i, j)` for every `j > i` so query
            `i` cannot attend to future key positions -- the decoder's
            masked self-attention sublayer. Encoder self-attention and
            cross-attention both leave this False (bidirectional).

    Returns:
        Attention output, shape (n_q, dv).
    """
    dk = Q.cols
    scores = matmul(Q, transpose(K))
    inv = 1.0 / math.sqrt(dk)
    for i in range(len(scores.data)):
        scores.data[i] *= inv
    mask = None
    if causal:
        mask = [[j > i for j in range(scores.cols)] for i in range(scores.rows)]
    w = softmax_rows(scores, mask=mask)
    return matmul(w, V)


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=False, kv_source=None):
    """Multi-head attention: project to Q/K/V, split into heads, attend, merge.

    Reused as-is for every sublayer type in this file: self-attention
    (`kv_source=None`, K/V come from `X`), masked self-attention
    (`causal=True`), and cross-attention (`kv_source=enc_out`, Q comes from
    the decoder but K/V come from the encoder output).

    Args:
        X: Query-side input, shape (n_q, d) -- always the source of Q.
        Wq, Wk, Wv: Q/K/V projections, each shape (d, d).
        Wo: Output projection, shape (d, d), applied after heads are
            concatenated -- the only place heads are allowed to mix.
        n_heads: Number of attention heads; must divide `d` evenly.
        causal: Passed through to `scaled_dot_product_attention` for every
            head.
        kv_source: If given, K/V are projected from this instead of `X` --
            cross-attention's encoder output. Its row count may differ from
            `X`'s (e.g. `src_len != tgt_len`).

    Returns:
        Output, shape (n_q, d).
    """
    Q = matmul(X, Wq)
    kv_input = kv_source if kv_source is not None else X
    K = matmul(kv_input, Wk)
    V = matmul(kv_input, Wv)
    d_head = Q.cols // n_heads
    head_outs = []
    for h in range(n_heads):
        Qh = Matrix(Q.rows, d_head, data=[Q.get(i, h * d_head + j) for i in range(Q.rows) for j in range(d_head)])
        Kh = Matrix(K.rows, d_head, data=[K.get(i, h * d_head + j) for i in range(K.rows) for j in range(d_head)])
        Vh = Matrix(V.rows, d_head, data=[V.get(i, h * d_head + j) for i in range(V.rows) for j in range(d_head)])
        head_outs.append(scaled_dot_product_attention(Qh, Kh, Vh, causal=causal))
    concat = Matrix(X.rows, Q.cols)
    for h, H in enumerate(head_outs):
        for i in range(H.rows):
            for j in range(d_head):
                concat.set(i, h * d_head + j, H.get(i, j))
    return matmul(concat, Wo)


# --- 6. Per-block weight bundles --------------------------------------------------
class BlockParams:
    """Self-attention + FFN weights shared by one encoder or decoder block.

    Args:
        d: Model/embedding dimension.
        n_heads: Number of self-attention heads; must divide `d` evenly.
        ffn_expansion: FFN hidden-dim ratio (`h = int(d * ffn_expansion)`).
            2017 default is 4x with ReLU; SwiGLU typically uses ~2.6x to
            match total parameter count (three matrices instead of two) --
            this class doesn't enforce that, callers pick `ffn_expansion`
            to match `use_swiglu`.
        rng: `random.Random` instance driving weight initialization.
        use_swiglu: FFN variant used by `encoder_block` / `decoder_block`:
            SwiGLU (2026 default, needs `W1`/`W2`/`W3`) if True, plain
            ReLU-FFN (2017, needs `W1`/`W2` only) if False.
        use_rmsnorm: Normalization variant used by `encoder_block` /
            `decoder_block`: RMSNorm (2026 default) if True, LayerNorm
            (2017) if False.
    """

    def __init__(self, d, n_heads, ffn_expansion, rng, use_swiglu=True, use_rmsnorm=True):
        self.d = d
        self.n_heads = n_heads
        self.use_swiglu = use_swiglu
        self.use_rmsnorm = use_rmsnorm
        self.Wq = randn(d, d, rng)
        self.Wk = randn(d, d, rng)
        self.Wv = randn(d, d, rng)
        self.Wo = randn(d, d, rng)
        h = int(d * ffn_expansion)
        if use_swiglu:
            self.W1 = randn(d, h, rng)
            self.W2 = randn(h, d, rng)
            self.W3 = randn(d, h, rng)
        else:
            self.W1 = randn(d, h, rng)
            self.W2 = randn(h, d, rng)


class CrossAttentionParams:
    """Extra Q/K/V/O weights for a decoder block's cross-attention sublayer.

    Only decoder blocks need this: queries come from the decoder's own
    (already self-attended) hidden state, but keys/values come from the
    encoder output, so cross-attention cannot reuse `BlockParams`'
    self-attention weights.

    Args:
        d: Model/embedding dimension.
        rng: `random.Random` instance driving weight initialization.
    """

    def __init__(self, d, rng):
        self.Wq = randn(d, d, rng)
        self.Wk = randn(d, d, rng)
        self.Wv = randn(d, d, rng)
        self.Wo = randn(d, d, rng)


# --- 7. Blocks: the six pieces wired together -------------------------------------
def encoder_block(x, p):
    """One bidirectional encoder block: self-attention + FFN, pre-norm, residual.

        x -> norm -> MHA(self) -> +x -> norm -> FFN -> +x -> out

    No masking -- every position attends to every position.

    Args:
        x: Input hidden states, shape (n, d).
        p: `BlockParams` for this block; `p.use_rmsnorm` / `p.use_swiglu`
            select which norm/FFN variant runs.

    Returns:
        Output hidden states, shape (n, d).
    """
    norm = rms_norm if p.use_rmsnorm else layer_norm
    h = norm(x)
    a = multi_head_attention(h, p.Wq, p.Wk, p.Wv, p.Wo, p.n_heads)
    x = add(x, a)
    h = norm(x)
    f = ffn_swiglu(h, p.W1, p.W2, p.W3) if p.use_swiglu else ffn_relu(h, p.W1, p.W2)
    return add(x, f)


def decoder_block(x, enc_out, p, cross):
    """One decoder block: masked self-attention, cross-attention, FFN; pre-norm, residual.

        x -> norm -> MHA(masked self) -> +x -> norm -> MHA(cross) -> +x -> norm -> FFN -> +x -> out

    Three sublayers instead of the encoder's two -- the middle
    cross-attention sublayer is the only place information flows from the
    encoder into the decoder. A decoder-only architecture (GPT) drops it
    and keeps just masked self-attention + FFN.

    Args:
        x: Decoder input hidden states, shape (tgt_n, d).
        enc_out: Encoder output, shape (src_n, d) -- supplies K/V for
            cross-attention; `enc_out.rows` need not equal `x.rows`.
        p: `BlockParams` for this block's self-attention + FFN sublayers.
        cross: `CrossAttentionParams` for this block's cross-attention
            sublayer.

    Returns:
        Output hidden states, shape (tgt_n, d).
    """
    norm = rms_norm if p.use_rmsnorm else layer_norm
    h = norm(x)
    a = multi_head_attention(h, p.Wq, p.Wk, p.Wv, p.Wo, p.n_heads, causal=True)
    x = add(x, a)
    h = norm(x)
    a = multi_head_attention(h, cross.Wq, cross.Wk, cross.Wv, cross.Wo, p.n_heads, kv_source=enc_out)
    x = add(x, a)
    h = norm(x)
    f = ffn_swiglu(h, p.W1, p.W2, p.W3) if p.use_swiglu else ffn_relu(h, p.W1, p.W2)
    return add(x, f)


# --- 8. Output head ----------------------------------------------------------------
def output_projection(x, norm_fn, W_out):
    """Final norm + linear projection to vocabulary logits.

    Args:
        x: Final decoder hidden states, shape (n, d).
        norm_fn: `layer_norm` or `rms_norm`, matching the stack's norm
            choice -- applied once more before the projection.
        W_out: Output projection, shape (d, vocab_size).

    Returns:
        Logits, shape (n, vocab_size): one row per target position, ready
        for a softmax/argmax over the vocabulary (no loss/training here).
    """
    return matmul(norm_fn(x), W_out)


def run_transformer(src, tgt, enc_params, dec_params, cross_params, norm_fn, W_out):
    """Run a full encoder+decoder forward pass, ending in vocabulary logits.

    Args:
        src: Source hidden states, shape (src_n, d) -- stands in for token
            embeddings with positional signal already added (see
            `../Positional_Encoding/`; kept out of scope here).
        tgt: Target hidden states, shape (tgt_n, d), same caveat as `src`.
        enc_params: List of `BlockParams`, one per encoder layer.
        dec_params: List of `BlockParams`, one per decoder layer.
        cross_params: List of `CrossAttentionParams`, one per decoder layer
            (same length as `dec_params`).
        norm_fn: `layer_norm` or `rms_norm`, applied as the final norm
            before `W_out`.
        W_out: Output projection, shape (d, vocab_size).

    Returns:
        Tuple of (enc_out, dec_out, logits):
            enc_out: Encoder output, shape (src_n, d).
            dec_out: Decoder output (pre-final-norm), shape (tgt_n, d).
            logits: Vocabulary logits, shape (tgt_n, vocab_size).
    """
    enc_out = src
    for p in enc_params:
        enc_out = encoder_block(enc_out, p)

    dec_out = tgt
    for p, cross in zip(dec_params, cross_params):
        dec_out = decoder_block(dec_out, enc_out, p, cross)

    logits = output_projection(dec_out, norm_fn, W_out)
    return enc_out, dec_out, logits


# --- 9. Demo -------------------------------------------------------------------------
def demo_transformer_stack(label, src, tgt, d, n_heads, ffn_expansion, n_layers,
                            vocab_size, use_swiglu, use_rmsnorm, seed):
    """Build one 2-layer encoder+decoder stack and run it end to end.

    Args:
        label: Printed heading, e.g. "2017 classic" or "2026 modern".
        src, tgt: Toy source/target hidden states (see `run_transformer`).
        d, n_heads, ffn_expansion, n_layers: Stack hyperparameters.
        vocab_size: Output projection's vocabulary size.
        use_swiglu: FFN variant, threaded into every `BlockParams`.
        use_rmsnorm: Norm variant, threaded into every `BlockParams`.
        seed: RNG seed for this stack's weights, kept separate per stack so
            the classic/modern comparison isn't confounded by shared random
            weights -- only the norm/FFN choice differs.

    Returns:
        Tuple of (enc_out, dec_out, logits), see `run_transformer`.
    """
    rng = random.Random(seed)
    enc_params = [BlockParams(d, n_heads, ffn_expansion, rng, use_swiglu, use_rmsnorm)
                  for _ in range(n_layers)]
    dec_params = [BlockParams(d, n_heads, ffn_expansion, rng, use_swiglu, use_rmsnorm)
                  for _ in range(n_layers)]
    cross_params = [CrossAttentionParams(d, rng) for _ in range(n_layers)]
    norm_fn = rms_norm if use_rmsnorm else layer_norm
    W_out = randn(d, vocab_size, rng)

    enc_out, dec_out, logits = run_transformer(
        src, tgt, enc_params, dec_params, cross_params, norm_fn, W_out
    )

    norm_name = "RMSNorm" if use_rmsnorm else "LayerNorm"
    ffn_name = "SwiGLU" if use_swiglu else "ReLU"
    print(f"=== {label}: {norm_name} + {ffn_name}-FFN ===")
    print(f"encoder output shape: ({enc_out.rows}, {enc_out.cols})")
    print(f"decoder output shape: ({dec_out.rows}, {dec_out.cols})")
    print(f"logits shape:         ({logits.rows}, {logits.cols})  (tgt_len, vocab_size)")
    print("decoder output, row 0 (first 4 dims): "
          + "  ".join(f"{v:+.3f}" for v in dec_out.row(0)[:4]))
    print("logits, row 0 (first 4 of vocab):     "
          + "  ".join(f"{v:+.3f}" for v in logits.row(0)[:4]))
    return enc_out, dec_out, logits


def main():
    """Run the 2017-classic and 2026-modern block configs on the same toy
    source/target pair and confirm both produce matching shapes end to end
    with only the norm/FFN functions swapped."""
    d = 8
    n_heads = 2
    src_len, tgt_len = 6, 5
    n_layers = 2
    vocab_size = 12

    data_rng = random.Random(42)
    src = randn(src_len, d, data_rng, scale=0.5)
    tgt = randn(tgt_len, d, data_rng, scale=0.5)

    demo_transformer_stack(
        "2017 classic", src, tgt, d, n_heads, ffn_expansion=4.0,
        n_layers=n_layers, vocab_size=vocab_size,
        use_swiglu=False, use_rmsnorm=False, seed=1,
    )
    print()
    demo_transformer_stack(
        "2026 modern", src, tgt, d, n_heads, ffn_expansion=2.6,
        n_layers=n_layers, vocab_size=vocab_size,
        use_swiglu=True, use_rmsnorm=True, seed=2,
    )
    print()
    print("Both configs share the same encoder_block/decoder_block wiring --")
    print("only the norm (LayerNorm vs. RMSNorm) and FFN (ReLU vs. SwiGLU)")
    print("functions differ, and shapes match end to end.")


if __name__ == "__main__":
    main()
