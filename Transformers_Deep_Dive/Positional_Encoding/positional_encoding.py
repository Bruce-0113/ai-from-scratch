"""Absolute sinusoidal positional encoding, RoPE (Rotary Position Embeddings),
and ALiBi (Attention with Linear Biases), built from scratch in pure Python
-- no NumPy/PyTorch needed, unlike the Self_Attention / Multi_Head_Attention
scripts next door.

Mirrors the "Positional Encoding" lesson from
rohitg00/ai-engineering-from-scratch
(phases/07-transformers-deep-dive/04-positional-encoding): attention itself
is permutation-invariant -- shuffle the input tokens and, with no positional
signal, `softmax(QK^T / sqrt(dk)) @ V` produces the same set of outputs,
just reordered. That is fatal for sequential data like language, so position
has to be injected somewhere. This script implements the three dominant
ways production models do it:

1. sinusoidal   -- absolute positional encoding added to the input
                    embeddings before the first attention layer:
                    PE[pos, 2i] = sin(pos / 10000^(2i/d)),
                    PE[pos, 2i+1] = cos(pos / 10000^(2i/d)).
                    Fixed (untrained), but a model never learns what to do
                    with positions past the max length it saw in training.
2. apply_rope   -- rotate each (Q, K) vector's consecutive dimension pairs
                    by a position-dependent angle theta_i = base^(-2i/d).
                    The dot product of a rotated query and key then depends
                    only on their *relative* distance (m - n), not their
                    absolute positions -- see `demo_rope_relative_property`.
3. alibi_bias   -- skip modifying Q/K/embeddings altogether and instead
                    subtract a per-head, distance-proportional penalty
                    directly from the attention scores before softmax:
                    scores[i, j] -= slope_h * |i - j|.

4. ascii_heatmap -- ASCII-art view of a position/bias matrix, reused across
                    all three demos.
5. demo_*        -- sinusoidal PE heatmap, RoPE rotation, RoPE's
                    relative-distance property (Build It Step 4), and
                    per-head ALiBi bias.

Run directly (`python positional_encoding.py`) to reproduce all demos.
"""

import math
import random
import sys


# --- 1. Absolute sinusoidal positional encoding ---------------------------------
def sinusoidal(N, d):
    """Build an (N, d) absolute sinusoidal positional encoding table.

    Args:
        N: Number of positions (sequence length) to encode.
        d: Embedding dimension. Should be even -- each pair of dimensions
            (2i, 2i+1) shares one frequency, `2i` getting sin and `2i+1`
            getting cos. If `d` is odd, the last column is left at 0.0
            (it has no cos partner).

    Returns:
        List of N rows, each a list of d floats. Add this directly to the
        input embeddings (`X' = X + PE[:N]`) before the first attention
        layer. It is a fixed table -- nothing here is learned -- so a model
        trained with `N` up to some `max_len` has no learned behavior for
        positions beyond it (see `demo_sinusoidal` and the README's notes on
        extrapolation).
    """
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe


# --- 2. RoPE: rotate Q/K by position-dependent angles ---------------------------
def apply_rope(x, pos, base=10000):
    """Rotate a Q or K vector by an angle that grows with `pos`.

    Each consecutive pair of dimensions (x[2i], x[2i+1]) is treated as a 2D
    point and rotated by `theta_i = pos / base**(2i/d)` radians:
        out[2i]   = x[2i] * cos(theta_i) - x[2i+1] * sin(theta_i)
        out[2i+1] = x[2i] * sin(theta_i) + x[2i+1] * cos(theta_i)

    Args:
        x: Query or key vector for a single token, as a list of floats.
            `len(x)` should be even -- there is no cos partner for a
            trailing odd dimension.
        pos: Absolute position of this token in the sequence. Applying this
            to Q at position `m` and K at position `n` *separately* (not to
            their dot product) is what makes the resulting `q' . k'` depend
            only on `m - n` -- see `demo_rope_relative_property`.
        base: Frequency base; larger `base` slows the rotation of the
            high-index (already-slow) pairs even further, which is the
            single knob NTK-aware / YaRN scaling adjust to extend a
            RoPE model's usable context length without retraining from
            scratch.

    Returns:
        Rotated vector, a new list the same length as `x`.
    """
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out


# --- 3. ALiBi: linear distance bias on attention scores --------------------------
def alibi_bias(n_heads, seq_len):
    """Build the per-head ALiBi bias matrix subtracted from attention scores.

    Args:
        n_heads: Number of attention heads. Each head `h` (1-indexed, per
            the original paper) gets its own slope
            `m_h = 2 ** (-8 * h / n_heads)`, geometrically spaced so
            different heads penalize distance at very different rates --
            some heads stay almost local, others look nearly uniform.
        seq_len: Sequence length; the bias matrix is (seq_len, seq_len).

    Returns:
        List of `n_heads` matrices, each `seq_len x seq_len`, where
        `bias[h][i][j] = -slope_h * abs(i - j)`. Add `bias[h]` to head h's
        raw attention scores *before* softmax -- no change to Q, K, V, or
        the embeddings themselves, which is why ALiBi needs zero extra
        training cost and extrapolates well past the trained sequence
        length (the penalty is just a linear function of distance, defined
        for any `i, j`, not a fixed-size table like `sinusoidal`).
    """
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax


# --- 4. Displaying matrices -------------------------------------------------------
def ascii_heatmap(matrix, chars=" ░▒▓█"):
    """Render a 2D matrix of floats as an ASCII-art heatmap, one row per line.

    Args:
        matrix: Sequence of rows, each a sequence of floats. Values can be
            of either sign (e.g. sinusoidal's [-1, 1] or ALiBi's <= 0 bias)
            -- shading is by each value's position between the matrix's own
            min and max, not by magnitude alone.
        chars: Shading characters from lightest (lowest value) to darkest
            (highest value).
    """
    flat = [v for row in matrix for v in row]
    lo, hi = min(flat), max(flat)
    span = hi - lo or 1.0
    for row in matrix:
        line = "".join(chars[int((v - lo) / span * (len(chars) - 1))] for v in row)
        print(line)


# --- 5. Demos ----------------------------------------------------------------------
def demo_sinusoidal(seq_len=16, d_model=8):
    """Build a sinusoidal PE table and show the stripe-widening pattern.

    Reproduces Exercise 1 ("visualize the PE matrix as a heatmap"): low
    dimensions (small i) complete a sin/cos cycle every few positions, while
    high dimensions oscillate far more slowly -- so the heatmap shows narrow
    stripes on the left widening into broad, slow-changing bands on the
    right.
    """
    print("=== 1. Sinusoidal positional encoding ===")
    pe = sinusoidal(seq_len, d_model)
    print(f"PE shape: ({seq_len}, {d_model})\n")
    ascii_heatmap(pe)
    print(f"\nPE[0] (position 0): {[f'{v:.3f}' for v in pe[0]]}")
    print(f"PE[1] (position 1): {[f'{v:.3f}' for v in pe[1]]}")


def demo_rope(dim=8, seed=42):
    """Rotate one toy vector at increasing positions and show the effect.

    `pos=0` rotates every pair by angle 0 (cos=1, sin=0), so the output
    should equal the input exactly -- a quick sanity check that `apply_rope`
    is a true rotation, not an arbitrary transform.
    """
    print("\n=== 2. RoPE: rotating a vector at different positions ===")
    rng = random.Random(seed)
    x = [rng.uniform(-1, 1) for _ in range(dim)]
    print("Original vector (pos=0 should leave this unchanged):")
    print(f"  x            = {[f'{v:.3f}' for v in x]}")
    for pos in (0, 1, 8, 64):
        rotated = apply_rope(x, pos)
        print(f"  rotated(pos={pos:2d}) = {[f'{v:.3f}' for v in rotated]}")


def demo_rope_relative_property(dim=8, shift=100, seed=42):
    """Build It Step 4: confirm a RoPE attention score depends only on m - n.

    Rotates the same (q, k) pair at (pos_a, pos_b) and again at
    (pos_a + shift, pos_b + shift). If RoPE truly encodes relative distance,
    `q' . k'` must match in both cases (within floating-point tolerance)
    even though every absolute position changed -- this is the property
    that lets a RoPE model's attention behavior generalize to positions it
    never saw during training.
    """
    print("\n=== 3. RoPE relative-distance property ===")
    rng = random.Random(seed)
    q = [rng.uniform(-1, 1) for _ in range(dim)]
    k = [rng.uniform(-1, 1) for _ in range(dim)]
    pos_a, pos_b = 5, 2

    def rotated_score(pos_q, pos_k):
        q_rot = apply_rope(q, pos_q)
        k_rot = apply_rope(k, pos_k)
        return sum(a * b for a, b in zip(q_rot, k_rot))

    score_before = rotated_score(pos_a, pos_b)
    score_after = rotated_score(pos_a + shift, pos_b + shift)
    print(f"(pos_q={pos_a:3d}, pos_k={pos_b:3d})                     m-n={pos_a - pos_b}   q'.k' = {score_before:.6f}")
    print(f"(pos_q={pos_a + shift:3d}, pos_k={pos_b + shift:3d})  (shift={shift})  m-n={pos_a - pos_b}   q'.k' = {score_after:.6f}")
    print(f"Relative-distance invariance PASS: {math.isclose(score_before, score_after, abs_tol=1e-9)}")


def demo_alibi(n_heads=4, seq_len=8):
    """Build It Step 3: compute per-head ALiBi bias and show one head's matrix.

    Prints every head's slope (geometrically spaced per `alibi_bias`) plus
    an ASCII heatmap of head 0's bias matrix, where the diagonal (distance
    0) is lightest and the far corners (largest |i - j|) are darkest --
    the "closer tokens get less penalty" shape ALiBi adds before softmax.
    """
    print("\n=== 4. ALiBi: per-head linear distance bias ===")
    bias = alibi_bias(n_heads, seq_len)
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    print(f"n_heads={n_heads}  seq_len={seq_len}")
    print(f"slopes: {[f'{s:.4f}' for s in slopes]}\n")
    print(f"Bias matrix for head 0 (slope={slopes[0]:.4f}), darker = more negative:")
    ascii_heatmap(bias[0])
    print(f"\nrow 0 (query position 0): {[f'{v:.2f}' for v in bias[0][0]]}")


def main():
    """Run every demo end to end: sinusoidal PE, RoPE rotation, RoPE's
    relative-distance property, and per-head ALiBi bias."""
    demo_sinusoidal()
    demo_rope()
    demo_rope_relative_property()
    demo_alibi()


if __name__ == "__main__":
    # ascii_heatmap prints block-shading characters (░▒▓█); force UTF-8 so
    # this doesn't crash under a non-UTF-8 console codepage (e.g. Windows cp950).
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
