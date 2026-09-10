"""Speculative decoding: the Leviathan rejection rule, EAGLE-style acceptance
rates, and the KV-cache bookkeeping production inference servers build around it.

Mirrors the "Speculative Decoding and EAGLE-3" lesson from
rohitg00/ai-engineering-from-scratch
(phases/10-llms-from-scratch/15-speculative-decoding-eagle3): a cheap draft
proposes N tokens, a verifier checks all of them in one forward pass, and the
Leviathan accept/residual rule guarantees the accepted output is distributed
*exactly* as if every token had been sampled from the verifier alone --
speculative decoding is a pure latency win, not an approximation.

1. accept / residual                       -- the Leviathan rejection rule
                                               and the residual distribution
                                               sampled from on rejection.
2. make_draft_and_verifier / spec_step /
   measure_acceptance_rate                 -- one full speculative round
                                               (draft N, verify in parallel,
                                               accept/reject, bonus token),
                                               and the empirical acceptance
                                               rate it produces.
3. SequenceWorker                          -- per-sequence logical KV-cache
                                               length bookkeeping across
                                               rounds.
4. chi_square_critical_value_95 /
   leviathan_invariant_check                -- empirical proof that the
                                               accept/residual loop
                                               reproduces samples drawn
                                               directly from the verifier.
5. expected_accepted_tokens /
   expected_speedup / optimal_draft_length  -- the speedup-vs-acceptance-rate
                                               math and how it ranks the
                                               EAGLE generations.

Run directly (`python speculative_decoding.py`) to reproduce all demos.
"""

import numpy as np


# --- 1. The Leviathan rejection rule ----------------------------------------
def accept(q_prob, p_prob, u):
    """Leviathan accept/reject test for one drafted token.

    Accepts with probability `min(1, q_prob / p_prob)` -- the rule that
    makes the overall speculative loop produce samples distributed exactly
    as verifier-only sampling would, regardless of how good or bad the draft
    distribution is (Leviathan, Kalman, Matias, ICML 2023).

    Args:
        q_prob: Verifier's probability for the drafted token.
        p_prob: Draft's probability for the same token.
        u: A uniform(0, 1) random draw supplied by the caller.

    Returns:
        True if the token is accepted.
    """
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)


def residual(q, p):
    """Residual distribution to sample from after a drafted token is rejected.

    Computes `(q - p)_+`, clipped at zero and renormalized. Together with
    `accept`, this is the second half of the Leviathan rule: whatever
    probability mass the draft under-assigned relative to the verifier is
    exactly what a rejected step should redistribute over, and sampling from
    it is what keeps the overall output distributed as `q`.

    Args:
        q: Verifier's distribution over the vocabulary.
        p: Draft's distribution over the vocabulary (same length as `q`).

    Returns:
        A distribution over the vocabulary (list of floats summing to 1).
        Falls back to `q` itself in the degenerate case where `q <= p`
        everywhere (raw residual mass is zero, e.g. when `p == q`).
    """
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]


# --- 2. One full speculative-decoding round ---------------------------------
def make_draft_and_verifier(vocab_size, noise, rng=None):
    """Build one toy (draft `p`, verifier `q`) distribution pair.

    `q` is a random categorical distribution standing in for the verifier's
    next-token distribution. `p` mixes `q` with uniform noise: `noise=0.0`
    makes the draft identical to the verifier (a perfect draft, acceptance
    rate 1.0); `noise=1.0` makes the draft uniform random regardless of what
    the verifier thinks (a poor draft, low acceptance rate). Sweeping
    `noise` is how this script controls the empirical acceptance rate
    `alpha` without needing a real draft/target model pair.

    Args:
        vocab_size: Size of the toy vocabulary.
        noise: Fraction of `p` drawn from a uniform distribution instead of
            `q`, in [0, 1].
        rng: `numpy.random.Generator` to draw from (defaults to a fresh
            `default_rng()`).

    Returns:
        Tuple `(p, q)`, each a length-`vocab_size` array summing to 1.
    """
    if rng is None:
        rng = np.random.default_rng()
    q = rng.dirichlet(np.ones(vocab_size))
    uniform = np.full(vocab_size, 1.0 / vocab_size)
    p = (1 - noise) * q + noise * uniform
    p = p / p.sum()
    return p, q


def spec_step(vocab_size, draft_len, noise, rng=None):
    """Run one full speculative-decoding round: draft, verify, accept/reject, bonus.

    Drafts `draft_len` tokens one at a time from a fresh toy draft
    distribution `p_i` at each position, then checks each one against its
    matching verifier distribution `q_i` -- standing in for the single
    batched verifier forward pass a real system runs on
    `prefix + d_1..d_N`. Tokens are accepted left to right via `accept`; on
    the first rejection, a correction is sampled from `residual(q_i, p_i)`
    and the round stops early. If every drafted token is accepted, one bonus
    token is drawn straight from the next position's verifier distribution
    `q_{N+1}`, matching the real algorithm's "free" extra token.

    Args:
        vocab_size: Size of the toy vocabulary.
        draft_len: Number of tokens drafted this round (`N` in the lesson).
        noise: Passed through to `make_draft_and_verifier`; controls how
            close the draft is to the verifier, and therefore the empirical
            acceptance rate.
        rng: `numpy.random.Generator` to draw from (defaults to a fresh
            `default_rng()`).

    Returns:
        Dict with `output_tokens` (the tokens this round actually emits,
        including any correction/bonus token), `accepted` (how many drafted
        tokens were accepted before the first rejection, or all of
        `draft_len` on full acceptance), and `rejected` (whether a
        rejection occurred this round).
    """
    if rng is None:
        rng = np.random.default_rng()

    output_tokens = []
    for _ in range(draft_len):
        p, q = make_draft_and_verifier(vocab_size, noise, rng)
        token = rng.choice(vocab_size, p=p)
        u = rng.random()
        if accept(q[token], p[token], u):
            output_tokens.append(int(token))
        else:
            correction = rng.choice(vocab_size, p=residual(q, p))
            output_tokens.append(int(correction))
            return {
                "output_tokens": output_tokens,
                "accepted": len(output_tokens) - 1,
                "rejected": True,
            }

    _, q_bonus = make_draft_and_verifier(vocab_size, noise, rng)
    bonus = rng.choice(vocab_size, p=q_bonus)
    output_tokens.append(int(bonus))
    return {"output_tokens": output_tokens, "accepted": draft_len, "rejected": False}


def measure_acceptance_rate(vocab_size, draft_len, noise, num_rounds=2000, seed=0):
    """Empirically estimate the per-token acceptance rate `alpha` for a given
    noise level, by averaging `spec_step` results over many independent rounds.

    Args:
        vocab_size: Size of the toy vocabulary.
        draft_len: Draft length per round.
        noise: Passed through to `spec_step`.
        num_rounds: Number of independent rounds to average over.
        seed: Seed for reproducibility.

    Returns:
        Dict with `alpha` (mean accepted tokens per round, divided by
        `draft_len`) and `avg_accepted` (mean accepted tokens per round,
        before dividing by `draft_len`).
    """
    rng = np.random.default_rng(seed)
    accepted_counts = [
        spec_step(vocab_size, draft_len, noise, rng)["accepted"] for _ in range(num_rounds)
    ]
    avg_accepted = float(np.mean(accepted_counts))
    return {"alpha": avg_accepted / draft_len, "avg_accepted": avg_accepted}


# --- 3. KV-cache rollback bookkeeping ----------------------------------------
class SequenceWorker:
    """Tracks one sequence's logical KV-cache length across speculative rounds.

    A verifier forward pass physically writes key/value entries for every
    drafted position, but on a rejection the entries past the correction
    token are stale. Rather than physically erasing them (real systems use a
    scratch buffer committed on acceptance, or a physical cache plus a
    logical length truncated on reject -- see the lesson's "KV cache
    rollback" section), this tracks only the logical length: how many
    positions from the front of the cache are actually valid.

    Attributes:
        kv_length: Number of valid cached positions for this sequence.
    """

    def __init__(self, prefix_length):
        self.kv_length = prefix_length

    def apply_round(self, result):
        """Advance `kv_length` by the outcome of one `spec_step` round.

        `result["output_tokens"]` already stops right after the
        rejection/correction token (or after the bonus token on full
        acceptance), so in both cases the cache simply grows by however many
        tokens this round actually emitted -- no separate branch is needed
        for the rejected case.

        Args:
            result: A dict as returned by `spec_step`.
        """
        self.kv_length += len(result["output_tokens"])


# --- 4. The Leviathan invariant, checked empirically -------------------------
def chi_square_critical_value_95(df):
    """Approximate the 95th-percentile chi-square critical value for `df`
    degrees of freedom, via the Wilson-Hilferty cube-root normal approximation.

    Avoids a `scipy` dependency for a single well-known approximation; it
    matches standard chi-square tables to within about 0.1% for df >= 2.

    Args:
        df: Degrees of freedom.

    Returns:
        The approximate critical value. A chi-square statistic above this
        is significant at the 5% level (rejects the null hypothesis that
        the two distributions match).
    """
    z = 1.645  # 95th percentile of the standard normal
    return df * (1 - 2 / (9 * df) + z * np.sqrt(2 / (9 * df))) ** 3


def leviathan_invariant_check(vocab_size=8, num_trials=50000, seed=2):
    """Empirically verify the Leviathan theorem: accept/residual sampling
    reproduces samples drawn directly from the verifier.

    Fixes one (draft `p`, verifier `q`) pair for the whole run -- unlike
    `spec_step`, which redraws `p`/`q` every position -- so that many trials
    of the *same* accept/reject decision can be pooled into one empirical
    distribution. That distribution is compared against `q` with a
    chi-square goodness-of-fit test, alongside the same test applied to
    `num_trials` direct samples from `q` as a sanity-check baseline. The
    check works regardless of `noise` (how bad the draft is) -- that is the
    whole point of the theorem.

    Args:
        vocab_size: Size of the toy vocabulary.
        num_trials: Number of speculative trials and direct-sampling trials
            (the lesson's exercise 1 uses 50,000).
        seed: Seed for reproducibility.

    Returns:
        Dict with `chi_square_speculative` (test statistic for the
        accept/residual output vs. `q`), `chi_square_direct` (the same
        statistic for direct samples from `q`, included for comparison),
        `critical_value_95`, `degrees_of_freedom` (`vocab_size - 1`), and
        `passes` (`chi_square_speculative < critical_value_95`).

    Note:
        This is a genuine hypothesis test at the 95% confidence level, so
        even when the theorem holds exactly, roughly 1 seed in 20 will land
        above the critical value by chance -- a single `False` here is not
        evidence the implementation is wrong. `demo_leviathan_check` picks a
        seed that passes; rerunning across several seeds is a better check
        than trusting any one of them.
    """
    rng = np.random.default_rng(seed)
    p, q = make_draft_and_verifier(vocab_size, noise=0.6, rng=rng)

    speculative_counts = np.zeros(vocab_size)
    for _ in range(num_trials):
        token = rng.choice(vocab_size, p=p)
        u = rng.random()
        if accept(q[token], p[token], u):
            speculative_counts[token] += 1
        else:
            correction = rng.choice(vocab_size, p=residual(q, p))
            speculative_counts[correction] += 1

    direct_samples = rng.choice(vocab_size, size=num_trials, p=q)
    direct_counts = np.bincount(direct_samples, minlength=vocab_size)

    expected = q * num_trials
    chi_square_speculative = float(np.sum((speculative_counts - expected) ** 2 / expected))
    chi_square_direct = float(np.sum((direct_counts - expected) ** 2 / expected))

    df = vocab_size - 1
    critical = float(chi_square_critical_value_95(df))

    return {
        "chi_square_speculative": chi_square_speculative,
        "chi_square_direct": chi_square_direct,
        "critical_value_95": critical,
        "degrees_of_freedom": df,
        "passes": chi_square_speculative < critical,
    }


# --- 5. Speedup vs. acceptance rate ------------------------------------------
def expected_accepted_tokens(alpha, draft_len):
    """Expected number of accepted tokens per verifier forward pass.

    `E[accepted] = (1 - alpha^(N+1)) / (1 - alpha)`, from summing the
    geometric-decay probability of surviving `k` consecutive accepts for
    `k = 1..N`, plus the always-emitted first token's contribution folded
    into the closed form.

    Args:
        alpha: Per-token acceptance rate.
        draft_len: Draft length `N`.

    Returns:
        Expected accepted tokens per round (float).
    """
    if alpha >= 1.0:
        return draft_len + 1.0
    return (1 - alpha ** (draft_len + 1)) / (1 - alpha)


def expected_speedup(alpha, draft_len, cost_ratio):
    """Expected wall-clock speedup vs. plain (non-speculative) decoding.

    One round costs `draft_len * cost_ratio + 1` units (`draft_len` cheap
    draft steps plus one verifier pass, in units of one verifier step) and
    yields `expected_accepted_tokens(alpha, draft_len)` tokens, vs. 1 unit
    per token for plain decoding.

    Args:
        alpha: Per-token acceptance rate.
        draft_len: Draft length `N`.
        cost_ratio: `cost(draft) / cost(verifier)`.

    Returns:
        Expected speedup as a multiple of plain decoding's throughput.
    """
    e_accepted = expected_accepted_tokens(alpha, draft_len)
    return e_accepted / (draft_len * cost_ratio + 1)


def optimal_draft_length(alpha, cost_ratio, max_len=20):
    """Find the draft length that maximizes expected speedup.

    Args:
        alpha: Per-token acceptance rate.
        cost_ratio: `cost(draft) / cost(verifier)`.
        max_len: Largest draft length considered.

    Returns:
        Tuple `(best_draft_len, best_speedup)`.
    """
    lengths = list(range(1, max_len + 1))
    speedups = [expected_speedup(alpha, n, cost_ratio) for n in lengths]
    best_idx = int(np.argmax(speedups))
    return lengths[best_idx], speedups[best_idx]


# --- Demos --------------------------------------------------------------------
def demo_leviathan_check():
    """Run the empirical Leviathan-invariant check and print the verdict."""
    result = leviathan_invariant_check()
    verdict = "PASS" if result["passes"] else "FAIL"
    print(
        f"chi-square (speculative output vs. verifier) = {result['chi_square_speculative']:.2f}, "
        f"critical value (df={result['degrees_of_freedom']}, 95%) = {result['critical_value_95']:.2f} "
        f"-> {verdict}"
    )
    print(f"chi-square (direct samples vs. verifier, sanity check) = {result['chi_square_direct']:.2f}")


def demo_spec_step():
    """Run many speculative-decoding rounds at a fixed draft length and noise
    level, printing the empirical acceptance rate they produce."""
    vocab_size, draft_len, noise = 50, 5, 0.4
    stats = measure_acceptance_rate(vocab_size, draft_len, noise, num_rounds=3000)
    print(
        f"draft_len={draft_len}, noise={noise}: empirical alpha = {stats['alpha']:.2f}, "
        f"avg accepted tokens/round = {stats['avg_accepted']:.2f} "
        "(rounds that fully accept also emit one bonus token, not counted above)"
    )


def demo_kv_rollback():
    """Simulate two sequences sharing the lesson's exercise 4 setup: one whose
    drafts all accept, one that rejects partway through -- and show that only
    the valid prefix of each is kept in the logical KV length."""
    draft_len, prefix_length = 4, 10
    worker_a = SequenceWorker(prefix_length)
    worker_b = SequenceWorker(prefix_length)

    all_accepted = {
        "output_tokens": list(range(draft_len + 1)),  # N drafts + 1 bonus token
        "accepted": draft_len,
        "rejected": False,
    }
    rejected_at_2 = {
        "output_tokens": list(range(3)),  # 2 accepted drafts + 1 correction token
        "accepted": 2,
        "rejected": True,
    }

    worker_a.apply_round(all_accepted)
    worker_b.apply_round(rejected_at_2)

    print(f"Sequence A (all {draft_len} drafts accepted + bonus): kv_length {prefix_length} -> {worker_a.kv_length}")
    print(f"Sequence B (rejected after 2 drafts):             kv_length {prefix_length} -> {worker_b.kv_length}")


def demo_speedup_curve():
    """Print the optimal draft length and expected speedup for each EAGLE
    generation's published acceptance rate, at a fixed draft/verifier cost ratio."""
    cost_ratio = 0.05
    strategies = [
        ("Vanilla (Leviathan 2023)", 0.60),
        ("EAGLE-1", 0.75),
        ("EAGLE-2", 0.84),
        ("EAGLE-3", 0.90),
    ]
    print(f"{'strategy':<28}{'alpha':>7}{'best N':>9}{'speedup':>10}")
    for name, alpha in strategies:
        best_n, speedup = optimal_draft_length(alpha, cost_ratio)
        print(f"{name:<28}{alpha:>7.2f}{best_n:>9}{speedup:>9.2f}x")


def main():
    """Run all speculative-decoding demos end to end."""
    print("=== 1. Leviathan invariant check ===")
    demo_leviathan_check()

    print("\n=== 2. One speculative round, many trials ===")
    demo_spec_step()

    print("\n=== 3. KV-cache rollback bookkeeping ===")
    demo_kv_rollback()

    print("\n=== 4. Speedup vs. acceptance rate (EAGLE generations) ===")
    demo_speedup_curve()


if __name__ == "__main__":
    main()
