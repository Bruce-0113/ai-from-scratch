"""LLM inference-serving optimizations: KV caching, batching, prefix caching,
and speculative decoding.

Mirrors the "Inference Optimization" lesson from rohitg00/ai-engineering-from-scratch
(phases/10-llms-from-scratch/12-inference-optimization): prefill is compute-bound
(the whole prompt is processed in parallel), decode is memory-bound (one token
at a time, paying the full cost of streaming model weights from memory for a
single FLOP-cheap step) -- every technique below targets one side of that split.

1. KVCache / MultiHeadAttention        -- avoid recomputing key/value
                                           projections for already-seen
                                           tokens; the core decode-side fix.
2. Request / simulate_*_batching /
   batching_stats                      -- static batching (wait for the
                                           slowest request in a fixed batch)
                                           vs. continuous batching (slot in a
                                           new request the moment one slot
                                           frees up).
3. TrieNode / PrefixCache              -- reuse cached KV state for a shared
                                           prompt prefix across requests (the
                                           RadixAttention/SGLang idea).
4. DraftModel / TargetModel /
   speculative_decode /
   compare_speculation_strategies      -- a cheap draft model proposes several
                                           tokens, a target model verifies them
                                           in one forward pass.
5. MODEL_CONFIGS / kv_cache_memory /
   memory_budget                       -- capacity planning: how much KV
                                           cache a model leaves room for on a
                                           given GPU.

Run directly (`python inference_optimization.py`) to reproduce all five demos.
"""

import numpy as np


# --- 1. KV cache + attention ------------------------------------------------
class KVCache:
    """Preallocated per-layer key/value cache for autoregressive decoding.

    Without a cache, generating token N+1 recomputes the key/value
    projections for tokens 1..N all over again even though they never change
    once written -- an O(N) waste per step that caching turns into O(1)
    (append-only).

    Attributes:
        k_cache: Array of shape (num_layers, num_heads, max_seq_len,
            head_dim) holding cached keys, preallocated once up front.
        v_cache: Same shape as `k_cache`, holding cached values.
        seq_len: Number of positions currently filled in the cache (shared
            across all layers, since every layer advances in lockstep).
    """

    def __init__(self, num_layers, num_heads, head_dim, max_seq_len, dtype=np.float16):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        self.k_cache = np.zeros(
            (num_layers, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        self.v_cache = np.zeros(
            (num_layers, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        self.seq_len = 0

    def update(self, layer_idx, new_keys, new_values):
        """Write `new_keys`/`new_values` for `layer_idx` at the current position.

        Args:
            layer_idx: Which transformer layer's cache slot to write into.
            new_keys: Array of shape (num_heads, num_new_tokens, head_dim).
            new_values: Same shape as `new_keys`.

        Returns:
            Tuple `(keys, values)` for every token seen so far at this layer,
            each of shape (num_heads, seq_len + num_new_tokens, head_dim) --
            i.e. the previously cached tokens plus the ones just written.
        """
        num_new = new_keys.shape[1]
        end = self.seq_len + num_new
        self.k_cache[layer_idx, :, self.seq_len:end, :] = new_keys
        self.v_cache[layer_idx, :, self.seq_len:end, :] = new_values
        return (
            self.k_cache[layer_idx, :, :end, :],
            self.v_cache[layer_idx, :, :end, :]
        )

    def advance(self, num_tokens):
        """Mark `num_tokens` newly written positions as part of the cache.

        Must be called once per forward pass (after `update` has run for
        every layer), with `num_tokens` equal to how many new tokens that
        pass processed: the full prompt length on the prefill pass, 1 on
        each decode step. Calling this with the wrong count desyncs
        `seq_len` from what `k_cache`/`v_cache` actually hold, and the next
        `update` will silently overwrite already-cached tokens instead of
        appending after them.
        """
        self.seq_len += num_tokens

    def memory_bytes(self):
        """Total bytes allocated for the cache, filled or not (`k_cache` + `v_cache`)."""
        return self.k_cache.nbytes + self.v_cache.nbytes

    def used_bytes(self):
        """Bytes actually holding data, for the `seq_len` tokens written so far."""
        per_token = 2 * self.num_layers * self.num_heads * self.head_dim * np.dtype(self.dtype).itemsize
        return per_token * self.seq_len


def scaled_dot_product_attention(query, keys, values):
    """Causal scaled dot-product attention: softmax(QK^T / sqrt(d)) @ V.

    Args:
        query: Array of shape (batch, num_heads, seq_len_q, head_dim).
        keys: Array of shape (batch, num_heads, seq_len_k, head_dim).
        values: Same shape as `keys`.

    A causal mask is applied only when `seq_len_q > 1` -- a prefill pass over
    a fresh prompt, where `seq_len_k == seq_len_q` and position i may attend
    only to positions <= i. A single-token decode step (`seq_len_q == 1`)
    needs no mask: the new token is already the most recent position, so it
    is always allowed to see everything already in `keys`/`values`.

    Returns:
        Attention output of shape (batch, num_heads, seq_len_q, head_dim).
    """
    head_dim = query.shape[-1]
    scores = np.matmul(query, keys.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    seq_len_q = scores.shape[-2]
    seq_len_k = scores.shape[-1]
    if seq_len_q > 1:
        mask = np.triu(np.ones((seq_len_q, seq_len_k), dtype=np.float32), k=seq_len_k - seq_len_q + 1)
        scores = scores + mask * (-1e9)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return np.matmul(attn_weights, values)


class MultiHeadAttention:
    """Multi-head self-attention with optional KV caching for autoregressive decoding.

    Weights are randomly initialized -- this class demonstrates the
    inference-time mechanics (projection -> cache -> attention -> output),
    not a trained model -- but the pipeline mirrors a real transformer
    attention block.
    """

    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * scale

    def forward(self, x, kv_cache=None, layer_idx=0):
        """Run one attention forward pass, optionally reading/writing `kv_cache`.

        Args:
            x: Input of shape (batch, seq_len, d_model). `seq_len` is the
                full prompt length on a prefill pass, or 1 on each decode step.
            kv_cache: Optional `KVCache` to read prior keys/values from and
                write this pass's keys/values into. Assumes batch size 1 when
                a cache is supplied, since `KVCache.update` caches a single
                sequence's keys/values per layer.
            layer_idx: Which cache slot (transformer layer) to read/write.

        Returns:
            Attention output of shape (batch, seq_len, d_model).
        """
        batch, seq_len, d_model = x.shape
        Q = np.matmul(x, self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.matmul(x, self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.matmul(x, self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            K_full, V_full = kv_cache.update(layer_idx, K[0], V[0])
            K = K_full[np.newaxis, :, :, :]
            V = V_full[np.newaxis, :, :, :]
            kv_cache.advance(seq_len)

        attn_out = scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, -1, d_model)
        return np.matmul(attn_out, self.W_o)


# --- 2. Batching -------------------------------------------------------------
class Request:
    """A single generation request moving through a batching simulation.

    Attributes:
        request_id: Identifier for the request.
        prompt_tokens: Prompt length (not used by the timing math below, kept
            for realism/reporting).
        output_tokens: Number of tokens this request needs generated before
            it is done.
        arrival_step: Simulated time step at which the request arrives.
        tokens_generated: Running count of tokens generated so far.
        start_step: Time step the request entered a batch (set by the simulator).
        end_step: Time step the request finished (set by the simulator).
    """

    def __init__(self, request_id, prompt_tokens, output_tokens, arrival_step):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.arrival_step = arrival_step
        self.tokens_generated = 0
        self.start_step = None
        self.end_step = None

    def is_done(self):
        """Whether this request has generated all of its `output_tokens`."""
        return self.tokens_generated >= self.output_tokens


def simulate_static_batching(requests, batch_size):
    """Simulate naive static batching: fixed-size batches that all wait for the
    slowest member before the next batch starts.

    Requests are grouped strictly in arrival order into batches of
    `batch_size`; every request in a batch is treated as finishing only once
    the batch's longest request (`max(output_tokens)`) has finished, so short
    requests sit idle behind long ones -- the throughput problem continuous
    batching (below) fixes.

    Args:
        requests: Iterable of `Request` objects (arrival order does not
            matter; they are sorted by `arrival_step`).
        batch_size: Maximum number of requests processed together.

    Returns:
        List of completed `Request` objects with `start_step`/`end_step` set.
    """
    step = 0
    completed = []
    queue = list(requests)
    queue.sort(key=lambda r: r.arrival_step)

    while queue:
        batch = []
        while queue and len(batch) < batch_size:
            r = queue.pop(0)
            r.start_step = max(step, r.arrival_step)
            batch.append(r)

        if batch:
            step = max(step, max(r.start_step for r in batch))
            max_output = max(r.output_tokens for r in batch)
            for r in batch:
                r.tokens_generated = r.output_tokens
                r.end_step = step + max_output
            step += max_output
            completed.extend(batch)

    return completed


def simulate_continuous_batching(requests, batch_size):
    """Simulate continuous (in-flight) batching: a request slots into the
    active batch as soon as a slot frees up, instead of waiting for the
    whole batch to finish.

    Each time step, every active request generates exactly one token; any
    request that finishes vacates its slot immediately, which the next
    waiting request fills on the following step -- so a short request never
    has to wait behind a long one already in the batch.

    Args:
        requests: Iterable of `Request` objects.
        batch_size: Maximum number of requests decoded concurrently.

    Returns:
        List of completed `Request` objects with `start_step`/`end_step` set.
    """
    step = 0
    completed = []
    queue = sorted(requests, key=lambda r: r.arrival_step)
    queue_idx = 0
    active = []
    waiting = []

    while queue_idx < len(queue) or active or waiting:
        while queue_idx < len(queue) and queue[queue_idx].arrival_step <= step:
            waiting.append(queue[queue_idx])
            queue_idx += 1

        while waiting and len(active) < batch_size:
            r = waiting.pop(0)
            r.start_step = step
            active.append(r)

        if not active:
            if waiting:
                step += 1
                continue
            elif queue_idx < len(queue):
                step = queue[queue_idx].arrival_step
                continue
            else:
                break

        for r in active:
            r.tokens_generated += 1

        done = [r for r in active if r.is_done()]
        for r in done:
            r.end_step = step + 1
            completed.append(r)
        active = [r for r in active if not r.is_done()]

        step += 1

    return completed


def batching_stats(completed):
    """Summarize latency and throughput across a batching simulation's output.

    Args:
        completed: List of finished `Request` objects (as returned by
            `simulate_static_batching` / `simulate_continuous_batching`).

    Returns:
        Dict with `avg_latency`, `p50_latency`, `p99_latency` (all in
        simulated time steps, measured from `arrival_step` to `end_step`),
        `total_time`, and `throughput` (output tokens per time step).
    """
    latencies = [r.end_step - r.arrival_step for r in completed]
    total_time = max(r.end_step for r in completed) - min(r.arrival_step for r in completed)
    total_tokens = sum(r.output_tokens for r in completed)
    return {
        "avg_latency": np.mean(latencies),
        "p50_latency": np.median(latencies),
        "p99_latency": np.percentile(latencies, 99),
        "total_time": total_time,
        "throughput": total_tokens / total_time if total_time > 0 else 0,
    }


# --- 3. Prefix caching --------------------------------------------------------
class TrieNode:
    """One node of the `PrefixCache` trie: one token id per edge.

    Attributes:
        children: Mapping of token_id -> child `TrieNode`.
        kv_data: Cached KV state for the token that reaches this node via its
            parent's edge, or `None` if this node's token has not been cached.
        hit_count: Number of lookups that passed through this node.
    """

    def __init__(self):
        self.children = {}
        self.kv_data = None
        self.hit_count = 0


class PrefixCache:
    """Trie-based cache of KV state keyed by token-id prefix.

    Requests that share a prompt prefix (a system prompt, few-shot examples,
    a retrieved document) can reuse the KV state already computed for that
    prefix instead of recomputing it -- the idea behind RadixAttention/SGLang
    prefix caching. Longer shared prefixes mean more reused computation.

    Attributes:
        max_entries: Cap on total trie nodes, so the cache cannot grow
            unbounded; `insert` stops early once the cap is reached.
        total_entries: Current number of trie nodes across all inserted sequences.
        hits: Number of `lookup` calls that matched at least one cached token.
        misses: Number of `lookup` calls that matched nothing.
    """

    def __init__(self, max_entries=1000):
        self.root = TrieNode()
        self.max_entries = max_entries
        self.total_entries = 0
        self.hits = 0
        self.misses = 0

    def _walk(self, token_ids):
        """Follow `token_ids` down the trie as far as cached children allow.

        Returns:
            Tuple `(node, depth)`: the deepest matching node and how many
            leading tokens of `token_ids` matched an existing path.
        """
        node = self.root
        depth = 0
        for tid in token_ids:
            if tid not in node.children:
                break
            node = node.children[tid]
            depth += 1
        return node, depth

    def lookup(self, token_ids):
        """Find the longest cached prefix of `token_ids` and its KV entries.

        Args:
            token_ids: Token ids for an incoming request's prompt.

        Returns:
            Tuple `(depth, kv_entries)`: `depth` is how many leading tokens
            matched a cached prefix (0 on a full miss), and `kv_entries` is
            the cached KV data for those matched tokens, in order.
        """
        node, depth = self._walk(token_ids)
        if depth > 0:
            self.hits += 1
            current = self.root
            for tid in token_ids[:depth]:
                current = current.children[tid]
                current.hit_count += 1
            kv_entries = []
            current = self.root
            for tid in token_ids[:depth]:
                current = current.children[tid]
                if current.kv_data is not None:
                    kv_entries.append(current.kv_data)
            return depth, kv_entries
        self.misses += 1
        return 0, []

    def insert(self, token_ids, kv_per_token):
        """Cache `token_ids` and their per-token KV data.

        Tokens already present in the trie are reused rather than duplicated,
        so inserting a sequence that shares a prefix with an existing one
        only adds nodes for the new suffix.

        Args:
            token_ids: Token ids to insert.
            kv_per_token: KV data for each position in `token_ids` (same length).

        Returns:
            Number of tokens actually cached: `len(token_ids)` normally, or
            the index at which `max_entries` was reached if the cache filled
            up partway through.
        """
        node = self.root
        for i, tid in enumerate(token_ids):
            if tid not in node.children:
                if self.total_entries >= self.max_entries:
                    return i
                node.children[tid] = TrieNode()
                self.total_entries += 1
            node = node.children[tid]
            if i < len(kv_per_token):
                node.kv_data = kv_per_token[i]
        return len(token_ids)

    def hit_rate(self):
        """Fraction of `lookup` calls that matched a cached prefix (0.0 if none yet)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# --- 4. Speculative decoding ---------------------------------------------------
class DraftModel:
    """Stand-in for a small, cheap "draft" model used to propose candidate tokens.

    Real draft/target token distributions are not simulated; instead
    `acceptance_rate` directly parametrizes how often a proposed token is
    accepted, letting `speculative_decode` model different drafting
    strategies (see `compare_speculation_strategies`) without needing an
    actual small model.
    """

    def __init__(self, vocab_size, acceptance_rate=0.8):
        self.vocab_size = vocab_size
        self.acceptance_rate = acceptance_rate

    def generate(self, context, num_tokens):
        """Propose `num_tokens` candidate token ids (uniform random stand-in)."""
        tokens = np.random.randint(0, self.vocab_size, size=num_tokens)
        return tokens

    def get_probs(self, context, token):
        """Return a random probability distribution over the vocabulary,
        standing in for the draft model's next-token distribution."""
        probs = np.random.dirichlet(np.ones(self.vocab_size))
        return probs


class TargetModel:
    """Stand-in for the large "target" model that verifies the draft's proposals."""

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def get_probs(self, context, tokens=None):
        """Return the target distribution(s) (random stand-ins) used to verify tokens.

        Args:
            context: Tokens generated so far (unused by this stand-in).
            tokens: If given, one distribution is returned per token in
                `tokens` (batched verification of several draft tokens at
                once); otherwise a single distribution is returned.
        """
        if tokens is not None:
            return [np.random.dirichlet(np.ones(self.vocab_size)) for _ in tokens]
        return np.random.dirichlet(np.ones(self.vocab_size))


def speculative_decode(draft_model, target_model, context, num_speculative=5,
                       draft_cost=1.0, target_cost=10.0, verify_cost=12.0):
    """Simulate speculative decoding: draft several tokens cheaply, verify them
    in one target-model pass, keep the accepted prefix.

    Each round, `draft_model` proposes `num_speculative` tokens; `target_model`
    verifies all of them in a single batched call. Tokens are accepted one at
    a time until the first rejection (a fresh token is sampled from the
    target distribution in its place) or all are accepted, in which case a
    bonus token is drawn "for free" from the target model.

    Note:
        The accept/reject decision itself is a Bernoulli draw against
        `draft_model.acceptance_rate` -- the empirical acceptance rate that
        differs by drafting strategy (see `compare_speculation_strategies`).
        `acceptance_prob` is computed from `target_p`/`draft_p` to show the
        textbook rejection-sampling formula real speculative decoding uses,
        but it is not itself sampled against here.

    Args:
        draft_model: `DraftModel` proposing candidate tokens.
        target_model: `TargetModel` verifying them.
        context: Initial token ids to condition on.
        num_speculative: Draft tokens proposed per round.
        draft_cost: Simulated cost of drafting one token.
        target_cost: Simulated cost of generating one token with the target
            model alone (used only for the `sequential_cost` baseline).
        verify_cost: Simulated cost of one batched target-model verification
            pass, regardless of how many draft tokens it verifies.

    Returns:
        Dict with `total_tokens` generated, `speculative_cost` actually
        spent, `sequential_cost` a plain (no speculation) baseline would have
        spent generating the same number of tokens, `speedup`
        (`sequential_cost / speculative_cost`), `avg_accepted` tokens per
        round, and `acceptance_rate` (`avg_accepted / num_speculative`).
    """
    total_tokens = 0
    total_cost = 0.0
    accepted_counts = []
    context = list(context)

    max_tokens = 100

    while total_tokens < max_tokens:
        draft_tokens = draft_model.generate(context, num_speculative)
        total_cost += draft_cost * num_speculative

        target_probs = target_model.get_probs(context, draft_tokens)
        total_cost += verify_cost

        accepted = 0
        for i, token in enumerate(draft_tokens):
            draft_p = draft_model.get_probs(context + list(draft_tokens[:i]), token)
            target_p = target_probs[i]

            r = np.random.random()
            acceptance_prob = min(1.0, target_p[token] / (draft_p[token] + 1e-10))

            if r < draft_model.acceptance_rate:
                accepted += 1
                context.append(token)
                total_tokens += 1
            else:
                new_token = np.random.choice(draft_model.vocab_size, p=target_p)
                context.append(new_token)
                total_tokens += 1
                break

        accepted_counts.append(accepted)

        if accepted == num_speculative:
            bonus_probs = target_model.get_probs(context)
            bonus_token = np.random.choice(draft_model.vocab_size, p=bonus_probs)
            context.append(bonus_token)
            total_tokens += 1

    sequential_cost = total_tokens * target_cost
    return {
        "total_tokens": total_tokens,
        "speculative_cost": total_cost,
        "sequential_cost": sequential_cost,
        "speedup": sequential_cost / total_cost if total_cost > 0 else 1.0,
        "avg_accepted": np.mean(accepted_counts),
        "acceptance_rate": np.mean(accepted_counts) / num_speculative,
    }


def compare_speculation_strategies(vocab_size=1000, num_trials=20):
    """Average `speculative_decode` results over several trials for a handful
    of real drafting strategies (identified by their published acceptance
    rates) plus a no-speculation baseline.

    Args:
        vocab_size: Vocabulary size shared by draft and target models.
        num_trials: Independent trials averaged per strategy (results are
            stochastic, so a single trial is noisy).

    Returns:
        Dict mapping strategy name -> `{speedup, acceptance_rate, avg_accepted}`,
        each averaged across `num_trials` runs of `speculative_decode`.
    """
    results = {}

    for name, acceptance_rate, spec_tokens in [
        ("Draft-target (8B->70B)", 0.78, 5),
        ("EAGLE", 0.85, 6),
        ("N-gram", 0.50, 4),
        ("No speculation", 0.0, 0),
    ]:
        if spec_tokens == 0:
            results[name] = {
                "speedup": 1.0,
                "acceptance_rate": 0.0,
                "avg_accepted": 0.0,
            }
            continue

        trial_results = []
        for _ in range(num_trials):
            draft = DraftModel(vocab_size, acceptance_rate=acceptance_rate)
            target = TargetModel(vocab_size)
            context = list(np.random.randint(0, vocab_size, size=10))
            result = speculative_decode(draft, target, context, num_speculative=spec_tokens)
            trial_results.append(result)

        results[name] = {
            "speedup": np.mean([r["speedup"] for r in trial_results]),
            "acceptance_rate": np.mean([r["acceptance_rate"] for r in trial_results]),
            "avg_accepted": np.mean([r["avg_accepted"] for r in trial_results]),
        }

    return results


# --- 5. Capacity planning ------------------------------------------------------
# Rough per-model shapes used for KV-cache sizing. `num_kv_heads` is the
# grouped-query-attention (GQA) head count when `gqa` is True -- fewer KV
# heads than query heads, which is what keeps KV-cache size down on modern
# models; GPT-4-est's estimated shape has no GQA (`num_kv_heads` == query heads).
MODEL_CONFIGS = {
    "Llama-3-8B": {
        "num_layers": 32, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 8, "gqa": True,
    },
    "Llama-3-70B": {
        "num_layers": 80, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 70, "gqa": True,
    },
    "Llama-3-405B": {
        "num_layers": 126, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 405, "gqa": True,
    },
    "Mistral-7B": {
        "num_layers": 32, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 7, "gqa": True,
    },
    "GPT-4-est": {
        "num_layers": 120, "num_kv_heads": 96, "head_dim": 128,
        "model_params_b": 1800, "gqa": False,
    },
}


def kv_cache_memory(config, seq_len, dtype_bytes=2):
    """Compute KV-cache size for one request of length `seq_len`.

    Args:
        config: One of `MODEL_CONFIGS`'s value dicts (`num_layers`,
            `num_kv_heads`, `head_dim`).
        seq_len: Number of cached tokens (prompt + generated so far).
        dtype_bytes: Bytes per cached value (2 for fp16/bf16, 1 for fp8).

    Returns:
        Dict with `per_token_bytes`/`per_token_kb` and
        `total_bytes`/`total_mb`/`total_gb` for `seq_len` tokens. Both keys
        and values are cached, hence the factor of 2 in `per_token_bytes`.
    """
    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * dtype_bytes
    total = per_token * seq_len
    return {
        "per_token_bytes": per_token,
        "per_token_kb": per_token / 1024,
        "total_bytes": total,
        "total_mb": total / (1024 ** 2),
        "total_gb": total / (1024 ** 3),
    }


def memory_budget(config, gpu_memory_gb, model_dtype_bytes=2, kv_dtype_bytes=2):
    """Estimate how many tokens of KV cache fit on a GPU after model weights.

    Reserves space for the model's weights and a flat 10% overhead (CUDA
    context, activations, fragmentation), then converts whatever remains
    into a token budget using `kv_cache_memory`'s per-token cost.

    Args:
        config: One of `MODEL_CONFIGS`'s value dicts.
        gpu_memory_gb: Total GPU memory in GB.
        model_dtype_bytes: Bytes per model weight (2 for fp16/bf16).
        kv_dtype_bytes: Bytes per cached KV value.

    Returns:
        On failure: `{"error": ..., "model_memory_gb": ...}` if the model's
        weights alone do not fit in `gpu_memory_gb` (this ignores
        multi-GPU/tensor-parallel deployment, so a "does not fit" result
        just means it does not fit on *one* GPU of this size).
        On success: dict with `gpu_memory_gb`, `model_memory_gb`,
        `overhead_gb`, `available_for_kv_gb`, `max_total_tokens`, and
        `max_users_at_2k`/`max_users_at_4k`/`max_users_at_32k` -- how many
        concurrent requests of that context length the remaining memory
        supports.
    """
    model_memory_gb = config["model_params_b"] * 1e9 * model_dtype_bytes / (1024 ** 3)
    overhead_gb = gpu_memory_gb * 0.1
    available_for_kv = gpu_memory_gb - model_memory_gb - overhead_gb

    if available_for_kv <= 0:
        return {"error": "Model does not fit in GPU memory", "model_memory_gb": model_memory_gb}

    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * kv_dtype_bytes
    max_tokens = int(available_for_kv * (1024 ** 3) / per_token)

    return {
        "gpu_memory_gb": gpu_memory_gb,
        "model_memory_gb": round(model_memory_gb, 1),
        "overhead_gb": round(overhead_gb, 1),
        "available_for_kv_gb": round(available_for_kv, 1),
        "max_total_tokens": max_tokens,
        "max_users_at_2k": max_tokens // 2048,
        "max_users_at_4k": max_tokens // 4096,
        "max_users_at_32k": max_tokens // 32768,
    }


# --- Demos ---------------------------------------------------------------------
def demo_kv_cache():
    """Run one prefill pass plus a few decode steps through a tiny attention
    layer, printing how the KV cache grows and how much memory it uses."""
    d_model, num_heads, max_seq_len = 64, 4, 32
    layer = MultiHeadAttention(d_model, num_heads)
    cache = KVCache(num_layers=1, num_heads=num_heads, head_dim=d_model // num_heads, max_seq_len=max_seq_len)

    prompt = np.random.randn(1, 6, d_model).astype(np.float32)
    layer.forward(prompt, kv_cache=cache, layer_idx=0)
    print(f"After prefill (6 tokens): cache.seq_len = {cache.seq_len}")

    for _ in range(3):
        next_token = np.random.randn(1, 1, d_model).astype(np.float32)
        layer.forward(next_token, kv_cache=cache, layer_idx=0)
    print(f"After 3 decode steps:     cache.seq_len = {cache.seq_len}")
    print(f"Cache memory: {cache.memory_bytes() / 1024:.1f} KB allocated, "
          f"{cache.used_bytes() / 1024:.1f} KB in use")


def demo_batching():
    """Run the same synthetic request mix through static and continuous
    batching, printing latency/throughput side by side."""
    def make_requests():
        return [
            Request(i, prompt_tokens=50, output_tokens=int(np.random.randint(10, 100)), arrival_step=i * 3)
            for i in range(20)
        ]

    np.random.seed(0)
    static_stats = batching_stats(simulate_static_batching(make_requests(), batch_size=4))
    np.random.seed(0)
    continuous_stats = batching_stats(simulate_continuous_batching(make_requests(), batch_size=4))

    print(f"{'metric':<15}{'static':>12}{'continuous':>12}")
    for key in static_stats:
        print(f"{key:<15}{static_stats[key]:>12.2f}{continuous_stats[key]:>12.2f}")


def demo_prefix_cache():
    """Show prefix-cache hits when several requests share a long system-prompt prefix."""
    cache = PrefixCache()
    system_prompt = list(range(100))  # stand-in for a long shared system prompt

    for i in range(5):
        user_suffix = list(np.random.randint(1000, 2000, size=10))
        tokens = system_prompt + user_suffix
        depth, _ = cache.lookup(tokens)
        cache.insert(tokens, kv_per_token=[f"kv_{t}" for t in tokens])
        print(f"Request {i}: matched {depth} cached tokens before falling back to computing the rest")

    print(f"Prefix cache hit rate: {cache.hit_rate():.0%}")


def demo_speculative_decoding():
    """Print speculative-decoding speedup for a few real drafting strategies."""
    results = compare_speculation_strategies()
    print(f"{'strategy':<28}{'speedup':>9}{'acceptance':>13}{'avg_accepted':>15}")
    for name, stats in results.items():
        print(f"{name:<28}{stats['speedup']:>8.2f}x{stats['acceptance_rate']:>12.0%}{stats['avg_accepted']:>15.2f}")


def demo_memory_budget():
    """Print KV-cache footprint and concurrent-request capacity for a few
    models sharing a single 80GB GPU."""
    for name in ["Llama-3-8B", "Llama-3-70B", "GPT-4-est"]:
        config = MODEL_CONFIGS[name]
        mem = kv_cache_memory(config, seq_len=4096)
        budget = memory_budget(config, gpu_memory_gb=80)
        print(f"{name}: {mem['total_mb']:.0f} MB KV cache per request @ 4K context")
        if "error" in budget:
            print(f"  {budget['error']} (needs {budget['model_memory_gb']:.1f} GB just for weights)")
        else:
            print(f"  Fits ~{budget['max_users_at_4k']} concurrent 4K-context requests on an 80GB GPU")


def main():
    """Run all five inference-optimization demos end to end."""
    print("=== 1. KV cache ===")
    demo_kv_cache()

    print("\n=== 2. Static vs. continuous batching ===")
    demo_batching()

    print("\n=== 3. Prefix caching ===")
    demo_prefix_cache()

    print("\n=== 4. Speculative decoding ===")
    demo_speculative_decoding()

    print("\n=== 5. KV-cache memory budgeting ===")
    demo_memory_budget()


if __name__ == "__main__":
    main()
