"""Prompt caching (context caching) for LLM API calls.

Reusing a stable prefix (system prompt, tool defs, few-shot examples, or
retrieved documents) across requests lets a provider skip recomputing the
attention KV-cache for that prefix, cutting both cost and time-to-first-token
for the cached portion. All three major providers expose this, with
different trade-offs:

- Anthropic (Claude): explicit `cache_control` breakpoints. Cache reads cost
  ~0.1x the base input price, cache writes cost ~1.25x (5-minute TTL) or ~2x
  (1-hour TTL). Caching is a strict *prefix* match -- render order is
  tools -> system -> messages, and a single byte changed anywhere in that
  prefix (a timestamp, a reordered tool, a rephrased instruction) invalidates
  every breakpoint at or after that position.
- OpenAI: fully automatic for any prompt prefix >= 1,024 tokens that matches
  a recent request -- no code changes needed, ~50% discount on cache reads.
- Gemini: explicit, named `CachedContent` objects with their own TTL and
  storage billing, aimed at reuse across many requests over hours/days
  rather than a single conversation.

The functions below are illustrative call patterns, not a runnable
end-to-end demo -- callers are expected to pass in real prompt content
(a large rubric, the code under review, few-shot examples, ...); calling
any of them still requires a real API key for that provider.
"""

import anthropic


def anthropic_explicit_cache_demo(rubric: str, code_a: str, code_b: str):
    """Cache a large, stable system prompt across two Claude review calls.

    `cache_control` sits on the system block, so the (potentially huge)
    `rubric` is written to the cache once and read on every later call,
    regardless of what code is being reviewed. The first call is a cache
    *write* (`usage.cache_creation_input_tokens`, paid at ~1.25x); every
    later call whose prefix still matches is a cache *read*
    (`usage.cache_read_input_tokens`, paid at ~0.1x).
    """
    client = anthropic.Anthropic()
    system = [
        {
            "type": "text",
            "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + rubric,
            "cache_control": {"type": "ephemeral"},  # default TTL: 5 minutes
        }
    ]

    def review(code: str):
        return client.messages.create(
            model="claude-opus-5",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": code}],
        )

    response_a = review(code_a)
    # response_a.usage -> cache_creation_input_tokens=15023 (paid at ~1.25x), cache_read_input_tokens=0

    response_b = review(code_b)
    # response_b.usage -> cache_creation_input_tokens=0, cache_read_input_tokens=15023 (paid at ~0.1x)

    return response_a, response_b


def anthropic_extended_ttl_block(rubric: str) -> dict:
    """Build a system content block using the 1-hour TTL instead of the 5-minute default.

    The 1-hour TTL costs a larger write premium (~2x vs ~1.25x) but keeps the
    cache warm across gaps of 5-60 minutes between requests that share the
    same prefix -- worth it once the prefix is read back at least ~3 times
    within the hour (2x + 0.2x reads < 3x uncached).
    """
    return {
        "type": "text",
        "text": rubric,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }


def openai_automatic_cache_demo(system_prompt: str, user_msg: str) -> int:
    """OpenAI caches automatically -- there is no `cache_control` field to set.

    Any request whose first >=1,024 tokens match a recently-seen prefix is
    served from cache automatically. The discounted token count comes back
    on the response as `usage.prompt_tokens_details.cached_tokens`.
    """
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt},  # long and stable
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.usage.prompt_tokens_details.cached_tokens


def gemini_explicit_cache_demo(rubric: str, few_shot_examples: str, code: str):
    """Gemini caches via an explicit, named `CachedContent` object.

    Unlike Anthropic/OpenAI's per-request breakpoints, the cache is created
    once up front (with its own TTL) and referenced by name on every later
    call -- suited to reusing a large corpus across many requests spread
    over hours or days rather than a single back-to-back conversation.
    """
    from google import genai
    from google.genai import types

    client = genai.Client()
    cache = client.caches.create(
        model="gemini-3-pro",
        config=types.CreateCachedContentConfig(
            display_name="rubric-v3",
            system_instruction=rubric,
            contents=[few_shot_examples],
            ttl="3600s",
        ),
    )

    return client.models.generate_content(
        model="gemini-3-pro",
        contents=["Review this code:\n" + code],
        config=types.GenerateContentConfig(cached_content=cache.name),
    )
