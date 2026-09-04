"""Long-context evaluation: NIAH, multi-needle, RULER-style tracing, LongBench v2.

Covers four complementary pieces of a long-context evaluation battery, for
answering "how much of the advertised context window is actually usable":

1. Needle-in-a-haystack (`build_haystack`, `score_niah`): plant a single fact
   ("needle") at a controlled depth inside a filler-text "haystack" of a given
   token length, then ask the model to retrieve it. Sweeping `depth_ratio` and
   `total_tokens` produces the classic NIAH depth x length heatmap - the
   baseline every frontier model should already pass.
2. Multi-needle retrieval (`build_multi_needle`): plant several needles at
   fixed depths (10%, 40%, 70%) instead of one. A question that requires all
   of them ("what are the three magic words?") tests whether the model can
   hold multiple facts in attention at once - single-needle success does not
   predict multi-needle success.
3. RULER-style multi-hop variable tracing (the module-level `haystack` /
   `question` below): a chain of variable assignments (X1 -> X2 -> X3) buried
   in filler text, where answering requires chaining every hop rather than
   retrieving one isolated fact. Frontier models that saturate NIAH at 128k
   often drop to 50-70% accuracy on this kind of task.
4. LongBench v2 evaluation (`eval_model_on_longbench`): runs a model against
   a real-world, human-annotated long-context QA benchmark
   (`THUDM/LongBench-v2` from Hugging Face) instead of a synthetic needle,
   scored by exact match after normalization.

`tokenize` (used by `build_haystack`) and `normalize` (used by
`eval_model_on_longbench`) are not defined in this file - callers must supply
their own (e.g. a real tokenizer's encode/decode, and a
lowercase-and-strip-punctuation normalizer, respectively). `model` in
`score_niah`/`eval_model_on_longbench` is any object exposing
`.complete(prompt, max_tokens) -> str`.

Importing this module eagerly downloads `THUDM/LongBench-v2` via
`load_dataset` at module scope - this is not lazy, and there is no
`if __name__ == "__main__":` guard around it.
"""

from datasets import load_dataset

def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    """Build a single-needle haystack: filler text with `needle` inserted at a depth.

    Repeats/tiles `filler_text`'s tokens until there are enough to fill the
    non-needle portion of the haystack (`total_tokens - len(needle_tokens)`),
    then splices the needle's tokens in at `depth_ratio` of the way through
    that filler body. `depth_ratio=0.0` puts the needle at the very start,
    `1.0` at the very end - sweeping this alongside `total_tokens` produces
    the classic NIAH depth x length grid.

    Args:
        filler_text: Background text to repeat until it fills the haystack.
            Must tokenize to at least one token.
        needle: The fact/sentence to hide inside the haystack.
        depth_ratio: Where to insert the needle, as a fraction of the filler
            body's length. Must be in [0.0, 1.0].
        total_tokens: Target haystack length in tokens, including the
            needle. Must be positive; if smaller than the needle's own
            token count, the haystack ends up being just the needle.

    Returns:
        The haystack as a single whitespace-joined string.

    Raises:
        ValueError: If `depth_ratio` is outside [0, 1], `total_tokens` is
            not positive, or `filler_text` tokenizes to an empty list.
    """
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    """Score a single needle-in-a-haystack retrieval attempt.

    Prompts `model` with the haystack and question, then checks whether the
    expected answer appears anywhere in the model's completion - a lenient,
    case-insensitive substring match rather than an exact-match check.

    Args:
        model: Object exposing `.complete(prompt, max_tokens) -> str`.
        haystack: Haystack text, e.g. produced by `build_haystack`.
        question: The question to ask about the hidden needle.
        expected: Expected answer substring to look for in the model's
            response.

    Returns:
        1 if `expected` (case-insensitive) appears in the model's answer,
        else 0.
    """
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0


def build_multi_needle(filler, needles, total_tokens):
    """Build a multi-needle haystack with needles planted at fixed depths.

    Interleaves slices of `filler` with each needle in `needles`, planting
    them at roughly 10%, 40%, and 70% of the way through `total_tokens`
    (via `zip`, so only the first 3 needles are used - any beyond that are
    silently ignored). Unlike `build_haystack`, this slices `filler`
    directly by position rather than tokenizing it first, so the caller is
    responsible for `filler`'s units matching `total_tokens`.

    A question that requires every planted needle (e.g. "what are the
    three magic words?") tests whether the model can hold multiple facts
    in attention at once, which single-needle NIAH
    (`build_haystack`/`score_niah`) does not exercise.

    Args:
        filler: Background text/sequence to slice for the surrounding
            context.
        needles: The facts to plant, in the order they should appear.
            Only the first 3 are placed, since there are 3 hard-coded
            depths.
        total_tokens: Nominal total length used to compute slice
            boundaries into `filler`.

    Returns:
        The multi-needle haystack as a single whitespace-joined string.
    """
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)


# RULER-style multi-hop variable tracing: answering `question` correctly
# requires chaining all three assignments (X1 -> X2 -> X3), not just
# retrieving one isolated fact the way single-needle NIAH does.
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"

longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    """Evaluate a model on one task subset of LongBench v2.

    Filters the module-level `longbench["test"]` split down to rows whose
    `task` matches `subset`, prompts `model` with each row's context and
    question, and scores by exact match between the normalized completion
    and the normalized gold answer.

    Args:
        model: Object exposing `.complete(prompt, max_tokens) -> str`.
        subset: Which LongBench v2 task category to evaluate, e.g.
            "single-doc-qa", "multi-doc-qa", "long-icl", "long-dialogue",
            "code-repo-understanding", or "long-structured-data".

    Returns:
        Accuracy over the matching rows, as a float in [0.0, 1.0].

    Raises:
        ZeroDivisionError: If no rows in the test split match `subset`.
    """
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)