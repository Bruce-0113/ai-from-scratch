"""LLM-as-judge evaluation: NLI-based faithfulness, answer relevance, and G-Eval.

Covers four complementary pieces of an LLM/RAG evaluation pipeline, mirroring
the RAGAS + DeepEval stack used in production:

1. Faithfulness (`atomic_claims`, `faithfulness`): a RAGAS-style,
   reference-free metric. An LLM decomposes the answer into atomic factual
   claims, then each claim is checked against the retrieved context with an
   NLI entailment model (not the judge LLM itself) - faithfulness is the
   fraction of claims the context actually supports. Low faithfulness means
   the answer contains unsupported claims, i.e. hallucination.
2. Answer relevance (`answer_relevance`): also reference-free. An LLM
   generates `n` hypothetical questions the answer could be responding to;
   if those questions embed far from the real question, the answer is
   drifting off-topic even if every claim in it happens to be faithful.
3. G-Eval (`metric`, `test`): a DeepEval `GEval` metric that uses an LLM
   judge directly, guided by explicit chain-of-thought evaluation steps
   (a rubric) instead of an implicit "score 0-1" prompt, which makes the
   judge's output more stable. Needs an expected/reference output, unlike
   the two metrics above.
4. CI regression gate (`test_rag_system`): a pytest-style test that reruns
   DeepEval's `FaithfulnessMetric` / `ContextualRelevancyMetric` on a fixed
   regression set and fails the build if either metric drops below
   threshold - the pattern for blocking merges on quality regressions.

Both `nli` (an entailment classifier) and `llm`/`encoder` (an LLM callable
and an embedding model) are pluggable: `nli` need only accept a
`{"text": premise, "text_pair": hypothesis}` dict à la `transformers`
`text-classification` pipelines, `llm` need only be `str -> str`, and
`encoder` need only implement `.encode(texts, normalize_embeddings=True)`.

Every LLM-as-judge metric here (faithfulness's claim extraction, relevance's
question generation, and G-Eval) is only as trustworthy as its calibration
against human labels - see the module's companion README for why an
uncalibrated judge, or a judge from the same model family as the system
under test, silently inflates scores.

`load_regression_cases` in `test_rag_system` is illustrative, matching the
`docs/en.md` reference this file is based on - it is not defined here, and
the module-level `metric.measure(test)` / `print(...)` call below runs
eagerly at import time, before `test_rag_system` is ever invoked.
"""

from typing import Callable
from transformers import pipeline
import numpy as np
from sentence_transformers import SentenceTransformer
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    """Decompose an answer into a list of atomic factual claims via an LLM.

    Each returned line is meant to be a single, independently-checkable
    factual statement, so that `faithfulness` can verify claims one at a
    time against the retrieved context rather than judging the answer as
    one indivisible block of text.

    Args:
        answer: The generated answer text to decompose.
        llm: Callable that takes a prompt string and returns the raw
            generated text (one claim per line).

    Returns:
        List of claim strings, one per non-empty line the LLM returned.
        May contain blank lines if the LLM's formatting is inconsistent -
        callers should not assume every element is non-empty.
    """
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    """RAGAS-style faithfulness: fraction of the answer's claims entailed by context.

    Splits `answer` into atomic claims via `atomic_claims`, then runs NLI
    once per claim with `context` as premise and the claim as hypothesis
    (via the module-level `nli` pipeline). A claim counts as "supported"
    only if entailment scores above 0.5. This uses a dedicated NLI model as
    the checker rather than asking the judge LLM directly, which is cheaper
    and avoids the judge simply trusting its own earlier claim extraction.

    Args:
        answer: The generated answer to score.
        context: The retrieved/source text the answer should be grounded in.
        llm: Callable used to decompose `answer` into claims (see
            `atomic_claims`); not used for the entailment check itself.

    Returns:
        Fraction of claims supported by `context`, in [0.0, 1.0]. Returns
        0.0 if the answer yields no claims at all.
    """
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)


# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    """Reference-free relevance: does the answer address the question asked?

    Asks `llm` to generate `n` hypothetical questions the answer could be a
    response to, embeds those alongside the real `question` (with
    `encoder`, normalized so the dot product below is cosine similarity),
    and averages the similarity between the real question and each
    generated one. An answer that actually addresses `question` should make
    the LLM regenerate questions close to it in embedding space; an answer
    that drifts off-topic will generate questions that embed far away, even
    if every individual claim in the answer is itself faithful to context.

    Args:
        question: The original question the answer is supposed to address.
        answer: The generated answer being scored.
        encoder: Embedding model exposing
            `.encode(texts, normalize_embeddings=True) -> ndarray`.
        llm: Callable that takes a prompt string and returns the raw
            generated text (one candidate question per line).
        n: Number of hypothetical questions to generate and average over.

    Returns:
        Mean cosine similarity between `question` and the `n` generated
        questions, roughly in [-1.0, 1.0] (in practice close to [0, 1] for
        on-topic embeddings). Returns 0.0 if the LLM produced no usable
        question lines.
    """
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)


# G-Eval: an LLM-judge metric guided by an explicit rubric (`evaluation_steps`)
# rather than an implicit "score 0-1" prompt, which makes judge output more
# stable across runs. Requires an `expected_output`, unlike the two metrics
# above, so it fits offline evaluation against a labeled/reference dataset
# more than live, reference-free monitoring.
metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)


def test_rag_system():
    """CI regression gate: fail the build if RAG quality drops below threshold.

    Pytest-discoverable test that reruns DeepEval's `FaithfulnessMetric`
    (threshold 0.85) and `ContextualRelevancyMetric` (threshold 0.7) over a
    fixed regression set on every call, so it can be wired into CI to block
    merges that regress retrieval or generation quality. `load_regression_cases`
    is a placeholder (see module docstring) - it must return an iterable of
    DeepEval `LLMTestCase`-like objects with an `id` for the assertion
    messages to be meaningful.

    Raises:
        AssertionError: If any case's faithfulness score falls below 0.85
            or contextual relevancy score falls below 0.7.
    """
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"