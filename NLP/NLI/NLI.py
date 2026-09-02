"""Natural Language Inference (NLI) demo using facebook/bart-large-mnli.

Covers three production uses of NLI, all built on the same premise/hypothesis
entailment classifier:

1. Direct entailment/contradiction/neutral classification between two texts.
2. Zero-shot text classification (verbalize each candidate label as a
   hypothesis, pick the label with the highest entailment score).
3. RAG faithfulness checking (`is_faithful`): does the retrieved context
   entail the generated answer, or is the answer a hallucination?
"""

from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]


zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}


def is_faithful(answer, context, threshold=0.5):
    """Check whether `answer` is entailed by `context` (RAG faithfulness check).

    Runs NLI with the retrieved context as premise and the generated answer
    as hypothesis, then compares the entailment score against `threshold`.
    A low entailment score means the answer is not supported by the context,
    i.e. a likely hallucination.

    Args:
        answer: The generated text to verify (hypothesis).
        context: The source/retrieved text the answer should be grounded in (premise).
        threshold: Minimum entailment score required to consider the answer faithful.

    Returns:
        True if the entailment score exceeds `threshold`, False otherwise.
    """
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold