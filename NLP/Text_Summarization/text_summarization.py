"""Text summarization: extractive TextRank, abstractive BART, and ROUGE evaluation.

Covers three complementary pieces of a summarization pipeline:

1. TextRank extractive summarization (`sentence_split`, `similarity`,
   `textrank`): treat a document as a graph of sentences, weight edges by
   how much vocabulary two sentences share, and rank sentences with
   PageRank's power-iteration algorithm. The top-`k` sentences are pulled
   out verbatim, in original order, rather than generated - the "tells you
   what the document said" style of summary.
2. Abstractive summarization via a fine-tuned encoder-decoder transformer
   (`facebook/bart-large-cnn`, through `transformers.pipeline`): generates
   a new summary token by token instead of copying sentences - the "tells
   you what the author meant" style, at the cost of a small risk of
   hallucinated details not present in the source.
3. ROUGE evaluation (`rouge_score`): scores a generated summary against a
   reference summary by n-gram / longest-common-subsequence overlap
   (ROUGE-1, ROUGE-2, ROUGE-L) - the standard automatic metric for
   comparing either style of summary to a human-written reference.

The ROUGE block at the bottom is illustrative: `reference_summary` and
`generated_summary` are placeholders (e.g. a human-written reference and
`summary[0]["summary_text"]` from the BART call above), not variables
defined in this file.
"""

import math
import re
from collections import Counter
from transformers import pipeline
from rouge_score import rouge_scorer


# --- 1. TextRank extractive summarization -----------------------------------
def sentence_split(text):
    """Split text into sentences on sentence-ending punctuation.

    Splits after '.', '!', or '?' followed by whitespace, via a regex
    lookbehind that keeps the punctuation attached to the preceding
    sentence. A plain heuristic - it doesn't handle abbreviations (e.g.
    "Dr.", "U.S.") or decimal numbers the way a trained sentence
    tokenizer (e.g. nltk's punkt) would.

    Args:
        text: Raw text to split.

    Returns:
        List of sentence strings.
    """
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    """Word-overlap similarity between two sentences, TextRank-style.

    Counts shared words (case-insensitive, whitespace-tokenized, with
    repeats counted via `Counter` intersection) and normalizes by the sum
    of the logs of each sentence's word count, following the sentence
    similarity measure from the original TextRank paper (Mihalcea &
    Tarau, 2004): normalizing by log-length keeps long sentences from
    scoring higher than short ones simply by containing more words.

    Args:
        s1: First sentence.
        s2: Second sentence.

    Returns:
        A non-negative similarity score. 0.0 if both sentences are empty
        (the log-length denominator would otherwise be zero).
    """
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    """Rank sentences with TextRank and return the top-`k` in their original order.

    Builds a complete weighted graph where each node is a sentence and
    each edge weight is `similarity(sentences[i], sentences[j])`, then
    runs the power-iteration form of PageRank over that graph: each
    sentence's score is redistributed to its neighbors in proportion to
    edge weight, damped by `damping` and floored by `(1 - damping)`,
    stopping once every score changes by less than `epsilon` between
    steps (or after `iterations` steps, whichever comes first). The
    `top_k` highest-scoring sentences are then resorted back into their
    original document order, so the extract reads as a coherent excerpt
    rather than a list sorted by score.

    Args:
        text: The article/document to summarize.
        top_k: Number of sentences to extract. If the document already
            has `top_k` or fewer sentences, all of them are returned
            as-is and no graph is built.
        damping: PageRank damping factor - the probability of following
            a graph edge vs. jumping to a random sentence (0.85 is the
            standard PageRank/TextRank default).
        iterations: Maximum number of power-iteration steps.
        epsilon: Convergence threshold - iteration stops early once every
            sentence's score changes by less than this between steps.

    Returns:
        List of up to `top_k` sentence strings, in their original order
        of appearance.
    """
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]


# --- 2. Abstractive summarization (BART) ------------------------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""  # replace with the real source text

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])


# --- 3. ROUGE evaluation ------------------------------------------------------
# Illustrative only (see module docstring): reference_summary (a human-written
# summary) and generated_summary (e.g. summary[0]["summary_text"] above) are
# placeholders, not defined in this file.
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})