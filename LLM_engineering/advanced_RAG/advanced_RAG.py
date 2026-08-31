"""Advanced retrieval-augmented generation (RAG) utilities.

Includes BM25 lexical search, hybrid search via reciprocal rank fusion,
a lightweight heuristic reranker, HyDE (Hypothetical Document Embeddings)
query expansion, parent-child chunking, and simple faithfulness/recall
evaluation metrics for a RAG pipeline.
"""

import math
from collections import Counter

class BM25:
    """Okapi BM25 lexical ranking index over a fixed corpus of documents."""

    def __init__(self, k1=1.2, b=0.75):
        """Initialize the index.

        Args:
            k1: Term-frequency saturation parameter; higher values let
                repeated term occurrences keep contributing to the score.
            b: Length-normalization strength, from 0 (no normalization)
                to 1 (full normalization by document length).
        """
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = {}
        self.n_docs = 0

    def index(self, documents):
        """Build the index over a corpus, computing document lengths and
        document frequencies for every term.

        Args:
            documents: List of raw document strings to index.
        """
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query, doc_idx):
        """Compute the BM25 relevance score of one document against a query.

        Args:
            query: Query text.
            doc_idx: Index of the document (in ``self.docs``) to score.

        Returns:
            The BM25 score as a float.
        """
        query_words = query.lower().split()
        doc_words = self.docs[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = Counter(doc_words)
        score = 0.0

        for term in query_words:
            if term not in word_counts:
                continue
            tf = word_counts[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def search(self, query, top_k=10):
        """Rank all indexed documents against a query and return the top matches.

        Args:
            query: Query text.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(doc_idx, score)`` tuples sorted by descending score.
        """
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def reciprocal_rank_fusion(ranked_lists, k=60):
    """Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    Each document's fused score is the sum of ``1 / (k + rank + 1)`` across
    every ranked list it appears in, rewarding documents that rank highly
    (or consistently) across multiple retrieval methods.

    Args:
        ranked_lists: Iterable of ranked lists, each a sequence of
            ``(doc_id, score)`` tuples ordered from most to least relevant.
        k: RRF smoothing constant; larger values reduce the influence of
            rank position.

    Returns:
        A list of ``(doc_id, fused_score)`` tuples sorted by descending
        fused score.
    """
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def hybrid_search(query, chunks, vector_embeddings, vocab, idf, bm25_index, top_k=5, fusion_k=60):
    """Combine dense vector search and BM25 lexical search via RRF.

    Retrieves a wider candidate pool from both a TF-IDF vector index and a
    BM25 index, then fuses the two ranked lists into a single result.

    Args:
        query: Query text.
        chunks: List of text chunks the corpus was built from.
        vector_embeddings: Precomputed embeddings for ``chunks``, used for
            the vector similarity search.
        vocab: Vocabulary mapping used to embed the query via TF-IDF.
        idf: Inverse document frequency weights used to embed the query.
        bm25_index: A fitted :class:`BM25` instance over ``chunks``.
        top_k: Number of fused results to return.
        fusion_k: RRF smoothing constant passed to
            :func:`reciprocal_rank_fusion`.

    Returns:
        A list of ``(doc_id, fused_score)`` tuples, the top ``top_k`` results.
    """
    query_emb = tfidf_embed(query, vocab, idf)
    vector_results = search(query_emb, vector_embeddings, top_k=top_k * 3)
    bm25_results = bm25_index.search(query, top_k=top_k * 3)
    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=fusion_k)
    return fused[:top_k]


def rerank(query, candidates, chunks):
    """Re-score retrieval candidates using lexical overlap heuristics.

    Combines content-word overlap, query-bigram matches within the chunk,
    an early-position boost, and the original retrieval score into a single
    heuristic relevance score.

    Args:
        query: Query text.
        candidates: List of ``(doc_id, initial_score)`` tuples from a prior
            retrieval step.
        chunks: List of text chunks, indexable by ``doc_id``.

    Returns:
        A list of ``(doc_id, rerank_score)`` tuples sorted by descending
        rerank score.
    """
    query_words = set(query.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that",
                  "with", "from", "be", "has", "have", "had", "not", "but"}
    query_terms = query_words - stop_words

    scored = []
    for doc_id, initial_score in candidates:
        chunk = chunks[doc_id].lower()
        chunk_words = set(chunk.split())

        term_overlap = len(query_terms & chunk_words)

        query_bigrams = set()
        q_list = [w for w in query.lower().split() if w not in stop_words]
        for i in range(len(q_list) - 1):
            query_bigrams.add(q_list[i] + " " + q_list[i + 1])
        bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)

        position_boost = 0
        for term in query_terms:
            pos = chunk.find(term)
            if pos != -1 and pos < len(chunk) // 3:
                position_boost += 0.5

        rerank_score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, rerank_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def hyde_generate_hypothesis(query):
    """Generate a hypothetical answer document for a query (HyDE).

    Fills a template based on the query's leading word (e.g. "what"/"how")
    to produce a synthetic passage that reads like a plausible answer. This
    hypothesis is embedded and used for retrieval instead of the raw query,
    since it lexically resembles the documents being searched.

    Args:
        query: Query text.

    Returns:
        A synthesized hypothetical-answer string.
    """
    templates = {
        "what": "The answer to '{query}' is as follows: Based on our documentation, {topic} involves specific policies and procedures that define how the process works.",
        "how": "To address '{query}': The process involves several steps. First, you need to initiate the request. Then, the system processes it according to the defined rules.",
        "default": "Regarding '{query}': Our records indicate specific details and policies related to this topic that provide a comprehensive answer."
    }
    query_lower = query.lower()
    if query_lower.startswith("what"):
        template = templates["what"]
    elif query_lower.startswith("how"):
        template = templates["how"]
    else:
        template = templates["default"]

    topic_words = [w for w in query.lower().split()
                   if w not in {"what", "is", "the", "how", "do", "does", "a", "an",
                                "for", "of", "to", "in", "on", "at", "by", "and", "or"}]
    topic = " ".join(topic_words) if topic_words else "this topic"

    return template.format(query=query, topic=topic)


def hyde_search(query, chunks, vector_embeddings, vocab, idf, top_k=5):
    """Retrieve chunks by embedding a HyDE hypothesis instead of the raw query.

    Args:
        query: Query text.
        chunks: List of text chunks the corpus was built from (unused
            directly here but kept for interface symmetry with other
            search functions).
        vector_embeddings: Precomputed embeddings for ``chunks``.
        vocab: Vocabulary mapping used to embed the hypothesis via TF-IDF.
        idf: Inverse document frequency weights used to embed the hypothesis.
        top_k: Number of results to return.

    Returns:
        A tuple of ``(results, hypothesis)`` where ``results`` is the list
        of ``(doc_id, score)`` tuples from the vector search and
        ``hypothesis`` is the generated hypothetical-answer text.
    """
    hypothesis = hyde_generate_hypothesis(query)
    hypothesis_emb = tfidf_embed(hypothesis, vocab, idf)
    results = search(hypothesis_emb, vector_embeddings, top_k)
    return results, hypothesis


def create_parent_child_chunks(text, parent_size=200, child_size=50):
    """Split text into small child chunks nested within larger parent chunks.

    Child chunks are used for precise retrieval matching, while their
    parent chunks provide broader surrounding context once a child is
    retrieved.

    Args:
        text: Full text to split.
        parent_size: Number of words per parent chunk.
        child_size: Number of words per child chunk.

    Returns:
        A tuple ``(parents, children, child_to_parent)`` where ``parents``
        and ``children`` are lists of text chunks, and ``child_to_parent``
        maps each child chunk's index to its parent chunk's index.
    """
    words = text.split()
    parents = []
    children = []
    child_to_parent = {}

    parent_idx = 0
    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parent_text = " ".join(words[start:parent_end])
        parents.append(parent_text)

        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = " ".join(words[child_start:child_end])
            child_idx = len(children)
            children.append(child_text)
            child_to_parent[child_idx] = parent_idx
            child_start += child_size

        parent_idx += 1
        start += parent_size

    return parents, children, child_to_parent


def evaluate_faithfulness(answer, retrieved_chunks):
    """Estimate how well a generated answer is grounded in retrieved context.

    Splits the answer into sentences and checks, for each one, whether
    enough of its content words also appear in the retrieved chunks. This
    is a simple lexical-overlap proxy for faithfulness/groundedness.

    Args:
        answer: Generated answer text to evaluate.
        retrieved_chunks: List of context chunks that were retrieved and
            given to the generator.

    Returns:
        A tuple ``(score, ungrounded)`` where ``score`` is the fraction of
        answer sentences considered grounded (0.0 to 1.0), and
        ``ungrounded`` is the list of sentences that were not.
    """
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    grounded = 0
    ungrounded = []
    context = " ".join(retrieved_chunks).lower()

    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                      "to", "of", "in", "for", "on", "at", "by", "it", "this", "that"}
        content_words = words - stop_words
        if not content_words:
            grounded += 1
            continue

        matched = sum(1 for w in content_words if w in context)
        ratio = matched / len(content_words) if content_words else 0

        if ratio >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    score = grounded / len(answer_sentences) if answer_sentences else 1.0
    return score, ungrounded


def evaluate_retrieval_recall(queries_with_relevant, retrieval_fn, k=5):
    """Compute average recall@k of a retrieval function over a labeled query set.

    Args:
        queries_with_relevant: Iterable of ``(query, relevant_indices)``
            pairs, where ``relevant_indices`` are the doc indices considered
            relevant ground truth for that query.
        retrieval_fn: Callable ``retrieval_fn(query, k)`` returning a list
            of ``(doc_idx, score)`` tuples.
        k: Number of results to retrieve per query.

    Returns:
        A tuple ``(avg_recall, results)`` where ``avg_recall`` is the mean
        recall across all queries, and ``results`` is a list of per-query
        dicts with keys ``query``, ``recall``, ``hits``, and
        ``total_relevant``.
    """
    total_recall = 0.0
    results = []

    for query, relevant_indices in queries_with_relevant:
        retrieved = retrieval_fn(query, k)
        retrieved_indices = set(idx for idx, _ in retrieved)
        relevant_set = set(relevant_indices)
        hits = len(retrieved_indices & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({
            "query": query,
            "recall": recall,
            "hits": hits,
            "total_relevant": len(relevant_set)
        })

    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0
    return avg_recall, results