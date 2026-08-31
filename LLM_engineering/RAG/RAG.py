"""Minimal Retrieval-Augmented Generation (RAG) pipeline built from scratch.

Implements text chunking, TF-IDF embeddings, cosine-similarity search, and
prompt construction without any external ML libraries. Useful as a reference
for how a RAG pipeline works end-to-end before reaching for a vector database
or a real embedding model.
"""

import math
from collections import Counter


def chunk_text(text, chunk_size=200, overlap=50):
    """Split text into overlapping chunks of whole words.

    Args:
        text: The raw document text to split.
        chunk_size: Number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        A list of chunk strings.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_vocabulary(documents):
    """Build a sorted vocabulary of unique lowercase words across documents."""
    vocab = set()
    for doc in documents:
        vocab.update(doc.lower().split())
    return sorted(vocab)


def compute_tf(text, vocab):
    """Compute term frequency of text for each word in vocab."""
    words = text.lower().split()
    count = Counter(words)
    total = len(words)
    return [count.get(word, 0) / total for word in vocab]


def compute_idf(documents, vocab):
    """Compute smoothed inverse document frequency for each word in vocab."""
    n = len(documents)
    idf = []
    for word in vocab:
        doc_count = sum(1 for doc in documents if word in doc.lower().split())
        idf.append(math.log((n + 1) / (doc_count + 1)) + 1)
    return idf


def tfidf_embed(text, vocab, idf):
    """Embed text as a TF-IDF vector aligned to vocab."""
    tf = compute_tf(text, vocab)
    return [t * i for t, i in zip(tf, idf)]


def cosine_similarity(a, b):
    """Compute cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search(query_embedding, stored_embeddings, top_k=5):
    """Return the indices and scores of the top_k most similar embeddings.

    Args:
        query_embedding: Embedding vector of the query.
        stored_embeddings: List of embedding vectors to search against.
        top_k: Number of top results to return.

    Returns:
        A list of (index, similarity_score) tuples sorted by descending score.
    """
    scores = []
    for i, emb in enumerate(stored_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        scores.append((i, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def build_rag_prompt(query, retrieved_chunks):
    """Build a grounded prompt that instructs the model to answer only from context."""
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    )
    rag_prompt = f"""Answer the question based ONLY on the following context.
If the context doesn't contain enough information, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""
    return rag_prompt


class RAGPipeline:
    """A minimal end-to-end RAG pipeline using TF-IDF retrieval."""

    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.vocab = []
        self.idf = []

    def index(self, documents):
        """Chunk and embed a list of documents, building the vocabulary and IDF table."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunk_text(doc))
        self.chunks = all_chunks
        self.vocab = build_vocabulary(all_chunks)
        self.idf = compute_idf(all_chunks, self.vocab)
        self.embeddings = [
            tfidf_embed(chunk, self.vocab, self.idf)
            for chunk in all_chunks
        ]

    def query(self, question, top_k=5):
        """Retrieve the top_k most relevant chunks and build a RAG prompt for the question.

        Returns:
            A tuple of (prompt, retrieved) where retrieved is a list of
            (chunk_text, similarity_score) pairs.
        """
        query_emb = tfidf_embed(question, self.vocab, self.idf)
        results = search(query_emb, self.embeddings, top_k)
        retrieved = [(self.chunks[i], score) for i, score in results]
        prompt = build_rag_prompt(
            question, [chunk for chunk, _ in retrieved]
        )
        return prompt, retrieved


def simple_generate(prompt, retrieved_chunks):
    """Generate a naive answer by picking the sentence with the most word overlap with the question.

    This stands in for a real LLM call: it scores each sentence in the
    retrieved chunks by how many query words it shares and returns the best
    match, so the pipeline can be demoed without an actual generation model.
    """
    query_words = set(prompt.lower().split("question:")[-1].split())
    best_sentence = ""
    best_score = 0
    for chunk in retrieved_chunks:
        for sentence in chunk.split("."):
            sentence = sentence.strip()
            if not sentence:
                continue
            words = set(sentence.lower().split())
            overlap = len(query_words & words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
    return best_sentence if best_sentence else "I don't have enough information."
