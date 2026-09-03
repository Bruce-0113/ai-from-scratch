"""Embedding models for semantic search: bi-encoders, Matryoshka truncation, and hybrid multi-vector retrieval.

Covers three complementary techniques for turning text into vectors you can
rank/retrieve with, roughly from simplest to most involved:

1. Bi-encoder dense retrieval (`SentenceTransformer` / bge-small-en-v1.5):
   encode queries and documents independently into a single normalized
   vector each; cosine similarity (here, a plain dot product, since vectors
   are L2-normalized) ranks documents by semantic closeness to the query.
2. Matryoshka-style dimension truncation (`truncate`): for models trained
   with a Matryoshka objective, the leading dimensions of the embedding are
   themselves a valid lower-dimensional embedding, so a vector can be sliced
   short and re-normalized to save storage/compute at a small accuracy cost.
3. BGE-M3 hybrid multi-vector retrieval: one model emits three complementary
   signals per passage - dense (single vector, semantic), sparse/lexical
   (learned term weights, keyword-style matching), and ColBERT (one vector
   per token, fine-grained MaxSim) - combined into a single weighted score.
4. MTEB evaluation: benchmarking the bi-encoder from step 1 against standard
   retrieval tasks (ArguAna, SciFact, NFCorpus) instead of a toy example.

The BGE-M3 scoring block (step 3) is illustrative pseudocode mirroring the
model's documented three-signal formula, not a runnable snippet: `q_lex`/
`d_lex`/`q_col`/`d_col` (per-query/doc lexical weights and ColBERT vectors)
are never computed here, and `dense_score` is a hardcoded placeholder.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from mteb import MTEB

# --- 1. Bi-encoder dense retrieval -----------------------------------------
# Encode the corpus and the query independently (a "bi-encoder"), each into
# one normalized vector. Because normalize_embeddings=True makes every
# vector unit-length, the dot product below is equivalent to cosine
# similarity, so ranking by `scores` ranks documents by semantic closeness
# to the query.
encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))  # (doc_index, score), best match first


# --- 2. Matryoshka-style dimension truncation ------------------------------
def truncate(vectors, dim):
    """Truncate normalized embeddings to their first `dim` dimensions and re-normalize.

    For models trained with a Matryoshka objective, the leading dimensions
    of the embedding already form a valid lower-dimensional embedding, so
    slicing them out and rescaling to unit length trades a little accuracy
    for less storage/compute (e.g. cutting a 384-dim vector to 128 dims).

    Args:
        vectors: Array of shape (n_vectors, orig_dim), rows assumed unit-norm.
        dim: Number of leading dimensions to keep (must be <= orig_dim).

    Returns:
        Array of shape (n_vectors, dim), each row re-normalized to unit L2 norm.
    """
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)


# --- 3. BGE-M3 hybrid multi-vector retrieval -------------------------------
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:      (n_docs, 1024) - single semantic vector per doc
# output["lexical_weights"]: list of dict {token_id: weight} - learned term
#                            weights for sparse/keyword-style matching
# output["colbert_vecs"]:    list of (n_tokens, 1024) arrays - one vector per
#                            token, compared against query tokens via MaxSim


# Illustrative only (see module docstring): combining the three signals into
# one hybrid score. q_lex/d_lex and q_col/d_col would need to be encoded for
# a specific query the same way `output` was encoded for the corpus above;
# dense_score stands in for an actual cosine similarity over dense_vecs.
dense_score = 0.5  # cosine similarity over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score


# --- 4. MTEB benchmark evaluation ------------------------------------------
# Evaluate the step-1 bi-encoder against standard retrieval benchmarks (MTEB:
# Massive Text Embedding Benchmark) rather than relying on the toy 3-document
# example above. A useful leaderboard sanity check, but always validate on
# your own domain data too - a high MTEB rank is necessary, not sufficient.
tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
