"""Chunking strategies for RAG: fixed, recursive, semantic, parent-document, and contextual retrieval.

Splitting documents into retrievable chunks influences retrieval quality
as much as the choice of embedding model. This file builds five
complementary strategies, roughly simplest to most involved, plus a
recall@k harness to compare them empirically instead of assuming a
"best" default:

1. Fixed-size chunking (`chunk_fixed`): naive fixed-character windows
   with optional overlap. Cheapest baseline; ignores sentence/paragraph
   boundaries, so chunks can break mid-sentence.
2. Recursive chunking (`chunk_recursive`): LangChain's
   `RecursiveCharacterTextSplitter` pattern - try splitting on paragraph
   breaks first, then line breaks, then sentences, then words, greedily
   packing pieces back together up to `size`. The common production
   default.
3. Semantic chunking (`chunk_semantic`): embed each sentence, cut where
   cosine similarity between adjacent sentences drops below a threshold,
   with a `min_chars` floor to avoid tiny fragments and a `max_chars`
   ceiling that falls back to recursive chunking.
4. Parent-document chunking (`chunk_parent_child` + `retrieve_parent`):
   index small "child" chunks for precise retrieval, but return their
   larger "parent" chunk so the caller gets full surrounding context.
5. Contextual retrieval (`contextualize_chunks`): Anthropic's pattern of
   prepending each chunk with an LLM-generated summary of where it sits
   in the document, before indexing, to reduce ambiguity for chunks that
   read poorly out of context.

`recall_at_k` closes the loop: given labeled (query, gold_chunk_indices)
pairs, it measures what fraction of queries retrieve a correct chunk in
the top k - the way to actually pick a strategy for a given corpus
rather than assuming a "best" default (e.g. recursive chunking can beat
semantic chunking on some benchmarks despite being the simpler option).

Two things this file assumes are supplied by the caller rather than
defined here: `split_sentences` (a sentence-splitting utility used by
`chunk_semantic`, e.g. nltk.sent_tokenize or a regex splitter) and `llm`
(an object exposing `.batch(prompts) -> list[str]`, used by
`contextualize_chunks`). `encoder` parameters throughout expect a
SentenceTransformer-like model: `.encode(list[str],
normalize_embeddings=True) -> np.ndarray`, with embeddings assumed unit
L2-normalized so dot products double as cosine similarity.
"""

import numpy as np


def chunk_fixed(text, size=512, overlap=0):
    """Split text into fixed-size character windows with optional overlap.

    Simplest baseline: slices `text` every `size` characters, stepping by
    `size - overlap` so consecutive windows share `overlap` characters.
    Ignores sentence/word boundaries entirely, so chunks can break
    mid-word or mid-sentence.

    Args:
        text: The full text to chunk.
        size: Chunk length in characters.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        List of chunk strings covering `text` end to end (the final
        chunk may be shorter than `size` if the text doesn't divide
        evenly).
    """
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    """Recursively split text on a priority list of separators until each chunk fits `size`.

    Mimics LangChain's `RecursiveCharacterTextSplitter`: finds the first
    separator in `seps` that actually occurs in `text`, splits on it, and
    greedily re-merges the resulting pieces so each returned chunk is as
    close to `size` as possible without exceeding it. Any single piece
    still longer than `size` after that split recurses into
    `chunk_recursive` with the remaining, finer-grained separators
    (`seps[1:]`, falling back to a plain space once separators run out).
    If none of `seps` appear anywhere in `text`, falls back to
    `chunk_fixed`.

    Args:
        text: The text to chunk.
        size: Maximum chunk length in characters.
        seps: Separators to try, in priority order from coarsest
            (paragraph break) to finest (word break).

    Returns:
        List of non-empty chunk strings, each at most `size` characters
        unless a single atomic piece (e.g. one very long "sentence")
        can't be split further by the remaining separators.
    """
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)


def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    """Group sentences into chunks by cutting where semantic similarity drops.

    Encodes every sentence with `encoder`, then walks through them in
    order comparing each to the previous one via cosine similarity (dot
    product of unit-normalized embeddings). Starts a new chunk when the
    similarity to the previous sentence falls below `threshold`, but only
    once the current chunk has already reached `min_chars` - this keeps a
    single similarity dip from producing a tiny fragment. Any resulting
    chunk longer than `max_chars` is further split with `chunk_recursive`.

    Args:
        text: The full text to chunk.
        encoder: A sentence-embedding model exposing
            `.encode(list[str], normalize_embeddings=True) -> np.ndarray`.
        threshold: Minimum cosine similarity between adjacent sentences
            required to keep them in the same chunk. Higher values
            produce more, smaller chunks.
        min_chars: Minimum accumulated chunk length (characters) before
            a similarity drop is allowed to start a new chunk.
        max_chars: Maximum chunk length; oversized chunks are recursively
            split via `chunk_recursive` instead of returned as-is.

    Returns:
        List of chunk strings, each a whitespace-joined group of
        sentences. Empty list if `text` contains no sentences.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result


def chunk_parent_child(text, parent_size=2048, child_size=256):
    """Build a two-level parent/child chunk index for parent-document retrieval.

    Splits `text` into large "parent" chunks via `chunk_recursive`, then
    splits each parent further into small "child" chunks. The idea:
    retrieval matches against the small, topically focused child chunks,
    but the context returned to the caller is the larger parent chunk,
    which preserves surrounding context a lone child chunk would lose.

    Args:
        text: The full text to chunk.
        parent_size: Maximum parent chunk length in characters.
        child_size: Maximum child chunk length in characters.

    Returns:
        List of dicts, one per child chunk, each with keys:
            "child": the child chunk text (what gets embedded/searched).
            "parent_idx": index of the parent chunk it was split from.
            "parent": the full parent chunk text (what gets returned).
    """
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    """Retrieve deduplicated parent chunks for a query via child-chunk similarity search.

    Embeds every child chunk in `mapping` plus the query, ranks children
    by cosine similarity to the query, then walks the ranked children
    keeping their parent chunks in order while skipping any parent
    already seen - since several children can map to the same parent,
    naively returning one parent per top child would waste context on
    duplicates.

    Args:
        child_query: The query text to search with.
        mapping: List of child/parent dicts as produced by
            `chunk_parent_child`.
        encoder: A sentence-embedding model exposing
            `.encode(list[str], normalize_embeddings=True) -> np.ndarray`.
        top_k: Number of top-scoring child chunks to consider.

    Returns:
        List of unique parent chunk strings, ordered by the rank of
        their best-matching child. Length is at most `top_k` but
        typically shorter once duplicate parents are collapsed.
    """
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents


def contextualize_chunks(document, chunks, llm):
    """Prepend an LLM-generated situating summary to each chunk (Anthropic's contextual retrieval).

    For every chunk, prompts `llm` to write a short summary of where
    that chunk sits within `document`, then prefixes the chunk with its
    generated summary before it gets indexed. The extra context helps
    retrieval surface chunks whose content is ambiguous when read in
    isolation, at the cost of one LLM call per chunk at index time.

    Args:
        document: The full source document, given to the LLM as context
            for writing each summary.
        chunks: List of chunk strings to contextualize.
        llm: An LLM client exposing `.batch(list[str]) -> list[str]`,
            called once with all prompts.

    Returns:
        List of strings, one per input chunk, each formatted as
        "{generated_context}\\n\\n{original_chunk}".
    """
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]


def recall_at_k(queries, corpus_chunks, encoder, k=5):
    """Compute recall@k: the fraction of queries whose relevant chunk is retrieved in the top k.

    For each (query, gold_idxs) pair, embeds the query, ranks all
    `corpus_chunks` by cosine similarity, and counts a hit if any of the
    top-`k` ranked indices is in `gold_idxs`. This is the metric to
    benchmark chunking strategies against each other on real data,
    rather than assuming one strategy is universally best.

    Args:
        queries: List of (query_text, gold_idxs) tuples, where gold_idxs
            is a set/list of indices into `corpus_chunks` considered
            correct for that query.
        corpus_chunks: List of chunk strings making up the searchable
            corpus.
        encoder: A sentence-embedding model exposing
            `.encode(list[str], normalize_embeddings=True) -> np.ndarray`.
        k: Number of top-ranked chunks to consider per query.

    Returns:
        Recall@k as a float in [0, 1]: hits divided by number of queries.
    """
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)