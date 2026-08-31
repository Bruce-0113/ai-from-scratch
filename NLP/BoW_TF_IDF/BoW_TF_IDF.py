import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def build_vocab(docs):
    """Build a vocabulary mapping each unique token to an integer index.

    Args:
        docs: List of documents, where each document is a list of tokens.

    Returns:
        A dict mapping token -> index, in first-seen order.
    """
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def bag_of_words(docs, vocab):
    """Convert documents into a bag-of-words count matrix.

    Args:
        docs: List of documents, where each document is a list of tokens.
        vocab: Dict mapping token -> index, as produced by build_vocab.

    Returns:
        A list of rows (one per document), each row a list of token counts
        aligned to vocab indices.
    """
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix

def bag_of_words_sample():
    """Demo: build a vocabulary and bag-of-words matrix for two toy documents."""
    docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
    vocab = build_vocab(docs)
    print(vocab)
    print(bag_of_words(docs, vocab))

def term_frequency(doc_bow, doc_length):
    """Compute term frequency (TF) for a single document's bag-of-words row.

    Args:
        doc_bow: List of token counts for one document.
        doc_length: Total token count in the document (sum of doc_bow).

    Returns:
        List of term frequencies (count / doc_length), 0 if doc_length is 0.
    """
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    """Count how many documents each vocabulary token appears in.

    Args:
        bow_matrix: List of bag-of-words rows, one per document.

    Returns:
        List of document frequencies (DF), aligned to vocab indices.
    """
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    """Compute smoothed inverse document frequency (IDF) for each token.

    Uses the smoothed formula log((n_docs + 1) / (df + 1)) + 1 to avoid
    division by zero and avoid zeroing out terms that appear in every doc.

    Args:
        df: List of document frequencies, one per vocabulary token.
        n_docs: Total number of documents in the corpus.

    Returns:
        List of IDF values aligned to vocab indices.
    """
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]


def tfidf(bow_matrix):
    """Compute TF-IDF scores for a corpus given its bag-of-words matrix.

    Args:
        bow_matrix: List of bag-of-words rows, one per document.

    Returns:
        List of TF-IDF rows, one per document, aligned to vocab indices.
    """
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out

def tfidf_sample():
    """Demo: build vocabulary, bag-of-words, and TF-IDF for three toy documents."""
    docs = [
        ["the", "cat", "sat"],
        ["the", "dog", "sat"],
        ["the", "cat", "ran"],]

    vocab = build_vocab(docs)
    bow = bag_of_words(docs, vocab)
    print(tfidf(bow))

def l2_normalize(matrix):
    """L2-normalize each row of a matrix so it has unit Euclidean norm.

    Args:
        matrix: List of numeric rows (e.g. TF-IDF vectors).

    Returns:
        List of normalized rows; a row of all zeros stays all zeros.
    """
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out

def scikit_learn_sample():
    """Demo: reproduce bag-of-words and TF-IDF using scikit-learn's vectorizers.

    Uses CountVectorizer for bag-of-words counts and TfidfVectorizer for
    TF-IDF scores, as a reference comparison against the manual implementation.
    """
    docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

    bow_vectorizer = CountVectorizer()
    bow = bow_vectorizer.fit_transform(docs)
    print(bow_vectorizer.get_feature_names_out())
    print(bow.toarray())

    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(docs)
    print(tfidf.toarray().round(3))