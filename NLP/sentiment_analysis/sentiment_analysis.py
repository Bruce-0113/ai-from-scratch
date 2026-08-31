import math
import numpy as np
from collections import Counter


POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]


def train_nb(docs_by_class, vocab, alpha=1.0):
    """Train a Multinomial Naive Bayes classifier with Laplace smoothing.

    Args:
        docs_by_class: Mapping of class label -> list of tokenized documents.
        vocab: Iterable of all vocabulary tokens.
        alpha: Laplace smoothing factor.

    Returns:
        A tuple of (class_priors, class_word_probs), where class_priors maps
        each class to its prior probability and class_word_probs maps each
        class to a dict of word -> conditional probability.
    """
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    """Predict the most likely class for a document using trained Naive Bayes parameters.

    Args:
        doc: Tokenized document to classify.
        class_priors: Mapping of class label -> prior probability.
        class_word_probs: Mapping of class label -> dict of word -> conditional probability.

    Returns:
        The predicted class label with the highest log-probability score.
    """
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)


def sigmoid(x):
    """Compute the numerically stable sigmoid function.

    Args:
        x: Input value or array.

    Returns:
        The sigmoid of x, with input clipped to [-20, 20] to avoid overflow.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    """Train a logistic regression classifier via batch gradient descent.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary target labels of shape (n_samples,).
        epochs: Number of gradient descent iterations.
        lr: Learning rate.
        l2: L2 regularization strength.

    Returns:
        A tuple of (w, b): the learned weight vector and bias term.
    """
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    """Predict binary class labels using a trained logistic regression model.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        w: Learned weight vector.
        b: Learned bias term.

    Returns:
        An array of predicted binary labels (0 or 1).
    """
    return (sigmoid(X @ w + b) >= 0.5).astype(int)


NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    """Tag tokens following a negation word with a NOT_ prefix until a terminator.

    Args:
        tokens: List of string tokens.

    Returns:
        A new list of tokens where words appearing after a negation word
        (e.g. "not", "never") and before a sentence terminator are prefixed
        with "NOT_".
    """
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out