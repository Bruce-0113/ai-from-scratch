"""
Part-of-speech tagging demo.

Contains two from-scratch POS taggers built on simple statistical models
(Most-Frequent-Tag baseline and a bigram HMM decoded with Viterbi), plus a
spaCy-based tagging example for comparison.
"""

import math
from collections import Counter, defaultdict

import spacy


def train_mft(train_examples):
    """Train a Most-Frequent-Tag (MFT) baseline tagger.

    For each word, remembers the tag it was seen with most often in
    training. Also tracks the single most common tag overall, used as a
    fallback for unseen words.

    Args:
        train_examples: Iterable of (tokens, tags) pairs, where tokens and
            tags are parallel lists of words and their gold POS tags.

    Returns:
        A tuple (word_best, default_tag):
            word_best: dict mapping lowercased word -> most frequent tag.
            default_tag: fallback tag for words not seen in training.
    """
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    """Tag tokens using the MFT baseline.

    Args:
        tokens: List of words to tag.
        word_best: dict from `train_mft`, mapping lowercased word -> tag.
        default_tag: Fallback tag for words not in `word_best`.

    Returns:
        List of predicted tags, one per input token.
    """
    return [word_best.get(t.lower(), default_tag) for t in tokens]


def train_hmm(train_examples, alpha=0.01):
    """Estimate transition and emission counts for a bigram HMM tagger.

    Transitions are tag -> tag counts (including <BOS>/<EOS> boundaries);
    emissions are tag -> word counts. Raw counts are returned rather than
    normalized probabilities so that Laplace smoothing can be applied
    consistently at decode time (see `log_prob` and `viterbi`).

    Args:
        train_examples: Iterable of (tokens, tags) pairs.
        alpha: Smoothing constant (unused during counting; kept for API
            symmetry with the decoding functions that consume it).

    Returns:
        A tuple (transitions, emissions, tags, vocab):
            transitions: dict[tag_or_BOS][tag_or_EOS] -> count.
            emissions: dict[tag][lowercased_word] -> count.
            tags: set of all tags seen in training.
            vocab: set of all lowercased words seen in training.
    """
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    """Compute a Laplace-smoothed log probability from a count table.

    Args:
        table: dict[given] -> Counter(key -> count), e.g. transitions or
            emissions from `train_hmm`.
        given: The conditioning context (e.g. previous tag).
        key: The event being scored (e.g. current tag or word).
        smooth_denom: Precomputed denominator, typically
            sum(table[given].values()) + alpha * (num_possible_keys + 1).
        alpha: Laplace smoothing constant added to the numerator.

    Returns:
        log((count(given, key) + alpha) / smooth_denom).
    """
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    """Decode the most likely tag sequence for a sentence via Viterbi.

    Runs the standard dynamic-programming Viterbi algorithm over a bigram
    HMM whose transition/emission counts come from `train_hmm`, applying
    Laplace smoothing for unseen tag/word combinations.

    Args:
        tokens: List of words in the sentence to tag.
        transitions: Transition counts from `train_hmm`.
        emissions: Emission counts from `train_hmm`.
        tags: Set of all possible tags.
        vocab: Set of all known words (used to size the smoothing
            denominator for unseen emissions).
        alpha: Laplace smoothing constant.

    Returns:
        List of predicted tags, one per input token, forming the highest
        scoring path through the trellis.
    """
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]


def demo_spacy_tagging(text):
    """Tag and print POS/dependency info for a sentence using spaCy.

    Args:
        text: Raw sentence to parse and tag.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for token in doc:
        print(
            f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} "
            f"dep={token.dep_:10s} head={token.head.text}"
        )


if __name__ == "__main__":
    demo_spacy_tagging("The cats were running at 3pm.")
