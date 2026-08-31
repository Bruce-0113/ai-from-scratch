import re
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# nltk.download("punkt_tab")
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger_eng")


def tokenize(text):
    """Split text into word (including contractions), number, and punctuation tokens.

    Args:
        text: Raw input string.

    Returns:
        list[str]: Extracted tokens in order of appearance.
    """
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)

def stem_step_1a(word):
    """Apply Porter stemmer step 1a plural-suffix rules to a single word.

    Args:
        word: The word to stem.

    Returns:
        str: The stemmed word.
    """
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word

LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    """Look up a word's lemma in LEMMA_TABLE, falling back to simple suffix rules.

    Args:
        word: The word to lemmatize.
        pos: Coarse POS tag (e.g. "VERB", "NOUN", "ADJ") used for the lookup.

    Returns:
        str: The lemmatized (lowercased) word.
    """
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()

def preprocess(text, pos_tagger=None):
    """Run the custom tokenize -> stem -> (tag) -> lemmatize pipeline on text.

    Args:
        text: Raw input string.
        pos_tagger: Optional callable(tokens) -> list[(token, pos)]; if omitted,
            every token is tagged as "NOUN".

    Returns:
        dict: {"tokens": list[str], "stems": list[str], "lemmas": list[str]}.
    """
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}

def nltk_pos_to_wordnet(tag):
    """Map an NLTK POS tag to the single-letter POS code WordNet's lemmatizer expects."""
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


def run_nltk_pipeline(text):
    """Tokenize, stem, POS-tag, and lemmatize text using NLTK, then print each step."""
    tokens = word_tokenize(text)
    stems = [PorterStemmer().stem(t) for t in tokens]
    tagged = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]

    print("tokens:", tokens)
    print("stems: ", stems)
    print("tags:  ", tagged)
    print("lemmas:", lemmas)


def run_spacy_pipeline(text):
    """Tokenize, lemmatize, and POS-tag text using spaCy, then print each token's info."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for token in doc:
        print(token.text, token.lemma_, token.pos_)


# tokenize
print(tokenize("The cats weren't running at 3pm."))
# stem step 1a
print([stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]])
# lemmatize
print(lemmatize("running", "VERB"))
print(lemmatize("cats", "NOUN"))
print(lemmatize("better", "ADJ"))
print(lemmatize("watched", "VERB"))
# nltk pipeline
run_nltk_pipeline("The cats were running.")
# spacy pipline
run_spacy_pipeline("The cats were running.")
