"""Byte-level BPE tokenizer, built from scratch and benchmarked against a
character-level baseline and OpenAI's production tokenizer (tiktoken).

Mirrors the "Build It" walkthrough from the tokenizers lesson in
rohitg00/ai-engineering-from-scratch (phases/10-llms-from-scratch/01-tokenizers):

1. CharTokenizer             -- naive one-codepoint-per-token baseline
2. BPETokenizer              -- byte-level BPE: count pairs, merge the most
                                 frequent, repeat
3. demo_roundtrip            -- encode/decode roundtrip + compression ratio
                                 vs. raw bytes
4. demo_tiktoken_comparison  -- token count vs. tiktoken's cl100k_base encoding
5. analyze_vocabulary        -- vocab usage stats over the trained tokenizer

Run directly (`python tokenizers.py`) to reproduce the walkthrough end to end.
"""

from collections import Counter

import tiktoken


class CharTokenizer:
    """Maps each Unicode code point to its integer ordinal and back.

    No training needed and no unknown tokens, but each character is its own
    token, so sequences are several times longer than a subword tokenizer
    would produce for the same text.
    """

    def encode(self, text):
        """Encode text as a list of Unicode code points (`ord` per character)."""
        return [ord(c) for c in text]

    def decode(self, tokens):
        """Decode a list of Unicode code points back into text (`chr` per token)."""
        return "".join(chr(t) for t in tokens)


class BPETokenizer:
    """Byte-level Byte Pair Encoding tokenizer, the algorithm behind GPT-2/3/4.

    Trains by repeatedly merging the most frequent adjacent pair of tokens in
    a byte-encoded corpus, starting from the 256 possible raw bytes. The
    learned merges double as the vocabulary: each merge both compresses the
    training corpus and adds one new token, so the vocabulary grows by
    exactly `num_merges` on top of the base 256 bytes.

    Attributes:
        merges: Ordered mapping of (token_a, token_b) -> merged_token_id, in
            the order the merges were learned. Order matters at encode time:
            merges must be replayed in this order so that, e.g., "th" can
            form before "the" does.
        vocab: Mapping of token_id -> the raw bytes it expands to, used to
            decode token IDs back into text.
    """

    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def _get_pairs(self, tokens):
        """Count occurrences of every adjacent token pair in `tokens`.

        Returns:
            Counter mapping (token_i, token_i+1) -> occurrence count.
        """
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_pair(self, tokens, pair, new_token):
        """Replace every non-overlapping occurrence of `pair` in `tokens` with `new_token`.

        Scans left to right; once a pair is merged, its second element is
        consumed and cannot also serve as the start of the next pair.
        """
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def train(self, text, num_merges):
        """Learn up to `num_merges` BPE merges from `text`.

        Args:
            text: Training corpus. Encoded as UTF-8 bytes before merging, so
                the base vocabulary is the 256 possible byte values.
            num_merges: Maximum number of merge operations to learn. Training
                stops early if no adjacent pair remains.

        Returns:
            self, so training can be chained: `BPETokenizer().train(text, 40)`.
        """
        tokens = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            new_token = 256 + i
            tokens = self._merge_pair(tokens, best_pair, new_token)
            self.merges[best_pair] = new_token
            self.vocab[new_token] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

        return self

    def encode(self, text):
        """Encode `text` into token IDs by replaying learned merges in order.

        Byte pairs that were never merged during training fall back to raw
        bytes, which is why compression is worse on text outside the
        training distribution.

        Returns:
            List of token IDs.
        """
        tokens = list(text.encode("utf-8"))
        for pair, new_token in self.merges.items():
            tokens = self._merge_pair(tokens, pair, new_token)
        return tokens

    def decode(self, tokens):
        """Decode token IDs previously produced by `encode` back into text.

        Invalid UTF-8 byte sequences are replaced rather than raising, since
        an arbitrary slice of tokens is not guaranteed to fall on a valid
        UTF-8 character boundary.
        """
        byte_sequence = b"".join(self.vocab[t] for t in tokens)
        return byte_sequence.decode("utf-8", errors="replace")


def analyze_vocabulary(tokenizer, test_texts):
    """Print vocabulary coverage stats for `tokenizer` over `test_texts`.

    Reports overall compression (tokens per character), the 10 most-used
    tokens (a proxy for the corpus's Zipf distribution), and how much of the
    trained vocabulary went unused on this sample -- a sign the training
    corpus was too small or too narrow relative to `num_merges`.
    """
    total_tokens = 0
    total_chars = 0
    token_usage = Counter()

    for text in test_texts:
        encoded = tokenizer.encode(text)
        total_tokens += len(encoded)
        total_chars += len(text)
        for t in encoded:
            token_usage[t] += 1

    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Total tokens across all texts: {total_tokens}")
    print(f"Total characters: {total_chars}")
    print(f"Avg tokens per character: {total_tokens / total_chars:.2f}")

    print("\nMost used tokens:")
    for token_id, count in token_usage.most_common(10):
        token_bytes = tokenizer.vocab[token_id]
        display = token_bytes.decode("utf-8", errors="replace")
        print(f"  Token {token_id:4d}: '{display}' (used {count} times)")

    unused = [t for t in tokenizer.vocab if t not in token_usage]
    print(f"\nUnused tokens: {len(unused)} out of {len(tokenizer.vocab)}")


def demo_roundtrip(tokenizer, sentences):
    """Encode/decode each sentence, printing token count, compression ratio
    vs. raw UTF-8 bytes, and whether the roundtrip is lossless."""
    for sentence in sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded)
        raw_bytes = len(sentence.encode("utf-8"))
        ratio = len(encoded) / raw_bytes
        print(f"'{sentence}'")
        print(f"  Tokens: {len(encoded)} (from {raw_bytes} bytes) -- ratio: {ratio:.2f}")
        print(f"  Roundtrip: {'PASS' if decoded == sentence else 'FAIL'}")


def demo_tiktoken_comparison(tokenizer, texts):
    """Encode each text with both `tokenizer` and OpenAI's cl100k_base
    encoding, printing token counts side by side to show how a small,
    corpus-specific vocabulary compares to a production tokenizer trained on
    a massive corpus with ~100k merges."""
    enc = tiktoken.get_encoding("cl100k_base")

    for text in texts:
        our_tokens = tokenizer.encode(text)
        tiktoken_tokens = enc.encode(text)
        tiktoken_pieces = [enc.decode([t]) for t in tiktoken_tokens]
        print(f"'{text}'")
        print(f"  Our BPE:   {len(our_tokens)} tokens")
        print(f"  tiktoken:  {len(tiktoken_tokens)} tokens -> {tiktoken_pieces}")


def main():
    """Train the BPE tokenizer on a small corpus and run all three demos:
    roundtrip, tiktoken comparison, and vocabulary analysis."""
    corpus = (
        "The cat sat on the mat. The cat ate the rat. "
        "The dog sat on the log. The dog ate the frog. "
        "Natural language processing is the study of how computers "
        "understand and generate human language. "
        "Tokenization is the first step in any NLP pipeline."
    )
    tokenizer = BPETokenizer().train(corpus, num_merges=40)

    test_sentences = [
        "The cat sat on the mat.",
        "Natural language processing",
        "tokenization pipeline",
        "unhappiness",
    ]
    demo_roundtrip(tokenizer, test_sentences)

    comparison_texts = [
        "The cat sat on the mat.",
        "unhappiness",
        "Hello, world!",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "Geschwindigkeitsbegrenzung",
    ]
    demo_tiktoken_comparison(tokenizer, comparison_texts)

    analyze_vocabulary(tokenizer, test_sentences + comparison_texts)


if __name__ == "__main__":
    main()
