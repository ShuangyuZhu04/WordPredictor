"""
N-Gram Language Model for Word Prediction.

Supports unigram, bigram, and trigram context. Predictions are updated
per-keystroke by filtering the vocabulary on the current prefix.
"""

from collections import defaultdict, Counter
import re
import math
from typing import List, Tuple, Optional
import nltk


class NGramPredictor:
    """Word predictor backed by n-gram frequency tables.

    Parameters
    ----------
    n : int
        The order of the model (1 = unigram, 2 = bigram, 3 = trigram).
    """

    # ------------------------------------------------------------------ #
    #  Construction & training                                            #
    # ------------------------------------------------------------------ #
    def __init__(self, n: int = 3):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        # ngram_counts[order][(context_tuple)] -> Counter of next words
        # order ranges from 1 (unigram) to n
        self.ngram_counts: dict[int, dict[tuple, Counter]] = {
            order: defaultdict(Counter) for order in range(1, n + 1)
        }
        self.vocabulary: set[str] = set()
        self.word_freq: Counter = Counter()  # raw unigram counts
        self._total_tokens: int = 0

    # ---- text normalisation ------------------------------------------ #
    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z]+(?:['\-][a-z]+)*", text.lower())

    # ---- training ---------------------------------------------------- #
    def train(self, corpus: str) -> None:
        """Build n-gram tables from a raw text corpus."""
        tokens = self.tokenize(corpus)
        self._total_tokens += len(tokens)
        self.vocabulary.update(tokens)
        self.word_freq.update(tokens)

        for order in range(1, self.n + 1):
            for i in range(len(tokens) - order + 1):
                window = tokens[i : i + order]
                context = tuple(window[:-1])  # empty tuple for unigram
                word = window[-1]
                self.ngram_counts[order][context][word] += 1

    # ------------------------------------------------------------------ #
    #  Prediction                                                         #
    # ------------------------------------------------------------------ #
    def predict(
        self,
        context: List[str],
        prefix: str = "",
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return the *top_k* most likely next words.

        Parameters
        ----------
        context : list[str]
            The preceding words (already typed and confirmed).
        prefix : str
            The partially-typed current word (may be empty).
        top_k : int
            How many suggestions to return.

        Returns
        -------
        list of (word, probability) pairs sorted by descending probability.
        """
        prefix = prefix.lower().strip()
        context = [w.lower() for w in context]

        # Try the highest-order model first, then back off.
        candidates: Counter = Counter()
        for order in range(self.n, 0, -1):
            ctx_len = order - 1
            ctx = tuple(context[-ctx_len:]) if ctx_len > 0 else ()

            if ctx in self.ngram_counts[order]:
                counts = self.ngram_counts[order][ctx]
                # Filter by prefix
                filtered = {w: c for w, c in counts.items() if w.startswith(prefix)}
                if filtered:
                    total = sum(filtered.values())
                    for w, c in filtered.items():
                        candidates[w] += c / total
                    break  # stop at the highest useful order

        results = [(w, round(s, 4)) for w, s in candidates.most_common(top_k)]
        return results

    # ------------------------------------------------------------------ #
    #  Convenience helpers                                                #
    # ------------------------------------------------------------------ #
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    def __repr__(self) -> str:
        return (
            f"NGramPredictor(n={self.n}, "
            f"vocab={self.vocabulary_size()}, "
            f"tokens={self._total_tokens})"
        )
