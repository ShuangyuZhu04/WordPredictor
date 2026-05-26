"""
Spell Corrector for the Word Predictor system.

Uses Damerau-Levenshtein edit distance weighted by corpus frequency
to suggest corrections for misspelled words.
"""

from collections import Counter
from typing import List, Tuple, Optional
import re


class SpellCorrector:
    """Corpus-frequency-aware spell corrector.

    Parameters
    ----------
    max_edit_distance : int
        Maximum edit distance to consider (1 or 2 recommended).
    """

    def __init__(self, max_edit_distance: int = 2):
        self.max_edit_distance = max_edit_distance
        self.word_freq: Counter = Counter()
        self.vocabulary: set[str] = set()
        self._total: int = 0

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #
    def train(self, corpus: str) -> None:
        """Learn word frequencies from raw text."""
        tokens = re.findall(r"[a-zA-Z]+", corpus.lower())
        self.word_freq.update(tokens)
        self.vocabulary.update(tokens)
        self._total += len(tokens)

    def train_from_counter(self, freq: Counter, vocab: set) -> None:
        """Initialise directly from an existing frequency counter."""
        self.word_freq = freq
        self.vocabulary = vocab
        self._total = sum(freq.values())

    # ------------------------------------------------------------------ #
    #  Edit-distance computation                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def damerau_levenshtein(s: str, t: str) -> int:
        """Compute the Damerau-Levenshtein distance between two strings."""
        len_s, len_t = len(s), len(t)
        # Quick reject
        if abs(len_s - len_t) > 2:
            return abs(len_s - len_t)

        d = [[0] * (len_t + 1) for _ in range(len_s + 1)]
        for i in range(len_s + 1):
            d[i][0] = i
        for j in range(len_t + 1):
            d[0][j] = j

        for i in range(1, len_s + 1):
            for j in range(1, len_t + 1):
                cost = 0 if s[i - 1] == t[j - 1] else 1
                d[i][j] = min(
                    d[i - 1][j] + 1,        # deletion
                    d[i][j - 1] + 1,         # insertion
                    d[i - 1][j - 1] + cost,  # substitution
                )
                # transposition
                if (
                    i > 1
                    and j > 1
                    and s[i - 1] == t[j - 2]
                    and s[i - 2] == t[j - 1]
                ):
                    d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
        return d[len_s][len_t]

    # ------------------------------------------------------------------ #
    #  Candidate generation (Peter Norvig-style edits)                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _edits1(word: str) -> set[str]:
        """All strings that are one edit away from *word*."""
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word: str) -> set[str]:
        """All strings that are two edits away from *word*."""
        return {e2 for e1 in self._edits1(word) for e2 in self._edits1(e1)}

    def _known(self, words: set[str]) -> set[str]:
        """Filter to words that exist in the vocabulary."""
        return words & self.vocabulary

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def correct(
        self,
        word: str,
        top_k: int = 5,
    ) -> List[Tuple[str, int, float]]:
        """Suggest corrections for *word*.

        Returns
        -------
        list of (candidate, edit_distance, score) sorted by descending score.
        The score combines closeness (inverse distance) and corpus frequency.
        """
        word = word.lower().strip()

        # Very short strings produce huge candidate sets with no value
        if len(word) < 2:
            return []

        # If the word is already known, return it first
        if word in self.vocabulary:
            candidates = {word}
        else:
            candidates = set()

        # Generate candidates at increasing edit distances
        ed1 = self._known(self._edits1(word))
        candidates |= ed1

        # Only fall back to the expensive edits2 when edits1 found too
        # few candidates AND the word is long enough for edits2 to be
        # meaningful (short words produce ~50K junk candidates).
        if (
            self.max_edit_distance >= 2
            and len(candidates) < top_k
            and len(word) >= 4
        ):
            ed2 = self._known(self._edits2(word))
            candidates |= ed2

        if not candidates:
            return []

        # Score: frequency / (distance + 1)^2  — strongly prefer close matches
        scored: List[Tuple[str, int, float]] = []
        for c in candidates:
            dist = self.damerau_levenshtein(word, c)
            if dist > self.max_edit_distance:
                continue
            freq = self.word_freq.get(c, 1)
            score = freq / ((dist + 1) ** 2)
            scored.append((c, dist, round(score, 4)))

        scored.sort(key=lambda x: -x[2])
        return scored[:top_k]

    def is_known(self, word: str) -> bool:
        return word.lower().strip() in self.vocabulary

    def __repr__(self) -> str:
        return (
            f"SpellCorrector(max_edit={self.max_edit_distance}, "
            f"vocab={len(self.vocabulary)})"
        )