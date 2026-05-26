"""
Corpus loader – centralised data loading for the Word Predictor project.

Uses the Brown corpus from NLTK (~1.15 M words, 500 texts across 15 genres).
Handles download, preprocessing, and train/test splitting in one place.

Usage:
    from corpus_loader import load_corpus
    train, test = load_corpus()
"""

from __future__ import annotations

import os
import re
import random
from typing import Tuple

# -------------------------------------------------------------------- #
#  NLTK bootstrap – download only if missing                            #
# -------------------------------------------------------------------- #

def _ensure_nltk_data() -> None:
    """Download the Brown corpus + Punkt tokenizer if not already cached."""
    import nltk

    for resource in ["brown", "punkt_tab"]:
        try:
            nltk.data.find(
                f"corpora/{resource}" if resource == "brown"
                else f"tokenizers/{resource}"
            )
        except LookupError:
            print(f"  Downloading NLTK resource: {resource} …")
            nltk.download(resource, quiet=True)


# -------------------------------------------------------------------- #
#  Public API                                                           #
# -------------------------------------------------------------------- #

def load_corpus(
    categories: list[str] | None = None,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_train_words: int | None = None,
) -> Tuple[str, str]:
    """Load the Brown corpus and return (train_text, test_text).

    Parameters
    ----------
    categories : list[str] or None
        Brown corpus categories to use, e.g. ["news", "fiction"].
        None = use all 15 categories.
    test_ratio : float
        Fraction of sentences reserved for the test set (default 10 %).
    seed : int
        Random seed for reproducible splits.
    max_train_words : int or None
        If set, truncate the training text to approximately this many
        words.  Useful for faster Transformer training during
        development.

    Returns
    -------
    (train_text, test_text) – plain strings, one sentence per line.

    Available categories
    --------------------
    adventure, belles_lettres, editorial, fiction, government, hobbies,
    humor, learned, lore, mystery, news, religion, reviews, romance,
    science_fiction
    """
    _ensure_nltk_data()

    from nltk.corpus import brown

    # Load sentences (each is a list of word-tokens)
    if categories:
        sents = brown.sents(categories=categories)
    else:
        sents = brown.sents()

    # Convert to plain-text strings
    all_sentences: list[str] = []
    for sent_tokens in sents:
        # Brown corpus tokens include punctuation; join with spaces
        text = " ".join(sent_tokens)
        # Light cleanup: normalise whitespace around punctuation
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        all_sentences.append(text)

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(all_sentences)

    split_idx = max(1, int(len(all_sentences) * (1 - test_ratio)))
    train_sents = all_sentences[:split_idx]
    test_sents = all_sentences[split_idx:]

    train_text = "\n".join(train_sents)
    test_text = "\n".join(test_sents)

    # Optional truncation for faster dev cycles
    if max_train_words is not None:
        words = train_text.split()
        if len(words) > max_train_words:
            train_text = " ".join(words[:max_train_words])

    # Stats
    train_words = len(train_text.split())
    test_words = len(test_text.split())
    print(f"  Brown corpus loaded:")
    print(f"    Categories : {categories or 'all'}")
    print(f"    Train      : {len(train_sents):,} sentences, {train_words:,} words")
    print(f"    Test       : {len(test_sents):,} sentences, {test_words:,} words")

    return train_text, test_text


# -------------------------------------------------------------------- #
#  Quick self-test                                                      #
# -------------------------------------------------------------------- #

if __name__ == "__main__":
    train, test = load_corpus(categories=None, max_train_words=200_000)
    print(f"\n  First 200 chars of train:\n    {train[:200]}…")
    print(f"\n  First 200 chars of test:\n    {test[:200]}…")
