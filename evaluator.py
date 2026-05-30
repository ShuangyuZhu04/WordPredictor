"""
Evaluation framework for word prediction models.

Simulates a user typing character-by-character, checking the model's
top-k suggestions after every keystroke, and "clicking" the correct
word as soon as it appears.

Metrics
-------
- keystrokes_without : total keystrokes with no prediction  = sum (len(word) + 1)
- keystrokes_with    : total keystrokes with prediction aid
      * word found after typing c chars  =>  c + 1 (click) + 1 (space) = c + 2
      * word never found                 =>  len(word) + 1  (typed in full)
- saved_keystrokes   : keystrokes_without - keystrokes_with
- proportion_saved   : saved_keystrokes / keystrokes_without
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Protocol

from spell_corrector import SpellCorrector

# ===================================================================== #
#  Predictor protocol (any model that exposes .predict() will work)      #
# ===================================================================== #


class PredictorLike(Protocol):
    """Structural type shared by NGramPredictor, SmallTransformerPredictor,
    and GPT2Predictor."""

    def predict(
        self,
        context: List[str],
        prefix: str,
        top_k: int,
    ) -> List[Tuple[str, float]]: ...


# ===================================================================== #
#  Per-word result                                                       #
# ===================================================================== #


@dataclass
class WordResult:
    """Evaluation result for a single word."""

    target: str
    keystrokes_without: int  # len(word) + 1
    keystrokes_with: int  # c + 2  or  len(word) + 1
    chars_typed: int  # how many chars the user typed before match
    found: bool  # was the word found in top-k?
    found_at_prefix_len: int  # prefix length when found (-1 if never)


# ===================================================================== #
#  Aggregate results                                                     #
# ===================================================================== #


@dataclass
class EvalResult:
    """Aggregate evaluation result for a full text."""

    model_name: str
    top_k: int
    total_words: int = 0
    words_found: int = 0
    keystrokes_without: int = 0
    keystrokes_with: int = 0
    word_results: List[WordResult] = field(default_factory=list)

    # ---- derived metrics -------------------------------------------- #
    @property
    def saved_keystrokes(self) -> int:
        return self.keystrokes_without - self.keystrokes_with

    @property
    def proportion_saved(self) -> float:
        if self.keystrokes_without == 0:
            return 0.0
        return self.saved_keystrokes / self.keystrokes_without

    @property
    def hit_rate(self) -> float:
        """Fraction of words successfully predicted before fully typed."""
        if self.total_words == 0:
            return 0.0
        return self.words_found / self.total_words

    @property
    def avg_prefix_len_when_found(self) -> float:
        """Average number of characters typed before match (over hits)."""
        hits = [wr for wr in self.word_results if wr.found]
        if not hits:
            return 0.0
        return sum(wr.found_at_prefix_len for wr in hits) / len(hits)


# ===================================================================== #
#  Typo simulator                                                        #
# ===================================================================== #


class TypoSimulator:
    """Introduce realistic typos into words for spell-correction evaluation.

    Each character position has an independent probability `p` of being
    corrupted.  The corruption type is chosen uniformly from:
      deletion, insertion, substitution, adjacent-key swap.
    """

    QWERTY_NEIGHBOURS: dict[str, str] = {
        "a": "sqwz",
        "b": "vghn",
        "c": "xdfv",
        "d": "sfcer",
        "e": "wrsdf",
        "f": "dgcvr",
        "g": "fhtbv",
        "h": "gjybn",
        "i": "ujko",
        "j": "hkunm",
        "k": "jloi",
        "l": "kop",
        "m": "njk",
        "n": "bhjm",
        "o": "iklp",
        "p": "ol",
        "q": "wa",
        "r": "edft",
        "s": "awedxz",
        "t": "rfgy",
        "u": "yhji",
        "v": "cfgb",
        "w": "qase",
        "x": "zsdc",
        "y": "tghu",
        "z": "asx",
    }

    def __init__(self, typo_rate: float = 0.15, seed: int = 42):
        self.typo_rate = typo_rate
        self.rng = random.Random(seed)

    def corrupt(self, word: str) -> str:
        """Return a possibly-corrupted version of *word*."""
        if len(word) <= 2:
            return word  # too short to safely corrupt

        chars = list(word.lower())
        result: list[str] = []

        for i, ch in enumerate(chars):
            if self.rng.random() >= self.typo_rate:
                result.append(ch)
                continue

            op = self.rng.choice(["delete", "insert", "substitute", "swap"])

            if op == "delete":
                pass  # skip this character
            elif op == "insert":
                extra = self.rng.choice("abcdefghijklmnopqrstuvwxyz")
                result.append(extra)
                result.append(ch)
            elif op == "substitute":
                neighbours = self.QWERTY_NEIGHBOURS.get(ch, "")
                if neighbours:
                    result.append(self.rng.choice(neighbours))
                else:
                    result.append(ch)
            elif op == "swap" and i + 1 < len(chars):
                result.append(chars[i + 1])
                chars[i + 1] = ch  # will be appended next iteration
            else:
                result.append(ch)

        corrupted = "".join(result)
        return corrupted if corrupted else word


# ===================================================================== #
#  Evaluator                                                             #
# ===================================================================== #


class Evaluator:
    """Simulate keystroke-by-keystroke typing and measure saved keystrokes.

    Parameters
    ----------
    top_k : int
        Number of suggestions the model shows to the "user" at each step.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z]+", text.lower())

    # ------------------------------------------------------------------ #
    #  Core simulation (one word)                                         #
    # ------------------------------------------------------------------ #
    def _simulate_word(
        self,
        predictor: PredictorLike,
        context: List[str],
        target: str,
        typed_word: Optional[str] = None,
        spell_corrector: Optional[SpellCorrector] = None,
    ) -> WordResult:
        """Simulate typing *typed_word* (defaults to *target*) and check
        when *target* appears in the model's top-k suggestions.

        Parameters
        ----------
        predictor        : the word-prediction model
        context          : preceding confirmed words
        target           : the correct word the user intends to type
        typed_word       : the (possibly misspelled) characters the user
                           actually types - defaults to *target*
        spell_corrector  : optional corrector that augments suggestions

        Returns
        -------
        WordResult with keystroke counts.
        """
        if typed_word is None:
            typed_word = target

        ks_without = len(target) + 1  # full word + space
        target_lower = target.lower()

        # Try prefix lengths 0, 1, 2, ..., len(typed_word)
        for c in range(len(typed_word) + 1):
            prefix = typed_word[:c].lower()

            # --- 1. Get base predictions from the model ---------------- #
            preds = predictor.predict(context, prefix=prefix, top_k=self.top_k)
            suggestion_words = {w.lower() for w, _ in preds}

            # --- 2. Augment with spell-correction suggestions ---------- #
            #   Skip short prefixes (< 3 chars): they generate massive
            #   candidate sets via edits2 and are never meaningful typos.
            if (
                spell_corrector
                and len(prefix) >= 3
                and not spell_corrector.is_known(prefix)
            ):
                corrections = spell_corrector.correct(prefix, top_k=self.top_k)
                for corrected_word, _dist, _score in corrections:
                    suggestion_words.add(corrected_word.lower())
                    # Also query model with corrected prefix
                    if corrected_word.lower() != prefix:
                        extra = predictor.predict(
                            context,
                            prefix=corrected_word.lower(),
                            top_k=self.top_k,
                        )
                        for w, _ in extra:
                            suggestion_words.add(w.lower())

            # --- 3. Check if target is among suggestions --------------- #
            if target_lower in suggestion_words:
                ks_with = c + 2  # c typed chars + 1 click + 1 space
                return WordResult(
                    target=target,
                    keystrokes_without=ks_without,
                    keystrokes_with=ks_with,
                    chars_typed=c,
                    found=True,
                    found_at_prefix_len=c,
                )

        # Word was never found - user types it in full
        return WordResult(
            target=target,
            keystrokes_without=ks_without,
            keystrokes_with=ks_without,
            chars_typed=len(typed_word),
            found=False,
            found_at_prefix_len=-1,
        )

    # ------------------------------------------------------------------ #
    #  Full-text evaluation                                               #
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        predictor: PredictorLike,
        test_text: str,
        model_name: str = "model",
        spell_corrector: Optional[SpellCorrector] = None,
        typo_simulator: Optional[TypoSimulator] = None,
        verbose: bool = True,
    ) -> EvalResult:
        """Run the full keystroke simulation on *test_text*.

        Parameters
        ----------
        predictor        : the word-prediction model
        test_text        : raw text to type
        model_name       : label for reporting
        spell_corrector  : optional corrector (augments suggestions)
        typo_simulator   : if provided, corrupts words before "typing"
        verbose          : if True, print progress every 200 words

        Returns
        -------
        EvalResult with aggregate metrics.
        """
        import time as _time

        words = self.tokenize(test_text)
        result = EvalResult(model_name=model_name, top_k=self.top_k)
        total = len(words)
        t0 = _time.time()

        for i, target in enumerate(words):
            context = words[max(0, i - 5) : i]  # up to 5 preceding words

            # Optionally corrupt the word to simulate typos
            typed = typo_simulator.corrupt(target) if typo_simulator else target

            wr = self._simulate_word(
                predictor,
                context,
                target,
                typed_word=typed,
                spell_corrector=spell_corrector,
            )

            result.total_words += 1
            result.keystrokes_without += wr.keystrokes_without
            result.keystrokes_with += wr.keystrokes_with
            if wr.found:
                result.words_found += 1
            result.word_results.append(wr)

            # Progress indicator
            if verbose and ((i + 1) % 200 == 0 or i + 1 == total):
                elapsed = _time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(
                    f"\r  [{model_name}] {i+1}/{total} words "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)  ",
                    end="",
                    flush=True,
                )

        if verbose and total > 0:
            print()  # newline after progress

        return result

    # ------------------------------------------------------------------ #
    #  Reporting                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def print_table(results: List[EvalResult]) -> None:
        """Print a comparative table of evaluation results."""
        header = (
            f"{'Model':<35s} | {'Top-k':>5s} | {'Words':>5s} | "
            f"{'Hits':>5s} | {'Hit %':>6s} | "
            f"{'KS w/o':>7s} | {'KS w/':>7s} | {'Saved':>7s} | "
            f"{'% Saved':>7s} | {'Avg pfx':>7s}"
        )
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for r in results:
            print(
                f"{r.model_name:<35s} | "
                f"{r.top_k:>5d} | "
                f"{r.total_words:>5d} | "
                f"{r.words_found:>5d} | "
                f"{r.hit_rate:>5.1%} | "
                f"{r.keystrokes_without:>7d} | "
                f"{r.keystrokes_with:>7d} | "
                f"{r.saved_keystrokes:>7d} | "
                f"{r.proportion_saved:>6.1%} | "
                f"{r.avg_prefix_len_when_found:>7.2f}"
            )
        print(sep)

    @staticmethod
    def print_detail(result: EvalResult, max_rows: int = 20) -> None:
        """Print per-word breakdown (first *max_rows* words)."""
        print(f"\n  Per-word detail for: {result.model_name}")
        print(
            f"  {'Word':<20s} {'Found':>6s} {'Prefix':>7s} "
            f"{'KS w/o':>7s} {'KS w/':>7s} {'Saved':>6s}"
        )
        print(f"  {'-' * 55}")
        for wr in result.word_results[:max_rows]:
            pfx = str(wr.found_at_prefix_len) if wr.found else "-"
            saved = wr.keystrokes_without - wr.keystrokes_with
            print(
                f"  {wr.target:<20s} {'Y' if wr.found else 'N':>6s} "
                f"{pfx:>7s} {wr.keystrokes_without:>7d} "
                f"{wr.keystrokes_with:>7d} {saved:>6d}"
            )
