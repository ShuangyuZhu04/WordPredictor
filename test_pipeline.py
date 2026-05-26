"""
Demo & test script for the N-Gram Predictor + Spell Corrector pipeline.

Run:  python test_pipeline.py
"""

from ngram_predictor import NGramPredictor
from spell_corrector import SpellCorrector
from corpus_loader import load_corpus

# -------------------------------------------------------------------- #
#  Load Brown corpus (downloads automatically on first run)             #
# -------------------------------------------------------------------- #
CORPUS, _ = load_corpus(
    categories=["news"],       # single category keeps this demo fast
    max_train_words=20_000,    # cap size for quick testing
)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main() -> None:
    # ---- 1. Train both models on the same corpus -------------------- #
    ngram = NGramPredictor(n=3)
    ngram.train(CORPUS)

    spell = SpellCorrector(max_edit_distance=2)
    spell.train_from_counter(ngram.word_freq, ngram.vocabulary)

    print(f"Trained: {ngram}")
    print(f"Trained: {spell}")

    # ---- 2. N-gram prediction --------------------------------------- #
    section("N-Gram Prediction")

    tests = [
        # (context words, prefix being typed, description)
        ([], "p", "No context, prefix 'p'"),
        (["the", "united"], "s", "Context='the united', prefix='s'"),
        (["the"], "g", "Context='the', prefix='g'"),
        (["said", "the"], "", "Context='said the', no prefix"),
        (["new", "york"], "c", "Context='new york', prefix='c'"),
    ]

    for ctx, prefix, desc in tests:
        preds = ngram.predict(ctx, prefix=prefix, top_k=5)
        print(f"\n  {desc}")
        print(f"    Context: {ctx}  |  Prefix: '{prefix}'")
        for word, prob in preds:
            print(f"      {word:20s}  p={prob:.4f}")

    # ---- 3. Keystroke-by-keystroke simulation ------------------------ #
    section("Keystroke Simulation: typing 'gov'")

    ctx = ["the", "new"]
    for i in range(1, 4):
        partial = "gov"[:i]
        preds = ngram.predict(ctx, prefix=partial, top_k=3)
        words = [w for w, _ in preds]
        print(f"  Typed so far: '{partial}'  →  suggestions: {words}")

    # ---- 4. Spell correction ---------------------------------------- #
    section("Spell Correction")

    misspellings = ["goverment", "presiden", "committe", "politcal", "teh"]
    for typo in misspellings:
        corrections = spell.correct(typo, top_k=3)
        print(f"\n  Typo: '{typo}'")
        for word, dist, score in corrections:
            print(f"      {word:20s}  dist={dist}  score={score:.4f}")

    # ---- 5. Combined pipeline: correct → predict -------------------- #
    section("Combined Pipeline")

    user_input = "the goverment"          # 'goverment' is misspelled
    tokens = user_input.lower().split()
    last_token = tokens[-1]
    context = tokens[:-1]

    print(f"  User typed: '{user_input}'")
    print(f"  Last token: '{last_token}'  |  Context: {context}")

    if not spell.is_known(last_token):
        corrections = spell.correct(last_token, top_k=3)
        print(f"\n  Spell corrections for '{last_token}':")
        for word, dist, score in corrections:
            print(f"      {word:20s}  dist={dist}  score={score:.4f}")

        # Use the best correction as context and predict next word
        if corrections:
            best = corrections[0][0]
            print(f"\n  Best correction: '{best}' → predicting next word …")
            preds = ngram.predict(context + [best], prefix="", top_k=5)
            for word, prob in preds:
                print(f"      {word:20s}  p={prob:.4f}")
    else:
        # Word is known — just predict next
        preds = ngram.predict(tokens, prefix="", top_k=5)
        print(f"\n  Next-word predictions:")
        for word, prob in preds:
            print(f"      {word:20s}  p={prob:.4f}")


if __name__ == "__main__":
    main()
