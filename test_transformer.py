"""
Test & comparison script for the Transformer predictor.

Trains a SmallTransformerPredictor on the Brown corpus, then compares
both models' predictions side by side.

Run:  python test_transformer.py
"""

import time
from ngram_predictor import NGramPredictor
from transformer_predictor import SmallTransformerPredictor
from corpus_loader import load_corpus

# -------------------------------------------------------------------- #
#  Load Brown corpus                                                    #
# -------------------------------------------------------------------- #
CORPUS, _ = load_corpus(
    categories=["news"],
    max_train_words=20_000,    # keep training fast for this demo
)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def compare_predictions(
    ngram: NGramPredictor,
    transformer: SmallTransformerPredictor,
    context: list[str],
    prefix: str,
    label: str,
) -> None:
    """Print N-gram vs Transformer predictions side by side."""
    print(f"\n  {label}")
    print(f"  Context: {context}  |  Prefix: '{prefix}'")

    ng_preds = ngram.predict(context, prefix=prefix, top_k=5)
    tf_preds = transformer.predict(context, prefix=prefix, top_k=5)

    print(f"  {'N-Gram':>30s}    {'Transformer':>30s}")
    print(f"  {'─' * 30}    {'─' * 30}")
    max_rows = max(len(ng_preds), len(tf_preds))
    for i in range(max_rows):
        ng_str = (
            f"{ng_preds[i][0]:>20s} p={ng_preds[i][1]:.4f}"
            if i < len(ng_preds) else ""
        )
        tf_str = (
            f"{tf_preds[i][0]:>20s} p={tf_preds[i][1]:.4f}"
            if i < len(tf_preds) else ""
        )
        print(f"  {ng_str:>30s}    {tf_str:>30s}")


def main() -> None:
    # ---- 1. Train N-Gram -------------------------------------------- #
    section("Training N-Gram Model")
    ngram = NGramPredictor(n=3)
    ngram.train(CORPUS)
    print(f"  {ngram}")

    # ---- 2. Train Transformer --------------------------------------- #
    section("Training Small Transformer (BPE + Causal LM)")
    transformer = SmallTransformerPredictor(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=64,
    )
    t0 = time.time()
    losses = transformer.train(
        CORPUS,
        epochs=60,
        lr=3e-3,
        batch_size=16,
        seq_len=32,
        log_every=15,
    )
    elapsed = time.time() - t0
    print(f"\n  {transformer}")
    print(f"  Trained in {elapsed:.1f}s  |  Final loss: {losses[-1]:.4f}")

    # ---- 3. Side-by-side comparison --------------------------------- #
    section("Side-by-Side Predictions")

    tests = [
        ([], "p", "No context, prefix 'p'"),
        (["the", "united"], "s", "Context='the united', prefix='s'"),
        (["the"], "g", "Context='the', prefix='g'"),
        (["said", "the"], "", "Context='said the', no prefix"),
        (["in", "the"], "c", "Context='in the', prefix='c'"),
        (["new"], "y", "Context='new', prefix='y'"),
    ]

    for ctx, prefix, desc in tests:
        compare_predictions(ngram, transformer, ctx, prefix, desc)

    # ---- 4. Keystroke simulation ------------------------------------ #
    section("Keystroke Simulation (Transformer): typing 'gov'")
    ctx = ["the", "new"]
    for i in range(1, 4):
        partial = "gov"[:i]
        preds = transformer.predict(ctx, prefix=partial, top_k=3)
        words = [w for w, _ in preds]
        print(f"  Typed: '{partial}'  →  {words}")

    # ---- 5. Inference speed ----------------------------------------- #
    section("Inference Latency")
    import statistics

    times_ng, times_tf = [], []
    for _ in range(50):
        t0 = time.time()
        ngram.predict(["the"], prefix="q", top_k=5)
        times_ng.append(time.time() - t0)

        t0 = time.time()
        transformer.predict(["the"], prefix="q", top_k=5)
        times_tf.append(time.time() - t0)

    print(f"  N-Gram     median: {statistics.median(times_ng)*1000:.2f} ms")
    print(f"  Transformer median: {statistics.median(times_tf)*1000:.2f} ms")

    # ---- 6. Save model ---------------------------------------------- #
    section("Persistence")
    transformer.save("/tmp/transformer_checkpoint")
    transformer2 = SmallTransformerPredictor(
        vocab_size=1000, d_model=64, n_heads=4, n_layers=2, max_seq_len=64
    )
    transformer2.load("/tmp/transformer_checkpoint")
    preds = transformer2.predict(["the", "united"], prefix="s", top_k=3)
    print(f"  Loaded model predicts 'the united s…': {preds}")


if __name__ == "__main__":
    main()