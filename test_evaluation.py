"""
Comparative evaluation of word-prediction models.

Runs the keystroke-saving simulation across:
  1. N-Gram (trigram)
  2. N-Gram + Spell Correction
  3. Small Transformer (BPE + causal LM)
  4. Small Transformer + Spell Correction

On two test sets:
  A. Clean text (no typos)          - measures pure prediction quality
  B. Text with simulated typos      - measures spell-correction benefit

Plus ablation studies:
  C. Effect of top-k on keystroke savings
  D. N-gram order comparison (n = 2, 3, 4)
  E. BPE vocabulary size comparison (500, 1000, 2000)
  F. Training data volume effect (30K, 60K, 100K words)

Run:  python test_evaluation.py
"""

import sys, time

# Fix Windows GBK encoding for Unicode symbols (Y N etc.)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from ngram_predictor import NGramPredictor
from transformer_predictor import SmallTransformerPredictor
from spell_corrector import SpellCorrector
from evaluator import Evaluator, TypoSimulator
from corpus_loader import load_corpus


# ===================================================================== #
#  Helpers                                                               #
# ===================================================================== #

def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    sys.stdout.flush()


def elapsed(t0: float) -> str:
    """Format seconds since t0 as mm:ss."""
    s = time.time() - t0
    return f"{int(s)//60:02d}:{int(s)%60:02d}"


def truncate(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def train_transformer(corpus: str, vocab_size: int = 1000,
                      d_model: int = 64, n_heads: int = 4,
                      n_layers: int = 2, epochs: int = 40,
                      label: str = "") -> SmallTransformerPredictor:
    """Train a SmallTransformerPredictor and print summary."""
    t = SmallTransformerPredictor(
        vocab_size=vocab_size, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, max_seq_len=64,
    )
    t0 = time.time()
    losses = t.train(corpus, epochs=epochs, lr=3e-3,
                     batch_size=16, seq_len=32, log_every=10)
    train_secs = time.time() - t0
    tag = f" [{label}]" if label else ""
    print(f"  {t}{tag}  (trained in {train_secs:.1f}s, "
          f"final loss {losses[-1]:.4f})")
    sys.stdout.flush()
    return t


# ===================================================================== #
#  Load Brown corpus with proper train/test split                        #
# ===================================================================== #

TRAIN_CORPUS, TEST_CLEAN = load_corpus(
    categories=["news", "editorial"],   # two genres for richer vocab
    test_ratio=0.1,                     # 10 % held out for evaluation
    max_train_words=100_000,            # more data for better fit
)

# Fixed subsets for fair cross-model comparison
TRANSFORMER_EVAL_WORDS = 500
TEST_SHARED = truncate(TEST_CLEAN, TRANSFORMER_EVAL_WORDS)
TEST_FULL   = TEST_CLEAN


def main() -> None:
    t_global = time.time()

    # ================================================================= #
    #  1. Train baseline models on the full corpus                       #
    # ================================================================= #
    section("Training Baseline Models")

    ngram = NGramPredictor(n=3)
    ngram.train(TRAIN_CORPUS)
    print(f"  {ngram}  [{elapsed(t_global)}]")

    spell = SpellCorrector(max_edit_distance=2)
    spell.train_from_counter(ngram.word_freq, ngram.vocabulary)
    print(f"  {spell}  [{elapsed(t_global)}]")

    print(f"  Training Transformer (this may take a few minutes)...")
    sys.stdout.flush()
    transformer = train_transformer(TRAIN_CORPUS, label="baseline")
    print(f"  >> All baseline models trained [{elapsed(t_global)}]")

    evaluator = Evaluator(top_k=5)
    typo_sim = TypoSimulator(typo_rate=0.15, seed=42)

    # ================================================================= #
    #  2. Evaluation A - Clean text (no typos)                           #
    # ================================================================= #
    section("Evaluation A: Clean Text (no typos)")

    results_clean = [
        evaluator.evaluate(ngram, TEST_FULL,
                           model_name="N-Gram (trigram)"),
        evaluator.evaluate(ngram, TEST_FULL,
                           model_name="N-Gram + SpellCorrector",
                           spell_corrector=spell),
        evaluator.evaluate(transformer, TEST_SHARED,
                           model_name="Transformer (small)"),
        evaluator.evaluate(transformer, TEST_SHARED,
                           model_name="Transformer + SpellCorrector",
                           spell_corrector=spell),
    ]

    Evaluator.print_table(results_clean)
    Evaluator.print_detail(results_clean[0], max_rows=15)

    # ---- Fair head-to-head on same subset ---------------------------- #
    print("\n  -- Fair comparison on same "
          f"{TRANSFORMER_EVAL_WORDS}-word subset --")
    head_to_head = [
        evaluator.evaluate(ngram, TEST_SHARED,
                           model_name="N-Gram (trigram) [subset]"),
        evaluator.evaluate(ngram, TEST_SHARED,
                           model_name="N-Gram + Spell [subset]",
                           spell_corrector=spell),
        evaluator.evaluate(transformer, TEST_SHARED,
                           model_name="Transformer [subset]"),
        evaluator.evaluate(transformer, TEST_SHARED,
                           model_name="Transformer + Spell [subset]",
                           spell_corrector=spell),
    ]
    Evaluator.print_table(head_to_head)
    print(f"  >> Eval A done [{elapsed(t_global)}]")

    # ================================================================= #
    #  3. Evaluation B - Text with simulated typos                       #
    # ================================================================= #
    section("Evaluation B: Simulated Typos (rate=15%)")

    results_typo = [
        evaluator.evaluate(ngram, TEST_FULL,
                           model_name="N-Gram (trigram)",
                           typo_simulator=typo_sim),
        evaluator.evaluate(ngram, TEST_FULL,
                           model_name="N-Gram + SpellCorrector",
                           spell_corrector=spell,
                           typo_simulator=typo_sim),
        evaluator.evaluate(transformer, TEST_SHARED,
                           model_name="Transformer (small)",
                           typo_simulator=typo_sim),
        evaluator.evaluate(transformer, TEST_SHARED,
                           model_name="Transformer + SpellCorrector",
                           spell_corrector=spell,
                           typo_simulator=typo_sim),
    ]

    Evaluator.print_table(results_typo)
    Evaluator.print_detail(results_typo[1], max_rows=15)

    # ---- Fair head-to-head with typos -------------------------------- #
    print("\n  -- Fair comparison on same "
          f"{TRANSFORMER_EVAL_WORDS}-word subset (typos) --")
    head_to_head_typo = [
        evaluator.evaluate(ngram, TEST_SHARED,
                           model_name="N-Gram [subset]",
                           typo_simulator=typo_sim),
        evaluator.evaluate(ngram, TEST_SHARED,
                           model_name="N-Gram + Spell [subset]",
                           spell_corrector=spell,
                           typo_simulator=typo_sim),
        evaluator.evaluate(transformer, TEST_SHARED,
                           model_name="Transformer [subset]",
                           typo_simulator=typo_sim),
        evaluator.evaluate(transformer, TEST_SHARED,
                           model_name="Transformer + Spell [subset]",
                           spell_corrector=spell,
                           typo_simulator=typo_sim),
    ]
    Evaluator.print_table(head_to_head_typo)
    print(f"  >> Eval B done [{elapsed(t_global)}]")

    # ================================================================= #
    #  4. Evaluation C - Varying top-k                                   #
    # ================================================================= #
    section("Evaluation C: Effect of Top-k on Keystroke Savings")

    k_values = [1, 3, 5, 10]
    results_k: list = []

    for k in k_values:
        ev = Evaluator(top_k=k)
        r = ev.evaluate(ngram, TEST_FULL,
                        model_name=f"N-Gram  (k={k})")
        results_k.append(r)

    for k in [3, 5, 10]:
        ev = Evaluator(top_k=k)
        r = ev.evaluate(transformer, TEST_SHARED,
                        model_name=f"Transformer (k={k})")
        results_k.append(r)

    Evaluator.print_table(results_k)
    print(f"  >> Eval C done [{elapsed(t_global)}]")

    # ================================================================= #
    #  5. Experiment D - N-gram order comparison                         #
    # ================================================================= #
    section("Experiment D: N-gram Order (n = 2, 3, 4)")

    results_order: list = []
    for n in [2, 3, 4]:
        ng = NGramPredictor(n=n)
        ng.train(TRAIN_CORPUS)
        r = evaluator.evaluate(ng, TEST_FULL,
                               model_name=f"N-Gram (n={n})")
        results_order.append(r)
        print(f"  n={n}:  vocab={ng.vocabulary_size()}, "
              f"tokens={ng._total_tokens}")

    Evaluator.print_table(results_order)
    print(f"  >> Experiment D done [{elapsed(t_global)}]")

    # ================================================================= #
    #  6. Experiment E - BPE vocabulary size comparison                   #
    # ================================================================= #
    section("Experiment E: BPE Vocabulary Size (500, 1000, 2000)")

    results_bpe: list = []
    for v in [500, 1000, 2000]:
        t = train_transformer(TRAIN_CORPUS, vocab_size=v,
                              label=f"vocab={v}")
        r = evaluator.evaluate(t, TEST_SHARED,
                               model_name=f"Transformer (V={v})")
        results_bpe.append(r)

    Evaluator.print_table(results_bpe)
    print(f"  >> Experiment E done [{elapsed(t_global)}]")

    # ================================================================= #
    #  7. Experiment F - Training data volume effect                     #
    # ================================================================= #
    section("Experiment F: Training Data Volume")

    results_volume: list = []
    volume_corpus_full, _ = load_corpus(
        categories=["news", "editorial"],
        test_ratio=0.1,
        max_train_words=None,        # get the maximum available
    )

    for size in [30_000, 60_000, 100_000]:
        sub_corpus = truncate(volume_corpus_full, size)
        actual = len(sub_corpus.split())

        # N-Gram
        ng = NGramPredictor(n=3)
        ng.train(sub_corpus)
        r_ng = evaluator.evaluate(ng, TEST_FULL,
                                  model_name=f"N-Gram ({size//1000}K words)")
        results_volume.append(r_ng)

        # Transformer
        t = train_transformer(sub_corpus, label=f"{size//1000}K words")
        r_tf = evaluator.evaluate(t, TEST_SHARED,
                                  model_name=f"Transformer ({size//1000}K words)")
        results_volume.append(r_tf)

    Evaluator.print_table(results_volume)
    print(f"  >> Experiment F done [{elapsed(t_global)}]")

    # ================================================================= #
    #  8. Summary                                                        #
    # ================================================================= #
    section("Summary")

    best_clean = max(results_clean, key=lambda r: r.proportion_saved)
    best_typo = max(results_typo, key=lambda r: r.proportion_saved)

    print(f"  Best model on clean text:  {best_clean.model_name}")
    print(f"    => {best_clean.proportion_saved:.1%} keystrokes saved "
          f"({best_clean.saved_keystrokes}/{best_clean.keystrokes_without})")
    print()
    print(f"  Best model with typos:     {best_typo.model_name}")
    print(f"    => {best_typo.proportion_saved:.1%} keystrokes saved "
          f"({best_typo.saved_keystrokes}/{best_typo.keystrokes_without})")

    # Spell correction benefit
    ng_no_spell_typo = results_typo[0].proportion_saved
    ng_spell_typo = results_typo[1].proportion_saved
    spell_lift = ng_spell_typo - ng_no_spell_typo
    print(f"\n  Key observations:")
    print(f"    * Spell correction improves N-Gram by "
          f"{spell_lift:+.1%} under typo conditions")

    # N-Gram vs Transformer
    ng_clean = results_clean[0].proportion_saved
    tf_clean = results_clean[2].proportion_saved
    print(f"    * N-Gram saves {ng_clean:.1%} vs "
          f"Transformer saves {tf_clean:.1%} on clean text")

    # Top-k
    k1 = next(r for r in results_k if r.model_name == "N-Gram  (k=1)")
    k10 = next(r for r in results_k if r.model_name == "N-Gram  (k=10)")
    print(f"    * Increasing k from 1=>10 improves N-Gram from "
          f"{k1.proportion_saved:.1%} to {k10.proportion_saved:.1%}")

    # N-gram order
    best_order = max(results_order, key=lambda r: r.proportion_saved)
    print(f"    * Best N-gram order: {best_order.model_name} "
          f"({best_order.proportion_saved:.1%})")

    # BPE vocab
    best_bpe = max(results_bpe, key=lambda r: r.proportion_saved)
    print(f"    * Best BPE vocab size: {best_bpe.model_name} "
          f"({best_bpe.proportion_saved:.1%})")

    # Data volume
    vol_ngrams = [r for r in results_volume if "N-Gram" in r.model_name]
    vol_trans = [r for r in results_volume if "Transformer" in r.model_name]
    if vol_ngrams:
        print(f"    * N-Gram KS% by data volume: "
              + ", ".join(f"{r.model_name}: {r.proportion_saved:.1%}"
                         for r in vol_ngrams))
    if vol_trans:
        print(f"    * Transformer KS% by data volume: "
              + ", ".join(f"{r.model_name}: {r.proportion_saved:.1%}"
                         for r in vol_trans))

    print(f"\n  Total time: {elapsed(t_global)}")


if __name__ == "__main__":
    main()