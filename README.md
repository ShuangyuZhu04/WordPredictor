# Word Predictor

A real-time word prediction system that suggests the next whole word as you type, powered by two language models and an optional spell corrector. Trained on the Brown corpus (~1.15 million words across 15 genres), the system implements the full pipeline from data loading through model training and evaluation to an interactive GUI demo.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the interactive demo
python app.py
# -> first run downloads Brown corpus (~3 MB), then opens http://localhost:7860

# 3. Run the evaluation suite
python test_evaluation.py
```

No pre-trained models or external datasets need to be downloaded manually. The Brown corpus is fetched automatically by NLTK on first run and cached locally at `~/nltk_data/`.

---

## Features

- **Two language models** with a shared prediction interface:
  - **N-Gram (trigram)** -- frequency tables with backoff from trigram to bigram to unigram. Sub-millisecond inference.
  - **Transformer (causal LM)** -- BPE subword tokenizer and a GPT-style decoder trained from scratch. Uses a dual-path prediction strategy to aggregate subword token probabilities back into whole-word suggestions.
- **Spell correction** -- Damerau-Levenshtein edit distance combined with corpus frequency ranking. Recovers from typos like `goverment -> government` and `teh -> the`.
- **Real-time GUI** -- Gradio interface with keystroke-by-keystroke suggestion updates, clickable word pills, and live model switching.
- **Rigorous evaluation** -- keystroke-saving simulation on held-out Brown corpus data, comparing clean text, noisy (typo) text, varying top-k values, and three ablation studies (n-gram order, BPE vocabulary size, training data volume).
- **Centralised data pipeline** -- a single `corpus_loader.py` handles NLTK download, preprocessing, and reproducible train/test splitting, shared by all scripts.

---

## Project Structure

```
word-predictor/
|-- corpus_loader.py            # Brown corpus loading, preprocessing, train/test split
|-- ngram_predictor.py          # N-Gram language model (trigram with backoff)
|-- transformer_predictor.py    # BPE tokenizer + causal Transformer LM
|-- spell_corrector.py          # Edit-distance spell corrector
|-- evaluator.py                # Keystroke-saving evaluation framework
|-- app.py                      # Gradio GUI
|-- test_pipeline.py            # Demo: N-Gram + spell correction
|-- test_transformer.py         # Demo: Transformer vs N-Gram comparison
|-- test_evaluation.py          # Full comparative evaluation (6 experiments)
`-- requirements.txt            # Python dependencies
```

---

## Data: The Brown Corpus

The Brown corpus is a classic NLP benchmark containing approximately 1.15 million words sampled from 500 texts across 15 genres. It is distributed via NLTK and downloaded automatically on first run (~3 MB).

### Available Genres

| Category | Description | Category | Description |
|---|---|---|---|
| `news` | Press reportage | `government` | Government documents |
| `editorial` | Editorials | `learned` | Academic prose |
| `fiction` | General fiction | `science_fiction` | Science fiction |
| `mystery` | Mystery fiction | `adventure` | Adventure fiction |
| `romance` | Romance fiction | `humor` | Humour |
| `lore` | Popular lore | `hobbies` | Hobbies |
| `belles_lettres` | Literary criticism | `reviews` | Reviews |
| `religion` | Religious texts | | |

### How the Data is Split

`corpus_loader.py` shuffles the corpus sentences with a fixed random seed and splits them by a configurable ratio (default 90/10). This ensures a reproducible, non-overlapping train/test partition. Each script controls which genres and how many words to use:

| Script | Genres | Train Words | Purpose |
|---|---|---|---|
| `app.py` | news | 30,000 | Fast GUI startup |
| `test_pipeline.py` | news | 20,000 | Quick demo |
| `test_transformer.py` | news | 20,000 | Model comparison |
| `test_evaluation.py` | news + editorial | 100,000 | Full evaluation |

---

## Architecture

### Shared Prediction Interface

All models expose the same method signature, making them interchangeable throughout the GUI and evaluation code:

```python
def predict(
    context: list[str],   # preceding confirmed words
    prefix: str,          # partially typed current word
    top_k: int,           # number of suggestions to return
) -> list[tuple[str, float]]:
    ...
```

### N-Gram Model (`NGramPredictor`)

Builds n-gram frequency tables (up to trigram by default) during training. At prediction time it tries the highest-order model first, backing off to lower orders when no matches are found. The vocabulary is filtered on the current prefix at each step, so suggestions update with every keystroke.

### Transformer Model (`SmallTransformerPredictor`)

Two-phase training:

1. **BPE tokenizer** -- trained from scratch on the corpus using HuggingFace `tokenizers`. This satisfies the subword tokenization requirement.
2. **Causal Transformer** -- a GPT-style decoder trained with a standard cross-entropy language modelling objective.

The critical challenge is mapping subword token probabilities back to whole-word suggestions. This is handled with a **dual-path prediction strategy**:

- **Path A (context-only)**: the model receives only the context words (without the current prefix), predicts the next token, and filters the results by prefix match. This works well when the prefix is short or empty, avoiding the problem of BPE misinterpreting a partial word.
- **Path B (context + prefix)**: when the prefix is 2+ characters, it is appended to the context so the model can use the partial word to predict its continuation. This is more effective for longer prefixes.
- **Word assembly**: within each path, top-scoring BPE tokens that decode to known corpus words are accepted directly (fast path). For subword fragments, a greedy extension of up to 4 decoding steps attempts to build a complete word. Results from both paths are merged, de-duplicated, and ranked by probability.

All suggestions are validated against the corpus vocabulary, guaranteeing that only whole words are returned.

#### Model Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `vocab_size` | 1,000 | BPE tokens balanced against corpus size to avoid underfitting |
| `d_model` | 64 | Small hidden dimension to match the ~100K token training set |
| `n_heads` | 4 | Standard multi-head attention |
| `n_layers` | 2 | Lightweight depth suited to data volume |
| `max_seq_len` | 64 | Context window |
| `epochs` | 40 | Smaller model needs more passes to converge |

A `GPT2Predictor` class is also provided as a drop-in replacement for production use with pre-trained GPT-2 weights (requires HuggingFace Hub access).

### Spell Corrector (`SpellCorrector`)

Generates edit-distance-1 and edit-distance-2 candidates using Norvig-style edits (deletions, insertions, substitutions, transpositions), filters against the known vocabulary, and scores by `frequency / (distance + 1)^2`. Uses full Damerau-Levenshtein distance to handle transpositions like `teh -> the`.

### GUI (`app.py`)

Built with Gradio. On every keystroke the `input` event fires, the current text is parsed into `(context, prefix)`, and the selected model is queried. Five suggestion buttons update their labels in real-time. Clicking a suggestion splices the word into the textbox (replacing the current prefix), appends a trailing space, and immediately triggers a fresh prediction round. A status bar shows the parsed context, prefix, hit count, and inference latency.

---

## Evaluation

### Metric Definition

The proportion of saved keystrokes is computed by simulating a user typing character-by-character:

| Scenario | Keystrokes per Word |
|---|---|
| Without prediction | `len(word) + 1` (all characters + space) |
| With prediction (word found after typing `c` characters) | `c + 2` (typed characters + click + space) |
| With prediction (word never found) | `len(word) + 1` (same as without) |

The simulation checks the model's top-k suggestions after every prefix length from 0 (no characters typed) through the full word, and "clicks" the correct suggestion as soon as it appears.

### Evaluation Dimensions

`test_evaluation.py` runs six evaluation experiments:

1. **Evaluation A -- Clean text**: held-out Brown sentences (10% of corpus, never seen during training), measuring pure prediction quality. Includes a fair head-to-head comparison on the same 500-word subset for both models.
2. **Evaluation B -- Simulated typos** (15% character-level error rate): measures the benefit of spell correction under realistic noisy input. Typos are generated using QWERTY-aware character mutations (adjacent-key substitutions, deletions, insertions, transpositions).
3. **Evaluation C -- Top-k sweep** (k = 1, 3, 5, 10): measures how suggestion list size affects keystroke savings.
4. **Experiment D -- N-gram order** (n = 2, 3, 4): determines the optimal context length for the N-Gram model.
5. **Experiment E -- BPE vocabulary size** (V = 500, 1000, 2000): explores the trade-off between subword granularity and word-level prediction quality.
6. **Experiment F -- Training data volume** (30K, 60K, 100K words): quantifies the marginal return of additional training data for both models.

### Running the Evaluation

```bash
python test_evaluation.py
```

Total runtime is approximately 2.5 hours on a single GPU (NVIDIA RTX-class) or longer on CPU. The script prints six comparative tables, per-word detail breakdowns, and a summary with key observations.

---

### Results

Evaluation on held-out Brown corpus data (news + editorial genres; 14,602 words for N-Gram, 493-word subset for Transformer).

#### Evaluation A -- Clean Text (no typos, top-k = 5)

| Model | Words | Hit Rate | KS Saved |
|---|---|---|---|
| N-Gram (trigram) | 14,602 | 90.3% | 27.9% |
| N-Gram + Spell Correction | 14,602 | 91.0% | 28.3% |
| Transformer (small) | 493 | 54.6% | 12.5% |
| Transformer + Spell Correction | 493 | 81.9% | 14.9% |

Fair head-to-head on the same 493-word subset:

| Model | Hit Rate | KS Saved |
|---|---|---|
| N-Gram (trigram) | 91.9% | 29.7% |
| N-Gram + Spell Correction | 92.1% | 30.1% |
| Transformer (small) | 54.6% | 12.5% |
| Transformer + Spell Correction | 81.9% | 14.9% |

#### Evaluation B -- Simulated Typos (15% error rate, top-k = 5)

| Model | Words | Hit Rate | KS Saved |
|---|---|---|---|
| N-Gram (trigram) | 14,602 | 71.3% | 21.4% |
| N-Gram + Spell Correction | 14,602 | 86.9% | 22.0% |
| Transformer (small) | 493 | 50.7% | 11.8% |
| Transformer + Spell Correction | 493 | 83.8% | 8.8% |

#### Evaluation C -- Effect of Top-k (clean text)

| Model | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| N-Gram KS Saved | 16.1% | 25.1% | 27.9% | 30.8% |
| Transformer KS Saved | -- | 10.7% | 12.5% | 14.8% |

#### Experiment D -- N-gram Order (clean text, top-k = 5)

| N-gram Order | Hit Rate | KS Saved |
|---|---|---|
| Bigram (n=2) | 90.7% | **29.1%** |
| Trigram (n=3) | 90.3% | 27.9% |
| 4-gram (n=4) | 90.2% | 27.8% |

#### Experiment E -- BPE Vocabulary Size (493-word subset, top-k = 5)

| BPE Vocab | Training Loss | Hit Rate | KS Saved |
|---|---|---|---|
| V=500 | 3.69 | 52.7% | 11.2% |
| V=1000 | 4.13 | 55.8% | 13.3% |
| V=2000 | 4.47 | 55.0% | **15.0%** |

#### Experiment F -- Training Data Volume (top-k = 5)

| Data Volume | N-Gram KS Saved | Transformer KS Saved |
|---|---|---|
| 30K words | 24.2% | 10.6% |
| 60K words | 26.6% | 12.8% |
| 100K words | 27.9% | 12.8% |

---

### Discussion

**N-Gram vs Transformer.** On the same 493-word subset, the N-Gram saves 29.7% of keystrokes compared to the Transformer's 12.5%. The N-Gram dominates because it directly memorises word-level co-occurrences, while the Transformer must decode subword tokens back into whole words -- a lossy process when the BPE vocabulary (1,000 tokens) is much smaller than the corpus vocabulary (13,296 words). Despite the dual-path prediction strategy improving the Transformer's KS% from 5.3% (original single-path) to 12.5%, the fundamental bottleneck remains the subword-to-word conversion. With a larger corpus or pre-trained GPT-2 weights, the Transformer's generalisation from subword representations would be expected to close this gap.

**Spell correction: robustness vs efficiency.** On N-Gram, the corrector boosts typo-condition hit rate by 15.6 percentage points (71.3% to 86.9%) while also improving keystroke savings (+0.6 pp). On the Transformer, it raises hit rate dramatically (50.7% to 83.8%) but *decreases* keystroke savings from 11.8% to 8.8%. The reason is that spell-corrected suggestions require more prefix characters before a match is found (average prefix length increases from 0.85 to 2.79), consuming the keystrokes that would otherwise be saved. This reveals that spell correction's primary value is **robustness** (coverage under noisy input), not efficiency.

**N-gram order: bigram wins.** The bigram model (29.1% KS saved) outperforms both the trigram (27.9%) and the 4-gram (27.8%). This is a counter-intuitive result: with only 100K training words, higher-order n-grams suffer from data sparsity. The trigram and 4-gram models frequently back off to lower-order estimates anyway, but their conditional distributions are more fragmented, leading to less concentrated top-k suggestions. The bigram avoids this sparsity issue entirely, producing more focused predictions. On a larger corpus, we would expect the trigram to overtake the bigram as its frequency tables become better populated.

**BPE vocabulary size: larger is better despite higher loss.** The V=2000 model achieves the best keystroke savings (15.0%) despite having the highest training loss (4.47 vs 4.13 for V=1000). This is because a larger BPE vocabulary maps more common words to single tokens, increasing the proportion of predictions that take the fast path (direct token-to-word match) and reducing reliance on the lossy greedy extension. The higher cross-entropy loss reflects the harder classification problem (predicting over 2,000 classes vs 1,000), not worse language modelling quality.

**Training data volume: N-Gram scales linearly, Transformer saturates.** N-Gram keystroke savings increase steadily with data volume (24.2% at 30K to 27.9% at 100K), reflecting the direct relationship between frequency table coverage and prediction quality. The Transformer, by contrast, plateaus at 60K words (12.8%), with no improvement from 60K to 100K. This indicates that the 201K-parameter model has reached its capacity limit: it cannot absorb more data without increasing model size. A larger model (more layers or wider embeddings) would be needed to benefit from additional training data.

**Top-k diminishing returns.** The largest marginal gain is k=1 to k=3 (+9.0 pp savings); k=5 to k=10 adds only +2.9 pp while doubling the screen space used. k=5 is the practical optimum, balancing keystroke savings against UI clutter.

---

## Usage Examples

### Loading the Corpus

```python
from corpus_loader import load_corpus

# Use specific genres
train, test = load_corpus(categories=["news", "fiction"])

# Use all genres, cap training size
train, test = load_corpus(max_train_words=100_000)

# Use all genres, full corpus
train, test = load_corpus()
```

### N-Gram Prediction

```python
from ngram_predictor import NGramPredictor
from corpus_loader import load_corpus

train, _ = load_corpus(categories=["news"])

model = NGramPredictor(n=3)
model.train(train)

model.predict(["the", "united"], prefix="s", top_k=3)
# -> [('states', 0.82), ('steel', 0.10), ...]
```

### Spell Correction

```python
from spell_corrector import SpellCorrector

spell = SpellCorrector(max_edit_distance=2)
spell.train(train)

spell.correct("goverment", top_k=3)
# -> [('government', 1, 12.5), ('movement', 2, 0.8), ...]
```

### Transformer Prediction

```python
from transformer_predictor import SmallTransformerPredictor

model = SmallTransformerPredictor(
    vocab_size=1000, d_model=64, n_heads=4, n_layers=2
)
model.train(train, epochs=40)

model.predict(["the", "united"], prefix="s", top_k=3)
# -> [('states', 0.91), ('steel', 0.05), ('senate', 0.04)]

model.save("./checkpoint")
model.load("./checkpoint")
```

### Evaluation

```python
from evaluator import Evaluator

ev = Evaluator(top_k=5)
result = ev.evaluate(model, test, model_name="N-Gram")

print(f"Keystrokes saved: {result.proportion_saved:.1%}")
print(f"Hit rate: {result.hit_rate:.1%}")
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | >= 2.0 | Deep learning framework for the Transformer |
| `transformers` | >= 4.30 | Model architecture utilities |
| `tokenizers` | >= 0.15 | BPE subword tokenizer training |
| `gradio` | >= 4.0 | Interactive GUI |
| `nltk` | >= 3.8 | Brown corpus access |

Python 3.10+ required. Install with:

```bash
pip install -r requirements.txt
```

A GPU is optional but recommended. Training and evaluation complete in approximately 2.5 hours with a GPU, or longer on CPU.

---

## Extending the Project

### Use More Brown Genres

```python
# Train on all 15 genres for maximum vocabulary coverage
train, test = load_corpus(categories=None, max_train_words=200_000)
```

### Use a Custom Corpus

```python
# Any plain text works -- just pass a string to .train()
with open("my_corpus.txt") as f:
    corpus = f.read()

model = NGramPredictor(n=3)
model.train(corpus)
```

### Swap in Pre-trained GPT-2

When HuggingFace Hub is reachable, replace `SmallTransformerPredictor` with `GPT2Predictor` for production-quality predictions:

```python
from transformer_predictor import GPT2Predictor

gpt2 = GPT2Predictor(model_name="gpt2")
gpt2.load()
gpt2.predict(["the"], prefix="g", top_k=5)
```

### Adjust Typo Simulation

```python
from evaluator import TypoSimulator

# Higher error rate for stress testing
typo = TypoSimulator(typo_rate=0.25, seed=123)
result = evaluator.evaluate(model, test, typo_simulator=typo)
```
