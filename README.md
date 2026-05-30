# Word Predictor

A word prediction system that suggests the next word while you type.
It includes a trigram n-gram model, a frequency-aware spell corrector,
and a small Transformer language model trained from scratch. The models
are compared with a keystroke-saving simulation on the Brown corpus.

## Files

- `ngram_predictor.py` - trigram model with weighted back-off and prefix filtering
- `spell_corrector.py` - Norvig-style edit candidates + Damerau-Levenshtein distance
- `transformer_predictor.py` - small causal Transformer + BPE tokenizer (trained from scratch)
- `corpus_loader.py` - loads and splits the Brown corpus (via NLTK)
- `evaluator.py` - keystroke-saving simulation and typo simulator
- `test_evaluation.py` - full comparative evaluation (Experiments A to F)
- `app.py` - interactive Gradio GUI

## Requirements

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

The main dependencies are `torch`, `tokenizers`, `nltk` and `gradio`.
The Brown corpus and the Punkt tokenizer are downloaded automatically by
NLTK on first run, so an internet connection is needed the first time.

## How to run

### 1. Reproduce the evaluation (tables in the report)

```bash
python test_evaluation.py
```

This trains the models and runs all experiments. It prints the result
tables to the terminal. The full run takes roughly 25 minutes on a CPU,
mostly because of training the Transformer several times. Output is also
saved in `evaluation_results_updated.txt`.

### 2. Run the interactive demo

```bash
python app.py
```

Then open http://localhost:7860 in a browser. Pick a model, start typing,
and click a suggestion to insert it. The trained Transformer is cached in
`./transformer_checkpoint` so later launches are faster.

## Notes

- The n-gram model and spell corrector are evaluated on the full test set
  (~14.6k words). The Transformer is slower at prediction, so it is
  evaluated on a fixed 500-word subset; a separate "subset" comparison
  in the report runs all models on the same 500 words for fairness.
- Spell correction is only applied to prefixes of length >= 3.
- Main settings: Brown corpus (news + editorial), 100k training words,
  top-k = 5, typo rate 15%.

## Data

The Brown corpus is downloaded through NLTK and is not included here.
The only data file provided is `evaluation_results_updated.txt`, which is
the raw output of `test_evaluation.py`.