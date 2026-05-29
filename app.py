"""
Word Predictor – Interactive GUI

A real-time word prediction demo built with Gradio.  Suggestions update
after every keystroke and can be clicked to insert the word.

Launch:
    python app.py

Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import os
import time
import re
from typing import List, Tuple

import gradio as gr

from ngram_predictor import NGramPredictor
from spell_corrector import SpellCorrector
from transformer_predictor import SmallTransformerPredictor
from corpus_loader import load_corpus

# ===================================================================== #
#  Load Brown corpus (downloads automatically on first run)              #
# ===================================================================== #

TRAIN_CORPUS, _ = load_corpus(
    categories=["news"],  # single genre for fast startup
    max_train_words=30_000,  # balance vocab richness vs speed
)

# Directory where the trained Transformer is cached between runs
TRANSFORMER_CHECKPOINT = "./transformer_checkpoint"


# ===================================================================== #
#  Initialise models (runs once at startup)                              #
# ===================================================================== #


def load_models():
    """Train and return all models + the spell corrector."""
    print("Training N-Gram model …")
    ngram = NGramPredictor(n=3)
    ngram.train(TRAIN_CORPUS)
    print(f"  {ngram}")

    print("Training spell corrector …")
    spell = SpellCorrector(max_edit_distance=2)
    spell.train_from_counter(ngram.word_freq, ngram.vocabulary)
    print(f"  {spell}")

    transformer = SmallTransformerPredictor(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=64,
    )
    if os.path.exists(os.path.join(TRANSFORMER_CHECKPOINT, "model.pt")):
        print("Loading saved Transformer model …")
        transformer.load(TRANSFORMER_CHECKPOINT)
    else:
        print("Training Transformer model …")
        transformer.train(
            TRAIN_CORPUS,
            epochs=40,
            lr=3e-3,
            batch_size=16,
            seq_len=32,
            log_every=20,
        )
        transformer.save(TRANSFORMER_CHECKPOINT)
    print(f"  {transformer}")

    return ngram, spell, transformer


NGRAM, SPELL, TRANSFORMER = load_models()

MODEL_CHOICES = [
    "N-Gram",
    "N-Gram + Spell Correction",
    "Transformer",
    "Transformer + Spell Correction",
]

TOP_K = 5


# ===================================================================== #
#  Core prediction logic                                                 #
# ===================================================================== #


def parse_input(text: str) -> Tuple[List[str], str]:
    """Split the input text into confirmed context words and a prefix.

    - If the text ends with a space → prefix is "" (user finished a word).
    - Otherwise → prefix is the last partial token.
    """
    if not text:
        return [], ""

    if text.endswith(" "):
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return words[-5:], ""
    else:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        if not words:
            return [], ""
        prefix = words[-1]
        context = words[-6:-1]
        return context, prefix


def get_predictions(text: str, model_name: str) -> List[Tuple[str, float]]:
    """Run the selected model and return predictions."""
    context, prefix = parse_input(text)

    use_spell = "Spell" in model_name
    use_transformer = "Transformer" in model_name
    predictor = TRANSFORMER if use_transformer else NGRAM

    # Base predictions
    preds = predictor.predict(context, prefix=prefix, top_k=TOP_K)
    suggestion_set = {w: p for w, p in preds}

    # Augment with spell-corrected predictions
    if use_spell and prefix and not SPELL.is_known(prefix):
        corrections = SPELL.correct(prefix, top_k=TOP_K)
        for corrected_word, _dist, _score in corrections:
            if corrected_word.lower() not in suggestion_set:
                suggestion_set[corrected_word.lower()] = _score / 100
            # Also query predictor with corrected prefix
            extra = predictor.predict(context, prefix=corrected_word.lower(), top_k=3)
            for w, p in extra:
                if w not in suggestion_set:
                    suggestion_set[w] = p * 0.8

    ranked = sorted(suggestion_set.items(), key=lambda x: -x[1])[:TOP_K]
    return ranked


def insert_suggestion(text: str, suggestion: str) -> str:
    """Replace the current prefix with the selected suggestion."""
    if not suggestion:
        return text
    if not text or text.endswith(" "):
        return (text or "") + suggestion + " "
    else:
        # Find the last space and replace everything after it
        last_space = text.rfind(" ")
        if last_space == -1:
            return suggestion + " "
        else:
            return text[: last_space + 1] + suggestion + " "


# ===================================================================== #
#  Gradio interface                                                      #
# ===================================================================== #

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&display=swap');

.gradio-container {
    max-width: 760px !important;
    margin: 0 auto !important;
    font-family: 'DM Sans', sans-serif !important;
}

.title-bar {
    text-align: center;
    padding: 20px 0 8px 0;
}
.title-bar h1 {
    font-size: 1.75rem;
    font-weight: 600;
    color: var(--body-text-color);
    margin: 0;
    letter-spacing: -0.5px;
}
.title-bar p {
    color: var(--body-text-color-subdued);
    font-size: 0.9rem;
    margin: 4px 0 0 0;
}

/* Suggestion pills */
.suggestion-row {
    display: flex;
    gap: 8px;
    min-height: 46px;
    flex-wrap: wrap;
}
.suggestion-row button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    padding: 8px 18px !important;
    transition: all 0.15s ease !important;
    min-width: 60px !important;
}

/* Status bar */
.status-text {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--body-text-color-subdued) !important;
    padding: 8px 4px !important;
    line-height: 1.6 !important;
}
.status-text span {
    display: inline;
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(css=CUSTOM_CSS, title="Word Predictor") as demo:

        # ---- Header ------------------------------------------------- #
        gr.HTML(
            '<div class="title-bar">'
            "<h1>Word Predictor</h1>"
            "<p>Type below — suggestions update with every keystroke</p>"
            "</div>"
        )

        # ---- Model selector ----------------------------------------- #
        model_selector = gr.Radio(
            choices=MODEL_CHOICES,
            value=MODEL_CHOICES[0],
            label="Model",
            interactive=True,
        )

        # ---- Text input --------------------------------------------- #
        text_input = gr.Textbox(
            label="Start typing",
            placeholder="e.g. the quick brown …",
            lines=3,
            max_lines=6,
            autofocus=True,
            interactive=True,
        )

        # ---- Suggestion buttons ------------------------------------- #
        gr.HTML(
            '<p style="margin:8px 0 2px;font-weight:500;'
            'font-size:0.85rem;color:var(--body-text-color-subdued)">'
            "Suggestions  (click to insert)</p>"
        )

        with gr.Row(elem_classes="suggestion-row"):
            btns = [
                gr.Button(
                    value="…",
                    visible=True,
                    variant="secondary",
                    size="sm",
                    interactive=False,
                )
                for _ in range(TOP_K)
            ]

        # ---- Status / debug info ------------------------------------ #
        status = gr.HTML(
            value='<div class="status-text">Ready</div>',
        )

        # ============================================================= #
        #  Event handlers                                                #
        # ============================================================= #

        def on_text_change(text: str, model_name: str):
            """Fired on every keystroke.  Returns updated buttons + status."""
            context, prefix = parse_input(text)

            t0 = time.time()
            preds = get_predictions(text, model_name)
            elapsed_ms = (time.time() - t0) * 1000

            # Build button updates
            btn_updates = []
            for i in range(TOP_K):
                if i < len(preds):
                    word, prob = preds[i]
                    label = f"{word}  ({prob:.0%})"
                    btn_updates.append(
                        gr.Button(value=label, interactive=True, visible=True)
                    )
                else:
                    btn_updates.append(
                        gr.Button(value="…", interactive=False, visible=True)
                    )

            ctx_str = " ".join(context) if context else "—"
            pfx_str = f'"{prefix}"' if prefix else '""'
            status_html = (
                f'<div class="status-text">'
                f"context: <b>{ctx_str}</b> &nbsp;│&nbsp; "
                f"prefix: <b>{pfx_str}</b> &nbsp;│&nbsp; "
                f"hits: <b>{len(preds)}</b> &nbsp;│&nbsp; "
                f"latency: <b>{elapsed_ms:.0f} ms</b>"
                f"</div>"
            )

            return btn_updates + [status_html]

        # Wire the input event to all outputs
        text_input.input(
            fn=on_text_change,
            inputs=[text_input, model_selector],
            outputs=btns + [status],
        )

        # Also re-predict when the model selector changes
        model_selector.change(
            fn=on_text_change,
            inputs=[text_input, model_selector],
            outputs=btns + [status],
        )

        # ---- Suggestion click handlers ------------------------------ #
        def make_click_handler(btn_index: int):
            """Create a click handler for the i-th suggestion button."""

            def handler(text: str, btn_label: str, model_name: str):
                # Extract the word from the button label ("word  (85%)")
                word = btn_label.split("(")[0].strip()
                if not word or word == "…":
                    return [text] + [gr.Button()] * TOP_K + [""]

                new_text = insert_suggestion(text, word)

                # Immediately re-predict for the new state
                return [new_text] + list(on_text_change(new_text, model_name))

            return handler

        for i, btn in enumerate(btns):
            btn.click(
                fn=make_click_handler(i),
                inputs=[text_input, btn, model_selector],
                outputs=[text_input] + btns + [status],
            )

    return demo


# ===================================================================== #
#  Entry point                                                           #
# ===================================================================== #

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
