"""
Transformer-based Word Predictor with Subword (BPE) Tokenisation.

Two concrete implementations sharing the same public API:

  SmallTransformerPredictor  – trains a lightweight causal LM from scratch
                                on any text corpus.  Uses a BPE tokenizer
                                built with HuggingFace `tokenizers`.

  GPT2Predictor              – wraps a pre-trained GPT-2 model from
                                HuggingFace (requires internet for first
                                download).

Both expose the same .predict(context, prefix, top_k) interface as the
NGramPredictor so they can be used interchangeably.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ===================================================================== #
#  BPE tokenizer helper                                                  #
# ===================================================================== #


def build_bpe_tokenizer(
    corpus: str,
    vocab_size: int = 500,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a byte-pair-encoding tokenizer from raw text.

    Returns a HuggingFace `tokenizers.Tokenizer` object.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )

    # The trainer expects an iterator of strings (one per "document")
    lines = [line.strip() for line in corpus.split("\n") if line.strip()]
    tokenizer.train_from_iterator(lines, trainer=trainer)
    return tokenizer


# ===================================================================== #
#  Small Causal Transformer (from scratch)                               #
# ===================================================================== #


class CausalTransformerLM(nn.Module):
    """Minimal GPT-style causal language model.

    Parameters
    ----------
    vocab_size   : number of BPE tokens
    d_model      : embedding / hidden dimension
    n_heads      : number of attention heads
    n_layers     : number of decoder blocks
    max_seq_len  : maximum context length
    dropout      : dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self, input_ids: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids    : (batch, seq_len)  token indices
        padding_mask : (batch, seq_len)  True where padded

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        causal = self._causal_mask(T, device)

        x = self.transformer(
            x,
            mask=causal,
            src_key_padding_mask=padding_mask,
        )
        x = self.ln_f(x)
        return self.lm_head(x)


# ===================================================================== #
#  SmallTransformerPredictor                                             #
# ===================================================================== #


class SmallTransformerPredictor:
    """Self-contained Transformer word predictor.

    Trains both a BPE tokenizer and a small causal LM from raw text.

    Parameters
    ----------
    vocab_size  : BPE vocabulary size
    d_model     : Transformer hidden dimension
    n_heads     : attention heads
    n_layers    : Transformer decoder layers
    max_seq_len : max context window
    """

    def __init__(
        self,
        vocab_size: int = 500,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 64,
        device: Optional[str] = None,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer: Optional[Tokenizer] = None
        self.model: Optional[CausalTransformerLM] = None
        self._pad_id: int = 0
        self._word_cache: dict[str, int] = {}  # whole-word → token-id
        self._corpus_words: set[str] = set()  # actual words from training

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #
    def train(
        self,
        corpus: str,
        epochs: int = 40,
        lr: float = 3e-3,
        batch_size: int = 8,
        seq_len: int = 32,
        log_every: int = 10,
    ) -> List[float]:
        """Train the BPE tokenizer + Transformer LM from raw text.

        Returns a list of per-epoch average losses.
        """
        # 1. Build BPE tokenizer
        self.tokenizer = build_bpe_tokenizer(corpus, vocab_size=self.vocab_size)
        actual_vocab = self.tokenizer.get_vocab_size()
        self._pad_id = self.tokenizer.token_to_id("<pad>")

        # Build the set of actual whole words from the corpus
        self._corpus_words = set(re.findall(r"[a-zA-Z]+", corpus.lower()))

        # Cache which BPE tokens are whole words (for fast filtering later)
        self._build_word_cache()

        # 2. Encode corpus
        encoded = self.tokenizer.encode(corpus.strip())
        token_ids = encoded.ids
        if len(token_ids) < seq_len + 1:
            # Repeat corpus so we have enough training data
            repeats = (seq_len * batch_size * 4) // len(token_ids) + 1
            token_ids = token_ids * repeats

        # 3. Create training batches  (input, target shifted by 1)
        data = torch.tensor(token_ids, dtype=torch.long)
        batches = []
        for i in range(0, len(data) - seq_len - 1, seq_len):
            inp = data[i : i + seq_len]
            tgt = data[i + 1 : i + seq_len + 1]
            if len(inp) == seq_len and len(tgt) == seq_len:
                batches.append((inp, tgt))

        # 4. Initialise model
        self.model = CausalTransformerLM(
            vocab_size=actual_vocab,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            max_seq_len=self.max_seq_len,
        ).to(self.device)

        # 5. Training loop
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self._pad_id)

        epoch_losses: List[float] = []
        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            n_batches = 0
            for start in range(0, len(batches), batch_size):
                batch = batches[start : start + batch_size]
                if not batch:
                    continue
                inp = torch.stack([b[0] for b in batch]).to(self.device)
                tgt = torch.stack([b[1] for b in batch]).to(self.device)

                logits = self.model(inp)  # (B, T, V)
                loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)
            if epoch % log_every == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}")

        self.model.eval()
        return epoch_losses

    # ------------------------------------------------------------------ #
    #  Whole-word token cache                                             #
    # ------------------------------------------------------------------ #
    def _build_word_cache(self) -> None:
        """Identify which BPE tokens correspond to complete whole words."""
        vocab = self.tokenizer.get_vocab()
        self._word_cache = {}
        for token_str, token_id in vocab.items():
            # Skip special tokens
            if token_str.startswith("<") and token_str.endswith(">"):
                continue
            # A "whole word" token is purely alphabetic (no sub-word markers)
            clean = token_str.strip()
            if clean.isalpha() and len(clean) > 1:
                self._word_cache[clean.lower()] = token_id

    # ------------------------------------------------------------------ #
    #  Prediction                                                         #
    # ------------------------------------------------------------------ #
    def _encode_context(self, context: List[str]) -> List[int]:
        """Encode context words into token IDs."""
        context_text = " ".join(context).strip()
        if not context_text:
            return [self.tokenizer.token_to_id("<bos>")]
        return self.tokenizer.encode(context_text).ids

    def _get_next_probs(self, input_ids: List[int]) -> torch.Tensor:
        """Run the model and return softmax probabilities for next token."""
        ids = input_ids[-self.max_seq_len :]
        inp = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits = self.model(inp)
        return F.softmax(logits[0, -1, :], dim=-1)

    @torch.no_grad()
    def predict(
        self,
        context: List[str],
        prefix: str = "",
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Predict the next whole word using dual-path strategy.

        Path A – context only (no prefix fed to model):
            Get raw next-token predictions, collect whole words that
            match the prefix.  Best when prefix is empty or short.

        Path B – context + prefix (prefix fed to model):
            Let the model see the partial word and predict its
            continuation.  Best when prefix is long enough for BPE
            to encode meaningfully.

        Results from both paths are merged and de-duplicated.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not trained – call .train() first.")

        prefix = prefix.lower().strip()
        word_scores: dict[str, float] = {}

        # ---- Path A: context-only predictions ---- #
        ctx_ids = self._encode_context(context)
        probs_a = self._get_next_probs(ctx_ids)

        raw_top_n = min(200, probs_a.shape[0])
        top_probs_a, top_ids_a = probs_a.topk(raw_top_n)

        extend_budget_a = 10

        for prob_val, token_id in zip(top_probs_a.tolist(), top_ids_a.tolist()):
            token_str = self.tokenizer.decode([token_id]).strip().lower()
            if not token_str or not token_str.isalpha():
                continue

            # Fast path: token is a known whole word matching prefix
            if token_str in self._corpus_words and token_str.startswith(prefix):
                word_scores[token_str] = word_scores.get(token_str, 0.0) + prob_val
                continue

            # Extension path: subword → try to build a whole word
            if extend_budget_a > 0 and token_str.startswith(prefix[: len(token_str)]):
                extend_budget_a -= 1
                extended = self._greedy_extend(
                    ctx_ids + [token_id], token_str, prob_val
                )
                for word, word_prob in extended:
                    if word.startswith(prefix):
                        word_scores[word] = word_scores.get(word, 0.0) + word_prob

        # ---- Path B: context + prefix (only if prefix >= 2 chars) ---- #
        if len(prefix) >= 2:
            ctx_plus_prefix = " ".join(context).strip() + " " + prefix
            enc_b = self.tokenizer.encode(ctx_plus_prefix)
            ids_b = enc_b.ids

            probs_b = self._get_next_probs(ids_b)
            top_probs_b, top_ids_b = probs_b.topk(raw_top_n)
            extend_budget_b = 10

            # Figure out which tokens belong to the prefix encoding
            ctx_only_enc = (
                self.tokenizer.encode(" ".join(context).strip()).ids if context else []
            )
            prefix_token_ids = ids_b[len(ctx_only_enc) :]

            for prob_val, token_id in zip(top_probs_b.tolist(), top_ids_b.tolist()):
                # Decode prefix tokens + this continuation token
                full_ids = prefix_token_ids + [token_id]
                decoded = self.tokenizer.decode(full_ids).strip().lower()

                if not decoded or not decoded.isalpha():
                    continue

                # Fast path: decoded is a complete known word
                if decoded in self._corpus_words and decoded.startswith(prefix):
                    word_scores[decoded] = (
                        word_scores.get(decoded, 0.0) + prob_val * 0.8
                    )
                    continue

                # Extension path
                if extend_budget_b > 0 and decoded.startswith(prefix):
                    extend_budget_b -= 1
                    extended = self._greedy_extend(
                        ids_b + [token_id], decoded, prob_val * 0.8
                    )
                    for word, word_prob in extended:
                        if word.startswith(prefix):
                            word_scores[word] = word_scores.get(word, 0.0) + word_prob

        # Sort and return
        ranked = sorted(word_scores.items(), key=lambda x: -x[1])[:top_k]
        total = sum(s for _, s in ranked) or 1.0
        return [(w, round(s / total, 4)) for w, s in ranked]

    @torch.no_grad()
    def _greedy_extend(
        self,
        token_ids: List[int],
        current_text: str,
        base_prob: float,
        max_steps: int = 4,
    ) -> List[Tuple[str, float]]:
        """Greedily extend a subword sequence until a known word forms.

        Fast: single forward pass per step (no beam branching).
        """
        results: list[Tuple[str, float]] = []
        ids = list(token_ids)
        cum_prob = base_prob
        text = current_text

        for _ in range(max_steps):
            probs = self._get_next_probs(ids)
            best_prob, best_id = probs.max(dim=-1)
            ids.append(best_id.item())
            cum_prob *= best_prob.item()

            new_token = self.tokenizer.decode([best_id.item()]).strip().lower()
            text += new_token

            if not text.isalpha():
                break
            if text in self._corpus_words:
                results.append((text, cum_prob))
            if cum_prob < 1e-6:
                break

        return results

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(directory, "model.pt"))
        self.tokenizer.save(os.path.join(directory, "tokenizer.json"))
        # Save the corpus vocabulary for whole-word validation
        with open(os.path.join(directory, "corpus_words.txt"), "w") as f:
            f.write("\n".join(sorted(self._corpus_words)))
        print(f"  Saved model + tokenizer to {directory}/")

    def load(self, directory: str) -> None:
        self.tokenizer = Tokenizer.from_file(os.path.join(directory, "tokenizer.json"))
        actual_vocab = self.tokenizer.get_vocab_size()
        self._pad_id = self.tokenizer.token_to_id("<pad>")
        self._build_word_cache()

        # Load corpus vocabulary
        words_path = os.path.join(directory, "corpus_words.txt")
        if os.path.exists(words_path):
            with open(words_path) as f:
                self._corpus_words = set(f.read().strip().split("\n"))
        else:
            self._corpus_words = set(self._word_cache.keys())

        self.model = CausalTransformerLM(
            vocab_size=actual_vocab,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            max_seq_len=self.max_seq_len,
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(
                os.path.join(directory, "model.pt"),
                map_location=self.device,
                weights_only=True,
            )
        )
        self.model.eval()
        print(f"  Loaded model + tokenizer from {directory}/")

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        return (
            f"SmallTransformerPredictor("
            f"vocab={self.vocab_size}, "
            f"d={self.d_model}, "
            f"layers={self.n_layers}, "
            f"params={n_params:,})"
        )


# ===================================================================== #
#  GPT2Predictor  (for use when HuggingFace Hub is reachable)            #
# ===================================================================== #


class GPT2Predictor:
    """Word predictor wrapping the pre-trained GPT-2 model.

    Requires internet to download weights on first use.
    After that, models are cached locally.

    Parameters
    ----------
    model_name : str   e.g. 'gpt2', 'gpt2-medium'
    """

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Download (or load from cache) the GPT-2 model."""
        from transformers import GPT2LMHeadModel, GPT2Tokenizer  # noqa: E402

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Loaded {self.model_name}: {n_params // 1_000_000}M params")

    @torch.no_grad()
    def predict(
        self,
        context: List[str],
        prefix: str = "",
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Predict the next whole word (same API as NGramPredictor)."""
        if self.model is None:
            raise RuntimeError("Call .load() first.")

        prefix = prefix.lower().strip()
        context_text = " ".join(context).strip()
        if prefix:
            context_text += " " + prefix

        if not context_text:
            context_text = " "

        input_ids = self.tokenizer.encode(context_text, return_tensors="pt").to(
            self.device
        )
        logits = self.model(input_ids).logits
        next_logits = logits[0, -1, :]
        probs = F.softmax(next_logits, dim=-1)

        # GPT-2 BPE: tokens starting with 'Ġ' mark word boundaries
        raw_top_k = min(300, probs.shape[0])
        top_probs, top_ids = probs.topk(raw_top_k)

        word_scores: dict[str, float] = {}

        for prob_val, token_id in zip(top_probs.tolist(), top_ids.tolist()):
            token_str = self.tokenizer.decode([token_id]).strip().lower()
            if not token_str or not token_str.isalpha():
                continue
            if token_str.startswith(prefix):
                word_scores[token_str] = word_scores.get(token_str, 0.0) + prob_val

        ranked = sorted(word_scores.items(), key=lambda x: -x[1])[:top_k]
        total = sum(s for _, s in ranked) or 1.0
        return [(w, round(s / total, 4)) for w, s in ranked]

    def __repr__(self) -> str:
        status = "loaded" if self.model else "not loaded"
        return f"GPT2Predictor(model='{self.model_name}', {status})"
