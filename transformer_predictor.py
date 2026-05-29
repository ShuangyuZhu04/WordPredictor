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
from collections import Counter
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
    vocab_size: int = 1000,
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
#  Small Causal Transformer                                             #
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
            norm_first=True,
            activation="gelu",
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
        vocab_size: int = 1000,
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
        self._word_freq: dict[str, int] = {}  # whole-word → corpus frequency

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

        # Build the set of actual whole words from the corpus + their counts.
        # \b...\b avoids extracting letter fragments glued to digits, e.g.
        # "20th" / "1st" / "2nd" would otherwise yield bogus words th/st/nd.
        words = re.findall(r"\b[a-z]+(?:['-][a-z]+)*\b", corpus.lower())
        self._word_freq = dict(Counter(words))
        self._corpus_words = set(self._word_freq)

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

    def _is_valid_prediction_word(self, word: str) -> bool:
        word = word.lower().strip()

        if not word:
            return False

        if len(word) == 1 and word not in {"a", "i"}:
            return False

        if not re.fullmatch(r"[a-z]+(?:['-][a-z]+)*", word):
            return False

        return word in self._corpus_words

    @torch.no_grad()
    def _score_batched(self, ctx_ids, cand, top_k):
        if not cand:
            return []

        pad = self._pad_id
        seqs, meta = [], []

        for w in cand:
            wt = self.tokenizer.encode(w).ids
            if not wt:
                continue

            seq = (ctx_ids + wt)[-self.max_seq_len :]
            start = len(seq) - len(wt) - 1

            if start < 0:
                continue

            seqs.append(seq)
            meta.append((w, wt, start))

        if not seqs:
            return []

        L = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), L), pad, dtype=torch.long, device=self.device)

        for i, s in enumerate(seqs):
            inp[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)

        logp = F.log_softmax(self.model(inp), dim=-1)

        scored = []

        for i, (w, wt, start) in enumerate(meta):
            score = sum(logp[i, start + j, wt[j]].item() for j in range(len(wt)))
            scored.append((w, score))

        scored.sort(key=lambda x: -x[1])
        top = scored[:top_k]

        if not top:
            return []

        # stable softmax
        m = top[0][1]
        exps = [math.exp(s - m) for _, s in top]
        tot = sum(exps) or 1.0

        return [(w, round(e / tot, 4)) for (w, _), e in zip(top, exps)]

    @torch.no_grad()
    def predict(self, context, prefix="", top_k=5, max_candidates=1000):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not trained – call .train() first.")
        prefix = prefix.lower().strip()
        ctx_ids = self._encode_context(context)

        # 有 prefix:先按词频裁到 max_candidates,再批量打分
        freq = self._word_freq
        cand = [
            w
            for w in self._corpus_words
            if self._is_valid_prediction_word(w) and w.startswith(prefix)
        ]
        cand.sort(key=lambda w: -freq.get(w, 0))
        cand = cand[:max_candidates]
        return self._score_batched(ctx_ids, cand, top_k)

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(directory, "model.pt"))
        self.tokenizer.save(os.path.join(directory, "tokenizer.json"))
        # Save the corpus vocabulary + frequencies for whole-word validation
        with open(os.path.join(directory, "corpus_words.txt"), "w") as f:
            for w in sorted(self._word_freq, key=lambda x: (-self._word_freq[x], x)):
                f.write(f"{w}\t{self._word_freq[w]}\n")
        print(f"  Saved model + tokenizer to {directory}/")

    def load(self, directory: str) -> None:
        self.tokenizer = Tokenizer.from_file(os.path.join(directory, "tokenizer.json"))
        actual_vocab = self.tokenizer.get_vocab_size()
        self._pad_id = self.tokenizer.token_to_id("<pad>")
        self._build_word_cache()

        # Load corpus vocabulary + frequencies
        words_path = os.path.join(directory, "corpus_words.txt")
        if os.path.exists(words_path):
            self._word_freq = {}
            with open(words_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    word = parts[0]
                    count = int(parts[1]) if len(parts) > 1 else 1
                    self._word_freq[word] = count
            self._corpus_words = set(self._word_freq)
        else:
            self._corpus_words = set(self._word_cache.keys())
            self._word_freq = {w: 1 for w in self._corpus_words}

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
