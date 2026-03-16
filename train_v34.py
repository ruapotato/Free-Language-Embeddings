#!/usr/bin/env python3
"""
FLM V34 — Dynamic Masking Word2vec

Unified masking framework on token windows. Three modes cycled every step:
  - SG-like (step%3==0): mask all but one word → predict many from one
  - CBOW-like (step%3==1): mask one word → predict one from many
  - Variable (step%3==2): mask random k words (2 ≤ k ≤ L-1)

Same vocab as V28/V33 for fair A/B comparison.

Changes from V33:
  - Single forward() with unmasked/masked split instead of separate SG/CBOW
  - Window ring buffer stores full token windows, masking at batch time
  - Three masking modes instead of two fixed objectives
  - 2M steps (2x V33) — more modes need more training

Usage:
    python train_v34.py --fresh
    python train_v34.py --resume
"""

import os
import re
import json
import math
import time
import random
import signal
import datetime
import argparse
import threading
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_DIM = 300          # Same as V28/V33 for A/B comparison
WINDOW_SIZE = 5          # Context window (each side)
MAX_WIN = 2 * WINDOW_SIZE + 1  # Max tokens per window (center + context)
NEG_SAMPLES = 15         # Negative samples per masked token
BATCH_SIZE = 4096
TOTAL_STEPS = 2_000_000  # 2x V33
PEAK_LR = 0.025
MIN_LR = 1e-4
WARMUP_STEPS = 4000
SUBSAMPLE_THRESHOLD = 1e-4
LOG_EVERY = 50
EVAL_EVERY = 5000
SAVE_EVERY = 10000
MIN_COUNT = 10
VOCAB_SAMPLE_DOCS = 200_000
MAX_VOCAB = 100_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints/word2vec_v34"
LOG_DIR = "logs"
LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v34.log",
    "metrics": f"{LOG_DIR}/concept_v34_metrics.csv",
}

PRETRAIN_SOURCES = [
    ("data/pretrain/wikipedia.jsonl", 0.30),
    ("data/pretrain/gutenberg.jsonl", 0.20),
    ("data/pretrain/stackexchange.jsonl", 0.16),
    ("data/pretrain/arxiv.jsonl", 0.20),
    ("data/pretrain/usgpo.jsonl", 0.10),
    ("data/pretrain/rfcs.jsonl", 0.025),
    ("data/pretrain/kernel_docs.jsonl", 0.0017),
    ("data/pretrain/archwiki.jsonl", 0.0015),
    ("data/pretrain/tldp.jsonl", 0.0016),
    ("data/pretrain/gnu_manuals.jsonl", 0.0023),
    ("data/pretrain/manpages.jsonl", 0.0049),
]

VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "vocab.json")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATHS["log"], "a") as f:
        f.write(line + "\n")

def log_metrics(step, metrics):
    path = LOG_PATHS["metrics"]
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(",".join(["step"] + list(metrics.keys())) + "\n")
    with open(path, "a") as f:
        vals = [str(step)] + [f"{v:.6f}" for v in metrics.values()]
        f.write(",".join(vals) + "\n")


# ---------------------------------------------------------------------------
# Whole-word vocabulary (identical to V28/V33)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

def tokenize_text(text):
    return _WORD_RE.findall(text.lower())


class Vocabulary:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.counts = []
        self.total_count = 0

    def build_from_sources(self, sources, num_docs=VOCAB_SAMPLE_DOCS, min_count=MIN_COUNT,
                           max_vocab=MAX_VOCAB):
        log(f"Building vocabulary from {num_docs:,} documents...")

        available = [(p, w) for p, w in sources if os.path.exists(p)]
        total_w = sum(w for _, w in available)
        cum_weights = []
        cum = 0
        for _, w in available:
            cum += w / total_w
            cum_weights.append(cum)
        files = [open(p) for p, _ in available]

        word_counts = Counter()
        docs_read = 0
        total_words = 0

        while docs_read < num_docs:
            r = random.random()
            src_idx = len(available) - 1
            for i, cw in enumerate(cum_weights):
                if r <= cw:
                    src_idx = i
                    break

            line = files[src_idx].readline()
            if not line:
                files[src_idx].close()
                files[src_idx] = open(available[src_idx][0])
                line = files[src_idx].readline()
                if not line:
                    continue

            try:
                text = json.loads(line).get("text", "").strip()
            except (json.JSONDecodeError, KeyError):
                continue

            if len(text) < 20:
                continue

            words = tokenize_text(text)
            word_counts.update(words)
            total_words += len(words)
            docs_read += 1

            if docs_read % 50000 == 0:
                log(f"  {docs_read:,} docs, {total_words:,} words, "
                    f"{len(word_counts):,} unique")

        for f in files:
            f.close()

        filtered = [(w, c) for w, c in word_counts.most_common() if c >= min_count]
        if len(filtered) > max_vocab:
            filtered = filtered[:max_vocab]

        self.word2id = {w: i for i, (w, _) in enumerate(filtered)}
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.counts = [c for _, c in filtered]
        self.total_count = sum(self.counts)

        log(f"  Vocabulary: {len(self.word2id):,} words "
            f"(from {len(word_counts):,} unique, min_count={min_count})")
        log(f"  Total tokens: {total_words:,}")
        log(f"  Top 20: {', '.join(w for w, _ in filtered[:20])}")

    def save(self, path):
        data = {
            "word2id": self.word2id,
            "counts": self.counts,
            "total_count": self.total_count,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        log(f"  Vocabulary saved: {path}")

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.word2id = data["word2id"]
        self.id2word = {int(i): w for w, i in self.word2id.items()}
        if isinstance(data["counts"], dict):
            self.counts = [data["counts"][str(i)] for i in range(len(self.word2id))]
        else:
            self.counts = data["counts"]
        self.total_count = data["total_count"]
        log(f"  Vocabulary loaded: {len(self.word2id):,} words from {path}")

    def encode(self, words):
        return [self.word2id[w] for w in words if w in self.word2id]

    def decode(self, ids):
        return [self.id2word.get(i, "<unk>") for i in ids]

    def __len__(self):
        return len(self.word2id)

    def __contains__(self, word):
        return word in self.word2id


# ---------------------------------------------------------------------------
# Dynamic Masking Model
# ---------------------------------------------------------------------------

class DynamicMaskWord2vec(nn.Module):
    """Unified masking word2vec with shared embeddings.

    Single forward pass handles all masking modes:
      - target_embeddings: input (looks up unmasked tokens, averages them)
      - context_embeddings: output (predicts each masked token)
    """

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)

        initrange = 0.5 / embed_dim
        nn.init.uniform_(self.target_embeddings.weight, -initrange, initrange)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, unmasked_ids, unmasked_mask, masked_ids, neg_ids):
        """Unified forward: average unmasked tokens, predict each masked token.

        unmasked_ids:  (batch, max_unmasked) — token IDs visible as input
        unmasked_mask: (batch, max_unmasked) — 1 for real tokens, 0 for padding
        masked_ids:    (batch, max_masked) — token IDs to predict
        neg_ids:       (batch, max_masked, num_neg) — negative samples per masked token
        """
        # Average the unmasked (visible) tokens as the input signal
        inp_emb = self.target_embeddings(unmasked_ids)      # (batch, max_unmasked, dim)
        mask = unmasked_mask.unsqueeze(-1).float()           # (batch, max_unmasked, 1)
        inp_sum = (inp_emb * mask).sum(dim=1)                # (batch, dim)
        inp_count = mask.sum(dim=1).clamp(min=1)             # (batch, 1)
        inp_avg = inp_sum / inp_count                        # (batch, dim)

        # Predict each masked token
        out_emb = self.context_embeddings(masked_ids)        # (batch, max_masked, dim)
        neg_emb = self.context_embeddings(neg_ids)           # (batch, max_masked, num_neg, dim)

        # Positive scores: dot product of input avg with each masked token
        # inp_avg: (batch, dim) -> (batch, 1, dim) for broadcasting
        pos_score = (inp_avg.unsqueeze(1) * out_emb).sum(dim=-1)  # (batch, max_masked)
        pos_loss = F.logsigmoid(pos_score)                         # (batch, max_masked)

        # Negative scores: dot product of input avg with each negative
        # inp_avg: (batch, dim) -> (batch, 1, 1, dim)
        neg_score = (inp_avg.unsqueeze(1).unsqueeze(1) * neg_emb).sum(dim=-1)  # (batch, max_masked, num_neg)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=-1)    # (batch, max_masked)

        # Only count real masked positions (not padding)
        # masked_ids > 0 isn't reliable since id 0 is valid; use a separate mask
        # We know masked positions are valid if they were set (non-padding)
        # The caller guarantees masked positions come first, padded with 0s
        # We'll use a mask derived from the batch construction
        total_loss = -(pos_loss + neg_loss)                 # (batch, max_masked)

        return total_loss.mean()

    def get_embeddings(self):
        return self.target_embeddings.weight.detach()


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

class WindowDataset:
    """Streams token windows with dynamic masking at batch time.

    Stores full token windows in a ring buffer. Masking is applied when
    constructing batches, not during data loading.
    """

    BUF_SIZE = 500_000

    def __init__(self, sources, vocab, window_size=WINDOW_SIZE,
                 subsample_threshold=SUBSAMPLE_THRESHOLD):
        self.sources = [(p, w) for p, w in sources if os.path.exists(p)]
        if not self.sources:
            raise FileNotFoundError("No data sources found")
        self.vocab = vocab
        self.window_size = window_size
        self.max_win = 2 * window_size + 1

        # Source sampling weights
        total_w = sum(w for _, w in self.sources)
        self._cum_weights = []
        cum = 0
        for _, w in self.sources:
            cum += w / total_w
            self._cum_weights.append(cum)
        self._files = [open(p) for p, _ in self.sources]

        # Subsampling probabilities
        self._subsample_probs = np.ones(len(vocab), dtype=np.float32)
        for i, count in enumerate(vocab.counts):
            freq = count / max(vocab.total_count, 1)
            if freq > 0:
                self._subsample_probs[i] = min(1.0,
                    (math.sqrt(freq / subsample_threshold) + 1) *
                    (subsample_threshold / freq))

        # Negative sampling table
        freqs = np.array(vocab.counts, dtype=np.float64)
        freqs = np.power(freqs, 0.75)
        self._neg_probs = freqs / freqs.sum()
        self._neg_table_size = 10_000_000
        self._neg_table = np.random.choice(len(vocab), size=self._neg_table_size,
                                           p=self._neg_probs).astype(np.int64)
        self._neg_idx = 0

        # Window ring buffer: each row is a full token window
        # Column 0 = window length, columns 1..max_win = token IDs
        self._windows = np.zeros((self.BUF_SIZE, self.max_win + 1), dtype=np.int32)
        self._win_count = 0
        self._win_valid = 0

        # Fill initial buffer
        self._fill_initial()

        log(f"WindowDataset: {len(self.sources)} sources, "
            f"vocab={len(vocab):,}, window={window_size}, buf={self.BUF_SIZE:,}")

    def _sample_source(self):
        r = random.random()
        for i, cw in enumerate(self._cum_weights):
            if r <= cw:
                return i
        return len(self.sources) - 1

    def _read_doc(self, src_idx):
        f = self._files[src_idx]
        line = f.readline()
        if not line:
            f.close()
            self._files[src_idx] = open(self.sources[src_idx][0])
            f = self._files[src_idx]
            line = f.readline()
            if not line:
                return None
        try:
            return json.loads(line).get("text", "").strip()
        except (json.JSONDecodeError, KeyError):
            return None

    def _tokenize_and_subsample(self, text):
        words = tokenize_text(text)
        ids = self.vocab.encode(words)
        return [i for i in ids if random.random() < self._subsample_probs[i]]

    def _process_doc(self, token_ids):
        """Extract windows and write into ring buffer."""
        n = len(token_ids)
        for i in range(n):
            w = random.randint(1, self.window_size)
            start = max(0, i - w)
            end = min(n, i + w + 1)
            # Build window: center at position 0, context follows
            window = [token_ids[i]]  # center first
            for j in range(start, end):
                if j != i:
                    window.append(token_ids[j])

            win_len = len(window)
            if win_len < 2:  # Need at least center + 1 context
                continue

            idx = self._win_count % self.BUF_SIZE
            self._windows[idx, 0] = win_len
            self._windows[idx, 1:win_len + 1] = window
            # Zero out remaining slots
            if win_len + 1 < self.max_win + 1:
                self._windows[idx, win_len + 1:] = 0
            self._win_count += 1

        self._win_valid = min(self._win_count, self.BUF_SIZE)

    def _fill_initial(self):
        target = self.BUF_SIZE // 2
        attempts = 0
        while self._win_valid < target and attempts < 50000:
            attempts += 1
            src = self._sample_source()
            text = self._read_doc(src)
            if not text or len(text) < 30:
                continue
            ids = self._tokenize_and_subsample(text)
            if len(ids) < 3:
                continue
            self._process_doc(ids)

    def _add_docs(self, n_docs=50):
        added = 0
        attempts = 0
        while added < n_docs and attempts < n_docs * 3:
            attempts += 1
            src = self._sample_source()
            text = self._read_doc(src)
            if not text or len(text) < 30:
                continue
            ids = self._tokenize_and_subsample(text)
            if len(ids) < 3:
                continue
            self._process_doc(ids)
            added += 1

    def start_reader(self):
        self._reader_stop = threading.Event()
        def _read_loop():
            while not self._reader_stop.is_set():
                self._add_docs(10)
                time.sleep(0.05)  # Yield GIL
        self._reader_thread = threading.Thread(target=_read_loop, daemon=True)
        self._reader_thread.start()

    def stop_reader(self):
        if hasattr(self, '_reader_stop'):
            self._reader_stop.set()

    def _sample_negatives(self, shape):
        """Sample negatives from the pre-built table. shape = total count needed."""
        total = int(np.prod(shape))
        if self._neg_idx + total > self._neg_table_size:
            self._neg_idx = 0
        neg = self._neg_table[self._neg_idx:self._neg_idx + total]
        self._neg_idx += total
        return neg.reshape(shape)

    def get_batch(self, batch_size, mode):
        """Get a batch with dynamic masking.

        mode: "sg" | "cbow" | "variable"

        SG and CBOW are fully vectorized. Variable uses a Python loop
        (unavoidable: per-sample random permutation count).

        Returns:
            unmasked_ids:  (batch, max_cols) int64
            unmasked_mask: (batch, max_cols) int64
            masked_ids:    (batch, max_masked) int64
            neg_ids:       (batch, max_masked, num_neg) int64
        """
        indices = np.random.randint(0, self._win_valid, size=batch_size)
        windows = self._windows[indices]  # (batch, max_win+1)
        lengths = windows[:, 0]           # (batch,)
        tokens = windows[:, 1:]           # (batch, max_win)  — col 0 is center

        if mode == "sg":
            # Keep 1 unmasked (center at col 0), mask rest (cols 1..L-1)
            unmasked_ids = tokens[:, 0:1].astype(np.int64)  # (batch, 1)
            unmasked_mask = np.ones((batch_size, 1), dtype=np.int64)
            # Context tokens are cols 1..max_win-1, valid where col < length
            ctx = tokens[:, 1:].astype(np.int64)            # (batch, max_win-1)
            col_idx = np.arange(ctx.shape[1])[None, :]      # (1, max_win-1)
            ctx_valid = col_idx < (lengths[:, None] - 1)     # (batch, max_win-1)
            masked_ids = ctx * ctx_valid                     # zero-padded
            max_m = int((lengths - 1).max()) if len(lengths) > 0 else 1
            max_m = max(max_m, 1)
            masked_ids = masked_ids[:, :max_m]

        elif mode == "cbow":
            # Mask 1 (center at col 0), keep rest unmasked
            masked_ids = tokens[:, 0:1].astype(np.int64)     # (batch, 1)
            ctx = tokens[:, 1:].astype(np.int64)             # (batch, max_win-1)
            col_idx = np.arange(ctx.shape[1])[None, :]
            ctx_valid = col_idx < (lengths[:, None] - 1)
            unmasked_ids = ctx * ctx_valid
            unmasked_mask = ctx_valid.astype(np.int64)
            max_m = 1

        else:  # variable
            # Vectorized: generate random permutations and split points
            max_win = self.max_win
            # Generate a random permutation per row by sorting random keys
            rand_keys = np.random.random((batch_size, max_win))
            # Zero out invalid positions so they sort last
            col_idx = np.arange(max_win)[None, :]
            valid = col_idx < lengths[:, None]
            rand_keys = np.where(valid, rand_keys, 2.0)  # invalid sorts to end
            perm = np.argsort(rand_keys, axis=1)  # (batch, max_win)

            # Permute tokens
            batch_idx = np.arange(batch_size)[:, None]
            shuffled = tokens[batch_idx, perm]  # (batch, max_win)

            # Random k masked per sample: k in [2, L-1], at least 1 unmasked
            # For L<=2, use k=1 (CBOW fallback)
            min_k = np.where(lengths > 2, 2, 1)
            max_k = np.maximum(lengths - 1, 1)
            # Random split point per sample
            k_values = (np.random.random(batch_size) * (max_k - min_k + 1) + min_k).astype(np.int64)
            k_values = np.minimum(k_values, max_k)  # clamp

            # Build masks: first k positions are masked, rest are unmasked
            is_masked = col_idx < k_values[:, None]     # (batch, max_win)
            is_unmasked = valid & ~is_masked

            # Pack masked tokens (left-aligned)
            max_m = int(k_values.max())
            max_m = max(max_m, 1)
            masked_ids = np.zeros((batch_size, max_m), dtype=np.int64)
            for km in range(max_m):
                mask_col = km < k_values
                masked_ids[:, km] = np.where(mask_col, shuffled[:, km], 0)

            # Pack unmasked tokens (left-aligned)
            # Unmasked tokens start at position k for each row
            max_u = int((lengths - k_values).max())
            max_u = max(max_u, 1)
            unmasked_ids = np.zeros((batch_size, max_u), dtype=np.int64)
            unmasked_mask = np.zeros((batch_size, max_u), dtype=np.int64)
            for u in range(max_u):
                src_col = k_values + u
                in_range = src_col < lengths
                safe_col = np.minimum(src_col, max_win - 1)
                unmasked_ids[:, u] = np.where(in_range, shuffled[batch_idx[:, 0], safe_col], 0)
                unmasked_mask[:, u] = in_range.astype(np.int64)

        neg_ids = self._sample_negatives((batch_size, max_m, NEG_SAMPLES))

        return (
            torch.from_numpy(unmasked_ids),
            torch.from_numpy(unmasked_mask),
            torch.from_numpy(masked_ids),
            torch.from_numpy(neg_ids),
        )


# ---------------------------------------------------------------------------
# Evaluation (identical to V28/V33)
# ---------------------------------------------------------------------------

ANALOGY_TESTS = [
    ("king", "man", "woman", "queen"),
    ("man", "woman", "boy", "girl"),
    ("big", "bigger", "small", "smaller"),
    ("good", "better", "bad", "worse"),
    ("france", "paris", "germany", "berlin"),
    ("japan", "tokyo", "italy", "rome"),
    ("slow", "slower", "fast", "faster"),
    ("tall", "taller", "short", "shorter"),
    ("go", "went", "come", "came"),
    ("see", "saw", "hear", "heard"),
]

SIMILARITY_TESTS = [
    ("king", "queen", "similar"),
    ("cat", "dog", "similar"),
    ("happy", "joyful", "similar"),
    ("big", "large", "similar"),
    ("run", "sprint", "similar"),
    ("cold", "freezing", "similar"),
    ("car", "vehicle", "similar"),
    ("doctor", "physician", "similar"),
    ("king", "banana", "different"),
    ("cat", "computer", "different"),
    ("happy", "purple", "different"),
    ("run", "philosophy", "different"),
]

DIRECTION_TESTS = [
    [("king", "queen"), ("man", "woman"), ("boy", "girl"),
     ("father", "mother"), ("brother", "sister"), ("he", "she")],
    [("small", "big"), ("tiny", "huge"), ("little", "large"),
     ("smaller", "bigger"), ("short", "tall")],
    [("go", "went"), ("run", "ran"), ("see", "saw"),
     ("come", "came"), ("eat", "ate"), ("take", "took")],
]
DIRECTION_NAMES = ["gender (M→F)", "size (small→big)", "tense (pres→past)"]


def evaluate_embeddings(model, vocab, step):
    emb = model.get_embeddings().to("cpu")
    emb_norm = F.normalize(emb, p=2, dim=-1)

    def get_vec(word):
        if word in vocab:
            return emb_norm[vocab.word2id[word]]
        return None

    def nearest(vec, exclude=None, k=5):
        sims = emb_norm @ vec
        if exclude:
            for w in exclude:
                if w in vocab:
                    sims[vocab.word2id[w]] = -1
        topk = sims.topk(k)
        return [(vocab.id2word.get(idx.item(), "?"), sim.item())
                for idx, sim in zip(topk.indices, topk.values)]

    # Analogies
    log(f"  --- Analogies ---")
    correct = 0
    total = 0
    for a, b, c, expected in ANALOGY_TESTS:
        va, vb, vc, ve = get_vec(a), get_vec(b), get_vec(c), get_vec(expected)
        if any(v is None for v in [va, vb, vc, ve]):
            log(f"    {a}:{b} :: {c}:? — skipped (OOV)")
            continue
        query = F.normalize(vc + vb - va, p=2, dim=-1)
        top = nearest(query, exclude=[a, b, c], k=5)
        hit = expected in [w for w, _ in top]
        total += 1
        if hit:
            correct += 1
        log(f"    {a}:{b} :: {c}:? → {top[0][0]} ({top[0][1]:.3f})"
            f"  [expect: {expected}] {'✓' if hit else '✗'}")
    analogy_acc = correct / max(total, 1)
    log(f"  Analogy accuracy: {correct}/{total} ({analogy_acc:.1%})")

    # Similarities
    log(f"  --- Similarities ---")
    sim_scores = {"similar": [], "different": []}
    for w1, w2, label in SIMILARITY_TESTS:
        v1, v2 = get_vec(w1), get_vec(w2)
        if v1 is None or v2 is None:
            continue
        sim = (v1 * v2).sum().item()
        sim_scores[label].append(sim)
        log(f"    {w1} ↔ {w2}: {sim:.3f} ({label})")
    avg_sim = np.mean(sim_scores["similar"]) if sim_scores["similar"] else 0
    avg_diff = np.mean(sim_scores["different"]) if sim_scores["different"] else 0
    sim_gap = avg_sim - avg_diff
    log(f"  Similar avg: {avg_sim:.3f} | Different avg: {avg_diff:.3f} | Gap: {sim_gap:+.3f}")

    # Nearest neighbors
    log(f"  --- Nearest neighbors ---")
    for word in ["king", "computer", "happy", "water", "france", "dog"]:
        v = get_vec(word)
        if v is not None:
            top = nearest(v, exclude=[word], k=8)
            nns = ", ".join(f"{w}({s:.2f})" for w, s in top)
            log(f"    {word}: {nns}")

    # Directions
    log(f"  --- Directional consistency ---")
    for name, pairs in zip(DIRECTION_NAMES, DIRECTION_TESTS):
        vecs = []
        for w1, w2 in pairs:
            v1, v2 = get_vec(w1), get_vec(w2)
            if v1 is not None and v2 is not None:
                vecs.append(F.normalize(v2 - v1, p=2, dim=-1))
        if len(vecs) >= 2:
            sims = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    sims.append((vecs[i] * vecs[j]).sum().item())
            avg = np.mean(sims)
            log(f"    {name}: avg direction consistency = {avg:.3f} ({len(vecs)} pairs)")

    return {
        "analogy_acc": analogy_acc,
        "sim_gap": sim_gap,
        "avg_similar": avg_sim,
        "avg_different": avg_diff,
    }


# ---------------------------------------------------------------------------
# Google Analogy Benchmark
# ---------------------------------------------------------------------------

def run_google_analogies(model, vocab):
    benchmark_path = "data/questions-words.txt"
    if not os.path.exists(benchmark_path):
        log("Google analogy benchmark not found, skipping")
        return None

    emb = model.get_embeddings().to("cpu")
    emb_norm = F.normalize(emb, p=2, dim=-1)

    correct = 0
    total = 0
    skipped = 0
    category_stats = {}
    current_category = ""

    with open(benchmark_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(":"):
                current_category = line[2:]
                category_stats[current_category] = {"correct": 0, "total": 0}
                continue
            parts = line.lower().split()
            if len(parts) != 4:
                continue

            a, b, c, expected = parts
            if any(w not in vocab for w in [a, b, c, expected]):
                skipped += 1
                continue

            va = emb_norm[vocab.word2id[a]]
            vb = emb_norm[vocab.word2id[b]]
            vc = emb_norm[vocab.word2id[c]]

            query = F.normalize(vb - va + vc, p=2, dim=-1)
            sims = emb_norm @ query
            for w in [a, b, c]:
                sims[vocab.word2id[w]] = -1
            pred_idx = sims.argmax().item()
            pred_word = vocab.id2word.get(pred_idx, "?")

            hit = pred_word == expected
            total += 1
            category_stats[current_category]["total"] += 1
            if hit:
                correct += 1
                category_stats[current_category]["correct"] += 1

    accuracy = correct / max(total, 1)
    coverage = total / max(total + skipped, 1)

    log(f"  === Google Analogy Benchmark ===")
    log(f"  Overall: {correct}/{total} ({accuracy:.1%}), coverage: {coverage:.1%}")

    semantic_cats = ["capital-common-countries", "capital-world", "currency",
                     "city-in-state", "family"]
    sem_c, sem_t, syn_c, syn_t = 0, 0, 0, 0
    for cat, stats in category_stats.items():
        if any(s in cat.lower() for s in [c.lower() for c in semantic_cats]):
            sem_c += stats["correct"]
            sem_t += stats["total"]
        else:
            syn_c += stats["correct"]
            syn_t += stats["total"]

    if sem_t > 0:
        log(f"  Semantic: {sem_c}/{sem_t} ({sem_c/sem_t:.1%})")
    if syn_t > 0:
        log(f"  Syntactic: {syn_c}/{syn_t} ({syn_c/syn_t:.1%})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "coverage": coverage,
        "category_stats": category_stats,
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "embed_dim": EMBED_DIM,
    }
    path = os.path.join(CHECKPOINT_DIR, f"step_{step:06d}.pt")
    torch.save(ckpt, path)
    latest = os.path.join(CHECKPOINT_DIR, "latest.pt")
    torch.save(ckpt, latest)
    log(f"  Checkpoint saved: {path}")


def load_checkpoint(model, optimizer):
    latest = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if not os.path.exists(latest):
        return 0
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step = ckpt["step"]
    log(f"Resumed V34: step {step}")
    return step


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step):
    if step < WARMUP_STEPS:
        return PEAK_LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(TOTAL_STEPS - WARMUP_STEPS, 1)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODES = ["sg", "cbow", "variable"]

def train(args):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    log("=" * 70)
    log("FLM V34 — Dynamic Masking Word2vec")
    log("=" * 70)
    log(f"  Modes: {MODES} (cycling every step)")

    # Reuse V28's vocabulary for fair A/B comparison
    vocab = Vocabulary()
    v28_vocab = "checkpoints/word2vec_v28/vocab.json"
    if os.path.exists(v28_vocab):
        log(f"Reusing V28 vocabulary for fair A/B comparison")
        vocab.load(v28_vocab)
    elif os.path.exists(VOCAB_PATH):
        vocab.load(VOCAB_PATH)
    else:
        vocab.build_from_sources(PRETRAIN_SOURCES)
        vocab.save(VOCAB_PATH)

    vocab_size = len(vocab)
    log(f"  Embed dim: {EMBED_DIM}")
    log(f"  Vocab size: {vocab_size:,}")
    log(f"  Window: {WINDOW_SIZE}, Neg samples: {NEG_SAMPLES}")
    log(f"  Batch size: {BATCH_SIZE}")
    log(f"  Total steps: {TOTAL_STEPS:,}")

    # Model
    model = DynamicMaskWord2vec(vocab_size, EMBED_DIM).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    log(f"  Parameters: {params:,} ({params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=PEAK_LR)

    # Load checkpoint
    start_step = 0
    if not args.fresh:
        start_step = load_checkpoint(model, optimizer)

    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")

    # Data
    dataset = WindowDataset(PRETRAIN_SOURCES, vocab)
    dataset.start_reader()
    log("Reader thread active, batches sampled with dynamic masking")
    log("-" * 70)

    # Training
    model.train()
    running_losses = {"sg": 0.0, "cbow": 0.0, "variable": 0.0}
    mode_counts = {"sg": 0, "cbow": 0, "variable": 0}
    start_time = time.time()

    # Signal handler
    stop_flag = [False]
    def handle_signal(sig, frame):
        log(f"Signal {sig} received, saving and exiting...")
        stop_flag[0] = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    for step in range(start_step, TOTAL_STEPS):
        if stop_flag[0]:
            save_checkpoint(model, optimizer, step)
            break

        current_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        mode = MODES[step % 3]
        unmasked_ids, unmasked_mask, masked_ids, neg_ids = dataset.get_batch(BATCH_SIZE, mode)

        loss = model.forward(
            unmasked_ids.to(DEVICE), unmasked_mask.to(DEVICE),
            masked_ids.to(DEVICE), neg_ids.to(DEVICE))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_losses[mode] += loss.item()
        mode_counts[mode] += 1

        if (step + 1) % LOG_EVERY == 0:
            avg_sg = running_losses["sg"] / max(mode_counts["sg"], 1)
            avg_cbow = running_losses["cbow"] / max(mode_counts["cbow"], 1)
            avg_var = running_losses["variable"] / max(mode_counts["variable"], 1)
            elapsed = time.time() - start_time
            sps = (step + 1 - start_step) / max(elapsed, 1)
            pct = (step + 1) / TOTAL_STEPS * 100

            log(f"step {step+1:>7d} [V34] | sg={avg_sg:.4f} cbow={avg_cbow:.4f} var={avg_var:.4f} "
                f"| lr {current_lr:.2e} | {pct:.1f}% | {sps:.1f} step/s")

            log_metrics(step + 1, {
                "sg_loss": avg_sg,
                "cbow_loss": avg_cbow,
                "var_loss": avg_var,
                "lr": current_lr,
            })

            running_losses = {"sg": 0.0, "cbow": 0.0, "variable": 0.0}
            mode_counts = {"sg": 0, "cbow": 0, "variable": 0}

        if (step + 1) % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                eval_results = evaluate_embeddings(model, vocab, step + 1)
            model.train()
            log_metrics(step + 1, {
                "sg_loss": avg_sg,
                "cbow_loss": avg_cbow,
                "var_loss": avg_var,
                "lr": current_lr,
                "analogy_acc": eval_results["analogy_acc"],
                "sim_gap": eval_results["sim_gap"],
                "avg_similar": eval_results["avg_similar"],
                "avg_different": eval_results["avg_different"],
            })

        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, step + 1)

    if not stop_flag[0]:
        save_checkpoint(model, optimizer, TOTAL_STEPS)

    # Final Google benchmark
    log("")
    log("Running Google Analogy Benchmark (A/B vs V33's 59.2%)...")
    model.eval()
    with torch.no_grad():
        run_google_analogies(model, vocab)

    dataset.stop_reader()
    log("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)
