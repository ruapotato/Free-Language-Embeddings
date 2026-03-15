#!/usr/bin/env python3
"""
FLM V28 — Word2vec Skip-Gram with Negative Sampling

Classic word2vec trained from scratch on DFSG-compliant data. Skip-gram
architecture: given a target word, predict context words within a window.
Uses negative sampling for efficient training.

Known to produce beautiful geometric properties:
  king - man + woman ≈ queen

Uses a whole-word vocabulary built from the training data (not subword
tokenization). This is critical — word2vec needs whole words like "king"
and "queen", not subword pieces like "##ing".

Architecture:
    - Embedding matrix: vocab_size × embed_dim
    - Context matrix: vocab_size × embed_dim (separate output embeddings)
    - Skip-gram objective with negative sampling
    - Subsampling of frequent words (Mikolov et al. 2013)

Usage:
    python train_v28.py --fresh
    python train_v28.py --resume
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
import queue
import threading
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_DIM = 300          # Classic word2vec dimension
WINDOW_SIZE = 5          # Context window (each side)
NEG_SAMPLES = 15         # Negative samples per positive pair
BATCH_SIZE = 4096        # Large batches work well for word2vec
TOTAL_STEPS = 500_000
PEAK_LR = 0.025          # Word2vec uses higher LR than transformers
MIN_LR = 1e-4
WARMUP_STEPS = 1000
SUBSAMPLE_THRESHOLD = 1e-4  # Subsampling threshold for frequent words
LOG_EVERY = 50
EVAL_EVERY = 5000
SAVE_EVERY = 10000
MIN_COUNT = 10           # Minimum word frequency to include in vocab
VOCAB_SAMPLE_DOCS = 200_000  # Documents to sample for building vocab
MAX_VOCAB = 100_000      # Maximum vocabulary size

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints/word2vec_v28"
LOG_DIR = "logs"
LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v28.log",
    "metrics": f"{LOG_DIR}/concept_v28_metrics.csv",
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
# Whole-word vocabulary
# ---------------------------------------------------------------------------

# Simple tokenization: lowercase, split on non-alpha, keep words 1-30 chars
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

def tokenize_text(text):
    """Simple whole-word tokenizer. Lowercase, split on non-alpha."""
    return _WORD_RE.findall(text.lower())


class Vocabulary:
    """Whole-word vocabulary with frequency counts."""

    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.counts = []  # frequency count per id
        self.total_count = 0

    def build_from_sources(self, sources, num_docs=VOCAB_SAMPLE_DOCS, min_count=MIN_COUNT,
                           max_vocab=MAX_VOCAB):
        """Build vocabulary by scanning documents from data sources."""
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
            # Sample source
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

        # Filter by min_count, cap at max_vocab
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
        # Handle both list and dict formats
        if isinstance(data["counts"], dict):
            self.counts = [data["counts"][str(i)] for i in range(len(self.word2id))]
        else:
            self.counts = data["counts"]
        self.total_count = data["total_count"]
        log(f"  Vocabulary loaded: {len(self.word2id):,} words from {path}")

    def encode(self, words):
        """Convert word list to id list, skipping unknown words."""
        return [self.word2id[w] for w in words if w in self.word2id]

    def decode(self, ids):
        """Convert id list to word list."""
        return [self.id2word.get(i, "<unk>") for i in ids]

    def __len__(self):
        return len(self.word2id)

    def __contains__(self, word):
        return word in self.word2id


# ---------------------------------------------------------------------------
# Word2vec Model
# ---------------------------------------------------------------------------

class SkipGram(nn.Module):
    """Skip-gram with negative sampling."""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)

        # Init: small uniform (Mikolov style)
        initrange = 0.5 / embed_dim
        nn.init.uniform_(self.target_embeddings.weight, -initrange, initrange)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, target_ids, context_ids, neg_ids):
        """
        target_ids:  (batch,)
        context_ids: (batch,)
        neg_ids:     (batch, num_neg)

        Returns scalar loss.
        """
        target_emb = self.target_embeddings(target_ids)     # (batch, dim)
        context_emb = self.context_embeddings(context_ids)   # (batch, dim)
        neg_emb = self.context_embeddings(neg_ids)           # (batch, num_neg, dim)

        # Positive: log σ(u_c · v_t)
        pos_score = (target_emb * context_emb).sum(dim=-1)
        pos_loss = F.logsigmoid(pos_score)

        # Negative: Σ log σ(-u_n · v_t)
        neg_score = torch.bmm(neg_emb, target_emb.unsqueeze(-1)).squeeze(-1)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=-1)

        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_embeddings(self):
        """Return the target embeddings."""
        return self.target_embeddings.weight.detach()


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

class WordPairDataset:
    """Streams (target, context) skip-gram pairs using whole-word vocabulary."""

    def __init__(self, sources, vocab, window_size=WINDOW_SIZE,
                 subsample_threshold=SUBSAMPLE_THRESHOLD):
        self.sources = [(p, w) for p, w in sources if os.path.exists(p)]
        if not self.sources:
            raise FileNotFoundError("No data sources found")
        self.vocab = vocab
        self.window_size = window_size
        self.subsample_threshold = subsample_threshold

        # Source sampling weights
        total_w = sum(w for _, w in self.sources)
        self._cum_weights = []
        cum = 0
        for _, w in self.sources:
            cum += w / total_w
            self._cum_weights.append(cum)
        self._files = [open(p) for p, _ in self.sources]

        # Precompute subsampling probabilities
        self._subsample_probs = np.ones(len(vocab), dtype=np.float32)
        for i, count in enumerate(vocab.counts):
            freq = count / max(vocab.total_count, 1)
            if freq > 0:
                self._subsample_probs[i] = min(1.0,
                    (math.sqrt(freq / subsample_threshold) + 1) *
                    (subsample_threshold / freq))

        # Negative sampling table: unigram^0.75 (Mikolov et al.)
        freqs = np.array(vocab.counts, dtype=np.float64)
        freqs = np.power(freqs, 0.75)
        self._neg_probs = freqs / freqs.sum()
        self._neg_table_size = 10_000_000
        self._neg_table = torch.from_numpy(
            np.random.choice(len(vocab), size=self._neg_table_size, p=self._neg_probs)
        ).long()
        self._neg_idx = 0

        # Pair buffer
        self._pair_buf = []
        self._MIN_BUF = 100000

        log(f"WordPairDataset: {len(self.sources)} sources, "
            f"vocab={len(vocab):,}, window={window_size}")

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
        """Tokenize to whole words, convert to IDs, subsample frequent words."""
        words = tokenize_text(text)
        ids = self.vocab.encode(words)
        # Subsample
        return [i for i in ids if random.random() < self._subsample_probs[i]]

    def _extract_pairs(self, token_ids):
        """Extract (target, context) skip-gram pairs."""
        pairs = []
        n = len(token_ids)
        for i in range(n):
            # Dynamic window: uniform from 1 to window_size
            w = random.randint(1, self.window_size)
            for j in range(max(0, i - w), min(n, i + w + 1)):
                if j != i:
                    pairs.append((token_ids[i], token_ids[j]))
        return pairs

    def _refill_buffer(self):
        attempts = 0
        while len(self._pair_buf) < self._MIN_BUF and attempts < 20000:
            attempts += 1
            src = self._sample_source()
            text = self._read_doc(src)
            if not text or len(text) < 30:
                continue
            ids = self._tokenize_and_subsample(text)
            if len(ids) < 3:
                continue
            self._pair_buf.extend(self._extract_pairs(ids))
        random.shuffle(self._pair_buf)

    def sample_negatives(self, batch_size, num_neg):
        """Sample negatives from unigram^0.75 distribution."""
        total_needed = batch_size * num_neg
        if self._neg_idx + total_needed > self._neg_table_size:
            self._neg_idx = 0
        neg = self._neg_table[self._neg_idx:self._neg_idx + total_needed]
        self._neg_idx += total_needed
        return neg.view(batch_size, num_neg)

    def get_batch(self, batch_size):
        while len(self._pair_buf) < batch_size:
            self._refill_buffer()
        pairs = [self._pair_buf.pop() for _ in range(batch_size)]
        targets = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        contexts = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        negatives = self.sample_negatives(batch_size, NEG_SAMPLES)
        return targets, contexts, negatives


class PrefetchBuffer:
    def __init__(self, dataset, batch_size=BATCH_SIZE, buf_size=8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.q = queue.Queue(maxsize=buf_size)
        self._stop = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._fill, daemon=True)
        self._thread.start()

    def _fill(self):
        while not self._stop.is_set():
            try:
                batch = self.dataset.get_batch(self.batch_size)
                self.q.put(batch, timeout=1.0)
            except Exception:
                continue

    def get(self):
        return self.q.get()

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# Evaluation
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
    """Run analogy, similarity, and direction evaluation."""
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
        results = []
        for idx, sim in zip(topk.indices, topk.values):
            word = vocab.id2word.get(idx.item(), "?")
            results.append((word, sim.item()))
        return results

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

    # Nearest neighbors for key words
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
    log(f"Resumed V28: step {step}")
    return step


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step):
    """Linear warmup, then cosine decay."""
    if step < WARMUP_STEPS:
        return PEAK_LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(TOTAL_STEPS - WARMUP_STEPS, 1)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    log("=" * 70)
    log("FLM V28 — Word2vec Skip-Gram (Whole-Word Vocabulary)")
    log("=" * 70)

    # Build or load vocabulary
    vocab = Vocabulary()
    if args.fresh or not os.path.exists(VOCAB_PATH):
        vocab.build_from_sources(PRETRAIN_SOURCES)
        vocab.save(VOCAB_PATH)
    else:
        vocab.load(VOCAB_PATH)

    vocab_size = len(vocab)
    log(f"  Embed dim: {EMBED_DIM}")
    log(f"  Vocab size: {vocab_size:,}")
    log(f"  Window: {WINDOW_SIZE}, Neg samples: {NEG_SAMPLES}")
    log(f"  Batch size: {BATCH_SIZE}")
    log(f"  Total steps: {TOTAL_STEPS:,}")

    # Model
    model = SkipGram(vocab_size, EMBED_DIM).to(DEVICE)
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
    dataset = WordPairDataset(PRETRAIN_SOURCES, vocab)
    prefetch = PrefetchBuffer(dataset, BATCH_SIZE)
    prefetch.start()
    log("Prefetch started")
    log("-" * 70)

    # Training
    model.train()
    running_loss = 0.0
    loss_count = 0
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

        targets, contexts, negatives = prefetch.get()
        targets = targets.to(DEVICE)
        contexts = contexts.to(DEVICE)
        negatives = negatives.to(DEVICE)

        loss = model(targets, contexts, negatives)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loss_count += 1

        if (step + 1) % LOG_EVERY == 0:
            avg_loss = running_loss / loss_count
            elapsed = time.time() - start_time
            sps = (step + 1 - start_step) / max(elapsed, 1)
            pct = (step + 1) / TOTAL_STEPS * 100

            log(f"step {step+1:>7d} [V28] | loss={avg_loss:.4f} | "
                f"lr {current_lr:.2e} | {pct:.1f}% | {sps:.1f} step/s")

            log_metrics(step + 1, {
                "loss": avg_loss,
                "lr": current_lr,
            })

            running_loss = 0.0
            loss_count = 0

        if (step + 1) % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                evaluate_embeddings(model, vocab, step + 1)
            model.train()

        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, step + 1)

    if not stop_flag[0]:
        save_checkpoint(model, optimizer, TOTAL_STEPS)

    prefetch.stop()
    log("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)
