#!/usr/bin/env python3
"""
FLM V29 — 3D Word2vec Skip-Gram

Same as V28 but with EMBED_DIM=3. Every word lives at a real 3D coordinate.
No dimensionality reduction needed — the embedding IS the visualization.

The question: how much semantic structure can 3 dimensions capture?

Architecture:
    - Embedding matrix: vocab_size × 3
    - Context matrix: vocab_size × 3 (separate output embeddings)
    - Skip-gram objective with negative sampling
    - Subsampling of frequent words (Mikolov et al. 2013)

Reuses V28's 100K whole-word vocabulary.

Usage:
    python train_v29.py --fresh
    python train_v29.py --resume
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

EMBED_DIM = 3              # The whole point: true 3D embeddings
WINDOW_SIZE = 5
NEG_SAMPLES = 15
BATCH_SIZE = 4096
TOTAL_STEPS = 500_000
PEAK_LR = 0.025
MIN_LR = 1e-4
WARMUP_STEPS = 1000
SUBSAMPLE_THRESHOLD = 1e-4
LOG_EVERY = 50
EVAL_EVERY = 5000
SAVE_EVERY = 10000
MIN_COUNT = 10
MAX_VOCAB = 100_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints/word2vec_v29"
LOG_DIR = "logs"
LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v29.log",
    "metrics": f"{LOG_DIR}/concept_v29_metrics.csv",
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

# Reuse V28's vocabulary
VOCAB_PATH = "checkpoints/word2vec_v28/vocab.json"

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
# Whole-word vocabulary (same as V28)
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

    def __len__(self):
        return len(self.word2id)

    def __contains__(self, word):
        return word in self.word2id


# ---------------------------------------------------------------------------
# Word2vec Model
# ---------------------------------------------------------------------------

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)

        # With only 3 dims, use larger init range
        initrange = 0.5 / embed_dim
        nn.init.uniform_(self.target_embeddings.weight, -initrange, initrange)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, target_ids, context_ids, neg_ids):
        target_emb = self.target_embeddings(target_ids)
        context_emb = self.context_embeddings(context_ids)
        neg_emb = self.context_embeddings(neg_ids)

        pos_score = (target_emb * context_emb).sum(dim=-1)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(neg_emb, target_emb.unsqueeze(-1)).squeeze(-1)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=-1)

        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_embeddings(self):
        return self.target_embeddings.weight.detach()


# ---------------------------------------------------------------------------
# Data Pipeline (same as V28)
# ---------------------------------------------------------------------------

class WordPairDataset:
    def __init__(self, sources, vocab, window_size=WINDOW_SIZE,
                 subsample_threshold=SUBSAMPLE_THRESHOLD):
        self.sources = [(p, w) for p, w in sources if os.path.exists(p)]
        if not self.sources:
            raise FileNotFoundError("No data sources found")
        self.vocab = vocab
        self.window_size = window_size
        self.subsample_threshold = subsample_threshold

        total_w = sum(w for _, w in self.sources)
        self._cum_weights = []
        cum = 0
        for _, w in self.sources:
            cum += w / total_w
            self._cum_weights.append(cum)
        self._files = [open(p) for p, _ in self.sources]

        self._subsample_probs = np.ones(len(vocab), dtype=np.float32)
        for i, count in enumerate(vocab.counts):
            freq = count / max(vocab.total_count, 1)
            if freq > 0:
                self._subsample_probs[i] = min(1.0,
                    (math.sqrt(freq / subsample_threshold) + 1) *
                    (subsample_threshold / freq))

        freqs = np.array(vocab.counts, dtype=np.float64)
        freqs = np.power(freqs, 0.75)
        self._neg_probs = freqs / freqs.sum()
        self._neg_table_size = 10_000_000
        self._neg_table = torch.from_numpy(
            np.random.choice(len(vocab), size=self._neg_table_size, p=self._neg_probs)
        ).long()
        self._neg_idx = 0

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
        words = tokenize_text(text)
        ids = self.vocab.encode(words)
        return [i for i in ids if random.random() < self._subsample_probs[i]]

    def _extract_pairs(self, token_ids):
        pairs = []
        n = len(token_ids)
        for i in range(n):
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
    log(f"Resumed V29: step {step}")
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

def train(args):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    log("=" * 70)
    log("FLM V29 — 3D Word2vec Skip-Gram")
    log("=" * 70)

    # Load vocabulary (reuse V28's)
    vocab = Vocabulary()
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"V28 vocab not found: {VOCAB_PATH}. Train V28 first.")
    vocab.load(VOCAB_PATH)

    vocab_size = len(vocab)
    log(f"  Embed dim: {EMBED_DIM} (TRUE 3D)")
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

            log(f"step {step+1:>7d} [V29] | loss={avg_loss:.4f} | "
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
