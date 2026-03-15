#!/usr/bin/env python3
"""
FLM V33 — Mixed Skip-Gram + CBOW Word2vec

Combines both word2vec training objectives on the same embedding matrix:
  - Skip-gram: given a center word, predict each context word
  - CBOW: given all context words (averaged), predict the center word

Two different views of the same co-occurrence data, both shaping the
same embeddings. Similar to how V25's joint reconstruction + prediction
gave two training signals.

Changes from V28:
  - Mixed skip-gram + CBOW loss (configurable weight)
  - More training data (500K docs for vocab)
  - More training steps (1M)
  - Same 300d, 100K vocab, same eval for A/B comparison

Usage:
    python train_v33.py --fresh
    python train_v33.py --resume
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

EMBED_DIM = 300          # Same as V28 for A/B comparison
WINDOW_SIZE = 5          # Context window (each side)
NEG_SAMPLES = 15         # Negative samples per positive pair
BATCH_SIZE = 4096
TOTAL_STEPS = 1_000_000  # 2x V28
PEAK_LR = 0.025
MIN_LR = 1e-4
WARMUP_STEPS = 2000
SUBSAMPLE_THRESHOLD = 1e-4
LOG_EVERY = 50
EVAL_EVERY = 5000
SAVE_EVERY = 10000
MIN_COUNT = 10
VOCAB_SAMPLE_DOCS = 200_000  # Same as V28
MAX_VOCAB = 100_000

# Loss mixing: total_loss = SG_WEIGHT * sg_loss + CBOW_WEIGHT * cbow_loss
SG_WEIGHT = 0.5
CBOW_WEIGHT = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints/word2vec_v33"
LOG_DIR = "logs"
LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v33.log",
    "metrics": f"{LOG_DIR}/concept_v33_metrics.csv",
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
# Whole-word vocabulary (identical to V28)
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
# Mixed Skip-Gram + CBOW Model
# ---------------------------------------------------------------------------

class MixedWord2vec(nn.Module):
    """Combined skip-gram and CBOW with shared embeddings.

    Both objectives share the same two embedding matrices:
      - target_embeddings: used as INPUT for both SG and CBOW
        (SG: looks up center word; CBOW: looks up + averages context words)
      - context_embeddings: used as OUTPUT for both
        (SG: looks up context word to predict; CBOW: looks up center word to predict)
    """

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)

        initrange = 0.5 / embed_dim
        nn.init.uniform_(self.target_embeddings.weight, -initrange, initrange)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward_skipgram(self, target_ids, context_ids, neg_ids):
        """Skip-gram: given center word, predict context word.

        target_ids:  (batch,)
        context_ids: (batch,)
        neg_ids:     (batch, num_neg)
        """
        target_emb = self.target_embeddings(target_ids)       # (batch, dim)
        context_emb = self.context_embeddings(context_ids)     # (batch, dim)
        neg_emb = self.context_embeddings(neg_ids)             # (batch, num_neg, dim)

        pos_score = (target_emb * context_emb).sum(dim=-1)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(neg_emb, target_emb.unsqueeze(-1)).squeeze(-1)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=-1)

        return -(pos_loss + neg_loss).mean()

    def forward_cbow(self, center_ids, context_ids, context_mask, neg_ids):
        """CBOW: given context words (averaged), predict center word.

        center_ids:   (batch,)
        context_ids:  (batch, max_ctx) — padded context word IDs
        context_mask: (batch, max_ctx) — 1 for real context, 0 for padding
        neg_ids:      (batch, num_neg)
        """
        # Average context word embeddings (using target_embeddings as input)
        ctx_emb = self.target_embeddings(context_ids)          # (batch, max_ctx, dim)
        # Mask out padding
        mask = context_mask.unsqueeze(-1).float()              # (batch, max_ctx, 1)
        ctx_sum = (ctx_emb * mask).sum(dim=1)                  # (batch, dim)
        ctx_count = mask.sum(dim=1).clamp(min=1)               # (batch, 1)
        ctx_avg = ctx_sum / ctx_count                          # (batch, dim)

        # Predict center word (using context_embeddings as output)
        center_emb = self.context_embeddings(center_ids)       # (batch, dim)
        neg_emb = self.context_embeddings(neg_ids)             # (batch, num_neg, dim)

        pos_score = (ctx_avg * center_emb).sum(dim=-1)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(neg_emb, ctx_avg.unsqueeze(-1)).squeeze(-1)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=-1)

        return -(pos_loss + neg_loss).mean()

    def get_embeddings(self):
        return self.target_embeddings.weight.detach()


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

class MixedDataset:
    """Streams both skip-gram pairs AND CBOW windows.

    Uses numpy ring buffers with random index sampling (no shuffle).
    Single-threaded — call from one prefetch thread only.
    """

    BUF_SIZE = 500_000  # Fixed ring buffer size

    def __init__(self, sources, vocab, window_size=WINDOW_SIZE,
                 subsample_threshold=SUBSAMPLE_THRESHOLD):
        self.sources = [(p, w) for p, w in sources if os.path.exists(p)]
        if not self.sources:
            raise FileNotFoundError("No data sources found")
        self.vocab = vocab
        self.window_size = window_size
        self.max_ctx = 2 * window_size

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
        self._neg_table = torch.from_numpy(
            np.random.choice(len(vocab), size=self._neg_table_size, p=self._neg_probs)
        ).long()
        self._neg_idx = 0

        # Numpy ring buffers — fixed size, overwrite oldest
        # SG: two arrays (targets, contexts)
        self._sg_targets = np.zeros(self.BUF_SIZE, dtype=np.int32)
        self._sg_contexts = np.zeros(self.BUF_SIZE, dtype=np.int32)
        self._sg_count = 0  # total items written (wraps via modulo)
        self._sg_valid = 0  # min(count, BUF_SIZE)

        # CBOW: center, context_ids (padded), context_lens
        self._cbow_centers = np.zeros(self.BUF_SIZE, dtype=np.int32)
        self._cbow_ctx = np.zeros((self.BUF_SIZE, self.max_ctx), dtype=np.int32)
        self._cbow_lens = np.zeros(self.BUF_SIZE, dtype=np.int32)
        self._cbow_count = 0
        self._cbow_valid = 0

        # Fill initial buffer
        self._fill_initial()

        log(f"MixedDataset: {len(self.sources)} sources, "
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
        """Extract windows and write directly into ring buffers."""
        n = len(token_ids)
        for i in range(n):
            w = random.randint(1, self.window_size)
            ctx_ids = []
            for j in range(max(0, i - w), min(n, i + w + 1)):
                if j != i:
                    # Write SG pair
                    idx = self._sg_count % self.BUF_SIZE
                    self._sg_targets[idx] = token_ids[i]
                    self._sg_contexts[idx] = token_ids[j]
                    self._sg_count += 1
                    ctx_ids.append(token_ids[j])

            if ctx_ids:
                # Write CBOW window
                idx = self._cbow_count % self.BUF_SIZE
                self._cbow_centers[idx] = token_ids[i]
                self._cbow_ctx[idx, :] = 0
                self._cbow_ctx[idx, :len(ctx_ids)] = ctx_ids
                self._cbow_lens[idx] = len(ctx_ids)
                self._cbow_count += 1

        self._sg_valid = min(self._sg_count, self.BUF_SIZE)
        self._cbow_valid = min(self._cbow_count, self.BUF_SIZE)

    def _fill_initial(self):
        """Fill buffers to at least half capacity."""
        target = self.BUF_SIZE // 2
        attempts = 0
        while self._sg_valid < target and attempts < 50000:
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
        """Read a few more documents into the ring buffer."""
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

    def sample_negatives(self, batch_size, num_neg):
        total_needed = batch_size * num_neg
        if self._neg_idx + total_needed > self._neg_table_size:
            self._neg_idx = 0
        neg = self._neg_table[self._neg_idx:self._neg_idx + total_needed]
        self._neg_idx += total_needed
        return neg.view(batch_size, num_neg)

    def start_reader(self):
        """Start background thread that continuously reads docs into ring buffer.

        Sleeps between batches to yield the GIL — Python threading means
        CPU-bound reader work blocks the main training thread otherwise.
        """
        self._reader_stop = threading.Event()
        def _read_loop():
            while not self._reader_stop.is_set():
                self._add_docs(10)
                # Yield GIL so training thread can run at full GPU speed
                time.sleep(0.05)
        self._reader_thread = threading.Thread(target=_read_loop, daemon=True)
        self._reader_thread.start()

    def stop_reader(self):
        if hasattr(self, '_reader_stop'):
            self._reader_stop.set()

    def get_sg_batch(self, batch_size):
        """Get a skip-gram batch via random index sampling. Instant — no I/O."""
        indices = np.random.randint(0, self._sg_valid, size=batch_size)
        targets = torch.from_numpy(self._sg_targets[indices].astype(np.int64))
        contexts = torch.from_numpy(self._sg_contexts[indices].astype(np.int64))
        negatives = self.sample_negatives(batch_size, NEG_SAMPLES)
        return targets, contexts, negatives

    def get_cbow_batch(self, batch_size):
        """Get a CBOW batch via random index sampling. Instant — no I/O."""
        indices = np.random.randint(0, self._cbow_valid, size=batch_size)
        centers = torch.from_numpy(self._cbow_centers[indices].astype(np.int64))
        context_ids = torch.from_numpy(self._cbow_ctx[indices].astype(np.int64))
        # Vectorized mask creation
        lens = self._cbow_lens[indices]
        arange = np.arange(self.max_ctx)
        context_mask = torch.from_numpy((arange[None, :] < lens[:, None]).astype(np.int64))
        negatives = self.sample_negatives(batch_size, NEG_SAMPLES)
        return centers, context_ids, context_mask, negatives



# ---------------------------------------------------------------------------
# Evaluation (identical to V28 for fair comparison)
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
# Google Analogy Benchmark (for final comparison with V28)
# ---------------------------------------------------------------------------

def run_google_analogies(model, vocab):
    """Run the full Google analogy benchmark for A/B comparison with V28."""
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

    # Semantic vs syntactic
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
    log(f"Resumed V33: step {step}")
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
    log("FLM V33 — Mixed Skip-Gram + CBOW Word2vec")
    log("=" * 70)
    log(f"  SG weight: {SG_WEIGHT}, CBOW weight: {CBOW_WEIGHT}")

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
    model = MixedWord2vec(vocab_size, EMBED_DIM).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    log(f"  Parameters: {params:,} ({params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=PEAK_LR)

    # Load checkpoint
    start_step = 0
    if not args.fresh:
        start_step = load_checkpoint(model, optimizer)

    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")

    # Data — reader thread fills ring buffer, batches sampled directly (instant)
    dataset = MixedDataset(PRETRAIN_SOURCES, vocab)
    dataset.start_reader()  # Background thread continuously reads docs into ring buffer
    log("Reader thread active, batches sampled directly (no prefetch queue)")
    log("-" * 70)

    # Training
    model.train()
    running_sg_loss = 0.0
    running_cbow_loss = 0.0
    sg_count = 0
    cbow_count = 0
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

        if step % 2 == 0:
            sg_targets, sg_contexts, sg_negatives = dataset.get_sg_batch(BATCH_SIZE)
            loss = model.forward_skipgram(
                sg_targets.to(DEVICE), sg_contexts.to(DEVICE), sg_negatives.to(DEVICE))
            running_sg_loss += loss.item()
            sg_count += 1
        else:
            cbow_centers, cbow_ctx_ids, cbow_ctx_mask, cbow_negatives = dataset.get_cbow_batch(BATCH_SIZE)
            loss = model.forward_cbow(
                cbow_centers.to(DEVICE), cbow_ctx_ids.to(DEVICE),
                cbow_ctx_mask.to(DEVICE), cbow_negatives.to(DEVICE))
            running_cbow_loss += loss.item()
            cbow_count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % LOG_EVERY == 0:
            avg_sg = running_sg_loss / max(sg_count, 1)
            avg_cbow = running_cbow_loss / max(cbow_count, 1)
            elapsed = time.time() - start_time
            sps = (step + 1 - start_step) / max(elapsed, 1)
            pct = (step + 1) / TOTAL_STEPS * 100

            log(f"step {step+1:>7d} [V33] | sg={avg_sg:.4f} cbow={avg_cbow:.4f} "
                f"| lr {current_lr:.2e} | {pct:.1f}% | {sps:.1f} step/s")

            log_metrics(step + 1, {
                "sg_loss": avg_sg,
                "cbow_loss": avg_cbow,
                "lr": current_lr,
            })

            running_sg_loss = 0.0
            running_cbow_loss = 0.0
            sg_count = 0
            cbow_count = 0

        if (step + 1) % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                evaluate_embeddings(model, vocab, step + 1)
            model.train()

        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, step + 1)

    if not stop_flag[0]:
        save_checkpoint(model, optimizer, TOTAL_STEPS)

    # Final Google benchmark
    log("")
    log("Running Google Analogy Benchmark (A/B vs V28's 43.9%)...")
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
