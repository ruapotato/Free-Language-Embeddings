#!/usr/bin/env python3
"""
FLM V31 — Phrase-level Word2vec

Aggressive word-level BPE: start with whole words, keep merging the most
frequent adjacent pairs until sentences are ~3 tokens long. Then train
word2vec skip-gram on the phrase tokens.

The idea: if "the king sat on" is a single token and "his throne" is another,
the skip-gram vectors should capture sentence-level meaning while retaining
the geometric properties that word2vec is known for.

Pipeline:
    Phase 1: Build phrase vocabulary via word-level BPE
    Phase 2: Train skip-gram on phrase tokens (300d)

Usage:
    python train_v31.py --build-vocab          # Phase 1: build phrase vocab
    python train_v31.py --fresh                # Phase 2: train (builds vocab if needed)
    python train_v31.py --resume               # Resume training
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
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Phase 1: Vocabulary
VOCAB_SAMPLE_DOCS = 20_000        # Documents to sample for building phrase vocab
MAX_SENTENCES = 200_000           # Cap total sentences to control memory
TARGET_AVG_TOKENS = 3.5           # Target average tokens per sentence
MAX_VOCAB = 500_000               # Maximum vocabulary size
MIN_PHRASE_COUNT = 3              # Minimum frequency to keep a phrase
MERGE_BATCH_SIZE = 500            # Merges per round before recount

# Phase 2: Training
EMBED_DIM = 300
WINDOW_SIZE = 3                   # Smaller window since tokens are phrases
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints/phrase_v31"
LOG_DIR = "logs"
LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v31.log",
    "metrics": f"{LOG_DIR}/concept_v31_metrics.csv",
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

VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "phrase_vocab.json")

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
# Phase 1: Build phrase vocabulary via word-level BPE
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

def tokenize_to_words(text):
    return _WORD_RE.findall(text.lower())


def split_sentences(text):
    """Split text into sentences (simple split on .!? followed by space/newline)."""
    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 5]


def load_sentences(sources, num_docs):
    """Load sentences from data sources."""
    available = [(p, w) for p, w in sources if os.path.exists(p)]
    total_w = sum(w for _, w in available)
    cum_weights = []
    cum = 0
    for _, w in available:
        cum += w / total_w
        cum_weights.append(cum)
    files = [open(p) for p, _ in available]

    sentences = []
    docs_read = 0

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

        if len(text) < 30:
            continue

        for sent in split_sentences(text):
            words = tokenize_to_words(sent)
            if 3 <= len(words) <= 30:
                sentences.append(words)
                if len(sentences) >= MAX_SENTENCES:
                    break

        docs_read += 1
        if len(sentences) >= MAX_SENTENCES:
            break
        if docs_read % 10000 == 0:
            log(f"  Loaded {docs_read:,} docs, {len(sentences):,} sentences")

    for f in files:
        f.close()

    return sentences


def build_phrase_vocab(sources):
    """Build phrase vocabulary by iterative BPE merging on word sequences."""
    log("=" * 70)
    log("Phase 1: Building Phrase Vocabulary")
    log("=" * 70)

    # Load sentences as word sequences
    log(f"Loading sentences from {VOCAB_SAMPLE_DOCS:,} documents...")
    sentences = load_sentences(sources, VOCAB_SAMPLE_DOCS)
    log(f"  Total sentences: {len(sentences):,}")

    # Initial word frequencies
    word_counts = Counter()
    for sent in sentences:
        word_counts.update(sent)
    log(f"  Unique words: {len(word_counts):,}")

    # Filter rare words — replace with <unk>
    min_word_count = 5
    valid_words = {w for w, c in word_counts.items() if c >= min_word_count}
    log(f"  Words with count >= {min_word_count}: {len(valid_words):,}")

    # Convert sentences to token sequences (words as initial tokens)
    # Use string tokens — phrases will be "word1 word2 word3" joined by spaces
    corpus = []
    for sent in sentences:
        tokens = [w if w in valid_words else "<unk>" for w in sent]
        corpus.append(tokens)

    # Base vocabulary: all valid words
    token2id = {"<unk>": 0}
    for w in sorted(valid_words):
        token2id[w] = len(token2id)

    base_vocab_size = len(token2id)
    log(f"  Base vocabulary: {base_vocab_size:,} words")

    # Compute initial stats
    total_tokens = sum(len(s) for s in corpus)
    avg_tokens = total_tokens / len(corpus)
    log(f"  Initial avg tokens/sentence: {avg_tokens:.1f}")
    log(f"  Total tokens: {total_tokens:,}")
    log(f"  Target avg tokens/sentence: {TARGET_AVG_TOKENS}")

    # BPE merge loop — apply top merge, rescan, repeat
    merge_rules = []
    merge_num = 0

    while avg_tokens > TARGET_AVG_TOKENS and len(token2id) < MAX_VOCAB:
        # Count all adjacent pairs
        pair_counts = Counter()
        for sent in corpus:
            for i in range(len(sent) - 1):
                pair_counts[(sent[i], sent[i+1])] += 1

        if not pair_counts:
            break

        # Get THE single most frequent pair
        (tok_a, tok_b), count = pair_counts.most_common(1)[0]

        if count < MIN_PHRASE_COUNT:
            log(f"  Top pair count={count} < {MIN_PHRASE_COUNT}, stopping")
            break

        # Create merged token
        merged = f"{tok_a} {tok_b}"
        if merged not in token2id:
            token2id[merged] = len(token2id)
        merge_rules.append((tok_a, tok_b, merged))
        merge_num += 1

        # Apply this single merge across entire corpus
        for si in range(len(corpus)):
            sent = corpus[si]
            if len(sent) < 2:
                continue
            new_sent = []
            i = 0
            while i < len(sent):
                if i < len(sent) - 1 and sent[i] == tok_a and sent[i+1] == tok_b:
                    new_sent.append(merged)
                    i += 2
                else:
                    new_sent.append(sent[i])
                    i += 1
            corpus[si] = new_sent

        total_tokens = sum(len(s) for s in corpus)
        avg_tokens = total_tokens / len(corpus)

        if merge_num % 1000 == 0 or merge_num <= 10 or avg_tokens <= TARGET_AVG_TOKENS + 0.5:
            word_count = len(merged.split())
            log(f"  Merge {merge_num:,}: vocab={len(token2id):,} | "
                f"avg_tokens={avg_tokens:.2f} | count={count:,} | "
                f"[{word_count}w] \"{merged}\"")

    log(f"\n  Final vocabulary: {len(token2id):,} tokens")
    log(f"  Final avg tokens/sentence: {avg_tokens:.2f}")
    log(f"  Total merges: {len(merge_rules):,}")
    log(f"  Base words: {base_vocab_size:,}")
    log(f"  Phrase tokens: {len(token2id) - base_vocab_size:,}")

    # Count token frequencies in final corpus
    token_counts = Counter()
    for sent in corpus:
        token_counts.update(sent)

    # Show longest phrases
    phrases_by_len = sorted(token2id.keys(), key=lambda t: -len(t.split()))
    log(f"\n  Longest phrases:")
    for p in phrases_by_len[:20]:
        wc = len(p.split())
        freq = token_counts.get(p, 0)
        log(f"    [{wc} words, freq={freq:,}] \"{p}\"")

    # Show some example sentence tokenizations
    log(f"\n  Example tokenizations:")
    for sent in corpus[:15]:
        original_words = sum(len(t.split()) for t in sent)
        tokens_str = " | ".join(sent)
        log(f"    [{len(sent)} tokens, {original_words} words] {tokens_str}")

    # Save vocabulary
    id2token = {i: t for t, i in token2id.items()}
    counts_list = [token_counts.get(id2token[i], 0) for i in range(len(token2id))]
    total_count = sum(counts_list)

    vocab_data = {
        "token2id": token2id,
        "counts": counts_list,
        "total_count": total_count,
        "merge_rules": merge_rules,
        "base_vocab_size": base_vocab_size,
        "avg_tokens_per_sentence": avg_tokens,
        "num_sentences_sampled": len(sentences),
    }

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab_data, f)
    log(f"\n  Vocabulary saved: {VOCAB_PATH}")

    return token2id, counts_list, total_count, merge_rules


# ---------------------------------------------------------------------------
# Phrase tokenizer (apply merge rules to new text)
# ---------------------------------------------------------------------------

class PhraseTokenizer:
    """Tokenize text using learned BPE merge rules."""

    def __init__(self, token2id, merge_rules):
        self.token2id = token2id
        self.id2token = {i: t for t, i in token2id.items()}
        self.merge_rules = merge_rules
        # Build merge priority lookup
        self._merge_map = {}
        for priority, (a, b, merged) in enumerate(merge_rules):
            self._merge_map[(a, b)] = (merged, priority)

    def tokenize(self, text):
        """Tokenize text to phrase token IDs."""
        words = tokenize_to_words(text)
        tokens = [w if w in self.token2id else "<unk>" for w in words]

        # Apply merge rules greedily (highest frequency merges first,
        # which is the order they were learned)
        for tok_a, tok_b, merged in self.merge_rules:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == tok_a and tokens[i+1] == tok_b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self.token2id[t] for t in tokens if t in self.token2id]

    def decode(self, ids):
        """Convert token IDs back to text."""
        tokens = [self.id2token.get(i, "<unk>") for i in ids]
        return " ".join(tokens)

    def __len__(self):
        return len(self.token2id)


# ---------------------------------------------------------------------------
# Word2vec Model (same as V28)
# ---------------------------------------------------------------------------

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)
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
# Data Pipeline (adapted for phrase tokens)
# ---------------------------------------------------------------------------

class PhrasePairDataset:
    """Streams (target, context) skip-gram pairs from phrase-tokenized text."""

    def __init__(self, sources, tokenizer, counts, total_count,
                 window_size=WINDOW_SIZE, subsample_threshold=SUBSAMPLE_THRESHOLD):
        self.sources = [(p, w) for p, w in sources if os.path.exists(p)]
        if not self.sources:
            raise FileNotFoundError("No data sources found")
        self.tokenizer = tokenizer
        self.window_size = window_size

        total_w = sum(w for _, w in self.sources)
        self._cum_weights = []
        cum = 0
        for _, w in self.sources:
            cum += w / total_w
            self._cum_weights.append(cum)
        self._files = [open(p) for p, _ in self.sources]

        # Subsampling
        vocab_size = len(tokenizer)
        self._subsample_probs = np.ones(vocab_size, dtype=np.float32)
        for i, count in enumerate(counts):
            freq = count / max(total_count, 1)
            if freq > 0:
                self._subsample_probs[i] = min(1.0,
                    (math.sqrt(freq / subsample_threshold) + 1) *
                    (subsample_threshold / freq))

        # Negative sampling table
        freqs = np.array(counts, dtype=np.float64) + 1  # add 1 to avoid zero
        freqs = np.power(freqs, 0.75)
        self._neg_probs = freqs / freqs.sum()
        self._neg_table_size = 10_000_000
        self._neg_table = torch.from_numpy(
            np.random.choice(vocab_size, size=self._neg_table_size, p=self._neg_probs)
        ).long()
        self._neg_idx = 0

        self._pair_buf = []
        self._MIN_BUF = 100000

        log(f"PhrasePairDataset: {len(self.sources)} sources, "
            f"vocab={vocab_size:,}, window={window_size}")

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
            ids = self.tokenizer.tokenize(text)
            # Subsample
            ids = [i for i in ids if random.random() < self._subsample_probs[i]]
            if len(ids) < 2:
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

def evaluate_embeddings(model, tokenizer, step):
    """Basic evaluation: nearest neighbors for phrases, similarity tests."""
    emb = model.get_embeddings().to("cpu")
    emb_norm = F.normalize(emb, p=2, dim=-1)

    def get_vec(token):
        if token in tokenizer.token2id:
            return emb_norm[tokenizer.token2id[token]]
        return None

    def nearest(vec, exclude_ids=None, k=8):
        sims = emb_norm @ vec
        if exclude_ids:
            for idx in exclude_ids:
                sims[idx] = -1
        topk = sims.topk(k)
        return [(tokenizer.id2token[i.item()], s.item()) for i, s in zip(topk.indices, topk.values)]

    # Nearest neighbors for sample tokens
    log(f"  --- Nearest neighbors ---")
    sample_tokens = []
    # Find some interesting phrase tokens
    for token, tid in sorted(tokenizer.token2id.items(), key=lambda x: -len(x[0].split()))[:5]:
        sample_tokens.append(token)
    # Add some word tokens
    for word in ["king", "computer", "happy", "france", "water"]:
        if word in tokenizer.token2id:
            sample_tokens.append(word)

    for token in sample_tokens[:10]:
        v = get_vec(token)
        if v is not None:
            top = nearest(v, exclude_ids=[tokenizer.token2id[token]], k=6)
            nns = ", ".join(f'"{w}"({s:.2f})' for w, s in top)
            log(f"    \"{token}\" → {nns}")

    # Similarity between related phrases
    log(f"  --- Phrase similarities ---")
    test_pairs = [
        ("the", "a"),
        ("king", "queen"),
        ("happy", "sad"),
    ]
    for t1, t2 in test_pairs:
        v1, v2 = get_vec(t1), get_vec(t2)
        if v1 is not None and v2 is not None:
            sim = (v1 * v2).sum().item()
            log(f"    \"{t1}\" ↔ \"{t2}\": {sim:.3f}")


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
    log(f"Resumed V31: step {step}")
    return step


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
    log("FLM V31 — Phrase-level Word2vec (Aggressive BPE)")
    log("=" * 70)

    # Phase 1: Build or load vocabulary
    if args.build_vocab or (args.fresh and not os.path.exists(VOCAB_PATH)):
        token2id, counts, total_count, merge_rules = build_phrase_vocab(PRETRAIN_SOURCES)
    else:
        log("Loading phrase vocabulary...")
        with open(VOCAB_PATH) as f:
            vdata = json.load(f)
        token2id = vdata["token2id"]
        counts = vdata["counts"]
        total_count = vdata["total_count"]
        merge_rules = [tuple(m) for m in vdata["merge_rules"]]
        log(f"  Vocabulary: {len(token2id):,} tokens ({vdata.get('base_vocab_size', '?')} base words)")
        log(f"  Avg tokens/sentence: {vdata.get('avg_tokens_per_sentence', '?'):.2f}")

    if args.build_vocab:
        return  # Just building vocab, done

    # Phase 2: Training
    log("\n" + "=" * 70)
    log("Phase 2: Training Skip-Gram")
    log("=" * 70)

    tokenizer = PhraseTokenizer(token2id, merge_rules)
    vocab_size = len(tokenizer)

    log(f"  Vocab size: {vocab_size:,}")
    log(f"  Embed dim: {EMBED_DIM}")
    log(f"  Window: {WINDOW_SIZE}, Neg samples: {NEG_SAMPLES}")
    log(f"  Batch size: {BATCH_SIZE}")

    # Model
    model = SkipGram(vocab_size, EMBED_DIM).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    log(f"  Parameters: {params:,} ({params/1e6:.1f}M)")

    optimizer = torch.optim.SparseAdam(model.parameters(), lr=PEAK_LR)

    start_step = 0
    if not args.fresh:
        start_step = load_checkpoint(model, optimizer)

    log(f"  LR: {PEAK_LR} → {MIN_LR} (cosine) | Steps: {start_step} → {TOTAL_STEPS}")

    # Data
    dataset = PhrasePairDataset(PRETRAIN_SOURCES, tokenizer, counts, total_count)
    prefetch = PrefetchBuffer(dataset, BATCH_SIZE)
    prefetch.start()
    log("Prefetch started")
    log("-" * 70)

    # Training loop
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

            log(f"step {step+1:>7d} [V31] | loss={avg_loss:.4f} | "
                f"lr {current_lr:.2e} | {pct:.1f}% | {sps:.1f} step/s")

            log_metrics(step + 1, {"loss": avg_loss, "lr": current_lr})
            running_loss = 0.0
            loss_count = 0

        if (step + 1) % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                evaluate_embeddings(model, tokenizer, step + 1)
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
    parser.add_argument("--build-vocab", action="store_true")
    args = parser.parse_args()
    train(args)
