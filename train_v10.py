"""
FLM V10 — Non-Autoregressive Concept Autoencoder (Reconstruction First)
================================================================================
Key changes from V9:
  1. NON-AUTOREGRESSIVE DECODER: Learned position queries cross-attend to concept
     vectors. Each position independently predicts its token. No teacher forcing,
     no causal mask. Concept vectors are the ONLY information source.
  2. RECONSTRUCTION ONLY: No geometry losses until the model can reliably
     reconstruct inputs at ALL lengths. Zero NCE, zero word-order, zero slot
     isolation — just pure reconstruction CE.
  3. PER-LENGTH METRICS: Track token accuracy and exact-match per length bucket
     (short/medium/long) so we know reconstruction works everywhere.
  4. GEOMETRY GATE: Geometry losses unlock ONLY when exact-match EMA > 90%.

V9 post-mortem: recon loss was 0.15 but outputs were garbage. The autoregressive
decoder cheated by copying from teacher-forced tokens, barely using concept vectors.
Word-order sensitivity was 0.97 (nearly identical for "dog bit man" vs "man bit dog").
Reconstruction collapsed on anything >5 tokens.

Usage:
    python train_v10.py --fresh          # start from scratch
    python train_v10.py --resume         # resume from V10 checkpoint
    python train_v10.py --from-v9        # warm-start encoder from V9, fresh decoder
    python train_v10.py --eval-only      # diagnostics only
"""

import os
import sys
import json
import time
import math
import signal
import random
import datetime
import threading
import queue
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from concept_model import (ConceptConfig, ConceptAutoencoderV10,
                           reconstruction_loss, flat_info_nce_loss,
                           flat_word_order_info_nce, slot_decorrelation_loss,
                           slot_isolation_loss, flat_similarity_matrix,
                           margin_paraphrase_loss, margin_negative_loss,
                           margin_word_order_loss, batch_repulsion_loss,
                           hard_repulsion_loss)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v10"
LOG_DIR = "logs"

MODEL_CONFIG = dict(
    vocab_size=30522,  # BERT tokenizer
    enc_hidden=384,
    enc_layers=6,
    enc_heads=6,
    enc_intermediate=1536,
    num_concepts=32,
    concept_dim=32,
    dec_hidden=384,
    dec_layers=6,
    dec_heads=6,
    dec_intermediate=1536,
    max_seq_len=128,
    dropout=0.1,
)

# Training hyperparameters
BATCH_SIZE = 64           # larger since no pair/axis batches eating VRAM
PEAK_LR = 3e-4
WARMUP_STEPS = 2000
TOTAL_STEPS = 200_000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0

# Geometry gate — only unlock when reconstruction is solid
EXACT_MATCH_GATE = 0.90   # 90% exact-match on eval set before geometry
EXACT_MATCH_EMA_DECAY = 0.9   # fast-tracking: responds in ~10 evals not 200

# Phase 2: Geometry loss weights (activated after EM gate opens)
NCE_TEMPERATURE = 0.07
NCE_WEIGHT = 2.0
WO_WEIGHT = 1.5
DECORR_WEIGHT = 1.0
SLOT_ISO_WEIGHT = 1.0
MARGIN_PARA_WEIGHT = 5.0
MARGIN_NEG_WEIGHT = 3.0
MARGIN_WO_WEIGHT = 2.0
REPULSION_WEIGHT = 2.0
HARD_REPULSION_WEIGHT = 3.0
PAIR_BATCH_SIZE = 32          # smaller than recon to avoid OOM with extra fwd passes
AXIS_BATCH_SIZE = 32
GEO_WARMUP_STEPS = 5000       # linearly ramp geometry weight from 0→1 over this many steps

# Dynamic length sampling — oversample lengths the model is worst at
DYNAMIC_SAMPLING_ALPHA = 1.0   # exponent: linear reweighting (2.0 was too aggressive)
DYNAMIC_UPDATE_EVERY = 500     # re-compute weights at eval time

# Logging
LOG_EVERY = 50
EVAL_EVERY = 500
CHECKPOINT_EVERY = 5000

# Length buckets for per-length tracking
LENGTH_BUCKETS = {
    "short":  (1, 10),
    "medium": (11, 30),
    "long":   (31, 128),
}

LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v10.log",
    "metrics": f"{LOG_DIR}/concept_v10_metrics.csv",
}

# Diagnostic sentences for reconstruction spot-checks
RECON_TEST_SENTENCES = [
    "the cat sat on the mat",
    "the dog bit the man",
    "the man bit the dog",
    "she runs every morning before breakfast",
    "the purple man licked the sucker",
    "three big red cars drove quickly north",
    "artificial intelligence will change the world",
    "he certainly did not enjoy the terrible movie yesterday",
    "alice likes bob but bob does not like alice",
    "the quick brown fox jumps over the lazy dog near the river",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    log_file = LOG_PATHS.get("log")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")


def log_metrics(step, recon_loss, token_acc, exact_match,
                acc_short, acc_med, acc_long, em_short, em_med, em_long,
                lr, elapsed_hours):
    metrics_file = LOG_PATHS.get("metrics")
    if not metrics_file:
        return
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    write_header = not os.path.exists(metrics_file)
    with open(metrics_file, "a") as f:
        if write_header:
            f.write("timestamp,step,recon_loss,token_acc,exact_match,"
                    "acc_short,acc_med,acc_long,em_short,em_med,em_long,"
                    "lr,elapsed_hours\n")
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts},{step},{recon_loss:.6f},{token_acc:.4f},{exact_match:.4f},"
                f"{acc_short:.4f},{acc_med:.4f},{acc_long:.4f},"
                f"{em_short:.4f},{em_med:.4f},{em_long:.4f},"
                f"{lr:.6e},{elapsed_hours:.4f}\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class ReconstructionDataset:
    """Loads training texts, supports dynamic length-weighted sampling."""

    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = []
        # Length-bucketed storage for dynamic sampling
        self.buckets = {name: [] for name in LENGTH_BUCKETS}
        self.bucket_weights = {name: 1.0 for name in LENGTH_BUCKETS}
        self._load_texts()
        self.idx = 0

    def _load_texts(self):
        data_dir = Path("data/pairs")
        if not data_dir.exists():
            return
        for path in sorted(data_dir.glob("*.jsonl")):
            if path.name.startswith("eval_"):
                continue
            with open(path) as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        a = doc.get("text_a", "").strip()
                        b = doc.get("text_b", "").strip()
                        for text in [a, b]:
                            if len(text) > 10:
                                self.texts.append(text)
                                # Rough char→token estimate for bucketing
                                approx_tokens = len(text.split()) + 2
                                for bname, (lo, hi) in LENGTH_BUCKETS.items():
                                    if lo <= approx_tokens <= hi:
                                        self.buckets[bname].append(text)
                                        break
                                else:
                                    self.buckets["long"].append(text)
                    except (json.JSONDecodeError, KeyError):
                        continue
        for bname in self.buckets:
            random.shuffle(self.buckets[bname])
        random.shuffle(self.texts)
        log(f"  Reconstruction texts: {len(self.texts):,}")
        for bname in LENGTH_BUCKETS:
            log(f"    {bname}: {len(self.buckets[bname]):,}")

    def update_weights(self, bucket_em):
        """Update sampling weights based on per-bucket exact-match.

        Buckets with lower EM get higher weight (more sampling).
        weight ∝ (1 - em)^alpha, with a floor so good buckets still get data.
        """
        for bname in LENGTH_BUCKETS:
            em = bucket_em.get(bname, 0.0)
            # Higher alpha = more aggressive focus on weak buckets
            # Floor at 0.15 so solved buckets still get enough to maintain quality
            self.bucket_weights[bname] = max(0.15, (1.0 - em) ** DYNAMIC_SAMPLING_ALPHA)
        total = sum(self.bucket_weights.values())
        for bname in self.bucket_weights:
            self.bucket_weights[bname] /= total
        log(f"  Dynamic weights: " + " | ".join(
            f"{b}={self.bucket_weights[b]:.2f}" for b in LENGTH_BUCKETS))

    def get_batch(self, batch_size):
        """Sample a batch with dynamic length weighting."""
        texts = []
        bucket_names = list(LENGTH_BUCKETS.keys())
        weights = [self.bucket_weights[b] for b in bucket_names]

        for _ in range(batch_size):
            # Pick a bucket proportional to weights
            bname = random.choices(bucket_names, weights=weights, k=1)[0]
            bucket = self.buckets[bname]
            if bucket:
                texts.append(bucket[random.randint(0, len(bucket) - 1)])
            else:
                # Fallback to uniform
                texts.append(self.texts[random.randint(0, len(self.texts) - 1)])

        enc = self.tokenizer(texts, max_length=self.max_len,
                             padding=True, truncation=True,
                             return_tensors="pt")
        return enc


class PairDataset:
    """Loads paraphrase and hard negative pairs for geometry training."""

    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pos_pairs = []
        self.hard_neg_pairs = []
        self._load_pairs()
        random.shuffle(self.pos_pairs)
        random.shuffle(self.hard_neg_pairs)
        self.pos_idx = 0
        self.hn_idx = 0

    def _load_pairs(self):
        data_dir = Path("data/pairs")
        if not data_dir.exists():
            return
        for path in sorted(data_dir.glob("*.jsonl")):
            if path.name.startswith("eval_"):
                continue
            with open(path) as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        a = doc.get("text_a", "").strip()
                        b = doc.get("text_b", "").strip()
                        if not a or not b:
                            continue
                        label = doc.get("label", 1)
                        ptype = doc.get("type", "")
                        if label == 1:
                            self.pos_pairs.append((a, b))
                        elif ptype in ("contradiction", "hard_negative"):
                            self.hard_neg_pairs.append((a, b))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        log(f"  Positive pairs: {len(self.pos_pairs):,}")
        log(f"  Hard negative pairs: {len(self.hard_neg_pairs):,}")

    def _tokenize_pair(self, texts_a, texts_b):
        enc_a = self.tokenizer(texts_a, max_length=self.max_len,
                               padding=True, truncation=True, return_tensors="pt")
        enc_b = self.tokenizer(texts_b, max_length=self.max_len,
                               padding=True, truncation=True, return_tensors="pt")
        return enc_a, enc_b

    def get_pos_batch(self, batch_size):
        texts_a, texts_b = [], []
        for _ in range(batch_size):
            if self.pos_idx >= len(self.pos_pairs):
                random.shuffle(self.pos_pairs)
                self.pos_idx = 0
            a, b = self.pos_pairs[self.pos_idx]
            texts_a.append(a)
            texts_b.append(b)
            self.pos_idx += 1
        return self._tokenize_pair(texts_a, texts_b)

    def get_hn_batch(self, batch_size):
        if not self.hard_neg_pairs:
            return None, None
        texts_a, texts_b = [], []
        for _ in range(batch_size):
            if self.hn_idx >= len(self.hard_neg_pairs):
                random.shuffle(self.hard_neg_pairs)
                self.hn_idx = 0
            a, b = self.hard_neg_pairs[self.hn_idx]
            texts_a.append(a)
            texts_b.append(b)
            self.hn_idx += 1
        return self._tokenize_pair(texts_a, texts_b)


class ConceptAxisDataset:
    """Loads synthetic concept axis pairs for slot isolation training."""

    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs_by_slot = {i: [] for i in range(32)}
        self._load()
        self.indices = {i: 0 for i in range(32)}

    def _load(self):
        data_dir = Path("data/concept_axes")
        combined = data_dir / "all_axes.jsonl"
        if not combined.exists():
            log(f"  WARNING: No concept axis data at {combined}")
            return
        count = 0
        with open(combined) as f:
            for line in f:
                doc = json.loads(line)
                slot = doc["slot"]
                self.pairs_by_slot[slot].append((doc["base"], doc["variant"]))
                count += 1
        for slot_id in range(32):
            random.shuffle(self.pairs_by_slot[slot_id])
        self.active_slots = [s for s in range(32) if self.pairs_by_slot[s]]
        log(f"  Concept axis pairs: {count:,} across {len(self.active_slots)} slots")

    def get_batch(self, batch_size):
        bases, variants, slot_ids = [], [], []
        sampled_slots = random.choices(self.active_slots, k=batch_size)
        for slot in sampled_slots:
            pairs = self.pairs_by_slot[slot]
            idx = self.indices[slot]
            if idx >= len(pairs):
                random.shuffle(pairs)
                idx = 0
            base, var = pairs[idx]
            self.indices[slot] = idx + 1
            bases.append(base)
            variants.append(var)
            slot_ids.append(slot)
        enc_base = self.tokenizer(bases, max_length=self.max_len,
                                   padding=True, truncation=True, return_tensors="pt")
        enc_var = self.tokenizer(variants, max_length=self.max_len,
                                  padding=True, truncation=True, return_tensors="pt")
        return enc_base, enc_var, torch.tensor(slot_ids, dtype=torch.long)


def shuffle_word_order(input_ids, attention_mask, tokenizer):
    """Swap 2 random content tokens per sequence for word-order contrastive pairs.

    Returns shuffled input_ids (same shape). Skips special tokens [CLS], [SEP], [PAD].
    """
    shuffled = input_ids.clone()
    special = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    for i in range(shuffled.shape[0]):
        content_positions = [
            j for j in range(shuffled.shape[1])
            if attention_mask[i, j].item() == 1 and shuffled[i, j].item() not in special
        ]
        if len(content_positions) >= 2:
            a, b = random.sample(content_positions, 2)
            shuffled[i, a], shuffled[i, b] = shuffled[i, b].clone(), shuffled[i, a].clone()
    return shuffled


class PrefetchBuffer:
    """Pre-tokenizes batches in background thread to keep GPU fed."""

    def __init__(self, dataset, device, batch_size=64, buf_size=4):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.q = queue.Queue(maxsize=buf_size)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._fill, daemon=True,
                                        name="prefetch-recon")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _fill(self):
        while not self._stop.is_set():
            try:
                batch = self.dataset.get_batch(self.batch_size)
                self.q.put(batch, timeout=1.0)
            except queue.Full:
                continue

    def get(self):
        return self.q.get()


class PairPrefetchBuffer:
    """Background prefetch for pair/axis data (only used during Phase 2)."""

    def __init__(self, pair_dataset, axis_dataset, device,
                 pair_batch_size=80, axis_batch_size=80, buf_size=3):
        self.pair_ds = pair_dataset
        self.axis_ds = axis_dataset
        self.device = device
        self.pair_bs = pair_batch_size
        self.axis_bs = axis_batch_size
        self.pos_q = queue.Queue(maxsize=buf_size)
        self.hn_q = queue.Queue(maxsize=buf_size)
        self.axis_q = queue.Queue(maxsize=buf_size)
        self._stop = threading.Event()
        self._threads = []

    def start(self):
        for name, fn in [("prefetch-pos", self._fill_pos),
                          ("prefetch-hn", self._fill_hn),
                          ("prefetch-axis", self._fill_axis)]:
            t = threading.Thread(target=fn, daemon=True, name=name)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()

    def _fill_pos(self):
        while not self._stop.is_set():
            try:
                batch = self.pair_ds.get_pos_batch(self.pair_bs)
                self.pos_q.put(batch, timeout=1.0)
            except queue.Full:
                continue

    def _fill_hn(self):
        while not self._stop.is_set():
            try:
                batch = self.pair_ds.get_hn_batch(self.pair_bs)
                self.hn_q.put(batch, timeout=1.0)
            except queue.Full:
                continue

    def _fill_axis(self):
        if not self.axis_ds.active_slots:
            return
        while not self._stop.is_set():
            try:
                batch = self.axis_ds.get_batch(self.axis_bs)
                self.axis_q.put(batch, timeout=1.0)
            except queue.Full:
                continue

    def get_pos(self):
        return self.pos_q.get()

    def get_hn(self):
        return self.hn_q.get()

    def get_axis(self):
        return self.axis_q.get()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _unwrap(model):
    return model._orig_mod if hasattr(model, '_orig_mod') else model


@torch.no_grad()
def evaluate_reconstruction(model, tokenizer, device="cuda"):
    """Evaluate parallel reconstruction on diagnostic sentences.

    Returns per-length-bucket token accuracy and exact-match rate,
    plus individual sentence results for logging.
    """
    model.eval()
    m = _unwrap(model)

    # Tokenize all test sentences
    enc = tokenizer(RECON_TEST_SENTENCES, max_length=128, padding=True,
                    truncation=True, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Parallel decode
    concepts = m.encode(input_ids, attention_mask)
    logits = m.decode_parallel(concepts, seq_len=input_ids.shape[1])
    predicted = logits.argmax(dim=-1)

    # Per-sentence results
    results = []
    bucket_correct = {b: 0 for b in LENGTH_BUCKETS}
    bucket_total = {b: 0 for b in LENGTH_BUCKETS}
    bucket_exact = {b: 0 for b in LENGTH_BUCKETS}
    bucket_count = {b: 0 for b in LENGTH_BUCKETS}

    for i, text in enumerate(RECON_TEST_SENTENCES):
        mask = attention_mask[i].bool()
        tgt = input_ids[i][mask]
        pred = predicted[i][mask]
        seq_len = mask.sum().item()

        correct = (tgt == pred).sum().item()
        total = seq_len
        exact = (tgt == pred).all().item()

        decoded = tokenizer.decode(pred, skip_special_tokens=True)
        results.append((text, decoded, correct / max(total, 1), exact))

        # Bucket
        for bname, (lo, hi) in LENGTH_BUCKETS.items():
            if lo <= seq_len <= hi:
                bucket_correct[bname] += correct
                bucket_total[bname] += total
                bucket_exact[bname] += int(exact)
                bucket_count[bname] += 1
                break

    # Compute bucket metrics
    bucket_acc = {}
    bucket_em = {}
    for bname in LENGTH_BUCKETS:
        if bucket_total[bname] > 0:
            bucket_acc[bname] = bucket_correct[bname] / bucket_total[bname]
        else:
            bucket_acc[bname] = 0.0
        if bucket_count[bname] > 0:
            bucket_em[bname] = bucket_exact[bname] / bucket_count[bname]
        else:
            bucket_em[bname] = 0.0

    # Overall
    all_correct = sum(bucket_correct.values())
    all_total = sum(bucket_total.values())
    all_exact = sum(bucket_exact.values())
    all_count = sum(bucket_count.values())
    overall_acc = all_correct / max(all_total, 1)
    overall_em = all_exact / max(all_count, 1)

    model.train()
    return results, overall_acc, overall_em, bucket_acc, bucket_em


@torch.no_grad()
def evaluate_batch_metrics(model, tokenizer, dataset, device="cuda",
                           num_batches=5):
    """Evaluate token accuracy and exact-match on random training data.

    More representative than the fixed diagnostic sentences.
    """
    model.eval()
    m = _unwrap(model)

    bucket_correct = {b: 0 for b in LENGTH_BUCKETS}
    bucket_total = {b: 0 for b in LENGTH_BUCKETS}
    bucket_exact = {b: 0 for b in LENGTH_BUCKETS}
    bucket_count = {b: 0 for b in LENGTH_BUCKETS}

    # Save and temporarily reset weights to uniform for unbiased eval
    saved_weights = dict(dataset.bucket_weights)
    dataset.bucket_weights = {b: 1.0 for b in LENGTH_BUCKETS}

    for _ in range(num_batches):
        enc = dataset.get_batch(BATCH_SIZE)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        concepts = m.encode(input_ids, attention_mask)
        logits = m.decode_parallel(concepts, seq_len=input_ids.shape[1])
        predicted = logits.argmax(dim=-1)

        for i in range(input_ids.shape[0]):
            mask = attention_mask[i].bool()
            tgt = input_ids[i][mask]
            pred = predicted[i][mask]
            seq_len = mask.sum().item()

            correct = (tgt == pred).sum().item()
            exact = (tgt == pred).all().item()

            for bname, (lo, hi) in LENGTH_BUCKETS.items():
                if lo <= seq_len <= hi:
                    bucket_correct[bname] += correct
                    bucket_total[bname] += seq_len
                    bucket_exact[bname] += int(exact)
                    bucket_count[bname] += 1
                    break

    bucket_acc = {}
    bucket_em = {}
    for bname in LENGTH_BUCKETS:
        bucket_acc[bname] = (bucket_correct[bname] / bucket_total[bname]
                             if bucket_total[bname] > 0 else 0.0)
        bucket_em[bname] = (bucket_exact[bname] / bucket_count[bname]
                            if bucket_count[bname] > 0 else 0.0)

    all_correct = sum(bucket_correct.values())
    all_total = sum(bucket_total.values())
    all_exact = sum(bucket_exact.values())
    all_count = sum(bucket_count.values())

    # Restore dynamic weights
    dataset.bucket_weights = saved_weights

    model.train()
    return {
        "token_acc": all_correct / max(all_total, 1),
        "exact_match": all_exact / max(all_count, 1),
        "bucket_acc": bucket_acc,
        "bucket_em": bucket_em,
    }


# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def cosine_lr(step, total_steps, peak_lr, warmup_steps):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scaler, config, step, loss,
                    exact_match_ema, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = model.state_dict()
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    ckpt = {
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "loss": loss,
        "exact_match_ema": exact_match_ema,
        "version": "v10",
        "timestamp": datetime.datetime.now().isoformat(),
    }
    path = os.path.join(checkpoint_dir, f"step_{step:06d}.pt")
    torch.save(ckpt, path)
    latest = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(ckpt, latest)
    log(f"  Checkpoint saved: {path}")

    # Keep last 3 + every 50K
    all_ckpts = sorted(Path(checkpoint_dir).glob("step_*.pt"))
    to_keep = set()
    for c in all_ckpts:
        step_num = int(c.stem.split("_")[1])
        if step_num % 50000 == 0:
            to_keep.add(c)
    for c in all_ckpts[-3:]:
        to_keep.add(c)
    for c in all_ckpts:
        if c not in to_keep:
            c.unlink()


def load_checkpoint(path, device="cuda"):
    log(f"Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoderV10(config).to(device)
    state = ckpt["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR,
                                  betas=BETAS, weight_decay=WEIGHT_DECAY)
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler = torch.amp.GradScaler("cuda")
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    step = ckpt.get("step", 0)
    exact_match_ema = ckpt.get("exact_match_ema", 0.0)
    total, _ = model.count_parameters()
    log(f"Resumed: {total:,} params | step {step} | em_ema={exact_match_ema:.3f}")
    return model, optimizer, scaler, config, step, exact_match_ema


def load_encoder_from_v9(v9_path, config, device="cuda"):
    """Load encoder + bottleneck weights from V9, fresh parallel decoder."""
    log(f"Loading encoder from V9 checkpoint: {v9_path}")
    ckpt = torch.load(v9_path, map_location="cpu", weights_only=False)
    v9_state = ckpt["model_state_dict"]
    v9_state = {k.replace("_orig_mod.", ""): v for k, v in v9_state.items()}

    model = ConceptAutoencoderV10(config).to(device)

    # Copy encoder + bottleneck + embed_tokens weights
    model_state = model.state_dict()
    loaded = 0
    for key in model_state:
        if key in v9_state and model_state[key].shape == v9_state[key].shape:
            # Only copy encoder, bottleneck, and embedding weights
            if any(key.startswith(p) for p in
                   ["embed_tokens", "enc_layers", "enc_norm", "bottleneck"]):
                model_state[key] = v9_state[key]
                loaded += 1

    model.load_state_dict(model_state)
    total, _ = model.count_parameters()
    log(f"Loaded {loaded} encoder/bottleneck tensors from V9")
    log(f"Model: {total:,} params | parallel decoder initialized fresh")
    return model


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False, eval_only=False, from_v9=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V10 — NON-AUTOREGRESSIVE CONCEPT AUTOENCODER (Recon First)")
    log("=" * 70)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    log(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    model_config = dict(MODEL_CONFIG)
    model_config["vocab_size"] = tokenizer.vocab_size

    exact_match_ema = 0.0

    if resume_from is None and not fresh and from_v9 is None:
        ckpt_dir = Path(CHECKPOINT_DIR)
        latest = ckpt_dir / "latest.pt"
        if latest.exists():
            resume_from = str(latest)

    if resume_from:
        model, optimizer, scaler, config, start_step, exact_match_ema = \
            load_checkpoint(resume_from, device)
    elif from_v9:
        config = ConceptConfig(**model_config)
        model = load_encoder_from_v9(from_v9, config, device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=PEAK_LR, betas=BETAS,
            weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0
    else:
        log("Starting fresh training...")
        config = ConceptConfig(**model_config)
        model = ConceptAutoencoderV10(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=PEAK_LR, betas=BETAS,
            weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0
        total, _ = model.count_parameters()
        log(f"Model: {total:,} params ({total/1e6:.1f}M)")
        log(f"Bottleneck: {config.num_concepts} concepts x {config.concept_dim} dim "
            f"= {config.total_concept_dim} total dims")

    if eval_only:
        log("\n--- RECONSTRUCTION EVAL ---")
        results, acc, em, bacc, bem = evaluate_reconstruction(
            model, tokenizer, device)
        log(f"  Overall: token_acc={acc:.3f} exact_match={em:.3f}")
        for bname in LENGTH_BUCKETS:
            log(f"  {bname:>8s}: acc={bacc[bname]:.3f} em={bem[bname]:.3f}")
        log("")
        for orig, decoded, tacc, exact in results:
            status = "OK" if exact else "DIFF"
            log(f"  [{status}] {orig}")
            log(f"       -> {decoded}")
        return

    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()

    log("Loading data...")
    recon_dataset = ReconstructionDataset(tokenizer, max_len=config.max_seq_len)

    if len(recon_dataset.texts) == 0:
        log("ERROR: No training texts found.")
        return

    # Load pair/axis data (used when geometry gate opens)
    pair_dataset = PairDataset(tokenizer, max_len=config.max_seq_len)
    axis_dataset = ConceptAxisDataset(tokenizer, max_len=config.max_seq_len)

    log(f"\nTraining plan (V10 — Non-Autoregressive Recon First):")
    log(f"  Encoder: {config.enc_hidden}h x {config.enc_layers}L x {config.enc_heads}heads")
    log(f"  Decoder: PARALLEL {config.dec_hidden}h x {config.dec_layers}L x {config.dec_heads}heads")
    log(f"  Bottleneck: {config.num_concepts} x {config.concept_dim} = {config.total_concept_dim} dims")
    log(f"  Batch: {BATCH_SIZE}")
    log(f"  Peak LR: {PEAK_LR} | Steps: {start_step} -> {TOTAL_STEPS}")
    log(f"  Losses: reconstruction CE only (geometry gated at {EXACT_MATCH_GATE:.0%} EM)")
    log(f"  Data: {len(recon_dataset.texts):,} texts | "
        f"{len(pair_dataset.pos_pairs):,} pos pairs | "
        f"{len(pair_dataset.hard_neg_pairs):,} hard neg pairs")
    log("-" * 70)

    # Start prefetch (pair prefetch starts lazily when gate opens)
    prefetch = PrefetchBuffer(recon_dataset, device, batch_size=BATCH_SIZE)
    prefetch.start()
    pair_prefetch = None
    geometry_active = exact_match_ema >= EXACT_MATCH_GATE
    geo_start_step = None  # track when geometry activated for warmup ramp

    if geometry_active:
        log("Geometry already active from checkpoint, starting pair/axis prefetch")
        pair_prefetch = PairPrefetchBuffer(
            pair_dataset, axis_dataset, device,
            pair_batch_size=PAIR_BATCH_SIZE,
            axis_batch_size=AXIS_BATCH_SIZE)
        pair_prefetch.start()

    log("Prefetch buffer started")

    # Initial eval
    log("\n--- INITIAL RECONSTRUCTION ---")
    results, acc, em, bacc, bem = evaluate_reconstruction(
        model, tokenizer, device)
    log(f"  Overall: token_acc={acc:.3f} exact_match={em:.3f}")
    for orig, decoded, tacc, exact in results[:5]:
        status = "OK" if exact else "DIFF"
        log(f"  [{status}] {orig}")
        log(f"       -> {decoded}")

    losses = []
    start_time = time.time()

    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        log("Shutdown signal received, saving checkpoint...")
        shutdown_requested = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    m = _unwrap(model)

    for step in range(start_step, TOTAL_STEPS):
        if shutdown_requested:
            break

        current_lr = cosine_lr(step, TOTAL_STEPS, PEAK_LR, WARMUP_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)

        # --- Reconstruction (the ONLY loss in Phase 1) ---
        recon_enc = prefetch.get()
        input_ids = recon_enc["input_ids"].to(device, non_blocking=True)
        attention_mask = recon_enc["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, concepts = model(input_ids, attention_mask)
            r_loss = reconstruction_loss(logits, input_ids)
            total_loss = r_loss

        r_loss_val = r_loss.item()
        geo_losses = {}

        # --- Phase 2: Geometry losses (behind EM gate, with warmup ramp) ---
        if geometry_active and pair_prefetch is not None:
            if geo_start_step is None:
                geo_start_step = step
            geo_ramp = min(1.0, (step - geo_start_step) / GEO_WARMUP_STEPS)
            geo_losses["ramp"] = geo_ramp
            with torch.amp.autocast("cuda", dtype=torch.float16):
                # Word-order contrastive (use recon batch)
                shuffled_ids = shuffle_word_order(input_ids, attention_mask, tokenizer)
                concepts_orig = m.encode(input_ids, attention_mask)
                concepts_shuf = m.encode(shuffled_ids, attention_mask)
                wo_nce_loss, wo_sim = flat_word_order_info_nce(
                    concepts_orig.flatten(1, 2), concepts_shuf.flatten(1, 2),
                    temperature=NCE_TEMPERATURE)
                total_loss = total_loss + geo_ramp * WO_WEIGHT * wo_nce_loss
                mwo_loss = margin_word_order_loss(
                    concepts_orig.flatten(1, 2), concepts_shuf.flatten(1, 2))
                total_loss = total_loss + geo_ramp * MARGIN_WO_WEIGHT * mwo_loss
                geo_losses["wo_nce"] = wo_nce_loss.item()
                geo_losses["wo_sim"] = wo_sim
                geo_losses["m_wo"] = mwo_loss.item()

                # Decorrelation + repulsion (use recon batch concepts)
                decorr = slot_decorrelation_loss(concepts_orig)
                total_loss = total_loss + geo_ramp * DECORR_WEIGHT * decorr
                geo_losses["decorr"] = decorr.item()

                batch_rep = batch_repulsion_loss(concepts_orig.flatten(1, 2))
                total_loss = total_loss + geo_ramp * REPULSION_WEIGHT * batch_rep
                geo_losses["b_rep"] = batch_rep.item()

                hard_rep = hard_repulsion_loss(concepts_orig.flatten(1, 2))
                total_loss = total_loss + geo_ramp * HARD_REPULSION_WEIGHT * hard_rep
                geo_losses["h_rep"] = hard_rep.item()

                # Positive pair NCE + margin
                pos_enc_a, pos_enc_b = pair_prefetch.get_pos()
                pos_ids_a = pos_enc_a["input_ids"].to(device, non_blocking=True)
                pos_mask_a = pos_enc_a["attention_mask"].to(device, non_blocking=True)
                pos_ids_b = pos_enc_b["input_ids"].to(device, non_blocking=True)
                pos_mask_b = pos_enc_b["attention_mask"].to(device, non_blocking=True)
                c_a = m.encode(pos_ids_a, pos_mask_a)
                c_b = m.encode(pos_ids_b, pos_mask_b)
                nce_loss = flat_info_nce_loss(
                    c_a.flatten(1, 2), c_b.flatten(1, 2),
                    temperature=NCE_TEMPERATURE)
                total_loss = total_loss + geo_ramp * NCE_WEIGHT * nce_loss
                geo_losses["nce"] = nce_loss.item()

                m_para = margin_paraphrase_loss(
                    c_a.flatten(1, 2), c_b.flatten(1, 2))
                total_loss = total_loss + geo_ramp * MARGIN_PARA_WEIGHT * m_para
                geo_losses["m_para"] = m_para.item()

                # Hard negative margin
                hn_enc_a, hn_enc_b = pair_prefetch.get_hn()
                if hn_enc_a is not None:
                    hn_ids_a = hn_enc_a["input_ids"].to(device, non_blocking=True)
                    hn_mask_a = hn_enc_a["attention_mask"].to(device, non_blocking=True)
                    hn_ids_b = hn_enc_b["input_ids"].to(device, non_blocking=True)
                    hn_mask_b = hn_enc_b["attention_mask"].to(device, non_blocking=True)
                    c_hn_a = m.encode(hn_ids_a, hn_mask_a)
                    c_hn_b = m.encode(hn_ids_b, hn_mask_b)
                    m_neg = margin_negative_loss(
                        c_hn_a.flatten(1, 2), c_hn_b.flatten(1, 2))
                    total_loss = total_loss + geo_ramp * MARGIN_NEG_WEIGHT * m_neg
                    geo_losses["m_neg"] = m_neg.item()

                # Slot isolation (concept axis pairs)
                if axis_dataset.active_slots:
                    ax_enc_base, ax_enc_var, ax_slots = pair_prefetch.get_axis()
                    ax_ids_b = ax_enc_base["input_ids"].to(device, non_blocking=True)
                    ax_mask_b = ax_enc_base["attention_mask"].to(device, non_blocking=True)
                    ax_ids_v = ax_enc_var["input_ids"].to(device, non_blocking=True)
                    ax_mask_v = ax_enc_var["attention_mask"].to(device, non_blocking=True)
                    c_base = m.encode(ax_ids_b, ax_mask_b)
                    c_var = m.encode(ax_ids_v, ax_mask_v)
                    # Average slot isolation across sampled slots
                    unique_slots = ax_slots.unique()
                    iso_total = 0.0
                    for s in unique_slots:
                        s_mask = ax_slots == s
                        iso_total = iso_total + slot_isolation_loss(
                            c_base[s_mask], c_var[s_mask], s.item())
                    iso_loss = iso_total / max(len(unique_slots), 1)
                    total_loss = total_loss + geo_ramp * SLOT_ISO_WEIGHT * iso_loss
                    geo_losses["slot_iso"] = iso_loss.item()

        if torch.isnan(total_loss):
            log(f"NaN loss at step {step}, skipping")
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        losses.append(r_loss_val)

        # --- Logging ---
        if (step + 1) % LOG_EVERY == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            pct = (step + 1) / TOTAL_STEPS * 100
            phase = "GEOM" if geometry_active else "RECON"
            geo_str = ""
            if geo_losses:
                geo_str = " | " + " ".join(
                    f"{k}={v:.4f}" for k, v in geo_losses.items())
            log(f"step {step+1:>7d} [{phase}] | loss {avg_loss:.4f} "
                f"(recon={r_loss_val:.4f}{geo_str}) | "
                f"em_ema={exact_match_ema:.3f} | "
                f"lr {current_lr:.2e} | {pct:.1f}%")

        # --- Eval ---
        if (step + 1) % EVAL_EVERY == 0:
            # Batch metrics on random training data
            batch_metrics = evaluate_batch_metrics(
                model, tokenizer, recon_dataset, device, num_batches=5)
            token_acc = batch_metrics["token_acc"]
            exact_match = batch_metrics["exact_match"]
            bacc = batch_metrics["bucket_acc"]
            bem = batch_metrics["bucket_em"]

            # Update EMA
            exact_match_ema = (EXACT_MATCH_EMA_DECAY * exact_match_ema +
                               (1 - EXACT_MATCH_EMA_DECAY) * exact_match)

            log(f"  EVAL: token_acc={token_acc:.3f} exact_match={exact_match:.3f} "
                f"em_ema={exact_match_ema:.3f}")
            log(f"    short: acc={bacc['short']:.3f} em={bem['short']:.3f} | "
                f"medium: acc={bacc['medium']:.3f} em={bem['medium']:.3f} | "
                f"long: acc={bacc['long']:.3f} em={bem['long']:.3f}")

            # Diagnostic sentences
            results, _, _, _, _ = evaluate_reconstruction(
                model, tokenizer, device)
            for orig, decoded, tacc, exact in results:
                status = "OK" if exact else "DIFF"
                log(f"    [{status}] ({tacc:.0%}) {orig}")
                log(f"           -> {decoded}")

            # Log metrics
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            log_metrics(step + 1, avg_loss, token_acc, exact_match,
                        bacc.get("short", 0), bacc.get("medium", 0),
                        bacc.get("long", 0),
                        bem.get("short", 0), bem.get("medium", 0),
                        bem.get("long", 0),
                        current_lr, elapsed / 3600)

            # Dynamic length sampling: update weights based on per-bucket EM
            recon_dataset.update_weights(bem)

            # Geometry gate check
            if exact_match_ema >= EXACT_MATCH_GATE and not geometry_active:
                geometry_active = True
                log(f"  >>> GEOMETRY GATE OPEN (em_ema={exact_match_ema:.3f} >= {EXACT_MATCH_GATE})")
                log(f"  >>> Activating geometry losses: NCE, word-order, margins, repulsion, slot isolation")
                # Start pair/axis prefetch
                pair_prefetch = PairPrefetchBuffer(
                    pair_dataset, axis_dataset, device,
                    pair_batch_size=PAIR_BATCH_SIZE,
                    axis_batch_size=AXIS_BATCH_SIZE)
                pair_prefetch.start()
                log(f"  >>> Pair/axis prefetch started")
            elif geometry_active:
                log(f"  >>> GEOMETRY ACTIVE (em_ema={exact_match_ema:.3f})")

        # --- Checkpoint ---
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            save_checkpoint(model, optimizer, scaler, config, step + 1,
                            avg_loss, exact_match_ema, CHECKPOINT_DIR)

        if shutdown_requested:
            break

    prefetch.stop()
    if pair_prefetch is not None:
        pair_prefetch.stop()
    if losses:
        avg_loss = sum(losses[-100:]) / len(losses[-100:])
        save_checkpoint(model, optimizer, scaler, config,
                        step + 1 if not shutdown_requested else step,
                        avg_loss, exact_match_ema, CHECKPOINT_DIR)
    log("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train concept autoencoder V10")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--from-v9", type=str, default=None,
                        help="Path to V9 checkpoint (loads encoder, fresh decoder)")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    with open(".train_pid", "w") as f:
        f.write(str(os.getpid()))

    train(resume_from=args.resume, fresh=args.fresh,
          eval_only=args.eval_only, from_v9=args.from_v9)
