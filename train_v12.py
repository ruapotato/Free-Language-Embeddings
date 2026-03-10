"""
FLM V12 — Non-Autoregressive Concept Autoencoder (Pure Reconstruction + Geometry Logging)
================================================================================
Key improvements over V11:
  1. Decoder padding mask fix — attention_mask passed through self-attention in
     the parallel decoder, so the model can't learn to depend on padding context.
     (The fix lives in concept_model.py; V12 just trains fresh with it.)
  2. Geometry probing every eval step — clustering gap, analogies, direction
     consistency, word order sensitivity, effective rank. Logged to CSV/stdout
     for dashboard tracking. NO geometry losses — still pure reconstruction.
  3. All lessons from V1-V11 baked in: diverse data, dynamic length sampling,
     cosine LR, parallel decoder, 32x32 bottleneck.

Usage:
    python train_v12.py --fresh          # start from scratch
    python train_v12.py --resume         # resume from V12 checkpoint
    python train_v12.py --from-v11       # warm-start from V11 (encoder only)
    python train_v12.py --eval-only      # diagnostics only
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
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from concept_model import (ConceptConfig, ConceptAutoencoderV10,
                           reconstruction_loss, flat_similarity_matrix)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v12"
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
BATCH_SIZE = 64
PEAK_LR = 3e-4
MIN_LR = 1e-5            # cosine decay floor — slow refinement at the end
WARMUP_STEPS = 2000
TOTAL_STEPS = 600_000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0

# EMA tracking (for dashboard, no gate)
EXACT_MATCH_EMA_DECAY = 0.9

# Dynamic length sampling
DYNAMIC_SAMPLING_ALPHA = 1.0
DYNAMIC_SAMPLING_FLOOR = 0.15

# Logging
LOG_EVERY = 50
EVAL_EVERY = 500
CHECKPOINT_EVERY = 5000

# Length buckets
LENGTH_BUCKETS = {
    "short":  (1, 10),
    "medium": (11, 30),
    "long":   (31, 128),
}

LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v12.log",
    "metrics": f"{LOG_DIR}/concept_v12_metrics.csv",
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
    # Code
    "def fibonacci ( n ) : return n if n < 2 else fibonacci ( n - 1 ) + fibonacci ( n - 2 )",
    # Math
    "the derivative of x squared plus three x equals two x plus three",
    # Logic
    "if all dogs are animals and all animals breathe then all dogs breathe",
]

# ---------------------------------------------------------------------------
# Geometry probing sentences (fixed, small — run every eval)
# ---------------------------------------------------------------------------

# Analogies: (a, b, c, expected_d, label)
GEOMETRY_ANALOGIES = [
    ("she ran", "she runs", "he walked", "he walks", "tense"),
    ("the movie was great", "the movie was terrible", "the food was great", "the food was terrible", "sentiment"),
    ("she is not here", "she is here", "he is not there", "he is there", "negation"),
    ("the cats", "the cat", "the dogs", "the dog", "plural"),
    ("he is happy", "she is happy", "he is tired", "she is tired", "subject"),
    ("the huge house", "the tiny house", "the huge tree", "the tiny tree", "size"),
]

# Clustering groups (8 groups × 6 sentences = 48 sentences, ~120 within + ~756 between pairs)
GEOMETRY_CLUSTERS = {
    "animals": ["the cat sat on the mat", "a dog ran in the park", "the bird flew over the tree",
                "fish swim in the ocean", "the horse galloped fast", "a rabbit hid in the grass"],
    "weather": ["it is raining heavily today", "the sun is shining brightly", "snow covered the ground",
                "a storm is approaching fast", "the wind blew the leaves", "fog rolled in at dawn"],
    "food": ["she cooked a delicious pasta", "the pizza was freshly baked", "he ate a bowl of rice",
             "the soup was too salty", "they ordered sushi for dinner", "she baked a chocolate cake"],
    "emotions": ["she was very happy today", "he felt sad and lonely", "the news made them angry",
                 "they were excited about it", "she felt anxious before", "he was proud of his work"],
    "technology": ["the computer crashed again", "she updated her phone", "the internet was slow today",
                   "he wrote a python program", "the server went down", "she debugged the code"],
    "music": ["she played the piano well", "he strummed the guitar", "the orchestra performed beautifully",
              "they sang a duet together", "the drums kept the beat", "she composed a new song"],
    "sports": ["he kicked the ball hard", "she ran the marathon", "the team won the game",
               "they scored in the last minute", "the coach praised the players", "she swam fifty laps"],
    "travel": ["they flew to paris last week", "the train arrived on time", "she drove across the country",
               "he packed his bags quickly", "the ship sailed at dawn", "we hiked up the mountain"],
}

# Direction consistency pairs: {attr: [(positive, negative), ...]}
# 5 attributes × 6 pairs each = 30 pairs, C(6,2)=15 cosines per attribute
GEOMETRY_DIRECTIONS = {
    "negation": [("the cat is not here", "the cat is here"), ("she did not run", "she did run"),
                 ("he is not happy", "he is happy"), ("they are not coming", "they are coming"),
                 ("it was not ready", "it was ready"), ("we did not agree", "we did agree")],
    "tense": [("she ran quickly", "she runs quickly"), ("he walked home", "he walks home"),
              ("they played outside", "they play outside"), ("it rained all day", "it rains all day"),
              ("she danced well", "she dances well"), ("he cooked dinner", "he cooks dinner")],
    "sentiment": [("the movie was great", "the movie was terrible"), ("she loves the food", "she hates the food"),
                  ("a wonderful day", "a horrible day"), ("he is kind", "he is cruel"),
                  ("the news was good", "the news was bad"), ("she enjoyed the book", "she disliked the book")],
    "plurality": [("the cats sat", "the cat sat"), ("the dogs ran", "the dog ran"),
                  ("the birds flew", "the bird flew"), ("the cars drove", "the car drove"),
                  ("the boys played", "the boy played"), ("the trees fell", "the tree fell")],
    "size": [("the big cat", "the small cat"), ("a large dog", "a small dog"),
             ("the huge house", "the tiny house"), ("a massive tree", "a little tree"),
             ("the big car", "the small car"), ("a large book", "a small book")],
}

# Word order pairs: should encode differently (8 pairs)
GEOMETRY_WORD_ORDER = [
    ("the dog bit the man", "the man bit the dog"),
    ("alice likes bob", "bob likes alice"),
    ("she gave him a book", "he gave her a book"),
    ("the cat chased the mouse", "the mouse chased the cat"),
    ("the teacher praised the student", "the student praised the teacher"),
    ("the boy pushed the girl", "the girl pushed the boy"),
    ("she told him a secret", "he told her a secret"),
    ("the car hit the truck", "the truck hit the car"),
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


METRICS_HEADER = ("timestamp,step,recon_loss,token_acc,exact_match,"
                  "acc_short,acc_med,acc_long,em_short,em_med,em_long,"
                  "lr,elapsed_hours,"
                  "analogy_avg,clustering_gap,direction_consistency,"
                  "word_order_sim,effective_rank90,effective_rank95\n")


def log_metrics(step, recon_loss, token_acc, exact_match,
                acc_short, acc_med, acc_long, em_short, em_med, em_long,
                lr, elapsed_hours, geo=None):
    metrics_file = LOG_PATHS.get("metrics")
    if not metrics_file:
        return
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    write_header = not os.path.exists(metrics_file)
    with open(metrics_file, "a") as f:
        if write_header:
            f.write(METRICS_HEADER)
        ts = datetime.datetime.now().isoformat()
        g = geo or {}
        f.write(f"{ts},{step},{recon_loss:.6f},{token_acc:.4f},{exact_match:.4f},"
                f"{acc_short:.4f},{acc_med:.4f},{acc_long:.4f},"
                f"{em_short:.4f},{em_med:.4f},{em_long:.4f},"
                f"{lr:.6e},{elapsed_hours:.4f},"
                f"{g.get('analogy_avg', 0):.4f},{g.get('clustering_gap', 0):.4f},"
                f"{g.get('direction_consistency', 0):.4f},"
                f"{g.get('word_order_sim', 0):.4f},"
                f"{g.get('rank90', 0)},{g.get('rank95', 0)}\n")


# ---------------------------------------------------------------------------
# Data loading — diverse sources
# ---------------------------------------------------------------------------

def _split_into_chunks(text, max_words=40):
    """Split text into sentence/paragraph-sized chunks."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.extend(words)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if len(c.split()) >= 3]


class ReconstructionDataset:
    """Loads diverse training texts, supports dynamic length-weighted sampling."""

    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.buckets = {name: [] for name in LENGTH_BUCKETS}
        self.bucket_weights = {name: 1.0 for name in LENGTH_BUCKETS}
        self.total_texts = 0
        self._load_all()

    def _bucket_text(self, text):
        approx_tokens = len(text.split()) + 2
        for bname, (lo, hi) in LENGTH_BUCKETS.items():
            if lo <= approx_tokens <= hi:
                self.buckets[bname].append(text)
                self.total_texts += 1
                return
        if approx_tokens > 0:
            self.buckets["long"].append(text)
            self.total_texts += 1

    def _load_all(self):
        # 1. Pair data
        pair_count = 0
        data_dir = Path("data/pairs")
        if data_dir.exists():
            for path in sorted(data_dir.glob("*.jsonl")):
                if path.name.startswith("eval_"):
                    continue
                with open(path) as f:
                    for line in f:
                        try:
                            doc = json.loads(line)
                            for key in ["text_a", "text_b"]:
                                text = doc.get(key, "").strip()
                                if len(text) > 10:
                                    self._bucket_text(text)
                                    pair_count += 1
                        except (json.JSONDecodeError, KeyError):
                            continue
        log(f"  Pair texts: {pair_count:,}")

        # 2. Pretrain data (diverse: code, math, docs, prose)
        MAX_CHUNKS_PER_SOURCE = 500_000
        pretrain_count = 0
        pretrain_dir = Path("data/pretrain")
        if pretrain_dir.exists():
            for path in sorted(pretrain_dir.glob("*.jsonl")):
                source = path.stem
                count = 0
                with open(path) as f:
                    for line in f:
                        if count >= MAX_CHUNKS_PER_SOURCE:
                            break
                        try:
                            doc = json.loads(line)
                            text = doc.get("text", "").strip()
                            if len(text) < 20:
                                continue
                            for chunk in _split_into_chunks(text):
                                self._bucket_text(chunk)
                                count += 1
                                if count >= MAX_CHUNKS_PER_SOURCE:
                                    break
                        except (json.JSONDecodeError, KeyError):
                            continue
                pretrain_count += count
                log(f"    {source}: {count:,} chunks")
        log(f"  Pretrain texts: {pretrain_count:,}")

        # 3. Conversations
        conv_count = 0
        conv_path = Path("data/oasst2_conversations.jsonl")
        if conv_path.exists():
            with open(conv_path) as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        text = doc.get("text", "").strip()
                        if len(text) < 20:
                            continue
                        for chunk in _split_into_chunks(text):
                            self._bucket_text(chunk)
                            conv_count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        log(f"  Conversation texts: {conv_count:,}")

        # Shuffle buckets
        for bname in self.buckets:
            random.shuffle(self.buckets[bname])

        log(f"  TOTAL: {self.total_texts:,}")
        for bname in LENGTH_BUCKETS:
            log(f"    {bname}: {len(self.buckets[bname]):,}")

    def update_weights(self, bucket_em):
        for bname in LENGTH_BUCKETS:
            em = bucket_em.get(bname, 0.0)
            self.bucket_weights[bname] = max(
                DYNAMIC_SAMPLING_FLOOR,
                (1.0 - em) ** DYNAMIC_SAMPLING_ALPHA)
        total = sum(self.bucket_weights.values())
        for bname in self.bucket_weights:
            self.bucket_weights[bname] /= total
        log(f"  Dynamic weights: " + " | ".join(
            f"{b}={self.bucket_weights[b]:.2f}" for b in LENGTH_BUCKETS))

    def get_batch(self, batch_size):
        texts = []
        bucket_names = list(LENGTH_BUCKETS.keys())
        weights = [self.bucket_weights[b] for b in bucket_names]

        for _ in range(batch_size):
            bname = random.choices(bucket_names, weights=weights, k=1)[0]
            bucket = self.buckets[bname]
            if bucket:
                texts.append(bucket[random.randint(0, len(bucket) - 1)])
            else:
                for bn in bucket_names:
                    if self.buckets[bn]:
                        texts.append(self.buckets[bn][random.randint(0, len(self.buckets[bn]) - 1)])
                        break

        enc = self.tokenizer(texts, max_length=self.max_len,
                             padding=True, truncation=True,
                             return_tensors="pt")
        return enc


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


# ---------------------------------------------------------------------------
# Evaluation — reconstruction
# ---------------------------------------------------------------------------

def _unwrap(model):
    return model._orig_mod if hasattr(model, '_orig_mod') else model


@torch.no_grad()
def evaluate_reconstruction(model, tokenizer, device="cuda"):
    """Evaluate parallel reconstruction on diagnostic sentences."""
    model.eval()
    m = _unwrap(model)

    enc = tokenizer(RECON_TEST_SENTENCES, max_length=128, padding=True,
                    truncation=True, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    concepts = m.encode(input_ids, attention_mask)
    logits = m.decode_parallel(concepts, seq_len=input_ids.shape[1],
                               attention_mask=attention_mask)
    predicted = logits.argmax(dim=-1)

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

        for bname, (lo, hi) in LENGTH_BUCKETS.items():
            if lo <= seq_len <= hi:
                bucket_correct[bname] += correct
                bucket_total[bname] += total
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
    overall_acc = all_correct / max(all_total, 1)
    overall_em = all_exact / max(all_count, 1)

    model.train()
    return results, overall_acc, overall_em, bucket_acc, bucket_em


@torch.no_grad()
def evaluate_batch_metrics(model, tokenizer, dataset, device="cuda",
                           num_batches=5):
    """Evaluate token accuracy and exact-match on random data (uniform sampling)."""
    model.eval()
    m = _unwrap(model)

    saved_weights = dict(dataset.bucket_weights)
    dataset.bucket_weights = {b: 1.0 for b in LENGTH_BUCKETS}

    bucket_correct = {b: 0 for b in LENGTH_BUCKETS}
    bucket_total = {b: 0 for b in LENGTH_BUCKETS}
    bucket_exact = {b: 0 for b in LENGTH_BUCKETS}
    bucket_count = {b: 0 for b in LENGTH_BUCKETS}

    for _ in range(num_batches):
        enc = dataset.get_batch(BATCH_SIZE)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        concepts = m.encode(input_ids, attention_mask)
        logits = m.decode_parallel(concepts, seq_len=input_ids.shape[1],
                                   attention_mask=attention_mask)
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

    dataset.bucket_weights = saved_weights

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

    model.train()
    return {
        "token_acc": all_correct / max(all_total, 1),
        "exact_match": all_exact / max(all_count, 1),
        "bucket_acc": bucket_acc,
        "bucket_em": bucket_em,
    }


# ---------------------------------------------------------------------------
# Geometry probing — lightweight, runs every eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_flat(model, tokenizer, texts, device):
    """Encode texts → flat normalized vectors [B, 1024]."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64,
                    return_tensors="pt").to(device)
    m = _unwrap(model)
    concepts = m.encode(enc["input_ids"], enc["attention_mask"])
    flat = concepts.view(concepts.shape[0], -1)
    return F.normalize(flat, p=2, dim=-1), concepts


@torch.no_grad()
def probe_geometry(model, tokenizer, device="cuda"):
    """Lightweight geometry probe — returns dict of metrics.

    Runs 6 analogies, 4 clustering groups, 3 direction attributes,
    4 word order pairs, and effective rank on ~30 sentences.
    Takes ~0.1s on GPU, negligible overhead.
    """
    model.eval()
    geo = {}

    # --- Analogies ---
    analogy_scores = []
    for a, b, c, d, label in GEOMETRY_ANALOGIES:
        va, _ = _encode_flat(model, tokenizer, [a], device)
        vb, _ = _encode_flat(model, tokenizer, [b], device)
        vc, _ = _encode_flat(model, tokenizer, [c], device)
        vd, _ = _encode_flat(model, tokenizer, [d], device)
        predicted = F.normalize(va - vb + vc, p=2, dim=-1)
        sim = F.cosine_similarity(predicted, vd).item()
        analogy_scores.append(sim)
    geo["analogy_avg"] = float(np.mean(analogy_scores))

    # --- Clustering gap ---
    group_concepts = {}
    for name, sents in GEOMETRY_CLUSTERS.items():
        _, concepts = _encode_flat(model, tokenizer, sents, device)
        group_concepts[name] = concepts

    within_sims = []
    for name, concepts in group_concepts.items():
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                within_sims.append(
                    flat_similarity_matrix(concepts[i:i+1], concepts[j:j+1]).item())

    between_sims = []
    gnames = list(group_concepts.keys())
    for i in range(len(gnames)):
        for j in range(i + 1, len(gnames)):
            ca, cb = group_concepts[gnames[i]], group_concepts[gnames[j]]
            for a in range(len(ca)):
                for b in range(len(cb)):
                    between_sims.append(
                        flat_similarity_matrix(ca[a:a+1], cb[b:b+1]).item())

    geo["clustering_gap"] = float(np.mean(within_sims) - np.mean(between_sims))

    # --- Direction consistency ---
    dir_scores = []
    for attr, pairs in GEOMETRY_DIRECTIONS.items():
        deltas = []
        for pos, neg in pairs:
            vp, _ = _encode_flat(model, tokenizer, [pos], device)
            vn, _ = _encode_flat(model, tokenizer, [neg], device)
            deltas.append(F.normalize(vp - vn, p=2, dim=-1))
        cons = []
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                cons.append(F.cosine_similarity(deltas[i], deltas[j]).item())
        dir_scores.append(float(np.mean(cons)))
    geo["direction_consistency"] = float(np.mean(dir_scores))

    # --- Word order sensitivity ---
    wo_sims = []
    for a, b in GEOMETRY_WORD_ORDER:
        va, _ = _encode_flat(model, tokenizer, [a], device)
        vb, _ = _encode_flat(model, tokenizer, [b], device)
        wo_sims.append(F.cosine_similarity(va, vb).item())
    geo["word_order_sim"] = float(np.mean(wo_sims))

    # --- Effective rank ---
    all_sents = []
    for sents in GEOMETRY_CLUSTERS.values():
        all_sents.extend(sents)
    all_sents.extend([
        "the president gave a speech", "she plays piano beautifully",
        "the train arrived on time", "he forgot his wallet at home",
        "the garden was full of flowers", "she read the book in one day",
        "the mountain was covered in snow", "they celebrated their anniversary",
        "the experiment failed completely", "she won the championship",
        "the baby cried all night", "he fixed the broken window",
        "the river flows to the sea", "she painted a beautiful portrait",
        "the fire spread quickly", "he whispered a secret to her",
        "the team won the game", "she discovered a new species",
    ])
    vecs, _ = _encode_flat(model, tokenizer, all_sents, device)
    vecs_np = vecs.cpu().numpy()
    vecs_np = vecs_np - vecs_np.mean(axis=0)
    _, s, _ = np.linalg.svd(vecs_np, full_matrices=False)
    var = s ** 2
    cumvar = np.cumsum(var / var.sum())
    geo["rank90"] = int(np.searchsorted(cumvar, 0.90) + 1)
    geo["rank95"] = int(np.searchsorted(cumvar, 0.95) + 1)

    model.train()
    return geo


# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def cosine_lr(step, total_steps, peak_lr, warmup_steps, min_lr=MIN_LR):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


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
        "version": "v12",
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


def load_encoder_from_v11(v11_path, config, device="cuda"):
    """Load encoder + bottleneck weights from V11, fresh parallel decoder."""
    log(f"Loading encoder from V11 checkpoint: {v11_path}")
    ckpt = torch.load(v11_path, map_location="cpu", weights_only=False)
    v11_state = ckpt["model_state_dict"]
    v11_state = {k.replace("_orig_mod.", ""): v for k, v in v11_state.items()}

    model = ConceptAutoencoderV10(config).to(device)

    model_state = model.state_dict()
    loaded = 0
    for key in model_state:
        if key in v11_state and model_state[key].shape == v11_state[key].shape:
            if any(key.startswith(p) for p in
                   ["embed_tokens", "enc_layers", "enc_norm", "bottleneck"]):
                model_state[key] = v11_state[key]
                loaded += 1

    model.load_state_dict(model_state)
    total, _ = model.count_parameters()
    log(f"Loaded {loaded} encoder/bottleneck tensors from V11")
    log(f"Model: {total:,} params | parallel decoder initialized fresh")
    return model


# ---------------------------------------------------------------------------
# Main training loop — pure reconstruction + geometry logging
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False, eval_only=False, from_v11=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V12 — NON-AUTOREGRESSIVE CONCEPT AUTOENCODER")
    log("  Pure Reconstruction + Geometry Logging + Decoder Padding Mask Fix")
    log("=" * 70)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    log(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    model_config = dict(MODEL_CONFIG)
    model_config["vocab_size"] = tokenizer.vocab_size

    exact_match_ema = 0.0

    if resume_from is None and not fresh and from_v11 is None:
        ckpt_dir = Path(CHECKPOINT_DIR)
        latest = ckpt_dir / "latest.pt"
        if latest.exists():
            resume_from = str(latest)

    if resume_from:
        model, optimizer, scaler, config, start_step, exact_match_ema = \
            load_checkpoint(resume_from, device)
    elif from_v11:
        config = ConceptConfig(**model_config)
        model = load_encoder_from_v11(from_v11, config, device)
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
        log("\n--- GEOMETRY PROBE ---")
        geo = probe_geometry(model, tokenizer, device)
        log(f"  analogy_avg={geo['analogy_avg']:.3f}")
        log(f"  clustering_gap={geo['clustering_gap']:+.4f}")
        log(f"  direction_consistency={geo['direction_consistency']:.3f}")
        log(f"  word_order_sim={geo['word_order_sim']:.3f}")
        log(f"  effective_rank: 90%={geo['rank90']}  95%={geo['rank95']}")
        return

    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()

    log("Loading data...")
    recon_dataset = ReconstructionDataset(tokenizer, max_len=config.max_seq_len)

    if recon_dataset.total_texts == 0:
        log("ERROR: No training texts found.")
        return

    log(f"\nTraining plan (V12 — Pure Reconstruction + Geometry Logging):")
    log(f"  Encoder: {config.enc_hidden}h x {config.enc_layers}L x {config.enc_heads}heads")
    log(f"  Decoder: PARALLEL {config.dec_hidden}h x {config.dec_layers}L x {config.dec_heads}heads")
    log(f"  Bottleneck: {config.num_concepts} x {config.concept_dim} = {config.total_concept_dim} dims")
    log(f"  Batch: {BATCH_SIZE}")
    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")
    log(f"  Data: {recon_dataset.total_texts:,} texts (diverse: prose, code, math, docs)")
    log(f"  NEW: Decoder attention_mask fix | Geometry probing every {EVAL_EVERY} steps")
    log("-" * 70)

    prefetch = PrefetchBuffer(recon_dataset, device, batch_size=BATCH_SIZE)
    prefetch.start()
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

    # Initial geometry
    geo = probe_geometry(model, tokenizer, device)
    log(f"  Initial geometry: analogy={geo['analogy_avg']:.3f} "
        f"cluster_gap={geo['clustering_gap']:+.4f} "
        f"dir_con={geo['direction_consistency']:.3f} "
        f"wo_sim={geo['word_order_sim']:.3f} "
        f"rank90={geo['rank90']}")

    losses = []
    start_time = time.time()

    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        log("Shutdown signal received, saving checkpoint...")
        shutdown_requested = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    for step in range(start_step, TOTAL_STEPS):
        if shutdown_requested:
            break

        current_lr = cosine_lr(step, TOTAL_STEPS, PEAK_LR, WARMUP_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)

        recon_enc = prefetch.get()
        input_ids = recon_enc["input_ids"].to(device, non_blocking=True)
        attention_mask = recon_enc["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, concepts = model(input_ids, attention_mask)
            r_loss = reconstruction_loss(logits, input_ids)

        r_loss_val = r_loss.item()

        if torch.isnan(r_loss):
            log(f"NaN loss at step {step}, skipping")
            continue

        scaler.scale(r_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        losses.append(r_loss_val)

        # --- Logging ---
        if (step + 1) % LOG_EVERY == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            pct = (step + 1) / TOTAL_STEPS * 100
            log(f"step {step+1:>7d} [RECON] | loss {avg_loss:.4f} "
                f"(recon={r_loss_val:.4f}) | "
                f"em_ema={exact_match_ema:.3f} | "
                f"lr {current_lr:.2e} | {pct:.1f}%")

        # --- Eval ---
        if (step + 1) % EVAL_EVERY == 0:
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

            # Geometry probe
            geo = probe_geometry(model, tokenizer, device)
            log(f"  GEOMETRY: analogy={geo['analogy_avg']:.3f} "
                f"cluster_gap={geo['clustering_gap']:+.4f} "
                f"dir_con={geo['direction_consistency']:.3f} "
                f"wo_sim={geo['word_order_sim']:.3f} "
                f"rank90={geo['rank90']} rank95={geo['rank95']}")

            # Log metrics (with geometry)
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            log_metrics(step + 1, avg_loss, token_acc, exact_match,
                        bacc.get("short", 0), bacc.get("medium", 0),
                        bacc.get("long", 0),
                        bem.get("short", 0), bem.get("medium", 0),
                        bem.get("long", 0),
                        current_lr, elapsed / 3600, geo=geo)

            # Dynamic length sampling
            recon_dataset.update_weights(bem)

        # --- Checkpoint ---
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            save_checkpoint(model, optimizer, scaler, config, step + 1,
                            avg_loss, exact_match_ema, CHECKPOINT_DIR)

        if shutdown_requested:
            break

    prefetch.stop()
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
    parser = argparse.ArgumentParser(description="Train concept autoencoder V12")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--from-v11", type=str, default=None,
                        help="Path to V11 checkpoint (loads encoder, fresh decoder)")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    with open(".train_pid", "w") as f:
        f.write(str(os.getpid()))

    train(resume_from=args.resume, fresh=args.fresh,
          eval_only=args.eval_only, from_v11=args.from_v11)
