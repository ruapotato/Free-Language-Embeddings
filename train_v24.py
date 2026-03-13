"""
FLM V24 — Sentence Compressor
================================================================================
Non-autoregressive autoencoder that compresses INDIVIDUAL SENTENCES into concept
vectors and reconstructs them. Each training sample is one sentence — NOT a
random document chunk.

This is Phase 1 of a two-model architecture:
  - Model 1 (this): sentence → concept vector → sentence (frozen after training)
  - Model 2 (later): autoregressive LM in concept space, predicts next sentence
    as a concept vector. Model 1 strapped on front/back as frozen bookends.

The LM will never see tokens. A prompt gets split into sentences, each sentence
becomes one concept vector via the frozen encoder, the LM predicts the next
concept vector(s), and the frozen decoder renders them back to text.

Architecture: 4L encoder (256d) → 64×16=1024 bottleneck + whitening → 4L decoder (256d)
Data: Individual sentences extracted from Wikipedia, Gutenberg, StackExchange, etc.

Usage:
    python train_v24.py --fresh            # start from scratch
    python train_v24.py                    # auto-resume
    python train_v24.py --eval-only        # diagnostics only
"""

import os
import re
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
from concept_model import (ConceptConfig, ConceptAutoencoderV24,
                           reconstruction_loss,
                           flat_similarity_matrix)
from geometry_data import GeometryDataGenerator, verify_splits

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v24"
LOG_DIR = "logs"

MODEL_CONFIG = dict(
    vocab_size=30522,       # bert-base-uncased (English-only)
    enc_hidden=256,
    enc_layers=4,
    enc_heads=4,
    enc_intermediate=1024,
    num_concepts=64,
    concept_dim=16,         # 64×16 = 1024 bottleneck (2x bigger than V21)
    dec_hidden=256,
    dec_layers=4,
    dec_heads=4,
    dec_intermediate=1024,
    max_seq_len=64,
    dropout=0.1,
)

# Training hyperparameters
BATCH_SIZE = 256            # tiny model, big batch
PEAK_LR = 3e-4
MIN_LR = 1e-5
WARMUP_STEPS = 2000
TOTAL_STEPS = 600_000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0

# EMA tracking
EXACT_MATCH_EMA_DECAY = 0.99

# Logging
LOG_EVERY = 50
EVAL_EVERY = 1000
CHECKPOINT_EVERY = 1000

LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v24.log",
    "metrics": f"{LOG_DIR}/concept_v24_metrics.csv",
}

# Diagnostic sentences for EN reconstruction
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
    "def fibonacci ( n ) : return n if n < 2 else fibonacci ( n - 1 ) + fibonacci ( n - 2 )",
    "the derivative of x squared plus three x equals two x plus three",
    "if all dogs are animals and all animals breathe then all dogs breathe",
]

# Pretrain data sources (order = priority; weights for sampling)
# Weights capped so NO source repeats. Each source seen at most once.
# Total run = ~4.9B tokens. Weights normalized automatically.
PRETRAIN_SOURCES = [
    ("data/pretrain/wikipedia.jsonl", 0.30),       # 4.75B avail, see ~1.5B
    ("data/pretrain/gutenberg.jsonl", 0.20),       # 3.5B avail, see ~1.0B
    ("data/pretrain/stackexchange.jsonl", 0.16),   # 0.8B avail, see ~0.8B (capped)
    ("data/pretrain/arxiv.jsonl", 0.20),           # 5.25B avail, see ~1.0B
    ("data/pretrain/usgpo.jsonl", 0.10),           # 17.75B avail, see ~0.5B
    ("data/pretrain/rfcs.jsonl", 0.025),           # 125M avail, see ~123M (capped)
    ("data/pretrain/kernel_docs.jsonl", 0.0017),   # 8.5M avail (capped)
    ("data/pretrain/archwiki.jsonl", 0.0015),      # 7.25M avail (capped)
    ("data/pretrain/tldp.jsonl", 0.0016),          # 8M avail (capped)
    ("data/pretrain/gnu_manuals.jsonl", 0.0023),   # 11.5M avail (capped)
    ("data/pretrain/manpages.jsonl", 0.0049),      # 24M avail (capped)
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


METRICS_HEADER = ("timestamp,step,recon_loss,token_acc,exact_match,em_ema,"
                  "lr,elapsed_hours,"
                  "analogy_avg,clustering_gap,direction_consistency,"
                  "word_order_sim,effective_rank90,effective_rank95\n")


def log_metrics(step, metrics):
    metrics_file = LOG_PATHS.get("metrics")
    if not metrics_file:
        return
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    write_header = not os.path.exists(metrics_file)
    with open(metrics_file, "a") as f:
        if write_header:
            f.write(METRICS_HEADER)
        m = metrics
        g = m.get("geo", {})
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts},{step},{m.get('recon_loss',0):.6f},"
                f"{m.get('token_acc',0):.4f},{m.get('exact_match',0):.4f},"
                f"{m.get('em_ema',0):.4f},{m.get('lr',0):.6e},"
                f"{m.get('elapsed_hours',0):.4f},"
                f"{g.get('analogy_avg',0):.4f},{g.get('clustering_gap',0):.4f},"
                f"{g.get('dir_consistency', g.get('direction_consistency', 0)):.4f},"
                f"{g.get('word_order_sim',0):.4f},"
                f"{g.get('rank90',0)},{g.get('rank95',0)}\n")


# ---------------------------------------------------------------------------
# Streaming data loader — reads from massive JSONL files
# ---------------------------------------------------------------------------

class SentenceDataset:
    """
    Streams INDIVIDUAL SENTENCES from JSONL pretrain files.

    Each training sample is one sentence — because at inference time,
    a prompt gets split into sentences, each becomes one concept vector,
    and the LM operates on the sequence of concept vectors.

    Documents are split on sentence boundaries (. ! ?).
    Sentences are buffered and served one per sample.
    Batches are bucketed by token length for minimal padding.
    """

    # Sentence splitting regex: split after . ! ? followed by space or end
    _SENT_RE = re.compile(r'(?<=[.!?])\s+')

    def __init__(self, tokenizer, max_len=128):
        self.tok = tokenizer
        self.max_len = max_len
        self.sources = []
        self.weights = []
        self.cum_weights = []

        # Buffer of extracted sentences (raw text strings)
        self._sentence_buf = []
        self._buf_target = 50_000  # refill when below this

        for path, weight in PRETRAIN_SOURCES:
            if os.path.exists(path):
                self.sources.append(path)
                self.weights.append(weight)
                log(f"  {os.path.basename(path)}: weight={weight:.0%}")
            else:
                log(f"  {os.path.basename(path)}: NOT FOUND, skipping")

        if not self.sources:
            raise RuntimeError("No pretrain data found!")

        # Normalize weights
        total_w = sum(self.weights)
        self.weights = [w / total_w for w in self.weights]
        cum = 0
        for w in self.weights:
            cum += w
            self.cum_weights.append(cum)

        # File handles
        self._handles = [None] * len(self.sources)

        log(f"  {len(self.sources)} sources loaded")

        # Initial fill
        self._refill_buffer()
        log(f"  Sentence buffer: {len(self._sentence_buf):,} sentences ready")

    def _sample_source(self):
        r = random.random()
        for i, cw in enumerate(self.cum_weights):
            if r <= cw:
                return i
        return len(self.sources) - 1

    def _read_line(self, source_idx):
        if self._handles[source_idx] is None:
            self._handles[source_idx] = open(self.sources[source_idx], "r")
        line = self._handles[source_idx].readline()
        if not line:
            self._handles[source_idx].close()
            self._handles[source_idx] = open(self.sources[source_idx], "r")
            line = self._handles[source_idx].readline()
            if not line:
                return None
        return line

    def _extract_sentences(self, text):
        """Split a document into individual sentences."""
        sentences = self._SENT_RE.split(text.strip())
        result = []
        for s in sentences:
            s = s.strip()
            # Filter: must be a real sentence (3+ words, not too long)
            word_count = len(s.split())
            if word_count >= 3 and word_count <= 50:
                result.append(s)
        return result

    def _refill_buffer(self):
        """Read documents and extract sentences until buffer is full."""
        attempts = 0
        while len(self._sentence_buf) < self._buf_target and attempts < 100_000:
            attempts += 1
            source_idx = self._sample_source()
            line = self._read_line(source_idx)
            if line is None:
                continue
            try:
                doc = json.loads(line)
                text = doc.get("text", "").strip()
                if len(text) < 20:
                    continue
                sents = self._extract_sentences(text)
                self._sentence_buf.extend(sents)
            except (json.JSONDecodeError, KeyError):
                continue
        random.shuffle(self._sentence_buf)

    def get_batch(self, batch_size):
        """Get a batch of tokenized individual sentences."""
        # Refill if running low
        if len(self._sentence_buf) < batch_size * 2:
            self._refill_buffer()

        # Grab sentences
        texts = []
        while len(texts) < batch_size and self._sentence_buf:
            texts.append(self._sentence_buf.pop())

        enc = self.tok(texts, max_length=self.max_len,
                       padding='longest', truncation=True, return_tensors="pt")
        return enc

    def close(self):
        for h in self._handles:
            if h is not None:
                h.close()


class PrefetchBuffer:
    """Pre-generates batches in background thread."""

    def __init__(self, dataset, batch_size=64, buf_size=8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.q = queue.Queue(maxsize=buf_size)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._fill, daemon=True,
                                        name="v24-prefetch")
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
            except Exception as e:
                # Log but don't crash the prefetch thread
                continue

    def get(self):
        return self.q.get()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(model):
    return model._orig_mod if hasattr(model, '_orig_mod') else model


@torch.no_grad()
def evaluate_reconstruction(model, tokenizer, device="cuda"):
    model.eval()
    m = _unwrap(model)
    results = []
    total_correct = 0
    total_tokens = 0
    total_exact = 0
    for text in RECON_TEST_SENTENCES:
        enc = tokenizer([text], max_length=128, padding=True,
                        truncation=True, return_tensors="pt").to(device)
        concepts = m.encode(enc["input_ids"], enc["attention_mask"])
        logits = m.decode(concepts, seq_len=enc["input_ids"].shape[1],
                          attention_mask=enc["attention_mask"])
        predicted = logits.argmax(dim=-1)
        mask = enc["attention_mask"][0].bool()
        tgt = enc["input_ids"][0][mask]
        pred = predicted[0][mask]
        correct = (tgt == pred).sum().item()
        total = mask.sum().item()
        exact = (tgt == pred).all().item()
        total_correct += correct
        total_tokens += total
        total_exact += int(exact)
        decoded = tokenizer.decode(pred, skip_special_tokens=True)
        results.append((text, decoded, correct / max(total, 1), exact))
    token_acc = total_correct / max(total_tokens, 1)
    exact_match = total_exact / max(len(RECON_TEST_SENTENCES), 1)
    model.train()
    return results, token_acc, exact_match


# ---------------------------------------------------------------------------
# Geometry probing (same probes as V19/V20 — measures emergent structure)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_flat(model, tokenizer, texts, device):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64,
                    return_tensors="pt").to(device)
    m = _unwrap(model)
    concepts = m.encode(enc["input_ids"], enc["attention_mask"])
    flat = concepts.view(concepts.shape[0], -1)
    return F.normalize(flat, p=2, dim=-1), concepts


def probe_geometry(model, tokenizer, device="cuda"):
    model.eval()
    geo = {}
    test_gen = GeometryDataGenerator(split="test", seed=42)

    # Analogies
    analogy_scores = []
    a_texts, b_texts, c_texts, d_texts = test_gen.analogy_batch(batch_size=20)
    for i in range(len(a_texts)):
        va, _ = _encode_flat(model, tokenizer, [a_texts[i]], device)
        vb, _ = _encode_flat(model, tokenizer, [b_texts[i]], device)
        vc, _ = _encode_flat(model, tokenizer, [c_texts[i]], device)
        vd, _ = _encode_flat(model, tokenizer, [d_texts[i]], device)
        predicted = F.normalize(va - vb + vc, p=2, dim=-1)
        sim = F.cosine_similarity(predicted, vd).item()
        analogy_scores.append(sim)
    geo["analogy_avg"] = sum(analogy_scores) / len(analogy_scores) if analogy_scores else 0

    # Clustering
    cluster_groups = test_gen.cluster_batch(n_groups=4, n_per_group=5)
    group_concepts = {}
    for name, sents in cluster_groups.items():
        _, concepts_3d = _encode_flat(model, tokenizer, sents, device)
        group_concepts[name] = concepts_3d
    within_sims, between_sims = [], []
    group_names = list(group_concepts.keys())
    for name in group_names:
        c = group_concepts[name]
        sim_mat = flat_similarity_matrix(c, c)
        n = sim_mat.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                within_sims.append(sim_mat[i, j].item())
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            ci = group_concepts[group_names[i]]
            cj = group_concepts[group_names[j]]
            for a in range(ci.shape[0]):
                for b in range(cj.shape[0]):
                    fi = ci[a].view(1, -1)
                    fj = cj[b].view(1, -1)
                    between_sims.append(F.cosine_similarity(fi, fj).item())
    avg_within = sum(within_sims) / len(within_sims) if within_sims else 0
    avg_between = sum(between_sims) / len(between_sims) if between_sims else 0
    geo["clustering_gap"] = avg_within - avg_between

    # Direction consistency
    dir_attr_pairs = test_gen.direction_batch(n_pairs_per_attr=4)
    all_dir_sims = []
    for attr_name, pairs in dir_attr_pairs.items():
        deltas = []
        for pos, neg in pairs:
            vp, _ = _encode_flat(model, tokenizer, [pos], device)
            vn, _ = _encode_flat(model, tokenizer, [neg], device)
            deltas.append(F.normalize(vp - vn, p=2, dim=-1))
        for a in range(len(deltas)):
            for b in range(a + 1, len(deltas)):
                sim = F.cosine_similarity(deltas[a], deltas[b]).item()
                all_dir_sims.append(sim)
    geo["dir_consistency"] = sum(all_dir_sims) / len(all_dir_sims) if all_dir_sims else 0

    # Word order
    wo_sims = []
    wo_origs, wo_swaps = test_gen.word_order_batch(batch_size=20)
    for i in range(len(wo_origs)):
        va, _ = _encode_flat(model, tokenizer, [wo_origs[i]], device)
        vb, _ = _encode_flat(model, tokenizer, [wo_swaps[i]], device)
        wo_sims.append(F.cosine_similarity(va, vb).item())
    geo["word_order_sim"] = sum(wo_sims) / len(wo_sims) if wo_sims else 1.0

    # Effective rank
    all_sents = test_gen.diverse_sentences(batch_size=60)
    vecs, _ = _encode_flat(model, tokenizer, all_sents, device)
    vecs_np = vecs.cpu().numpy()
    try:
        U, S, Vt = np.linalg.svd(vecs_np, full_matrices=False)
        S_norm = S / S.sum()
        cumsum = np.cumsum(S_norm)
        geo["rank90"] = int(np.searchsorted(cumsum, 0.90) + 1)
        geo["rank95"] = int(np.searchsorted(cumsum, 0.95) + 1)
    except Exception:
        geo["rank90"] = 0
        geo["rank95"] = 0

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
        "version": "v24",
        "timestamp": datetime.datetime.now().isoformat(),
    }
    path = os.path.join(checkpoint_dir, f"step_{step:06d}.pt")
    torch.save(ckpt, path)
    latest = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(ckpt, latest)
    log(f"  Checkpoint saved: {path}")

    # Prune old checkpoints (keep every 50K + last 3)
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
    log(f"Loading V24 checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        config = ConceptConfig(**cfg)
    else:
        config = cfg
    model = ConceptAutoencoderV24(config).to(device)
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
    log(f"Resumed V24: {total:,} params | step {step} | em_ema={exact_match_ema:.3f}")
    return model, optimizer, scaler, config, step, exact_match_ema


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False, eval_only=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V24 — TINY SENTENCE COMPRESSOR")
    log(f"  Bottleneck: 32x16 = 512 dims | {MODEL_CONFIG['dec_layers']}L decoder | Concept whitening")
    log("  Data: Wikipedia + Gutenberg + StackExchange + arXiv + USGPO")
    log("  Philosophy: structure from compression, not from auxiliary losses")
    log("=" * 70)

    verify_splits()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    eval_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    log(f"BERT-uncased tokenizer: vocab={tokenizer.vocab_size}")

    model_config = dict(MODEL_CONFIG)
    model_config["vocab_size"] = tokenizer.vocab_size

    exact_match_ema = 0.0

    if resume_from is None and not fresh:
        latest = Path(CHECKPOINT_DIR) / "latest.pt"
        if latest.exists():
            resume_from = str(latest)

    if resume_from:
        model, optimizer, scaler, config, start_step, exact_match_ema = \
            load_checkpoint(resume_from, device)
    else:
        log("Starting fresh training...")
        config = ConceptConfig(**model_config)
        model = ConceptAutoencoderV24(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=PEAK_LR, betas=BETAS,
            weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0
        total, _ = model.count_parameters()
        log(f"Model: {total:,} params ({total/1e6:.1f}M)")

    if eval_only:
        log("\n--- EN RECONSTRUCTION ---")
        results, acc, em = evaluate_reconstruction(model, eval_tokenizer, device)
        log(f"  token_acc={acc:.3f} exact_match={em:.3f}")
        for orig, decoded, tacc, exact in results:
            status = "OK" if exact else "DIFF"
            log(f"  [{status}] ({tacc:.0%}) {orig}")
            if not exact:
                log(f"       -> {decoded}")
        log("\n--- GEOMETRY (TEST) ---")
        geo = probe_geometry(model, eval_tokenizer, device)
        for k, v in geo.items():
            log(f"  {k}: {v}")
        return

    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()

    log("Loading data...")
    dataset = SentenceDataset(tokenizer, max_len=config.max_seq_len)

    total, _ = _unwrap(model).count_parameters()
    log(f"\nTraining plan (V24 — Tiny Sentence Compressor):")
    log(f"  Model: {total:,} params ({total/1e6:.1f}M)")
    log(f"  Bottleneck: {config.num_concepts}x{config.concept_dim} = "
        f"{config.num_concepts * config.concept_dim} dims")
    log(f"  Decoder: {config.dec_layers}L")
    log(f"  Whitening: ON (concept whitening on bottleneck)")
    log(f"  Batch: {BATCH_SIZE}")
    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")
    log(f"  Data: {len(dataset.sources)} sources, streaming")
    log("-" * 70)

    prefetch = PrefetchBuffer(dataset, batch_size=BATCH_SIZE)
    prefetch.start()
    log("Prefetch buffer started")

    loss_tracker = []
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

        batch = prefetch.get()

        with torch.amp.autocast("cuda", dtype=torch.float16):
            m = _unwrap(model) if not hasattr(model, 'encode') else model

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            concepts = m.encode(input_ids, attention_mask)
            logits = m.decode(concepts, input_ids.shape[1], attention_mask)
            loss = reconstruction_loss(logits, input_ids)

        if torch.isnan(loss):
            log(f"NaN loss at step {step}, skipping")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        loss_tracker.append(loss.item())

        # --- Logging ---
        if (step + 1) % LOG_EVERY == 0:
            avg_loss = np.mean(loss_tracker[-100:]) if loss_tracker else 0
            pct = (step + 1) / TOTAL_STEPS * 100
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1 - start_step) / max(elapsed, 1)
            tokens_per_sec = steps_per_sec * BATCH_SIZE * config.max_seq_len
            log(f"step {step+1:>7d} [V24] | en={avg_loss:.4f} | "
                f"em_ema={exact_match_ema:.3f} | "
                f"lr {current_lr:.2e} | {pct:.1f}% | "
                f"{tokens_per_sec/1000:.0f}K tok/s")

        # --- Eval ---
        if (step + 1) % EVAL_EVERY == 0:
            results, acc, em = evaluate_reconstruction(
                model, eval_tokenizer, device)
            exact_match_ema = (EXACT_MATCH_EMA_DECAY * exact_match_ema +
                               (1 - EXACT_MATCH_EMA_DECAY) * em)
            log(f"  EN EVAL: token_acc={acc:.3f} exact_match={em:.3f} "
                f"em_ema={exact_match_ema:.3f}")
            for orig, decoded, tacc, exact in results:
                status = "OK" if exact else "DIFF"
                log(f"    [{status}] ({tacc:.0%}) {orig}")
                if not exact:
                    log(f"           -> {decoded}")

            # Geometry probing
            geo = probe_geometry(model, eval_tokenizer, device)
            log(f"  GEOMETRY (TEST): analogy={geo['analogy_avg']:.3f} "
                f"cluster_gap={geo['clustering_gap']:+.4f} "
                f"dir_con={geo.get('dir_consistency', 0):.3f} "
                f"wo_sim={geo['word_order_sim']:.3f} "
                f"rank90={geo['rank90']} rank95={geo['rank95']}")

            elapsed = time.time() - start_time
            avg_loss = np.mean(loss_tracker[-100:]) if loss_tracker else 0
            metrics_dict = {
                "recon_loss": avg_loss,
                "token_acc": acc, "exact_match": em,
                "em_ema": exact_match_ema, "lr": current_lr,
                "elapsed_hours": elapsed / 3600, "geo": geo,
            }
            log_metrics(step + 1, metrics_dict)

        # --- Checkpoint ---
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = np.mean(loss_tracker[-100:]) if loss_tracker else 0
            save_checkpoint(model, optimizer, scaler, config, step + 1,
                            avg_loss, exact_match_ema, CHECKPOINT_DIR)

        if shutdown_requested:
            break

    prefetch.stop()
    dataset.close()
    if loss_tracker:
        avg_loss = np.mean(loss_tracker[-100:])
        save_checkpoint(model, optimizer, scaler, config,
                        step + 1 if not shutdown_requested else step,
                        avg_loss, exact_match_ema, CHECKPOINT_DIR)
    log("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    train(resume_from=args.resume, fresh=args.fresh, eval_only=args.eval_only)
