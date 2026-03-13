"""
FLM V23 — Contrastive Concept Encoder (SimCSE-style)
================================================================================
Encoder-only architecture trained with contrastive learning. No decoder.
The concept bottleneck is directly optimized as a semantic representation space.

Architecture: 12L encoder (768d) → 32×16=512 bottleneck + whitening → concept vector
Training: InfoNCE loss with dropout augmentation (same text, two dropout masks)
Data: Wikipedia + Gutenberg + StackExchange + arXiv + USGPO (~300GB, streaming)

Key changes from V22:
- No decoder — all params go into the encoder (253M, 24L)
- Contrastive loss instead of reconstruction loss
- Gradient checkpointing → batch 512 (massive negatives for contrastive)
- Directly optimizes concept space geometry

Usage:
    python train_v23.py --fresh            # start from scratch
    python train_v23.py                    # auto-resume
    python train_v23.py --eval-only        # diagnostics only
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
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from pathlib import Path
from concept_model import (ConceptConfig, ConceptEncoderV23,
                           flat_similarity_matrix)
from geometry_data import GeometryDataGenerator, verify_splits

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v23"
LOG_DIR = "logs"

MODEL_CONFIG = dict(
    vocab_size=30522,       # bert-base-uncased (English-only)
    enc_hidden=768,
    enc_layers=12,
    enc_heads=12,
    enc_intermediate=3072,
    num_concepts=32,
    concept_dim=16,         # 32×16 = 512 bottleneck
    max_seq_len=128,
    dropout=0.1,
)

# Training hyperparameters
BATCH_SIZE = 256            # big batch for contrastive learning (more negatives = better)
PEAK_LR = 1e-4             # conservative for contrastive
MIN_LR = 1e-6
WARMUP_STEPS = 2000
TOTAL_STEPS = 600_000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0

# InfoNCE
TEMPERATURE = 0.05         # standard for SimCSE

# Logging
LOG_EVERY = 50
EVAL_EVERY = 5000
CHECKPOINT_EVERY = 5000

LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v23.log",
    "metrics": f"{LOG_DIR}/concept_v23_metrics.csv",
}


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


METRICS_HEADER = ("timestamp,step,contrastive_loss,alignment,uniformity,"
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
        f.write(f"{ts},{step},{m.get('contrastive_loss',0):.6f},"
                f"{m.get('alignment',0):.4f},{m.get('uniformity',0):.4f},"
                f"{m.get('lr',0):.6e},"
                f"{m.get('elapsed_hours',0):.4f},"
                f"{g.get('analogy_avg',0):.4f},{g.get('clustering_gap',0):.4f},"
                f"{g.get('dir_consistency', g.get('direction_consistency', 0)):.4f},"
                f"{g.get('word_order_sim',0):.4f},"
                f"{g.get('rank90',0)},{g.get('rank95',0)}\n")


# ---------------------------------------------------------------------------
# Streaming data loader — reads from massive JSONL files
# ---------------------------------------------------------------------------

# Pretrain data sources (same as V21/V22)
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


class StreamingPretrainDataset:
    """Streams text from multiple JSONL pretrain files with weighted sampling."""

    def __init__(self, tokenizer, max_len=128):
        self.tok = tokenizer
        self.max_len = max_len
        self.sources = []
        self.weights = []
        self.cum_weights = []

        for path, weight in PRETRAIN_SOURCES:
            if os.path.exists(path):
                self.sources.append(path)
                self.weights.append(weight)
                log(f"  {os.path.basename(path)}: weight={weight:.0%}")
            else:
                log(f"  {os.path.basename(path)}: NOT FOUND, skipping")

        if not self.sources:
            raise RuntimeError("No pretrain data found!")

        total_w = sum(self.weights)
        self.weights = [w / total_w for w in self.weights]
        cum = 0
        for w in self.weights:
            cum += w
            self.cum_weights.append(cum)

        self._handles = [None] * len(self.sources)
        log(f"  {len(self.sources)} sources loaded")

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

    def _get_text(self, source_idx):
        for _ in range(10):
            line = self._read_line(source_idx)
            if line is None:
                return None
            try:
                doc = json.loads(line)
                text = doc.get("text", "").strip()
                if len(text) > 20:
                    return text
            except (json.JSONDecodeError, KeyError):
                continue
        return None

    def _chunk_text(self, text):
        """Extract a random ~max_len token chunk using character-level estimation.
        ~5 chars per BERT-uncased token on average."""
        max_chars = self.max_len * 5
        if len(text) <= max_chars:
            return text
        start = random.randint(0, len(text) - max_chars)
        space_pos = text.find(" ", start)
        if space_pos != -1 and space_pos < start + 50:
            start = space_pos + 1
        chunk = text[start:start + max_chars]
        last_space = chunk.rfind(" ")
        if last_space > max_chars // 2:
            chunk = chunk[:last_space]
        return chunk.strip()

    def get_batch(self, batch_size):
        texts = []
        while len(texts) < batch_size:
            source_idx = self._sample_source()
            text = self._get_text(source_idx)
            if text is None:
                continue
            chunk = self._chunk_text(text)
            if len(chunk) > 10:
                texts.append(chunk)

        enc = self.tok(texts, max_length=self.max_len,
                       padding=True, truncation=True, return_tensors="pt")
        return enc

    def close(self):
        for h in self._handles:
            if h is not None:
                h.close()


class PrefetchBuffer:
    """Pre-generates batches in background thread."""

    def __init__(self, dataset, batch_size=512, buf_size=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.q = queue.Queue(maxsize=buf_size)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._fill, daemon=True,
                                        name="v23-prefetch")
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
            except Exception:
                continue

    def get(self):
        return self.q.get()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(model):
    return model._orig_mod if hasattr(model, '_orig_mod') else model


def enable_gradient_checkpointing(model):
    """Replace model.encode with gradient-checkpointed version."""
    m = _unwrap(model)
    original_encode = m.encode

    def checkpointed_encode(input_ids, attention_mask=None):
        bsz, seq_len = input_ids.shape
        x = m.embed_tokens(input_ids)
        attn_mask = None
        if attention_mask is not None:
            pad_mask = torch.zeros(bsz, 1, 1, seq_len, device=x.device, dtype=x.dtype)
            pad_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))
            attn_mask = pad_mask
        for layer in m.enc_layers:
            x = grad_checkpoint(layer, x, m.enc_rope_cos, m.enc_rope_sin,
                                attn_mask, use_reentrant=False)
        x = m.enc_norm(x)
        concepts = m.bottleneck(x)
        bsz2, K, D = concepts.shape
        flat = concepts.reshape(bsz2, -1)
        whitened = m.whitening(flat)
        return whitened.reshape(bsz2, K, D)

    m.encode = checkpointed_encode
    log("Gradient checkpointing enabled")


def info_nce_loss(z1, z2, temperature=TEMPERATURE):
    """
    InfoNCE contrastive loss (SimCSE-style).
    z1, z2: (batch, dim) L2-normalized concept vectors
    Each z1[i] should be close to z2[i] (positive pair),
    and far from z2[j] for j != i (negatives).
    """
    # Similarity matrix: (batch, batch)
    sim = z1 @ z2.T / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    # Cross-entropy in both directions
    loss_12 = F.cross_entropy(sim, labels)
    loss_21 = F.cross_entropy(sim.T, labels)
    return (loss_12 + loss_21) / 2


@torch.no_grad()
def compute_alignment_uniformity(z1, z2):
    """
    Alignment: mean L2 distance between positive pairs (lower = better).
    Uniformity: how spread out the representations are (lower = more uniform).
    From Wang & Isola, "Understanding Contrastive Representation Learning".
    """
    # Alignment: avg distance between positive pairs
    alignment = (z1 - z2).norm(dim=-1).pow(2).mean().item()
    # Uniformity: log of avg pairwise gaussian kernel
    z = torch.cat([z1, z2], dim=0)
    sq_pdist = torch.cdist(z, z).pow(2)
    n = z.shape[0]
    # Mask diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    uniformity = torch.log(torch.exp(-2 * sq_pdist[mask]).mean()).item()
    return alignment, uniformity


# ---------------------------------------------------------------------------
# Geometry probing (same probes as V19-V22)
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
                    checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    m = _unwrap(model)
    state = m.state_dict()
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    ckpt = {
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "loss": loss,
        "version": "v23",
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
    log(f"Loading V23 checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        config = ConceptConfig(**cfg)
    else:
        config = cfg
    model = ConceptEncoderV23(config).to(device)
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
    total, _ = model.count_parameters()
    log(f"Resumed V23: {total:,} params | step {step}")
    return model, optimizer, scaler, config, step


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False, eval_only=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V23 — CONTRASTIVE CONCEPT ENCODER (SimCSE-style)")
    log(f"  Encoder: {MODEL_CONFIG['enc_layers']}L ({MODEL_CONFIG['enc_hidden']}d)")
    log(f"  Bottleneck: 32x16 = 512 dims | Concept whitening")
    log("  Loss: InfoNCE with dropout augmentation")
    log("  Data: Wikipedia + Gutenberg + StackExchange + arXiv + USGPO")
    log("  No decoder — all params in encoder, concept space directly optimized")
    log("=" * 70)

    verify_splits()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    log(f"BERT-uncased tokenizer: vocab={tokenizer.vocab_size}")

    model_config = dict(MODEL_CONFIG)
    model_config["vocab_size"] = tokenizer.vocab_size

    if resume_from is None and not fresh:
        latest = Path(CHECKPOINT_DIR) / "latest.pt"
        if latest.exists():
            resume_from = str(latest)

    if resume_from:
        model, optimizer, scaler, config, start_step = \
            load_checkpoint(resume_from, device)
    else:
        log("Starting fresh training...")
        config = ConceptConfig(**model_config)
        model = ConceptEncoderV23(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=PEAK_LR, betas=BETAS,
            weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0
        total, _ = model.count_parameters()
        log(f"Model: {total:,} params ({total/1e6:.1f}M)")

    # Enable gradient checkpointing for memory efficiency
    enable_gradient_checkpointing(model)

    if eval_only:
        log("\n--- GEOMETRY (TEST) ---")
        geo = probe_geometry(model, tokenizer, device)
        for k, v in geo.items():
            log(f"  {k}: {v}")
        return

    model.train()

    log("Loading data...")
    dataset = StreamingPretrainDataset(tokenizer, max_len=config.max_seq_len)

    total, _ = _unwrap(model).count_parameters()
    log(f"\nTraining plan (V23 — Contrastive):")
    log(f"  Model: {total:,} params ({total/1e6:.1f}M)")
    log(f"  Bottleneck: {config.num_concepts}x{config.concept_dim} = "
        f"{config.num_concepts * config.concept_dim} dims")
    log(f"  Temperature: {TEMPERATURE}")
    log(f"  Batch: {BATCH_SIZE} (= {BATCH_SIZE} positives + {BATCH_SIZE} negatives)")
    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")
    log(f"  Data: {len(dataset.sources)} sources, streaming")
    log("-" * 70)

    prefetch = PrefetchBuffer(dataset, batch_size=BATCH_SIZE, buf_size=4)
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
            m = _unwrap(model)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            # Two forward passes with different dropout → positive pairs
            z1 = m.concept_vector(input_ids, attention_mask)
            z2 = m.concept_vector(input_ids, attention_mask)

            loss = info_nce_loss(z1, z2)

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
            log(f"step {step+1:>7d} [V23] | cl={avg_loss:.4f} | "
                f"lr {current_lr:.2e} | {pct:.1f}% | "
                f"{tokens_per_sec/1000:.0f}K tok/s")

        # --- Eval ---
        if (step + 1) % EVAL_EVERY == 0:
            model.eval()

            # Compute alignment & uniformity on a sample
            sample_batch = prefetch.get()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                s_ids = sample_batch["input_ids"][:64].to(device)
                s_mask = sample_batch["attention_mask"][:64].to(device)
                sz1 = m.concept_vector(s_ids, s_mask)
                sz2 = m.concept_vector(s_ids, s_mask)
            alignment, uniformity = compute_alignment_uniformity(
                sz1.float(), sz2.float())
            log(f"  CONTRASTIVE EVAL: alignment={alignment:.4f} "
                f"uniformity={uniformity:.4f}")

            # Geometry probing
            geo = probe_geometry(model, tokenizer, device)
            log(f"  GEOMETRY (TEST): analogy={geo['analogy_avg']:.3f} "
                f"cluster_gap={geo['clustering_gap']:+.4f} "
                f"dir_con={geo.get('dir_consistency', 0):.3f} "
                f"wo_sim={geo['word_order_sim']:.3f} "
                f"rank90={geo['rank90']} rank95={geo['rank95']}")

            elapsed = time.time() - start_time
            avg_loss = np.mean(loss_tracker[-100:]) if loss_tracker else 0
            metrics_dict = {
                "contrastive_loss": avg_loss,
                "alignment": alignment, "uniformity": uniformity,
                "lr": current_lr,
                "elapsed_hours": elapsed / 3600, "geo": geo,
            }
            log_metrics(step + 1, metrics_dict)

            model.train()

        # --- Checkpoint ---
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = np.mean(loss_tracker[-100:]) if loss_tracker else 0
            save_checkpoint(model, optimizer, scaler, config, step + 1,
                            avg_loss, CHECKPOINT_DIR)

        if shutdown_requested:
            break

    prefetch.stop()
    dataset.close()
    if loss_tracker:
        avg_loss = np.mean(loss_tracker[-100:])
        save_checkpoint(model, optimizer, scaler, config,
                        step + 1 if not shutdown_requested else step,
                        avg_loss, CHECKPOINT_DIR)
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
