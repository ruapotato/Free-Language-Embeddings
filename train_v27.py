#!/usr/bin/env python3
"""
FLM V27 — Contrastive Sentence Autoencoder

Reconstruction + SimCSE contrastive + VICReg regularization, trained jointly
from scratch. Based on research findings:

- SimCSE: encode same sentence twice with different dropout → positive pair,
  rest of batch → negatives. Creates alignment + uniformity on hypersphere.
- VICReg: variance (prevent collapse) + covariance (decorrelate dims) on
  bottleneck, no whitening layer needed.
- L2 normalization on bottleneck output (forces unit hypersphere).
- Frequency-weighted reconstruction (downweight stopwords).
- Reconstruction keeps encoder grounded; contrastive shapes geometry.

Architecture: same encoder/bottleneck/decoder as V24 (25.9M params)
Loss: L_recon + alpha*L_contrastive + beta*L_variance + gamma*L_covariance

Usage:
    python train_v27.py --fresh
    python train_v27.py --resume
"""

import os
import re
import json
import math
import time
import random
import signal
import queue
import datetime
import threading
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer
from concept_model import ConceptConfig, ConceptAutoencoderV24

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v27"
LOG_DIR = "logs"

V24_CONFIG = dict(
    vocab_size=30522,
    enc_hidden=256,
    enc_layers=4,
    enc_heads=4,
    enc_intermediate=1024,
    num_concepts=64,
    concept_dim=16,
    dec_hidden=256,
    dec_layers=4,
    dec_heads=4,
    dec_intermediate=1024,
    max_seq_len=64,
    dropout=0.3,  # higher dropout for stronger SimCSE augmentation
)

CONCEPT_DIM = V24_CONFIG["num_concepts"] * V24_CONFIG["concept_dim"]  # 1024

# Loss weights
RECON_WEIGHT = 1.0
CONTRASTIVE_WEIGHT = 0.1       # SimCSE InfoNCE, ramped up
CONTRASTIVE_RAMP = 5000        # ramp contrastive from 0 to full over N steps
CONTRASTIVE_TEMP = 0.05        # temperature for InfoNCE
VICREG_VAR_WEIGHT = 0.1        # variance regularization
VICREG_COV_WEIGHT = 0.04       # covariance regularization

# Training
BATCH_SIZE = 64
PEAK_LR = 3e-4
MIN_LR = 1e-5
WARMUP_STEPS = 2000
TOTAL_STEPS = 400_000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0

LOG_EVERY = 50
EVAL_EVERY = 2000
CHECKPOINT_EVERY = 5000

LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v27.log",
    "metrics": f"{LOG_DIR}/concept_v27_metrics.csv",
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
# Token frequency weights for reconstruction loss
# ---------------------------------------------------------------------------

FREQ_WEIGHTS = None  # (vocab_size,) tensor, populated after tokenizer loaded

def init_freq_weights(tokenizer, device):
    """Build per-token reconstruction weights. Downweight top-100 most common tokens."""
    global FREQ_WEIGHTS
    # Approximate: tokens with short surface forms or common words get lower weight
    # We use a simple heuristic: special tokens and single-char tokens get 0.1 weight
    weights = torch.ones(tokenizer.vocab_size, device=device)

    # Downweight special tokens
    for tid in tokenizer.all_special_ids:
        weights[tid] = 0.1

    # Downweight common stopwords
    stopwords = [
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'am', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'must',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'my', 'your', 'his', 'its', 'our', 'their',
        'this', 'that', 'these', 'those',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
        'and', 'or', 'but', 'not', 'if', 'so', 'than', 'too', 'very',
        '.', ',', '!', '?', ';', ':', '-', "'", '"',
    ]
    for w in stopwords:
        ids = tokenizer.encode(w, add_special_tokens=False)
        for tid in ids:
            weights[tid] = 0.3  # not zero, still reconstruct, just less weight

    FREQ_WEIGHTS = weights
    n_down = (weights < 1.0).sum().item()
    log(f"Freq weights: {n_down} tokens downweighted out of {tokenizer.vocab_size}")


def freq_weighted_recon_loss(logits, targets, attention_mask):
    """Reconstruction CE with per-token frequency weights."""
    B, S, V = logits.shape
    loss_per_token = F.cross_entropy(
        logits.view(-1, V), targets.view(-1), reduction='none'
    ).view(B, S)  # (B, S)

    # Per-token weights from frequency table
    token_weights = FREQ_WEIGHTS[targets]  # (B, S)

    weighted = loss_per_token * token_weights * attention_mask.float()
    total_weight = (token_weights * attention_mask.float()).sum()
    if total_weight > 0:
        return weighted.sum() / total_weight
    return weighted.sum()


# ---------------------------------------------------------------------------
# Contrastive loss (SimCSE-style InfoNCE)
# ---------------------------------------------------------------------------

def simcse_loss(z1, z2, temperature=CONTRASTIVE_TEMP):
    """InfoNCE contrastive loss between two views of the same batch.

    z1, z2: (B, D) L2-normalized embeddings from two forward passes
            with different dropout masks.

    Positive pairs: (z1[i], z2[i]) — same sentence, different dropout
    Negative pairs: all other combinations in the batch
    """
    B = z1.shape[0]

    # Similarity matrix: (B, B) for z1 vs z2
    sim = z1 @ z2.T / temperature  # (B, B)

    # Labels: diagonal (each z1[i] matches z2[i])
    labels = torch.arange(B, device=z1.device)

    # Cross-entropy: row i should have max at column i
    loss = F.cross_entropy(sim, labels)

    return loss


# ---------------------------------------------------------------------------
# VICReg losses (variance + covariance)
# ---------------------------------------------------------------------------

def vicreg_variance_loss(z, target_std=1.0):
    """Encourage each dimension to have at least target_std variance.
    Prevents collapse where all embeddings become identical.

    z: (B, D) embeddings
    """
    std = z.std(dim=0)  # (D,)
    # Hinge loss: only penalize if std < target
    loss = F.relu(target_std - std).mean()
    return loss


def vicreg_covariance_loss(z):
    """Penalize off-diagonal covariance (decorrelate dimensions).

    z: (B, D) embeddings
    """
    B, D = z.shape
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (B - 1)  # (D, D)

    # Zero out diagonal (we don't penalize self-correlation)
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0)

    # Penalize squared off-diagonal elements
    loss = (off_diag ** 2).sum() / D
    return loss


# ---------------------------------------------------------------------------
# Dataset — single sentences
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text):
    sents = _SENT_RE.split(text.strip())
    result = []
    for s in sents:
        s = s.strip()
        words = len(s.split())
        if 3 <= words <= 50:
            result.append(s)
    return result


class SentenceDataset:
    def __init__(self, sources, tokenizer, max_seq_len=V24_CONFIG["max_seq_len"]):
        self.sources = [(p, w) for p, w in sources if os.path.exists(p)]
        if not self.sources:
            raise FileNotFoundError("No data sources found")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        total_w = sum(w for _, w in self.sources)
        self._cum_weights = []
        cum = 0
        for _, w in self.sources:
            cum += w / total_w
            self._cum_weights.append(cum)
        self._files = [open(p) for p, _ in self.sources]
        self._sent_buf = []
        self._MIN_BUF = 2000
        log(f"SentenceDataset: {len(self.sources)} sources")

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

    def _refill_buffer(self):
        attempts = 0
        while len(self._sent_buf) < self._MIN_BUF and attempts < 50000:
            attempts += 1
            src = self._sample_source()
            text = self._read_doc(src)
            if not text or len(text) < 30:
                continue
            self._sent_buf.extend(split_sentences(text))
        random.shuffle(self._sent_buf)

    def get_batch(self, batch_size):
        while len(self._sent_buf) < batch_size:
            self._refill_buffer()
        sents = [self._sent_buf.pop() for _ in range(batch_size)]
        enc = self.tokenizer(
            sents, max_length=self.max_seq_len,
            padding='max_length', truncation=True, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "raw_sentences": sents,
        }


class PrefetchBuffer:
    def __init__(self, dataset, batch_size=64, buf_size=8):
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

def save_checkpoint(model, optimizer, scaler, config, step, metrics, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    ckpt = {
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "metrics": metrics,
        "version": "v27",
        "timestamp": datetime.datetime.now().isoformat(),
    }
    path = os.path.join(checkpoint_dir, f"step_{step:06d}.pt")
    torch.save(ckpt, path)
    torch.save(ckpt, os.path.join(checkpoint_dir, "latest.pt"))
    log(f"  Checkpoint saved: {path}")

    all_ckpts = sorted(Path(checkpoint_dir).glob("step_*.pt"))
    to_keep = set()
    for c in all_ckpts:
        sn = int(c.stem.split("_")[1])
        if sn % 50000 == 0:
            to_keep.add(c)
    for c in all_ckpts[-3:]:
        to_keep.add(c)
    for c in all_ckpts:
        if c not in to_keep:
            c.unlink()


def load_checkpoint(path, device="cuda"):
    log(f"Loading V27 checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoderV24(config).to(device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR,
                                  betas=BETAS, weight_decay=WEIGHT_DECAY)
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler = torch.amp.GradScaler("cuda")
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    step = ckpt["step"]
    log(f"Resumed V27: step {step}")
    return model, optimizer, scaler, config, step


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, tokenizer, device="cuda"):
    model.eval()
    test_sents = [
        "the cat sat on the mat.",
        "artificial intelligence will change the world.",
        "she runs every morning before breakfast.",
        "the quick brown fox jumps over the lazy dog.",
        "water boils at one hundred degrees celsius.",
        "he played guitar at the concert last night.",
        "the president gave a speech about climate change.",
        "please pass the salt and pepper.",
        "the children played happily in the park.",
        "birds fly south when winter approaches.",
    ]
    enc = tokenizer(test_sents, max_length=V24_CONFIG["max_seq_len"],
                    padding='max_length', truncation=True, return_tensors="pt").to(device)
    exact = 0
    for i in range(len(test_sents)):
        ids = enc["input_ids"][i:i+1]
        mask = enc["attention_mask"][i:i+1]
        c = model.encode(ids, mask)
        # L2 normalize for consistency
        c_flat = c.view(1, -1)
        c_norm = F.normalize(c_flat, p=2, dim=-1)
        c_3d = c_norm.view(1, V24_CONFIG["num_concepts"], V24_CONFIG["concept_dim"])
        logits = model.decode(c_3d, ids.shape[1], mask)
        pred = logits.argmax(dim=-1)[0]
        pred_mask = mask[0].bool()
        decoded = tokenizer.decode(pred[pred_mask], skip_special_tokens=True)
        match = decoded.strip() == test_sents[i].strip()
        exact += int(match)
        status = "OK" if match else "DIFF"
        log(f"  [{status}] {test_sents[i]}")
        if not match:
            log(f"       -> {decoded}")
    em_rate = exact / len(test_sents)
    log(f"  Exact match: {exact}/{len(test_sents)} ({em_rate:.0%})")

    # Quick geometry check: within vs between group similarity
    groups = {
        "animals": ["the cat is sleeping.", "the dog is running.", "birds fly south."],
        "weather": ["it is raining outside.", "the sun is shining.", "a storm is coming."],
        "food": ["the pizza was great.", "she cooked dinner.", "i love cake."],
    }
    all_vecs = []
    all_labels = []
    for label, sents in groups.items():
        e = tokenizer(sents, max_length=64, padding=True, truncation=True, return_tensors="pt").to(device)
        for j in range(len(sents)):
            c = model.encode(e["input_ids"][j:j+1], e["attention_mask"][j:j+1])
            v = F.normalize(c.view(1, -1), p=2, dim=-1)[0]
            all_vecs.append(v)
            all_labels.append(label)
    vecs = torch.stack(all_vecs)
    sim_mat = vecs @ vecs.T
    within, between = [], []
    for i in range(len(all_vecs)):
        for j in range(i+1, len(all_vecs)):
            s = sim_mat[i, j].item()
            if all_labels[i] == all_labels[j]:
                within.append(s)
            else:
                between.append(s)
    gap = np.mean(within) - np.mean(between) if within and between else 0
    log(f"  Geometry: within={np.mean(within):.3f} between={np.mean(between):.3f} gap={gap:+.3f}")

    model.train()
    return em_rate, gap


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V27 — CONTRASTIVE SENTENCE AUTOENCODER")
    log("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    init_freq_weights(tokenizer, device)

    # --- Model setup ---
    if resume_from:
        model, optimizer, scaler, config, start_step = \
            load_checkpoint(resume_from, device)
    else:
        config = ConceptConfig(**V24_CONFIG)
        model = ConceptAutoencoderV24(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR,
                                      betas=BETAS, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0

    total, _ = model.count_parameters()
    log(f"\nArchitecture:")
    log(f"  Encoder/Decoder: {total:,} params")
    log(f"  Bottleneck: {config.num_concepts}x{config.concept_dim} = {CONCEPT_DIM}d + L2 norm")
    log(f"  Dropout: {config.dropout} (used for SimCSE augmentation)")
    log(f"\nLosses:")
    log(f"  Reconstruction: weight={RECON_WEIGHT} (freq-weighted)")
    log(f"  Contrastive (SimCSE): weight={CONTRASTIVE_WEIGHT}, temp={CONTRASTIVE_TEMP}, ramp={CONTRASTIVE_RAMP} steps")
    log(f"  VICReg variance: weight={VICREG_VAR_WEIGHT}")
    log(f"  VICReg covariance: weight={VICREG_COV_WEIGHT}")
    log(f"\nTraining:")
    log(f"  Batch: {BATCH_SIZE}")
    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")

    # Compile
    model = torch.compile(model)
    log("Model compiled")

    # --- Data ---
    dataset = SentenceDataset(PRETRAIN_SOURCES, tokenizer)
    prefetch = PrefetchBuffer(dataset, batch_size=BATCH_SIZE)
    prefetch.start()
    log("Prefetch started")
    log("-" * 70)

    # --- Training state ---
    recon_tracker = []
    contrastive_tracker = []
    var_tracker = []
    cov_tracker = []
    em_ema = 0.0
    start_time = time.time()
    nan_count = 0

    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        log("Shutdown signal received, saving checkpoint...")
        shutdown_requested = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    K = V24_CONFIG["num_concepts"]
    D = V24_CONFIG["concept_dim"]

    for step in range(start_step, TOTAL_STEPS):
        if shutdown_requested:
            break

        current_lr = cosine_lr(step, TOTAL_STEPS, PEAK_LR, WARMUP_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Ramp contrastive weight
        if step < CONTRASTIVE_RAMP:
            ctr_weight = CONTRASTIVE_WEIGHT * step / CONTRASTIVE_RAMP
        else:
            ctr_weight = CONTRASTIVE_WEIGHT

        optimizer.zero_grad(set_to_none=True)
        batch = prefetch.get()

        with torch.amp.autocast("cuda", dtype=torch.float16):
            input_ids = batch["input_ids"].to(device)      # (B, S)
            attn_mask = batch["attention_mask"].to(device)  # (B, S)
            B, S = input_ids.shape

            # === Forward pass 1 (with dropout) ===
            model.train()  # ensure dropout is active
            concepts_1 = model.encode(input_ids, attn_mask)  # (B, K, D)
            z1 = F.normalize(concepts_1.view(B, -1), p=2, dim=-1)  # (B, CONCEPT_DIM)

            # === Forward pass 2 (same input, different dropout + noise) ===
            concepts_2 = model.encode(input_ids, attn_mask)  # (B, K, D)
            # Add small Gaussian noise to create stronger augmentation
            noise = torch.randn_like(concepts_2) * 0.1
            concepts_2 = concepts_2 + noise
            z2 = F.normalize(concepts_2.view(B, -1), p=2, dim=-1)  # (B, CONCEPT_DIM)

            # === Reconstruction loss (from first encoding) ===
            concepts_1_norm = z1.view(B, K, D)
            logits = model.decode(concepts_1_norm, S, attn_mask)  # (B, S, V)
            recon_loss = freq_weighted_recon_loss(logits, input_ids, attn_mask)

            # === Contrastive loss (SimCSE) ===
            ctr_loss = simcse_loss(z1, z2, CONTRASTIVE_TEMP)

            # === VICReg losses (on z1, could also average with z2) ===
            # Cast to float32 for stability in variance/covariance computation
            z1_f32 = z1.float()
            var_loss = vicreg_variance_loss(z1_f32)
            cov_loss = vicreg_covariance_loss(z1_f32)

            # === Total loss ===
            loss = (RECON_WEIGHT * recon_loss +
                    ctr_weight * ctr_loss +
                    VICREG_VAR_WEIGHT * var_loss +
                    VICREG_COV_WEIGHT * cov_loss)

            # Exact match tracking
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                for b in range(B):
                    valid = attn_mask[b].bool()
                    match = (preds[b][valid] == input_ids[b][valid]).all().item()
                    em_ema = 0.999 * em_ema + 0.001 * float(match)

        if torch.isnan(loss):
            nan_count += 1
            if nan_count <= 3 or nan_count % 100 == 0:
                log(f"NaN loss at step {step} ({nan_count} consecutive)")
            if nan_count >= 50:
                log("Too many NaN losses, stopping")
                break
            continue

        nan_count = 0
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        recon_tracker.append(recon_loss.item())
        contrastive_tracker.append(ctr_loss.item())
        var_tracker.append(var_loss.item())
        cov_tracker.append(cov_loss.item())

        # --- Logging ---
        if (step + 1) % LOG_EVERY == 0:
            avg_recon = np.mean(recon_tracker[-100:])
            avg_ctr = np.mean(contrastive_tracker[-100:])
            avg_var = np.mean(var_tracker[-100:])
            avg_cov = np.mean(cov_tracker[-100:])
            pct = (step + 1) / TOTAL_STEPS * 100
            elapsed = time.time() - start_time
            sps = (step + 1 - start_step) / max(elapsed, 1)
            log(f"step {step+1:>7d} [V27] | recon={avg_recon:.4f} ctr={avg_ctr:.4f} "
                f"var={avg_var:.4f} cov={avg_cov:.4f} | "
                f"em={em_ema:.3f} ctr_w={ctr_weight:.3f} | "
                f"lr {current_lr:.2e} | {pct:.1f}% | {sps:.1f} step/s")

            log_metrics(step + 1, {
                "recon_loss": avg_recon,
                "contrastive_loss": avg_ctr,
                "variance_loss": avg_var,
                "covariance_loss": avg_cov,
                "em_ema": em_ema,
                "contrastive_weight": ctr_weight,
                "lr": current_lr,
            })

        # --- Eval ---
        if (step + 1) % EVAL_EVERY == 0:
            log("\n--- EVALUATION ---")
            evaluate(model, tokenizer, device)
            log("")

        # --- Checkpoint ---
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_recon = np.mean(recon_tracker[-100:])
            avg_ctr = np.mean(contrastive_tracker[-100:])
            save_checkpoint(model, optimizer, scaler, config, step + 1,
                          {"recon": avg_recon, "contrastive": avg_ctr, "em_ema": em_ema},
                          CHECKPOINT_DIR)

    if recon_tracker:
        avg_recon = np.mean(recon_tracker[-100:])
        save_checkpoint(model, optimizer, scaler, config, step + 1,
                      {"recon": avg_recon, "em_ema": em_ema},
                      CHECKPOINT_DIR)

    prefetch.stop()
    log("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLM V27 — Contrastive Sentence Autoencoder")
    parser.add_argument("--fresh", action="store_true", help="Train from scratch")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    args = parser.parse_args()

    if not args.fresh and not args.resume:
        latest = Path(CHECKPOINT_DIR) / "latest.pt"
        if latest.exists():
            args.resume = str(latest)

    train(resume_from=args.resume, fresh=args.fresh)
