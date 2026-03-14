#!/usr/bin/env python3
"""
FLM V26 — Masked Sentence Modeling

Encoder/decoder autoencoder where random content words are masked in the
*decoder target*. The encoder sees the full sentence, the bottleneck must
encode enough meaning for the decoder to reconstruct everything INCLUDING
the masked words.

Unlike plain reconstruction (V24), masking forces the bottleneck to encode
semantic content, not just surface tokens — because the decoder can't just
copy, it must infer masked words from the concept vector.

Loss = reconstruction CE on ALL tokens, with extra weight on masked tokens.
This keeps the bottleneck faithful to the full sentence while adding pressure
to encode meaning.

Usage:
    python train_v26.py --fresh                     # train from scratch
    python train_v26.py --v24-init path/to/v24.pt   # init from V24 checkpoint
    python train_v26.py --resume                     # resume from latest
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
from concept_model import ConceptConfig, ConceptAutoencoderV24, reconstruction_loss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v26"
LOG_DIR = "logs"

# Encoder/decoder config (same as V24)
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
    dropout=0.0,
)

# Masking config
MASK_PROB = 0.25          # probability of masking each content word
MASK_WEIGHT = 5.0         # extra weight on masked token loss
MIN_MASKS = 1             # always mask at least 1 token per sentence
MASK_RAMP_STEPS = 5000    # ramp mask_prob from 0 to full over this many steps

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
    "log": f"{LOG_DIR}/concept_v26.log",
    "metrics": f"{LOG_DIR}/concept_v26_metrics.csv",
}

# Data sources
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
# Masking logic
# ---------------------------------------------------------------------------

# BERT special tokens and common stopwords to avoid masking
STOPWORD_IDS = None  # populated after tokenizer is loaded

def init_stopwords(tokenizer):
    """Build set of token IDs that should NOT be masked (stopwords, special tokens, punctuation)."""
    global STOPWORD_IDS
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'shall', 'can', 'must',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'its', 'our', 'their',
        'this', 'that', 'these', 'those',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
        'and', 'or', 'but', 'not', 'if', 'so', 'than', 'too', 'very',
        '.', ',', '!', '?', ';', ':', '-', "'", '"', '(', ')', '[', ']',
    }
    special_ids = set(tokenizer.all_special_ids)
    stop_ids = set()
    for w in stopwords:
        ids = tokenizer.encode(w, add_special_tokens=False)
        stop_ids.update(ids)
    STOPWORD_IDS = stop_ids | special_ids


def create_mask_targets(input_ids, attention_mask, mask_prob, tokenizer):
    """Create masked targets for a batch of sentences.

    Args:
        input_ids: (B, S) token IDs — encoder sees these unmodified
        attention_mask: (B, S)
        mask_prob: float, probability of masking each content token

    Returns:
        mask_weights: (B, S) per-token loss weights (1.0 for normal, MASK_WEIGHT for masked)
        num_masked: total number of masked tokens in batch
    """
    B, S = input_ids.shape
    mask_weights = torch.ones(B, S, device=input_ids.device)

    num_masked = 0
    for b in range(B):
        # Find content token positions (not special, not stopword, not padding)
        content_positions = []
        for s in range(S):
            if attention_mask[b, s] == 0:
                continue
            tid = input_ids[b, s].item()
            if tid in STOPWORD_IDS:
                continue
            content_positions.append(s)

        if not content_positions:
            continue

        # Randomly select tokens to mask
        n_mask = max(MIN_MASKS, int(len(content_positions) * mask_prob))
        n_mask = min(n_mask, len(content_positions))
        masked_pos = random.sample(content_positions, n_mask)

        for pos in masked_pos:
            mask_weights[b, pos] = MASK_WEIGHT
            num_masked += 1

    return mask_weights, num_masked


def weighted_reconstruction_loss(logits, targets, mask_weights, attention_mask):
    """CE loss with per-token weights.

    Args:
        logits: (B, S, V)
        targets: (B, S)
        mask_weights: (B, S) per-token weights
        attention_mask: (B, S)
    """
    B, S, V = logits.shape

    # Standard CE per token
    loss_per_token = F.cross_entropy(
        logits.view(-1, V), targets.view(-1), reduction='none'
    ).view(B, S)

    # Apply mask weights and attention mask
    weighted = loss_per_token * mask_weights * attention_mask.float()

    # Normalize by total weight (not just count)
    total_weight = (mask_weights * attention_mask.float()).sum()
    if total_weight > 0:
        return weighted.sum() / total_weight
    return weighted.sum()


# ---------------------------------------------------------------------------
# Dataset — single sentences (not windows like V25)
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
    """Streams individual sentences from JSONL document sources."""

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
            doc = json.loads(line)
            return doc.get("text", "").strip()
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
            sents = split_sentences(text)
            self._sent_buf.extend(sents)
        random.shuffle(self._sent_buf)

    def get_batch(self, batch_size):
        """Get a batch of tokenized sentences.

        Returns dict with:
            input_ids: (B, S)
            attention_mask: (B, S)
            raw_sentences: list of strings
        """
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

    state = model.state_dict()
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    ckpt = {
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "metrics": metrics,
        "version": "v26",
        "timestamp": datetime.datetime.now().isoformat(),
    }
    path = os.path.join(checkpoint_dir, f"step_{step:06d}.pt")
    torch.save(ckpt, path)
    latest = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(ckpt, latest)
    log(f"  Checkpoint saved: {path}")

    # Prune old (keep every 50K + last 3)
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
    log(f"Loading V26 checkpoint from {path}...")
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
    metrics = ckpt.get("metrics", {})
    log(f"Resumed V26: step {step}")
    return model, optimizer, scaler, config, step, metrics


def load_v24_init(path, device="cuda"):
    log(f"Initializing from V24 checkpoint: {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoderV24(config).to(device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    total, _ = model.count_parameters()
    log(f"  V24: {total:,} params | loaded from step {ckpt['step']}")
    return model, config


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
        logits = model.decode(c, ids.shape[1], mask)
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

    model.train()
    return em_rate


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(v24_init=None, resume_from=None, fresh=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V26 — MASKED SENTENCE MODELING")
    log("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    init_stopwords(tokenizer)
    log(f"Stopword token IDs: {len(STOPWORD_IDS)}")

    # --- Model setup ---
    if resume_from:
        model, optimizer, scaler, config, start_step, _ = \
            load_checkpoint(resume_from, device)
    else:
        if v24_init:
            model, config = load_v24_init(v24_init, device)
        else:
            config = ConceptConfig(**V24_CONFIG)
            model = ConceptAutoencoderV24(config).to(device)

        total, _ = model.count_parameters()
        log(f"Model: {total:,} params")

        optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR,
                                      betas=BETAS, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0

    total, _ = model.count_parameters()
    log(f"\nArchitecture:")
    log(f"  Encoder/Decoder: {total:,} params")
    log(f"  Bottleneck: {config.num_concepts}x{config.concept_dim} = "
        f"{config.num_concepts * config.concept_dim}d")
    log(f"  Masking: {MASK_PROB:.0%} of content words, weight={MASK_WEIGHT}x")
    log(f"  Mask ramp: 0 -> {MASK_PROB:.0%} over {MASK_RAMP_STEPS} steps")
    log(f"  Batch: {BATCH_SIZE}")
    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")

    # Compile for speed
    model = torch.compile(model)
    log("Model compiled with torch.compile")

    # --- Data ---
    dataset = SentenceDataset(PRETRAIN_SOURCES, tokenizer)
    prefetch = PrefetchBuffer(dataset, batch_size=BATCH_SIZE)
    prefetch.start()
    log("Prefetch started")
    log("-" * 70)

    # --- Training state ---
    loss_tracker = []
    mask_loss_tracker = []
    unmask_loss_tracker = []
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

    for step in range(start_step, TOTAL_STEPS):
        if shutdown_requested:
            break

        current_lr = cosine_lr(step, TOTAL_STEPS, PEAK_LR, WARMUP_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Ramp mask probability
        if step < MASK_RAMP_STEPS:
            current_mask_prob = MASK_PROB * step / MASK_RAMP_STEPS
        else:
            current_mask_prob = MASK_PROB

        optimizer.zero_grad(set_to_none=True)

        batch = prefetch.get()

        with torch.amp.autocast("cuda", dtype=torch.float16):
            input_ids = batch["input_ids"].to(device)      # (B, S)
            attn_mask = batch["attention_mask"].to(device)  # (B, S)

            # Encoder sees full sentence
            concepts = model.encode(input_ids, attn_mask)  # (B, K, D)

            # Decoder must reconstruct full sentence
            logits = model.decode(concepts, input_ids.shape[1], attn_mask)  # (B, S, V)

            # Create mask weights (higher weight on masked content tokens)
            mask_weights, num_masked = create_mask_targets(
                input_ids, attn_mask, current_mask_prob, tokenizer
            )

            # Weighted reconstruction loss
            loss = weighted_reconstruction_loss(logits, input_ids, mask_weights, attn_mask)

            # Also compute separate masked/unmasked loss for monitoring
            with torch.no_grad():
                B, S, V = logits.shape
                per_token = F.cross_entropy(
                    logits.view(-1, V), input_ids.view(-1), reduction='none'
                ).view(B, S)
                is_masked = (mask_weights > 1.5).float()
                is_unmasked = (mask_weights <= 1.5).float() * attn_mask.float()
                masked_sum = (per_token * is_masked).sum()
                masked_count = is_masked.sum()
                unmasked_sum = (per_token * is_unmasked).sum()
                unmasked_count = is_unmasked.sum()

                masked_loss = (masked_sum / masked_count).item() if masked_count > 0 else 0
                unmasked_loss = (unmasked_sum / unmasked_count).item() if unmasked_count > 0 else 0

            # Exact match tracking (on unmasked reconstruction)
            with torch.no_grad():
                preds = logits.argmax(dim=-1)  # (B, S)
                for b in range(input_ids.shape[0]):
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

        loss_tracker.append(loss.item())
        mask_loss_tracker.append(masked_loss)
        unmask_loss_tracker.append(unmasked_loss)

        # --- Logging ---
        if (step + 1) % LOG_EVERY == 0:
            avg_loss = np.mean(loss_tracker[-100:])
            avg_masked = np.mean(mask_loss_tracker[-100:])
            avg_unmasked = np.mean(unmask_loss_tracker[-100:])
            pct = (step + 1) / TOTAL_STEPS * 100
            elapsed = time.time() - start_time
            sps = (step + 1 - start_step) / max(elapsed, 1)
            log(f"step {step+1:>7d} [V26] | loss={avg_loss:.4f} "
                f"masked={avg_masked:.4f} unmasked={avg_unmasked:.4f} | "
                f"em={em_ema:.3f} mask_p={current_mask_prob:.2f} | "
                f"lr {current_lr:.2e} | {pct:.1f}% | {sps:.1f} step/s")

            log_metrics(step + 1, {
                "loss": avg_loss,
                "masked_loss": avg_masked,
                "unmasked_loss": avg_unmasked,
                "em_ema": em_ema,
                "mask_prob": current_mask_prob,
                "lr": current_lr,
            })

        # --- Eval ---
        if (step + 1) % EVAL_EVERY == 0:
            log("\n--- EVALUATION ---")
            evaluate(model, tokenizer, device)
            log("")

        # --- Checkpoint ---
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = np.mean(loss_tracker[-100:])
            avg_masked = np.mean(mask_loss_tracker[-100:])
            avg_unmasked = np.mean(unmask_loss_tracker[-100:])
            save_checkpoint(model, optimizer, scaler, config, step + 1,
                          {"loss": avg_loss, "masked_loss": avg_masked,
                           "unmasked_loss": avg_unmasked, "em_ema": em_ema},
                          CHECKPOINT_DIR)

    # Final save
    if loss_tracker:
        avg_loss = np.mean(loss_tracker[-100:])
        save_checkpoint(model, optimizer, scaler, config, step + 1,
                      {"loss": avg_loss, "em_ema": em_ema},
                      CHECKPOINT_DIR)

    prefetch.stop()
    log("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLM V26 — Masked Sentence Modeling")
    parser.add_argument("--fresh", action="store_true", help="Train from scratch")
    parser.add_argument("--v24-init", type=str, help="Initialize from V24 checkpoint")
    parser.add_argument("--resume", type=str, help="Resume V26 training")
    args = parser.parse_args()

    if not args.fresh and not args.v24_init and not args.resume:
        latest = Path(CHECKPOINT_DIR) / "latest.pt"
        if latest.exists():
            args.resume = str(latest)

    train(v24_init=args.v24_init, resume_from=args.resume, fresh=args.fresh)
