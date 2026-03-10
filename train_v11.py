"""
FLM V11 — Non-Autoregressive Concept Autoencoder (Pure Reconstruction)
================================================================================
Key insight: the parallel decoder forces concept vectors to encode ALL meaning.
Geometry (analogies, word-order sensitivity, slot structure) emerges naturally
from reconstruction alone — no explicit geometry losses needed.

Trains on diverse data: prose, code, math, technical docs, conversations.
Full sentences/paragraphs in, same out. The bottleneck must learn to compress
any structured information into 32 slots x 32 dims = 1024 dimensions.

Usage:
    python train_v11.py --fresh          # start from scratch
    python train_v11.py --resume         # resume from V11 checkpoint
    python train_v11.py --from-v9        # warm-start encoder from V9, fresh decoder
    python train_v11.py --eval-only      # diagnostics only
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
from concept_model import ConceptConfig, ConceptAutoencoderV10, reconstruction_loss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v11"
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
TOTAL_STEPS = 300_000
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
    "log": f"{LOG_DIR}/concept_v11.log",
    "metrics": f"{LOG_DIR}/concept_v11_metrics.csv",
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
# Data loading — diverse sources
# ---------------------------------------------------------------------------

def _split_into_chunks(text, max_words=40):
    """Split text into sentence/paragraph-sized chunks.

    Tries to split on sentence boundaries first, falls back to word chunks.
    Each chunk is a complete idea — full sentences in, full sentences out.
    """
    # Split on sentence boundaries
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
        """Add text to appropriate length bucket."""
        approx_tokens = len(text.split()) + 2  # rough estimate
        for bname, (lo, hi) in LENGTH_BUCKETS.items():
            if lo <= approx_tokens <= hi:
                self.buckets[bname].append(text)
                self.total_texts += 1
                return
        # Over max bucket → still add to long
        if approx_tokens > 0:
            self.buckets["long"].append(text)
            self.total_texts += 1

    def _load_all(self):
        # 1. Pair data (text_a / text_b from paraphrase datasets)
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
        # Cap per source to avoid loading 30GB+ into RAM
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
        """Update sampling weights based on per-bucket exact-match."""
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
        """Sample a batch with dynamic length weighting."""
        texts = []
        bucket_names = list(LENGTH_BUCKETS.keys())
        weights = [self.bucket_weights[b] for b in bucket_names]

        for _ in range(batch_size):
            bname = random.choices(bucket_names, weights=weights, k=1)[0]
            bucket = self.buckets[bname]
            if bucket:
                texts.append(bucket[random.randint(0, len(bucket) - 1)])
            else:
                # Fallback: pick from any non-empty bucket
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
# Evaluation
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
    logits = m.decode_parallel(concepts, seq_len=input_ids.shape[1])
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

    # Uniform sampling for unbiased eval
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
        "version": "v11",
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

    model_state = model.state_dict()
    loaded = 0
    for key in model_state:
        if key in v9_state and model_state[key].shape == v9_state[key].shape:
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
# Main training loop — pure reconstruction
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False, eval_only=False, from_v9=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V11 — NON-AUTOREGRESSIVE CONCEPT AUTOENCODER (Pure Recon)")
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

    if recon_dataset.total_texts == 0:
        log("ERROR: No training texts found.")
        return

    log(f"\nTraining plan (V11 — Pure Reconstruction):")
    log(f"  Encoder: {config.enc_hidden}h x {config.enc_layers}L x {config.enc_heads}heads")
    log(f"  Decoder: PARALLEL {config.dec_hidden}h x {config.dec_layers}L x {config.dec_heads}heads")
    log(f"  Bottleneck: {config.num_concepts} x {config.concept_dim} = {config.total_concept_dim} dims")
    log(f"  Batch: {BATCH_SIZE}")
    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")
    log(f"  Data: {recon_dataset.total_texts:,} texts (diverse: prose, code, math, docs)")
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

            # Log metrics
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            log_metrics(step + 1, avg_loss, token_acc, exact_match,
                        bacc.get("short", 0), bacc.get("medium", 0),
                        bacc.get("long", 0),
                        bem.get("short", 0), bem.get("medium", 0),
                        bem.get("long", 0),
                        current_lr, elapsed / 3600)

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
    parser = argparse.ArgumentParser(description="Train concept autoencoder V11")
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
