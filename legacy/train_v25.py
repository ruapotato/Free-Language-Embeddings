#!/usr/bin/env python3
"""
FLM V25 — Unified Sentence Compressor + Sentence-Level Language Model

Joint training: V24 encoder/decoder + causal sentence predictor.
The encoder/decoder learn concept geometry FROM the prediction task,
like word2vec learns word geometry from context prediction.

Architecture:
    sentences → [V24 Encoder] → concept vectors → [Causal Predictor] → predicted next concept
                                      ↓                                        ↓
                              [V24 Decoder]                            [V24 Decoder]
                                      ↓                                        ↓
                              reconstruction CE                         prediction CE

Both losses backprop through the encoder, shaping the concept space
so that sentence meaning is predictable from context.

Usage:
    python train_v25.py --fresh                    # train from scratch
    python train_v25.py --v24-init path/to/v24.pt  # init encoder/decoder from V24
    python train_v25.py --resume path/to/v25.pt    # resume V25 training
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
from model import HamnerConfig, HamnerBlock, precompute_rope

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v25"
LOG_DIR = "logs"

# V24 encoder/decoder config (same as V24)
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

# Sentence predictor config
PRED_HIDDEN = 512
PRED_LAYERS = 8
PRED_HEADS = 8
PRED_KV_HEADS = 4
PRED_FFN = 2048
MAX_SENTENCES = 16      # max sentences per context window
WINDOW_SIZE = 8          # sentences per training example

# Training
BATCH_SIZE = 16
PEAK_LR = 3e-4
MIN_LR = 1e-5
WARMUP_STEPS = 2000
TOTAL_STEPS = 600_000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0
RECON_WEIGHT = 0.5       # weight for reconstruction loss (prediction loss = 1.0)

LOG_EVERY = 50
EVAL_EVERY = 1000
CHECKPOINT_EVERY = 2000

LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v25.log",
    "metrics": f"{LOG_DIR}/concept_v25_metrics.csv",
}

# Data sources (same as V24)
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

# Test sentences for evaluation
EVAL_PROMPTS = [
    ["The cat sat on the mat.", "The dog chased the ball."],
    ["The sun rose over the mountains.", "Birds began to sing in the trees."],
    ["She opened the door carefully.", "The room was dark and quiet."],
    ["Water boils at one hundred degrees.", "Ice melts at zero degrees."],
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
# Sentence Predictor — causal transformer on concept vectors
# ---------------------------------------------------------------------------

CONCEPT_DIM = V24_CONFIG["num_concepts"] * V24_CONFIG["concept_dim"]  # 64*16 = 1024

class SentencePredictor(nn.Module):
    """Causal transformer that predicts next sentence concept vectors.

    Input:  (batch, num_sentences, CONCEPT_DIM) — sequence of concept vectors
    Output: (batch, num_sentences, CONCEPT_DIM) — predicted next concept vectors
    """

    def __init__(self, concept_dim=CONCEPT_DIM, hidden=PRED_HIDDEN,
                 num_layers=PRED_LAYERS, num_heads=PRED_HEADS,
                 num_kv_heads=PRED_KV_HEADS, ffn_dim=PRED_FFN,
                 max_sentences=MAX_SENTENCES):
        super().__init__()
        self.concept_dim = concept_dim
        self.hidden = hidden
        self.max_sentences = max_sentences

        # Project concept vectors to/from predictor hidden dim
        self.proj_in = nn.Linear(concept_dim, hidden)
        self.proj_out = nn.Linear(hidden, concept_dim)

        # Transformer layers (using HamnerBlock for RoPE + SwiGLU + GQA)
        block_config = HamnerConfig(
            hidden_size=hidden,
            num_layers=num_layers,
            num_attention_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_experts=1,
            num_active_experts=1,
            expert_intermediate_size=ffn_dim,
            max_seq_len=max_sentences,
            rope_theta=10000.0,
            rms_norm_eps=1e-5,
            use_differential_attention=False,
            gradient_checkpointing=False,
        )
        self.layers = nn.ModuleList([
            HamnerBlock(block_config, layer_idx=i) for i in range(num_layers)
        ])
        from model import RMSNorm
        self.norm = RMSNorm(hidden)

        # Precompute RoPE for sentence positions
        self.rope_cos, self.rope_sin = precompute_rope(
            hidden // num_heads, max_sentences, theta=10000.0
        )

    def forward(self, x, attention_mask=None):
        """
        x: (batch, num_sentences, concept_dim)
        attention_mask: (batch, 1, num_sents, num_sents) causal mask, or None
        Returns: (batch, num_sentences, concept_dim) predicted next vectors
        """
        B, S, _ = x.shape
        device = x.device

        h = self.proj_in(x)  # (B, S, hidden)

        rope_cos = self.rope_cos[:S].to(device)
        rope_sin = self.rope_sin[:S].to(device)

        # Build causal mask if not provided
        if attention_mask is None:
            causal = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool))
            attention_mask = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

        for layer in self.layers:
            h, _ = layer(h, rope_cos, rope_sin, attention_mask)

        h = self.norm(h)
        return self.proj_out(h)  # (B, S, concept_dim)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ---------------------------------------------------------------------------
# Document Dataset — yields windows of consecutive sentences
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text):
    """Split document into sentences."""
    sents = _SENT_RE.split(text.strip())
    result = []
    for s in sents:
        s = s.strip()
        words = len(s.split())
        if 3 <= words <= 50:
            result.append(s)
    return result


class DocumentDataset:
    """Streams windows of consecutive sentences from JSONL document sources."""

    def __init__(self, sources, tokenizer, window_size=WINDOW_SIZE,
                 max_seq_len=V24_CONFIG["max_seq_len"]):
        self.sources = [(p, w) for p, w in sources if os.path.exists(p)]
        if not self.sources:
            raise FileNotFoundError(f"No data sources found")
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_seq_len = max_seq_len

        # Weighted sampling
        total_w = sum(w for _, w in self.sources)
        self._cum_weights = []
        cum = 0
        for _, w in self.sources:
            cum += w / total_w
            self._cum_weights.append(cum)

        # File handles
        self._files = [open(p) for p, _ in self.sources]

        # Buffer of sentence windows: list of [sent1, sent2, ..., sentW]
        self._window_buf = []
        self._MIN_BUF = 500

        log(f"DocumentDataset: {len(self.sources)} sources, window={window_size}")

    def _sample_source(self):
        r = random.random()
        for i, cw in enumerate(self._cum_weights):
            if r <= cw:
                return i
        return len(self.sources) - 1

    def _read_doc(self, src_idx):
        """Read one document from source, return text or None."""
        f = self._files[src_idx]
        line = f.readline()
        if not line:
            # Reopen
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
        """Fill buffer with sentence windows from documents."""
        attempts = 0
        while len(self._window_buf) < self._MIN_BUF and attempts < 50000:
            attempts += 1
            src = self._sample_source()
            text = self._read_doc(src)
            if not text or len(text) < 30:
                continue
            sents = split_sentences(text)
            if len(sents) < self.window_size:
                continue
            # Extract all possible windows from this document
            for start in range(0, len(sents) - self.window_size + 1,
                             max(1, self.window_size // 2)):  # overlapping windows
                window = sents[start:start + self.window_size]
                self._window_buf.append(window)
        random.shuffle(self._window_buf)

    def get_batch(self, batch_size):
        """Get a batch of sentence windows, tokenized.

        Returns dict with:
            input_ids: (batch, window_size, max_seq_len)
            attention_mask: (batch, window_size, max_seq_len)
            raw_sentences: list of list of strings
        """
        while len(self._window_buf) < batch_size:
            self._refill_buffer()

        windows = [self._window_buf.pop() for _ in range(batch_size)]

        # Tokenize all sentences in all windows
        all_input_ids = []
        all_attention_mask = []
        for window in windows:
            enc = self.tokenizer(
                window, max_length=self.max_seq_len,
                padding='max_length', truncation=True, return_tensors="pt"
            )
            all_input_ids.append(enc["input_ids"])
            all_attention_mask.append(enc["attention_mask"])

        return {
            "input_ids": torch.stack(all_input_ids),           # (B, W, S)
            "attention_mask": torch.stack(all_attention_mask),  # (B, W, S)
            "raw_sentences": windows,
        }


class PrefetchBuffer:
    """Background thread that pre-generates batches."""

    def __init__(self, dataset, batch_size=16, buf_size=8):
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

def save_checkpoint(v24_model, predictor, optimizer, scaler, v24_config,
                    step, metrics, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)

    v24_state = v24_model.state_dict()
    v24_state = {k.replace("_orig_mod.", ""): v for k, v in v24_state.items()}
    pred_state = predictor.state_dict()
    pred_state = {k.replace("_orig_mod.", ""): v for k, v in pred_state.items()}

    ckpt = {
        "v24_state_dict": v24_state,
        "pred_state_dict": pred_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "v24_config": v24_config.__dict__,
        "pred_config": {
            "hidden": PRED_HIDDEN,
            "num_layers": PRED_LAYERS,
            "num_heads": PRED_HEADS,
            "num_kv_heads": PRED_KV_HEADS,
            "ffn_dim": PRED_FFN,
            "max_sentences": MAX_SENTENCES,
        },
        "step": step,
        "metrics": metrics,
        "version": "v25",
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


def load_v25_checkpoint(path, device="cuda"):
    """Load a V25 checkpoint (V24 + predictor)."""
    log(f"Loading V25 checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    v24_config = ConceptConfig(**ckpt["v24_config"])
    v24_model = ConceptAutoencoderV24(v24_config).to(device)
    v24_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["v24_state_dict"].items()}
    v24_model.load_state_dict(v24_state, strict=True)

    pc = ckpt["pred_config"]
    predictor = SentencePredictor(
        hidden=pc["hidden"], num_layers=pc["num_layers"],
        num_heads=pc["num_heads"], num_kv_heads=pc["num_kv_heads"],
        ffn_dim=pc["ffn_dim"], max_sentences=pc["max_sentences"],
    ).to(device)
    pred_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["pred_state_dict"].items()}
    predictor.load_state_dict(pred_state, strict=True)

    all_params = list(v24_model.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=PEAK_LR,
                                  betas=BETAS, weight_decay=WEIGHT_DECAY)
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler = torch.amp.GradScaler("cuda")
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    step = ckpt["step"]
    metrics = ckpt.get("metrics", {})
    log(f"Resumed V25: step {step}")
    return v24_model, predictor, optimizer, scaler, v24_config, step, metrics


def load_v24_init(path, device="cuda"):
    """Initialize V24 encoder/decoder from a V24 checkpoint."""
    log(f"Initializing V24 from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    v24_config = ConceptConfig(**ckpt["config"])
    v24_model = ConceptAutoencoderV24(v24_config).to(device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    v24_model.load_state_dict(state, strict=True)
    total, _ = v24_model.count_parameters()
    log(f"  V24: {total:,} params | loaded from step {ckpt['step']}")
    return v24_model, v24_config


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(v24_model, predictor, tokenizer, device="cuda"):
    """Generate next-sentence predictions from eval prompts."""
    v24_model.eval()
    predictor.eval()

    for prompt_sents in EVAL_PROMPTS:
        # Encode prompt sentences
        enc = tokenizer(prompt_sents, max_length=V24_CONFIG["max_seq_len"],
                       padding='max_length', truncation=True, return_tensors="pt").to(device)

        # Encode each sentence → concept vector
        concepts = []
        for i in range(len(prompt_sents)):
            ids = enc["input_ids"][i:i+1]
            mask = enc["attention_mask"][i:i+1]
            c = v24_model.encode(ids, mask)       # (1, K, D)
            concepts.append(c.view(1, -1))         # (1, concept_dim)

        concept_seq = torch.stack(concepts, dim=1)  # (1, num_sents, concept_dim)

        # Predict next sentence
        predicted = predictor(concept_seq)           # (1, num_sents, concept_dim)
        next_concept = predicted[0, -1]              # (concept_dim,)

        # Decode predicted concept vector
        next_concept_3d = next_concept.view(1, V24_CONFIG["num_concepts"],
                                            V24_CONFIG["concept_dim"])
        dummy_mask = torch.ones(1, V24_CONFIG["max_seq_len"],
                              dtype=torch.long, device=device)
        logits = v24_model.decode(next_concept_3d, V24_CONFIG["max_seq_len"], dummy_mask)
        tokens = logits.argmax(dim=-1)[0]
        generated = tokenizer.decode(tokens, skip_special_tokens=True)

        # Also test reconstruction of input sentences
        prompt_str = " | ".join(prompt_sents)
        log(f"  Prompt: {prompt_str}")
        log(f"  Predicted next: {generated}")

    # Test reconstruction quality (still works?)
    recon_sents = [
        "the cat sat on the mat",
        "artificial intelligence will change the world",
        "she runs every morning before breakfast",
    ]
    enc = tokenizer(recon_sents, max_length=V24_CONFIG["max_seq_len"],
                   padding='max_length', truncation=True, return_tensors="pt").to(device)
    exact = 0
    for i in range(len(recon_sents)):
        ids = enc["input_ids"][i:i+1]
        mask = enc["attention_mask"][i:i+1]
        c = v24_model.encode(ids, mask)
        logits = v24_model.decode(c, ids.shape[1], mask)
        pred = logits.argmax(dim=-1)[0]
        pred_mask = mask[0].bool()
        decoded = tokenizer.decode(pred[pred_mask], skip_special_tokens=True)
        match = decoded.strip() == recon_sents[i].strip()
        exact += int(match)
        status = "OK" if match else "DIFF"
        log(f"  Recon [{status}]: {recon_sents[i]}")
        if not match:
            log(f"           -> {decoded}")
    log(f"  Reconstruction: {exact}/{len(recon_sents)}")

    v24_model.train()
    predictor.train()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(v24_init=None, resume_from=None, fresh=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V25 — UNIFIED SENTENCE COMPRESSOR + LM")
    log("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # --- Model setup ---
    if resume_from:
        v24_model, predictor, optimizer, scaler, v24_config, start_step, _ = \
            load_v25_checkpoint(resume_from, device)
    else:
        if v24_init:
            v24_model, v24_config = load_v24_init(v24_init, device)
        else:
            v24_config = ConceptConfig(**V24_CONFIG)
            v24_model = ConceptAutoencoderV24(v24_config).to(device)
            total, _ = v24_model.count_parameters()
            log(f"V24 fresh init: {total:,} params")

        predictor = SentencePredictor().to(device)
        pred_total, _ = predictor.count_parameters()
        log(f"Predictor: {pred_total:,} params")

        all_params = list(v24_model.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=PEAK_LR,
                                      betas=BETAS, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0

    v24_total, _ = v24_model.count_parameters()
    pred_total, _ = predictor.count_parameters()
    log(f"\nArchitecture:")
    log(f"  V24 Encoder/Decoder: {v24_total:,} params")
    log(f"  Sentence Predictor:  {pred_total:,} params")
    log(f"  Total:               {v24_total + pred_total:,} params")
    log(f"  Concept dim: {CONCEPT_DIM} ({V24_CONFIG['num_concepts']}x{V24_CONFIG['concept_dim']})")
    log(f"  Predictor: {PRED_LAYERS}L, {PRED_HIDDEN}d, {PRED_HEADS}h, {PRED_FFN} FFN")
    log(f"  Window: {WINDOW_SIZE} sentences, max {MAX_SENTENCES}")
    log(f"  Batch: {BATCH_SIZE}")
    log(f"  Losses: recon (weight={RECON_WEIGHT}) + prediction")
    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")

    # --- Data ---
    dataset = DocumentDataset(PRETRAIN_SOURCES, tokenizer)
    prefetch = PrefetchBuffer(dataset, batch_size=BATCH_SIZE)
    prefetch.start()
    log("Prefetch started")
    log("-" * 70)

    # --- Training state ---
    recon_tracker = []
    pred_tracker = []
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

        optimizer.zero_grad(set_to_none=True)

        batch = prefetch.get()

        with torch.amp.autocast("cuda", dtype=torch.float16):
            input_ids = batch["input_ids"].to(device)       # (B, W, S)
            attn_mask = batch["attention_mask"].to(device)   # (B, W, S)
            B, W, S = input_ids.shape

            # --- Encode all sentences → concept vectors ---
            # Reshape to (B*W, S) for batch encoding
            flat_ids = input_ids.view(B * W, S)
            flat_mask = attn_mask.view(B * W, S)

            concepts_3d = v24_model.encode(flat_ids, flat_mask)  # (B*W, K, D)
            concepts_flat = concepts_3d.view(B * W, -1)          # (B*W, concept_dim)
            concepts_seq = concepts_flat.view(B, W, -1)          # (B, W, concept_dim)

            # --- Reconstruction loss (all sentences) ---
            recon_logits = v24_model.decode(concepts_3d, S, flat_mask)  # (B*W, S, vocab)
            recon_loss = reconstruction_loss(recon_logits, flat_ids)

            # --- Prediction loss ---
            # Input to predictor: sentences 0..W-2
            # Target: sentences 1..W-1
            pred_input = concepts_seq[:, :-1, :]     # (B, W-1, concept_dim)
            target_ids = input_ids[:, 1:, :]         # (B, W-1, S)
            target_mask = attn_mask[:, 1:, :]        # (B, W-1, S)

            # Predict next sentence concept vectors
            predicted = predictor(pred_input)        # (B, W-1, concept_dim)

            # Decode predicted vectors → token logits → CE loss
            pred_concepts_3d = predicted.view(B * (W - 1),
                                              V24_CONFIG["num_concepts"],
                                              V24_CONFIG["concept_dim"])
            pred_target_ids = target_ids.reshape(B * (W - 1), S)
            pred_target_mask = target_mask.reshape(B * (W - 1), S)

            pred_logits = v24_model.decode(pred_concepts_3d, S, pred_target_mask)
            pred_loss = reconstruction_loss(pred_logits, pred_target_ids)

            # --- Total loss ---
            loss = RECON_WEIGHT * recon_loss + pred_loss

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

        # Clip gradients for both models
        all_params = list(v24_model.parameters()) + list(predictor.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        recon_tracker.append(recon_loss.item())
        pred_tracker.append(pred_loss.item())

        # --- Logging ---
        if (step + 1) % LOG_EVERY == 0:
            avg_recon = np.mean(recon_tracker[-100:]) if recon_tracker else 0
            avg_pred = np.mean(pred_tracker[-100:]) if pred_tracker else 0
            pct = (step + 1) / TOTAL_STEPS * 100
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1 - start_step) / max(elapsed, 1)
            log(f"step {step+1:>7d} [V25] | recon={avg_recon:.4f} pred={avg_pred:.4f} | "
                f"lr {current_lr:.2e} | {pct:.1f}% | {steps_per_sec:.1f} step/s")

            log_metrics(step + 1, {
                "recon_loss": avg_recon,
                "pred_loss": avg_pred,
                "total_loss": RECON_WEIGHT * avg_recon + avg_pred,
                "lr": current_lr,
            })

        # --- Eval ---
        if (step + 1) % EVAL_EVERY == 0:
            log("\n--- EVALUATION ---")
            evaluate(v24_model, predictor, tokenizer, device)
            log("")

        # --- Checkpoint ---
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_recon = np.mean(recon_tracker[-100:]) if recon_tracker else 0
            avg_pred = np.mean(pred_tracker[-100:]) if pred_tracker else 0
            save_checkpoint(v24_model, predictor, optimizer, scaler, v24_config,
                          step + 1, {"recon": avg_recon, "pred": avg_pred},
                          CHECKPOINT_DIR)

    # Final save
    if recon_tracker:
        avg_recon = np.mean(recon_tracker[-100:])
        avg_pred = np.mean(pred_tracker[-100:])
        save_checkpoint(v24_model, predictor, optimizer, scaler, v24_config,
                      step + 1, {"recon": avg_recon, "pred": avg_pred},
                      CHECKPOINT_DIR)

    log("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLM V25 — Unified Sentence LM")
    parser.add_argument("--fresh", action="store_true", help="Train from scratch")
    parser.add_argument("--v24-init", type=str, help="Initialize from V24 checkpoint")
    parser.add_argument("--resume", type=str, help="Resume V25 training")
    args = parser.parse_args()

    if not args.fresh and not args.v24_init and not args.resume:
        # Default: try to resume from latest, else init from V24 latest
        latest = Path(CHECKPOINT_DIR) / "latest.pt"
        if latest.exists():
            args.resume = str(latest)
        else:
            v24_latest = Path("checkpoints/concept_v24/latest.pt")
            if v24_latest.exists():
                args.v24_init = str(v24_latest)

    train(v24_init=args.v24_init, resume_from=args.resume, fresh=args.fresh)
