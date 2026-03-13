"""
FLM Concept-Space LM — Phase 2
================================================================================
Autoregressive LM that operates entirely in concept space. Never sees tokens.

Architecture:
  - Same transformer as V4 (HamnerModel-style: GQA, RoPE, SwiGLU)
  - Input: concept vectors from frozen V24 encoder (1024-dim per sentence)
  - Output: predict next concept vector (1024-dim)
  - Loss: MSE on predicted vs actual concept vectors

Pipeline at training time:
  1. Load a document, split into sentences
  2. Encode each sentence through frozen V24 → sequence of concept vectors
  3. Feed concept vector sequence to LM, predict next concept vector autoregressively
  4. Loss = MSE(predicted_concept[i], actual_concept[i+1])

Pipeline at inference time:
  1. Split prompt into sentences
  2. Encode each through frozen V24 encoder → concept vectors
  3. LM predicts next concept vector
  4. Frozen V24 decoder renders concept vector → text
  5. Repeat

Usage:
    python train_concept_lm.py --fresh            # start from scratch
    python train_concept_lm.py                    # auto-resume
    python train_concept_lm.py --eval-only        # diagnostics only
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
from dataclasses import dataclass
from model import HamnerConfig, HamnerBlock, RMSNorm, precompute_rope

# We'll load the frozen encoder from concept_model
from concept_model import ConceptConfig, ConceptAutoencoderV24

# ---------------------------------------------------------------------------
# Concept-Space LM Model
# ---------------------------------------------------------------------------

@dataclass
class ConceptLMConfig:
    """Config for the concept-space language model."""
    # Concept space dimensions (from V24 encoder)
    concept_dim: int = 1024         # 64 concepts x 16 dims = 1024 flat
    num_concepts: int = 64
    concept_slot_dim: int = 16

    # Transformer config (same scale as V4)
    hidden_size: int = 768
    num_layers: int = 20
    num_attention_heads: int = 12
    num_kv_heads: int = 4
    expert_intermediate_size: int = 2048
    max_seq_len: int = 128          # max sentences in a document
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    gradient_checkpointing: bool = True


class ConceptLM(nn.Module):
    """
    Autoregressive LM that operates on concept vectors instead of tokens.

    Input: (batch, seq_len, concept_dim) — sequence of concept vectors
    Output: (batch, seq_len, concept_dim) — predicted next concept vectors

    Each position is one sentence's concept vector (1024-dim).
    The model predicts the concept vector for the NEXT sentence.
    """

    def __init__(self, config: ConceptLMConfig):
        super().__init__()
        self.config = config

        # Project concept vectors into transformer hidden space
        self.concept_in = nn.Linear(config.concept_dim, config.hidden_size, bias=False)

        # Transformer backbone (reuses HamnerBlock from model.py)
        hamner_cfg = HamnerConfig(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            expert_intermediate_size=config.expert_intermediate_size,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            dropout=config.dropout,
            gradient_checkpointing=config.gradient_checkpointing,
            num_experts=1,
            num_active_experts=1,
            use_differential_attention=False,
        )

        head_dim = config.hidden_size // config.num_attention_heads
        rope_cos, rope_sin = precompute_rope(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self.layers = nn.ModuleList([
            HamnerBlock(hamner_cfg, i) for i in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Project back to concept space
        self.concept_out = nn.Linear(config.hidden_size, config.concept_dim, bias=False)

        self.apply(self._init_weights)
        self._gradient_checkpointing = config.gradient_checkpointing

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, concept_vectors, targets=None):
        """
        concept_vectors: (batch, seq_len, concept_dim) — input concept sequence
        targets: (batch, seq_len, concept_dim) — target concept vectors (shifted by 1)

        Returns dict with 'loss' and 'predicted_concepts'.
        """
        bsz, seq_len, _ = concept_vectors.shape
        device = concept_vectors.device

        # Project to hidden space
        x = self.concept_in(concept_vectors)

        # Causal mask (autoregressive: each position can only see previous)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        # Run through transformer
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(
                    layer, x, self.rope_cos, self.rope_sin, causal_mask,
                    use_reentrant=False,
                )
            else:
                x, _ = layer(x, self.rope_cos, self.rope_sin, causal_mask)

        x = self.final_norm(x)

        # Project back to concept space
        predicted = self.concept_out(x)

        loss = None
        if targets is not None:
            # MSE loss: predicted[i] should match targets[i] (= actual concept[i+1])
            loss = F.mse_loss(predicted, targets)

        return {"loss": loss, "predicted_concepts": predicted}

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    @torch.no_grad()
    def generate(self, concept_vectors, num_steps=10):
        """
        Autoregressively generate concept vectors.

        concept_vectors: (1, seq_len, concept_dim) — prompt concepts
        num_steps: how many new concept vectors to generate

        Returns: (1, seq_len + num_steps, concept_dim)
        """
        self.eval()
        for _ in range(num_steps):
            # Only use last max_seq_len positions
            ctx = concept_vectors[:, -self.config.max_seq_len:]
            out = self(ctx)
            next_concept = out["predicted_concepts"][:, -1:, :]  # last position's prediction
            concept_vectors = torch.cat([concept_vectors, next_concept], dim=1)
        return concept_vectors


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# V24 encoder checkpoint (frozen)
V24_CHECKPOINT = "checkpoints/concept_v24/latest.pt"

V24_CONFIG = None  # loaded from checkpoint

LM_CONFIG = ConceptLMConfig(
    concept_dim=1024,           # 64 * 16 from V24
    num_concepts=64,
    concept_slot_dim=16,
    hidden_size=768,
    num_layers=20,
    num_attention_heads=12,
    num_kv_heads=4,
    expert_intermediate_size=2048,
    max_seq_len=128,            # up to 128 sentences per document
    gradient_checkpointing=True,
)

# Training hyperparameters
BATCH_SIZE = 32
PEAK_LR = 2e-4
MIN_LR = 1e-5
WARMUP_STEPS = 2000
TOTAL_STEPS = 400_000
GRAD_CLIP = 1.0
MIN_SENTENCES = 3              # minimum sentences per document

# Logging
LOG_EVERY = 50
EVAL_EVERY = 1000
CHECKPOINT_EVERY = 1000

CHECKPOINT_DIR = "checkpoints/concept_lm"
LOG_PATHS = {
    "log": "logs/concept_lm.log",
    "metrics": "logs/concept_lm_metrics.csv",
    "stderr": "logs/concept_lm_stderr.log",
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

SAMPLE_PROMPTS = [
    "The meaning of life is important to understand.",
    "Once upon a time there was a small village.",
    "Hello! How are you doing today?",
    "The most important thing about technology is innovation.",
    "The derivative of x squared is two x.",
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


METRICS_HEADER = ("timestamp,step,mse_loss,cosine_sim,lr,"
                  "elapsed_hours,sentences_per_sec\n")


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
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts},{step},{m.get('mse_loss',0):.6f},"
                f"{m.get('cosine_sim',0):.4f},{m.get('lr',0):.6e},"
                f"{m.get('elapsed_hours',0):.4f},"
                f"{m.get('sentences_per_sec',0):.1f}\n")


# ---------------------------------------------------------------------------
# Document → sentence sequence dataset
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text):
    """Split text into individual sentences."""
    sentences = _SENT_RE.split(text.strip())
    result = []
    for s in sentences:
        s = s.strip()
        words = len(s.split())
        if words >= 3 and words <= 50:
            result.append(s)
    return result


class DocumentSentenceDataset:
    """
    Streams documents, splits into sentences, and returns sequences of
    sentences (as raw text) for encoding through the frozen V24.

    Each batch item is a list of sentences from one document.
    """

    def __init__(self, min_sentences=3):
        self.min_sentences = min_sentences
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

    def get_document_sentences(self):
        """Get a list of sentences from one document."""
        for _ in range(100):
            source_idx = self._sample_source()
            line = self._read_line(source_idx)
            if line is None:
                continue
            try:
                doc = json.loads(line)
                text = doc.get("text", "").strip()
                if len(text) < 50:
                    continue
                sentences = split_sentences(text)
                if len(sentences) >= self.min_sentences:
                    # Cap at max_seq_len sentences
                    return sentences[:LM_CONFIG.max_seq_len]
            except (json.JSONDecodeError, KeyError):
                continue
        return None

    def get_batch(self, batch_size):
        """Get a batch of document sentence lists."""
        docs = []
        while len(docs) < batch_size:
            sents = self.get_document_sentences()
            if sents is not None:
                docs.append(sents)
        return docs

    def close(self):
        for h in self._handles:
            if h is not None:
                h.close()


class PrefetchBuffer:
    """Pre-generates batches in background thread."""

    def __init__(self, dataset, batch_size=32, buf_size=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.q = queue.Queue(maxsize=buf_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            try:
                batch = self.dataset.get_batch(self.batch_size)
                self.q.put(batch, timeout=1)
            except queue.Full:
                continue
            except Exception as e:
                print(f"Prefetch error: {e}", flush=True)
                continue

    def get(self):
        return self.q.get()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Encode sentences through frozen V24
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_sentences(sentences, encoder, tokenizer, device):
    """
    Encode a list of sentences through frozen V24 encoder.

    Returns: (num_sentences, concept_dim) tensor of concept vectors
    """
    enc = tokenizer(sentences, max_length=64, padding='longest',
                    truncation=True, return_tensors="pt").to(device)
    concepts = encoder.encode(enc["input_ids"], enc["attention_mask"])
    # Flatten: (batch, num_concepts, concept_slot_dim) → (batch, concept_dim)
    flat = concepts.view(concepts.shape[0], -1)
    return flat


@torch.no_grad()
def encode_batch(doc_batch, encoder, tokenizer, device, max_seq_len):
    """
    Encode a batch of documents (each = list of sentences) into padded
    concept vector sequences.

    Returns:
        concept_seqs: (batch, max_len, concept_dim) — padded concept sequences
        lengths: list of actual sequence lengths
    """
    all_concepts = []
    lengths = []

    for sentences in doc_batch:
        # Encode all sentences in this document at once
        flat = encode_sentences(sentences, encoder, tokenizer, device)
        all_concepts.append(flat)
        lengths.append(flat.shape[0])

    # Pad to max length in batch
    max_len = min(max(lengths), max_seq_len)
    concept_dim = all_concepts[0].shape[1]
    padded = torch.zeros(len(doc_batch), max_len, concept_dim, device=device)

    for i, (concepts, length) in enumerate(zip(all_concepts, lengths)):
        actual = min(length, max_len)
        padded[i, :actual] = concepts[:actual]

    return padded, lengths


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, encoder, decoder, tokenizer, device):
    """Evaluate: generate sentences from prompts and measure prediction quality."""
    model.eval()
    m = encoder

    results = []
    for prompt in SAMPLE_PROMPTS:
        # Encode prompt sentence
        enc = tokenizer([prompt], max_length=64, padding='longest',
                        truncation=True, return_tensors="pt").to(device)
        concepts = m.encode(enc["input_ids"], enc["attention_mask"])
        flat = concepts.view(1, 1, -1)  # (1, 1, concept_dim)

        # Generate next concept vectors
        generated = model.generate(flat, num_steps=3)

        # Decode all generated concepts back to text
        decoded_sentences = []
        for i in range(generated.shape[1]):
            concept_vec = generated[0, i:i+1]  # (1, concept_dim)
            # Reshape to (1, num_concepts, concept_slot_dim)
            concept_3d = concept_vec.view(1, LM_CONFIG.num_concepts,
                                          LM_CONFIG.concept_slot_dim)
            # Decode — need to figure out seq_len; use 64 as max
            logits = decoder.decode(concept_3d, seq_len=64)
            tokens = logits.argmax(dim=-1)
            text = tokenizer.decode(tokens[0], skip_special_tokens=True)
            decoded_sentences.append(text)

        results.append({
            "prompt": prompt,
            "generated": decoded_sentences,
        })

    model.train()
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

shutdown_requested = False


def _handle_signal(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    log("Shutdown signal received, saving checkpoint...")


def save_checkpoint(model, optimizer, scaler, config, step, loss,
                    checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "loss": loss,
        "version": "concept_lm_v1",
    }
    path = os.path.join(checkpoint_dir, f"step_{step:06d}.pt")
    torch.save(state, path)
    latest = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(state, latest)
    log(f"  Checkpoint saved: {path}")


def train(resume_from=None, fresh=False, eval_only=False):
    global shutdown_requested
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load frozen V24 encoder/decoder ---
    log("Loading frozen V24 encoder/decoder...")
    ckpt_path = V24_CHECKPOINT
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"V24 checkpoint not found: {ckpt_path}")

    v24_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    v24_config = ConceptConfig(**v24_ckpt["config"])
    v24_model = ConceptAutoencoderV24(v24_config).to(device)
    v24_model.load_state_dict(v24_ckpt["model_state_dict"])
    log(f"  V24 loaded from {ckpt_path} (step {v24_ckpt.get('step', '?')})")
    del v24_ckpt

    if resume_from is None and not fresh:
        # Try auto-resume concept LM
        latest = os.path.join(CHECKPOINT_DIR, "latest.pt")
        if os.path.exists(latest):
            resume_from = latest

    v24_model.eval()
    for p in v24_model.parameters():
        p.requires_grad = False
    log("  V24 frozen (encoder + decoder)")

    # --- Load tokenizer ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    log(f"  Tokenizer: bert-base-uncased (vocab={tokenizer.vocab_size})")

    # --- Build concept LM ---
    config = LM_CONFIG
    model = ConceptLM(config).to(device)
    total_params, trainable_params = model.count_parameters()
    log(f"  Concept LM: {total_params:,} params ({total_params/1e6:.1f}M)")

    start_step = 0
    lm_ckpt = None
    if resume_from and os.path.exists(resume_from):
        lm_ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(lm_ckpt["model"])
        start_step = lm_ckpt["step"]
        log(f"  Resumed from {resume_from} (step {start_step})")

    model = torch.compile(model)
    if fresh:
        log("  Starting fresh training...")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scaler = torch.amp.GradScaler("cuda")

    if lm_ckpt is not None:
        if "optimizer" in lm_ckpt:
            optimizer.load_state_dict(lm_ckpt["optimizer"])
        if "scaler" in lm_ckpt:
            scaler.load_state_dict(lm_ckpt["scaler"])
        del lm_ckpt  # free memory

    # --- Data ---
    log("Loading data...")
    dataset = DocumentSentenceDataset(min_sentences=MIN_SENTENCES)
    prefetch = PrefetchBuffer(dataset, batch_size=BATCH_SIZE, buf_size=4)

    # --- Training info ---
    log("")
    log(f"Training plan (Concept-Space LM):")
    log(f"  Model: {total_params:,} params ({total_params/1e6:.1f}M)")
    log(f"  Concept dim: {config.concept_dim} ({config.num_concepts}x{config.concept_slot_dim})")
    log(f"  Transformer: {config.num_layers}L, {config.hidden_size}d, {config.num_attention_heads} heads")
    log(f"  Batch: {BATCH_SIZE}")
    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")
    log(f"  Frozen encoder: V24 ({V24_CHECKPOINT})")
    log("-" * 70)
    log("Prefetch buffer started")

    loss_tracker = []
    cosine_tracker = []
    start_time = time.time()

    for step in range(start_step, TOTAL_STEPS):
        # LR schedule: warmup then cosine decay
        if step < WARMUP_STEPS:
            current_lr = PEAK_LR * (step + 1) / WARMUP_STEPS
        else:
            progress = (step - WARMUP_STEPS) / max(TOTAL_STEPS - WARMUP_STEPS, 1)
            current_lr = MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * progress))

        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)

        # Get batch of documents (lists of sentences)
        doc_batch = prefetch.get()

        # Encode through frozen V24
        with torch.amp.autocast("cuda", dtype=torch.float16):
            concept_seqs, lengths = encode_batch(
                doc_batch, v24_model, tokenizer, device, config.max_seq_len
            )

            # Build input/target pairs (shifted by 1)
            # Input: concepts[:-1], Target: concepts[1:]
            inputs = concept_seqs[:, :-1, :]
            targets = concept_seqs[:, 1:, :]

            if inputs.shape[1] < 1:
                continue

            # Forward through concept LM
            out = model(inputs, targets=targets)
            loss = out["loss"]

        if loss is None or torch.isnan(loss):
            log(f"NaN/None loss at step {step}, skipping")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        loss_tracker.append(loss.item())

        # Cosine similarity between predicted and target (quality metric)
        with torch.no_grad():
            pred = out["predicted_concepts"].float()
            tgt = targets.float()
            cos_sim = F.cosine_similarity(pred.reshape(-1, config.concept_dim),
                                          tgt.reshape(-1, config.concept_dim),
                                          dim=-1).mean().item()
            cosine_tracker.append(cos_sim)

        # --- Logging ---
        if (step + 1) % LOG_EVERY == 0:
            avg_loss = np.mean(loss_tracker[-100:]) if loss_tracker else 0
            avg_cos = np.mean(cosine_tracker[-100:]) if cosine_tracker else 0
            pct = (step + 1) / TOTAL_STEPS * 100
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1 - start_step) / max(elapsed, 1)
            sents_per_sec = steps_per_sec * BATCH_SIZE * np.mean([
                min(l, config.max_seq_len) for l in lengths
            ])
            log(f"step {step+1:>7d} [CLM] | mse={avg_loss:.4f} | "
                f"cos={avg_cos:.3f} | lr {current_lr:.2e} | "
                f"{pct:.1f}% | {sents_per_sec:.0f} sent/s")

        # --- Eval ---
        if (step + 1) % EVAL_EVERY == 0:
            avg_loss = np.mean(loss_tracker[-100:]) if loss_tracker else 0
            avg_cos = np.mean(cosine_tracker[-100:]) if cosine_tracker else 0

            # Generate samples
            results = evaluate(model, v24_model, v24_model, tokenizer, device)
            for r in results:
                log(f"  PROMPT: {r['prompt']}")
                for i, sent in enumerate(r['generated']):
                    label = "INPUT " if i == 0 else f"GEN {i}"
                    log(f"    [{label}] {sent}")

            elapsed = time.time() - start_time
            metrics_dict = {
                "mse_loss": avg_loss,
                "cosine_sim": avg_cos,
                "lr": current_lr,
                "elapsed_hours": elapsed / 3600,
                "sentences_per_sec": 0,
            }
            log_metrics(step + 1, metrics_dict)

        # --- Checkpoint ---
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = np.mean(loss_tracker[-100:]) if loss_tracker else 0
            save_checkpoint(model, optimizer, scaler, config, step + 1,
                            avg_loss, CHECKPOINT_DIR)

        if shutdown_requested:
            save_checkpoint(model, optimizer, scaler, config, step + 1,
                            np.mean(loss_tracker[-100:]) if loss_tracker else 0,
                            CHECKPOINT_DIR)
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

    log("=" * 70)
    log("FLM CONCEPT-SPACE LM — Phase 2")
    log("  Autoregressive LM operating in concept space")
    log("  Input: concept vectors from frozen V24 encoder")
    log("  Output: predicted next concept vectors → decoded by frozen V24")
    log("=" * 70)

    train(resume_from=args.resume, fresh=args.fresh, eval_only=args.eval_only)
