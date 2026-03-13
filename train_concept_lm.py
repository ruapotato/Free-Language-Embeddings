"""
FLM Concept-Space LM — Phase 2 (Block-Causal)
================================================================================
Autoregressive LM that operates on concept SLOTS, not flattened vectors.

Each sentence = 64 concept slots x 16 dims (from frozen V24 encoder).
The LM sees slots as individual positions with block-causal attention:
  - Within a sentence: slots attend to each other bidirectionally
  - Across sentences: causal (can only see previous sentences)

This means concept slots interact and refine each other (like words in a
sentence), while sentence-level prediction remains autoregressive.

Key efficiency: at inference, the LM outputs 64 slots for the next sentence
and feeds them DIRECTLY back in as input. No decode→re-encode roundtrip.
V24 encoder/decoder only touch the edges (encoding prompt, decoding output).

Pipeline at training time:
  1. Load a document, split into sentences
  2. Encode each sentence through frozen V24 → (num_sentences, 64, 16)
  3. Input = sentence blocks 0..N-2, Target = sentence blocks 1..N-1
  4. Block-causal transformer predicts next sentence's slots
  5. Loss = MSE on valid (non-padding) positions only

Pipeline at inference time:
  1. Split prompt into sentences
  2. Encode each through frozen V24 → concept slot blocks
  3. LM predicts next 64 slots (one sentence)
  4. Feed predicted slots directly back into LM (no V24 roundtrip)
  5. When done generating, decode final output through frozen V24 decoder

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

from concept_model import ConceptConfig, ConceptAutoencoderV24

# ---------------------------------------------------------------------------
# Concept-Space LM Model (Block-Causal)
# ---------------------------------------------------------------------------

@dataclass
class ConceptLMConfig:
    """Config for the block-causal concept-space language model."""
    # Concept space (from V24 encoder)
    num_concepts: int = 64          # K — slots per sentence
    concept_slot_dim: int = 16      # dim per slot

    # Transformer
    hidden_size: int = 512
    num_layers: int = 16
    num_attention_heads: int = 8
    num_kv_heads: int = 4
    expert_intermediate_size: int = 2048
    max_sentences: int = 32         # max sentences in a sequence
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    gradient_checkpointing: bool = True


class ConceptLM(nn.Module):
    """
    Block-causal transformer operating on concept slots.

    Input: (batch, num_sentences * K, slot_dim) — interleaved concept slots
    Output: (batch, num_sentences * K, slot_dim) — predicted next sentence slots

    Each sentence is a block of K=64 concept slots (16-dim each).
    Slots within a sentence attend bidirectionally.
    Across sentences, attention is causal.

    Position encoding:
      - Slot-level RoPE (0..63, repeating per sentence) for intra-sentence structure
      - Learned sentence embeddings for inter-sentence ordering
    """

    def __init__(self, config: ConceptLMConfig):
        super().__init__()
        self.config = config
        K = config.num_concepts

        # Project concept slots to hidden space
        self.concept_in = nn.Linear(config.concept_slot_dim, config.hidden_size, bias=False)

        # Learned sentence position embeddings
        self.sentence_embed = nn.Embedding(config.max_sentences, config.hidden_size)

        # Slot-level RoPE (repeats every K positions)
        head_dim = config.hidden_size // config.num_attention_heads
        slot_cos, slot_sin = precompute_rope(head_dim, K, config.rope_theta)
        # Tile for max total sequence length
        max_total = config.max_sentences * K
        self.register_buffer("rope_cos", slot_cos.repeat(config.max_sentences, 1)[:max_total],
                             persistent=False)
        self.register_buffer("rope_sin", slot_sin.repeat(config.max_sentences, 1)[:max_total],
                             persistent=False)

        # Transformer backbone (reuses HamnerBlock)
        hamner_cfg = HamnerConfig(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            expert_intermediate_size=config.expert_intermediate_size,
            max_seq_len=max_total,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            dropout=config.dropout,
            gradient_checkpointing=config.gradient_checkpointing,
            num_experts=1,
            num_active_experts=1,
            use_differential_attention=False,
        )

        self.layers = nn.ModuleList([
            HamnerBlock(hamner_cfg, i) for i in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Project back to concept slot space
        self.concept_out = nn.Linear(config.hidden_size, config.concept_slot_dim, bias=False)

        self.apply(self._init_weights)
        self._gradient_checkpointing = config.gradient_checkpointing

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _build_mask(self, sentence_counts, max_sents, device, dtype):
        """Build block-causal attention mask with padding.

        Within a sentence block: bidirectional (all slots see each other)
        Across sentence blocks: causal (only see previous sentences)
        Padding sentences: masked out entirely
        """
        K = self.config.num_concepts
        bsz = len(sentence_counts)
        total = max_sents * K

        # Sentence index for each position (0, 0, ..., 1, 1, ..., 2, 2, ...)
        sent_idx = torch.arange(max_sents, device=device).repeat_interleave(K)  # (total,)

        # Block-causal: position i can attend to j if sent_idx[j] <= sent_idx[i]
        causal = sent_idx.unsqueeze(0) <= sent_idx.unsqueeze(1)  # (total, total)
        causal = causal.unsqueeze(0).expand(bsz, total, total).clone()  # (bsz, total, total)

        # Padding mask: prevent attending TO padding positions
        # (Don't mask padding FROM attending — softmax needs valid targets
        #  to avoid NaN. Padding outputs are ignored via valid_mask on loss.)
        for b in range(bsz):
            valid = sent_idx < sentence_counts[b]  # (total,) bool
            causal[b, :, ~valid] = False  # can't attend TO padding

        # Convert to additive mask
        mask = torch.where(causal, 0.0, float("-inf"))
        return mask.unsqueeze(1).to(dtype)  # (bsz, 1, total, total)

    def forward(self, concept_slots, sentence_counts, targets=None, valid_mask=None):
        """
        concept_slots: (batch, max_sents * K, slot_dim)
        sentence_counts: list[int] — actual sentence count per batch item
        targets: (batch, max_sents * K, slot_dim) — target slots (shifted by 1 sentence)
        valid_mask: (batch, max_sents * K) bool — True for non-padding positions

        Returns dict with 'loss' and 'predicted_concepts'.
        """
        bsz, total_len, _ = concept_slots.shape
        K = self.config.num_concepts
        max_sents = total_len // K
        device = concept_slots.device

        # Project slots to hidden space
        x = self.concept_in(concept_slots)  # (bsz, total, hidden)

        # Add sentence position embeddings (same embedding for all slots in a sentence)
        sent_ids = torch.arange(max_sents, device=device).repeat_interleave(K)[:total_len]
        x = x + self.sentence_embed(sent_ids)  # broadcast over batch

        # Block-causal + padding mask
        mask = self._build_mask(sentence_counts, max_sents, device, x.dtype)

        # Transformer layers
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(
                    layer, x, self.rope_cos, self.rope_sin, mask,
                    use_reentrant=False,
                )
            else:
                x, _ = layer(x, self.rope_cos, self.rope_sin, mask)

        x = self.final_norm(x)
        predicted = self.concept_out(x)  # (bsz, total, slot_dim)

        loss = None
        if targets is not None:
            diff = (predicted - targets) ** 2  # (bsz, total, slot_dim)
            if valid_mask is not None:
                diff = diff * valid_mask.unsqueeze(-1)
                loss = diff.sum() / (valid_mask.sum() * self.config.concept_slot_dim + 1e-8)
            else:
                loss = diff.mean()

        return {"loss": loss, "predicted_concepts": predicted}

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    @torch.no_grad()
    def generate(self, concept_slots, num_sentences_in, num_steps=3):
        """
        Autoregressively generate sentence blocks.

        concept_slots: (1, num_sentences_in * K, slot_dim) — prompt concept slots
        num_sentences_in: int — number of input sentences
        num_steps: how many new sentences to generate

        Returns: (1, (num_sentences_in + num_steps) * K, slot_dim)
        """
        self.eval()
        K = self.config.num_concepts
        current_sents = num_sentences_in

        for _ in range(num_steps):
            # Cap context to max_sentences
            if current_sents > self.config.max_sentences:
                # Keep last max_sentences worth of slots
                start = (current_sents - self.config.max_sentences) * K
                ctx = concept_slots[:, start:, :]
                ctx_sents = self.config.max_sentences
            else:
                ctx = concept_slots
                ctx_sents = current_sents

            out = self(ctx, [ctx_sents])
            # Last sentence block's predictions = next sentence
            next_block = out["predicted_concepts"][:, -K:, :]  # (1, K, slot_dim)
            concept_slots = torch.cat([concept_slots, next_block], dim=1)
            current_sents += 1

        return concept_slots


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

V24_CHECKPOINT = "checkpoints/concept_v24/latest.pt"

LM_CONFIG = ConceptLMConfig(
    num_concepts=64,
    concept_slot_dim=16,
    hidden_size=512,
    num_layers=16,
    num_attention_heads=8,
    num_kv_heads=4,
    expert_intermediate_size=2048,
    max_sentences=32,
    gradient_checkpointing=True,
)

# Training hyperparameters
BATCH_SIZE = 16
PEAK_LR = 2e-4
MIN_LR = 1e-5
WARMUP_STEPS = 2000
TOTAL_STEPS = 400_000
GRAD_CLIP = 1.0
MIN_SENTENCES = 3

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
    """Streams documents, splits into sentences, returns lists of sentences."""

    def __init__(self, min_sentences=3, max_sentences=32):
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
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
                    return sentences[:self.max_sentences]
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

    def __init__(self, dataset, batch_size=16, buf_size=4):
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
    Encode sentences through frozen V24 encoder.

    Returns: (num_sentences, K, slot_dim) — concept slot tensor (NOT flattened)
    """
    enc = tokenizer(sentences, max_length=64, padding='longest',
                    truncation=True, return_tensors="pt").to(device)
    concepts = encoder.encode(enc["input_ids"], enc["attention_mask"])
    # concepts: (num_sentences, K, slot_dim) — already in the right shape
    return concepts


@torch.no_grad()
def encode_batch(doc_batch, encoder, tokenizer, device, max_sentences, K):
    """
    Encode a batch of documents into padded concept slot sequences.

    Each document = list of sentences → (num_sents, K, slot_dim)
    Padded to (batch, max_sents_in_batch, K, slot_dim)

    For training with shifted targets:
      Input  = sentences 0..N-2  (predict from these)
      Target = sentences 1..N-1  (predict these)

    Returns:
        inputs: (batch, max_input_sents * K, slot_dim) — flattened input blocks
        targets: (batch, max_input_sents * K, slot_dim) — flattened target blocks
        input_counts: list[int] — actual input sentence count per item
        valid_mask: (batch, max_input_sents * K) bool — True for non-padding
    """
    all_concepts = []
    raw_counts = []

    for sentences in doc_batch:
        concepts = encode_sentences(sentences, encoder, tokenizer, device)
        all_concepts.append(concepts)  # (num_sents, K, slot_dim)
        raw_counts.append(concepts.shape[0])

    # After shifting: input has N-1 sentences, target has N-1 sentences
    input_counts = [max(c - 1, 1) for c in raw_counts]
    max_input_sents = min(max(input_counts), max_sentences - 1)
    slot_dim = all_concepts[0].shape[2]
    bsz = len(doc_batch)

    inputs = torch.zeros(bsz, max_input_sents * K, slot_dim, device=device)
    targets = torch.zeros(bsz, max_input_sents * K, slot_dim, device=device)
    valid_mask = torch.zeros(bsz, max_input_sents * K, device=device, dtype=torch.bool)

    for b, (concepts, n_input) in enumerate(zip(all_concepts, input_counts)):
        actual = min(n_input, max_input_sents)
        # Input: sentences 0..actual-1
        inp_slots = concepts[:actual].reshape(actual * K, slot_dim)
        inputs[b, :actual * K] = inp_slots
        # Target: sentences 1..actual (shifted by 1)
        tgt_slots = concepts[1:actual + 1].reshape(actual * K, slot_dim)
        targets[b, :actual * K] = tgt_slots
        # Valid positions
        valid_mask[b, :actual * K] = True
        input_counts[b] = actual

    return inputs, targets, input_counts, valid_mask


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, v24_model, tokenizer, device, config):
    """Generate sentences from prompts and decode back to text."""
    model.eval()
    K = config.num_concepts

    results = []
    for prompt in SAMPLE_PROMPTS:
        # Encode prompt sentence through V24
        enc = tokenizer([prompt], max_length=64, padding='longest',
                        truncation=True, return_tensors="pt").to(device)
        concepts = v24_model.encode(enc["input_ids"], enc["attention_mask"])
        # concepts: (1, K, slot_dim) — one sentence block
        prompt_slots = concepts.reshape(1, K, -1)  # (1, K, slot_dim)

        # Generate 3 more sentences
        all_slots = prompt_slots.reshape(1, K, -1)  # (1, 1*K, slot_dim)
        generated = model.generate(all_slots, num_sentences_in=1, num_steps=3)
        # generated: (1, (1+3)*K, slot_dim)

        # Decode each sentence block back to text
        num_total_sents = generated.shape[1] // K
        decoded = []
        for i in range(num_total_sents):
            block = generated[0, i * K:(i + 1) * K].unsqueeze(0)  # (1, K, slot_dim)
            logits = v24_model.decode(block, seq_len=64)
            tokens = logits.argmax(dim=-1)
            text = tokenizer.decode(tokens[0], skip_special_tokens=True)
            decoded.append(text)

        results.append({
            "prompt": prompt,
            "generated": decoded,
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


def save_checkpoint(model, optimizer, scaler, config, step, loss, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "loss": loss,
        "version": "concept_lm_block_causal_v1",
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
    v24_step = v24_ckpt.get("step", "?")
    log(f"  V24 loaded from {ckpt_path} (step {v24_step})")
    del v24_ckpt

    v24_model.eval()
    for p in v24_model.parameters():
        p.requires_grad = False
    log("  V24 frozen (encoder + decoder)")

    if resume_from is None and not fresh:
        latest = os.path.join(CHECKPOINT_DIR, "latest.pt")
        if os.path.exists(latest):
            resume_from = latest

    # --- Load tokenizer ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    log(f"  Tokenizer: bert-base-uncased (vocab={tokenizer.vocab_size})")

    # --- Build concept LM ---
    config = LM_CONFIG
    K = config.num_concepts
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
        del lm_ckpt

    # --- Data ---
    log("Loading data...")
    dataset = DocumentSentenceDataset(
        min_sentences=MIN_SENTENCES,
        max_sentences=config.max_sentences,
    )
    prefetch = PrefetchBuffer(dataset, batch_size=BATCH_SIZE, buf_size=4)

    # --- Training info ---
    log("")
    log("Training plan (Block-Causal Concept LM):")
    log(f"  Model: {total_params:,} params ({total_params/1e6:.1f}M)")
    log(f"  Slots per sentence: {K} x {config.concept_slot_dim}d")
    log(f"  Transformer: {config.num_layers}L, {config.hidden_size}d, "
        f"{config.num_attention_heads}h")
    log(f"  Max sentences: {config.max_sentences}")
    log(f"  Attention: block-causal (bidirectional within sentence, "
        f"causal across)")
    log(f"  Batch: {BATCH_SIZE}")
    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | "
        f"Steps: {start_step} -> {TOTAL_STEPS}")
    log(f"  Frozen V24: step {v24_step}")
    log("-" * 70)

    loss_tracker = []
    cosine_tracker = []
    start_time = time.time()

    for step in range(start_step, TOTAL_STEPS):
        # LR schedule
        if step < WARMUP_STEPS:
            current_lr = PEAK_LR * (step + 1) / WARMUP_STEPS
        else:
            progress = (step - WARMUP_STEPS) / max(TOTAL_STEPS - WARMUP_STEPS, 1)
            current_lr = MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (
                1 + math.cos(math.pi * progress))

        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)

        doc_batch = prefetch.get()

        with torch.amp.autocast("cuda", dtype=torch.float16):
            inputs, targets, input_counts, valid_mask = encode_batch(
                doc_batch, v24_model, tokenizer, device,
                config.max_sentences, K,
            )

            if inputs.shape[1] < K:
                continue

            out = model(inputs, input_counts, targets=targets,
                        valid_mask=valid_mask)
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

        # Cosine similarity (per-sentence, flattened slots)
        with torch.no_grad():
            pred = out["predicted_concepts"].float()
            tgt = targets.float()
            flat_dim = K * config.concept_slot_dim
            # Reshape to per-sentence vectors for cosine sim
            max_sents = pred.shape[1] // K
            pred_flat = pred.reshape(-1, max_sents, flat_dim).reshape(-1, flat_dim)
            tgt_flat = tgt.reshape(-1, max_sents, flat_dim).reshape(-1, flat_dim)
            # Only compute on valid sentences
            valid_sents = valid_mask.reshape(-1, max_sents, K).any(dim=-1).reshape(-1)
            if valid_sents.any():
                cos_sim = F.cosine_similarity(
                    pred_flat[valid_sents], tgt_flat[valid_sents], dim=-1
                ).mean().item()
            else:
                cos_sim = 0.0
            cosine_tracker.append(cos_sim)

        # --- Logging ---
        if (step + 1) % LOG_EVERY == 0:
            avg_loss = np.mean(loss_tracker[-100:]) if loss_tracker else 0
            avg_cos = np.mean(cosine_tracker[-100:]) if cosine_tracker else 0
            pct = (step + 1) / TOTAL_STEPS * 100
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1 - start_step) / max(elapsed, 1)
            avg_sents = np.mean(input_counts) if input_counts else 1
            sents_per_sec = steps_per_sec * BATCH_SIZE * avg_sents
            log(f"step {step+1:>7d} [CLM] | mse={avg_loss:.4f} | "
                f"cos={avg_cos:.3f} | lr {current_lr:.2e} | "
                f"{pct:.1f}% | {sents_per_sec:.0f} sent/s")

        # --- Eval ---
        if (step + 1) % EVAL_EVERY == 0:
            avg_loss = np.mean(loss_tracker[-100:]) if loss_tracker else 0
            avg_cos = np.mean(cosine_tracker[-100:]) if cosine_tracker else 0

            results = evaluate(model, v24_model, tokenizer, device, config)
            for r in results:
                log(f"  PROMPT: {r['prompt']}")
                for i, sent in enumerate(r['generated']):
                    label = "INPUT " if i == 0 else f"GEN {i}"
                    log(f"    [{label}] {sent}")

            elapsed = time.time() - start_time
            log_metrics(step + 1, {
                "mse_loss": avg_loss,
                "cosine_sim": avg_cos,
                "lr": current_lr,
                "elapsed_hours": elapsed / 3600,
                "sentences_per_sec": 0,
            })

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
    log("FLM CONCEPT-SPACE LM — Phase 2 (Block-Causal)")
    log("  Concept slots attend within sentences (bidirectional)")
    log("  Sentences attend causally (autoregressive)")
    log("  No encode/decode roundtrip between generation steps")
    log("=" * 70)

    train(resume_from=args.resume, fresh=args.fresh, eval_only=args.eval_only)
