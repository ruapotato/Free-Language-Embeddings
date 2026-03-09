"""
FLM V8 — Staged Slot Specialization
================================================================================
Key changes from V7:
  1. Three training phases: Foundation → Slot Focus → Joint Fine-tune
  2. Soft per-slot gradient scaling (not hard freezing)
     - Phase 2 focuses on one slot at a time with reduced gradient on others
     - This GUIDES specialization without destroying learned structure
  3. Per-slot isolation stats logged every eval (SLOT_STATS line)
  4. Stronger margin losses, lower repulsion/temperature targets
  5. Phase-aware logging (which phase, which focus slot)

V7 post-mortem: slot structure developed (28/32 assigned, iso=0.63) but global
flat cosine compressed into narrow band (para=0.94, unrelated=0.84, WO=0.71).
pos_sim stuck at 0.74. Margin losses too weak at 1.5 weight on 8.4 total loss.
Repulsion target 0.3 trivially satisfied at 0.03-0.05. Clustering gap collapsed
to 0.013 despite repulsion — it prevents total collapse but doesn't create
semantic structure.

Fix: staged slot specialization. Phase 2 cycles through each slot, applying
strong gradient to the focus slot and weak gradient to others. This forces
each slot to specialize for its semantic dimension. Combined with stronger
margins and tighter repulsion, this should spread the similarity space.

Usage:
    python train_v8.py --fresh          # start from scratch (Phase 1)
    python train_v8.py --from-v7       # init from V7 checkpoint
    python train_v8.py --resume         # resume from V8 checkpoint
    python train_v8.py --eval-only      # diagnostics only
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
from concept_model import (ConceptConfig, ConceptAutoencoder,
                           reconstruction_loss, flat_info_nce_loss,
                           flat_word_order_info_nce, slot_decorrelation_loss,
                           slot_isolation_loss, flat_similarity_matrix,
                           SlotClassifiers, per_slot_contrastive_loss,
                           margin_paraphrase_loss, margin_negative_loss,
                           margin_word_order_loss, per_slot_paraphrase_loss,
                           batch_repulsion_loss)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v8"
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
BATCH_SIZE = 40
PAIR_BATCH_SIZE = 80
PEAK_LR = 3e-4
WARMUP_STEPS = 2000
TOTAL_STEPS = 200_000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0

# V8: Lower temperature for sharper discrimination
NCE_TEMPERATURE = 0.04          # was 0.07 in V7

# Loss weights — V8 tuning (much stronger margins)
# Encoder/geometry losses:
NCE_WEIGHT = 2.0
WO_WEIGHT = 1.5
DECORR_WEIGHT = 1.0
SLOT_ISO_WEIGHT = 1.0
CLASSIFY_WEIGHT = 3.0
SLOT_CON_WEIGHT = 2.0
STS_WEIGHT = 1.0
# V8: cranked up from V7
MARGIN_PARA_WEIGHT = 8.0       # was 1.5 — overwhelmed by total loss
MARGIN_NEG_WEIGHT = 5.0        # was 1.0
MARGIN_WO_WEIGHT = 2.0         # was 1.0
SLOT_PARA_WEIGHT = 2.0
REPULSION_WEIGHT = 3.0         # was 1.5

# V8: tighter targets
MARGIN_PARA_TARGET = 0.92      # was 0.85 — give pos_sim room to climb
MARGIN_NEG_TARGET = 0.2        # was 0.3
MARGIN_WO_TARGET = 0.4         # was 0.5
REPULSION_TARGET = 0.05        # was 0.3 — trivially satisfied, keep pushing

# Decoder losses:
RECON_WEIGHT = 1.0
CROSS_RECON_WEIGHT = 0.5

# Gradient taper
TAPER_BOTTLENECK = 0.3

AXIS_BATCH_SIZE = 80
SLOT_CON_BATCH_SIZE = 40
SLOT_CON_SLOTS_PER_STEP = 2

# ===================== V8: PHASE SCHEDULE =====================
# Phase 1: Foundation — normal training, establish reconstruction + slot basics
PHASE1_STEPS = 30_000

# Phase 2: Slot Focus — cycle through each slot with gradient emphasis
STEPS_PER_SLOT = 500            # steps focusing on each slot
PHASE2_STEPS = 32 * STEPS_PER_SLOT  # = 16,000 steps total

# Phase 3: Joint Fine-tune — all slots equal, remaining steps
# Starts at PHASE1_STEPS + PHASE2_STEPS = 46,000

# Gradient scaling in Phase 2
SLOT_FOCUS_SCALE = 1.0          # full gradient for focus slot
SLOT_BACKGROUND_SCALE = 0.05    # 5% gradient for non-focus slots
# ===================== END PHASE SCHEDULE =====================

# Logging
LOG_EVERY = 50
EVAL_EVERY = 500
SAMPLE_EVERY = 2000
CHECKPOINT_EVERY = 5000
GEOMETRY_EVAL_EVERY = 2000

LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v8.log",
    "metrics": f"{LOG_DIR}/concept_v8_metrics.csv",
}

# Diagnostic pairs
DIAGNOSTIC_PAIRS = [
    ("the massive cat stepped on the rug",
     "there was a rug that a massive cat stepped on", "paraphrase"),
    ("the king died",
     "the monarch passed away", "paraphrase"),
    ("the cat sat on the mat",
     "le chat etait assis sur le tapis", "crosslingual"),
    ("the cat sat on the mat",
     "the stock market crashed today", "unrelated"),
    ("the dog bit the man",
     "the man bit the dog", "word_order"),
    ("the purple man licked the sucker",
     "the man licked a purple sucker", "binding"),
    ("alice likes bob",
     "bob likes alice", "word_order"),
    ("she gave him a book",
     "he gave her a book", "word_order"),
]

SLOT_NAMES = {
    0: "subject", 1: "object", 2: "animacy", 3: "age",
    4: "size", 5: "color", 6: "shape", 7: "material",
    8: "weight", 9: "temperature", 10: "action_type", 11: "manner",
    12: "speed", 13: "direction", 14: "location", 15: "spatial",
    16: "distance", 17: "tense", 18: "duration", 19: "time_ref",
    20: "number", 21: "degree", 22: "sentiment", 23: "emotion",
    24: "arousal", 25: "quality", 26: "difficulty", 27: "negation",
    28: "certainty", 29: "causation", 30: "formality", 31: "speech_act",
}


# ---------------------------------------------------------------------------
# Word-order augmentation
# ---------------------------------------------------------------------------

def shuffle_tokens(input_ids, attention_mask, pad_id=0, cls_id=101, sep_id=102):
    """Swap 2 random content tokens per sequence."""
    shuffled = input_ids.clone()
    for i in range(input_ids.shape[0]):
        mask = attention_mask[i].bool()
        ids = input_ids[i]
        content_pos = []
        for j in range(ids.shape[0]):
            if mask[j] and ids[j].item() not in (pad_id, cls_id, sep_id):
                content_pos.append(j)
        if len(content_pos) <= 2:
            continue
        for _ in range(10):
            idx = torch.randperm(len(content_pos))[:2]
            p1, p2 = content_pos[idx[0]], content_pos[idx[1]]
            if ids[p1] != ids[p2]:
                break
        shuffled[i, p1] = ids[p2]
        shuffled[i, p2] = ids[p1]
    return shuffled, attention_mask


# ---------------------------------------------------------------------------
# Gradient scaling (V8: per-slot + taper)
# ---------------------------------------------------------------------------

class GradientScale(torch.autograd.Function):
    """Scales gradients in the backward pass. Identity in forward."""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def gradient_taper(concepts, scale):
    """Taper reconstruction gradient reaching the encoder."""
    return GradientScale.apply(concepts, scale)


class SlotGradientScale(torch.autograd.Function):
    """Per-slot gradient scaling. Each slot gets its own scale factor.

    In forward: identity (returns concepts unchanged).
    In backward: multiplies gradient for slot k by slot_scales[k].

    This is the core of V8's staged specialization: during Phase 2,
    the focus slot gets scale=1.0 while others get scale=0.05.
    """
    @staticmethod
    def forward(ctx, concepts, slot_scales):
        # concepts: (B, K, D), slot_scales: (K,)
        ctx.save_for_backward(slot_scales)
        return concepts

    @staticmethod
    def backward(ctx, grad_output):
        slot_scales, = ctx.saved_tensors
        # Scale each slot's gradient: (B, K, D) * (1, K, 1)
        return grad_output * slot_scales.unsqueeze(0).unsqueeze(-1), None


def slot_gradient_scale(concepts, slot_scales):
    """Apply per-slot gradient scaling to concept vectors."""
    return SlotGradientScale.apply(concepts, slot_scales)


# ---------------------------------------------------------------------------
# V8: Phase management
# ---------------------------------------------------------------------------

def get_phase(step):
    """Returns (phase_number, focus_slot_or_-1)."""
    if step < PHASE1_STEPS:
        return 1, -1
    elif step < PHASE1_STEPS + PHASE2_STEPS:
        phase2_step = step - PHASE1_STEPS
        focus_slot = phase2_step // STEPS_PER_SLOT
        return 2, min(focus_slot, 31)
    else:
        return 3, -1


def get_slot_scales(step, device):
    """Return per-slot gradient scale tensor for current step."""
    phase, focus_slot = get_phase(step)
    if phase == 2:
        scales = torch.full((32,), SLOT_BACKGROUND_SCALE, device=device)
        scales[focus_slot] = SLOT_FOCUS_SCALE
        return scales
    else:
        return torch.ones(32, device=device)


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


def log_metrics(step, recon_loss, pos_sim, neg_sim, wo_sim,
                eff_rank, lr, elapsed_hours):
    metrics_file = LOG_PATHS.get("metrics")
    if not metrics_file:
        return
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    write_header = not os.path.exists(metrics_file)
    with open(metrics_file, "a") as f:
        if write_header:
            f.write("timestamp,step,recon_loss,pos_sim,neg_sim,"
                    "wo_sim,eff_rank,lr,elapsed_hours\n")
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts},{step},{recon_loss:.6f},{pos_sim:.4f},{neg_sim:.4f},"
                f"{wo_sim:.4f},{eff_rank},{lr:.6e},{elapsed_hours:.4f}\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class ReconstructionDataset:
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = []
        self._load_texts()
        random.shuffle(self.texts)
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
                        if len(a) > 10:
                            self.texts.append(a)
                        if len(b) > 10:
                            self.texts.append(b)
                    except (json.JSONDecodeError, KeyError):
                        continue
        log(f"  Reconstruction texts: {len(self.texts):,}")

    def get_batch(self, batch_size):
        texts = []
        for _ in range(batch_size):
            if self.idx >= len(self.texts):
                random.shuffle(self.texts)
                self.idx = 0
            texts.append(self.texts[self.idx])
            self.idx += 1
        enc = self.tokenizer(texts, max_length=self.max_len,
                             padding=True, truncation=True,
                             return_tensors="pt")
        return enc


class PairDataset:
    """Loads all pair types: positive, hard negative, graded."""
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pos_pairs = []
        self.hard_neg_pairs = []
        self.sts_pairs = []
        self._load_pairs()
        random.shuffle(self.pos_pairs)
        random.shuffle(self.hard_neg_pairs)
        random.shuffle(self.sts_pairs)
        self.pos_idx = 0
        self.hn_idx = 0
        self.sts_idx = 0

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
                        if "sim_score" in doc:
                            score = float(doc["sim_score"])
                            self.sts_pairs.append((a, b, score))
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
        log(f"  STS graded pairs: {len(self.sts_pairs):,}")

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

    def get_hard_neg_batch(self, batch_size):
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

    def get_sts_batch(self, batch_size):
        if not self.sts_pairs:
            return None, None, None
        texts_a, texts_b, scores = [], [], []
        for _ in range(batch_size):
            if self.sts_idx >= len(self.sts_pairs):
                random.shuffle(self.sts_pairs)
                self.sts_idx = 0
            a, b, s = self.sts_pairs[self.sts_idx]
            texts_a.append(a)
            texts_b.append(b)
            scores.append(s)
            self.sts_idx += 1
        enc_a, enc_b = self._tokenize_pair(texts_a, texts_b)
        return enc_a, enc_b, torch.tensor(scores, dtype=torch.float32)


class ConceptAxisDataset:
    """Loads synthetic concept axis pairs with concept_value labels."""
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs_by_slot = {i: [] for i in range(32)}
        self.label_vocab = {}
        self._load()
        self.indices = {i: 0 for i in range(32)}

    def _load(self):
        data_dir = Path("data/concept_axes")
        combined = data_dir / "all_axes.jsonl"
        if not combined.exists():
            log(f"  WARNING: No concept axis data at {combined}")
            return
        raw_by_slot = {i: [] for i in range(32)}
        count = 0
        with open(combined) as f:
            for line in f:
                doc = json.loads(line)
                slot = doc["slot"]
                cv = doc.get("concept_value", "")
                raw_by_slot[slot].append((doc["base"], doc["variant"], cv))
                count += 1

        for slot_id in range(32):
            values = sorted(set(cv for _, _, cv in raw_by_slot[slot_id]))
            self.label_vocab[slot_id] = {v: i for i, v in enumerate(values)}

        for slot_id in range(32):
            vocab = self.label_vocab[slot_id]
            for base, var, cv in raw_by_slot[slot_id]:
                self.pairs_by_slot[slot_id].append((base, var, vocab.get(cv, 0)))
            random.shuffle(self.pairs_by_slot[slot_id])

        counts = [len(self.pairs_by_slot[s]) for s in range(32)]
        sqrt_counts = [c ** 0.5 if c > 0 else 0.0 for c in counts]
        total_sqrt = sum(sqrt_counts)
        self.slot_weights = [sc / total_sqrt for sc in sqrt_counts]
        self.active_slots = [s for s in range(32) if counts[s] > 0]

        log(f"  Concept axis pairs: {count:,} across 32 slots")
        log(f"  Active slots: {len(self.active_slots)}/32")
        min_slot = min(range(32), key=lambda s: counts[s] if counts[s] > 0 else float('inf'))
        max_slot = max(range(32), key=lambda s: counts[s])
        log(f"  Data range: slot {min_slot} ({counts[min_slot]:,}) — slot {max_slot} ({counts[max_slot]:,})")
        log(f"  Label vocab sizes: {', '.join(f'{s}:{len(v)}' for s, v in sorted(self.label_vocab.items()) if v)}")

    def get_batch(self, batch_size):
        bases, variants, slot_ids, labels = [], [], [], []
        sampled_slots = random.choices(self.active_slots,
                                       weights=[self.slot_weights[s] for s in self.active_slots],
                                       k=batch_size)
        for slot in sampled_slots:
            pairs = self.pairs_by_slot[slot]
            idx = self.indices[slot]
            if idx >= len(pairs):
                random.shuffle(pairs)
                idx = 0
            base, var, label = pairs[idx]
            self.indices[slot] = idx + 1
            bases.append(base)
            variants.append(var)
            slot_ids.append(slot)
            labels.append(label)
        enc_base = self.tokenizer(bases, max_length=self.max_len,
                                   padding=True, truncation=True, return_tensors="pt")
        enc_var = self.tokenizer(variants, max_length=self.max_len,
                                  padding=True, truncation=True, return_tensors="pt")
        return (enc_base, enc_var,
                torch.tensor(slot_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    def get_slot_batch(self, slot_id, batch_size):
        pairs = self.pairs_by_slot[slot_id]
        if not pairs:
            return None, None, None
        bases, variants, labels = [], [], []
        for _ in range(batch_size):
            idx = self.indices[slot_id]
            if idx >= len(pairs):
                random.shuffle(pairs)
                idx = 0
            base, var, label = pairs[idx]
            self.indices[slot_id] = idx + 1
            bases.append(base)
            variants.append(var)
            labels.append(label)
        enc_base = self.tokenizer(bases, max_length=self.max_len,
                                   padding=True, truncation=True, return_tensors="pt")
        enc_var = self.tokenizer(variants, max_length=self.max_len,
                                  padding=True, truncation=True, return_tensors="pt")
        return enc_base, enc_var, torch.tensor(labels, dtype=torch.long)

    @property
    def num_classes_per_slot(self):
        return {s: len(v) for s, v in self.label_vocab.items() if len(v) > 1}


# ---------------------------------------------------------------------------
# Prefetch buffer
# ---------------------------------------------------------------------------

class PrefetchBuffer:
    """Pre-tokenizes batches in background threads to keep GPU fed."""

    def __init__(self, recon_dataset, pair_dataset, axis_dataset, device,
                 pair_batch_size=128, hard_neg_batch=64, buf_size=4):
        self.recon = recon_dataset
        self.pairs = pair_dataset
        self.axis = axis_dataset
        self.device = device
        self.pair_bs = pair_batch_size
        self.hn_bs = hard_neg_batch
        self.recon_q = queue.Queue(maxsize=buf_size)
        self.pos_q = queue.Queue(maxsize=buf_size)
        self.hn_q = queue.Queue(maxsize=buf_size)
        self.sts_q = queue.Queue(maxsize=buf_size)
        self.axis_q = queue.Queue(maxsize=buf_size)
        self._stop = threading.Event()
        self._threads = []

    def start(self):
        for name, fn in [("recon", self._fill_recon),
                         ("pos", self._fill_pos),
                         ("hn", self._fill_hn),
                         ("sts", self._fill_sts),
                         ("axis", self._fill_axis)]:
            t = threading.Thread(target=fn, daemon=True, name=f"prefetch-{name}")
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()

    def _fill_recon(self):
        while not self._stop.is_set():
            try:
                batch = self.recon.get_batch(BATCH_SIZE)
                self.recon_q.put(batch, timeout=1.0)
            except queue.Full:
                continue

    def _fill_pos(self):
        while not self._stop.is_set():
            try:
                batch = self.pairs.get_pos_batch(self.pair_bs)
                self.pos_q.put(batch, timeout=1.0)
            except queue.Full:
                continue

    def _fill_hn(self):
        while not self._stop.is_set():
            try:
                batch = self.pairs.get_hard_neg_batch(self.hn_bs)
                self.hn_q.put(batch, timeout=1.0)
            except queue.Full:
                continue

    def _fill_sts(self):
        while not self._stop.is_set():
            try:
                batch = self.pairs.get_sts_batch(min(64, self.pair_bs))
                self.sts_q.put(batch, timeout=1.0)
            except queue.Full:
                continue

    def _fill_axis(self):
        while not self._stop.is_set():
            try:
                batch = self.axis.get_batch(AXIS_BATCH_SIZE)
                self.axis_q.put(batch, timeout=1.0)
            except queue.Full:
                continue

    def get_recon(self):
        return self.recon_q.get()
    def get_pos(self):
        return self.pos_q.get()
    def get_hn(self):
        return self.hn_q.get()
    def get_sts(self):
        return self.sts_q.get()
    def get_axis(self):
        return self.axis_q.get()


# ---------------------------------------------------------------------------
# Loss helpers (flat cosine)
# ---------------------------------------------------------------------------

def hard_negative_info_nce(concepts_pos_a, concepts_pos_b,
                           concepts_hn_a, concepts_hn_b,
                           temperature=0.07):
    """InfoNCE with explicit hard negatives. Uses FLAT cosine similarity."""
    B_pos = concepts_pos_a.shape[0]
    sim_pos_raw = flat_similarity_matrix(concepts_pos_a, concepts_pos_b)

    if concepts_hn_a is not None:
        hn_all = torch.cat([concepts_hn_a, concepts_hn_b], dim=0)
        sim_hn_raw = flat_similarity_matrix(concepts_pos_a, hn_all)
        sim_all = torch.cat([sim_pos_raw, sim_hn_raw], dim=1) / temperature
    else:
        sim_all = sim_pos_raw / temperature

    labels = torch.arange(B_pos, device=sim_all.device)
    sim_pos_block = sim_pos_raw / temperature
    loss = (F.cross_entropy(sim_all, labels) +
            F.cross_entropy(sim_pos_block.T, labels)) / 2

    with torch.no_grad():
        pos_sim = sim_pos_raw.diag().mean().item()
        mask = ~torch.eye(B_pos, dtype=torch.bool, device=sim_pos_raw.device)
        neg_sim = sim_pos_raw[mask].mean().item()

    return loss, pos_sim, neg_sim


def sts_loss(concepts_a, concepts_b, target_scores):
    """Graded similarity using FLAT cosine."""
    sim_matrix = flat_similarity_matrix(concepts_a, concepts_b)
    pred_sim = sim_matrix.diag()
    loss = F.mse_loss(pred_sim, target_scores.to(pred_sim.device))
    return loss, pred_sim.mean().item()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _unwrap(model):
    return model._orig_mod if hasattr(model, '_orig_mod') else model


@torch.no_grad()
def run_diagnostics(model, tokenizer, device="cuda"):
    model.eval()
    m = _unwrap(model)
    results = []
    for text_a, text_b, pair_type in DIAGNOSTIC_PAIRS:
        enc_a = tokenizer(text_a, max_length=128, padding=True,
                          truncation=True, return_tensors="pt").to(device)
        enc_b = tokenizer(text_b, max_length=128, padding=True,
                          truncation=True, return_tensors="pt").to(device)
        concepts_a = m.encode(enc_a["input_ids"], enc_a["attention_mask"])
        concepts_b = m.encode(enc_b["input_ids"], enc_b["attention_mask"])
        sim = flat_similarity_matrix(concepts_a, concepts_b).item()
        results.append((text_a[:45], text_b[:45], sim, pair_type))
    model.train()
    return results


@torch.no_grad()
def measure_effective_rank(model, tokenizer, texts, device="cuda"):
    model.eval()
    m = _unwrap(model)
    enc = tokenizer(texts[:200], max_length=128, padding=True,
                    truncation=True, return_tensors="pt").to(device)
    concepts = m.encode(enc["input_ids"], enc["attention_mask"])
    flat = concepts.view(concepts.shape[0], -1).cpu().numpy()
    flat = flat - flat.mean(axis=0)
    _, s, _ = np.linalg.svd(flat, full_matrices=False)
    var = s ** 2
    var_ratio = var / var.sum()
    cumvar = np.cumsum(var_ratio)
    rank90 = int(np.searchsorted(cumvar, 0.90)) + 1
    rank95 = int(np.searchsorted(cumvar, 0.95)) + 1
    model.train()
    return rank90, rank95, var_ratio[:10]


@torch.no_grad()
def measure_slot_accuracy(model, axis_dataset, tokenizer, device="cuda"):
    """Returns per-slot dict with target_sim, other_sim, isolation, correct_slot."""
    model.eval()
    m = _unwrap(model)
    slot_scores = {}
    for slot_id in range(32):
        pairs = axis_dataset.pairs_by_slot[slot_id]
        if len(pairs) < 10:
            continue
        sample = random.sample(pairs, min(32, len(pairs)))
        bases = [p[0] for p in sample]
        variants = [p[1] for p in sample]
        enc_b = tokenizer(bases, max_length=128, padding=True,
                          truncation=True, return_tensors="pt").to(device)
        enc_v = tokenizer(variants, max_length=128, padding=True,
                          truncation=True, return_tensors="pt").to(device)
        c_base = m.encode(enc_b["input_ids"], enc_b["attention_mask"])
        c_var = m.encode(enc_v["input_ids"], enc_v["attention_mask"])
        normed_b = F.normalize(c_base, p=2, dim=-1)
        normed_v = F.normalize(c_var, p=2, dim=-1)
        slot_sims = (normed_b * normed_v).sum(dim=-1)
        target_sim = slot_sims[:, slot_id].mean().item()
        mask = torch.ones(32, dtype=torch.bool, device=device)
        mask[slot_id] = False
        other_sim = slot_sims[:, mask].mean().item()
        isolation = other_sim - target_sim
        avg_sims = slot_sims.mean(dim=0)
        most_changed = avg_sims.argmin().item()
        slot_scores[slot_id] = {
            "target_sim": target_sim,
            "other_sim": other_sim,
            "isolation": isolation,
            "correct_slot": most_changed == slot_id,
        }
    model.train()
    return slot_scores


@torch.no_grad()
def measure_geometry(model, tokenizer, device="cuda"):
    """Measure clustering gap and direction consistency using FLAT cosine."""
    model.eval()
    m = _unwrap(model)

    def encode_texts(texts):
        enc = tokenizer(texts, max_length=64, padding=True,
                        truncation=True, return_tensors="pt").to(device)
        return m.encode(enc["input_ids"], enc["attention_mask"])

    groups = {
        "animals": ["the cat sat on the mat", "a dog ran in the park",
                     "the bird flew over the tree", "fish swim in the ocean"],
        "weather": ["it is raining heavily today", "the sun is shining brightly",
                     "snow covered the ground", "a storm is approaching fast"],
        "food": ["she cooked a delicious pasta", "the pizza was freshly baked",
                  "he ate a bowl of rice", "the soup was too salty"],
        "emotions": ["she was very happy today", "he felt sad and lonely",
                      "the news made them angry", "they were excited about the trip"],
        "tech": ["the computer crashed again", "she updated her phone software",
                  "the internet was slow", "he wrote a python program"],
    }

    group_concepts = {}
    for name, sents in groups.items():
        group_concepts[name] = encode_texts(sents)

    within_sims = []
    for concepts in group_concepts.values():
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                sim = flat_similarity_matrix(concepts[i:i+1], concepts[j:j+1]).item()
                within_sims.append(sim)

    between_sims = []
    gnames = list(group_concepts.keys())
    for i in range(len(gnames)):
        for j in range(i + 1, len(gnames)):
            ca, cb = group_concepts[gnames[i]], group_concepts[gnames[j]]
            for a in range(len(ca)):
                for b in range(len(cb)):
                    sim = flat_similarity_matrix(ca[a:a+1], cb[b:b+1]).item()
                    between_sims.append(sim)

    clustering_gap = np.mean(within_sims) - np.mean(between_sims)

    directions = {
        "negation": [("the cat is not here", "the cat is here"),
                     ("she did not run", "she did run"),
                     ("he is not happy", "he is happy")],
        "sentiment": [("the movie was great", "the movie was terrible"),
                      ("she loves the food", "she hates the food"),
                      ("a wonderful day", "a horrible day")],
        "tense": [("she ran quickly", "she runs quickly"),
                  ("he walked home", "he walks home"),
                  ("they played outside", "they play outside")],
    }

    dir_consistencies = []
    for attr, pairs in directions.items():
        deltas = []
        for pos, neg in pairs:
            c_pos = encode_texts([pos])
            c_neg = encode_texts([neg])
            delta = c_pos.view(1, -1) - c_neg.view(1, -1)
            delta = F.normalize(delta, p=2, dim=-1)
            deltas.append(delta)
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                dir_consistencies.append(
                    F.cosine_similarity(deltas[i], deltas[j]).item())

    direction_consistency = np.mean(dir_consistencies)

    model.train()
    return {
        "clustering_gap": clustering_gap,
        "direction_consistency": direction_consistency,
        "within_sim": np.mean(within_sims),
        "between_sim": np.mean(between_sims),
    }


@torch.no_grad()
def test_reconstruction(model, tokenizer, device="cuda"):
    model.eval()
    m = _unwrap(model)
    test_texts = [
        "the purple man licked the sucker",
        "the man licked a purple sucker",
        "the dog bit the man",
        "the man bit the dog",
        "she runs every morning",
    ]
    results = []
    for text in test_texts:
        enc = tokenizer(text, max_length=128, padding=True,
                        truncation=True, return_tensors="pt").to(device)
        concepts = m.encode(enc["input_ids"], enc["attention_mask"])
        bos_id = tokenizer.cls_token_id or 101
        generated = [bos_id]
        for _ in range(len(enc["input_ids"][0]) + 5):
            dec_input = torch.tensor([generated], device=device)
            logits = m.decode(dec_input, concepts)
            next_token = logits[0, -1].argmax().item()
            if next_token == tokenizer.sep_token_id:
                break
            generated.append(next_token)
        decoded = tokenizer.decode(generated[1:], skip_special_tokens=True)
        results.append((text, decoded))
    model.train()
    return results


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

def save_checkpoint(model, classifiers, optimizer, scaler, config,
                    step, loss, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = model.state_dict()
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    ckpt = {
        "model_state_dict": state,
        "classifier_state_dict": classifiers.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "loss": loss,
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "v8",
    }
    path = os.path.join(checkpoint_dir, f"step_{step:06d}.pt")
    torch.save(ckpt, path)
    latest = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(ckpt, latest)
    log(f"  Checkpoint saved: {path}")

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


def load_checkpoint(path, axis_dataset, device="cuda"):
    log(f"Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoder(config).to(device)
    state = ckpt["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    classifiers = SlotClassifiers(
        config.concept_dim, axis_dataset.num_classes_per_slot).to(device)
    if "classifier_state_dict" in ckpt:
        classifiers.load_state_dict(ckpt["classifier_state_dict"])

    all_params = list(model.parameters()) + list(classifiers.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=PEAK_LR,
                                  betas=BETAS, weight_decay=WEIGHT_DECAY)
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, RuntimeError):
            log("  WARNING: Could not load optimizer state (param count changed)")

    scaler = torch.amp.GradScaler("cuda")
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    step = ckpt.get("step", 0)
    total, _ = model.count_parameters()
    cls_params = sum(p.numel() for p in classifiers.parameters())
    log(f"Resumed: {total:,} model + {cls_params:,} classifier params | step {step}")
    return model, classifiers, optimizer, scaler, config, step


def load_from_v7(axis_dataset, device="cuda"):
    """Initialize V8 from a V7 checkpoint."""
    v7_path = "checkpoints/concept_v7/latest.pt"
    log(f"Initializing from V7 checkpoint: {v7_path}")
    ckpt = torch.load(v7_path, map_location="cpu", weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoder(config).to(device)
    state = ckpt["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    classifiers = SlotClassifiers(
        config.concept_dim, axis_dataset.num_classes_per_slot).to(device)
    if "classifier_state_dict" in ckpt:
        try:
            classifiers.load_state_dict(ckpt["classifier_state_dict"])
            log("  Loaded V7 classifier weights")
        except (ValueError, RuntimeError):
            log("  WARNING: V7 classifier mismatch, using fresh classifiers")

    all_params = list(model.parameters()) + list(classifiers.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=PEAK_LR,
                                  betas=BETAS, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")

    total, _ = model.count_parameters()
    cls_params = sum(p.numel() for p in classifiers.parameters())
    log(f"V7->V8: {total:,} model + {cls_params:,} classifier params")
    return model, classifiers, optimizer, scaler, config


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False, from_v7=False, eval_only=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V8 — STAGED SLOT SPECIALIZATION")
    log("=" * 70)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    log(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    model_config = dict(MODEL_CONFIG)
    model_config["vocab_size"] = tokenizer.vocab_size

    log("Loading data...")
    recon_dataset = ReconstructionDataset(tokenizer, max_len=128)
    pair_dataset = PairDataset(tokenizer, max_len=128)
    axis_dataset = ConceptAxisDataset(tokenizer, max_len=128)

    if len(recon_dataset.texts) == 0:
        log("ERROR: No training texts found.")
        return

    if resume_from is None and not fresh and not from_v7:
        latest = Path(CHECKPOINT_DIR) / "latest.pt"
        if latest.exists():
            resume_from = str(latest)

    if resume_from:
        model, classifiers, optimizer, scaler, config, start_step = \
            load_checkpoint(resume_from, axis_dataset, device)
    elif from_v7:
        model, classifiers, optimizer, scaler, config = \
            load_from_v7(axis_dataset, device)
        start_step = 0
    else:
        log("Starting fresh training...")
        config = ConceptConfig(**model_config)
        model = ConceptAutoencoder(config).to(device)
        classifiers = SlotClassifiers(
            config.concept_dim, axis_dataset.num_classes_per_slot).to(device)
        all_params = list(model.parameters()) + list(classifiers.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=PEAK_LR,
                                      betas=BETAS, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0
        total, _ = model.count_parameters()
        cls_params = sum(p.numel() for p in classifiers.parameters())
        log(f"Model: {total:,} params ({total/1e6:.1f}M) + {cls_params:,} classifier")
        log(f"Bottleneck: {config.num_concepts} concepts x {config.concept_dim} dim "
            f"= {config.num_concepts * config.concept_dim} total dims")

    if eval_only:
        log("\n--- DIAGNOSTICS ---")
        results = run_diagnostics(model, tokenizer, device)
        for a, b, sim, ptype in results:
            log(f"  {sim:+.4f}  [{ptype:<12s}] {a}  <->  {b}")
        log("\n--- GEOMETRY ---")
        geo = measure_geometry(model, tokenizer, device)
        log(f"  Clustering gap: {geo['clustering_gap']:+.3f}")
        log(f"  Direction consistency: {geo['direction_consistency']:.3f}")
        log("\n--- RECONSTRUCTION ---")
        recon = test_reconstruction(model, tokenizer, device)
        for orig, decoded in recon:
            log(f"  IN:  {orig}")
            log(f"  OUT: {decoded}")
            log("")
        log("\n--- PER-SLOT STATS ---")
        slot_scores = measure_slot_accuracy(model, axis_dataset, tokenizer, device)
        for sid in sorted(slot_scores.keys()):
            s = slot_scores[sid]
            name = SLOT_NAMES.get(sid, f"slot_{sid}")
            status = "OK" if s["isolation"] > 0.1 else "WEAK" if s["isolation"] > 0 else "BAD"
            assigned = "Y" if s["correct_slot"] else "N"
            log(f"  [{status:>4s}] slot {sid:2d} ({name:>12s}): "
                f"target_sim={s['target_sim']:.3f} other_sim={s['other_sim']:.3f} "
                f"isolation={s['isolation']:+.3f} assigned={assigned}")
        return

    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()
    classifiers.train()

    rank_texts = random.sample(recon_dataset.texts, min(200, len(recon_dataset.texts)))

    log("\n--- INITIAL DIAGNOSTICS ---")
    results = run_diagnostics(model, tokenizer, device)
    for a, b, sim, ptype in results:
        log(f"  {sim:+.4f}  [{ptype:<12s}] {a}  <->  {b}")

    geo = measure_geometry(model, tokenizer, device)
    log(f"  Clustering gap: {geo['clustering_gap']:+.3f} | "
        f"Direction consistency: {geo['direction_consistency']:.3f}")

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
    HARD_NEG_BATCH = min(64, PAIR_BATCH_SIZE // 2)

    axis_total = sum(len(v) for v in axis_dataset.pairs_by_slot.values())

    log(f"\nTraining plan (V8 — Staged Slot Specialization):")
    log(f"  Encoder: {config.enc_hidden}h x {config.enc_layers}L x {config.enc_heads}heads")
    log(f"  Decoder: {config.dec_hidden}h x {config.dec_layers}L x {config.dec_heads}heads")
    log(f"  Bottleneck: {config.num_concepts} x {config.concept_dim} = "
        f"{config.num_concepts * config.concept_dim} dims")
    log(f"  V8 PHASES: P1=Foundation({PHASE1_STEPS}) -> P2=SlotFocus({PHASE2_STEPS}, "
        f"{STEPS_PER_SLOT}/slot) -> P3=JointFinetune(rest)")
    log(f"  Phase 2 scales: focus={SLOT_FOCUS_SCALE} background={SLOT_BACKGROUND_SCALE}")
    log(f"  NCE temp: {NCE_TEMPERATURE} | Margin targets: para>{MARGIN_PARA_TARGET} "
        f"neg<{MARGIN_NEG_TARGET} wo<{MARGIN_WO_TARGET} repul<{REPULSION_TARGET}")
    log(f"  Batch: {BATCH_SIZE} recon + {PAIR_BATCH_SIZE} pairs + {HARD_NEG_BATCH} hard neg + "
        f"{AXIS_BATCH_SIZE} axis + {SLOT_CON_SLOTS_PER_STEP}x{SLOT_CON_BATCH_SIZE} slot_con")
    log(f"  Peak LR: {PEAK_LR} | Steps: {start_step} -> {TOTAL_STEPS}")
    log(f"  Encoder: nce={NCE_WEIGHT} wo={WO_WEIGHT} decorr={DECORR_WEIGHT} "
        f"iso={SLOT_ISO_WEIGHT} cls={CLASSIFY_WEIGHT} scon={SLOT_CON_WEIGHT} sts={STS_WEIGHT}")
    log(f"  Margins: m_para={MARGIN_PARA_WEIGHT} m_neg={MARGIN_NEG_WEIGHT} "
        f"m_wo={MARGIN_WO_WEIGHT} slot_para={SLOT_PARA_WEIGHT} repul={REPULSION_WEIGHT}")
    log(f"  Decoder: recon={RECON_WEIGHT} cross_recon={CROSS_RECON_WEIGHT}")
    log(f"  Data: {len(pair_dataset.pos_pairs):,} pos + "
        f"{len(pair_dataset.hard_neg_pairs):,} hard neg + "
        f"{len(pair_dataset.sts_pairs):,} STS + "
        f"{axis_total:,} axis")
    log("-" * 70)

    # Start prefetch
    prefetch = PrefetchBuffer(recon_dataset, pair_dataset, axis_dataset, device,
                              pair_batch_size=PAIR_BATCH_SIZE,
                              hard_neg_batch=HARD_NEG_BATCH)
    prefetch.start()
    log("Prefetch buffer started (5 threads)")

    # Track phase transitions for logging
    prev_phase = -1
    prev_focus = -1

    for step in range(start_step, TOTAL_STEPS):
        if shutdown_requested:
            break

        # Phase management
        phase, focus_slot = get_phase(step)
        slot_scales = get_slot_scales(step, device)

        # Log phase transitions
        if phase != prev_phase or (phase == 2 and focus_slot != prev_focus):
            if phase == 1:
                log(f"  PHASE 1: Foundation training (steps {step}-{PHASE1_STEPS})")
            elif phase == 2:
                slot_name = SLOT_NAMES.get(focus_slot, f"slot_{focus_slot}")
                log(f"  PHASE 2: Focus slot {focus_slot} ({slot_name}) "
                    f"| scale={SLOT_FOCUS_SCALE} focus, {SLOT_BACKGROUND_SCALE} others "
                    f"| steps {step}-{step + STEPS_PER_SLOT}")
            elif phase == 3:
                log(f"  PHASE 3: Joint fine-tune (steps {step}-{TOTAL_STEPS})")
            prev_phase = phase
            prev_focus = focus_slot

        current_lr = cosine_lr(step, TOTAL_STEPS, PEAK_LR, WARMUP_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)

        # =================================================================
        # ENCODER PASS — geometry losses (with per-slot gradient scaling)
        # =================================================================

        recon_enc = prefetch.get_recon()
        input_ids = recon_enc["input_ids"].to(device, non_blocking=True)
        attention_mask = recon_enc["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            concepts_raw = m.encode(input_ids, attention_mask)

            # V8: apply per-slot gradient scaling for geometry losses
            concepts = slot_gradient_scale(concepts_raw, slot_scales)

            # Slot decorrelation
            decorr = slot_decorrelation_loss(concepts)
            geo_loss = DECORR_WEIGHT * decorr

            # Batch repulsion (V8: tighter target)
            repul_loss, repul_sim = batch_repulsion_loss(concepts, target_sim=REPULSION_TARGET)
            geo_loss = geo_loss + REPULSION_WEIGHT * repul_loss

            # Word-order InfoNCE
            shuffled_ids, shuffled_mask = shuffle_tokens(input_ids, attention_mask)
            concepts_shuf_raw = m.encode(shuffled_ids, shuffled_mask)
            concepts_shuf = slot_gradient_scale(concepts_shuf_raw, slot_scales)
            wo_loss, wo_sim_val = flat_word_order_info_nce(
                concepts, concepts_shuf, temperature=NCE_TEMPERATURE)
            geo_loss = geo_loss + WO_WEIGHT * wo_loss

            # Margin word-order (V8: tighter target)
            m_wo_loss, m_wo_sim = margin_word_order_loss(
                concepts, concepts_shuf, target_sim=MARGIN_WO_TARGET)
            geo_loss = geo_loss + MARGIN_WO_WEIGHT * m_wo_loss

        wo_loss_val = wo_loss.item()
        decorr_val = decorr.item()
        m_wo_val = m_wo_loss.item()
        repul_val = repul_loss.item()
        repul_sim_val = repul_sim

        # --- Slot isolation + classification + contrastive (synthetic) ---
        iso_loss_val = 0.0
        cls_loss_val = 0.0
        cls_acc_val = 0.0
        scon_loss_val = 0.0

        axis_enc_base, axis_enc_var, axis_slots, axis_labels = prefetch.get_axis()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            ax_ids_b = axis_enc_base["input_ids"].to(device, non_blocking=True)
            ax_mask_b = axis_enc_base["attention_mask"].to(device, non_blocking=True)
            ax_ids_v = axis_enc_var["input_ids"].to(device, non_blocking=True)
            ax_mask_v = axis_enc_var["attention_mask"].to(device, non_blocking=True)
            ax_labels = axis_labels.to(device, non_blocking=True)
            ax_slots_d = axis_slots.to(device, non_blocking=True)

            ax_concepts_b_raw = m.encode(ax_ids_b, ax_mask_b)
            ax_concepts_v_raw = m.encode(ax_ids_v, ax_mask_v)
            ax_concepts_b = slot_gradient_scale(ax_concepts_b_raw, slot_scales)
            ax_concepts_v = slot_gradient_scale(ax_concepts_v_raw, slot_scales)

            # Slot isolation loss
            iso_loss = torch.tensor(0.0, device=device)
            unique_slots = axis_slots.unique()
            for s in unique_slots:
                mask = axis_slots == s
                if mask.sum() < 2:
                    continue
                iso_loss = iso_loss + slot_isolation_loss(
                    ax_concepts_b[mask], ax_concepts_v[mask], s.item())
            iso_loss = iso_loss / max(len(unique_slots), 1)
            geo_loss = geo_loss + SLOT_ISO_WEIGHT * iso_loss
            iso_loss_val = iso_loss.item()

            # Per-slot classifiers
            cls_loss = classifiers(ax_concepts_v, ax_slots_d, ax_labels)
            geo_loss = geo_loss + CLASSIFY_WEIGHT * cls_loss
            cls_loss_val = cls_loss.item()

            with torch.no_grad():
                accs = classifiers.accuracy(ax_concepts_v, ax_slots_d, ax_labels)
                cls_acc_val = np.mean(list(accs.values())) if accs else 0.0

        # Per-slot contrastive (synthetic)
        # V8: In Phase 2, always include the focus slot
        with torch.amp.autocast("cuda", dtype=torch.float16):
            scon_loss = torch.tensor(0.0, device=device)
            scon_count = 0

            if phase == 2 and focus_slot in axis_dataset.active_slots:
                # Always train focus slot in Phase 2
                sample_slots = [focus_slot]
                remaining = [s for s in axis_dataset.active_slots if s != focus_slot]
                if remaining and SLOT_CON_SLOTS_PER_STEP > 1:
                    sample_slots.extend(random.sample(
                        remaining, min(SLOT_CON_SLOTS_PER_STEP - 1, len(remaining))))
            else:
                sample_slots = random.sample(
                    axis_dataset.active_slots,
                    min(SLOT_CON_SLOTS_PER_STEP, len(axis_dataset.active_slots)))

            for slot_id in sample_slots:
                enc_b, enc_v, s_labels = axis_dataset.get_slot_batch(
                    slot_id, SLOT_CON_BATCH_SIZE)
                if enc_b is None:
                    continue
                s_ids = enc_v["input_ids"].to(device, non_blocking=True)
                s_mask = enc_v["attention_mask"].to(device, non_blocking=True)
                s_labels_d = s_labels.to(device, non_blocking=True)
                s_concepts_raw = m.encode(s_ids, s_mask)
                s_concepts = slot_gradient_scale(s_concepts_raw, slot_scales)
                scon_loss = scon_loss + per_slot_contrastive_loss(
                    s_concepts, slot_id, s_labels_d, temperature=NCE_TEMPERATURE)
                scon_count += 1
            if scon_count > 0:
                scon_loss = scon_loss / scon_count
            geo_loss = geo_loss + SLOT_CON_WEIGHT * scon_loss
            scon_loss_val = scon_loss.item()

        # --- Paraphrase InfoNCE with hard negatives ---
        nce_loss_val = 0.0
        p_sim_val = 0.0
        n_sim_val = 0.0
        m_para_val = 0.0
        m_neg_val = 0.0
        slot_para_val = 0.0
        slot_para_sim = 0.0
        concepts_a_for_cross = None
        enc_b_for_cross = None

        if pair_dataset.pos_pairs:
            enc_a, enc_b = prefetch.get_pos()
            hn_enc_a, hn_enc_b = prefetch.get_hn()

            with torch.amp.autocast("cuda", dtype=torch.float16):
                ids_a = enc_a["input_ids"].to(device, non_blocking=True)
                mask_a = enc_a["attention_mask"].to(device, non_blocking=True)
                ids_b = enc_b["input_ids"].to(device, non_blocking=True)
                mask_b = enc_b["attention_mask"].to(device, non_blocking=True)
                concepts_a_raw = m.encode(ids_a, mask_a)
                concepts_b_raw = m.encode(ids_b, mask_b)
                concepts_a = slot_gradient_scale(concepts_a_raw, slot_scales)
                concepts_b = slot_gradient_scale(concepts_b_raw, slot_scales)

                concepts_hna = None
                concepts_hnb = None
                if hn_enc_a is not None:
                    hn_ids_a = hn_enc_a["input_ids"].to(device, non_blocking=True)
                    hn_mask_a = hn_enc_a["attention_mask"].to(device, non_blocking=True)
                    hn_ids_b = hn_enc_b["input_ids"].to(device, non_blocking=True)
                    hn_mask_b = hn_enc_b["attention_mask"].to(device, non_blocking=True)
                    concepts_hna_raw = m.encode(hn_ids_a, hn_mask_a)
                    concepts_hnb_raw = m.encode(hn_ids_b, hn_mask_b)
                    concepts_hna = slot_gradient_scale(concepts_hna_raw, slot_scales)
                    concepts_hnb = slot_gradient_scale(concepts_hnb_raw, slot_scales)

                # Flat cosine InfoNCE
                nce_loss, p_sim_val, n_sim_val = hard_negative_info_nce(
                    concepts_a, concepts_b,
                    concepts_hna, concepts_hnb,
                    temperature=NCE_TEMPERATURE)
                geo_loss = geo_loss + NCE_WEIGHT * nce_loss

                # Margin paraphrase (V8: higher target)
                m_para_loss, _ = margin_paraphrase_loss(
                    concepts_a, concepts_b, target_sim=MARGIN_PARA_TARGET)
                geo_loss = geo_loss + MARGIN_PARA_WEIGHT * m_para_loss
                m_para_val = m_para_loss.item()

                # Margin negative (V8: tighter target)
                if concepts_hna is not None:
                    m_neg_loss, _ = margin_negative_loss(
                        concepts_hna, concepts_hnb, target_sim=MARGIN_NEG_TARGET)
                    geo_loss = geo_loss + MARGIN_NEG_WEIGHT * m_neg_loss
                    m_neg_val = m_neg_loss.item()

                # Per-slot paraphrase on REAL data
                sp_loss, sp_sim = per_slot_paraphrase_loss(concepts_a, concepts_b)
                geo_loss = geo_loss + SLOT_PARA_WEIGHT * sp_loss
                slot_para_val = sp_loss.item()
                slot_para_sim = sp_sim

            nce_loss_val = nce_loss.item()

            # Save for cross-reconstruction (use raw, no slot scaling)
            concepts_a_for_cross = concepts_a_raw.detach()
            enc_b_for_cross = enc_b

        # --- STS graded loss ---
        sts_loss_val = 0.0
        sts_enc_a, sts_enc_b, sts_scores = prefetch.get_sts()
        if sts_enc_a is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                sts_ids_a = sts_enc_a["input_ids"].to(device, non_blocking=True)
                sts_mask_a = sts_enc_a["attention_mask"].to(device, non_blocking=True)
                sts_ids_b = sts_enc_b["input_ids"].to(device, non_blocking=True)
                sts_mask_b = sts_enc_b["attention_mask"].to(device, non_blocking=True)
                sts_ca_raw = m.encode(sts_ids_a, sts_mask_a)
                sts_cb_raw = m.encode(sts_ids_b, sts_mask_b)
                sts_ca = slot_gradient_scale(sts_ca_raw, slot_scales)
                sts_cb = slot_gradient_scale(sts_cb_raw, slot_scales)
                s_loss, _ = sts_loss(sts_ca, sts_cb, sts_scores)
                geo_loss = geo_loss + STS_WEIGHT * s_loss
            sts_loss_val = s_loss.item()

        # =================================================================
        # DECODER PASS — uses raw concepts with taper (no slot scaling)
        # Decoder trains all slots equally so reconstruction stays healthy
        # =================================================================

        with torch.amp.autocast("cuda", dtype=torch.float16):
            targets = input_ids[:, 1:]
            dec_input = input_ids[:, :-1]
            concepts_for_dec = gradient_taper(concepts_raw, TAPER_BOTTLENECK)
            logits = m.decode(dec_input, concepts_for_dec)
            r_loss = reconstruction_loss(logits, targets)
            dec_loss = RECON_WEIGHT * r_loss

        r_loss_val = r_loss.item()

        # Cross-reconstruction
        cross_recon_val = 0.0
        if concepts_a_for_cross is not None and enc_b_for_cross is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                cross_ids_b = enc_b_for_cross["input_ids"].to(device, non_blocking=True)
                cross_targets = cross_ids_b[:, 1:]
                cross_dec_input = cross_ids_b[:, :-1]
                cross_logits = m.decode(cross_dec_input, concepts_a_for_cross)
                cross_r = reconstruction_loss(cross_logits, cross_targets)
                dec_loss = dec_loss + CROSS_RECON_WEIGHT * cross_r
            cross_recon_val = cross_r.item()

        # =================================================================
        # COMBINED BACKWARD
        # =================================================================

        total_loss = geo_loss + dec_loss

        if torch.isnan(total_loss):
            log(f"NaN loss at step {step}, skipping")
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(classifiers.parameters()), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss_val = total_loss.item()
        losses.append(total_loss_val)

        # Log — V8 format includes phase info
        if (step + 1) % LOG_EVERY == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            pct = (step + 1) / TOTAL_STEPS * 100
            phase_tag = f"P{phase}"
            if phase == 2:
                phase_tag += f"s{focus_slot}"
            log(f"step {step+1:>7d} | loss {avg_loss:.4f} "
                f"(recon={r_loss_val:.3f} xrecon={cross_recon_val:.3f} "
                f"nce={nce_loss_val:.3f} wo={wo_loss_val:.3f} "
                f"decorr={decorr_val:.3f} iso={iso_loss_val:.3f} "
                f"cls={cls_loss_val:.3f} scon={scon_loss_val:.3f} "
                f"sts={sts_loss_val:.3f} "
                f"m_para={m_para_val:.3f} m_neg={m_neg_val:.3f} "
                f"m_wo={m_wo_val:.3f} sp={slot_para_val:.3f} "
                f"repul={repul_val:.3f}) | "
                f"p_sim={p_sim_val:.3f} n_sim={n_sim_val:.3f} "
                f"wo_sim={wo_sim_val:.3f} sp_sim={slot_para_sim:.3f} "
                f"r_sim={repul_sim_val:.3f} "
                f"cls_acc={cls_acc_val:.3f} | "
                f"lr {current_lr:.2e} | {phase_tag} {pct:.1f}%")

        # Eval — includes per-slot stats
        if (step + 1) % EVAL_EVERY == 0:
            results = run_diagnostics(model, tokenizer, device)
            word_order_sims = [sim for _, _, sim, pt in results if pt == "word_order"]
            para_sims = [sim for _, _, sim, pt in results if pt == "paraphrase"]
            unrelated_sims = [sim for _, _, sim, pt in results if pt == "unrelated"]

            avg_para = sum(para_sims) / len(para_sims) if para_sims else 0
            avg_wo = sum(word_order_sims) / len(word_order_sims) if word_order_sims else 0
            avg_neg = sum(unrelated_sims) / len(unrelated_sims) if unrelated_sims else 0

            rank90, rank95, top_var = measure_effective_rank(
                model, tokenizer, rank_texts, device)

            slot_scores = measure_slot_accuracy(
                model, axis_dataset, tokenizer, device)
            avg_iso = sum(s["isolation"] for s in slot_scores.values()) / max(len(slot_scores), 1)
            good_slots = sum(1 for s in slot_scores.values() if s["isolation"] > 0.1)
            correct_slots = sum(1 for s in slot_scores.values() if s["correct_slot"])

            log(f"  EVAL: para_sim={avg_para:.3f} wo_sim={avg_wo:.3f} "
                f"unrelated_sim={avg_neg:.3f} | rank90={rank90} rank95={rank95} | "
                f"slot_iso={avg_iso:.3f} ({good_slots}/32 good) "
                f"slot_assign={correct_slots}/32")

            for a, b, sim, ptype in results:
                log(f"    {sim:+.4f}  [{ptype:<12s}] {a} <-> {b}")

            # V8: Per-slot isolation stats (compact one-liner for dashboard)
            slot_iso_parts = []
            for sid in sorted(slot_scores.keys()):
                s = slot_scores[sid]
                assigned = "Y" if s["correct_slot"] else "N"
                slot_iso_parts.append(f"{sid}:{s['isolation']:+.2f}{assigned}")
            log(f"  SLOT_STATS: {' '.join(slot_iso_parts)}")

            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            log_metrics(step + 1, avg_loss, avg_para, avg_neg,
                        avg_wo, rank90, current_lr, elapsed / 3600)

        # Geometry eval
        if (step + 1) % GEOMETRY_EVAL_EVERY == 0:
            geo = measure_geometry(model, tokenizer, device)
            log(f"  GEO: clustering_gap={geo['clustering_gap']:+.3f} "
                f"dir_consistency={geo['direction_consistency']:.3f} "
                f"within={geo['within_sim']:.3f} between={geo['between_sim']:.3f}")

        # Reconstruction samples + detailed slot breakdown
        if (step + 1) % SAMPLE_EVERY == 0:
            log("--- RECONSTRUCTION SAMPLES ---")
            recon = test_reconstruction(model, tokenizer, device)
            for orig, decoded in recon:
                match = "OK" if orig.lower().strip() == decoded.lower().strip() else "DIFF"
                log(f"  [{match}] {orig}")
                log(f"        -> {decoded}")

            log("--- SLOT ISOLATION DETAIL ---")
            slot_scores = measure_slot_accuracy(
                model, axis_dataset, tokenizer, device)
            for sid in sorted(slot_scores.keys()):
                s = slot_scores[sid]
                name = SLOT_NAMES.get(sid, f"slot_{sid}")
                status = "OK" if s["isolation"] > 0.1 else "WEAK" if s["isolation"] > 0 else "BAD"
                assigned = "Y" if s["correct_slot"] else "N"
                log(f"  [{status:>4s}] slot {sid:2d} ({name:>12s}): "
                    f"target_sim={s['target_sim']:.3f} other_sim={s['other_sim']:.3f} "
                    f"isolation={s['isolation']:+.3f} assigned={assigned}")

        # Checkpoint
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            save_checkpoint(model, classifiers, optimizer, scaler, config,
                            step + 1, avg_loss, CHECKPOINT_DIR)

        if shutdown_requested:
            break

    prefetch.stop()
    if losses:
        avg_loss = sum(losses[-100:]) / len(losses[-100:])
        save_checkpoint(model, classifiers, optimizer, scaler, config,
                        step + 1 if not shutdown_requested else step,
                        avg_loss, CHECKPOINT_DIR)
    log("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train concept autoencoder V8")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--from-v7", action="store_true",
                        help="Initialize encoder/decoder from V7 checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    with open(".train_pid", "w") as f:
        f.write(str(os.getpid()))

    train(resume_from=args.resume, fresh=args.fresh,
          from_v7=args.from_v7, eval_only=args.eval_only)
