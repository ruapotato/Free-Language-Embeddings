"""
FLM V17 — Bottleneck + 3 Heads + Geo Gate + Programmatic Geometry
================================================================================
Key changes from V15:
  - 3 decoder heads: EN, Paraphrase, Parse (no FR/ES)
  - Bottleneck restored (geometry needs compact target to sculpt)
  - Geo gate: geometry losses activate after EN EM EMA > 0.5, ramp over 5K steps
  - Geo frequency ramp: every step for first 2K after gate, then ramp to 1/5
  - Programmatic geometry data via GeometryDataGenerator (train/test splits)
  - Fresh weights only

V15 showed strong dir_con improvement because geometry operated on the compact
bottleneck. V17 no-bottleneck experiment confirmed mean-pooling is too lossy
for geometry. Restoring bottleneck + adding frequency ramp for intensive
early geometry pressure.

Usage:
    python train_v17.py --fresh            # start from scratch
    python train_v17.py                    # auto-resume
    python train_v17.py --eval-only        # diagnostics only
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
from concept_model import (ConceptConfig, ConceptAutoencoderV17,
                           reconstruction_loss,
                           flat_similarity_matrix,
                           margin_word_order_loss,
                           hard_repulsion_loss, batch_repulsion_loss,
                           analogy_loss, direction_consistency_loss,
                           cluster_separation_loss)
from geometry_data import GeometryDataGenerator, verify_splits

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v17"
LOG_DIR = "logs"

MODEL_CONFIG = dict(
    vocab_size=30522,       # BERT (EN encoder + EN/para/parse decoders)
    fr_vocab_size=1,        # placeholder (no FR head)
    es_vocab_size=1,        # placeholder (no ES head)
    enc_hidden=384,
    enc_layers=6,
    enc_heads=6,
    enc_intermediate=1536,
    num_concepts=32,        # kept but unused by V17
    concept_dim=16,         # kept but unused by V17
    dec_hidden=384,
    dec_layers=6,           # 6 layers per head
    dec_heads=6,
    dec_intermediate=1536,
    max_seq_len=128,
    dropout=0.1,
)

# Training hyperparameters
BATCH_SIZE = 32
PEAK_LR = 2e-4
MIN_LR = 1e-5
WARMUP_STEPS = 2000
TOTAL_STEPS = 600_000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0

# Data sampling weights (no FR/ES)
DATA_WEIGHTS = {
    "para": 0.50,
    "parse": 0.50,
}

# Geometry loss config — GATE RESTORED
GEO_GATE_THRESHOLD = 0.5   # geo losses activate when EN EM EMA > 0.5
GEO_RAMP_STEPS = 5000      # ramp geo weight from 0→1 over 5K steps after gate opens
GEO_WO_WEIGHT = 2.0        # margin word order loss weight
GEO_WO_TARGET = 0.5        # target: wo pairs should have sim < 0.5
GEO_HREPUL_WEIGHT = 1.0    # hard repulsion weight
GEO_HREPUL_TARGET = 0.1    # target: worst pairs should have sim < 0.1
GEO_BREPUL_WEIGHT = 0.3    # batch repulsion weight
GEO_BREPUL_TARGET = 0.3    # target: random pairs should have sim < 0.3
GEO_ANALOGY_WEIGHT = 2.0   # analogy preservation weight
GEO_ANALOGY_TARGET = 0.9   # target: a-b+c should have sim > 0.9 with d
GEO_DIRCON_WEIGHT = 1.5    # direction consistency weight
GEO_DIRCON_TARGET = 0.8    # target: same-attribute directions should have sim > 0.8
GEO_CLUSTER_WEIGHT = 1.5   # cluster separation weight
GEO_CLUSTER_WITHIN = 0.5   # target: same-group sim > 0.5
GEO_CLUSTER_BETWEEN = 0.2  # target: different-group sim < 0.2
GEO_EVERY_STEP_FOR = 2000   # run geo every step for this many steps after gate
GEO_RAMP_TO_EVERY_N = 5     # then ramp to every-N over GEO_FREQ_RAMP_STEPS
GEO_FREQ_RAMP_STEPS = 10000 # steps to transition from every-1 to every-N

# EMA tracking
EXACT_MATCH_EMA_DECAY = 0.99

# Logging
LOG_EVERY = 50
EVAL_EVERY = 500
CHECKPOINT_EVERY = 5000

LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v17.log",
    "metrics": f"{LOG_DIR}/concept_v17_metrics.csv",
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

# Parse test pairs
PARSE_TEST_PAIRS = [
    ("the dog bit the man", "subject : the dog | action : bite | object : the man"),
    ("she runs every morning", "subject : she | action : run | location : every morning"),
    ("he did not enjoy the movie", "subject : he | action : enjoy | negation : true | object : the movie"),
    ("the cat chased the mouse quickly", "subject : the cat | action : chase | object : the mouse | manner : quickly"),
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
                  "word_order_sim,effective_rank90,effective_rank95,"
                  "para_loss,para_token_acc,parse_loss,parse_token_acc,"
                  "geo_scale,wo_loss,hrepul_loss,brepul_loss,analogy_loss,dircon_loss,cluster_loss\n")


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
                f"{g.get('direction_consistency',0):.4f},"
                f"{g.get('word_order_sim',0):.4f},"
                f"{g.get('rank90',0)},{g.get('rank95',0)},"
                f"{m.get('para_loss',0):.6f},{m.get('para_token_acc',0):.4f},"
                f"{m.get('parse_loss',0):.6f},{m.get('parse_token_acc',0):.4f},"
                f"{m.get('geo_scale',0):.4f},{m.get('wo_loss',0):.6f},"
                f"{m.get('hrepul_loss',0):.6f},{m.get('brepul_loss',0):.6f},"
                f"{m.get('analogy_loss',0):.6f},{m.get('dircon_loss',0):.6f},"
                f"{m.get('cluster_loss',0):.6f}\n")


# ---------------------------------------------------------------------------
# Data loading — para + parse only (no FR/ES)
# ---------------------------------------------------------------------------

class MultiSourceDataset:
    """Loads para + parse data sources (no FR/ES)."""

    def __init__(self, en_tok, max_len=128):
        self.en_tok = en_tok
        self.max_len = max_len

        self.para_pairs = []
        self.parse_pairs = []

        self._load()

    def _load(self):
        for src in ["paws.jsonl", "qqp.jsonl", "mrpc.jsonl"]:
            self._load_pairs(f"data/pairs/{src}", self.para_pairs, "PARA", require_label=1)
        path = Path("data/pairs/nli.jsonl")
        if path.exists():
            count = 0
            with open(path) as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        if doc.get("type") == "entailment" and doc.get("label") == 1:
                            a = doc["text_a"].strip()
                            b = doc["text_b"].strip()
                            if len(a) > 5 and len(b) > 5:
                                self.para_pairs.append((a, b))
                                count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
            log(f"  nli (entailment): {count:,} pairs")
        self._load_pairs("data/pairs/semantic_parse.jsonl", self.parse_pairs, "PARSE")
        for pairs in [self.para_pairs, self.parse_pairs]:
            random.shuffle(pairs)
        log(f"  TOTALS: PARA={len(self.para_pairs):,} PARSE={len(self.parse_pairs):,}")

    def _load_pairs(self, path, target, name, require_label=None):
        path = Path(path)
        if not path.exists():
            log(f"  {path}: not found, skipping")
            return
        count = 0
        with open(path) as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    if require_label is not None and doc.get("label") != require_label:
                        continue
                    a = doc.get("text_a", "").strip()
                    b = doc.get("text_b", "").strip()
                    if len(a) > 5 and len(b) > 5:
                        target.append((a, b))
                        count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        log(f"  {path.name}: {count:,} pairs [{name}]")

    def get_batch(self, batch_size, head):
        if head == "para":
            pairs = self.para_pairs
        elif head == "parse":
            pairs = self.parse_pairs
        else:
            raise ValueError(f"Unknown head: {head}")

        tok = self.en_tok  # always EN tokenizer

        indices = [random.randint(0, len(pairs) - 1) for _ in range(batch_size)]
        en_texts = [pairs[i][0] for i in indices]
        tgt_texts = [pairs[i][1] for i in indices]

        en_enc = self.en_tok(en_texts, max_length=self.max_len,
                             padding=True, truncation=True, return_tensors="pt")
        tgt_enc = tok(tgt_texts, max_length=self.max_len,
                      padding=True, truncation=True, return_tensors="pt")
        return en_enc, tgt_enc


class HydraPrefetchBuffer:
    """Pre-tokenizes batches in background."""

    def __init__(self, dataset, device, batch_size=32, buf_size=8):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.q = queue.Queue(maxsize=buf_size)
        self._stop = threading.Event()
        self._thread = None
        heads = list(DATA_WEIGHTS.keys())
        weights = [DATA_WEIGHTS[h] for h in heads]
        total = sum(weights)
        self._heads = heads
        self._cum_weights = []
        cum = 0
        for w in weights:
            cum += w / total
            self._cum_weights.append(cum)

    def _sample_head(self):
        r = random.random()
        for i, cw in enumerate(self._cum_weights):
            if r <= cw:
                return self._heads[i]
        return self._heads[-1]

    def start(self):
        self._thread = threading.Thread(target=self._fill, daemon=True,
                                        name="hydra-prefetch")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _fill(self):
        while not self._stop.is_set():
            try:
                head = self._sample_head()
                batch = self.dataset.get_batch(self.batch_size, head)
                self.q.put((head, batch), timeout=1.0)
            except queue.Full:
                continue

    def get(self):
        return self.q.get()


# ---------------------------------------------------------------------------
# Geometry batch generation (programmatic via GeometryDataGenerator)
# ---------------------------------------------------------------------------

def _encode_concepts(model, tokenizer, texts, device):
    """Encode texts → flat bottleneck vectors (B, num_concepts * concept_dim)."""
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    enc = tokenizer(texts, max_length=64, padding=True, truncation=True, return_tensors="pt")
    concepts = m.encode(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return concepts.view(concepts.shape[0], -1)  # (B, num_concepts * concept_dim)


def get_word_order_batch(geo_gen, tokenizer, model, device, batch_size=16):
    """Get a batch of word-order swap pairs, encode via bottleneck."""
    orig_texts, swap_texts = geo_gen.word_order_batch(batch_size)
    return _encode_concepts(model, tokenizer, orig_texts, device), \
           _encode_concepts(model, tokenizer, swap_texts, device)


def get_analogy_batch(geo_gen, tokenizer, model, device, batch_size=6):
    """Get a batch of analogy quads, encode via bottleneck."""
    a_texts, b_texts, c_texts, d_texts = geo_gen.analogy_batch(batch_size)
    return (_encode_concepts(model, tokenizer, a_texts, device),
            _encode_concepts(model, tokenizer, b_texts, device),
            _encode_concepts(model, tokenizer, c_texts, device),
            _encode_concepts(model, tokenizer, d_texts, device))


def get_direction_batch(geo_gen, model, tokenizer, device):
    """Encode direction pairs via bottleneck."""
    attr_pairs = geo_gen.direction_batch(n_pairs_per_attr=3)
    direction_groups = []
    for attr_name, pairs in attr_pairs.items():
        a_texts = [p[0] for p in pairs]
        b_texts = [p[1] for p in pairs]
        flat_a = _encode_concepts(model, tokenizer, a_texts, device)
        flat_b = _encode_concepts(model, tokenizer, b_texts, device)
        directions = F.normalize(flat_a - flat_b, p=2, dim=-1)
        direction_groups.append(directions)
    return direction_groups


def get_cluster_batch(geo_gen, model, tokenizer, device, n_groups=3, n_per_group=3):
    """Encode cluster sentences via bottleneck."""
    groups = geo_gen.cluster_batch(n_groups=n_groups, n_per_group=n_per_group)
    group_concepts = []
    for name, sents in groups.items():
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        enc = tokenizer(sents, max_length=64, padding=True, truncation=True, return_tensors="pt").to(device)
        concepts = m.encode(enc["input_ids"], enc["attention_mask"])  # (N, num_concepts, concept_dim)
        group_concepts.append(concepts)

    within_concepts = group_concepts  # list of (N, num_concepts, concept_dim)
    between_pairs = []
    for i in range(len(group_concepts)):
        for j in range(i + 1, len(group_concepts)):
            idx_a = random.randint(0, group_concepts[i].shape[0] - 1)
            idx_b = random.randint(0, group_concepts[j].shape[0] - 1)
            between_pairs.append((group_concepts[i][idx_a], group_concepts[j][idx_b]))

    return within_concepts, between_pairs


# ---------------------------------------------------------------------------
# Evaluation
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
        encoder_output = m.encode(enc["input_ids"], enc["attention_mask"])
        logits = m.decode_en(encoder_output, seq_len=enc["input_ids"].shape[1],
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


@torch.no_grad()
def evaluate_translation(model, en_tok, tgt_tok, test_pairs, decode_fn_name, device="cuda"):
    """Generic eval for any decoder head (para, parse)."""
    model.eval()
    m = _unwrap(model)
    decode_fn = getattr(m, decode_fn_name)

    results = []
    total_correct = 0
    total_tokens = 0

    for en_text, ref_text in test_pairs:
        en_enc = en_tok([en_text], max_length=128, padding=True,
                        truncation=True, return_tensors="pt").to(device)
        ref_enc = tgt_tok([ref_text], max_length=128, padding=True,
                          truncation=True, return_tensors="pt").to(device)

        encoder_output = m.encode(en_enc["input_ids"], en_enc["attention_mask"])
        logits = decode_fn(encoder_output, seq_len=ref_enc["input_ids"].shape[1],
                           attention_mask=ref_enc["attention_mask"])
        predicted = logits.argmax(dim=-1)
        mask = ref_enc["attention_mask"][0].bool()
        tgt = ref_enc["input_ids"][0][mask]
        pred = predicted[0][mask]
        correct = (tgt == pred).sum().item()
        total = mask.sum().item()
        total_correct += correct
        total_tokens += total
        decoded = tgt_tok.decode(pred, skip_special_tokens=True)
        results.append((en_text, ref_text, decoded, correct / max(total, 1)))

    token_acc = total_correct / max(total_tokens, 1)
    model.train()
    return results, token_acc


# ---------------------------------------------------------------------------
# Geometry probing (uses TEST split for genuine generalization testing)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_flat(model, tokenizer, texts, device):
    """Encode texts to flat bottleneck vectors. Returns (normalized_flat, concepts)."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64,
                    return_tensors="pt").to(device)
    m = _unwrap(model)
    concepts = m.encode(enc["input_ids"], enc["attention_mask"])  # (B, num_concepts, concept_dim)
    flat = concepts.view(concepts.shape[0], -1)  # (B, num_concepts * concept_dim)
    return F.normalize(flat, p=2, dim=-1), concepts


@torch.no_grad()
def probe_geometry(model, tokenizer, device="cuda"):
    """Probe geometry using TEST split — measures genuine generalization."""
    model.eval()
    geo = {}

    # Use test split with fixed seed for reproducible eval
    test_gen = GeometryDataGenerator(split="test", seed=42)

    # Analogies (from test vocab)
    analogy_scores = []
    for _ in range(5):  # 5 batches of 6 = 30 analogy tests
        a_texts, b_texts, c_texts, d_texts = test_gen.analogy_batch(6)
        for i in range(len(a_texts)):
            va, _ = _encode_flat(model, tokenizer, [a_texts[i]], device)
            vb, _ = _encode_flat(model, tokenizer, [b_texts[i]], device)
            vc, _ = _encode_flat(model, tokenizer, [c_texts[i]], device)
            vd, _ = _encode_flat(model, tokenizer, [d_texts[i]], device)
            predicted = F.normalize(va - vb + vc, p=2, dim=-1)
            sim = F.cosine_similarity(predicted, vd).item()
            analogy_scores.append(sim)
    geo["analogy_avg"] = float(np.mean(analogy_scores))

    # Clustering gap (from test vocab)
    cluster_groups = test_gen.cluster_batch(n_groups=6, n_per_group=5)
    group_concepts = {}
    for name, sents in cluster_groups.items():
        _, concepts_3d = _encode_flat(model, tokenizer, sents, device)  # (N, 1, D)
        group_concepts[name] = concepts_3d

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

    # Direction consistency (from test vocab)
    dir_scores = []
    attr_pairs = test_gen.direction_batch(n_pairs_per_attr=5)
    for attr, pairs in attr_pairs.items():
        deltas = []
        for pos, neg in pairs:
            vp, _ = _encode_flat(model, tokenizer, [pos], device)
            vn, _ = _encode_flat(model, tokenizer, [neg], device)
            deltas.append(F.normalize(vp - vn, p=2, dim=-1))
        cons = []
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                cons.append(F.cosine_similarity(deltas[i], deltas[j]).item())
        if cons:
            dir_scores.append(float(np.mean(cons)))
    geo["direction_consistency"] = float(np.mean(dir_scores)) if dir_scores else 0.0

    # Word order sensitivity (from test vocab)
    wo_origs, wo_swaps = test_gen.word_order_batch(16)
    wo_sims = []
    for i in range(len(wo_origs)):
        va, _ = _encode_flat(model, tokenizer, [wo_origs[i]], device)
        vb, _ = _encode_flat(model, tokenizer, [wo_swaps[i]], device)
        wo_sims.append(F.cosine_similarity(va, vb).item())
    geo["word_order_sim"] = float(np.mean(wo_sims))

    # Effective rank (diverse test sentences)
    all_sents = test_gen.diverse_sentences(batch_size=60)
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
                    exact_match_ema, geo_gate_step, checkpoint_dir):
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
        "geo_gate_step": geo_gate_step,
        "version": "v17",
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
    log(f"Loading V17 checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoderV17(config).to(device)
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
    geo_gate_step = ckpt.get("geo_gate_step", None)
    total, _ = model.count_parameters()
    log(f"Resumed V17: {total:,} params | step {step} | em_ema={exact_match_ema:.3f} | geo_gate_step={geo_gate_step}")
    return model, optimizer, scaler, config, step, exact_match_ema, geo_gate_step


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False, eval_only=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V17 — NO BOTTLENECK + GEO GATE + PROGRAMMATIC GEOMETRY")
    log(f"  Bottleneck: {config.num_concepts}×{config.concept_dim} = {config.num_concepts * config.concept_dim}d")
    log("  Heads: EN recon | EN para | Semantic parse (3 heads, 6L each)")
    log("  Geometry: programmatic data, recon-gated (EM EMA > 0.5)")
    log("=" * 70)

    # Verify geometry data splits are clean
    verify_splits()
    log("Geometry vocab splits verified: zero train/test overlap")

    from transformers import AutoTokenizer

    en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    en_tok_eval  = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Separate instance for geo loss functions — avoids "Already borrowed" from
    # concurrent fast-tokenizer use between the prefetch thread and main loop.
    geo_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    log(f"EN tokenizer: vocab={en_tokenizer.vocab_size}")

    model_config = dict(MODEL_CONFIG)
    model_config["vocab_size"] = en_tokenizer.vocab_size

    exact_match_ema = 0.0
    geo_gate_step = None  # step when geo gate opened (None = not yet)
    geo_every_n = 0       # current geo frequency (0 = off)

    if resume_from is None and not fresh:
        latest = Path(CHECKPOINT_DIR) / "latest.pt"
        if latest.exists():
            resume_from = str(latest)

    if resume_from:
        model, optimizer, scaler, config, start_step, exact_match_ema, geo_gate_step = \
            load_checkpoint(resume_from, device)
    else:
        log("Starting fresh training...")
        config = ConceptConfig(**model_config)
        model = ConceptAutoencoderV17(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=PEAK_LR, betas=BETAS,
            weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0
        total, _ = model.count_parameters()
        log(f"Model: {total:,} params ({total/1e6:.1f}M)")

    if eval_only:
        log("\n--- EN RECONSTRUCTION ---")
        results, acc, em = evaluate_reconstruction(model, en_tok_eval, device)
        log(f"  token_acc={acc:.3f} exact_match={em:.3f}")
        for orig, decoded, tacc, exact in results:
            status = "OK" if exact else "DIFF"
            log(f"  [{status}] ({tacc:.0%}) {orig}")
            if not exact:
                log(f"       -> {decoded}")

        log("\n--- PARSE ---")
        parse_results, parse_acc = evaluate_translation(
            model, en_tok_eval, en_tok_eval, PARSE_TEST_PAIRS, "decode_parse", device)
        log(f"  Parse token_acc={parse_acc:.3f}")
        for en, ref, pred, tacc in parse_results:
            log(f"  [{tacc:.0%}] {en} -> {pred}")
            log(f"       ref: {ref}")

        log("\n--- GEOMETRY (TEST SPLIT) ---")
        geo = probe_geometry(model, en_tok_eval, device)
        for k, v in geo.items():
            log(f"  {k}: {v}")
        return

    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()

    log("Loading data...")
    dataset = MultiSourceDataset(en_tokenizer, max_len=config.max_seq_len)

    # Create geometry data generators
    train_geo_gen = GeometryDataGenerator(split="train")
    log(f"Geometry generator: train split (programmatic, 18K+ word order combos)")

    total, _ = _unwrap(model).count_parameters()
    log(f"\nTraining plan (V17 — No Bottleneck + Geo Gate + Programmatic Geometry):")
    log(f"  Model: {total:,} params ({total/1e6:.1f}M)")
    log(f"  Bottleneck: {config.num_concepts}×{config.concept_dim} = {config.num_concepts * config.concept_dim}d")
    log(f"  Heads: EN({config.dec_layers}L) PARA({config.dec_layers}L) PARSE({config.dec_layers}L)")
    log(f"  Geometry: wo={GEO_WO_WEIGHT} hrepul={GEO_HREPUL_WEIGHT} "
        f"brepul={GEO_BREPUL_WEIGHT} analogy={GEO_ANALOGY_WEIGHT} "
        f"dircon={GEO_DIRCON_WEIGHT} cluster={GEO_CLUSTER_WEIGHT}")
    log(f"  Geo: GATED (EM EMA > {GEO_GATE_THRESHOLD}), ramp 0→1 over {GEO_RAMP_STEPS} steps")
    log(f"  Geo freq: every step for {GEO_EVERY_STEP_FOR} steps, then ramp to 1/{GEO_RAMP_TO_EVERY_N} over {GEO_FREQ_RAMP_STEPS} steps")
    log(f"  Batch: {BATCH_SIZE}")
    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")
    log(f"  Sampling: " + " ".join(f"{h}={w:.0%}" for h, w in DATA_WEIGHTS.items()))
    log("-" * 70)

    prefetch = HydraPrefetchBuffer(dataset, device, batch_size=BATCH_SIZE)
    prefetch.start()
    log("Prefetch buffer started")

    PAD_ID = en_tokenizer.pad_token_id or 0

    DECODE_FNS = {
        "para": "decode_para",
        "parse": "decode_parse",
    }

    loss_trackers = {h: [] for h in ["en", "para", "parse"]}
    geo_trackers = {"wo": [], "hrepul": [], "brepul": [], "analogy": [], "dircon": [], "cluster": []}
    head_counts = {h: 0 for h in DATA_WEIGHTS}

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

        # Get batch
        head, (en_enc, tgt_enc) = prefetch.get()
        head_counts[head] += 1

        en_ids = en_enc["input_ids"].to(device, non_blocking=True)
        en_mask = en_enc["attention_mask"].to(device, non_blocking=True)
        tgt_ids = tgt_enc["input_ids"].to(device, non_blocking=True)
        tgt_mask = tgt_enc["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            m = _unwrap(model) if not hasattr(model, 'encode') else model

            # Encode → bottleneck concepts (B, num_concepts, concept_dim)
            concepts = m.encode(en_ids, en_mask)

            # EN reconstruction (always)
            en_logits = m.decode_en(concepts, en_ids.shape[1], en_mask)
            r_loss = reconstruction_loss(en_logits, en_ids)

            # Secondary head (para or parse)
            decode_fn = getattr(m, DECODE_FNS[head])
            head_logits = decode_fn(concepts, tgt_ids.shape[1], tgt_mask)
            h_loss = F.cross_entropy(
                head_logits.reshape(-1, head_logits.shape[-1]),
                tgt_ids.reshape(-1),
                ignore_index=PAD_ID)

            total_loss = r_loss + h_loss

            # --- Geometry losses (GATED — activate after EM EMA > threshold) ---
            # Check geo gate
            if geo_gate_step is None and exact_match_ema >= GEO_GATE_THRESHOLD:
                geo_gate_step = step
                log(f"  *** GEO GATE OPENED at step {step} (em_ema={exact_match_ema:.3f}) ***")

            if geo_gate_step is not None:
                steps_since_gate = step - geo_gate_step
                geo_scale = min(1.0, steps_since_gate / max(1, GEO_RAMP_STEPS))
            else:
                geo_scale = 0.0

            # Geo frequency: every step for first 2K steps, then ramp to every-5
            if geo_scale > 0 and geo_gate_step is not None:
                steps_since_gate = step - geo_gate_step
                if steps_since_gate <= GEO_EVERY_STEP_FOR:
                    geo_every_n = 1
                elif steps_since_gate <= GEO_EVERY_STEP_FOR + GEO_FREQ_RAMP_STEPS:
                    # Linear ramp from 1 to GEO_RAMP_TO_EVERY_N
                    progress = (steps_since_gate - GEO_EVERY_STEP_FOR) / GEO_FREQ_RAMP_STEPS
                    geo_every_n = max(1, int(1 + progress * (GEO_RAMP_TO_EVERY_N - 1)))
                else:
                    geo_every_n = GEO_RAMP_TO_EVERY_N
                run_geo = (step % geo_every_n == 0)
            else:
                run_geo = False

            wo_val = 0.0
            hr_val = 0.0
            br_val = 0.0
            an_val = 0.0
            dc_val = 0.0
            cl_val = 0.0

            if run_geo:
                # Word-order loss: programmatic swap pairs (bottleneck)
                orig_pooled, swap_pooled = get_word_order_batch(
                    train_geo_gen, geo_tokenizer, m, device, batch_size=16)
                # (B, D) — view(B, -1) is a no-op on (B, D)
                wo_loss, _ = margin_word_order_loss(orig_pooled, swap_pooled,
                                                     target_sim=GEO_WO_TARGET)
                total_loss = total_loss + geo_scale * GEO_WO_WEIGHT * wo_loss
                wo_val = wo_loss.item()

                # Hard repulsion on batch concepts (flattened bottleneck)
                batch_pooled = concepts.view(concepts.shape[0], -1)  # (B, num_concepts * concept_dim)
                hr_loss, _ = hard_repulsion_loss(batch_pooled,
                                                  target_sim=GEO_HREPUL_TARGET,
                                                  top_k=8)
                total_loss = total_loss + geo_scale * GEO_HREPUL_WEIGHT * hr_loss
                hr_val = hr_loss.item()

                # Batch repulsion — same pooled vectors
                br_loss, _ = batch_repulsion_loss(batch_pooled,
                                                   target_sim=GEO_BREPUL_TARGET)
                total_loss = total_loss + geo_scale * GEO_BREPUL_WEIGHT * br_loss
                br_val = br_loss.item()

                # Analogy preservation: programmatic quads (bottleneck)
                c_a, c_b, c_c, c_d = get_analogy_batch(
                    train_geo_gen, geo_tokenizer, m, device, batch_size=6)
                # (B, D) — pass directly
                an_loss, _ = analogy_loss(c_a, c_b, c_c, c_d,
                                          target_sim=GEO_ANALOGY_TARGET)
                total_loss = total_loss + geo_scale * GEO_ANALOGY_WEIGHT * an_loss
                an_val = an_loss.item()

                # Direction consistency: programmatic pairs (bottleneck)
                dir_groups = get_direction_batch(train_geo_gen, m, geo_tokenizer, device)
                dc_loss, _ = direction_consistency_loss(dir_groups,
                                                        target_sim=GEO_DIRCON_TARGET)
                total_loss = total_loss + geo_scale * GEO_DIRCON_WEIGHT * dc_loss
                dc_val = dc_loss.item()

                # Cluster separation: programmatic groups (bottleneck)
                within_c, between_c = get_cluster_batch(
                    train_geo_gen, m, geo_tokenizer, device, n_groups=3, n_per_group=3)
                cl_loss, _, _ = cluster_separation_loss(
                    within_c, between_c,
                    within_target=GEO_CLUSTER_WITHIN,
                    between_target=GEO_CLUSTER_BETWEEN)
                total_loss = total_loss + geo_scale * GEO_CLUSTER_WEIGHT * cl_loss
                cl_val = cl_loss.item()

        r_val = r_loss.item()
        h_val = h_loss.item()
        loss_trackers["en"].append(r_val)
        loss_trackers[head].append(h_val)
        geo_trackers["wo"].append(wo_val)
        geo_trackers["hrepul"].append(hr_val)
        geo_trackers["brepul"].append(br_val)
        geo_trackers["analogy"].append(an_val)
        geo_trackers["dircon"].append(dc_val)
        geo_trackers["cluster"].append(cl_val)

        if torch.isnan(total_loss):
            log(f"NaN loss at step {step}, skipping")
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # --- Logging ---
        if (step + 1) % LOG_EVERY == 0:
            avg_en = np.mean(loss_trackers["en"][-100:]) if loss_trackers["en"] else 0
            pct = (step + 1) / TOTAL_STEPS * 100
            head_losses = []
            for h in ["para", "parse"]:
                if loss_trackers[h]:
                    head_losses.append(f"{h}={np.mean(loss_trackers[h][-50:]):.3f}")
            hl_str = " ".join(head_losses) if head_losses else "warming up"
            geo_str = f"geo={geo_scale:.2f}"
            if geo_scale > 0:
                avg_wo = np.mean(geo_trackers["wo"][-50:])
                avg_hr = np.mean(geo_trackers["hrepul"][-50:])
                avg_an = np.mean(geo_trackers["analogy"][-50:])
                avg_dc = np.mean(geo_trackers["dircon"][-50:])
                avg_cl = np.mean(geo_trackers["cluster"][-50:])
                geo_str += f" wo={avg_wo:.3f} hr={avg_hr:.3f} an={avg_an:.3f} dc={avg_dc:.3f} cl={avg_cl:.3f}"
            log(f"step {step+1:>7d} [V17+GEO] | en={avg_en:.4f} {hl_str} | "
                f"em_ema={exact_match_ema:.3f} | {geo_str} | "
                f"lr {current_lr:.2e} | {pct:.1f}%")

        # --- Eval ---
        if (step + 1) % EVAL_EVERY == 0:
            results, acc, em = evaluate_reconstruction(
                model, en_tok_eval, device)
            exact_match_ema = (EXACT_MATCH_EMA_DECAY * exact_match_ema +
                               (1 - EXACT_MATCH_EMA_DECAY) * em)
            log(f"  EN EVAL: token_acc={acc:.3f} exact_match={em:.3f} "
                f"em_ema={exact_match_ema:.3f}")
            for orig, decoded, tacc, exact in results:
                status = "OK" if exact else "DIFF"
                log(f"    [{status}] ({tacc:.0%}) {orig}")
                if not exact:
                    log(f"           -> {decoded}")

            parse_results, parse_acc = evaluate_translation(
                model, en_tok_eval, en_tok_eval, PARSE_TEST_PAIRS, "decode_parse", device)
            log(f"  PARSE EVAL: token_acc={parse_acc:.3f}")
            for en, ref, pred, tacc in parse_results:
                log(f"    [{tacc:.0%}] {en} -> {pred}")
                log(f"           ref: {ref}")

            geo = probe_geometry(model, en_tok_eval, device)
            log(f"  GEOMETRY (TEST): analogy={geo['analogy_avg']:.3f} "
                f"cluster_gap={geo['clustering_gap']:+.4f} "
                f"dir_con={geo['direction_consistency']:.3f} "
                f"wo_sim={geo['word_order_sim']:.3f} "
                f"rank90={geo['rank90']} rank95={geo['rank95']}")

            total_heads = sum(head_counts.values()) or 1
            dist = " ".join(f"{h}={head_counts[h]/total_heads:.0%}" for h in DATA_WEIGHTS)
            geo_freq = geo_every_n if geo_scale > 0 else 0
            log(f"  HEAD DIST: {dist} | geo_scale={geo_scale:.2f} | geo_freq=1/{geo_freq} | geo_gate_step={geo_gate_step}")

            elapsed = time.time() - start_time
            avg_en = np.mean(loss_trackers["en"][-100:]) if loss_trackers["en"] else 0
            avg_para = np.mean(loss_trackers["para"][-50:]) if loss_trackers["para"] else 0
            avg_parse = np.mean(loss_trackers["parse"][-50:]) if loss_trackers["parse"] else 0
            log_metrics(step + 1, {
                "recon_loss": avg_en, "token_acc": acc, "exact_match": em,
                "em_ema": exact_match_ema, "lr": current_lr,
                "elapsed_hours": elapsed / 3600, "geo": geo,
                "para_loss": avg_para, "para_token_acc": 0,
                "parse_loss": avg_parse, "parse_token_acc": parse_acc,
                "geo_scale": geo_scale,
                "wo_loss": np.mean(geo_trackers["wo"][-50:]) if geo_trackers["wo"] else 0,
                "hrepul_loss": np.mean(geo_trackers["hrepul"][-50:]) if geo_trackers["hrepul"] else 0,
                "brepul_loss": np.mean(geo_trackers["brepul"][-50:]) if geo_trackers["brepul"] else 0,
                "analogy_loss": np.mean(geo_trackers["analogy"][-50:]) if geo_trackers["analogy"] else 0,
                "dircon_loss": np.mean(geo_trackers["dircon"][-50:]) if geo_trackers["dircon"] else 0,
                "cluster_loss": np.mean(geo_trackers["cluster"][-50:]) if geo_trackers["cluster"] else 0,
            })

        # --- Checkpoint ---
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = np.mean(loss_trackers["en"][-100:]) if loss_trackers["en"] else 0
            save_checkpoint(model, optimizer, scaler, config, step + 1,
                            avg_loss, exact_match_ema, geo_gate_step,
                            CHECKPOINT_DIR)

        if shutdown_requested:
            break

    prefetch.stop()
    if loss_trackers["en"]:
        avg_loss = np.mean(loss_trackers["en"][-100:])
        save_checkpoint(model, optimizer, scaler, config,
                        step + 1 if not shutdown_requested else step,
                        avg_loss, exact_match_ema, geo_gate_step,
                        CHECKPOINT_DIR)
    log("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train concept autoencoder V17 (No Bottleneck + Geo Gate + Programmatic Geo)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    train(resume_from=args.resume, fresh=args.fresh, eval_only=args.eval_only)
