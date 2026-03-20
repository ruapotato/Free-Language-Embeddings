"""
FLM V4 — Train Semantic Encoder (Model 1)
==========================================
Contrastive training on paraphrase pairs. No next-token prediction.

Usage:
    python train_encoder.py                  # fresh training
    python train_encoder.py --resume         # resume from latest checkpoint
    python train_encoder.py --eval-only      # run diagnostics only
"""

import os
import sys
import json
import time
import math
import signal
import random
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from encoder_model import (EncoderConfig, SemanticEncoder,
                          adaptive_contrastive_loss, graded_similarity_loss)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/encoder_v4"
LOG_DIR = "logs"

ENCODER_CONFIG = dict(
    vocab_size=32000,
    hidden_size=384,
    num_layers=8,
    num_heads=6,
    intermediate_size=1536,
    max_seq_len=128,
    dropout=0.1,
    output_dim=512,
)

# Training hyperparameters
BATCH_SIZE = 192
PEAK_LR = 3e-4
WARMUP_STEPS = 1000
TOTAL_STEPS = 100_000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0
NEG_BATCH_SIZE = 192       # negative pairs per step
POS_TARGET = 0.9           # pull positive pairs above this
NEG_TARGET = 0.3           # push negative pairs below this
NEG_WEIGHT = 2.0           # weight negative loss vs positive loss
GRADED_WEIGHT = 0.5        # weight for graded similarity regression loss
GRADED_BATCH_SIZE = 96     # graded pairs per step

# Logging
LOG_EVERY = 50
EVAL_EVERY = 500
SAMPLE_EVERY = 1000
CHECKPOINT_EVERY = 2000
UMAP_EVERY = 500

LOG_PATHS = {
    "log": f"{LOG_DIR}/encoder_v4.log",
    "metrics": f"{LOG_DIR}/encoder_v4_metrics.csv",
}

# Diagnostic pairs — printed at start and periodically
DIAGNOSTIC_PAIRS = [
    # Paraphrases — should be >0.95
    ("the massive cat stepped on the rug",
     "there was a rug that a massive cat stepped on"),
    ("the king died",
     "the monarch passed away"),
    ("she runs every morning",
     "every morning she goes for a run"),
    # Cross-lingual — should be >0.90
    ("the cat sat on the mat",
     "le chat était assis sur le tapis"),
    # Different meaning — should be <0.3
    ("the cat sat on the mat",
     "the stock market crashed today"),
    # Hard negative — should be <0.4
    ("the dog bit the man",
     "the man bit the dog"),
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


def log_metrics(step, loss, para_sim, nonpara_sim, sep_ratio, crosslingual_sim,
                hard_neg_sim, lr, elapsed_hours):
    metrics_file = LOG_PATHS.get("metrics")
    if not metrics_file:
        return
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    write_header = not os.path.exists(metrics_file)
    with open(metrics_file, "a") as f:
        if write_header:
            f.write("timestamp,step,loss,para_sim,nonpara_sim,sep_ratio,"
                    "crosslingual_sim,hard_neg_sim,lr,elapsed_hours\n")
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts},{step},{loss:.6f},{para_sim:.4f},{nonpara_sim:.4f},"
                f"{sep_ratio:.4f},{crosslingual_sim:.4f},{hard_neg_sim:.4f},"
                f"{lr:.6e},{elapsed_hours:.4f}\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class PairDataset:
    """Loads paraphrase pairs from data/pairs/*.jsonl.

    Each line: {"text_a": "...", "text_b": "...", "label": 1/0, "source": "..."}
    Loads both positive (label=1) and negative (label=0) pairs.
    Negatives include hard negatives from PAWS and non-paraphrases from QQP/MRPC.
    """

    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pos_pairs = []
        self.neg_pairs = []
        self.graded_pairs = []  # (text_a, text_b, sim_score)
        self._load_pairs()
        random.shuffle(self.pos_pairs)
        random.shuffle(self.neg_pairs)
        random.shuffle(self.graded_pairs)
        self.pos_idx = 0
        self.neg_idx = 0
        self.graded_idx = 0

    def _load_pairs(self):
        data_dir = Path("data/pairs")
        if not data_dir.exists():
            log(f"WARNING: {data_dir} not found")
            return

        # Load per-source so we can balance
        pos_by_source = {}
        neg_by_source = {}

        graded_pairs_raw = []

        for path in sorted(data_dir.glob("*.jsonl")):
            if path.name.startswith("eval_"):
                continue
            source = path.stem
            pos_by_source[source] = []
            neg_by_source[source] = []
            graded_count = 0
            with open(path) as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        label = doc.get("label", 1)
                        pair_type = doc.get("type", "")

                        # Graded pairs (STS-B) go to separate pool
                        # Only keep clearly similar (>0.7) or clearly different (<0.3)
                        # The middle zone (0.3-0.7) fights the margin-based losses
                        if "sim_score" in doc:
                            score = doc["sim_score"]
                            if score > 0.7 or score < 0.3:
                                graded_pairs_raw.append(
                                    (doc["text_a"], doc["text_b"], score))
                                graded_count += 1
                            continue

                        pair = (doc["text_a"], doc["text_b"])
                        if label == 1:
                            pos_by_source[source].append(pair)
                        else:
                            neg_by_source[source].append(pair)
                    except (json.JSONDecodeError, KeyError):
                        continue
            pos_n = len(pos_by_source[source])
            neg_n = len(neg_by_source[source])
            extra = f" + {graded_count:,} graded" if graded_count else ""
            log(f"  {path.name}: {pos_n:,} pos + {neg_n:,} neg{extra}")

        # Balance positives: cap each source, oversample small sources
        # Target: roughly equal representation per source
        pos_counts = {s: len(p) for s, p in pos_by_source.items() if p}
        if pos_counts:
            median_count = sorted(pos_counts.values())[len(pos_counts) // 2]
            target_per_source = min(median_count * 2, 100_000)
            for source, pairs in pos_by_source.items():
                if not pairs:
                    continue
                if len(pairs) > target_per_source:
                    # Downsample large sources
                    random.shuffle(pairs)
                    sampled = pairs[:target_per_source]
                elif len(pairs) < target_per_source // 2:
                    # Oversample small sources
                    repeats = (target_per_source // len(pairs)) + 1
                    sampled = (pairs * repeats)[:target_per_source]
                else:
                    sampled = pairs
                self.pos_pairs.extend(sampled)
                log(f"    pos balanced: {source} {len(pairs):,} → {len(sampled):,}")

        # Balance negatives: oversample PAWS hard negatives
        neg_counts = {s: len(p) for s, p in neg_by_source.items() if p}
        if neg_counts:
            target_neg = max(neg_counts.values())
            for source, pairs in neg_by_source.items():
                if not pairs:
                    continue
                if source == "paws":
                    # PAWS hard negatives are the most valuable — oversample heavily
                    repeats = (target_neg // len(pairs)) + 1
                    sampled = (pairs * repeats)[:target_neg]
                else:
                    sampled = pairs
                self.neg_pairs.extend(sampled)
                log(f"    neg balanced: {source} {len(pairs):,} → {len(sampled):,}")

        # Graded pairs: oversample STS-B since it's small but valuable
        if graded_pairs_raw:
            target_graded = max(50_000, len(graded_pairs_raw))
            repeats = (target_graded // len(graded_pairs_raw)) + 1
            self.graded_pairs = (graded_pairs_raw * repeats)[:target_graded]
            log(f"    graded: {len(graded_pairs_raw):,} → {len(self.graded_pairs):,}")

        log(f"  Balanced total: {len(self.pos_pairs):,} pos, {len(self.neg_pairs):,} neg, "
            f"{len(self.graded_pairs):,} graded")

    def get_batch(self, batch_size):
        """Get a batch of tokenized positive pairs."""
        texts_a = []
        texts_b = []

        for _ in range(batch_size):
            if self.pos_idx >= len(self.pos_pairs):
                random.shuffle(self.pos_pairs)
                self.pos_idx = 0
            a, b = self.pos_pairs[self.pos_idx]
            texts_a.append(a)
            texts_b.append(b)
            self.pos_idx += 1

        enc_a = self.tokenizer(texts_a, max_length=self.max_len,
                               padding=True, truncation=True,
                               return_tensors="pt")
        enc_b = self.tokenizer(texts_b, max_length=self.max_len,
                               padding=True, truncation=True,
                               return_tensors="pt")

        return enc_a, enc_b

    def get_neg_batch(self, batch_size):
        """Get a batch of tokenized negative pairs (hard negatives)."""
        if not self.neg_pairs:
            return None, None
        texts_a = []
        texts_b = []

        for _ in range(batch_size):
            if self.neg_idx >= len(self.neg_pairs):
                random.shuffle(self.neg_pairs)
                self.neg_idx = 0
            a, b = self.neg_pairs[self.neg_idx]
            texts_a.append(a)
            texts_b.append(b)
            self.neg_idx += 1

        enc_a = self.tokenizer(texts_a, max_length=self.max_len,
                               padding=True, truncation=True,
                               return_tensors="pt")
        enc_b = self.tokenizer(texts_b, max_length=self.max_len,
                               padding=True, truncation=True,
                               return_tensors="pt")

        return enc_a, enc_b

    def get_graded_batch(self, batch_size):
        """Get a batch of graded similarity pairs with target scores."""
        if not self.graded_pairs:
            return None, None, None
        texts_a = []
        texts_b = []
        scores = []

        for _ in range(batch_size):
            if self.graded_idx >= len(self.graded_pairs):
                random.shuffle(self.graded_pairs)
                self.graded_idx = 0
            a, b, score = self.graded_pairs[self.graded_idx]
            texts_a.append(a)
            texts_b.append(b)
            scores.append(score)
            self.graded_idx += 1

        enc_a = self.tokenizer(texts_a, max_length=self.max_len,
                               padding=True, truncation=True,
                               return_tensors="pt")
        enc_b = self.tokenizer(texts_b, max_length=self.max_len,
                               padding=True, truncation=True,
                               return_tensors="pt")

        return enc_a, enc_b, torch.tensor(scores, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_diagnostics(model, tokenizer, device="cuda"):
    """Run diagnostic pairs and return similarity scores."""
    model.eval()
    results = []
    for text_a, text_b in DIAGNOSTIC_PAIRS:
        enc_a = tokenizer(text_a, max_length=128, padding=True,
                          truncation=True, return_tensors="pt").to(device)
        enc_b = tokenizer(text_b, max_length=128, padding=True,
                          truncation=True, return_tensors="pt").to(device)
        vec_a = model(enc_a["input_ids"], enc_a["attention_mask"])
        vec_b = model(enc_b["input_ids"], enc_b["attention_mask"])
        sim = F.cosine_similarity(vec_a, vec_b).item()
        results.append((text_a[:50], text_b[:50], sim))
    model.train()
    return results


@torch.no_grad()
def eval_geometry(model, eval_pairs, tokenizer, device="cuda", max_pairs=500):
    """Evaluate geometry quality metrics on held-out pairs.

    Returns dict with: para_sim, nonpara_sim, sep_ratio,
                        crosslingual_sim, hard_neg_sim
    """
    model.eval()

    para_sims = []
    nonpara_sims = []
    crosslingual_sims = []
    crosslingual_neg_sims = []
    hard_neg_sims = []

    for source, pairs in eval_pairs.items():
        sample = pairs[:max_pairs]
        for text_a, text_b, label, pair_type in sample:
            enc_a = tokenizer(text_a, max_length=128, padding=True,
                              truncation=True, return_tensors="pt").to(device)
            enc_b = tokenizer(text_b, max_length=128, padding=True,
                              truncation=True, return_tensors="pt").to(device)
            vec_a = model(enc_a["input_ids"], enc_a["attention_mask"])
            vec_b = model(enc_b["input_ids"], enc_b["attention_mask"])
            sim = F.cosine_similarity(vec_a, vec_b).item()

            if pair_type == "crosslingual":
                crosslingual_sims.append(sim)
            elif pair_type == "crosslingual_neg":
                crosslingual_neg_sims.append(sim)
            elif pair_type == "hard_negative":
                hard_neg_sims.append(sim)
            elif label == 1:
                para_sims.append(sim)
            else:
                nonpara_sims.append(sim)

    def safe_mean(lst, default=0.0):
        return sum(lst) / len(lst) if lst else default

    para_sim = safe_mean(para_sims)
    nonpara_sim = safe_mean(nonpara_sims, 0.5)
    sep_ratio = para_sim / max(nonpara_sim, 0.01)

    model.train()
    return {
        "para_sim": para_sim,
        "nonpara_sim": nonpara_sim,
        "sep_ratio": sep_ratio,
        "crosslingual_sim": safe_mean(crosslingual_sims),
        "crosslingual_neg_sim": safe_mean(crosslingual_neg_sims, 0.5),
        "hard_neg_sim": safe_mean(hard_neg_sims),
    }


@torch.no_grad()
def eval_concept_geometry(model, tokenizer, device="cuda"):
    """Evaluate concept space geometry quality.

    Tests clustering, directional consistency, and analogies.
    Returns summary metrics that track concept quality over training.
    """
    model.eval()

    def encode(text):
        enc = tokenizer(text, max_length=128, padding=True,
                        truncation=True, return_tensors="pt").to(device)
        vec = model(enc["input_ids"], enc["attention_mask"])
        return vec.cpu().numpy()[0]

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    # 1. Clustering: within-group vs between-group similarity
    groups = {
        "animals": ["a cat", "a dog", "a bird", "a fish"],
        "vehicles": ["a car", "a truck", "a bus", "a train"],
        "emotions": ["I am happy", "I am sad", "I am angry", "I am scared"],
    }
    group_vecs = {name: [encode(p) for p in phrases]
                  for name, phrases in groups.items()}

    within_sims = []
    between_sims = []
    for name, vecs in group_vecs.items():
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                within_sims.append(cos(vecs[i], vecs[j]))
        centroid = np.mean(vecs, axis=0)
        for other_name, other_vecs in group_vecs.items():
            if other_name != name:
                other_centroid = np.mean(other_vecs, axis=0)
                between_sims.append(cos(centroid, other_centroid))

    cluster_ratio = np.mean(within_sims) / max(np.mean(between_sims), 0.01)

    # 2. Directional consistency: "big X" - "X" should be consistent
    size_pairs = [("a cat", "a big cat"), ("a dog", "a big dog"),
                  ("a house", "a big house"), ("a car", "a big car")]
    deltas = []
    for base, modified in size_pairs:
        vb = encode(base)
        vm = encode(modified)
        d = vm - vb
        d = d / (np.linalg.norm(d) + 1e-8)
        deltas.append(d)
    dir_sims = [cos(deltas[i], deltas[j])
                for i in range(len(deltas)) for j in range(i + 1, len(deltas))]
    direction_consistency = float(np.mean(dir_sims))

    # 3. Analogies: king-man+woman≈queen, big cat - cat + dog ≈ big dog
    analogies = [
        ("a big cat", "a cat", "a dog", "a big dog"),
        ("he is happy", "he", "she", "she is happy"),
        ("a hot day", "hot", "cold", "a cold day"),
    ]
    analogy_sims = []
    for a, b, c, d in analogies:
        va, vb, vc, vd = encode(a), encode(b), encode(c), encode(d)
        result = va - vb + vc
        result = result / (np.linalg.norm(result) + 1e-8)
        analogy_sims.append(cos(result, vd))
    analogy_score = float(np.mean(analogy_sims))

    model.train()
    return {
        "cluster_ratio": cluster_ratio,
        "direction_consistency": direction_consistency,
        "analogy_score": analogy_score,
    }


def load_eval_pairs():
    """Load evaluation pairs from data/pairs/eval_*.jsonl."""
    eval_pairs = {}
    data_dir = Path("data/pairs")
    for path in sorted(data_dir.glob("eval_*.jsonl")):
        pairs = []
        with open(path) as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    pair_type = doc.get("type", "paraphrase")
                    pairs.append((doc["text_a"], doc["text_b"],
                                  doc.get("label", 1), pair_type))
                except (json.JSONDecodeError, KeyError):
                    continue
        eval_pairs[path.stem] = pairs
        log(f"  Eval set: {path.name} — {len(pairs):,} pairs")
    return eval_pairs


def build_tracking_set(eval_pairs, max_per_source=50):
    """Build a fixed set of sentences to track across training.

    Returns list of (text_a, text_b, pair_type) — same order every time.
    Includes concept-probing pairs so the animation shows geometry evolution.
    """
    tracking = []
    for source, pairs in sorted(eval_pairs.items()):
        for text_a, text_b, label, pair_type in pairs[:max_per_source]:
            tracking.append((text_a, text_b, pair_type))

    # Diagnostic pairs
    diag_types = ["paraphrase", "paraphrase", "paraphrase",
                  "crosslingual", "non_paraphrase", "hard_negative"]
    for (text_a, text_b), ptype in zip(DIAGNOSTIC_PAIRS, diag_types):
        tracking.append((text_a, text_b, ptype))

    # Concept geometry pairs — these are what we really want to see evolve
    concept_pairs = [
        # Animal cluster
        ("a cat", "a dog", "concept_cluster"),
        ("a cat", "a bird", "concept_cluster"),
        ("a dog", "a bird", "concept_cluster"),
        # Vehicle cluster
        ("a car", "a truck", "concept_cluster"),
        ("a car", "a bus", "concept_cluster"),
        # Cross-cluster (should be far apart)
        ("a cat", "a car", "concept_cross"),
        ("a dog", "a truck", "concept_cross"),
        ("I am happy", "a car", "concept_cross"),
        # Size direction
        ("a cat", "a big cat", "concept_size"),
        ("a dog", "a big dog", "concept_size"),
        ("a house", "a big house", "concept_size"),
        # Emotion direction
        ("I am happy", "I am sad", "concept_emotion"),
        ("I am happy", "I am angry", "concept_emotion"),
        # Analogies
        ("a big cat", "a big dog", "concept_analogy"),
        ("a small cat", "a small dog", "concept_analogy"),
    ]
    for a, b, ptype in concept_pairs:
        tracking.append((a, b, ptype))

    return tracking


@torch.no_grad()
def save_tracking_vectors(model, tracking_set, tokenizer, step, device="cuda"):
    """Save concept vectors for the fixed tracking set. Fast — just forward passes."""
    model.eval()
    vecs_a = []
    vecs_b = []

    # Batch encode for speed
    all_texts_a = [t[0] for t in tracking_set]
    all_texts_b = [t[1] for t in tracking_set]

    batch_size = 64
    for start in range(0, len(all_texts_a), batch_size):
        end = min(start + batch_size, len(all_texts_a))

        enc_a = tokenizer(all_texts_a[start:end], max_length=128,
                          padding=True, truncation=True, return_tensors="pt").to(device)
        enc_b = tokenizer(all_texts_b[start:end], max_length=128,
                          padding=True, truncation=True, return_tensors="pt").to(device)

        va = model(enc_a["input_ids"], enc_a["attention_mask"]).cpu().numpy()
        vb = model(enc_b["input_ids"], enc_b["attention_mask"]).cpu().numpy()
        vecs_a.append(va)
        vecs_b.append(vb)

    vecs_a = np.concatenate(vecs_a, axis=0)
    vecs_b = np.concatenate(vecs_b, axis=0)

    track_dir = Path("logs/tracking")
    track_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(track_dir / f"step_{step:06d}.npz",
                        vecs_a=vecs_a, vecs_b=vecs_b, step=step)
    log(f"  Tracking vectors saved (step {step})")
    model.train()


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

def save_checkpoint(model, optimizer, scaler, config, step, loss, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "loss": loss,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    path = os.path.join(checkpoint_dir, f"step_{step:06d}.pt")
    torch.save(ckpt, path)
    latest = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(ckpt, latest)
    log(f"  Checkpoint saved: {path}")

    # Cleanup: keep milestones (every 25k) + last 5
    all_ckpts = sorted(Path(checkpoint_dir).glob("step_*.pt"))
    to_keep = set()
    for c in all_ckpts:
        step_num = int(c.stem.split("_")[1])
        if step_num % 25000 == 0:
            to_keep.add(c)
    for c in all_ckpts[-5:]:
        to_keep.add(c)
    for c in all_ckpts:
        if c not in to_keep:
            c.unlink()


def load_checkpoint(path, device="cuda"):
    log(f"Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = EncoderConfig(**ckpt["config"])
    model = SemanticEncoder(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR,
                                   betas=BETAS, weight_decay=WEIGHT_DECAY)
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler = torch.amp.GradScaler("cuda")
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    step = ckpt.get("step", 0)
    total, _ = model.count_parameters()
    log(f"Resumed: {total:,} params | step {step}")
    return model, optimizer, scaler, config, step


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False, eval_only=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V4 — SEMANTIC ENCODER TRAINING")
    log("=" * 70)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    log(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    # Update config with actual vocab size
    model_config = dict(ENCODER_CONFIG)
    model_config["vocab_size"] = tokenizer.vocab_size

    # Resolve checkpoint
    if resume_from is None and not fresh:
        ckpt_dir = Path(CHECKPOINT_DIR)
        latest = ckpt_dir / "latest.pt"
        if latest.exists():
            resume_from = str(latest)

    # Initialize or resume
    if resume_from:
        model, optimizer, scaler, config, start_step = \
            load_checkpoint(resume_from, device)
    else:
        log("Starting fresh training...")
        config = EncoderConfig(**model_config)
        model = SemanticEncoder(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=PEAK_LR, betas=BETAS,
            weight_decay=WEIGHT_DECAY
        )
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0
        total, _ = model.count_parameters()
        log(f"Model: {total:,} params ({total/1e6:.1f}M)")

    # Load eval pairs
    eval_pairs = load_eval_pairs()

    # Eval-only mode
    if eval_only:
        log("\n--- DIAGNOSTICS ---")
        results = run_diagnostics(model, tokenizer, device)
        for a, b, sim in results:
            log(f"  {sim:+.4f}  {a}  ↔  {b}")
        if eval_pairs:
            metrics = eval_geometry(model, eval_pairs, tokenizer, device)
            log(f"\n  Para sim:        {metrics['para_sim']:.4f} (target >0.95)")
            log(f"  Non-para sim:    {metrics['nonpara_sim']:.4f} (target <0.30)")
            log(f"  Separation:      {metrics['sep_ratio']:.4f} (target >3.0)")
            log(f"  Cross-lingual+:  {metrics['crosslingual_sim']:.4f} (target >0.90)")
            log(f"  Cross-lingual-:  {metrics['crosslingual_neg_sim']:.4f} (target <0.30)")
            log(f"  Hard negative:   {metrics['hard_neg_sim']:.4f} (target <0.40)")
        return

    # Compile
    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()

    # Data
    log("Loading training pairs...")
    dataset = PairDataset(tokenizer, max_len=config.max_seq_len)
    if len(dataset.pos_pairs) == 0:
        log("ERROR: No training pairs found. Run build_pairs.py first.")
        return

    # Build tracking set for animation
    tracking_set = build_tracking_set(eval_pairs)
    log(f"  Tracking {len(tracking_set)} pairs for animation")

    # Save initial vectors (random weights — step 0)
    save_tracking_vectors(model, tracking_set, tokenizer, 0, device)

    # Save tracking metadata once
    track_dir = Path("logs/tracking")
    track_dir.mkdir(parents=True, exist_ok=True)
    with open(track_dir / "metadata.json", "w") as f:
        meta = [{"text_a": a, "text_b": b, "type": t} for a, b, t in tracking_set]
        json.dump(meta, f)

    # Run initial diagnostics
    log("\n--- INITIAL DIAGNOSTICS (random weights) ---")
    results = run_diagnostics(model, tokenizer, device)
    for a, b, sim in results:
        log(f"  {sim:+.4f}  {a}  ↔  {b}")

    # Training state
    losses = []
    start_time = time.time()

    # Graceful shutdown
    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        log("Shutdown signal received, saving checkpoint...")
        shutdown_requested = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log(f"\nTraining plan:")
    log(f"  Model: {config.hidden_size}h × {config.num_layers}L × {config.num_heads}heads → {config.output_dim}d")
    log(f"  Batch size: {BATCH_SIZE} (pos) + {NEG_BATCH_SIZE} (neg)")
    log(f"  Peak LR: {PEAK_LR} | Steps: {start_step} → {TOTAL_STEPS}")
    log(f"  Positive pairs: {len(dataset.pos_pairs):,}")
    log(f"  Negative pairs: {len(dataset.neg_pairs):,}")
    log(f"  Graded pairs:  {len(dataset.graded_pairs):,}")
    log(f"  Targets: pos_sim>{POS_TARGET} neg_sim<{NEG_TARGET} | neg_weight={NEG_WEIGHT} | graded_weight={GRADED_WEIGHT}")
    log("-" * 70)

    for step in range(start_step, TOTAL_STEPS):
        if shutdown_requested:
            break

        # LR schedule
        current_lr = cosine_lr(step, TOTAL_STEPS, PEAK_LR, WARMUP_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Get positive batch
        enc_a, enc_b = dataset.get_batch(BATCH_SIZE)
        ids_a = enc_a["input_ids"].to(device)
        mask_a = enc_a["attention_mask"].to(device)
        ids_b = enc_b["input_ids"].to(device)
        mask_b = enc_b["attention_mask"].to(device)

        # Get negative batch
        neg_enc_a, neg_enc_b = dataset.get_neg_batch(NEG_BATCH_SIZE)
        neg_ids_a = neg_enc_a["input_ids"].to(device)
        neg_mask_a = neg_enc_a["attention_mask"].to(device)
        neg_ids_b = neg_enc_b["input_ids"].to(device)
        neg_mask_b = neg_enc_b["attention_mask"].to(device)

        # Get graded batch (STS-B)
        graded_enc_a, graded_enc_b, graded_targets = \
            dataset.get_graded_batch(GRADED_BATCH_SIZE)

        # Forward + unified adaptive loss
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            z_pos_a = model(ids_a, mask_a)
            z_pos_b = model(ids_b, mask_b)
            z_neg_a = model(neg_ids_a, neg_mask_a)
            z_neg_b = model(neg_ids_b, neg_mask_b)

            pos_loss, neg_loss, batch_pos_sim, batch_neg_sim = \
                adaptive_contrastive_loss(z_pos_a, z_pos_b, z_neg_a, z_neg_b,
                                          pos_target=POS_TARGET,
                                          neg_target=NEG_TARGET)
            loss = pos_loss + NEG_WEIGHT * neg_loss

            # Graded similarity regression loss
            g_loss_val = 0.0
            if graded_targets is not None:
                g_ids_a = graded_enc_a["input_ids"].to(device)
                g_mask_a = graded_enc_a["attention_mask"].to(device)
                g_ids_b = graded_enc_b["input_ids"].to(device)
                g_mask_b = graded_enc_b["attention_mask"].to(device)
                g_targets = graded_targets.to(device)

                z_g_a = model(g_ids_a, g_mask_a)
                z_g_b = model(g_ids_b, g_mask_b)
                g_loss, _ = graded_similarity_loss(z_g_a, z_g_b, g_targets)
                loss = loss + GRADED_WEIGHT * g_loss
                g_loss_val = g_loss.item()

        if torch.isnan(loss):
            log(f"NaN loss at step {step}, skipping")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        pos_loss_val = pos_loss.item()
        neg_loss_val = neg_loss.item()
        losses.append(loss_val)

        # Log
        if (step + 1) % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            hours = elapsed / 3600
            pct = (step + 1) / TOTAL_STEPS * 100
            g_str = f" g={g_loss_val:.3f}" if g_loss_val > 0 else ""
            log(f"step {step+1:>7d} | loss {avg_loss:.4f} "
                f"(pos={pos_loss_val:.3f} neg={neg_loss_val:.3f}{g_str}) | "
                f"p_sim={batch_pos_sim:.3f} n_sim={batch_neg_sim:.3f} | "
                f"lr {current_lr:.2e} | {pct:.1f}%")

        # Eval geometry
        if (step + 1) % EVAL_EVERY == 0 and eval_pairs:
            metrics = eval_geometry(model, eval_pairs, tokenizer, device)
            concept = eval_concept_geometry(model, tokenizer, device)
            log(f"  EVAL: para={metrics['para_sim']:.3f} "
                f"non={metrics['nonpara_sim']:.3f} "
                f"sep={metrics['sep_ratio']:.2f} "
                f"xlang+={metrics['crosslingual_sim']:.3f} "
                f"xlang-={metrics['crosslingual_neg_sim']:.3f} "
                f"hard={metrics['hard_neg_sim']:.3f}")
            log(f"  CONCEPT: cluster={concept['cluster_ratio']:.2f} "
                f"direction={concept['direction_consistency']:.3f} "
                f"analogy={concept['analogy_score']:.3f}")
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            log_metrics(step + 1, avg_loss, metrics["para_sim"],
                        metrics["nonpara_sim"], metrics["sep_ratio"],
                        metrics["crosslingual_sim"], metrics["hard_neg_sim"],
                        current_lr, elapsed / 3600)

        # Diagnostics
        if (step + 1) % SAMPLE_EVERY == 0:
            log("--- DIAGNOSTIC PAIRS ---")
            results = run_diagnostics(model, tokenizer, device)
            for a, b, sim in results:
                log(f"  {sim:+.4f}  {a}  ↔  {b}")

        # Save tracking vectors for animation
        if (step + 1) % UMAP_EVERY == 0:
            save_tracking_vectors(model, tracking_set, tokenizer, step + 1, device)

        # Checkpoint
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            save_checkpoint(model, optimizer, scaler, config, step + 1,
                            avg_loss, CHECKPOINT_DIR)

        if shutdown_requested:
            break

    # Final save
    if losses:
        avg_loss = sum(losses[-100:]) / len(losses[-100:])
        save_checkpoint(model, optimizer, scaler, config, step + 1,
                        avg_loss, CHECKPOINT_DIR)

    # Final diagnostics
    log("\n--- FINAL DIAGNOSTICS ---")
    results = run_diagnostics(model, tokenizer, device)
    for a, b, sim in results:
        log(f"  {sim:+.4f}  {a}  ↔  {b}")

    elapsed = time.time() - start_time
    log("=" * 70)
    log(f"ENCODER TRAINING {'STOPPED' if shutdown_requested else 'COMPLETE'}")
    log(f"Final step: {step + 1} | Loss: {avg_loss:.4f} | Time: {elapsed/3600:.1f}h")
    log("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FLM V4 Encoder Training")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.checkpoint:
        train(resume_from=args.checkpoint, eval_only=args.eval_only)
    elif args.fresh:
        train(fresh=True, eval_only=args.eval_only)
    elif args.resume:
        train(eval_only=args.eval_only)
    else:
        train(fresh=True, eval_only=args.eval_only)
