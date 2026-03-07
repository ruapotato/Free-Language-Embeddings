"""
FLM V4 — Train Concept Autoencoder V4
=======================================
Hard negatives + per-dimension variance + smooth weight scheduling.

Key changes from V3:
  - Uses ALL pair data: positives, hard negatives (NLI contradictions, PAWS),
    graded STS similarity
  - Spectral spread loss: directly maximizes effective rank by penalizing
    uneven eigenvalue distribution
  - Smooth recon decay: recon weight = max(floor, 1/recon_loss) so it
    naturally lessens as reconstruction improves, never fully drops off
  - No stage freezing — all params train throughout

Usage:
    python train_concept.py --fresh          # fresh training
    python train_concept.py --resume         # resume from checkpoint
    python train_concept.py --eval-only      # diagnostics only
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
import torch.nn.functional as F
from pathlib import Path
from concept_model import (ConceptConfig, ConceptAutoencoder,
                           reconstruction_loss, info_nce_loss,
                           word_order_info_nce, slot_decorrelation_loss)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v4"
LOG_DIR = "logs"

MODEL_CONFIG = dict(
    vocab_size=30522,  # BERT tokenizer
    enc_hidden=384,
    enc_layers=6,
    enc_heads=6,
    enc_intermediate=1536,
    num_concepts=8,
    concept_dim=128,
    dec_hidden=384,
    dec_layers=6,
    dec_heads=6,
    dec_intermediate=1536,
    max_seq_len=128,
    dropout=0.1,
)

# Training hyperparameters
BATCH_SIZE = 64
PAIR_BATCH_SIZE = 128
PEAK_LR = 3e-4
WARMUP_STEPS = 2000
TOTAL_STEPS = 200_000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0

# InfoNCE temperature
NCE_TEMPERATURE = 0.07

# Loss weights (geometry losses scale up as recon improves)
RECON_FLOOR = 0.2             # minimum recon weight (never goes below this)
NCE_WEIGHT = 1.0
HARD_NEG_WEIGHT = 0.5
WO_WEIGHT = 0.5
DECORR_WEIGHT = 1.0
STS_WEIGHT = 0.5
DIMVAR_WEIGHT = 1.0           # per-dimension variance loss (rank pusher)

# Logging
LOG_EVERY = 50
EVAL_EVERY = 500
SAMPLE_EVERY = 2000
CHECKPOINT_EVERY = 5000

LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v4.log",
    "metrics": f"{LOG_DIR}/concept_v4_metrics.csv",
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


def get_recon_weight(recon_loss_val):
    """Smooth recon weight: high when recon is bad, decays as it improves.

    w = max(RECON_FLOOR, min(2.0, recon_loss_val))
    When recon=5.0 → w=2.0 (capped), when recon=0.5 → w=0.5, when recon=0.1 → w=0.2 (floor)
    """
    return max(RECON_FLOOR, min(2.0, recon_loss_val))


def get_geometry_weight(recon_loss_val):
    """Geometry ramps up as recon improves. Sigmoid-ish based on recon quality.

    When recon > 3.0: ~0.1 (minimal geometry)
    When recon ~ 1.0: ~0.5
    When recon < 0.3: ~1.0 (full geometry)
    """
    # Smooth ramp: 1 / (1 + recon_loss)
    return 1.0 / (1.0 + recon_loss_val)


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


# ---------------------------------------------------------------------------
# Prefetch buffer — tokenize on CPU threads while GPU trains
# ---------------------------------------------------------------------------

class PrefetchBuffer:
    """Pre-tokenizes batches in background threads to keep GPU fed."""

    def __init__(self, recon_dataset, pair_dataset, device,
                 pair_batch_size=128, hard_neg_batch=64, buf_size=4):
        self.recon = recon_dataset
        self.pairs = pair_dataset
        self.device = device
        self.pair_bs = pair_batch_size
        self.hn_bs = hard_neg_batch
        self.recon_q = queue.Queue(maxsize=buf_size)
        self.pos_q = queue.Queue(maxsize=buf_size)
        self.hn_q = queue.Queue(maxsize=buf_size)
        self.sts_q = queue.Queue(maxsize=buf_size)
        self._stop = threading.Event()
        self._threads = []

    def start(self):
        """Start background prefetch threads."""
        for name, fn in [("recon", self._fill_recon),
                         ("pos", self._fill_pos),
                         ("hn", self._fill_hn),
                         ("sts", self._fill_sts)]:
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

    def get_recon(self):
        return self.recon_q.get()

    def get_pos(self):
        return self.pos_q.get()

    def get_hn(self):
        return self.hn_q.get()

    def get_sts(self):
        return self.sts_q.get()


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def hard_negative_info_nce(concepts_pos_a, concepts_pos_b,
                           concepts_hn_a, concepts_hn_b,
                           temperature=0.07):
    """InfoNCE with explicit hard negatives mixed into the negative pool."""
    B_pos = concepts_pos_a.shape[0]

    flat_pa = F.normalize(concepts_pos_a.view(B_pos, -1), p=2, dim=-1)
    flat_pb = F.normalize(concepts_pos_b.view(B_pos, -1), p=2, dim=-1)

    if concepts_hn_a is not None:
        B_hn = concepts_hn_a.shape[0]
        flat_hna = F.normalize(concepts_hn_a.view(B_hn, -1), p=2, dim=-1)
        flat_hnb = F.normalize(concepts_hn_b.view(B_hn, -1), p=2, dim=-1)
        all_b = torch.cat([flat_pb, flat_hna, flat_hnb], dim=0)
    else:
        all_b = flat_pb

    sim_matrix = flat_pa @ all_b.T / temperature
    labels = torch.arange(B_pos, device=sim_matrix.device)

    sim_pos_block = flat_pa @ flat_pb.T / temperature
    loss = (F.cross_entropy(sim_matrix, labels) +
            F.cross_entropy(sim_pos_block.T, labels)) / 2

    with torch.no_grad():
        pos_sim = (flat_pa * flat_pb).sum(dim=-1).mean().item()
        mask = ~torch.eye(B_pos, dtype=torch.bool, device=flat_pa.device)
        neg_sim = (flat_pa @ flat_pb.T)[mask].mean().item()

    return loss, pos_sim, neg_sim


def sts_loss(concepts_a, concepts_b, target_scores):
    """Graded similarity: cosine similarity should match target score."""
    flat_a = F.normalize(concepts_a.view(concepts_a.shape[0], -1), p=2, dim=-1)
    flat_b = F.normalize(concepts_b.view(concepts_b.shape[0], -1), p=2, dim=-1)
    pred_sim = F.cosine_similarity(flat_a, flat_b)
    loss = F.mse_loss(pred_sim, target_scores.to(pred_sim.device))
    return loss, pred_sim.mean().item()


def dimension_variance_loss(concepts):
    """
    Push the model to use ALL dimensions by penalizing low per-dimension variance.

    For each dimension in the flattened concept vector, compute variance across
    the batch. Penalize dimensions with low variance via -log(var). No sample
    cap — works on full batch, O(B*D), and can push all 1024 dims.

    concepts: (B, K, D) — will be flattened to (B, K*D)
    """
    flat = concepts.view(concepts.shape[0], -1).float()  # (B, K*D)
    var = flat.var(dim=0)  # (K*D,) variance per dimension
    # -log(var) penalizes low-variance dims heavily, diminishing return on high-var
    loss = -torch.log(var.clamp(min=1e-8)).mean()
    return loss

    # KL divergence from uniform (want to minimize)
    # KL(uniform || p) = sum(uniform * log(uniform/p))
    loss = (uniform * (uniform.log() - p.log())).sum()

    return loss


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
        vec_a = m.concept_vector(enc_a["input_ids"], enc_a["attention_mask"])
        vec_b = m.concept_vector(enc_b["input_ids"], enc_b["attention_mask"])
        sim = F.cosine_similarity(vec_a, vec_b).item()
        results.append((text_a[:45], text_b[:45], sim, pair_type))
    model.train()
    return results


@torch.no_grad()
def measure_effective_rank(model, tokenizer, texts, device="cuda"):
    """Measure effective rank of concept representations via PCA."""
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

def save_checkpoint(model, optimizer, scaler, config, step, loss, checkpoint_dir):
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
        "timestamp": datetime.datetime.now().isoformat(),
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


def load_checkpoint(path, device="cuda"):
    log(f"Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoder(config).to(device)
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
    log(f"Resumed: {total:,} params | step {step}")
    return model, optimizer, scaler, config, step


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False, eval_only=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V4 — CONCEPT AUTOENCODER V4 (Hard Neg + Spectral Spread)")
    log("=" * 70)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    log(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    model_config = dict(MODEL_CONFIG)
    model_config["vocab_size"] = tokenizer.vocab_size

    if resume_from is None and not fresh:
        ckpt_dir = Path(CHECKPOINT_DIR)
        latest = ckpt_dir / "latest.pt"
        if latest.exists():
            resume_from = str(latest)

    if resume_from:
        model, optimizer, scaler, config, start_step = \
            load_checkpoint(resume_from, device)
    else:
        log("Starting fresh training...")
        config = ConceptConfig(**model_config)
        model = ConceptAutoencoder(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=PEAK_LR, betas=BETAS,
            weight_decay=WEIGHT_DECAY
        )
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0
        total, _ = model.count_parameters()
        log(f"Model: {total:,} params ({total/1e6:.1f}M)")
        log(f"Bottleneck: {config.num_concepts} concepts x {config.concept_dim} dim "
            f"= {config.total_concept_dim} total dims")

    if eval_only:
        log("\n--- DIAGNOSTICS ---")
        results = run_diagnostics(model, tokenizer, device)
        for a, b, sim, ptype in results:
            log(f"  {sim:+.4f}  [{ptype:<12s}] {a}  <->  {b}")
        log("\n--- RECONSTRUCTION ---")
        recon = test_reconstruction(model, tokenizer, device)
        for orig, decoded in recon:
            log(f"  IN:  {orig}")
            log(f"  OUT: {decoded}")
            log("")
        return

    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()

    log("Loading data...")
    recon_dataset = ReconstructionDataset(tokenizer, max_len=config.max_seq_len)
    pair_dataset = PairDataset(tokenizer, max_len=config.max_seq_len)

    if len(recon_dataset.texts) == 0:
        log("ERROR: No training texts found.")
        return

    rank_texts = random.sample(recon_dataset.texts, min(200, len(recon_dataset.texts)))

    log("\n--- INITIAL DIAGNOSTICS ---")
    results = run_diagnostics(model, tokenizer, device)
    for a, b, sim, ptype in results:
        log(f"  {sim:+.4f}  [{ptype:<12s}] {a}  <->  {b}")

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

    log(f"\nTraining plan (V4 — Hard Negatives + Spectral Spread):")
    log(f"  Encoder: {config.enc_hidden}h x {config.enc_layers}L x {config.enc_heads}heads")
    log(f"  Decoder: {config.dec_hidden}h x {config.dec_layers}L x {config.dec_heads}heads")
    log(f"  Bottleneck: {config.num_concepts} x {config.concept_dim} = {config.total_concept_dim} dims")
    log(f"  Batch: {BATCH_SIZE} recon + {PAIR_BATCH_SIZE} pairs + {HARD_NEG_BATCH} hard neg")
    log(f"  Peak LR: {PEAK_LR} | Steps: {start_step} -> {TOTAL_STEPS}")
    log(f"  Recon weight: smooth decay (floor={RECON_FLOOR})")
    log(f"  Geometry: nce={NCE_WEIGHT} wo={WO_WEIGHT} decorr={DECORR_WEIGHT} "
        f"dimvar={DIMVAR_WEIGHT} sts={STS_WEIGHT}")
    log(f"  Data: {len(pair_dataset.pos_pairs):,} pos + "
        f"{len(pair_dataset.hard_neg_pairs):,} hard neg + "
        f"{len(pair_dataset.sts_pairs):,} STS")
    log("-" * 70)

    # Start prefetch buffer
    prefetch = PrefetchBuffer(recon_dataset, pair_dataset, device,
                              pair_batch_size=PAIR_BATCH_SIZE,
                              hard_neg_batch=HARD_NEG_BATCH)
    prefetch.start()
    log("Prefetch buffer started (4 threads)")

    for step in range(start_step, TOTAL_STEPS):
        if shutdown_requested:
            break

        current_lr = cosine_lr(step, TOTAL_STEPS, PEAK_LR, WARMUP_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)

        # --- Reconstruction ---
        recon_enc = prefetch.get_recon()
        input_ids = recon_enc["input_ids"].to(device, non_blocking=True)
        attention_mask = recon_enc["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, concepts = model(input_ids, attention_mask)
            targets = input_ids[:, 1:]
            r_loss = reconstruction_loss(logits, targets)

        r_loss_val = r_loss.item()
        recon_w = get_recon_weight(r_loss_val)
        geo_w = get_geometry_weight(r_loss_val)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            total_loss = recon_w * r_loss

            # Slot decorrelation
            decorr = slot_decorrelation_loss(concepts)
            total_loss = total_loss + geo_w * DECORR_WEIGHT * decorr

            # Word-order InfoNCE
            shuffled_ids, shuffled_mask = shuffle_tokens(input_ids, attention_mask)
            concepts_shuf = m.encode(shuffled_ids, shuffled_mask)
            wo_loss, wo_sim_val = word_order_info_nce(
                concepts, concepts_shuf, temperature=NCE_TEMPERATURE)
            total_loss = total_loss + geo_w * WO_WEIGHT * wo_loss

        # Per-dimension variance loss (needs float32, outside autocast)
        dimvar = dimension_variance_loss(concepts)
        total_loss = total_loss + geo_w * DIMVAR_WEIGHT * dimvar

        wo_loss_val = wo_loss.item()
        decorr_val = decorr.item()
        dimvar_val = dimvar.item()

        # --- Paraphrase InfoNCE with hard negatives ---
        nce_loss_val = 0.0
        p_sim_val = 0.0
        n_sim_val = 0.0

        if pair_dataset.pos_pairs:
            enc_a, enc_b = prefetch.get_pos()
            hn_enc_a, hn_enc_b = prefetch.get_hn()

            with torch.amp.autocast("cuda", dtype=torch.float16):
                ids_a = enc_a["input_ids"].to(device, non_blocking=True)
                mask_a = enc_a["attention_mask"].to(device, non_blocking=True)
                ids_b = enc_b["input_ids"].to(device, non_blocking=True)
                mask_b = enc_b["attention_mask"].to(device, non_blocking=True)
                concepts_a = m.encode(ids_a, mask_a)
                concepts_b = m.encode(ids_b, mask_b)

                concepts_hna = None
                concepts_hnb = None
                if hn_enc_a is not None:
                    hn_ids_a = hn_enc_a["input_ids"].to(device, non_blocking=True)
                    hn_mask_a = hn_enc_a["attention_mask"].to(device, non_blocking=True)
                    hn_ids_b = hn_enc_b["input_ids"].to(device, non_blocking=True)
                    hn_mask_b = hn_enc_b["attention_mask"].to(device, non_blocking=True)
                    concepts_hna = m.encode(hn_ids_a, hn_mask_a)
                    concepts_hnb = m.encode(hn_ids_b, hn_mask_b)

                nce_loss, p_sim_val, n_sim_val = hard_negative_info_nce(
                    concepts_a, concepts_b,
                    concepts_hna, concepts_hnb,
                    temperature=NCE_TEMPERATURE)
                total_loss = total_loss + geo_w * NCE_WEIGHT * nce_loss

            nce_loss_val = nce_loss.item()

        # --- STS graded loss ---
        sts_loss_val = 0.0
        sts_enc_a, sts_enc_b, sts_scores = prefetch.get_sts()
        if sts_enc_a is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                sts_ids_a = sts_enc_a["input_ids"].to(device, non_blocking=True)
                sts_mask_a = sts_enc_a["attention_mask"].to(device, non_blocking=True)
                sts_ids_b = sts_enc_b["input_ids"].to(device, non_blocking=True)
                sts_mask_b = sts_enc_b["attention_mask"].to(device, non_blocking=True)
                sts_ca = m.encode(sts_ids_a, sts_mask_a)
                sts_cb = m.encode(sts_ids_b, sts_mask_b)
                s_loss, _ = sts_loss(sts_ca, sts_cb, sts_scores)
                total_loss = total_loss + geo_w * STS_WEIGHT * s_loss
            sts_loss_val = s_loss.item()

        if torch.isnan(total_loss):
            log(f"NaN loss at step {step}, skipping")
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss_val = total_loss.item()
        losses.append(total_loss_val)

        # Log
        if (step + 1) % LOG_EVERY == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            pct = (step + 1) / TOTAL_STEPS * 100
            log(f"step {step+1:>7d} | loss {avg_loss:.4f} "
                f"(recon={r_loss_val:.3f} nce={nce_loss_val:.3f} "
                f"wo={wo_loss_val:.3f} decorr={decorr_val:.3f} "
                f"dimvar={dimvar_val:.3f} sts={sts_loss_val:.3f}) "
                f"[rw={recon_w:.2f} gw={geo_w:.2f}] | "
                f"p_sim={p_sim_val:.3f} n_sim={n_sim_val:.3f} "
                f"wo_sim={wo_sim_val:.3f} | "
                f"lr {current_lr:.2e} | {pct:.1f}%")

        # Eval
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

            log(f"  EVAL: para_sim={avg_para:.3f} wo_sim={avg_wo:.3f} "
                f"unrelated_sim={avg_neg:.3f} | rank90={rank90} rank95={rank95}")

            for a, b, sim, ptype in results:
                log(f"    {sim:+.4f}  [{ptype:<12s}] {a} <-> {b}")

            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            log_metrics(step + 1, avg_loss, avg_para, avg_neg,
                        avg_wo, rank90, current_lr, elapsed / 3600)

        # Reconstruction samples
        if (step + 1) % SAMPLE_EVERY == 0:
            log("--- RECONSTRUCTION SAMPLES ---")
            recon = test_reconstruction(model, tokenizer, device)
            for orig, decoded in recon:
                match = "OK" if orig.lower().strip() == decoded.lower().strip() else "DIFF"
                log(f"  [{match}] {orig}")
                log(f"        -> {decoded}")

        # Checkpoint
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            save_checkpoint(model, optimizer, scaler, config, step + 1,
                            avg_loss, CHECKPOINT_DIR)

        if shutdown_requested:
            break

    prefetch.stop()
    if losses:
        avg_loss = sum(losses[-100:]) / len(losses[-100:])
        save_checkpoint(model, optimizer, scaler, config,
                        step + 1 if not shutdown_requested else step,
                        avg_loss, CHECKPOINT_DIR)
    log("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train concept autoencoder V4")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    with open(".train_pid", "w") as f:
        f.write(str(os.getpid()))

    train(resume_from=args.resume, fresh=args.fresh, eval_only=args.eval_only)
