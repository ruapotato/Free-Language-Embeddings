"""
FLM V4 — Train Concept Autoencoder
=====================================
Reconstruction + paraphrase training. The concept stack must encode enough
information to reconstruct text AND produce similar vectors for paraphrases.

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
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from concept_model import (ConceptConfig, ConceptAutoencoder,
                           reconstruction_loss, paraphrase_loss, negative_loss,
                           word_order_loss)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v2"
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
BATCH_SIZE = 64               # reconstruction batch
PARA_BATCH_SIZE = 64          # paraphrase pair batch
NEG_BATCH_SIZE = 64           # negative pair batch
PEAK_LR = 3e-4
WARMUP_STEPS = 2000
TOTAL_STEPS = 200_000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0

# Loss weights — scheduled dynamically in training loop
# Phase 1 (recon_loss > 2.0): recon=2.0, geometry=0.25 each — learn to reconstruct first
# Phase 2 (recon_loss < 2.0): recon=1.0, geometry=0.5 each — balance
# Phase 3 (recon_loss < 1.0): recon=0.5, geometry=1.0 each — focus on geometry
POS_TARGET = 0.9
NEG_TARGET = 0.3
WO_TARGET = 0.3               # shuffled sentences should be far from original

def get_loss_weights(recon_loss_val):
    """Schedule loss weights based on reconstruction quality."""
    if recon_loss_val > 2.0:
        return 2.0, 0.25, 0.25, 0.25  # recon, para, neg, wo
    elif recon_loss_val > 1.0:
        return 1.0, 0.5, 0.5, 0.5
    else:
        return 0.5, 1.0, 1.0, 1.0

# Logging
LOG_EVERY = 50
EVAL_EVERY = 500
SAMPLE_EVERY = 1000
CHECKPOINT_EVERY = 5000

LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v2.log",
    "metrics": f"{LOG_DIR}/concept_v2_metrics.csv",
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


# ---------------------------------------------------------------------------
# Word-order augmentation
# ---------------------------------------------------------------------------

def shuffle_tokens(input_ids, attention_mask, pad_id=0, cls_id=101, sep_id=102):
    """
    Swap 2 random content tokens per sequence to create word-order negatives.

    Minimal swap forces the model to encode every word's position, since any
    single swap must produce a different concept vector. Full permutation is
    too easy to detect (broken n-grams) and teaches the wrong lesson.
    """
    shuffled = input_ids.clone()
    for i in range(input_ids.shape[0]):
        mask = attention_mask[i].bool()
        ids = input_ids[i]
        # Find content token positions (not CLS, SEP, or PAD)
        content_pos = []
        for j in range(ids.shape[0]):
            if mask[j] and ids[j].item() not in (pad_id, cls_id, sep_id):
                content_pos.append(j)
        if len(content_pos) <= 2:
            continue
        # Pick 2 distinct positions with different tokens (so swap is visible)
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


def log_metrics(step, recon_loss, para_sim, neg_sim, word_order_sim,
                lr, elapsed_hours):
    metrics_file = LOG_PATHS.get("metrics")
    if not metrics_file:
        return
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    write_header = not os.path.exists(metrics_file)
    with open(metrics_file, "a") as f:
        if write_header:
            f.write("timestamp,step,recon_loss,para_sim,neg_sim,"
                    "word_order_sim,lr,elapsed_hours\n")
        ts = datetime.datetime.now().isoformat()
        f.write(f"{ts},{step},{recon_loss:.6f},{para_sim:.4f},{neg_sim:.4f},"
                f"{word_order_sim:.4f},{lr:.6e},{elapsed_hours:.4f}\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class ReconstructionDataset:
    """Loads raw text for reconstruction training.

    Sources: pair data (text_a and text_b separately), raw text files.
    """

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
    """Loads positive and negative pairs for paraphrase/negative training."""

    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pos_pairs = []
        self.neg_pairs = []
        self._load_pairs()
        random.shuffle(self.pos_pairs)
        random.shuffle(self.neg_pairs)
        self.pos_idx = 0
        self.neg_idx = 0

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
                        if "sim_score" in doc:
                            continue  # skip graded for now
                        label = doc.get("label", 1)
                        pair = (doc["text_a"], doc["text_b"])
                        if label == 1:
                            self.pos_pairs.append(pair)
                        else:
                            self.neg_pairs.append(pair)
                    except (json.JSONDecodeError, KeyError):
                        continue

        log(f"  Pair data: {len(self.pos_pairs):,} pos, {len(self.neg_pairs):,} neg")

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
        enc_a = self.tokenizer(texts_a, max_length=self.max_len,
                               padding=True, truncation=True, return_tensors="pt")
        enc_b = self.tokenizer(texts_b, max_length=self.max_len,
                               padding=True, truncation=True, return_tensors="pt")
        return enc_a, enc_b

    def get_neg_batch(self, batch_size):
        if not self.neg_pairs:
            return None, None
        texts_a, texts_b = [], []
        for _ in range(batch_size):
            if self.neg_idx >= len(self.neg_pairs):
                random.shuffle(self.neg_pairs)
                self.neg_idx = 0
            a, b = self.neg_pairs[self.neg_idx]
            texts_a.append(a)
            texts_b.append(b)
            self.neg_idx += 1
        enc_a = self.tokenizer(texts_a, max_length=self.max_len,
                               padding=True, truncation=True, return_tensors="pt")
        enc_b = self.tokenizer(texts_b, max_length=self.max_len,
                               padding=True, truncation=True, return_tensors="pt")
        return enc_a, enc_b


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _unwrap(model):
    """Get underlying model from torch.compile wrapper."""
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
def test_reconstruction(model, tokenizer, device="cuda"):
    """Encode then decode a few sentences to see if reconstruction works."""
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

        # Greedy decode
        bos_id = tokenizer.cls_token_id or 101  # [CLS] for BERT
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
    # Strip torch.compile prefix
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

    # Cleanup: keep milestones (every 50k) + last 3
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
    log("FLM V4 — CONCEPT AUTOENCODER TRAINING")
    log("=" * 70)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    log(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    model_config = dict(MODEL_CONFIG)
    model_config["vocab_size"] = tokenizer.vocab_size

    # Resolve checkpoint
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

    # Eval-only mode
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

    # Compile
    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()

    # Data
    log("Loading data...")
    recon_dataset = ReconstructionDataset(tokenizer, max_len=config.max_seq_len)
    pair_dataset = PairDataset(tokenizer, max_len=config.max_seq_len)

    if len(recon_dataset.texts) == 0:
        log("ERROR: No training texts found. Run build_pairs.py first.")
        return

    # Initial diagnostics
    log("\n--- INITIAL DIAGNOSTICS (random weights) ---")
    results = run_diagnostics(model, tokenizer, device)
    for a, b, sim, ptype in results:
        log(f"  {sim:+.4f}  [{ptype:<12s}] {a}  <->  {b}")

    # Training state
    losses = []
    start_time = time.time()

    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        log("Shutdown signal received, saving checkpoint...")
        shutdown_requested = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log(f"\nTraining plan:")
    log(f"  Encoder: {config.enc_hidden}h x {config.enc_layers}L x {config.enc_heads}heads")
    log(f"  Decoder: {config.dec_hidden}h x {config.dec_layers}L x {config.dec_heads}heads")
    log(f"  Bottleneck: {config.num_concepts} x {config.concept_dim} = {config.total_concept_dim} dims")
    log(f"  Batch: {BATCH_SIZE} recon + {PARA_BATCH_SIZE} para + {NEG_BATCH_SIZE} neg")
    log(f"  Peak LR: {PEAK_LR} | Steps: {start_step} -> {TOTAL_STEPS}")
    log(f"  Weights: SCHEDULED (P1: r=2.0 g=0.25 | P2: r=1.0 g=0.5 | P3: r=0.5 g=1.0)")
    log(f"  Texts: {len(recon_dataset.texts):,} | Pairs: {len(pair_dataset.pos_pairs):,} pos, {len(pair_dataset.neg_pairs):,} neg")
    log("-" * 70)

    for step in range(start_step, TOTAL_STEPS):
        if shutdown_requested:
            break

        current_lr = cosine_lr(step, TOTAL_STEPS, PEAK_LR, WARMUP_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)

        m = _unwrap(model)

        # --- Reconstruction + Word-order loss (shared batch) ---
        recon_enc = recon_dataset.get_batch(BATCH_SIZE)
        input_ids = recon_enc["input_ids"].to(device)
        attention_mask = recon_enc["attention_mask"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, concepts = model(input_ids, attention_mask)
            targets = input_ids[:, 1:]  # predict next token
            r_loss = reconstruction_loss(logits, targets)

        r_loss_val = r_loss.item()

        # Schedule loss weights based on reconstruction quality
        recon_w, para_w, neg_w, wo_w = get_loss_weights(r_loss_val)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            total_loss = recon_w * r_loss

            # Word-order: shuffle tokens in the same batch, push apart
            shuffled_ids, shuffled_mask = shuffle_tokens(input_ids, attention_mask)
            concepts_shuf = m.encode(shuffled_ids, shuffled_mask)
            wo_loss, wo_sim_val = word_order_loss(concepts, concepts_shuf,
                                                   target_sim=WO_TARGET)
            total_loss = total_loss + wo_w * wo_loss

        wo_loss_val = wo_loss.item()

        # --- Paraphrase loss (encode-only, no decoder) ---
        p_loss_val = 0.0
        p_sim_val = 0.0
        if pair_dataset.pos_pairs:
            enc_a, enc_b = pair_dataset.get_pos_batch(PARA_BATCH_SIZE)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                ids_a = enc_a["input_ids"].to(device)
                mask_a = enc_a["attention_mask"].to(device)
                ids_b = enc_b["input_ids"].to(device)
                mask_b = enc_b["attention_mask"].to(device)
                concepts_a = m.encode(ids_a, mask_a)
                concepts_b = m.encode(ids_b, mask_b)
                p_loss, p_sim_val = paraphrase_loss(concepts_a, concepts_b,
                                                     target_sim=POS_TARGET)
                total_loss = total_loss + para_w * p_loss
            p_loss_val = p_loss.item()

        # --- Negative loss (encode-only, no decoder) ---
        n_loss_val = 0.0
        n_sim_val = 0.0
        neg_enc_a, neg_enc_b = pair_dataset.get_neg_batch(NEG_BATCH_SIZE)
        if neg_enc_a is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                neg_ids_a = neg_enc_a["input_ids"].to(device)
                neg_mask_a = neg_enc_a["attention_mask"].to(device)
                neg_ids_b = neg_enc_b["input_ids"].to(device)
                neg_mask_b = neg_enc_b["attention_mask"].to(device)
                concepts_na = m.encode(neg_ids_a, neg_mask_a)
                concepts_nb = m.encode(neg_ids_b, neg_mask_b)
                n_loss, n_sim_val = negative_loss(concepts_na, concepts_nb,
                                                   target_sim=NEG_TARGET)
                total_loss = total_loss + neg_w * n_loss
            n_loss_val = n_loss.item()

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
            phase = "P1" if r_loss_val > 2.0 else ("P2" if r_loss_val > 1.0 else "P3")
            log(f"step {step+1:>7d} | loss {avg_loss:.4f} "
                f"(recon={r_loss_val:.3f} para={p_loss_val:.3f} neg={n_loss_val:.3f} wo={wo_loss_val:.3f}) [{phase} rw={recon_w}] | "
                f"p_sim={p_sim_val:.3f} n_sim={n_sim_val:.3f} wo_sim={wo_sim_val:.3f} | "
                f"lr {current_lr:.2e} | {pct:.1f}%")

        # Eval
        if (step + 1) % EVAL_EVERY == 0:
            results = run_diagnostics(model, tokenizer, device)
            # Find word order pairs
            word_order_sims = [sim for _, _, sim, pt in results if pt == "word_order"]
            para_sims = [sim for _, _, sim, pt in results if pt == "paraphrase"]
            unrelated_sims = [sim for _, _, sim, pt in results if pt == "unrelated"]

            avg_para = sum(para_sims) / len(para_sims) if para_sims else 0
            avg_wo = sum(word_order_sims) / len(word_order_sims) if word_order_sims else 0
            avg_neg = sum(unrelated_sims) / len(unrelated_sims) if unrelated_sims else 0

            log(f"  EVAL: para_sim={avg_para:.3f} word_order_sim={avg_wo:.3f} "
                f"unrelated_sim={avg_neg:.3f}")

            for a, b, sim, ptype in results:
                marker = ""
                if ptype == "word_order" and sim < 0.8:
                    marker = " !!!"  # celebrate word order sensitivity
                log(f"    {sim:+.4f}  [{ptype:<12s}] {a} <-> {b}{marker}")

            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            log_metrics(step + 1, avg_loss, avg_para, avg_neg,
                        avg_wo, current_lr, elapsed / 3600)

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

    # Final save
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
    parser = argparse.ArgumentParser(description="Train concept autoencoder")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    train(resume_from=args.resume, fresh=args.fresh, eval_only=args.eval_only)
