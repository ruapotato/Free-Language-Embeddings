"""
Worm Brain Sidecar Training Script
====================================
Trains a WormSidecarModel from random weights using dual optimization:

1. Standard backprop (AdamW) for transformer + sidecar projections
2. Hebbian plasticity for worm internal synaptic weights

Same hyperparameters as train_pretrain.py base stage for fair comparison.

Usage:
    python train_worm_sidecar.py                    # fresh training
    python train_worm_sidecar.py --resume           # resume from checkpoint
    python train_worm_sidecar.py --steps 10000      # quick test run
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
from collections import deque
from model import HamnerModel, HamnerConfig
from worm_sidecar import WormSidecarModel

# Import shared utilities from train_pretrain
from train_pretrain import (
    MultiSourceStreamer, ValidationSet,
    SAMPLE_PROMPTS, BETAS, WEIGHT_DECAY, GRAD_CLIP,
    BASE_DATA_RATIOS, MODEL_CONFIG,
)

# ---------------------------------------------------------------------------
# Config — matches base pretrain for fair comparison
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/worm_sidecar"
LOG_DIR = "logs"

BATCH_SIZE = 24
SEQ_LEN = 1024
LR = 2e-4
WARMUP_STEPS = 2000
MAX_STEPS = 400_000   # same as base pretrain
DECAY_FRACTION = 0.10

# Hebbian plasticity
HEBBIAN_EVERY = 10     # apply Hebbian update every K backprop steps
HEBBIAN_LR = 1e-5      # much smaller than backprop LR

# Worm simulation
WORM_SUBSTEPS = 10     # RK4 substeps per worm step
WORM_STRIDE = 64       # process every 64th token (1024/64 = 16 worm steps per batch)

# Logging/checkpointing
CHECKPOINT_EVERY = 1000
SAMPLE_EVERY = 500
LOG_EVERY = 50
VAL_EVERY = 500
KEEP_CHECKPOINTS = 10

METRICS_FILE = f"{LOG_DIR}/worm_sidecar_metrics.csv"
SAMPLES_FILE = f"{LOG_DIR}/worm_sidecar_samples.jsonl"
WORM_DIAG_FILE = f"{LOG_DIR}/worm_sidecar_diagnostics.jsonl"
LOG_FILE = f"{LOG_DIR}/worm_sidecar.log"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def log_metrics(step, loss, perplexity, lr, tokens_per_sec, tokens_total,
                elapsed_hours, val_loss=None):
    os.makedirs(LOG_DIR, exist_ok=True)
    write_header = not os.path.exists(METRICS_FILE)
    with open(METRICS_FILE, "a") as f:
        if write_header:
            f.write("timestamp,step,loss,perplexity,learning_rate,tokens_per_sec,"
                    "tokens_total,tokens_billions,elapsed_hours,val_loss\n")
        ts = datetime.datetime.now().isoformat()
        tokens_b = tokens_total / 1e9
        val_str = f"{val_loss:.6f}" if val_loss is not None else ""
        f.write(f"{ts},{step},{loss:.6f},{perplexity:.2f},{lr:.6e},"
                f"{tokens_per_sec:.0f},{tokens_total},{tokens_b:.4f},"
                f"{elapsed_hours:.4f},{val_str}\n")


def log_samples(step, tokens_total, samples_dict, worm_diag=None):
    os.makedirs(LOG_DIR, exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": step,
        "tokens_total": tokens_total,
        "tokens_billions": round(tokens_total / 1e9, 4),
        "samples": samples_dict,
    }
    if worm_diag:
        entry["worm_diagnostics"] = worm_diag
    with open(SAMPLES_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_worm_diagnostics(step, tokens_total, diag):
    os.makedirs(LOG_DIR, exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": step,
        "tokens_total": tokens_total,
        **diag,
    }
    with open(WORM_DIAG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# LR Schedule (same WSD as base pretrain)
# ---------------------------------------------------------------------------

def wsd_lr(step, max_steps, peak_lr, warmup_steps, decay_fraction):
    min_lr = peak_lr * 0.1
    decay_start = int(max_steps * (1 - decay_fraction))
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    elif step < decay_start:
        return peak_lr
    else:
        progress = (step - decay_start) / max(1, max_steps - decay_start)
        return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir):
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return str(latest)
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    return str(ckpts[-1]) if ckpts else None


def save_checkpoint(model, optimizer, scaler, config, step, loss,
                    tokens_total=0, extra=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    raw_state = model.state_dict()
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}

    ckpt_data = {
        "model_state_dict": clean_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "avg_loss": loss,
        "tokens_total": tokens_total,
        "timestamp": datetime.datetime.now().isoformat(),
        "training_type": "worm_sidecar",
    }
    if extra:
        ckpt_data.update(extra)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{step:07d}.pt")
    torch.save(ckpt_data, ckpt_path)
    latest_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    torch.save(ckpt_data, latest_path)
    log(f"  Checkpoint saved: {ckpt_path}")

    # Cleanup
    all_ckpts = sorted(Path(CHECKPOINT_DIR).glob("step_*.pt"))
    to_keep = set()
    for c in all_ckpts:
        step_num = int(c.stem.split("_")[1])
        if step_num % 10000 == 0:
            to_keep.add(c)
    for c in all_ckpts[-KEEP_CHECKPOINTS:]:
        to_keep.add(c)
    for c in all_ckpts:
        if c not in to_keep:
            c.unlink()


def load_checkpoint(path, device="cuda"):
    log(f"Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    config = HamnerConfig(**ckpt["config"])
    model = WormSidecarModel(config, substeps=WORM_SUBSTEPS, worm_stride=WORM_STRIDE).to(device)

    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)

    # Only optimize backprop-trainable params (exclude worm synaptic weights)
    backprop_params = _get_backprop_params(model)
    optimizer = torch.optim.AdamW(
        backprop_params, lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError):
            log("  Warning: could not restore optimizer state, starting fresh optimizer")

    scaler = torch.amp.GradScaler("cuda")
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    step = ckpt.get("step", 0)
    tokens_total = ckpt.get("tokens_total", 0)

    total_p, _ = model.count_parameters()
    sidecar_p, _ = model.count_sidecar_parameters()
    log(f"Resumed: {total_p:,} total params ({sidecar_p:,} sidecar) | "
        f"step {step} | tokens {tokens_total/1e9:.2f}B")

    return model, optimizer, scaler, config, step, tokens_total


# ---------------------------------------------------------------------------
# Parameter grouping
# ---------------------------------------------------------------------------

def _get_backprop_params(model):
    """Get parameters for AdamW (everything except worm synaptic weights)."""
    # Worm synaptic weights are updated by Hebbian learning, not backprop
    worm_buffer_names = {"worm.chem_weights", "worm.gap_weights"}
    params = []
    for name, param in model.named_parameters():
        if name not in worm_buffer_names:
            params.append(param)
    return params


# ---------------------------------------------------------------------------
# Text generation (adapted for sidecar model)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(model, tokenizer, prompts, device="cuda", max_tokens=60):
    model.eval()
    results = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        try:
            output = model.generate(
                input_ids, max_new_tokens=max_tokens,
                temperature=0.8, top_k=40, top_p=0.9,
                repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id or 0,
            )
            generated = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
            results.append(generated)
        except Exception as e:
            results.append(f"{prompt} [generation error: {e}]")
    model.train()
    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("WORM BRAIN SIDECAR TRAINING — From Random Weights")
    log("=" * 70)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    # Resolve checkpoint
    if resume_from is None and not fresh:
        resume_from = find_latest_checkpoint(CHECKPOINT_DIR)
        if resume_from:
            log(f"Found existing checkpoint: {resume_from}")

    # Initialize or resume
    if resume_from:
        model, optimizer, scaler, config, start_step, tokens_total = \
            load_checkpoint(resume_from, device)
    else:
        log("Starting fresh worm sidecar training...")
        config = HamnerConfig(**MODEL_CONFIG, vocab_size=tokenizer.vocab_size)
        model = WormSidecarModel(config, substeps=WORM_SUBSTEPS, worm_stride=WORM_STRIDE).to(device)

        backprop_params = _get_backprop_params(model)
        optimizer = torch.optim.AdamW(
            backprop_params, lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY
        )
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0
        tokens_total = 0

        total_p, _ = model.count_parameters()
        sidecar_p, _ = model.count_sidecar_parameters()
        log(f"Model: {total_p:,} total params ({sidecar_p:,} sidecar, "
            f"{total_p - sidecar_p:,} transformer)")
        log(f"Sidecar VRAM: ~{sidecar_p * 2 / 1024:.1f}KB (FP16)")

    # Compile: the transformer as a unit + worm RK4 integration.
    # The sidecar forward calls self.model() (compilable) and uses hooks
    # for layer 5 capture + feedback injection.
    if hasattr(torch, "compile"):
        log("Compiling transformer + worm RK4...")
        model.model = torch.compile(model.model)
        model.worm._integrate = torch.compile(model.worm._integrate)

    model.train()

    # Data stream (same as base pretrain)
    data = MultiSourceStreamer(
        tokenizer, seq_len=SEQ_LEN, ratios=BASE_DATA_RATIOS,
    )

    # Validation set
    val_set = ValidationSet(tokenizer, seq_len=SEQ_LEN, n_batches=5,
                            batch_size=BATCH_SIZE)

    # Training state
    losses = deque(maxlen=200)
    rolling_loss = deque(maxlen=100)
    start_time = time.time()
    best_val_loss = float("inf")

    # Graceful shutdown
    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        log("Shutdown signal received, saving checkpoint...")
        shutdown_requested = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    log(f"\nTraining from step {start_step} to {MAX_STEPS}")
    log(f"Batch size: {BATCH_SIZE} | Seq len: {SEQ_LEN}")
    log(f"LR: {LR} (WSD: warmup {WARMUP_STEPS}, decay last {DECAY_FRACTION*100:.0f}%)")
    log(f"Hebbian update every {HEBBIAN_EVERY} steps (lr={HEBBIAN_LR})")
    log(f"Worm substeps: {WORM_SUBSTEPS} | stride: {WORM_STRIDE} ({SEQ_LEN // WORM_STRIDE} worm steps/batch)")
    log(f"Data ratios: {BASE_DATA_RATIOS}")
    log(f"Tokens so far: {tokens_total:,}")
    log("-" * 70)

    for step in range(start_step, MAX_STEPS):
        if shutdown_requested:
            break

        # LR schedule
        current_lr = wsd_lr(step, MAX_STEPS, LR, WARMUP_STEPS, DECAY_FRACTION)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Get batch
        input_ids, labels = data.get_batch(BATCH_SIZE)
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward + backward (standard backprop)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

        if torch.isnan(loss):
            log(f"NaN loss at step {step}, skipping")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        losses.append(loss_val)
        rolling_loss.append(loss_val)
        tokens_total += BATCH_SIZE * SEQ_LEN

        # --- Hebbian plasticity update ---
        if (step + 1) % HEBBIAN_EVERY == 0:
            rolling_avg = sum(rolling_loss) / len(rolling_loss)
            current_avg = sum(list(rolling_loss)[-10:]) / min(10, len(rolling_loss))
            reward = max(0.0, rolling_avg - current_avg)

            da_level = model.worm.get_da_level()
            if da_level is not None:
                # Modulate Hebbian LR by reward signal
                effective_lr = HEBBIAN_LR * (1.0 + reward * 10.0)
                model.worm.hebbian_update(da_level, eta=effective_lr)

        # Log
        if (step + 1) % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(losses) / len(losses)
            tps = (step - start_step + 1) * BATCH_SIZE * SEQ_LEN / elapsed
            perplexity = math.exp(min(avg_loss, 20))
            hours = elapsed / 3600
            tokens_b = tokens_total / 1e9
            pct = (step + 1) / MAX_STEPS * 100

            log(f"step {step+1:>7d} | loss {avg_loss:.4f} | ppl {perplexity:.1f} | "
                f"lr {current_lr:.2e} | {tps:.0f} tok/s | "
                f"{tokens_b:.2f}B tokens ({pct:.1f}%) | {hours:.1f}h")
            log_metrics(step + 1, avg_loss, perplexity, current_lr, tps,
                        tokens_total, hours)

            # Log worm diagnostics
            diag = model.get_worm_diagnostics()
            if diag:
                log_worm_diagnostics(step + 1, tokens_total, diag)

        # Validation (uncompiled to avoid train/eval recompilation)
        if (step + 1) % VAL_EVERY == 0:
            with model.uncompiled():
                val_loss = val_set.evaluate(model, device)
            val_ppl = math.exp(min(val_loss, 20))
            improved = " *NEW BEST*" if val_loss < best_val_loss else ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            log(f"  VAL loss {val_loss:.4f} | ppl {val_ppl:.1f}{improved}")

            elapsed = time.time() - start_time
            avg_loss = sum(losses) / len(losses)
            tps = (step - start_step + 1) * BATCH_SIZE * SEQ_LEN / elapsed
            log_metrics(step + 1, avg_loss, math.exp(min(avg_loss, 20)),
                        current_lr, tps, tokens_total, elapsed / 3600,
                        val_loss=val_loss)

        # Generate samples
        if (step + 1) % SAMPLE_EVERY == 0:
            log("--- SAMPLE GENERATIONS ---")
            samples = generate_samples(model, tokenizer, SAMPLE_PROMPTS[:3], device)
            samples_dict = {}
            for i, sample in enumerate(samples):
                sample = sample[:300]
                log(f"  [{i+1}] {sample}")
                samples_dict[SAMPLE_PROMPTS[i]] = sample
            log("-" * 40)

            diag = model.get_worm_diagnostics()
            log_samples(step + 1, tokens_total, samples_dict, diag)

        # Checkpoint
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = sum(losses) / len(losses)
            save_checkpoint(
                model, optimizer, scaler, config, step + 1, avg_loss,
                tokens_total=tokens_total,
            )

        if shutdown_requested:
            break

    # Final save
    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    save_checkpoint(
        model, optimizer, scaler, config, step + 1, avg_loss,
        tokens_total=tokens_total,
    )

    elapsed = time.time() - start_time
    log("=" * 70)
    log(f"WORM SIDECAR TRAINING {'STOPPED' if shutdown_requested else 'COMPLETE'}")
    log(f"Final step: {step + 1} | Loss: {avg_loss:.4f} | Time: {elapsed/3600:.1f}h")
    log(f"Total tokens: {tokens_total:,} ({tokens_total/1e9:.2f}B)")
    log(f"Best val loss: {best_val_loss:.4f}")
    log("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Worm Brain Sidecar Training")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from specific checkpoint")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh (ignore existing checkpoints)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max steps")
    args = parser.parse_args()

    if args.steps is not None:
        MAX_STEPS = args.steps

    if args.checkpoint:
        train(resume_from=args.checkpoint)
    elif args.fresh:
        train(fresh=True)
    elif args.resume:
        train()  # auto-detects checkpoint
    else:
        train(fresh=True)  # default: fresh start
