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
from encoder_model import EncoderConfig, SemanticEncoder, nt_xent_loss

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
BATCH_SIZE = 256
PEAK_LR = 3e-4
WARMUP_STEPS = 1000
TOTAL_STEPS = 100_000
TEMPERATURE = 0.07
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0

# Logging
LOG_EVERY = 50
EVAL_EVERY = 500
SAMPLE_EVERY = 5000
CHECKPOINT_EVERY = 5000
UMAP_EVERY = 5000

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
    label=1 means paraphrase (positive pair), label=0 means non-paraphrase.
    We only use label=1 pairs for training (negatives come from in-batch).
    """

    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = []
        self._load_pairs()
        random.shuffle(self.pairs)
        self.idx = 0

    def _load_pairs(self):
        data_dir = Path("data/pairs")
        if not data_dir.exists():
            log(f"WARNING: {data_dir} not found")
            return

        for path in sorted(data_dir.glob("*.jsonl")):
            count = 0
            with open(path) as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        if doc.get("label", 1) == 1:  # positive pairs only
                            self.pairs.append((doc["text_a"], doc["text_b"]))
                            count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
            log(f"  Loaded {count:,} pairs from {path.name}")

        log(f"  Total training pairs: {len(self.pairs):,}")

    def get_batch(self, batch_size):
        """Get a batch of tokenized positive pairs."""
        texts_a = []
        texts_b = []

        for _ in range(batch_size):
            if self.idx >= len(self.pairs):
                random.shuffle(self.pairs)
                self.idx = 0
            a, b = self.pairs[self.idx]
            texts_a.append(a)
            texts_b.append(b)
            self.idx += 1

        enc_a = self.tokenizer(texts_a, max_length=self.max_len,
                               padding=True, truncation=True,
                               return_tensors="pt")
        enc_b = self.tokenizer(texts_b, max_length=self.max_len,
                               padding=True, truncation=True,
                               return_tensors="pt")

        return enc_a, enc_b


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
        "hard_neg_sim": safe_mean(hard_neg_sims),
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


@torch.no_grad()
def generate_umap(model, eval_pairs, tokenizer, step, device="cuda"):
    """Generate UMAP visualization of concept vectors."""
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model.eval()
    vectors = []
    labels = []

    # Collect vectors from eval pairs
    all_texts = []
    for source, pairs in eval_pairs.items():
        for text_a, text_b, label, pair_type in pairs[:250]:
            all_texts.append((text_a, source))
            all_texts.append((text_b, source))

    if len(all_texts) < 50:
        return

    for text, source in all_texts[:1000]:
        enc = tokenizer(text, max_length=128, padding=True,
                        truncation=True, return_tensors="pt").to(device)
        vec = model(enc["input_ids"], enc["attention_mask"])
        vectors.append(vec.cpu().numpy()[0])
        labels.append(source)

    vectors = np.array(vectors)
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = [l == label for l in labels]
        pts = embedding[mask]
        plt.scatter(pts[:, 0], pts[:, 1], c=[colors[i]], s=5, alpha=0.5, label=label)
    plt.legend(fontsize=8, markerscale=3)
    plt.title(f"Concept Vector UMAP — Step {step:,}")

    umap_dir = Path("logs/umap")
    umap_dir.mkdir(parents=True, exist_ok=True)
    path = umap_dir / f"step_{step:06d}.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  UMAP saved: {path}")
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
            log(f"  Cross-lingual:   {metrics['crosslingual_sim']:.4f} (target >0.90)")
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
    if len(dataset.pairs) == 0:
        log("ERROR: No training pairs found. Run build_pairs.py first.")
        return

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
    log(f"  Batch size: {BATCH_SIZE} | Temperature: {TEMPERATURE}")
    log(f"  Peak LR: {PEAK_LR} | Steps: {start_step} → {TOTAL_STEPS}")
    log(f"  Training pairs: {len(dataset.pairs):,}")
    log("-" * 70)

    for step in range(start_step, TOTAL_STEPS):
        if shutdown_requested:
            break

        # LR schedule
        current_lr = cosine_lr(step, TOTAL_STEPS, PEAK_LR, WARMUP_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Get batch
        enc_a, enc_b = dataset.get_batch(BATCH_SIZE)
        ids_a = enc_a["input_ids"].to(device)
        mask_a = enc_a["attention_mask"].to(device)
        ids_b = enc_b["input_ids"].to(device)
        mask_b = enc_b["attention_mask"].to(device)

        # Forward + loss
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            z_a = model(ids_a, mask_a)
            z_b = model(ids_b, mask_b)
            loss = nt_xent_loss(z_a, z_b, temperature=TEMPERATURE)

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

        # Log
        if (step + 1) % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            hours = elapsed / 3600
            pct = (step + 1) / TOTAL_STEPS * 100
            log(f"step {step+1:>7d} | loss {avg_loss:.4f} | "
                f"lr {current_lr:.2e} | {pct:.1f}% | {hours:.1f}h")

        # Eval geometry
        if (step + 1) % EVAL_EVERY == 0 and eval_pairs:
            metrics = eval_geometry(model, eval_pairs, tokenizer, device)
            log(f"  EVAL: para={metrics['para_sim']:.3f} "
                f"non={metrics['nonpara_sim']:.3f} "
                f"sep={metrics['sep_ratio']:.2f} "
                f"xlang={metrics['crosslingual_sim']:.3f} "
                f"hard={metrics['hard_neg_sim']:.3f}")
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

        # UMAP
        if (step + 1) % UMAP_EVERY == 0 and eval_pairs:
            generate_umap(model, eval_pairs, tokenizer, step + 1, device)

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
