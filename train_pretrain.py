"""
Hamner V3 Pretraining Script — DFSG-Compatible, Linux-Focused
=============================================================
SmolLM-135M architecture (30 layers × 576 hidden) with DFSG-compliant data
from Common Pile + existing Linux specialization data.

10B token test run with flat data mix (no staged ramp).

All data from data/pretrain/*.jsonl — no HuggingFace streaming.

Usage:
    python train_pretrain.py                     # fresh training
    python train_pretrain.py --resume            # resume from latest checkpoint
    python train_pretrain.py --checkpoint path   # resume from specific checkpoint
    python train_pretrain.py --steps 50000       # override max steps
    python train_pretrain.py --eval-only         # run LinuxBench eval only
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
from model import HamnerModel, HamnerConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/pretrain_v3"
LOG_DIR = "logs"

# Model config — SmolLM-135M architecture
MODEL_CONFIG = dict(
    vocab_size=49152,
    hidden_size=576,
    num_layers=30,
    num_attention_heads=9,
    num_kv_heads=3,
    num_experts=1,
    num_active_experts=1,
    expert_intermediate_size=1536,
    use_differential_attention=False,
    gradient_checkpointing=True,
    max_seq_len=2048,
    tie_word_embeddings=True,
)

# --- Hyperparameters (SmolLM recipe for 135M) ---
BATCH_SIZE = 32
SEQ_LEN = 2048
PEAK_LR = 1e-3
MIN_LR_RATIO = 0.1            # min LR = peak * 0.1
WARMUP_STEPS = 2000
DECAY_FRACTION = 0.10          # cosine decay in last 10%
MAX_STEPS = 152_588            # ~10B tokens at batch 32 * seq 2048

# AdamW
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# --- Logging ---
CHECKPOINT_EVERY = 2000
SAMPLE_EVERY = 1000
LOG_EVERY = 50
VAL_EVERY = 1000
EVAL_EVERY = 10_000            # LinuxBench eval frequency
KEEP_CHECKPOINTS = 10

SAMPLE_PROMPTS = [
    "The meaning of life is",
    "Once upon a time there was a",
    "To list all files in a directory, use the command",
    "The Linux kernel is",
    "def fibonacci(n):",
    "To configure a network interface on Debian, you can",
]

# ---------------------------------------------------------------------------
# Data sources — all from data/pretrain/*.jsonl
# ---------------------------------------------------------------------------

# Category assignments for each source file
DATA_CATEGORIES = {
    # General text (50%)
    "wikipedia":     "general",
    "gutenberg":     "general",
    # Technical Q&A (20%)
    "stackexchange": "technical",
    # Code (10%)
    "thestack":      "code",
    # Scientific (10%)
    "peS2o":         "science",
    "arxiv":         "science",
    # Linux specialization (3.6%)
    "manpages":      "linux",
    "kernel_docs":   "linux",
    "rfcs":          "linux",
    "archwiki":      "linux",
    "tldp":          "linux",
    "gnu_manuals":   "linux",
    "kernel_source": "linux",
    "foss_repos":    "linux",
    # Government/Legal (6.4%)
    "usgpo":         "government",
}

# Flat data mix ratios for 10B test run (no staged ramp)
STAGE_RATIOS = [
    # (start_fraction, general, technical, code, science, linux, government)
    (0.00, 0.50, 0.20, 0.10, 0.10, 0.036, 0.064),
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_PATHS = {
    "log": f"{LOG_DIR}/pretrain_v3.log",
    "metrics": f"{LOG_DIR}/pretrain_v3_metrics.csv",
    "samples": f"{LOG_DIR}/pretrain_v3_samples.jsonl",
}


def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    log_file = LOG_PATHS.get("log")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")


def log_metrics(step, loss, perplexity, lr, tokens_per_sec, tokens_total,
                elapsed_hours, val_loss=None, linux_bench_acc=None,
                stage_num=None):
    metrics_file = LOG_PATHS.get("metrics")
    if not metrics_file:
        return
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    write_header = not os.path.exists(metrics_file)
    with open(metrics_file, "a") as f:
        if write_header:
            f.write("timestamp,step,loss,perplexity,learning_rate,tokens_per_sec,"
                    "tokens_total,tokens_billions,elapsed_hours,val_loss,"
                    "linux_bench_acc,stage\n")
        ts = datetime.datetime.now().isoformat()
        tokens_b = tokens_total / 1e9
        val_str = f"{val_loss:.6f}" if val_loss is not None else ""
        lb_str = f"{linux_bench_acc:.4f}" if linux_bench_acc is not None else ""
        stage_str = str(stage_num) if stage_num is not None else ""
        f.write(f"{ts},{step},{loss:.6f},{perplexity:.2f},{lr:.6e},"
                f"{tokens_per_sec:.0f},{tokens_total},{tokens_b:.4f},"
                f"{elapsed_hours:.4f},{val_str},{lb_str},{stage_str}\n")


def log_samples(step, tokens_total, samples_dict):
    samples_file = LOG_PATHS.get("samples")
    if not samples_file:
        return
    os.makedirs(os.path.dirname(samples_file), exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "step": step,
        "tokens_total": tokens_total,
        "tokens_billions": round(tokens_total / 1e9, 4),
        "samples": samples_dict,
    }
    with open(samples_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Data: Local JSONL multi-source streaming
# ---------------------------------------------------------------------------

class LocalDataStreamer:
    """Streams from local JSONL files with evolving category ratios.

    All data comes from data/pretrain/*.jsonl files.
    Data mix evolves through 4 stages during training.
    Supports save/restore of file positions for resumable training.
    """

    def __init__(self, tokenizer, seq_len=2048, max_steps=400_000):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_steps = max_steps

        self.source_paths = {}  # name → Path
        self.source_files = {}  # name → open file handle
        self.source_sizes = {}  # name → file size (for weighting within category)
        self.token_buffers = {} # name → list of token ids
        self.current_stage = 0

        self._init_sources()

    def _init_sources(self):
        data_dir = Path("data/pretrain")
        for name, category in DATA_CATEGORIES.items():
            path = data_dir / f"{name}.jsonl"
            if path.exists() and path.stat().st_size > 0:
                self.source_paths[name] = path
                self.source_files[name] = open(path)
                self.source_sizes[name] = path.stat().st_size
                self.token_buffers[name] = []
                log(f"  Data source: {name} ({category}) — "
                    f"{path.stat().st_size / 1e6:.0f} MB")
            else:
                log(f"  WARNING: Missing data source: {path}")

        if not self.source_files:
            raise RuntimeError("No data sources found in data/pretrain/!")

        # Show category totals
        for cat in ["general", "technical", "code", "science", "linux", "government"]:
            cat_sources = [n for n, c in DATA_CATEGORIES.items()
                           if c == cat and n in self.source_files]
            cat_size = sum(self.source_sizes.get(n, 0) for n in cat_sources)
            log(f"  Category '{cat}': {len(cat_sources)} sources, "
                f"{cat_size / 1e6:.0f} MB")

    def _read_next_text(self, name):
        """Read next valid text from a source, looping at EOF."""
        f = self.source_files[name]
        while True:
            line = f.readline()
            if not line:
                # EOF — loop back to start
                f.seek(0)
                line = f.readline()
                if not line:
                    return ""
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                text = doc.get("text", "")
                if len(text) >= 50:
                    return text
            except json.JSONDecodeError:
                continue

    def get_state(self):
        """Save file positions and token buffers for resumable training."""
        state = {
            "file_positions": {},
            "token_buffers": {},
            "rng_python": random.getstate(),
        }
        for name, f in self.source_files.items():
            state["file_positions"][name] = f.tell()
            state["token_buffers"][name] = self.token_buffers.get(name, [])
        return state

    def set_state(self, state):
        """Restore file positions and token buffers from checkpoint."""
        if not state:
            return
        positions = state.get("file_positions", {})
        buffers = state.get("token_buffers", {})
        for name in self.source_files:
            if name in positions:
                self.source_files[name].seek(positions[name])
                log(f"  Restored {name} position: byte {positions[name]:,}")
            if name in buffers:
                self.token_buffers[name] = buffers[name]
        if "rng_python" in state:
            random.setstate(state["rng_python"])

    def get_stage_ratios(self, step):
        """Get category ratios for current training progress."""
        progress = step / self.max_steps

        # Find which stage we're in
        stage_idx = 0
        for i, (start_frac, *_) in enumerate(STAGE_RATIOS):
            if progress >= start_frac:
                stage_idx = i

        ratios = STAGE_RATIOS[stage_idx]
        _, general_r, technical_r, code_r, science_r, linux_r, government_r = ratios

        if stage_idx != self.current_stage:
            self.current_stage = stage_idx
            log(f"  DATA MIX: general={general_r:.0%} technical={technical_r:.0%} "
                f"code={code_r:.0%} science={science_r:.0%} "
                f"linux={linux_r:.1%} government={government_r:.1%}")

        return {
            "general": general_r, "technical": technical_r,
            "code": code_r, "science": science_r,
            "linux": linux_r, "government": government_r,
        }, stage_idx + 1

    def _choose_source(self, category_ratios):
        """Choose a data source based on category ratios and source sizes."""
        # First pick a category
        r = random.random()
        cumulative = 0.0
        chosen_cat = "general"
        for cat, ratio in category_ratios.items():
            cumulative += ratio
            if r < cumulative:
                chosen_cat = cat
                break

        # Then pick a source within that category, weighted by file size
        cat_sources = [n for n, c in DATA_CATEGORIES.items()
                       if c == chosen_cat and n in self.source_files]

        if not cat_sources:
            # Fallback to any available source
            cat_sources = list(self.source_files.keys())

        # Weight by file size
        weights = [self.source_sizes.get(n, 1) for n in cat_sources]
        total = sum(weights)
        weights = [w / total for w in weights]

        r = random.random()
        cumulative = 0.0
        for name, w in zip(cat_sources, weights):
            cumulative += w
            if r < cumulative:
                return name
        return cat_sources[-1]

    def get_batch(self, batch_size, step=0):
        """Get a batch of tokenized sequences."""
        category_ratios, stage_num = self.get_stage_ratios(step)

        input_ids = []
        for _ in range(batch_size):
            source = self._choose_source(category_ratios)
            buf = self.token_buffers[source]

            while len(buf) < self.seq_len:
                text = self._read_next_text(source)
                if not text:
                    break
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                tokens.append(self.tokenizer.eos_token_id or 0)
                buf.extend(tokens)
                self.token_buffers[source] = buf

            chunk = buf[:self.seq_len]
            self.token_buffers[source] = buf[self.seq_len:]

            input_ids.append(torch.tensor(chunk, dtype=torch.long))

        return torch.stack(input_ids), torch.stack(input_ids), stage_num


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationSet:
    """Held-out validation set from Wikipedia for tracking generalization."""

    def __init__(self, tokenizer, seq_len=2048, n_batches=5, batch_size=8):
        self.batches = []
        log("Building validation set from Wikipedia...")

        wiki_path = Path("data/pretrain/wikipedia.jsonl")
        if not wiki_path.exists():
            log("  WARNING: No wikipedia.jsonl for validation, skipping")
            return

        # Use the last ~5000 lines of wikipedia as validation
        lines = []
        with open(wiki_path) as f:
            for i, line in enumerate(f):
                if i >= 50000:  # Skip first 50K for training
                    lines.append(line.strip())
                if len(lines) >= 5000:
                    break

        buf = []
        line_idx = 0
        for _ in range(n_batches):
            batch_ids = []
            for _ in range(batch_size):
                while len(buf) < seq_len:
                    if line_idx >= len(lines):
                        line_idx = 0
                    try:
                        doc = json.loads(lines[line_idx])
                        text = doc.get("text", "")
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        tokens.append(tokenizer.eos_token_id or 0)
                        buf.extend(tokens)
                    except Exception:
                        pass
                    line_idx += 1

                chunk = buf[:seq_len]
                buf = buf[seq_len:]
                batch_ids.append(torch.tensor(chunk, dtype=torch.long))
            self.batches.append(torch.stack(batch_ids))

        log(f"  Validation set: {len(self.batches)} batches of {batch_size}")

    @torch.no_grad()
    def evaluate(self, model, device="cuda"):
        if not self.batches:
            return None
        model.eval()
        total_loss = 0.0
        for batch in self.batches:
            batch = batch.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(batch, labels=batch)
                total_loss += outputs["loss"].item()
        model.train()
        return total_loss / len(self.batches)


# ---------------------------------------------------------------------------
# LinuxBench evaluation during training
# ---------------------------------------------------------------------------

class LinuxBenchEvaluator:
    """Lightweight LinuxBench evaluator for periodic checks during training."""

    def __init__(self, tokenizer, n_questions=200, device="cuda"):
        self.tokenizer = tokenizer
        self.device = device
        self.questions = []
        self.max_len = 2048
        self.pad_id = tokenizer.pad_token_id or 0

        bench_path = Path("eval/linux_bench.json")
        if not bench_path.exists():
            log("  WARNING: eval/linux_bench.json not found, LinuxBench disabled")
            return

        with open(bench_path) as f:
            all_questions = json.load(f)

        # Use a fixed subset for consistency across runs
        random.seed(42)
        self.questions = random.sample(all_questions, min(n_questions, len(all_questions)))
        random.seed()  # re-randomize
        log(f"  LinuxBench: loaded {len(self.questions)} questions for periodic eval")

    @torch.no_grad()
    def evaluate(self, model):
        """Run LinuxBench evaluation, return accuracy."""
        if not self.questions:
            return None

        model.eval()
        correct = 0
        total = 0
        batch_size = 32

        for i in range(0, len(self.questions), batch_size):
            batch_q = self.questions[i:i + batch_size]

            prompts = []
            answers = []
            for q in batch_q:
                # Format: question + choices, predict answer token
                prompt = q["question"] + "\n"
                for key, val in q["choices"].items():
                    prompt += f"  {key}. {val}\n"
                prompt += "Answer:"
                prompts.append(prompt)
                answers.append(q["answer"])

            # Tokenize and batch
            encoded = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
            max_len = min(max(len(e) for e in encoded), self.max_len)

            # Left-pad
            input_ids = torch.full((len(encoded), max_len), self.pad_id,
                                   dtype=torch.long, device=self.device)
            attn_mask = torch.zeros((len(encoded), max_len),
                                    dtype=torch.long, device=self.device)
            for j, ids in enumerate(encoded):
                ids = ids[-max_len:]
                input_ids[j, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
                attn_mask[j, max_len - len(ids):] = 1

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids, attention_mask=attn_mask)

            last_logits = outputs["logits"][:, -1, :]
            log_probs = F.log_softmax(last_logits.float(), dim=-1)

            # Score each answer option (A, B, C, D)
            label_tokens = ["A", "B", "C", "D"]
            label_ids = [self.tokenizer.encode(l, add_special_tokens=False)[0]
                         for l in label_tokens]

            for j in range(len(batch_q)):
                scores = {l: log_probs[j, tid].item()
                          for l, tid in zip(label_tokens, label_ids)}
                predicted = max(scores, key=scores.get)
                if predicted == answers[j]:
                    correct += 1
                total += 1

        model.train()
        return correct / total if total > 0 else 0.0


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
    if ckpts:
        return str(ckpts[-1])
    return None


def save_checkpoint(model, optimizer, scaler, config, step, loss,
                    checkpoint_dir, tokens_total=0, extra=None,
                    data_streamer=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
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
        "training_type": "pretrain_v3",
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    if data_streamer is not None:
        ckpt_data["data_streamer_state"] = data_streamer.get_state()
    if extra:
        ckpt_data.update(extra)

    ckpt_path = os.path.join(checkpoint_dir, f"step_{step:07d}.pt")
    torch.save(ckpt_data, ckpt_path)
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(ckpt_data, latest_path)
    log(f"  Checkpoint saved: {ckpt_path}")

    # Cleanup: keep milestones (every 10k) + last N
    all_ckpts = sorted(Path(checkpoint_dir).glob("step_*.pt"))
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
    model = HamnerModel(config).to(device)

    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    scaler = torch.amp.GradScaler("cuda")
    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    step = ckpt.get("step", 0)
    loss = ckpt.get("avg_loss", float("inf"))
    tokens_total = ckpt.get("tokens_total", 0)

    # Restore RNG states if available
    if "torch_rng_state" in ckpt:
        torch.random.set_rng_state(ckpt["torch_rng_state"])
    if "cuda_rng_state" in ckpt and ckpt["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(ckpt["cuda_rng_state"])

    data_state = ckpt.get("data_streamer_state", None)

    total_p, _ = model.count_parameters()
    log(f"Resumed: {total_p:,} params | step {step} | loss {loss:.4f} | "
        f"tokens {tokens_total/1e9:.2f}B")

    return model, optimizer, scaler, config, step, tokens_total, data_state


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(model, tokenizer, prompts, device="cuda", max_tokens=80):
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
# LR Schedule: Warmup-Stable-Decay (WSD)
# ---------------------------------------------------------------------------

def wsd_lr(step, max_steps, peak_lr, warmup_steps, decay_fraction):
    """Warmup-Stable-Decay schedule (SmolLM2 style).
    - Linear warmup for warmup_steps
    - Stable at peak_lr for most of training
    - Cosine decay in the last decay_fraction of training
    - Min LR = peak_lr * MIN_LR_RATIO
    """
    min_lr = peak_lr * MIN_LR_RATIO
    decay_start = int(max_steps * (1 - decay_fraction))

    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    elif step < decay_start:
        return peak_lr
    else:
        progress = (step - decay_start) / max(1, max_steps - decay_start)
        return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False, max_steps_override=None,
          eval_only=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps = max_steps_override or MAX_STEPS

    log("=" * 70)
    log("HAMNER V3 PRETRAINING — SmolLM-135M Architecture, Common Pile Data")
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
    data_state = None
    if resume_from:
        model, optimizer, scaler, config, start_step, tokens_total, data_state = \
            load_checkpoint(resume_from, device)
        config.vocab_size = tokenizer.vocab_size
    else:
        log("Starting fresh training...")
        config = HamnerConfig(**MODEL_CONFIG)
        model = HamnerModel(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=PEAK_LR, betas=BETAS, weight_decay=WEIGHT_DECAY
        )
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0
        tokens_total = 0

        total_p, _ = model.count_parameters()
        log(f"Model: {total_p:,} params | {total_p*2/1e9:.2f}GB fp16")

    # LinuxBench evaluator
    linux_bench = LinuxBenchEvaluator(tokenizer, n_questions=200, device=device)

    # Eval-only mode
    if eval_only:
        log("Running LinuxBench evaluation only...")
        acc = linux_bench.evaluate(model)
        if acc is not None:
            log(f"LinuxBench accuracy: {acc:.1%} ({acc*200:.0f}/200)")
        else:
            log("LinuxBench not available")
        return

    # Compile for speed
    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile (first step will be slow)...")
        model = torch.compile(model)

    model.train()

    # Data stream
    log("Initializing data sources...")
    data = LocalDataStreamer(tokenizer, seq_len=SEQ_LEN, max_steps=max_steps)
    if data_state:
        log("Restoring data stream position from checkpoint...")
        data.set_state(data_state)

    # Validation set
    val_set = ValidationSet(tokenizer, seq_len=SEQ_LEN, n_batches=5,
                            batch_size=BATCH_SIZE)

    # Training state
    losses = []
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

    # Print training plan
    tokens_per_step = BATCH_SIZE * SEQ_LEN
    total_tokens_target = max_steps * tokens_per_step
    log(f"\nTraining plan:")
    log(f"  Model: {config.hidden_size}h × {config.num_layers}L × "
        f"{config.num_attention_heads}heads")
    log(f"  Batch size: {BATCH_SIZE} | Seq len: {SEQ_LEN} | "
        f"Tokens/step: {tokens_per_step:,}")
    log(f"  Peak LR: {PEAK_LR} (WSD: warmup {WARMUP_STEPS}, "
        f"decay last {DECAY_FRACTION*100:.0f}%)")
    log(f"  Steps: {start_step} → {max_steps}")
    log(f"  Target: ~{total_tokens_target / 1e9:.1f}B new tokens")
    log(f"  Tokens so far: {tokens_total:,} ({tokens_total/1e9:.2f}B)")
    for i, (start_frac, gen, tech, code, sci, linux, gov) in enumerate(STAGE_RATIOS):
        step_start = int(start_frac * max_steps)
        log(f"  Stage {i+1} (step {step_start:,}): "
            f"gen={gen:.0%} tech={tech:.0%} code={code:.0%} "
            f"sci={sci:.0%} linux={linux:.1%} gov={gov:.1%}")
    log("-" * 70)

    for step in range(start_step, max_steps):
        if shutdown_requested:
            break

        # LR schedule
        current_lr = wsd_lr(step, max_steps, PEAK_LR, WARMUP_STEPS,
                            DECAY_FRACTION)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Get batch
        input_ids, labels, stage_num = data.get_batch(BATCH_SIZE, step=step)
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward + backward
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
        tokens_total += tokens_per_step

        # Log
        if (step + 1) % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            tps = (step - start_step + 1) * tokens_per_step / elapsed
            perplexity = math.exp(min(avg_loss, 20))
            hours = elapsed / 3600
            tokens_b = tokens_total / 1e9
            pct = (step + 1) / max_steps * 100

            log(f"step {step+1:>7d} [S{stage_num}] | loss {avg_loss:.4f} | "
                f"ppl {perplexity:.1f} | lr {current_lr:.2e} | {tps:.0f} tok/s | "
                f"{tokens_b:.2f}B tokens ({pct:.1f}%) | {hours:.1f}h")
            log_metrics(step + 1, avg_loss, perplexity, current_lr, tps,
                        tokens_total, hours, stage_num=stage_num)

        # Validation
        if (step + 1) % VAL_EVERY == 0:
            val_loss = val_set.evaluate(model, device)
            if val_loss is not None:
                val_ppl = math.exp(min(val_loss, 20))
                improved = " *NEW BEST*" if val_loss < best_val_loss else ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                log(f"  VAL loss {val_loss:.4f} | ppl {val_ppl:.1f}{improved}")

                elapsed = time.time() - start_time
                avg_loss = sum(losses[-100:]) / len(losses[-100:])
                tps = (step - start_step + 1) * tokens_per_step / elapsed
                log_metrics(step + 1, avg_loss, math.exp(min(avg_loss, 20)),
                            current_lr, tps, tokens_total, elapsed / 3600,
                            val_loss=val_loss, stage_num=stage_num)

        # LinuxBench evaluation
        if (step + 1) % EVAL_EVERY == 0:
            log("--- LINUXBENCH EVALUATION ---")
            acc = linux_bench.evaluate(model)
            if acc is not None:
                log(f"  LinuxBench: {acc:.1%} ({acc*200:.0f}/200) "
                    f"[baseline: 25%]")
                elapsed = time.time() - start_time
                avg_loss = sum(losses[-100:]) / len(losses[-100:])
                tps = (step - start_step + 1) * tokens_per_step / elapsed
                log_metrics(step + 1, avg_loss, math.exp(min(avg_loss, 20)),
                            current_lr, tps, tokens_total, elapsed / 3600,
                            linux_bench_acc=acc, stage_num=stage_num)

        # Generate samples
        if (step + 1) % SAMPLE_EVERY == 0:
            log("--- SAMPLE GENERATIONS ---")
            samples = generate_samples(model, tokenizer,
                                       SAMPLE_PROMPTS[:4], device)
            samples_dict = {}
            for i, sample in enumerate(samples):
                sample = sample[:400]
                log(f"  [{i+1}] {sample}")
                samples_dict[SAMPLE_PROMPTS[i]] = sample
            log("-" * 40)
            log_samples(step + 1, tokens_total, samples_dict)

        # Checkpoint
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            save_checkpoint(
                model, optimizer, scaler, config, step + 1, avg_loss,
                CHECKPOINT_DIR, tokens_total=tokens_total,
                data_streamer=data,
            )

        if shutdown_requested:
            break

    # Final save
    avg_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else float("inf")
    save_checkpoint(
        model, optimizer, scaler, config, step + 1, avg_loss,
        CHECKPOINT_DIR, tokens_total=tokens_total,
        data_streamer=data,
    )

    # Final LinuxBench
    log("--- FINAL LINUXBENCH EVALUATION ---")
    acc = linux_bench.evaluate(model)
    if acc is not None:
        log(f"  LinuxBench: {acc:.1%}")

    elapsed = time.time() - start_time
    log("=" * 70)
    log(f"PRETRAINING V3 {'STOPPED' if shutdown_requested else 'COMPLETE'}")
    log(f"Final step: {step + 1} | Loss: {avg_loss:.4f} | Time: {elapsed/3600:.1f}h")
    log(f"Total tokens: {tokens_total:,} ({tokens_total/1e9:.2f}B)")
    log(f"Best val loss: {best_val_loss:.4f}")
    if acc is not None:
        log(f"Final LinuxBench: {acc:.1%}")
    log("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hamner V3 Pretraining")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from specific checkpoint")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh (ignore existing checkpoints)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max steps")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run LinuxBench evaluation only")
    args = parser.parse_args()

    if args.checkpoint:
        train(resume_from=args.checkpoint, max_steps_override=args.steps,
              eval_only=args.eval_only)
    elif args.fresh:
        train(fresh=True, max_steps_override=args.steps,
              eval_only=args.eval_only)
    elif args.resume:
        train(max_steps_override=args.steps, eval_only=args.eval_only)
    else:
        train(fresh=True, max_steps_override=args.steps,
              eval_only=args.eval_only)
