#!/usr/bin/env python3
"""
FLM V30 — Autoregressive LM with Word2vec Input Space

Uses V28's frozen 300d word2vec embeddings as both input and output space.
The model "thinks" in word2vec geometry — input is looked up from the frozen
embedding matrix, output is a dot product against it.

This means the model's predictions live in the same space where
king - man + woman = queen. We can potentially prompt it with vector
arithmetic and see what it generates.

Architecture:
    word_ids → Frozen V28 Embeddings (300d)
             → Linear(300, 512) + Positional Encoding
             → 8-layer Causal Transformer (512d, 8 heads, SwiGLU)
             → Linear(512, 300)
             → dot product with V28 embedding matrix
             → softmax → next word prediction

Usage:
    python train_v30.py --fresh
    python train_v30.py --resume
"""

import os
import re
import json
import math
import time
import random
import signal
import datetime
import argparse
import queue
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_DIM = 300           # V28 word2vec dimension (frozen)
HIDDEN_DIM = 512          # Transformer hidden dimension
NUM_LAYERS = 8
NUM_HEADS = 8
FF_MULT = 4              # FFN multiplier (SwiGLU: 2/3 * 4 * hidden = 1365)
DROPOUT = 0.1
MAX_SEQ_LEN = 256         # Context window in tokens

BATCH_SIZE = 16
TOTAL_STEPS = 500_000
PEAK_LR = 3e-4
MIN_LR = 1e-5
WARMUP_STEPS = 2000
GRAD_ACCUM = 4            # Effective batch = 16 * 4 = 64
GRAD_CLIP = 1.0
LOG_EVERY = 50
EVAL_EVERY = 5000
SAVE_EVERY = 10000
GENERATE_EVERY = 2500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints/lm_v30"
LOG_DIR = "logs"
LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v30.log",
    "metrics": f"{LOG_DIR}/concept_v30_metrics.csv",
}

# V28 paths
V28_CHECKPOINT = "checkpoints/word2vec_v28/latest.pt"
V28_VOCAB = "checkpoints/word2vec_v28/vocab.json"

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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATHS["log"], "a") as f:
        f.write(line + "\n")

def log_metrics(step, metrics):
    path = LOG_PATHS["metrics"]
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(",".join(["step"] + list(metrics.keys())) + "\n")
    with open(path, "a") as f:
        vals = [str(step)] + [f"{v:.6f}" for v in metrics.values()]
        f.write(",".join(vals) + "\n")


# ---------------------------------------------------------------------------
# Vocabulary (reuse V28)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

def tokenize_text(text):
    return _WORD_RE.findall(text.lower())


class Vocabulary:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.counts = []
        self.total_count = 0
        self.vocab_size = 0

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.word2id = data["word2id"]
        self.id2word = {int(i): w for w, i in self.word2id.items()}
        if isinstance(data["counts"], dict):
            self.counts = [data["counts"][str(i)] for i in range(len(self.word2id))]
        else:
            self.counts = data["counts"]
        self.total_count = data["total_count"]
        self.vocab_size = len(self.word2id)
        log(f"  Vocabulary loaded: {self.vocab_size:,} words from {path}")

    def encode(self, words):
        return [self.word2id[w] for w in words if w in self.word2id]

    def decode(self, ids):
        return [self.id2word.get(i, "<unk>") for i in ids]

    def __len__(self):
        return self.vocab_size

    def __contains__(self, word):
        return word in self.word2id


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ff_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with pre-norm
        normed = self.ln1(x)
        T = normed.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)
        # FFN with pre-norm
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class Word2vecLM(nn.Module):
    """Autoregressive LM that operates in word2vec embedding space."""

    def __init__(self, frozen_embeddings, hidden_dim, num_layers, num_heads, ff_dim,
                 dropout, max_seq_len):
        super().__init__()
        vocab_size, embed_dim = frozen_embeddings.shape

        # Frozen word2vec embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.copy_(frozen_embeddings)
        self.embedding.weight.requires_grad = False

        # Normalized copy for output dot product
        self.register_buffer("embed_norm",
            F.normalize(frozen_embeddings, p=2, dim=-1))

        # Input projection: word2vec space → transformer space
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Positional encoding (learned)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(hidden_dim)

        # Output projection: transformer space → word2vec space
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

        # Temperature for output logits
        self.log_temp = nn.Parameter(torch.tensor(0.0))

        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len) — word IDs

        Returns logits: (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape

        # Look up frozen word2vec embeddings
        x = self.embedding(input_ids)          # (B, T, 300)

        # Project into transformer space
        x = self.input_proj(x)                 # (B, T, 512)

        # Add positional encoding
        positions = torch.arange(T, device=input_ids.device)
        x = x + self.pos_embed(positions)

        # Transformer layers (causal mask handled by is_causal=True)
        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)                  # (B, T, 512)

        # Project back to word2vec space
        w2v_out = self.output_proj(x)          # (B, T, 300)

        # Normalize output vector
        w2v_out = F.normalize(w2v_out, p=2, dim=-1)

        # Dot product with normalized embeddings → logits
        temp = self.log_temp.exp().clamp(min=0.01, max=10.0)
        logits = (w2v_out @ self.embed_norm.T) / temp  # (B, T, vocab_size)

        return logits

    def generate(self, prompt_ids, max_new_tokens=50, temperature=0.8, top_k=50):
        """Autoregressive generation."""
        self.eval()
        ids = prompt_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop to max_seq_len
                context = ids[:, -self.max_seq_len:]
                logits = self.forward(context)
                # Take logits for last position
                next_logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    topk_vals, _ = next_logits.topk(top_k, dim=-1)
                    threshold = topk_vals[:, -1].unsqueeze(-1)
                    next_logits[next_logits < threshold] = float('-inf')

                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, next_id], dim=1)

        return ids

    def generate_from_vector(self, vec, max_new_tokens=50, temperature=0.8, top_k=50):
        """Generate starting from an arbitrary word2vec vector (for arithmetic prompts)."""
        self.eval()
        vec = F.normalize(vec.unsqueeze(0).unsqueeze(0), p=2, dim=-1)  # (1, 1, 300)

        with torch.no_grad():
            # Project into transformer space
            x = self.input_proj(vec)
            x = x + self.pos_embed(torch.tensor([0], device=vec.device))

            for layer in self.layers:
                x = layer(x)

            x = self.ln_final(x)
            w2v_out = F.normalize(self.output_proj(x), p=2, dim=-1)

            # First token from vector
            logits = (w2v_out @ self.embed_norm.T) / (self.log_temp.exp().clamp(0.01, 10.0))
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                topk_vals, _ = logits.topk(top_k, dim=-1)
                logits[logits < topk_vals[:, -1:]] = float('-inf')
            first_id = torch.multinomial(F.softmax(logits, dim=-1), 1)

            # Continue autoregressively
            ids = first_id
            for _ in range(max_new_tokens - 1):
                context = ids[:, -self.max_seq_len:]
                full_logits = self.forward(context)
                next_logits = full_logits[:, -1, :] / temperature
                if top_k > 0:
                    topk_vals, _ = next_logits.topk(top_k, dim=-1)
                    next_logits[next_logits < topk_vals[:, -1:]] = float('-inf')
                next_id = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
                ids = torch.cat([ids, next_id], dim=1)

        return ids


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

class SequenceDataset:
    """Streams sequences of word IDs for language modeling."""

    def __init__(self, sources, vocab, seq_len=MAX_SEQ_LEN):
        self.sources = [(p, w) for p, w in sources if os.path.exists(p)]
        if not self.sources:
            raise FileNotFoundError("No data sources found")
        self.vocab = vocab
        self.seq_len = seq_len

        total_w = sum(w for _, w in self.sources)
        self._cum_weights = []
        cum = 0
        for _, w in self.sources:
            cum += w / total_w
            self._cum_weights.append(cum)
        self._files = [open(p) for p, _ in self.sources]

        # Token buffer: accumulate tokens, then slice into sequences
        self._token_buf = []
        self._MIN_BUF = seq_len * 200

        log(f"SequenceDataset: {len(self.sources)} sources, "
            f"vocab={len(vocab):,}, seq_len={seq_len}")

    def _sample_source(self):
        r = random.random()
        for i, cw in enumerate(self._cum_weights):
            if r <= cw:
                return i
        return len(self.sources) - 1

    def _read_doc(self, src_idx):
        f = self._files[src_idx]
        line = f.readline()
        if not line:
            f.close()
            self._files[src_idx] = open(self.sources[src_idx][0])
            f = self._files[src_idx]
            line = f.readline()
            if not line:
                return None
        try:
            return json.loads(line).get("text", "").strip()
        except (json.JSONDecodeError, KeyError):
            return None

    def _refill_buffer(self):
        attempts = 0
        while len(self._token_buf) < self._MIN_BUF and attempts < 20000:
            attempts += 1
            src = self._sample_source()
            text = self._read_doc(src)
            if not text or len(text) < 50:
                continue
            ids = self.vocab.encode(tokenize_text(text))
            if len(ids) >= 10:
                self._token_buf.extend(ids)

    def get_batch(self, batch_size):
        """Return (input, target) where target is input shifted by 1."""
        needed = batch_size * (self.seq_len + 1)
        while len(self._token_buf) < needed:
            self._refill_buffer()

        batch_input = []
        batch_target = []
        for _ in range(batch_size):
            # Take seq_len + 1 tokens
            chunk = self._token_buf[:self.seq_len + 1]
            self._token_buf = self._token_buf[self.seq_len + 1:]
            batch_input.append(chunk[:-1])
            batch_target.append(chunk[1:])

        return (torch.tensor(batch_input, dtype=torch.long),
                torch.tensor(batch_target, dtype=torch.long))


class PrefetchBuffer:
    def __init__(self, dataset, batch_size, buf_size=8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.q = queue.Queue(maxsize=buf_size)
        self._stop = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._fill, daemon=True)
        self._thread.start()

    def _fill(self):
        while not self._stop.is_set():
            try:
                batch = self.dataset.get_batch(self.batch_size)
                self.q.put(batch, timeout=1.0)
            except Exception:
                continue

    def get(self):
        return self.q.get()

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# Evaluation & Generation
# ---------------------------------------------------------------------------

def evaluate_generation(model, vocab, step):
    """Generate sample text and test vector arithmetic prompts."""
    model.eval()

    # Text prompts
    prompts = [
        "the king sat on his",
        "the weather was cold and",
        "she walked through the forest",
        "the computer program crashed because",
        "in the beginning there was",
    ]

    log(f"  --- Generation samples ---")
    for prompt_text in prompts:
        words = tokenize_text(prompt_text)
        ids = vocab.encode(words)
        if len(ids) < 2:
            continue
        input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        output_ids = model.generate(input_ids, max_new_tokens=30, temperature=0.8, top_k=50)
        output_words = vocab.decode(output_ids[0].cpu().tolist())
        log(f"    \"{prompt_text}\" → {' '.join(output_words[len(ids):])}")

    # Vector arithmetic prompts
    log(f"  --- Vector arithmetic generation ---")
    arithmetic_prompts = [
        ("king - man + woman", ["king", "-man", "+woman"]),
        ("paris - france + germany", ["paris", "-france", "+germany"]),
        ("happy - good + bad", ["happy", "-good", "+bad"]),
    ]

    emb = model.embedding.weight.data
    emb_norm = F.normalize(emb, p=2, dim=-1)

    for desc, terms in arithmetic_prompts:
        vec = None
        valid = True
        for term in terms:
            sign = 1
            word = term
            if term.startswith("-"):
                sign = -1
                word = term[1:]
            elif term.startswith("+"):
                word = term[1:]
            if word not in vocab:
                valid = False
                break
            v = emb_norm[vocab.word2id[word]]
            vec = v * sign if vec is None else vec + v * sign

        if valid and vec is not None:
            # What's nearest to this vector?
            vec_norm = F.normalize(vec.unsqueeze(0), p=2, dim=-1)
            sims = (vec_norm @ emb_norm.T).squeeze()
            top5 = sims.topk(5)
            nn_words = [f"{vocab.id2word[i.item()]}({s:.2f})" for i, s in zip(top5.indices, top5.values)]
            log(f"    {desc} → nearest: {', '.join(nn_words)}")

            # Generate from this vector
            vec_dev = vec.to(DEVICE)
            output_ids = model.generate_from_vector(vec_dev, max_new_tokens=20, temperature=0.8)
            output_words = vocab.decode(output_ids[0].cpu().tolist())
            log(f"    {desc} → generates: {' '.join(output_words)}")

    model.train()


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, step):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt = {
        "model_state_dict": {k: v for k, v in model.state_dict().items()
                             if "embedding.weight" not in k and "embed_norm" not in k},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "config": {
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "ff_mult": FF_MULT,
            "dropout": DROPOUT,
            "max_seq_len": MAX_SEQ_LEN,
            "embed_dim": EMBED_DIM,
        },
    }
    path = os.path.join(CHECKPOINT_DIR, f"step_{step:06d}.pt")
    torch.save(ckpt, path)
    latest = os.path.join(CHECKPOINT_DIR, "latest.pt")
    torch.save(ckpt, latest)
    log(f"  Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, scheduler):
    latest = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if not os.path.exists(latest):
        return 0
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    # Load only non-frozen params
    state = ckpt["model_state_dict"]
    model_state = model.state_dict()
    for k, v in state.items():
        if k in model_state:
            model_state[k] = v
    model.load_state_dict(model_state)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    step = ckpt["step"]
    log(f"Resumed V30: step {step}")
    return step


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step):
    if step < WARMUP_STEPS:
        return PEAK_LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(TOTAL_STEPS - WARMUP_STEPS, 1)
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    log("=" * 70)
    log("FLM V30 — Autoregressive LM with Word2vec Input Space")
    log("=" * 70)

    # Load vocabulary
    vocab = Vocabulary()
    vocab.load(V28_VOCAB)

    # Load frozen word2vec embeddings
    log("Loading V28 word2vec embeddings...")
    v28_ckpt = torch.load(V28_CHECKPOINT, map_location="cpu", weights_only=False)
    frozen_embeddings = v28_ckpt["model_state_dict"]["target_embeddings.weight"]
    log(f"  Embeddings: {frozen_embeddings.shape} (frozen)")

    # Compute FFN dim (SwiGLU uses 2/3 of standard FFN)
    ff_dim = int(2 * HIDDEN_DIM * FF_MULT / 3)
    log(f"  Hidden: {HIDDEN_DIM}, FFN: {ff_dim}, Layers: {NUM_LAYERS}, Heads: {NUM_HEADS}")
    log(f"  Context: {MAX_SEQ_LEN} tokens, Batch: {BATCH_SIZE}")

    # Model
    model = Word2vecLM(
        frozen_embeddings=frozen_embeddings,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ff_dim=ff_dim,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    log(f"  Total params: {total_params:,} ({total_params/1e6:.1f}M)")
    log(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    log(f"  Frozen (embeddings): {frozen_params:,} ({frozen_params/1e6:.1f}M)")

    # Optimizer — only trainable parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=0.1
    )

    # Load checkpoint
    start_step = 0
    if not args.fresh:
        start_step = load_checkpoint(model, optimizer, None)

    log(f"  LR: {PEAK_LR} → {MIN_LR} (cosine) | Steps: {start_step} → {TOTAL_STEPS}")

    # Data
    dataset = SequenceDataset(PRETRAIN_SOURCES, vocab, seq_len=MAX_SEQ_LEN)
    prefetch = PrefetchBuffer(dataset, BATCH_SIZE)
    prefetch.start()
    log("Prefetch started")
    log("-" * 70)

    # Training
    model.train()
    running_loss = 0.0
    loss_count = 0
    start_time = time.time()

    stop_flag = [False]
    def handle_signal(sig, frame):
        log(f"Signal {sig} received, saving and exiting...")
        stop_flag[0] = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    for step in range(start_step, TOTAL_STEPS):
        if stop_flag[0]:
            save_checkpoint(model, optimizer, None, step)
            break

        current_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Gradient accumulation
        accum_loss = 0.0
        for accum_step in range(GRAD_ACCUM):
            input_ids, target_ids = prefetch.get()
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)

            logits = model(input_ids)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, vocab.vocab_size),
                target_ids.view(-1),
            ) / GRAD_ACCUM

            loss.backward()
            accum_loss += loss.item()

        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()

        running_loss += accum_loss
        loss_count += 1

        if (step + 1) % LOG_EVERY == 0:
            avg_loss = running_loss / loss_count
            ppl = math.exp(min(avg_loss, 20))  # cap to avoid overflow
            elapsed = time.time() - start_time
            sps = (step + 1 - start_step) / max(elapsed, 1)
            pct = (step + 1) / TOTAL_STEPS * 100
            temp = model.log_temp.exp().item()

            log(f"step {step+1:>7d} [V30] | loss={avg_loss:.4f} ppl={ppl:.1f} | "
                f"temp={temp:.3f} | lr {current_lr:.2e} | {pct:.1f}% | {sps:.1f} step/s")

            log_metrics(step + 1, {
                "loss": avg_loss,
                "ppl": ppl,
                "lr": current_lr,
                "temp": temp,
            })

            running_loss = 0.0
            loss_count = 0

        if (step + 1) % GENERATE_EVERY == 0:
            with torch.no_grad():
                evaluate_generation(model, vocab, step + 1)

        if (step + 1) % EVAL_EVERY == 0:
            # Compute validation-like metrics
            pass

        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, None, step + 1)

    if not stop_flag[0]:
        save_checkpoint(model, optimizer, None, TOTAL_STEPS)

    prefetch.stop()
    log("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)
