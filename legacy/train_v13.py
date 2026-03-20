"""
FLM V13 — Dual-Decoder Concept Autoencoder (EN→concepts→EN + EN→concepts→FR)
================================================================================
Key change from V12:
  Two parallel decoders share the same 32×32 concept bottleneck:
    1. EN decoder: reconstructs original English (same as V12)
    2. FR decoder: translates to French from the same concepts

  This forces language-independent meaning encoding because:
    - The FR decoder can't rely on English surface tokens
    - French word order differs from English
    - The shared bottleneck must satisfy BOTH decoders
    - Can't be bag-of-words: different languages have different bags

  Warm-starts encoder + bottleneck + EN decoder from V12.
  FR decoder starts fresh.

Usage:
    python train_v13.py                    # auto-resume V13 or warm-start from V12
    python train_v13.py --fresh            # start from scratch
    python train_v13.py --from-v12 PATH    # warm-start from specific V12 checkpoint
    python train_v13.py --eval-only        # diagnostics only
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
from concept_model import (ConceptConfig, ConceptAutoencoderV13,
                           reconstruction_loss, translation_loss,
                           flat_similarity_matrix)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v13"
LOG_DIR = "logs"

MODEL_CONFIG = dict(
    vocab_size=30522,       # BERT tokenizer (EN encoder + EN decoder)
    fr_vocab_size=32005,    # CamemBERT tokenizer (FR decoder)
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
BATCH_SIZE = 48            # slightly smaller — 2 decoders on translation steps
PEAK_LR = 2e-4            # lower than V12's 3e-4 since warm-starting
MIN_LR = 1e-5
WARMUP_STEPS = 2000
TOTAL_STEPS = 600_000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0

# Dual-decoder config
LAMBDA_FR = 1.0            # FR translation loss weight

# EMA tracking
EXACT_MATCH_EMA_DECAY = 0.9

# Logging
LOG_EVERY = 50
EVAL_EVERY = 500
CHECKPOINT_EVERY = 5000

LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v13.log",
    "metrics": f"{LOG_DIR}/concept_v13_metrics.csv",
}

# Diagnostic sentences for EN reconstruction spot-checks
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
    # Code
    "def fibonacci ( n ) : return n if n < 2 else fibonacci ( n - 1 ) + fibonacci ( n - 2 )",
    # Math
    "the derivative of x squared plus three x equals two x plus three",
    # Logic
    "if all dogs are animals and all animals breathe then all dogs breathe",
]

# Diagnostic sentences for FR translation spot-checks
TRANSLATION_TEST_PAIRS = [
    ("Resumption of the session", "Reprise de la session"),
    ("I declare resumed the session", "Je déclare reprise la session"),
    ("The vote will take place tomorrow", "Le vote aura lieu demain"),
    ("I would like to thank the Commission", "Je voudrais remercier la Commission"),
    ("The situation is very serious", "La situation est très grave"),
    ("We need to find a solution", "Nous devons trouver une solution"),
    ("The European Parliament has decided", "Le Parlement européen a décidé"),
    ("This is a very important issue", "C' est une question très importante"),
]

# ---------------------------------------------------------------------------
# Geometry probing sentences (same as V12)
# ---------------------------------------------------------------------------

GEOMETRY_ANALOGIES = [
    ("she ran", "she runs", "he walked", "he walks", "tense"),
    ("the movie was great", "the movie was terrible", "the food was great", "the food was terrible", "sentiment"),
    ("she is not here", "she is here", "he is not there", "he is there", "negation"),
    ("the cats", "the cat", "the dogs", "the dog", "plural"),
    ("he is happy", "she is happy", "he is tired", "she is tired", "subject"),
    ("the huge house", "the tiny house", "the huge tree", "the tiny tree", "size"),
]

GEOMETRY_CLUSTERS = {
    "animals": ["the cat sat on the mat", "a dog ran in the park", "the bird flew over the tree",
                "fish swim in the ocean", "the horse galloped fast", "a rabbit hid in the grass"],
    "weather": ["it is raining heavily today", "the sun is shining brightly", "snow covered the ground",
                "a storm is approaching fast", "the wind blew the leaves", "fog rolled in at dawn"],
    "food": ["she cooked a delicious pasta", "the pizza was freshly baked", "he ate a bowl of rice",
             "the soup was too salty", "they ordered sushi for dinner", "she baked a chocolate cake"],
    "emotions": ["she was very happy today", "he felt sad and lonely", "the news made them angry",
                 "they were excited about it", "she felt anxious before", "he was proud of his work"],
    "technology": ["the computer crashed again", "she updated her phone", "the internet was slow today",
                   "he wrote a python program", "the server went down", "she debugged the code"],
    "music": ["she played the piano well", "he strummed the guitar", "the orchestra performed beautifully",
              "they sang a duet together", "the drums kept the beat", "she composed a new song"],
    "sports": ["he kicked the ball hard", "she ran the marathon", "the team won the game",
               "they scored in the last minute", "the coach praised the players", "she swam fifty laps"],
    "travel": ["they flew to paris last week", "the train arrived on time", "she drove across the country",
               "he packed his bags quickly", "the ship sailed at dawn", "we hiked up the mountain"],
}

GEOMETRY_DIRECTIONS = {
    "negation": [("the cat is not here", "the cat is here"), ("she did not run", "she did run"),
                 ("he is not happy", "he is happy"), ("they are not coming", "they are coming"),
                 ("it was not ready", "it was ready"), ("we did not agree", "we did agree")],
    "tense": [("she ran quickly", "she runs quickly"), ("he walked home", "he walks home"),
              ("they played outside", "they play outside"), ("it rained all day", "it rains all day"),
              ("she danced well", "she dances well"), ("he cooked dinner", "he cooks dinner")],
    "sentiment": [("the movie was great", "the movie was terrible"), ("she loves the food", "she hates the food"),
                  ("a wonderful day", "a horrible day"), ("he is kind", "he is cruel"),
                  ("the news was good", "the news was bad"), ("she enjoyed the book", "she disliked the book")],
    "plurality": [("the cats sat", "the cat sat"), ("the dogs ran", "the dog ran"),
                  ("the birds flew", "the bird flew"), ("the cars drove", "the car drove"),
                  ("the boys played", "the boy played"), ("the trees fell", "the tree fell")],
    "size": [("the big cat", "the small cat"), ("a large dog", "a small dog"),
             ("the huge house", "the tiny house"), ("a massive tree", "a little tree"),
             ("the big car", "the small car"), ("a large book", "a small book")],
}

GEOMETRY_WORD_ORDER = [
    ("the dog bit the man", "the man bit the dog"),
    ("alice likes bob", "bob likes alice"),
    ("she gave him a book", "he gave her a book"),
    ("the cat chased the mouse", "the mouse chased the cat"),
    ("the teacher praised the student", "the student praised the teacher"),
    ("the boy pushed the girl", "the girl pushed the boy"),
    ("she told him a secret", "he told her a secret"),
    ("the car hit the truck", "the truck hit the car"),
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


METRICS_HEADER = ("timestamp,step,recon_loss,token_acc,exact_match,"
                  "acc_short,acc_med,acc_long,em_short,em_med,em_long,"
                  "lr,elapsed_hours,"
                  "analogy_avg,clustering_gap,direction_consistency,"
                  "word_order_sim,effective_rank90,effective_rank95,"
                  "fr_loss,fr_token_acc\n")


def log_metrics(step, recon_loss, token_acc, exact_match,
                acc_short, acc_med, acc_long, em_short, em_med, em_long,
                lr, elapsed_hours, geo=None, fr_loss=0.0, fr_token_acc=0.0):
    metrics_file = LOG_PATHS.get("metrics")
    if not metrics_file:
        return
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    write_header = not os.path.exists(metrics_file)
    with open(metrics_file, "a") as f:
        if write_header:
            f.write(METRICS_HEADER)
        ts = datetime.datetime.now().isoformat()
        g = geo or {}
        f.write(f"{ts},{step},{recon_loss:.6f},{token_acc:.4f},{exact_match:.4f},"
                f"{acc_short:.4f},{acc_med:.4f},{acc_long:.4f},"
                f"{em_short:.4f},{em_med:.4f},{em_long:.4f},"
                f"{lr:.6e},{elapsed_hours:.4f},"
                f"{g.get('analogy_avg', 0):.4f},{g.get('clustering_gap', 0):.4f},"
                f"{g.get('direction_consistency', 0):.4f},"
                f"{g.get('word_order_sim', 0):.4f},"
                f"{g.get('rank90', 0)},{g.get('rank95', 0)},"
                f"{fr_loss:.6f},{fr_token_acc:.4f}\n")


# ---------------------------------------------------------------------------
# Data loading — EN↔FR translation pairs (sole data source for V13)
# ---------------------------------------------------------------------------

class TranslationDataset:
    """Loads EN↔FR pairs from europarl for dual-decoder training."""

    def __init__(self, en_tokenizer, fr_tokenizer, max_len=128):
        self.en_tokenizer = en_tokenizer
        self.fr_tokenizer = fr_tokenizer
        self.max_len = max_len
        self.pairs = []  # list of (en_text, fr_text)
        self._load()

    def _load(self):
        # Load all EN→FR sources
        sources = ["europarl.jsonl", "tatoeba_enfr.jsonl", "wikimatrix_enfr.jsonl"]
        for src in sources:
            path = Path(f"data/pairs/{src}")
            if not path.exists():
                log(f"  {src}: not found, skipping")
                continue
            count = 0
            with open(path) as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        if doc.get("label") != 1:
                            continue
                        en = doc.get("text_a", "").strip()
                        fr = doc.get("text_b", "").strip()
                        if len(en) > 10 and len(fr) > 10:
                            self.pairs.append((en, fr))
                            count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
            log(f"  {src}: {count:,} pairs")
        random.shuffle(self.pairs)
        log(f"  Translation pairs total: {len(self.pairs):,}")

    def get_batch(self, batch_size):
        """Returns (en_encoding, fr_encoding) for a random batch."""
        indices = [random.randint(0, len(self.pairs) - 1) for _ in range(batch_size)]
        en_texts = [self.pairs[i][0] for i in indices]
        fr_texts = [self.pairs[i][1] for i in indices]

        en_enc = self.en_tokenizer(en_texts, max_length=self.max_len,
                                   padding=True, truncation=True,
                                   return_tensors="pt")
        fr_enc = self.fr_tokenizer(fr_texts, max_length=self.max_len,
                                   padding=True, truncation=True,
                                   return_tensors="pt")
        return en_enc, fr_enc


class PrefetchBuffer:
    """Pre-tokenizes batches in background thread."""

    def __init__(self, dataset, device, batch_size=48, buf_size=4, name="prefetch"):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.q = queue.Queue(maxsize=buf_size)
        self._stop = threading.Event()
        self._thread = None
        self._name = name

    def start(self):
        self._thread = threading.Thread(target=self._fill, daemon=True,
                                        name=self._name)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _fill(self):
        while not self._stop.is_set():
            try:
                batch = self.dataset.get_batch(self.batch_size)
                self.q.put(batch, timeout=1.0)
            except queue.Full:
                continue

    def get(self):
        return self.q.get()


# ---------------------------------------------------------------------------
# Evaluation — EN reconstruction
# ---------------------------------------------------------------------------

def _unwrap(model):
    return model._orig_mod if hasattr(model, '_orig_mod') else model


@torch.no_grad()
def evaluate_reconstruction(model, tokenizer, device="cuda"):
    """Evaluate EN reconstruction on fixed diagnostic sentences."""
    model.eval()
    m = _unwrap(model)

    enc = tokenizer(RECON_TEST_SENTENCES, max_length=128, padding=True,
                    truncation=True, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    concepts = m.encode(input_ids, attention_mask)
    logits = m.decode_parallel(concepts, seq_len=input_ids.shape[1],
                               attention_mask=attention_mask)
    predicted = logits.argmax(dim=-1)

    results = []
    total_correct = 0
    total_tokens = 0
    total_exact = 0

    for i, text in enumerate(RECON_TEST_SENTENCES):
        mask = attention_mask[i].bool()
        tgt = input_ids[i][mask]
        pred = predicted[i][mask]
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
def evaluate_translation(model, en_tokenizer, fr_tokenizer, device="cuda"):
    """Evaluate FR translation on fixed test pairs."""
    model.eval()
    m = _unwrap(model)

    en_texts = [p[0] for p in TRANSLATION_TEST_PAIRS]
    fr_texts = [p[1] for p in TRANSLATION_TEST_PAIRS]

    en_enc = en_tokenizer(en_texts, max_length=128, padding=True,
                          truncation=True, return_tensors="pt").to(device)
    fr_enc = fr_tokenizer(fr_texts, max_length=128, padding=True,
                          truncation=True, return_tensors="pt").to(device)

    concepts = m.encode(en_enc["input_ids"], en_enc["attention_mask"])
    fr_logits = m.decode_parallel_fr(concepts, seq_len=fr_enc["input_ids"].shape[1],
                                     attention_mask=fr_enc["attention_mask"])
    fr_predicted = fr_logits.argmax(dim=-1)

    results = []
    total_correct = 0
    total_tokens = 0

    for i in range(len(TRANSLATION_TEST_PAIRS)):
        mask = fr_enc["attention_mask"][i].bool()
        tgt = fr_enc["input_ids"][i][mask]
        pred = fr_predicted[i][mask]
        correct = (tgt == pred).sum().item()
        total = mask.sum().item()
        total_correct += correct
        total_tokens += total
        exact = (tgt == pred).all().item()
        decoded = fr_tokenizer.decode(pred, skip_special_tokens=True)
        results.append((en_texts[i], fr_texts[i], decoded, correct / max(total, 1), exact))

    token_acc = total_correct / max(total_tokens, 1)
    model.train()
    return results, token_acc


# ---------------------------------------------------------------------------
# Geometry probing (same as V12)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_flat(model, tokenizer, texts, device):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64,
                    return_tensors="pt").to(device)
    m = _unwrap(model)
    concepts = m.encode(enc["input_ids"], enc["attention_mask"])
    flat = concepts.view(concepts.shape[0], -1)
    return F.normalize(flat, p=2, dim=-1), concepts


@torch.no_grad()
def probe_geometry(model, tokenizer, device="cuda"):
    model.eval()
    geo = {}

    # --- Analogies ---
    analogy_scores = []
    for a, b, c, d, label in GEOMETRY_ANALOGIES:
        va, _ = _encode_flat(model, tokenizer, [a], device)
        vb, _ = _encode_flat(model, tokenizer, [b], device)
        vc, _ = _encode_flat(model, tokenizer, [c], device)
        vd, _ = _encode_flat(model, tokenizer, [d], device)
        predicted = F.normalize(va - vb + vc, p=2, dim=-1)
        sim = F.cosine_similarity(predicted, vd).item()
        analogy_scores.append(sim)
    geo["analogy_avg"] = float(np.mean(analogy_scores))

    # --- Clustering gap ---
    group_concepts = {}
    for name, sents in GEOMETRY_CLUSTERS.items():
        _, concepts = _encode_flat(model, tokenizer, sents, device)
        group_concepts[name] = concepts

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

    # --- Direction consistency ---
    dir_scores = []
    for attr, pairs in GEOMETRY_DIRECTIONS.items():
        deltas = []
        for pos, neg in pairs:
            vp, _ = _encode_flat(model, tokenizer, [pos], device)
            vn, _ = _encode_flat(model, tokenizer, [neg], device)
            deltas.append(F.normalize(vp - vn, p=2, dim=-1))
        cons = []
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                cons.append(F.cosine_similarity(deltas[i], deltas[j]).item())
        dir_scores.append(float(np.mean(cons)))
    geo["direction_consistency"] = float(np.mean(dir_scores))

    # --- Word order sensitivity ---
    wo_sims = []
    for a, b in GEOMETRY_WORD_ORDER:
        va, _ = _encode_flat(model, tokenizer, [a], device)
        vb, _ = _encode_flat(model, tokenizer, [b], device)
        wo_sims.append(F.cosine_similarity(va, vb).item())
    geo["word_order_sim"] = float(np.mean(wo_sims))

    # --- Effective rank ---
    all_sents = []
    for sents in GEOMETRY_CLUSTERS.values():
        all_sents.extend(sents)
    all_sents.extend([
        "the president gave a speech", "she plays piano beautifully",
        "the train arrived on time", "he forgot his wallet at home",
        "the garden was full of flowers", "she read the book in one day",
        "the mountain was covered in snow", "they celebrated their anniversary",
        "the experiment failed completely", "she won the championship",
        "the baby cried all night", "he fixed the broken window",
        "the river flows to the sea", "she painted a beautiful portrait",
        "the fire spread quickly", "he whispered a secret to her",
        "the team won the game", "she discovered a new species",
    ])
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
                    exact_match_ema, checkpoint_dir):
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
        "version": "v13",
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
    log(f"Loading V13 checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoderV13(config).to(device)
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
    total, _ = model.count_parameters()
    log(f"Resumed V13: {total:,} params | step {step} | em_ema={exact_match_ema:.3f}")
    return model, optimizer, scaler, config, step, exact_match_ema


def load_from_v12(v12_path, config, device="cuda"):
    """Warm-start from V12: load encoder + bottleneck + EN decoder, fresh FR decoder."""
    log(f"Loading from V12 checkpoint: {v12_path}")
    ckpt = torch.load(v12_path, map_location="cpu", weights_only=False)
    v12_state = ckpt["model_state_dict"]
    v12_state = {k.replace("_orig_mod.", ""): v for k, v in v12_state.items()}

    model = ConceptAutoencoderV13(config).to(device)
    model_state = model.state_dict()

    # FR-specific prefixes — these stay randomly initialized
    fr_prefixes = ("fr_dec_layers", "fr_dec_norm", "fr_lm_head", "position_queries_fr")

    loaded = 0
    for key in model_state:
        if key in v12_state and model_state[key].shape == v12_state[key].shape:
            if not any(key.startswith(p) for p in fr_prefixes):
                model_state[key] = v12_state[key]
                loaded += 1

    model.load_state_dict(model_state)
    total, _ = model.count_parameters()
    log(f"Loaded {loaded} tensors from V12 (encoder + bottleneck + EN decoder)")
    log(f"Model: {total:,} params | FR decoder initialized fresh")
    return model


# ---------------------------------------------------------------------------
# Main training loop — dual decoder
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False, eval_only=False, from_v12=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V13 — DUAL-DECODER CONCEPT AUTOENCODER")
    log("  EN→concepts→EN + EN→concepts→FR (shared bottleneck)")
    log("=" * 70)

    from transformers import AutoTokenizer

    en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    fr_tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    # Separate tokenizer instances for eval (avoids threading conflicts with prefetch)
    en_tokenizer_eval = AutoTokenizer.from_pretrained("bert-base-uncased")
    fr_tokenizer_eval = AutoTokenizer.from_pretrained("camembert-base")
    log(f"EN tokenizer: vocab_size={en_tokenizer.vocab_size}")
    log(f"FR tokenizer: vocab_size={fr_tokenizer.vocab_size}")

    model_config = dict(MODEL_CONFIG)
    model_config["vocab_size"] = en_tokenizer.vocab_size
    model_config["fr_vocab_size"] = fr_tokenizer.vocab_size

    exact_match_ema = 0.0

    # Auto-resume V13 only (never auto-fallback to V12)
    if resume_from is None and not fresh and from_v12 is None:
        ckpt_dir = Path(CHECKPOINT_DIR)
        latest = ckpt_dir / "latest.pt"
        if latest.exists():
            resume_from = str(latest)

    if resume_from:
        model, optimizer, scaler, config, start_step, exact_match_ema = \
            load_checkpoint(resume_from, device)
    elif from_v12:
        config = ConceptConfig(**model_config)
        model = load_from_v12(from_v12, config, device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=PEAK_LR, betas=BETAS,
            weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0
    else:
        log("Starting fresh training...")
        config = ConceptConfig(**model_config)
        model = ConceptAutoencoderV13(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=PEAK_LR, betas=BETAS,
            weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0
        total, _ = model.count_parameters()
        log(f"Model: {total:,} params ({total/1e6:.1f}M)")

    if eval_only:
        log("\n--- EN RECONSTRUCTION EVAL ---")
        results, acc, em = evaluate_reconstruction(
            model, en_tokenizer_eval, device)
        log(f"  Overall: token_acc={acc:.3f} exact_match={em:.3f}")
        for orig, decoded, tacc, exact in results:
            status = "OK" if exact else "DIFF"
            log(f"  [{status}] {orig}")
            log(f"       -> {decoded}")

        log("\n--- FR TRANSLATION EVAL ---")
        tr_results, tr_acc = evaluate_translation(
            model, en_tokenizer_eval, fr_tokenizer_eval, device)
        log(f"  FR token_acc={tr_acc:.3f}")
        for en, fr_ref, fr_pred, tacc, exact in tr_results:
            status = "OK" if exact else "DIFF"
            log(f"  [{status}] EN: {en}")
            log(f"       FR ref:  {fr_ref}")
            log(f"       FR pred: {fr_pred}")

        log("\n--- GEOMETRY PROBE ---")
        geo = probe_geometry(model, en_tokenizer_eval, device)
        log(f"  analogy_avg={geo['analogy_avg']:.3f}")
        log(f"  clustering_gap={geo['clustering_gap']:+.4f}")
        log(f"  direction_consistency={geo['direction_consistency']:.3f}")
        log(f"  word_order_sim={geo['word_order_sim']:.3f}")
        log(f"  effective_rank: 90%={geo['rank90']}  95%={geo['rank95']}")
        return

    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()

    log("Loading data...")
    trans_dataset = TranslationDataset(en_tokenizer, fr_tokenizer, max_len=config.max_seq_len)

    if len(trans_dataset.pairs) == 0:
        log("ERROR: No translation pairs found.")
        return

    total, _ = _unwrap(model).count_parameters()
    log(f"\nTraining plan (V13 — Dual-Decoder):")
    log(f"  Model: {total:,} params ({total/1e6:.1f}M)")
    log(f"  Encoder: {config.enc_hidden}h x {config.enc_layers}L x {config.enc_heads}heads")
    log(f"  EN Decoder: PARALLEL {config.dec_hidden}h x {config.dec_layers}L")
    log(f"  FR Decoder: PARALLEL {config.dec_hidden}h x {config.dec_layers}L")
    log(f"  Bottleneck: {config.num_concepts} x {config.concept_dim} = {config.total_concept_dim} dims")
    log(f"  Batch: {BATCH_SIZE}")
    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")
    log(f"  Data: {len(trans_dataset.pairs):,} EN↔FR pairs (both decoders every step)")
    log(f"  Loss: EN recon + {LAMBDA_FR}×FR translation")
    log("-" * 70)

    prefetch = PrefetchBuffer(trans_dataset, device, batch_size=BATCH_SIZE,
                              name="prefetch-trans")
    prefetch.start()
    log("Prefetch buffer started")

    # Initial evals
    log("\n--- INITIAL EN RECONSTRUCTION ---")
    results, acc, em = evaluate_reconstruction(
        model, en_tokenizer_eval, device)
    log(f"  Overall: token_acc={acc:.3f} exact_match={em:.3f}")
    for orig, decoded, tacc, exact in results[:5]:
        status = "OK" if exact else "DIFF"
        log(f"  [{status}] {orig}")
        log(f"       -> {decoded}")

    log("\n--- INITIAL FR TRANSLATION ---")
    tr_results, tr_acc = evaluate_translation(
        model, en_tokenizer_eval, fr_tokenizer_eval, device)
    log(f"  FR token_acc={tr_acc:.3f}")
    for en, fr_ref, fr_pred, tacc, exact in tr_results[:4]:
        log(f"  EN: {en}")
        log(f"  FR pred: {fr_pred}")
        log(f"  FR ref:  {fr_ref}")

    geo = probe_geometry(model, en_tokenizer_eval, device)
    log(f"  Initial geometry: analogy={geo['analogy_avg']:.3f} "
        f"cluster_gap={geo['clustering_gap']:+.4f} "
        f"dir_con={geo['direction_consistency']:.3f} "
        f"wo_sim={geo['word_order_sim']:.3f} "
        f"rank90={geo['rank90']}")

    losses_en = []
    losses_fr = []
    start_time = time.time()
    last_fr_acc = 0.0

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

        # Every step: EN recon + FR translation from the same pairs
        en_enc, fr_enc = prefetch.get()
        en_ids = en_enc["input_ids"].to(device, non_blocking=True)
        en_mask = en_enc["attention_mask"].to(device, non_blocking=True)
        fr_ids = fr_enc["input_ids"].to(device, non_blocking=True)
        fr_mask = fr_enc["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            en_logits, fr_logits, concepts = model(
                en_ids, en_mask,
                fr_seq_len=fr_ids.shape[1],
                fr_attention_mask=fr_mask)

            r_loss = reconstruction_loss(en_logits, en_ids)
            t_loss = translation_loss(fr_logits, fr_ids,
                                      pad_token_id=fr_tokenizer.pad_token_id)
            total_loss = r_loss + LAMBDA_FR * t_loss

        r_loss_val = r_loss.item()
        t_loss_val = t_loss.item()
        losses_en.append(r_loss_val)
        losses_fr.append(t_loss_val)

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
            avg_en = sum(losses_en[-100:]) / len(losses_en[-100:])
            avg_fr = sum(losses_fr[-100:]) / len(losses_fr[-100:])
            pct = (step + 1) / TOTAL_STEPS * 100
            log(f"step {step+1:>7d} [DUAL] | loss {avg_en + avg_fr:.4f} "
                f"(recon={r_loss_val:.4f} fr={t_loss_val:.4f}) | "
                f"em_ema={exact_match_ema:.3f} | "
                f"lr {current_lr:.2e} | {pct:.1f}%")

        # --- Eval ---
        if (step + 1) % EVAL_EVERY == 0:
            # EN diagnostic sentences
            results, acc, em = evaluate_reconstruction(
                model, en_tokenizer_eval, device)
            exact_match_ema = (EXACT_MATCH_EMA_DECAY * exact_match_ema +
                               (1 - EXACT_MATCH_EMA_DECAY) * em)

            log(f"  EN EVAL: token_acc={acc:.3f} exact_match={em:.3f} "
                f"em_ema={exact_match_ema:.3f}")
            for orig, decoded, tacc, exact in results:
                status = "OK" if exact else "DIFF"
                log(f"    [{status}] ({tacc:.0%}) {orig}")
                log(f"           -> {decoded}")

            # FR translation eval
            tr_results, tr_acc = evaluate_translation(
                model, en_tokenizer_eval, fr_tokenizer_eval, device)
            last_fr_acc = tr_acc
            log(f"  FR EVAL: token_acc={tr_acc:.3f}")
            for en, fr_ref, fr_pred, tacc, exact in tr_results:
                status = "OK" if exact else "DIFF"
                log(f"    [{status}] ({tacc:.0%}) {en}")
                log(f"           -> {fr_pred}")
                log(f"           ref: {fr_ref}")

            # Geometry probe
            geo = probe_geometry(model, en_tokenizer_eval, device)
            log(f"  GEOMETRY: analogy={geo['analogy_avg']:.3f} "
                f"cluster_gap={geo['clustering_gap']:+.4f} "
                f"dir_con={geo['direction_consistency']:.3f} "
                f"wo_sim={geo['word_order_sim']:.3f} "
                f"rank90={geo['rank90']} rank95={geo['rank95']}")

            # Log metrics
            elapsed = time.time() - start_time
            avg_en = sum(losses_en[-100:]) / len(losses_en[-100:])
            avg_fr = sum(losses_fr[-100:]) / len(losses_fr[-100:])
            log_metrics(step + 1, avg_en, acc, em,
                        0, 0, 0, 0, 0, 0,
                        current_lr, elapsed / 3600, geo=geo,
                        fr_loss=avg_fr, fr_token_acc=last_fr_acc)

        # --- Checkpoint ---
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = sum(losses_en[-100:]) / len(losses_en[-100:])
            save_checkpoint(model, optimizer, scaler, config, step + 1,
                            avg_loss, exact_match_ema, CHECKPOINT_DIR)

        if shutdown_requested:
            break

    prefetch.stop()
    if losses_en:
        avg_loss = sum(losses_en[-100:]) / len(losses_en[-100:])
        save_checkpoint(model, optimizer, scaler, config,
                        step + 1 if not shutdown_requested else step,
                        avg_loss, exact_match_ema, CHECKPOINT_DIR)
    log("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train concept autoencoder V13")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--from-v12", type=str, default=None,
                        help="Path to V12 checkpoint (loads encoder+EN decoder, fresh FR decoder)")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    with open(".train_pid", "w") as f:
        f.write(str(os.getpid()))

    train(resume_from=args.resume, fresh=args.fresh,
          eval_only=args.eval_only, from_v12=args.from_v12)
