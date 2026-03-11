"""
FLM V15 — Hydra + Geometry Losses
================================================================================
Key changes from V14:
  - Bigger bottleneck: 32×16 = 512 dims (was 16×16=256 in V14, 32×32=1024 in V13)
  - Same 5 parallel decoder heads (EN, FR, ES, Para, Parse)
  - GEOMETRY LOSSES RESTORED from V7-V9:
    - margin_word_order_loss: push word-order swaps below target similarity
    - hard_repulsion_loss: push apart most-similar unrelated pairs
    - batch_repulsion_loss: prevent global similarity collapse
  - Recon-gated: geometry losses only activate after EN recon EM EMA > 0.5
  - Dynamic loss weighting: learned log-variance per head (uncertainty weighting)

V7-V9 proved geometry losses work for word order (wo_sim 0.89-0.81) but had poor
reconstruction (autoregressive decoder). V10-V14 proved parallel decoder gets great
reconstruction but geometry collapses without explicit losses. V15 = both.

Usage:
    python train_v15.py --fresh            # start from scratch
    python train_v15.py                    # auto-resume
    python train_v15.py --eval-only        # diagnostics only
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
from concept_model import (ConceptConfig, ConceptAutoencoderV14,
                           reconstruction_loss, translation_loss,
                           flat_similarity_matrix,
                           margin_word_order_loss,
                           hard_repulsion_loss, batch_repulsion_loss)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v15"
LOG_DIR = "logs"

MODEL_CONFIG = dict(
    vocab_size=30522,       # BERT (EN encoder + EN/para/parse decoders)
    fr_vocab_size=32005,    # CamemBERT (FR decoder)
    es_vocab_size=31002,    # BETO (ES decoder)
    enc_hidden=384,
    enc_layers=6,
    enc_heads=6,
    enc_intermediate=1536,
    num_concepts=32,        # BIGGER: was 16 in V14
    concept_dim=16,         # 32×16 = 512 total dims
    dec_hidden=384,
    dec_layers=4,           # 4 layers per head (same as V14)
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

# Data sampling weights
DATA_WEIGHTS = {
    "fr": 0.25,
    "es": 0.25,
    "para": 0.25,     # bumped from 20% — paraphrase pairs ARE word-order data
    "parse": 0.25,
}

# Geometry loss config
GEO_GATE_THRESHOLD = 0.5   # EN EM EMA must exceed this to enable geo losses
GEO_RAMP_STEPS = 5000      # ramp geo weight from 0→1 over this many steps after gate opens
GEO_WO_WEIGHT = 2.0        # margin word order loss weight
GEO_WO_TARGET = 0.5        # target: wo pairs should have sim < 0.5
GEO_HREPUL_WEIGHT = 3.0    # hard repulsion weight
GEO_HREPUL_TARGET = 0.1    # target: worst pairs should have sim < 0.1
GEO_BREPUL_WEIGHT = 1.0    # batch repulsion weight
GEO_BREPUL_TARGET = 0.3    # target: random pairs should have sim < 0.3

# EMA tracking
EXACT_MATCH_EMA_DECAY = 0.99

# Logging
LOG_EVERY = 50
EVAL_EVERY = 500
CHECKPOINT_EVERY = 5000

LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v15.log",
    "metrics": f"{LOG_DIR}/concept_v15_metrics.csv",
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

# FR translation test pairs
FR_TEST_PAIRS = [
    ("Resumption of the session", "Reprise de la session"),
    ("The vote will take place tomorrow", "Le vote aura lieu demain"),
    ("I would like to thank the Commission", "Je voudrais remercier la Commission"),
    ("The situation is very serious", "La situation est très grave"),
    ("We need to find a solution", "Nous devons trouver une solution"),
    ("This is a very important issue", "C' est une question très importante"),
]

# ES translation test pairs
ES_TEST_PAIRS = [
    ("Resumption of the session", "Reanudación del período de sesiones"),
    ("The vote will take place tomorrow", "La votación tendrá lugar mañana"),
    ("The situation is very serious", "La situación es muy grave"),
    ("We need to find a solution", "Tenemos que encontrar una solución"),
    ("This is a very important issue", "Esta es una cuestión muy importante"),
]

# Parse test pairs
PARSE_TEST_PAIRS = [
    ("the dog bit the man", "subject : the dog | action : bite | object : the man"),
    ("she runs every morning", "subject : she | action : run | location : every morning"),
    ("he did not enjoy the movie", "subject : he | action : enjoy | negation : true | object : the movie"),
    ("the cat chased the mouse quickly", "subject : the cat | action : chase | object : the mouse | manner : quickly"),
]

# Word-order pairs for geometry loss (explicit training signal)
WORD_ORDER_PAIRS = [
    ("the dog bit the man", "the man bit the dog"),
    ("alice likes bob", "bob likes alice"),
    ("she gave him a book", "he gave her a book"),
    ("the cat chased the mouse", "the mouse chased the cat"),
    ("the teacher praised the student", "the student praised the teacher"),
    ("the boy pushed the girl", "the girl pushed the boy"),
    ("she told him a secret", "he told her a secret"),
    ("the car hit the truck", "the truck hit the car"),
    ("the mother hugged the child", "the child hugged the mother"),
    ("the doctor examined the patient", "the patient examined the doctor"),
    ("the wolf ate the sheep", "the sheep ate the wolf"),
    ("john called mary", "mary called john"),
    ("the police caught the thief", "the thief caught the police"),
    ("the bird watched the cat", "the cat watched the bird"),
    ("she served him coffee", "he served her coffee"),
    ("the king defeated the knight", "the knight defeated the king"),
]

# Geometry probing
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


METRICS_HEADER = ("timestamp,step,recon_loss,token_acc,exact_match,em_ema,"
                  "lr,elapsed_hours,"
                  "analogy_avg,clustering_gap,direction_consistency,"
                  "word_order_sim,effective_rank90,effective_rank95,"
                  "fr_loss,fr_token_acc,es_loss,es_token_acc,"
                  "para_loss,para_token_acc,parse_loss,parse_token_acc,"
                  "geo_scale,wo_loss,hrepul_loss,brepul_loss\n")


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
                f"{m.get('fr_loss',0):.6f},{m.get('fr_token_acc',0):.4f},"
                f"{m.get('es_loss',0):.6f},{m.get('es_token_acc',0):.4f},"
                f"{m.get('para_loss',0):.6f},{m.get('para_token_acc',0):.4f},"
                f"{m.get('parse_loss',0):.6f},{m.get('parse_token_acc',0):.4f},"
                f"{m.get('geo_scale',0):.4f},{m.get('wo_loss',0):.6f},"
                f"{m.get('hrepul_loss',0):.6f},{m.get('brepul_loss',0):.6f}\n")


# ---------------------------------------------------------------------------
# Data loading — multi-source (same as V14)
# ---------------------------------------------------------------------------

class MultiSourceDataset:
    """Loads all data sources for the hydra decoder heads."""

    def __init__(self, en_tok, fr_tok, es_tok, max_len=128):
        self.en_tok = en_tok
        self.fr_tok = fr_tok
        self.es_tok = es_tok
        self.max_len = max_len

        self.fr_pairs = []
        self.es_pairs = []
        self.para_pairs = []
        self.parse_pairs = []

        self._load()

    def _load(self):
        for src in ["europarl.jsonl", "tatoeba_enfr.jsonl", "wikimatrix_enfr.jsonl"]:
            self._load_pairs(f"data/pairs/{src}", self.fr_pairs, "FR")
        for src in ["europarl_enes.jsonl", "tatoeba_enes.jsonl", "wikimatrix_enes.jsonl"]:
            self._load_pairs(f"data/pairs/{src}", self.es_pairs, "ES")
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
        for pairs in [self.fr_pairs, self.es_pairs, self.para_pairs, self.parse_pairs]:
            random.shuffle(pairs)
        log(f"  TOTALS: FR={len(self.fr_pairs):,} ES={len(self.es_pairs):,} "
            f"PARA={len(self.para_pairs):,} PARSE={len(self.parse_pairs):,}")

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
        if head == "fr":
            pairs, tok = self.fr_pairs, self.fr_tok
        elif head == "es":
            pairs, tok = self.es_pairs, self.es_tok
        elif head == "para":
            pairs, tok = self.para_pairs, self.en_tok
        elif head == "parse":
            pairs, tok = self.parse_pairs, self.en_tok
        else:
            raise ValueError(f"Unknown head: {head}")

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
# Word-order batch generation (for geometry loss)
# ---------------------------------------------------------------------------

def get_word_order_batch(tokenizer, device, batch_size=16):
    """Get a batch of word-order swap pairs for geometry loss."""
    indices = [random.randint(0, len(WORD_ORDER_PAIRS) - 1)
               for _ in range(batch_size)]
    orig_texts = [WORD_ORDER_PAIRS[i][0] for i in indices]
    swap_texts = [WORD_ORDER_PAIRS[i][1] for i in indices]

    orig_enc = tokenizer(orig_texts, max_length=64, padding=True,
                         truncation=True, return_tensors="pt")
    swap_enc = tokenizer(swap_texts, max_length=64, padding=True,
                         truncation=True, return_tensors="pt")
    return (orig_enc["input_ids"].to(device), orig_enc["attention_mask"].to(device),
            swap_enc["input_ids"].to(device), swap_enc["attention_mask"].to(device))


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
        concepts = m.encode(enc["input_ids"], enc["attention_mask"])
        logits = m.decode_en(concepts, seq_len=enc["input_ids"].shape[1],
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

        concepts = m.encode(en_enc["input_ids"], en_enc["attention_mask"])
        logits = decode_fn(concepts, seq_len=ref_enc["input_ids"].shape[1],
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
# Geometry probing
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

    # Analogies
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

    # Clustering gap
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

    # Direction consistency
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

    # Word order sensitivity
    wo_sims = []
    for a, b in GEOMETRY_WORD_ORDER:
        va, _ = _encode_flat(model, tokenizer, [a], device)
        vb, _ = _encode_flat(model, tokenizer, [b], device)
        wo_sims.append(F.cosine_similarity(va, vb).item())
    geo["word_order_sim"] = float(np.mean(wo_sims))

    # Effective rank
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
        "version": "v15",
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
    log(f"Loading V15 checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoderV14(config).to(device)
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
    geo_gate_step = ckpt.get("geo_gate_step", -1)
    total, _ = model.count_parameters()
    log(f"Resumed V15: {total:,} params | step {step} | em_ema={exact_match_ema:.3f}")
    return model, optimizer, scaler, config, step, exact_match_ema, geo_gate_step


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False, eval_only=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V15 — HYDRA + GEOMETRY LOSSES")
    log("  Bottleneck: 32×16 = 512 dims")
    log("  Heads: EN recon | FR trans | ES trans | EN para | Semantic parse")
    log("  Geometry: margin_wo + hard_repul + batch_repul (recon-gated)")
    log("=" * 70)

    from transformers import AutoTokenizer

    en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    fr_tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    es_tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    en_tok_eval = AutoTokenizer.from_pretrained("bert-base-uncased")
    fr_tok_eval = AutoTokenizer.from_pretrained("camembert-base")
    es_tok_eval = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

    log(f"EN tokenizer: vocab={en_tokenizer.vocab_size}")
    log(f"FR tokenizer: vocab={fr_tokenizer.vocab_size}")
    log(f"ES tokenizer: vocab={es_tokenizer.vocab_size}")

    model_config = dict(MODEL_CONFIG)
    model_config["vocab_size"] = en_tokenizer.vocab_size
    model_config["fr_vocab_size"] = fr_tokenizer.vocab_size
    model_config["es_vocab_size"] = es_tokenizer.vocab_size

    exact_match_ema = 0.0
    geo_gate_step = -1  # step when geo gate opened (-1 = not yet)

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
        model = ConceptAutoencoderV14(config).to(device)  # reuses V14 model class
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

        log("\n--- FR TRANSLATION ---")
        fr_results, fr_acc = evaluate_translation(
            model, en_tok_eval, fr_tok_eval, FR_TEST_PAIRS, "decode_fr", device)
        log(f"  FR token_acc={fr_acc:.3f}")
        for en, ref, pred, tacc in fr_results:
            log(f"  [{tacc:.0%}] {en} -> {pred}")
            log(f"       ref: {ref}")

        log("\n--- ES TRANSLATION ---")
        es_results, es_acc = evaluate_translation(
            model, en_tok_eval, es_tok_eval, ES_TEST_PAIRS, "decode_es", device)
        log(f"  ES token_acc={es_acc:.3f}")
        for en, ref, pred, tacc in es_results:
            log(f"  [{tacc:.0%}] {en} -> {pred}")
            log(f"       ref: {ref}")

        log("\n--- PARSE ---")
        parse_results, parse_acc = evaluate_translation(
            model, en_tok_eval, en_tok_eval, PARSE_TEST_PAIRS, "decode_parse", device)
        log(f"  Parse token_acc={parse_acc:.3f}")
        for en, ref, pred, tacc in parse_results:
            log(f"  [{tacc:.0%}] {en} -> {pred}")
            log(f"       ref: {ref}")

        log("\n--- GEOMETRY ---")
        geo = probe_geometry(model, en_tok_eval, device)
        for k, v in geo.items():
            log(f"  {k}: {v}")
        return

    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()

    log("Loading data...")
    dataset = MultiSourceDataset(en_tokenizer, fr_tokenizer, es_tokenizer,
                                 max_len=config.max_seq_len)

    total, _ = _unwrap(model).count_parameters()
    log(f"\nTraining plan (V15 — Hydra + Geometry):")
    log(f"  Model: {total:,} params ({total/1e6:.1f}M)")
    log(f"  Bottleneck: {config.num_concepts}×{config.concept_dim} = "
        f"{config.num_concepts * config.concept_dim} dims")
    log(f"  Heads: EN({config.dec_layers}L) FR({config.dec_layers}L) "
        f"ES({config.dec_layers}L) PARA({config.dec_layers}L) PARSE({config.dec_layers}L)")
    log(f"  Geometry: wo={GEO_WO_WEIGHT} hrepul={GEO_HREPUL_WEIGHT} "
        f"brepul={GEO_BREPUL_WEIGHT} | gate@EM>{GEO_GATE_THRESHOLD}")
    log(f"  Batch: {BATCH_SIZE}")
    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")
    log(f"  Sampling: " + " ".join(f"{h}={w:.0%}" for h, w in DATA_WEIGHTS.items()))
    log("-" * 70)

    prefetch = HydraPrefetchBuffer(dataset, device, batch_size=BATCH_SIZE)
    prefetch.start()
    log("Prefetch buffer started")

    PAD_IDS = {
        "en": en_tokenizer.pad_token_id or 0,
        "fr": fr_tokenizer.pad_token_id or 1,
        "es": es_tokenizer.pad_token_id or 1,
        "para": en_tokenizer.pad_token_id or 0,
        "parse": en_tokenizer.pad_token_id or 0,
    }

    DECODE_FNS = {
        "fr": "decode_fr",
        "es": "decode_es",
        "para": "decode_para",
        "parse": "decode_parse",
    }

    loss_trackers = {h: [] for h in ["en", "fr", "es", "para", "parse"]}
    geo_trackers = {"wo": [], "hrepul": [], "brepul": []}
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

            # Encode
            concepts = m.encode(en_ids, en_mask)

            # EN reconstruction (always)
            en_logits = m.decode_en(concepts, en_ids.shape[1], en_mask)
            r_loss = reconstruction_loss(en_logits, en_ids)

            # Secondary head
            decode_fn = getattr(m, DECODE_FNS[head])
            head_logits = decode_fn(concepts, tgt_ids.shape[1], tgt_mask)
            h_loss = F.cross_entropy(
                head_logits.reshape(-1, head_logits.shape[-1]),
                tgt_ids.reshape(-1),
                ignore_index=PAD_IDS[head])

            total_loss = r_loss + h_loss

            # --- Geometry losses (recon-gated) ---
            # Check gate
            geo_scale = 0.0
            if exact_match_ema >= GEO_GATE_THRESHOLD:
                if geo_gate_step < 0:
                    geo_gate_step = step
                    log(f"  GEO GATE OPENED at step {step} (em_ema={exact_match_ema:.3f})")
                # Ramp
                steps_since_gate = step - geo_gate_step
                geo_scale = min(1.0, steps_since_gate / GEO_RAMP_STEPS)

            wo_val = 0.0
            hr_val = 0.0
            br_val = 0.0

            if geo_scale > 0:
                # Word-order loss: encode swap pairs, push apart
                wo_orig_ids, wo_orig_mask, wo_swap_ids, wo_swap_mask = \
                    get_word_order_batch(en_tokenizer, device, batch_size=16)
                c_orig = m.encode(wo_orig_ids, wo_orig_mask)
                c_swap = m.encode(wo_swap_ids, wo_swap_mask)
                wo_loss, _ = margin_word_order_loss(c_orig, c_swap,
                                                     target_sim=GEO_WO_TARGET)
                total_loss = total_loss + geo_scale * GEO_WO_WEIGHT * wo_loss
                wo_val = wo_loss.item()

                # Hard repulsion on batch concepts
                hr_loss, _ = hard_repulsion_loss(concepts,
                                                  target_sim=GEO_HREPUL_TARGET,
                                                  top_k=8)
                total_loss = total_loss + geo_scale * GEO_HREPUL_WEIGHT * hr_loss
                hr_val = hr_loss.item()

                # Batch repulsion
                br_loss, _ = batch_repulsion_loss(concepts,
                                                   target_sim=GEO_BREPUL_TARGET)
                total_loss = total_loss + geo_scale * GEO_BREPUL_WEIGHT * br_loss
                br_val = br_loss.item()

        r_val = r_loss.item()
        h_val = h_loss.item()
        loss_trackers["en"].append(r_val)
        loss_trackers[head].append(h_val)
        geo_trackers["wo"].append(wo_val)
        geo_trackers["hrepul"].append(hr_val)
        geo_trackers["brepul"].append(br_val)

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
            for h in ["fr", "es", "para", "parse"]:
                if loss_trackers[h]:
                    head_losses.append(f"{h}={np.mean(loss_trackers[h][-50:]):.3f}")
            hl_str = " ".join(head_losses) if head_losses else "warming up"
            geo_str = f"geo={geo_scale:.2f}"
            if geo_scale > 0:
                avg_wo = np.mean(geo_trackers["wo"][-50:])
                avg_hr = np.mean(geo_trackers["hrepul"][-50:])
                geo_str += f" wo={avg_wo:.3f} hr={avg_hr:.3f}"
            log(f"step {step+1:>7d} [HYDRA+GEO] | en={avg_en:.4f} {hl_str} | "
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

            fr_results, fr_acc = evaluate_translation(
                model, en_tok_eval, fr_tok_eval, FR_TEST_PAIRS, "decode_fr", device)
            log(f"  FR EVAL: token_acc={fr_acc:.3f}")
            for en, ref, pred, tacc in fr_results[:3]:
                log(f"    [{tacc:.0%}] {en} -> {pred}")
                log(f"           ref: {ref}")

            es_results, es_acc = evaluate_translation(
                model, en_tok_eval, es_tok_eval, ES_TEST_PAIRS, "decode_es", device)
            log(f"  ES EVAL: token_acc={es_acc:.3f}")
            for en, ref, pred, tacc in es_results[:3]:
                log(f"    [{tacc:.0%}] {en} -> {pred}")
                log(f"           ref: {ref}")

            parse_results, parse_acc = evaluate_translation(
                model, en_tok_eval, en_tok_eval, PARSE_TEST_PAIRS, "decode_parse", device)
            log(f"  PARSE EVAL: token_acc={parse_acc:.3f}")
            for en, ref, pred, tacc in parse_results:
                log(f"    [{tacc:.0%}] {en} -> {pred}")
                log(f"           ref: {ref}")

            geo = probe_geometry(model, en_tok_eval, device)
            log(f"  GEOMETRY: analogy={geo['analogy_avg']:.3f} "
                f"cluster_gap={geo['clustering_gap']:+.4f} "
                f"dir_con={geo['direction_consistency']:.3f} "
                f"wo_sim={geo['word_order_sim']:.3f} "
                f"rank90={geo['rank90']} rank95={geo['rank95']}")

            total_heads = sum(head_counts.values()) or 1
            dist = " ".join(f"{h}={head_counts[h]/total_heads:.0%}" for h in DATA_WEIGHTS)
            log(f"  HEAD DIST: {dist} | geo_scale={geo_scale:.2f} "
                f"gate_step={geo_gate_step}")

            elapsed = time.time() - start_time
            avg_en = np.mean(loss_trackers["en"][-100:]) if loss_trackers["en"] else 0
            avg_fr = np.mean(loss_trackers["fr"][-50:]) if loss_trackers["fr"] else 0
            avg_es = np.mean(loss_trackers["es"][-50:]) if loss_trackers["es"] else 0
            avg_para = np.mean(loss_trackers["para"][-50:]) if loss_trackers["para"] else 0
            avg_parse = np.mean(loss_trackers["parse"][-50:]) if loss_trackers["parse"] else 0
            log_metrics(step + 1, {
                "recon_loss": avg_en, "token_acc": acc, "exact_match": em,
                "em_ema": exact_match_ema, "lr": current_lr,
                "elapsed_hours": elapsed / 3600, "geo": geo,
                "fr_loss": avg_fr, "fr_token_acc": fr_acc,
                "es_loss": avg_es, "es_token_acc": es_acc,
                "para_loss": avg_para, "para_token_acc": 0,
                "parse_loss": avg_parse, "parse_token_acc": parse_acc,
                "geo_scale": geo_scale,
                "wo_loss": np.mean(geo_trackers["wo"][-50:]) if geo_trackers["wo"] else 0,
                "hrepul_loss": np.mean(geo_trackers["hrepul"][-50:]) if geo_trackers["hrepul"] else 0,
                "brepul_loss": np.mean(geo_trackers["brepul"][-50:]) if geo_trackers["brepul"] else 0,
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
    parser = argparse.ArgumentParser(description="Train concept autoencoder V15 (Hydra+Geo)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    train(resume_from=args.resume, fresh=args.fresh, eval_only=args.eval_only)
