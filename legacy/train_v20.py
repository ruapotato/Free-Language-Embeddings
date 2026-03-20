"""
FLM V20 — Focused 3-Head + NLI Graded Contrastive + WordNet Geometry
================================================================================
2 decoder heads (EN, FR) with 6L decoders. Freed params from V19's 9 heads.
Bottleneck: 32×16 = 512 dims.

Key changes from V19:
  - ConceptAutoencoderV20: 2 heads (EN, FR), 6L decoders
  - NLI 3-tier contrastive loss (entailment=close, neutral=mid, contradiction=far)
  - WordNet noun hierarchy distance loss
  - WordNet adjective/verb axis consistency loss
  - WordNet verb troponym chain ordering loss
  - Word-order margin loss (gated on EM EMA > 0.5)

Usage:
    python train_v20.py --fresh            # start from scratch
    python train_v20.py                    # auto-resume
    python train_v20.py --eval-only        # diagnostics only
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
from concept_model import (ConceptConfig, ConceptAutoencoderV20,
                           reconstruction_loss,
                           flat_similarity_matrix,
                           margin_word_order_loss,
                           nli_graded_contrastive_loss,
                           wordnet_hierarchy_loss,
                           wordnet_axis_consistency_loss,
                           wordnet_troponym_chain_loss)
from geometry_data import GeometryDataGenerator, verify_splits
from wordnet_data import WordNetDataGenerator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v20"
LOG_DIR = "logs"

MODEL_CONFIG = dict(
    vocab_size=119547,      # mBERT shared vocab
    enc_hidden=384,
    enc_layers=6,
    enc_heads=6,
    enc_intermediate=1536,
    num_concepts=32,
    concept_dim=16,         # 32×16 = 512 bottleneck
    dec_hidden=384,
    dec_layers=6,           # 6L per head (3 heads, more depth)
    dec_heads=6,
    dec_intermediate=1536,
    max_seq_len=128,
    dropout=0.1,
)

# Training hyperparameters
BATCH_SIZE = 24  # 3 heads + no contrastive double-encode on most steps
PEAK_LR = 2e-4
MIN_LR = 1e-5
WARMUP_STEPS = 2000
TOTAL_STEPS = 600_000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0

# Data sampling weights (decoder heads + bottleneck-only losses)
DATA_WEIGHTS = {
    "fr": 0.25,          # FR translation head
    "nli": 0.30,         # NLI 3-tier contrastive (bottleneck-only)
    "wn_noun": 0.18,     # WordNet noun hierarchy (bottleneck-only)
    "wn_axis": 0.17,     # WordNet axis consistency (bottleneck-only)
    "wn_tropo": 0.10,    # WordNet troponym chains (bottleneck-only)
}

# Loss weights
NLI_WEIGHT = 1.0
WN_NOUN_WEIGHT = 0.5
WN_AXIS_WEIGHT = 0.5
WN_TROPO_WEIGHT = 0.3

# Word order loss config
WO_WEIGHT = 2.0
WO_TARGET = 0.5
WO_GATE_THRESHOLD = 0.5

# EMA tracking
EXACT_MATCH_EMA_DECAY = 0.99

# Logging
LOG_EVERY = 50
EVAL_EVERY = 500
CHECKPOINT_EVERY = 5000

LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v20.log",
    "metrics": f"{LOG_DIR}/concept_v20_metrics.csv",
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

FR_TEST_PAIRS = [
    ("Resumption of the session", "Reprise de la session"),
    ("The vote will take place tomorrow", "Le vote aura lieu demain"),
    ("I would like to thank the Commission", "Je voudrais remercier la Commission"),
    ("The situation is very serious", "La situation est très grave"),
    ("We need to find a solution", "Nous devons trouver une solution"),
    ("This is a very important issue", "C' est une question très importante"),
]

HEAD_TEST_PAIRS = {
    "fr": FR_TEST_PAIRS,
}


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
                  "fr_loss,fr_token_acc,"
                  "nli_loss,wn_noun_loss,wn_axis_loss,wn_tropo_loss,wo_loss,"
                  "nli_ordering_acc,wn_noun_corr\n")


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
                f"{g.get('dir_consistency', g.get('direction_consistency', 0)):.4f},"
                f"{g.get('word_order_sim',0):.4f},"
                f"{g.get('rank90',0)},{g.get('rank95',0)},"
                f"{m.get('fr_loss',0):.6f},{m.get('fr_token_acc',0):.4f},"
                f"{m.get('nli_loss',0):.6f},{m.get('wn_noun_loss',0):.6f},"
                f"{m.get('wn_axis_loss',0):.6f},{m.get('wn_tropo_loss',0):.6f},"
                f"{m.get('wo_loss',0):.6f},"
                f"{m.get('nli_ordering_acc',0):.4f},"
                f"{m.get('wn_noun_corr',0):.4f}\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class V20Dataset:
    """Loads FR, Para, and NLI data. WordNet data is generated on-the-fly."""

    def __init__(self, tokenizer, max_len=128):
        self.tok = tokenizer
        self.max_len = max_len
        self.fr_pairs = []
        self.nli_entail = []
        self.nli_neutral = []
        self.nli_contra = []
        self._load()

    def _load(self):
        # FR data
        for src in ["europarl.jsonl", "tatoeba_enfr.jsonl", "wikimatrix_enfr.jsonl"]:
            self._load_pairs(f"data/pairs/{src}", self.fr_pairs, "FR")
        # NLI data (all 3 tiers)
        nli_path = Path("data/pairs/nli.jsonl")
        if nli_path.exists():
            e_count = n_count = c_count = 0
            with open(nli_path) as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        a = doc["text_a"].strip()
                        b = doc["text_b"].strip()
                        if len(a) <= 5 or len(b) <= 5:
                            continue
                        nli_type = doc.get("type", "")
                        if nli_type == "entailment":
                            self.nli_entail.append((a, b))
                            e_count += 1
                        elif nli_type == "neutral":
                            self.nli_neutral.append((a, b))
                            n_count += 1
                        elif nli_type == "contradiction":
                            self.nli_contra.append((a, b))
                            c_count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
            log(f"  NLI: entail={e_count:,} neutral={n_count:,} contra={c_count:,}")

        for pairs in [self.fr_pairs,
                      self.nli_entail, self.nli_neutral, self.nli_contra]:
            random.shuffle(pairs)

        log(f"  TOTALS: FR={len(self.fr_pairs):,} "
            f"NLI={len(self.nli_entail)+len(self.nli_neutral)+len(self.nli_contra):,}")

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

    def get_pair_batch(self, batch_size, head):
        """Get a batch of (en, target) pairs for a decoder head."""
        if head == "fr":
            pairs = self.fr_pairs
        else:
            raise ValueError(f"Unknown head: {head}")
        indices = [random.randint(0, len(pairs) - 1) for _ in range(batch_size)]
        en_texts = [pairs[i][0] for i in indices]
        tgt_texts = [pairs[i][1] for i in indices]
        en_enc = self.tok(en_texts, max_length=self.max_len,
                          padding=True, truncation=True, return_tensors="pt")
        tgt_enc = self.tok(tgt_texts, max_length=self.max_len,
                           padding=True, truncation=True, return_tensors="pt")
        return en_enc, tgt_enc

    def get_nli_batch(self, batch_size):
        """Get a balanced NLI batch with labels (0=entail, 1=neutral, 2=contradiction)."""
        per_type = batch_size // 3
        remainder = batch_size - 3 * per_type
        texts_a, texts_b, labels = [], [], []
        for pairs, label, count in [
            (self.nli_entail, 0, per_type + (1 if remainder > 0 else 0)),
            (self.nli_neutral, 1, per_type + (1 if remainder > 1 else 0)),
            (self.nli_contra, 2, per_type),
        ]:
            for _ in range(count):
                a, b = pairs[random.randint(0, len(pairs) - 1)]
                texts_a.append(a)
                texts_b.append(b)
                labels.append(label)
        # Shuffle within batch
        combined = list(zip(texts_a, texts_b, labels))
        random.shuffle(combined)
        texts_a, texts_b, labels = zip(*combined)
        enc_a = self.tok(list(texts_a), max_length=self.max_len,
                         padding=True, truncation=True, return_tensors="pt")
        enc_b = self.tok(list(texts_b), max_length=self.max_len,
                         padding=True, truncation=True, return_tensors="pt")
        return enc_a, enc_b, torch.tensor(labels, dtype=torch.long)


class V20PrefetchBuffer:
    """Pre-generates batches in background thread."""

    def __init__(self, dataset, wn_gen, tokenizer, device,
                 batch_size=24, buf_size=8):
        self.dataset = dataset
        self.wn_gen = wn_gen
        self.tok = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_len = 128
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

    def _sample_type(self):
        r = random.random()
        for i, cw in enumerate(self._cum_weights):
            if r <= cw:
                return self._heads[i]
        return self._heads[-1]

    def start(self):
        self._thread = threading.Thread(target=self._fill, daemon=True,
                                        name="v20-prefetch")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _fill(self):
        while not self._stop.is_set():
            try:
                batch_type = self._sample_type()

                if batch_type in ("fr", "para"):
                    en_enc, tgt_enc = self.dataset.get_pair_batch(
                        self.batch_size, batch_type)
                    self.q.put(("head", batch_type, en_enc, tgt_enc), timeout=1.0)

                elif batch_type == "nli":
                    enc_a, enc_b, labels = self.dataset.get_nli_batch(self.batch_size)
                    self.q.put(("nli", None, enc_a, enc_b, labels), timeout=1.0)

                elif batch_type == "wn_noun":
                    batch = self.wn_gen.noun_hierarchy_batch(self.batch_size)
                    enc_a = self.tok(batch["sentences_a"], max_length=self.max_len,
                                     padding=True, truncation=True, return_tensors="pt")
                    enc_b = self.tok(batch["sentences_b"], max_length=self.max_len,
                                     padding=True, truncation=True, return_tensors="pt")
                    distances = torch.tensor(batch["distances"], dtype=torch.float32)
                    self.q.put(("wn_noun", None, enc_a, enc_b, distances), timeout=1.0)

                elif batch_type == "wn_axis":
                    batch = self.wn_gen.adj_axis_batch(self.batch_size, contexts_per_axis=4)
                    # Flatten groups into pairs for encoding
                    all_a, all_b, group_sizes = [], [], []
                    for group in batch["groups"]:
                        all_a.extend(group["sentences_a"])
                        all_b.extend(group["sentences_b"])
                        group_sizes.append(len(group["sentences_a"]))
                    if all_a:
                        enc_a = self.tok(all_a, max_length=self.max_len,
                                         padding=True, truncation=True, return_tensors="pt")
                        enc_b = self.tok(all_b, max_length=self.max_len,
                                         padding=True, truncation=True, return_tensors="pt")
                        self.q.put(("wn_axis", group_sizes, enc_a, enc_b), timeout=1.0)

                elif batch_type == "wn_tropo":
                    batch = self.wn_gen.verb_troponym_batch(self.batch_size,
                                                            contexts_per_chain=3)
                    # Encode all sentences for all chains
                    all_sents = []
                    chain_info = []  # (n_levels, n_contexts_per_level)
                    for chain_data in batch["chains"]:
                        n_levels = len(chain_data["chain_verbs"])
                        n_ctx = len(chain_data["sentences_per_level"][0])
                        for level_sents in chain_data["sentences_per_level"]:
                            all_sents.extend(level_sents)
                        chain_info.append((n_levels, n_ctx))
                    if all_sents:
                        enc = self.tok(all_sents, max_length=self.max_len,
                                       padding=True, truncation=True, return_tensors="pt")
                        self.q.put(("wn_tropo", chain_info, enc), timeout=1.0)

            except queue.Full:
                continue
            except Exception:
                continue

    def get(self):
        return self.q.get()


# ---------------------------------------------------------------------------
# Geometry encoding helper
# ---------------------------------------------------------------------------

def _encode_concepts(model, tokenizer, texts, device):
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    enc = tokenizer(texts, max_length=64, padding=True, truncation=True, return_tensors="pt")
    concepts = m.encode(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return concepts.view(concepts.shape[0], -1)


def get_word_order_batch(geo_gen, tokenizer, model, device, batch_size=16):
    orig_texts, swap_texts = geo_gen.word_order_batch(batch_size)
    return _encode_concepts(model, tokenizer, orig_texts, device), \
           _encode_concepts(model, tokenizer, swap_texts, device)


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
def evaluate_head(model, tokenizer, test_pairs, head_name, device="cuda"):
    model.eval()
    m = _unwrap(model)
    results = []
    total_correct = 0
    total_tokens = 0
    for en_text, ref_text in test_pairs:
        en_enc = tokenizer([en_text], max_length=128, padding=True,
                           truncation=True, return_tensors="pt").to(device)
        ref_enc = tokenizer([ref_text], max_length=128, padding=True,
                            truncation=True, return_tensors="pt").to(device)
        concepts = m.encode(en_enc["input_ids"], en_enc["attention_mask"])
        logits = m.decode_head(head_name, concepts,
                               seq_len=ref_enc["input_ids"].shape[1],
                               attention_mask=ref_enc["attention_mask"])
        predicted = logits.argmax(dim=-1)
        mask = ref_enc["attention_mask"][0].bool()
        tgt = ref_enc["input_ids"][0][mask]
        pred = predicted[0][mask]
        correct = (tgt == pred).sum().item()
        total = mask.sum().item()
        total_correct += correct
        total_tokens += total
        decoded = tokenizer.decode(pred, skip_special_tokens=True)
        results.append((en_text, ref_text, decoded, correct / max(total, 1)))
    token_acc = total_correct / max(total_tokens, 1)
    model.train()
    return results, token_acc


@torch.no_grad()
def probe_nli_ordering(model, tokenizer, dataset, device="cuda"):
    """Check if entailment > neutral > contradiction in cosine sim."""
    model.eval()
    m = _unwrap(model)
    type_sims = {}
    for nli_type, pairs, label in [
        ("entailment", dataset.nli_entail, 0),
        ("neutral", dataset.nli_neutral, 1),
        ("contradiction", dataset.nli_contra, 2),
    ]:
        sample = [pairs[random.randint(0, len(pairs) - 1)] for _ in range(50)]
        texts_a = [p[0] for p in sample]
        texts_b = [p[1] for p in sample]
        enc_a = tokenizer(texts_a, max_length=128, padding=True,
                          truncation=True, return_tensors="pt").to(device)
        enc_b = tokenizer(texts_b, max_length=128, padding=True,
                          truncation=True, return_tensors="pt").to(device)
        va = m.concept_vector(enc_a["input_ids"], enc_a["attention_mask"])
        vb = m.concept_vector(enc_b["input_ids"], enc_b["attention_mask"])
        sims = F.cosine_similarity(va, vb, dim=-1)
        type_sims[nli_type] = sims.mean().item()

    ordered = (type_sims["entailment"] > type_sims["neutral"] >
               type_sims["contradiction"])
    model.train()
    return type_sims, ordered


# ---------------------------------------------------------------------------
# Geometry probing (same as V19)
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
    test_gen = GeometryDataGenerator(split="test", seed=42)

    # Analogies
    analogy_scores = []
    a_texts, b_texts, c_texts, d_texts = test_gen.analogy_batch(batch_size=20)
    for i in range(len(a_texts)):
        va, _ = _encode_flat(model, tokenizer, [a_texts[i]], device)
        vb, _ = _encode_flat(model, tokenizer, [b_texts[i]], device)
        vc, _ = _encode_flat(model, tokenizer, [c_texts[i]], device)
        vd, _ = _encode_flat(model, tokenizer, [d_texts[i]], device)
        predicted = F.normalize(va - vb + vc, p=2, dim=-1)
        sim = F.cosine_similarity(predicted, vd).item()
        analogy_scores.append(sim)
    geo["analogy_avg"] = sum(analogy_scores) / len(analogy_scores) if analogy_scores else 0

    # Clustering
    cluster_groups = test_gen.cluster_batch(n_groups=4, n_per_group=5)
    group_concepts = {}
    for name, sents in cluster_groups.items():
        _, concepts_3d = _encode_flat(model, tokenizer, sents, device)
        group_concepts[name] = concepts_3d
    within_sims, between_sims = [], []
    group_names = list(group_concepts.keys())
    for name in group_names:
        c = group_concepts[name]
        sim_mat = flat_similarity_matrix(c, c)
        n = sim_mat.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                within_sims.append(sim_mat[i, j].item())
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            ci = group_concepts[group_names[i]]
            cj = group_concepts[group_names[j]]
            for a in range(ci.shape[0]):
                for b in range(cj.shape[0]):
                    fi = ci[a].view(1, -1)
                    fj = cj[b].view(1, -1)
                    between_sims.append(F.cosine_similarity(fi, fj).item())
    avg_within = sum(within_sims) / len(within_sims) if within_sims else 0
    avg_between = sum(between_sims) / len(between_sims) if between_sims else 0
    geo["clustering_gap"] = avg_within - avg_between

    # Direction consistency
    dir_attr_pairs = test_gen.direction_batch(n_pairs_per_attr=4)
    all_dir_sims = []
    for attr_name, pairs in dir_attr_pairs.items():
        deltas = []
        for pos, neg in pairs:
            vp, _ = _encode_flat(model, tokenizer, [pos], device)
            vn, _ = _encode_flat(model, tokenizer, [neg], device)
            deltas.append(F.normalize(vp - vn, p=2, dim=-1))
        for a in range(len(deltas)):
            for b in range(a + 1, len(deltas)):
                sim = F.cosine_similarity(deltas[a], deltas[b]).item()
                all_dir_sims.append(sim)
    geo["dir_consistency"] = sum(all_dir_sims) / len(all_dir_sims) if all_dir_sims else 0

    # Word order
    wo_sims = []
    wo_origs, wo_swaps = test_gen.word_order_batch(batch_size=20)
    for i in range(len(wo_origs)):
        va, _ = _encode_flat(model, tokenizer, [wo_origs[i]], device)
        vb, _ = _encode_flat(model, tokenizer, [wo_swaps[i]], device)
        wo_sims.append(F.cosine_similarity(va, vb).item())
    geo["word_order_sim"] = sum(wo_sims) / len(wo_sims) if wo_sims else 1.0

    # Effective rank
    all_sents = test_gen.diverse_sentences(batch_size=60)
    vecs, _ = _encode_flat(model, tokenizer, all_sents, device)
    vecs_np = vecs.cpu().numpy()
    try:
        U, S, Vt = np.linalg.svd(vecs_np, full_matrices=False)
        S_norm = S / S.sum()
        cumsum = np.cumsum(S_norm)
        geo["rank90"] = int(np.searchsorted(cumsum, 0.90) + 1)
        geo["rank95"] = int(np.searchsorted(cumsum, 0.95) + 1)
    except Exception:
        geo["rank90"] = 0
        geo["rank95"] = 0

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
                    exact_match_ema, wo_gate_step, checkpoint_dir):
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
        "wo_gate_step": wo_gate_step,
        "version": "v20",
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
    log(f"Loading V20 checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoderV20(config).to(device)
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
    wo_gate_step = ckpt.get("wo_gate_step", -1)
    total, _ = model.count_parameters()
    log(f"Resumed V20: {total:,} params | step {step} | em_ema={exact_match_ema:.3f}")
    return model, optimizer, scaler, config, step, exact_match_ema, wo_gate_step


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False, eval_only=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V20 — FOCUSED 3-HEAD + NLI + WORDNET GEOMETRY")
    log("  Bottleneck: 32x16 = 512 dims | 6L decoders x 2 heads")
    log("  Heads: EN FR (shared mBERT + lm_head)")
    log("  Losses: recon + FR + NLI_3tier + WN_noun + WN_axis + WN_tropo + WO")
    log("=" * 70)

    verify_splits()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    eval_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    geo_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    wn_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    train_geo_gen = GeometryDataGenerator(split="train", seed=None)
    train_wn_gen = WordNetDataGenerator(split="train")

    log(f"mBERT tokenizer: vocab={tokenizer.vocab_size}")

    model_config = dict(MODEL_CONFIG)
    model_config["vocab_size"] = tokenizer.vocab_size

    exact_match_ema = 0.0
    wo_gate_step = -1

    if resume_from is None and not fresh:
        latest = Path(CHECKPOINT_DIR) / "latest.pt"
        if latest.exists():
            resume_from = str(latest)

    if resume_from:
        model, optimizer, scaler, config, start_step, exact_match_ema, wo_gate_step = \
            load_checkpoint(resume_from, device)
    else:
        log("Starting fresh training...")
        config = ConceptConfig(**model_config)
        model = ConceptAutoencoderV20(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=PEAK_LR, betas=BETAS,
            weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")
        start_step = 0
        total, _ = model.count_parameters()
        log(f"Model: {total:,} params ({total/1e6:.1f}M)")

    if eval_only:
        log("\n--- EN RECONSTRUCTION ---")
        results, acc, em = evaluate_reconstruction(model, eval_tokenizer, device)
        log(f"  token_acc={acc:.3f} exact_match={em:.3f}")
        for orig, decoded, tacc, exact in results:
            status = "OK" if exact else "DIFF"
            log(f"  [{status}] ({tacc:.0%}) {orig}")
            if not exact:
                log(f"       -> {decoded}")
        for head_name, test_pairs in HEAD_TEST_PAIRS.items():
            head_upper = head_name.upper()
            head_results, head_acc = evaluate_head(
                model, eval_tokenizer, test_pairs, head_name, device)
            log(f"  {head_upper} EVAL: token_acc={head_acc:.3f}")
            for en, ref, pred, tacc in head_results:
                log(f"  [{tacc:.0%}] {en} -> {pred}")
                log(f"       ref: {ref}")
        log("\n--- GEOMETRY (TEST) ---")
        geo = probe_geometry(model, eval_tokenizer, device)
        for k, v in geo.items():
            log(f"  {k}: {v}")
        return

    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()

    log("Loading data...")
    dataset = V20Dataset(tokenizer, max_len=config.max_seq_len)

    total, _ = _unwrap(model).count_parameters()
    log(f"\nTraining plan (V20 — Focused + NLI + WordNet):")
    log(f"  Model: {total:,} params ({total/1e6:.1f}M)")
    log(f"  Bottleneck: {config.num_concepts}x{config.concept_dim} = "
        f"{config.num_concepts * config.concept_dim} dims")
    log(f"  Heads: " + " ".join(f"{h.upper()}({config.dec_layers}L)"
                                for h in ConceptAutoencoderV20.HEAD_NAMES))
    log(f"  NLI: weight={NLI_WEIGHT} | WN_noun: {WN_NOUN_WEIGHT} | "
        f"WN_axis: {WN_AXIS_WEIGHT} | WN_tropo: {WN_TROPO_WEIGHT}")
    log(f"  Word order: weight={WO_WEIGHT} target={WO_TARGET} gate@EM>{WO_GATE_THRESHOLD}")
    log(f"  Batch: {BATCH_SIZE}")
    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")
    log(f"  Sampling: " + " ".join(f"{h}={w:.0%}" for h, w in DATA_WEIGHTS.items()))
    log("-" * 70)

    prefetch = V20PrefetchBuffer(dataset, train_wn_gen, wn_tokenizer, device,
                                  batch_size=BATCH_SIZE)
    prefetch.start()
    log("Prefetch buffer started")

    PAD_ID = tokenizer.pad_token_id or 0

    loss_trackers = {"en": [], "fr": [],
                     "nli": [], "wn_noun": [], "wn_axis": [], "wn_tropo": []}
    wo_tracker = []
    type_counts = {k: 0 for k in DATA_WEIGHTS}

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

        batch_data = prefetch.get()
        batch_type = batch_data[0]

        with torch.amp.autocast("cuda", dtype=torch.float16):
            m = _unwrap(model) if not hasattr(model, 'encode') else model

            if batch_type == "head":
                # Decoder head batch (FR or Para) — always includes EN recon
                head_name = batch_data[1]
                en_enc, tgt_enc = batch_data[2], batch_data[3]
                type_counts[head_name] = type_counts.get(head_name, 0) + 1

                en_ids = en_enc["input_ids"].to(device, non_blocking=True)
                en_mask = en_enc["attention_mask"].to(device, non_blocking=True)
                tgt_ids = tgt_enc["input_ids"].to(device, non_blocking=True)
                tgt_mask = tgt_enc["attention_mask"].to(device, non_blocking=True)

                concepts = m.encode(en_ids, en_mask)

                # EN reconstruction
                en_logits = m.decode_en(concepts, en_ids.shape[1], en_mask)
                r_loss = reconstruction_loss(en_logits, en_ids)

                # Head loss
                head_logits = m.decode_head(head_name, concepts,
                                            tgt_ids.shape[1], tgt_mask)
                h_loss = F.cross_entropy(
                    head_logits.reshape(-1, head_logits.shape[-1]),
                    tgt_ids.reshape(-1), ignore_index=PAD_ID)

                total_loss = r_loss + h_loss
                loss_trackers["en"].append(r_loss.item())
                loss_trackers[head_name].append(h_loss.item())

            elif batch_type == "nli":
                # NLI 3-tier contrastive — bottleneck only, no decoder
                type_counts["nli"] = type_counts.get("nli", 0) + 1
                enc_a, enc_b = batch_data[2], batch_data[3]
                labels = batch_data[4]

                ids_a = enc_a["input_ids"].to(device, non_blocking=True)
                mask_a = enc_a["attention_mask"].to(device, non_blocking=True)
                ids_b = enc_b["input_ids"].to(device, non_blocking=True)
                mask_b = enc_b["attention_mask"].to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                vec_a = m.concept_vector(ids_a, mask_a)
                vec_b = m.concept_vector(ids_b, mask_b)

                nli_loss = nli_graded_contrastive_loss(vec_a, vec_b, labels)
                total_loss = NLI_WEIGHT * nli_loss
                loss_trackers["nli"].append(nli_loss.item())

            elif batch_type == "wn_noun":
                # WordNet noun hierarchy — bottleneck only
                type_counts["wn_noun"] = type_counts.get("wn_noun", 0) + 1
                enc_a, enc_b = batch_data[2], batch_data[3]
                distances = batch_data[4]

                ids_a = enc_a["input_ids"].to(device, non_blocking=True)
                mask_a = enc_a["attention_mask"].to(device, non_blocking=True)
                ids_b = enc_b["input_ids"].to(device, non_blocking=True)
                mask_b = enc_b["attention_mask"].to(device, non_blocking=True)
                distances = distances.to(device, non_blocking=True)

                vec_a = m.concept_vector(ids_a, mask_a)
                vec_b = m.concept_vector(ids_b, mask_b)

                wn_loss = wordnet_hierarchy_loss(vec_a, vec_b, distances)
                total_loss = WN_NOUN_WEIGHT * wn_loss
                loss_trackers["wn_noun"].append(wn_loss.item())

            elif batch_type == "wn_axis":
                # WordNet axis consistency — bottleneck only
                type_counts["wn_axis"] = type_counts.get("wn_axis", 0) + 1
                group_sizes = batch_data[1]
                enc_a, enc_b = batch_data[2], batch_data[3]

                ids_a = enc_a["input_ids"].to(device, non_blocking=True)
                mask_a = enc_a["attention_mask"].to(device, non_blocking=True)
                ids_b = enc_b["input_ids"].to(device, non_blocking=True)
                mask_b = enc_b["attention_mask"].to(device, non_blocking=True)

                vec_a = m.concept_vector(ids_a, mask_a)
                vec_b = m.concept_vector(ids_b, mask_b)

                # Split into groups and compute delta vectors
                delta_groups = []
                offset = 0
                for gs in group_sizes:
                    group_a = vec_a[offset:offset + gs]
                    group_b = vec_b[offset:offset + gs]
                    deltas = F.normalize(group_a - group_b, dim=-1)
                    delta_groups.append(deltas)
                    offset += gs

                axis_loss = wordnet_axis_consistency_loss(delta_groups)
                total_loss = WN_AXIS_WEIGHT * axis_loss
                loss_trackers["wn_axis"].append(axis_loss.item())

            elif batch_type == "wn_tropo":
                # WordNet troponym chains — bottleneck only
                type_counts["wn_tropo"] = type_counts.get("wn_tropo", 0) + 1
                chain_info = batch_data[1]
                enc = batch_data[2]

                ids = enc["input_ids"].to(device, non_blocking=True)
                mask = enc["attention_mask"].to(device, non_blocking=True)

                all_vecs = m.concept_vector(ids, mask)

                # Split into chains and compute per-chain loss
                chain_losses = []
                offset = 0
                for n_levels, n_ctx in chain_info:
                    level_vecs = []
                    for lvl in range(n_levels):
                        lvl_vecs = all_vecs[offset:offset + n_ctx]
                        level_vecs.append(lvl_vecs.mean(dim=0))  # avg across contexts
                        offset += n_ctx
                    chain_loss = wordnet_troponym_chain_loss(level_vecs)
                    chain_losses.append(chain_loss)

                if chain_losses:
                    tropo_loss = sum(chain_losses) / len(chain_losses)
                else:
                    tropo_loss = torch.tensor(0.0, device=device)
                total_loss = WN_TROPO_WEIGHT * tropo_loss
                loss_trackers["wn_tropo"].append(tropo_loss.item())

            else:
                continue

            # --- Word order loss (gated on EM EMA > threshold) ---
            wo_val = 0.0
            if exact_match_ema >= WO_GATE_THRESHOLD:
                if wo_gate_step < 0:
                    wo_gate_step = step
                    log(f"  WO GATE OPENED at step {step} "
                        f"(em_ema={exact_match_ema:.3f})")
                c_orig, c_swap = get_word_order_batch(
                    train_geo_gen, geo_tokenizer, model, device, batch_size=16)
                wo_loss, _ = margin_word_order_loss(c_orig, c_swap,
                                                    target_sim=WO_TARGET)
                total_loss = total_loss + WO_WEIGHT * wo_loss
                wo_val = wo_loss.item()

        wo_tracker.append(wo_val)

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
            parts = [f"en={avg_en:.4f}"]
            for h in ["fr"]:
                if loss_trackers[h]:
                    parts.append(f"{h}={np.mean(loss_trackers[h][-50:]):.3f}")
            for h in ["nli", "wn_noun", "wn_axis", "wn_tropo"]:
                if loss_trackers[h]:
                    parts.append(f"{h}={np.mean(loss_trackers[h][-50:]):.3f}")
            avg_wo = np.mean(wo_tracker[-50:]) if wo_tracker else 0
            log(f"step {step+1:>7d} [V20] | {' '.join(parts)} | "
                f"em_ema={exact_match_ema:.3f} | wo={avg_wo:.3f} | "
                f"lr {current_lr:.2e} | {pct:.1f}%")

        # --- Eval ---
        if (step + 1) % EVAL_EVERY == 0:
            results, acc, em = evaluate_reconstruction(
                model, eval_tokenizer, device)
            exact_match_ema = (EXACT_MATCH_EMA_DECAY * exact_match_ema +
                               (1 - EXACT_MATCH_EMA_DECAY) * em)
            log(f"  EN EVAL: token_acc={acc:.3f} exact_match={em:.3f} "
                f"em_ema={exact_match_ema:.3f}")
            for orig, decoded, tacc, exact in results:
                status = "OK" if exact else "DIFF"
                log(f"    [{status}] ({tacc:.0%}) {orig}")
                if not exact:
                    log(f"           -> {decoded}")

            head_accs = {}
            for head_name, test_pairs in HEAD_TEST_PAIRS.items():
                head_upper = head_name.upper()
                head_results, head_acc = evaluate_head(
                    model, eval_tokenizer, test_pairs, head_name, device)
                head_accs[head_name] = head_acc
                log(f"  {head_upper} EVAL: token_acc={head_acc:.3f}")
                for en, ref, pred, tacc in head_results[:2]:
                    log(f"    [{tacc:.0%}] {en} -> {pred}")
                    log(f"           ref: {ref}")

            # NLI ordering probe
            nli_sims, nli_ordered = probe_nli_ordering(
                model, eval_tokenizer, dataset, device)
            log(f"  NLI PROBE: entail={nli_sims['entailment']:.3f} "
                f"neutral={nli_sims['neutral']:.3f} "
                f"contra={nli_sims['contradiction']:.3f} "
                f"ordered={'YES' if nli_ordered else 'NO'}")

            # Geometry probing
            geo = probe_geometry(model, eval_tokenizer, device)
            log(f"  GEOMETRY (TEST): analogy={geo['analogy_avg']:.3f} "
                f"cluster_gap={geo['clustering_gap']:+.4f} "
                f"dir_con={geo.get('dir_consistency', 0):.3f} "
                f"wo_sim={geo['word_order_sim']:.3f} "
                f"rank90={geo['rank90']} rank95={geo['rank95']}")

            total_types = sum(type_counts.values()) or 1
            dist = " ".join(f"{h}={type_counts[h]/total_types:.0%}"
                            for h in DATA_WEIGHTS)
            log(f"  TYPE DIST: {dist} | wo_gate_step={wo_gate_step}")

            elapsed = time.time() - start_time
            metrics_dict = {
                "recon_loss": avg_en if loss_trackers["en"] else 0,
                "token_acc": acc, "exact_match": em,
                "em_ema": exact_match_ema, "lr": current_lr,
                "elapsed_hours": elapsed / 3600, "geo": geo,
                "fr_loss": np.mean(loss_trackers["fr"][-50:]) if loss_trackers["fr"] else 0,
                "fr_token_acc": head_accs.get("fr", 0),
                "nli_loss": np.mean(loss_trackers["nli"][-50:]) if loss_trackers["nli"] else 0,
                "wn_noun_loss": np.mean(loss_trackers["wn_noun"][-50:]) if loss_trackers["wn_noun"] else 0,
                "wn_axis_loss": np.mean(loss_trackers["wn_axis"][-50:]) if loss_trackers["wn_axis"] else 0,
                "wn_tropo_loss": np.mean(loss_trackers["wn_tropo"][-50:]) if loss_trackers["wn_tropo"] else 0,
                "wo_loss": np.mean(wo_tracker[-50:]) if wo_tracker else 0,
                "nli_ordering_acc": 1.0 if nli_ordered else 0.0,
                "wn_noun_corr": 0.0,  # TODO: Spearman correlation probe
            }
            log_metrics(step + 1, metrics_dict)

        # --- Checkpoint ---
        if (step + 1) % CHECKPOINT_EVERY == 0:
            avg_loss = np.mean(loss_trackers["en"][-100:]) if loss_trackers["en"] else 0
            save_checkpoint(model, optimizer, scaler, config, step + 1,
                            avg_loss, exact_match_ema, wo_gate_step,
                            CHECKPOINT_DIR)

        if shutdown_requested:
            break

    prefetch.stop()
    if loss_trackers["en"]:
        avg_loss = np.mean(loss_trackers["en"][-100:])
        save_checkpoint(model, optimizer, scaler, config,
                        step + 1 if not shutdown_requested else step,
                        avg_loss, exact_match_ema, wo_gate_step,
                        CHECKPOINT_DIR)
    log("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    train(resume_from=args.resume, fresh=args.fresh, eval_only=args.eval_only)
