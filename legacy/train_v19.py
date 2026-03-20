"""
FLM V19 — Multilingual Hydra with shared mBERT tokenizer + contrastive loss
================================================================================
9 decoder heads (EN, FR, ES, DE, PT, ZH, JA, Paraphrase, Semantic Parse) sharing
one lm_head. All heads use bert-base-multilingual-cased (vocab_size=119547).
4L decoders per head (9 heads → need to keep params manageable).
Bottleneck: 32×16 = 512 dims.

Key changes from V18:
  - ConceptAutoencoderV19: 9 heads with shared lm_head, single mBERT tokenizer
  - NO explicit geometry losses (no direction_consistency, cluster_separation,
    analogy, hard_repulsion, batch_repulsion)
  - YES contrastive loss (InfoNCE) on bottleneck: translations/paraphrases
    should have similar concept vectors
  - YES word-order margin loss (gated on EM EMA > 0.5)
  - 4 new language heads: DE, PT, ZH, JA (from wikimatrix data)

Usage:
    python train_v19.py --fresh            # start from scratch
    python train_v19.py                    # auto-resume
    python train_v19.py --eval-only        # diagnostics only
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
from concept_model import (ConceptConfig, ConceptAutoencoderV19,
                           reconstruction_loss,
                           flat_similarity_matrix,
                           margin_word_order_loss)
from geometry_data import GeometryDataGenerator, verify_splits

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/concept_v19"
LOG_DIR = "logs"

MODEL_CONFIG = dict(
    vocab_size=119547,      # mBERT shared vocab for everything
    enc_hidden=384,
    enc_layers=6,
    enc_heads=6,
    enc_intermediate=1536,
    num_concepts=32,
    concept_dim=16,         # 32×16 = 512 bottleneck
    dec_hidden=384,
    dec_layers=4,           # 4L per head (9 heads, need to keep params reasonable)
    dec_heads=6,
    dec_intermediate=1536,
    max_seq_len=128,
    dropout=0.1,
)

# Training hyperparameters
BATCH_SIZE = 16  # smaller for 203M model + contrastive double-encode on 24GB
PEAK_LR = 2e-4
MIN_LR = 1e-5
WARMUP_STEPS = 2000
TOTAL_STEPS = 600_000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0

# Data sampling weights
DATA_WEIGHTS = {
    "fr": 0.12,
    "es": 0.12,
    "de": 0.12,
    "pt": 0.12,
    "zh": 0.12,
    "ja": 0.12,
    "para": 0.14,
    "parse": 0.14,
}

# Contrastive loss config
CONTRASTIVE_WEIGHT = 0.5   # weight for contrastive loss
CONTRASTIVE_TEMP = 0.07    # temperature for InfoNCE

# Word order loss config
WO_WEIGHT = 2.0            # word order margin loss weight
WO_TARGET = 0.5            # target sim for word order
WO_GATE_THRESHOLD = 0.5    # EN EM EMA threshold for word order activation

# EMA tracking
EXACT_MATCH_EMA_DECAY = 0.99

# Logging
LOG_EVERY = 50
EVAL_EVERY = 500
CHECKPOINT_EVERY = 5000

LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v19.log",
    "metrics": f"{LOG_DIR}/concept_v19_metrics.csv",
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

# DE translation test pairs
DE_TEST_PAIRS = [
    ("The situation is very serious", "Die Lage ist sehr ernst"),
    ("We need to find a solution", "Wir müssen eine Lösung finden"),
    ("This is a very important issue", "Dies ist eine sehr wichtige Frage"),
    ("The vote will take place tomorrow", "Die Abstimmung findet morgen statt"),
]

# PT translation test pairs
PT_TEST_PAIRS = [
    ("The situation is very serious", "A situação é muito grave"),
    ("We need to find a solution", "Precisamos encontrar uma solução"),
    ("This is a very important issue", "Esta é uma questão muito importante"),
    ("The vote will take place tomorrow", "A votação terá lugar amanhã"),
]

# ZH translation test pairs
ZH_TEST_PAIRS = [
    ("The situation is very serious", "情况非常严重"),
    ("We need to find a solution", "我们需要找到一个解决方案"),
    ("This is a very important issue", "这是一个非常重要的问题"),
    ("The cat sat on the mat", "猫坐在垫子上"),
]

# JA translation test pairs
JA_TEST_PAIRS = [
    ("The situation is very serious", "状況は非常に深刻です"),
    ("We need to find a solution", "解決策を見つける必要があります"),
    ("This is a very important issue", "これは非常に重要な問題です"),
    ("The cat sat on the mat", "猫はマットの上に座った"),
]

# Parse test pairs
PARSE_TEST_PAIRS = [
    ("the dog bit the man", "subject : the dog | action : bite | object : the man"),
    ("she runs every morning", "subject : she | action : run | location : every morning"),
    ("he did not enjoy the movie", "subject : he | action : enjoy | negation : true | object : the movie"),
    ("the cat chased the mouse quickly", "subject : the cat | action : chase | object : the mouse | manner : quickly"),
]

# Head → test pairs mapping
HEAD_TEST_PAIRS = {
    "fr": FR_TEST_PAIRS,
    "es": ES_TEST_PAIRS,
    "de": DE_TEST_PAIRS,
    "pt": PT_TEST_PAIRS,
    "zh": ZH_TEST_PAIRS,
    "ja": JA_TEST_PAIRS,
    "parse": PARSE_TEST_PAIRS,
}

# Decode function name mapping (head name → decode_head argument)
DECODE_FNS = {
    "fr": "fr", "es": "es", "de": "de", "pt": "pt",
    "zh": "zh", "ja": "ja", "para": "para", "parse": "parse",
}


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

def contrastive_loss(concepts_a, concepts_b, temperature=0.07):
    """InfoNCE contrastive loss on flattened bottleneck vectors."""
    a = F.normalize(concepts_a.view(concepts_a.shape[0], -1), dim=-1)
    b = F.normalize(concepts_b.view(concepts_b.shape[0], -1), dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(len(a), device=a.device)
    return F.cross_entropy(logits, labels)


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
                  "de_loss,de_token_acc,pt_loss,pt_token_acc,"
                  "zh_loss,zh_token_acc,ja_loss,ja_token_acc,"
                  "para_loss,para_token_acc,parse_loss,parse_token_acc,"
                  "ctr_loss,wo_loss\n")


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
                f"{m.get('es_loss',0):.6f},{m.get('es_token_acc',0):.4f},"
                f"{m.get('de_loss',0):.6f},{m.get('de_token_acc',0):.4f},"
                f"{m.get('pt_loss',0):.6f},{m.get('pt_token_acc',0):.4f},"
                f"{m.get('zh_loss',0):.6f},{m.get('zh_token_acc',0):.4f},"
                f"{m.get('ja_loss',0):.6f},{m.get('ja_token_acc',0):.4f},"
                f"{m.get('para_loss',0):.6f},{m.get('para_token_acc',0):.4f},"
                f"{m.get('parse_loss',0):.6f},{m.get('parse_token_acc',0):.4f},"
                f"{m.get('ctr_loss',0):.6f},{m.get('wo_loss',0):.6f}\n")


# ---------------------------------------------------------------------------
# Data loading — multi-source with shared mBERT tokenizer
# ---------------------------------------------------------------------------

class MultiSourceDataset:
    """Loads all data sources for the 9-head hydra decoder."""

    def __init__(self, tokenizer, max_len=128):
        self.tok = tokenizer
        self.max_len = max_len

        self.fr_pairs = []
        self.es_pairs = []
        self.de_pairs = []
        self.pt_pairs = []
        self.zh_pairs = []
        self.ja_pairs = []
        self.para_pairs = []
        self.parse_pairs = []

        self._load()

    def _load(self):
        # FR data
        for src in ["europarl.jsonl", "tatoeba_enfr.jsonl", "wikimatrix_enfr.jsonl"]:
            self._load_pairs(f"data/pairs/{src}", self.fr_pairs, "FR")
        # ES data
        for src in ["europarl_enes.jsonl", "tatoeba_enes.jsonl", "wikimatrix_enes.jsonl"]:
            self._load_pairs(f"data/pairs/{src}", self.es_pairs, "ES")
        # DE data
        for src in ["wikimatrix_ende.jsonl"]:
            self._load_pairs(f"data/pairs/{src}", self.de_pairs, "DE")
        # PT data
        for src in ["wikimatrix_enpt.jsonl"]:
            self._load_pairs(f"data/pairs/{src}", self.pt_pairs, "PT")
        # ZH data
        for src in ["wikimatrix_enzh.jsonl"]:
            self._load_pairs(f"data/pairs/{src}", self.zh_pairs, "ZH")
        # JA data
        for src in ["wikimatrix_enja.jsonl"]:
            self._load_pairs(f"data/pairs/{src}", self.ja_pairs, "JA")
        # Paraphrase data
        for src in ["paws.jsonl", "qqp.jsonl", "mrpc.jsonl"]:
            self._load_pairs(f"data/pairs/{src}", self.para_pairs, "PARA",
                             require_label=1)
        # NLI entailment pairs as paraphrases
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
        # Semantic parse data
        self._load_pairs("data/pairs/semantic_parse.jsonl", self.parse_pairs, "PARSE")

        for pairs in [self.fr_pairs, self.es_pairs, self.de_pairs,
                      self.pt_pairs, self.zh_pairs, self.ja_pairs,
                      self.para_pairs, self.parse_pairs]:
            random.shuffle(pairs)

        log(f"  TOTALS: FR={len(self.fr_pairs):,} ES={len(self.es_pairs):,} "
            f"DE={len(self.de_pairs):,} PT={len(self.pt_pairs):,} "
            f"ZH={len(self.zh_pairs):,} JA={len(self.ja_pairs):,} "
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

    def _get_pair_list(self, head):
        mapping = {
            "fr": self.fr_pairs, "es": self.es_pairs,
            "de": self.de_pairs, "pt": self.pt_pairs,
            "zh": self.zh_pairs, "ja": self.ja_pairs,
            "para": self.para_pairs, "parse": self.parse_pairs,
        }
        pairs = mapping.get(head)
        if pairs is None:
            raise ValueError(f"Unknown head: {head}")
        return pairs

    def get_batch(self, batch_size, head):
        pairs = self._get_pair_list(head)
        if not pairs:
            raise ValueError(f"No data for head '{head}'")

        indices = [random.randint(0, len(pairs) - 1) for _ in range(batch_size)]
        en_texts = [pairs[i][0] for i in indices]
        tgt_texts = [pairs[i][1] for i in indices]

        # Shared mBERT tokenizer for everything
        en_enc = self.tok(en_texts, max_length=self.max_len,
                          padding=True, truncation=True, return_tensors="pt")
        tgt_enc = self.tok(tgt_texts, max_length=self.max_len,
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

        # Build cumulative weights for head sampling
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
                head = self._heads[i]
                # Check if head has data
                pairs = self.dataset._get_pair_list(head)
                if pairs:
                    return head
                # Fallback to a head that has data
                break
        # Fallback: pick any head with data
        for h in self._heads:
            pairs = self.dataset._get_pair_list(h)
            if pairs:
                return h
        return self._heads[0]  # last resort

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
            except ValueError:
                # No data for this head, skip
                continue

    def get(self):
        return self.q.get()


# ---------------------------------------------------------------------------
# Geometry batch generation (word order only, using GeometryDataGenerator)
# ---------------------------------------------------------------------------

def _encode_concepts(model, tokenizer, texts, device):
    """Encode texts -> flat bottleneck vectors (B, num_concepts * concept_dim)."""
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    enc = tokenizer(texts, max_length=64, padding=True, truncation=True, return_tensors="pt")
    concepts = m.encode(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return concepts.view(concepts.shape[0], -1)


def get_word_order_batch(geo_gen, tokenizer, model, device, batch_size=16):
    """Get a batch of word-order swap pairs, encode via bottleneck."""
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
    """Evaluate a decoder head (translation, paraphrase, or parse)."""
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
    """Probe geometry using TEST split -- measures genuine generalization."""
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

    within_sims = []
    between_sims = []
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

    # Word order sensitivity
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
        "version": "v19",
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
    log(f"Loading V19 checkpoint from {path}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoderV19(config).to(device)
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
    log(f"Resumed V19: {total:,} params | step {step} | em_ema={exact_match_ema:.3f}")
    return model, optimizer, scaler, config, step, exact_match_ema, wo_gate_step


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(resume_from=None, fresh=False, eval_only=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("FLM V19 — MULTILINGUAL HYDRA + CONTRASTIVE LOSS")
    log("  Bottleneck: 32x16 = 512 dims | 4L decoders x 9 heads")
    log("  Heads: EN FR ES DE PT ZH JA Para Parse (shared mBERT + lm_head)")
    log("  Losses: recon + head + contrastive (InfoNCE) + word_order (gated)")
    log("  NO explicit geometry losses (contrastive replaces them)")
    log("=" * 70)

    verify_splits()

    from transformers import AutoTokenizer

    # Single shared mBERT tokenizer for all languages
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    # Separate instance for geometry/eval (avoids "Already borrowed" from
    # concurrent fast-tokenizer use between the prefetch thread and main loop)
    eval_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    geo_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    train_geo_gen = GeometryDataGenerator(split="train", seed=None)

    log(f"mBERT tokenizer: vocab={tokenizer.vocab_size}")

    model_config = dict(MODEL_CONFIG)
    model_config["vocab_size"] = tokenizer.vocab_size

    exact_match_ema = 0.0
    wo_gate_step = -1  # step when word order gate opened (-1 = not yet)

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
        model = ConceptAutoencoderV19(config).to(device)
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
            log(f"\n--- {head_upper} EVAL ---")
            head_results, head_acc = evaluate_head(
                model, eval_tokenizer, test_pairs, head_name, device)
            log(f"  {head_upper} token_acc={head_acc:.3f}")
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
    dataset = MultiSourceDataset(tokenizer, max_len=config.max_seq_len)

    total, _ = _unwrap(model).count_parameters()
    log(f"\nTraining plan (V19 — Multilingual Hydra + Contrastive):")
    log(f"  Model: {total:,} params ({total/1e6:.1f}M)")
    log(f"  Bottleneck: {config.num_concepts}x{config.concept_dim} = "
        f"{config.num_concepts * config.concept_dim} dims")
    log(f"  Heads: " + " ".join(f"{h.upper()}({config.dec_layers}L)"
                                for h in ConceptAutoencoderV19.HEAD_NAMES))
    log(f"  Contrastive: weight={CONTRASTIVE_WEIGHT} temp={CONTRASTIVE_TEMP}")
    log(f"  Word order: weight={WO_WEIGHT} target={WO_TARGET} gate@EM>{WO_GATE_THRESHOLD}")
    log(f"  Batch: {BATCH_SIZE}")
    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")
    log(f"  Sampling: " + " ".join(f"{h}={w:.0%}" for h, w in DATA_WEIGHTS.items()))
    log("-" * 70)

    prefetch = HydraPrefetchBuffer(dataset, device, batch_size=BATCH_SIZE)
    prefetch.start()
    log("Prefetch buffer started")

    # Shared pad token ID (mBERT uses 0 as pad)
    PAD_ID = tokenizer.pad_token_id or 0

    all_heads = list(DATA_WEIGHTS.keys())
    loss_trackers = {h: [] for h in ["en"] + all_heads}
    ctr_tracker = []
    wo_tracker = []
    head_counts = {h: 0 for h in all_heads}

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

            # Encode EN input
            concepts = m.encode(en_ids, en_mask)

            # EN reconstruction (always)
            en_logits = m.decode_en(concepts, en_ids.shape[1], en_mask)
            r_loss = reconstruction_loss(en_logits, en_ids)

            # Secondary head loss
            head_logits = m.decode_head(head, concepts, tgt_ids.shape[1], tgt_mask)
            h_loss = F.cross_entropy(
                head_logits.reshape(-1, head_logits.shape[-1]),
                tgt_ids.reshape(-1),
                ignore_index=PAD_ID)

            total_loss = r_loss + h_loss

            # --- Contrastive loss (skip for parse — structured output) ---
            ctr_val = 0.0
            if head != "parse":
                tgt_concepts = m.encode(tgt_ids, tgt_mask)
                ctr_loss = contrastive_loss(concepts, tgt_concepts,
                                            temperature=CONTRASTIVE_TEMP)
                total_loss = total_loss + CONTRASTIVE_WEIGHT * ctr_loss
                ctr_val = ctr_loss.item()

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

        r_val = r_loss.item()
        h_val = h_loss.item()
        loss_trackers["en"].append(r_val)
        loss_trackers[head].append(h_val)
        ctr_tracker.append(ctr_val)
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
            head_losses = []
            for h in all_heads:
                if loss_trackers[h]:
                    head_losses.append(f"{h}={np.mean(loss_trackers[h][-50:]):.3f}")
            hl_str = " ".join(head_losses) if head_losses else "warming up"
            avg_ctr = np.mean(ctr_tracker[-50:]) if ctr_tracker else 0
            avg_wo = np.mean(wo_tracker[-50:]) if wo_tracker else 0
            log(f"step {step+1:>7d} [V19] | en={avg_en:.4f} {hl_str} | "
                f"em_ema={exact_match_ema:.3f} | "
                f"ctr={avg_ctr:.3f} wo={avg_wo:.3f} | "
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

            # Per-head eval
            head_accs = {}
            for head_name, test_pairs in HEAD_TEST_PAIRS.items():
                head_upper = head_name.upper()
                head_results, head_acc = evaluate_head(
                    model, eval_tokenizer, test_pairs, head_name, device)
                head_accs[head_name] = head_acc
                log(f"  {head_upper} EVAL: token_acc={head_acc:.3f}")
                # Show first 2 examples
                for en, ref, pred, tacc in head_results[:2]:
                    log(f"    [{tacc:.0%}] {en} -> {pred}")
                    log(f"           ref: {ref}")

            # Geometry probing
            geo = probe_geometry(model, eval_tokenizer, device)
            log(f"  GEOMETRY (TEST): analogy={geo['analogy_avg']:.3f} "
                f"cluster_gap={geo['clustering_gap']:+.4f} "
                f"dir_con={geo.get('dir_consistency', 0):.3f} "
                f"wo_sim={geo['word_order_sim']:.3f} "
                f"rank90={geo['rank90']} rank95={geo['rank95']}")

            total_heads = sum(head_counts.values()) or 1
            dist = " ".join(f"{h}={head_counts[h]/total_heads:.0%}"
                            for h in all_heads)
            log(f"  HEAD DIST: {dist} | wo_gate_step={wo_gate_step}")

            elapsed = time.time() - start_time
            avg_en = np.mean(loss_trackers["en"][-100:]) if loss_trackers["en"] else 0
            metrics_dict = {
                "recon_loss": avg_en, "token_acc": acc, "exact_match": em,
                "em_ema": exact_match_ema, "lr": current_lr,
                "elapsed_hours": elapsed / 3600, "geo": geo,
                "ctr_loss": np.mean(ctr_tracker[-50:]) if ctr_tracker else 0,
                "wo_loss": np.mean(wo_tracker[-50:]) if wo_tracker else 0,
            }
            for h in all_heads:
                avg_h = np.mean(loss_trackers[h][-50:]) if loss_trackers[h] else 0
                metrics_dict[f"{h}_loss"] = avg_h
                metrics_dict[f"{h}_token_acc"] = head_accs.get(h, 0)
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
    parser = argparse.ArgumentParser(
        description="Train concept autoencoder V19 (multilingual hydra + contrastive)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    train(resume_from=args.resume, fresh=args.fresh, eval_only=args.eval_only)
