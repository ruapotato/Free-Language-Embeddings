#!/usr/bin/env python3
"""
FLM V32 — Path2vec: Skip-Gram Embeddings for Filesystem Paths

Builds word2vec-style embeddings for Unix filesystem path components.
Treats each path as a "sentence" of components and trains skip-gram
with negative sampling — identical to V28 but on filesystem data.

Data source: A Debian chroot with many packages installed, plus
mounted /proc, /sys, /dev for virtual filesystem structure.

The idea: just as word2vec learns king - man + woman = queen from
text co-occurrence, path2vec might learn structural relationships
like /usr/lib - lib + bin = /usr/bin.

Tokens = path components (split on '/')
"Sentences" = full paths

Phase 1: Build Debian chroot with lots of packages (requires sudo)
Phase 2: Walk filesystem, collect paths, build vocabulary
Phase 3: Train skip-gram on path components

Usage:
    python train_v32.py --setup-chroot   # Phase 1: build chroot (needs sudo)
    python train_v32.py --collect        # Phase 2: collect paths + build vocab
    python train_v32.py --fresh          # Phase 3: train from scratch
    python train_v32.py --resume         # Phase 3: resume training
    python train_v32.py --all            # All phases
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
import subprocess
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_DIM = 300          # Same as V28
WINDOW_SIZE = 3          # Smaller than V28 (paths are shorter than sentences)
NEG_SAMPLES = 15         # Same as V28
BATCH_SIZE = 4096        # Large batches for skip-gram
TOTAL_STEPS = 1_000_000
PEAK_LR = 0.025
MIN_LR = 1e-4
WARMUP_STEPS = 1000
SUBSAMPLE_THRESHOLD = 1e-3  # Higher than V28 — path components less Zipfian
LOG_EVERY = 50
EVAL_EVERY = 5000
SAVE_EVERY = 10000
MIN_COUNT = 3            # Lower than V28 — we want rare path components
MAX_VOCAB = 200_000      # Allow large vocab for all unique components

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints/path2vec_v32"
LOG_DIR = "logs"
LOG_PATHS = {
    "log": f"{LOG_DIR}/concept_v32.log",
    "metrics": f"{LOG_DIR}/concept_v32_metrics.csv",
}

CHROOT_DIR = "/tmp/debian_chroot"
PATHS_FILE = os.path.join(CHECKPOINT_DIR, "paths.json")
VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "vocab.json")

# Packages to install for maximum filesystem coverage
CHROOT_PACKAGES = [
    # Desktop environments & display
    "xfce4", "lxde", "i3", "openbox",
    # Development
    "build-essential", "cmake", "autoconf", "automake", "libtool",
    "git", "subversion", "mercurial",
    "python3", "python3-pip", "python3-venv", "python3-dev",
    "python3-numpy", "python3-scipy", "python3-matplotlib",
    "ruby", "perl", "lua5.4", "gawk",
    "gcc", "g++", "gfortran", "gdb", "valgrind", "strace",
    "default-jdk", "ant", "maven",
    "nodejs", "npm",
    "golang-go", "rustc", "cargo",
    # Editors
    "vim", "emacs-nox", "nano", "ed",
    # Servers
    "apache2", "nginx", "lighttpd",
    "postgresql", "mariadb-server", "sqlite3",
    "redis-server", "memcached",
    "postfix", "dovecot-imapd",
    "openssh-server", "vsftpd", "proftpd-basic",
    "bind9", "dnsmasq",
    "samba", "nfs-kernel-server",
    # System tools
    "systemd", "cron", "at", "logrotate",
    "rsyslog", "fail2ban", "ufw",
    "lvm2", "mdadm", "cryptsetup",
    "parted", "gdisk", "dosfstools", "ntfs-3g",
    "iproute2", "iptables", "nftables",
    "tcpdump", "nmap", "netcat-openbsd", "curl", "wget",
    "htop", "iotop", "sysstat", "dstat",
    "tmux", "screen",
    # Libraries (adds tons of /usr/lib and /usr/include paths)
    "libssl-dev", "libcurl4-openssl-dev", "libxml2-dev",
    "libsqlite3-dev", "libpq-dev", "libmysqlclient-dev",
    "libboost-all-dev", "libglib2.0-dev",
    "libgtk-3-dev", "libqt5-dev",
    "libpng-dev", "libjpeg-dev", "libtiff-dev",
    "libfreetype-dev", "libfontconfig-dev",
    "libx11-dev", "libxext-dev", "libxrandr-dev",
    "libasound2-dev", "libpulse-dev",
    "libavcodec-dev", "libavformat-dev", "libswscale-dev",
    # Documentation (lots of /usr/share/doc, /usr/share/man)
    "man-db", "manpages", "manpages-dev",
    "info", "texinfo",
    "doc-base",
    # TeX (enormous file tree)
    "texlive-base", "texlive-latex-base", "texlive-fonts-recommended",
    # Misc tools
    "imagemagick", "ffmpeg", "sox",
    "zip", "unzip", "p7zip-full", "xz-utils", "zstd",
    "jq", "xmlstarlet",
    "gnupg", "ca-certificates",
    "locales", "tzdata",
    "fonts-dejavu", "fonts-liberation", "fonts-noto",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
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
# Phase 1: Build Debian chroot
# ---------------------------------------------------------------------------

def setup_chroot():
    """Build a Debian chroot with many packages for maximum path coverage."""
    log("=" * 70)
    log("Phase 1: Building Debian chroot")
    log("=" * 70)

    if os.path.exists(CHROOT_DIR):
        log(f"Chroot already exists at {CHROOT_DIR}")
        log("To rebuild, remove it first: sudo rm -rf /tmp/debian_chroot")
        return

    # Bootstrap minimal Debian
    log(f"Running debootstrap into {CHROOT_DIR}...")
    result = subprocess.run(
        ["sudo", "debootstrap", "--variant=minbase", "trixie", CHROOT_DIR],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        log(f"debootstrap failed: {result.stderr}")
        raise RuntimeError("debootstrap failed")
    log("Base system installed.")

    # Mount virtual filesystems
    mount_virtual_fs()

    # Install packages in batches (some may fail, that's ok)
    log(f"Installing {len(CHROOT_PACKAGES)} packages...")
    batch_size = 20
    for i in range(0, len(CHROOT_PACKAGES), batch_size):
        batch = CHROOT_PACKAGES[i:i+batch_size]
        log(f"  Installing batch {i//batch_size + 1}: {', '.join(batch[:5])}...")
        result = subprocess.run(
            ["sudo", "chroot", CHROOT_DIR, "apt-get", "install", "-y",
             "--no-install-recommends"] + batch,
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
        )
        if result.returncode != 0:
            # Try packages individually
            for pkg in batch:
                subprocess.run(
                    ["sudo", "chroot", CHROOT_DIR, "apt-get", "install", "-y",
                     "--no-install-recommends", pkg],
                    capture_output=True, text=True, timeout=120,
                    env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
                )
        installed = subprocess.run(
            ["sudo", "chroot", CHROOT_DIR, "dpkg", "--list"],
            capture_output=True, text=True
        )
        pkg_count = installed.stdout.count("\nii ")
        log(f"  {pkg_count} packages installed so far")

    log("Package installation complete.")


def mount_virtual_fs():
    """Mount /proc, /sys, /dev into the chroot."""
    log("Mounting virtual filesystems...")
    mounts = [
        ("proc", f"{CHROOT_DIR}/proc", "proc"),
        ("sysfs", f"{CHROOT_DIR}/sys", "sysfs"),
        ("/dev", f"{CHROOT_DIR}/dev", None),  # bind mount
    ]
    for src, dst, fstype in mounts:
        os.makedirs(dst, exist_ok=True)
        if os.path.ismount(dst):
            log(f"  Already mounted: {dst}")
            continue
        if fstype:
            subprocess.run(["sudo", "mount", "-t", fstype, src, dst],
                         capture_output=True)
        else:
            subprocess.run(["sudo", "mount", "--bind", src, dst],
                         capture_output=True)
        log(f"  Mounted: {dst}")


def unmount_virtual_fs():
    """Unmount virtual filesystems from chroot."""
    log("Unmounting virtual filesystems...")
    for subdir in ["dev", "sys", "proc"]:
        path = f"{CHROOT_DIR}/{subdir}"
        if os.path.ismount(path):
            subprocess.run(["sudo", "umount", "-l", path], capture_output=True)
            log(f"  Unmounted: {path}")


# ---------------------------------------------------------------------------
# Phase 2: Collect paths and build vocabulary
# ---------------------------------------------------------------------------

def path_to_tokens(path):
    """Split a filesystem path into component tokens.

    /usr/share/doc/python3 → ["usr", "share", "doc", "python3"]

    Special handling:
    - Root "/" is skipped (implicit)
    - Empty components from double slashes are skipped
    - Very long components (>100 chars) are skipped
    """
    parts = [p for p in path.split("/") if p and len(p) <= 100]
    return parts


def collect_paths():
    """Walk the chroot filesystem and collect all paths."""
    log("=" * 70)
    log("Phase 2: Collecting filesystem paths")
    log("=" * 70)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Make sure virtual fs is mounted
    if os.path.exists(CHROOT_DIR):
        mount_virtual_fs()

    root = CHROOT_DIR if os.path.exists(CHROOT_DIR) else "/"
    log(f"Walking filesystem from: {root}")

    # Directories to skip even within chroot
    skip_prefixes = []
    if root == "/":
        skip_prefixes = ["/home", "/mnt", "/tmp", "/media", "/root"]

    all_paths = []
    path_count = 0
    dir_count = 0
    file_count = 0
    symlink_count = 0
    error_count = 0

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Get the path relative to root
        rel = dirpath[len(root):] if root != "/" else dirpath
        if not rel:
            rel = "/"

        # Skip excluded directories
        skip = False
        for prefix in skip_prefixes:
            if rel.startswith(prefix):
                skip = True
                break
        if skip:
            continue

        # Record the directory itself
        all_paths.append(rel)
        dir_count += 1

        # Record all files
        for fname in filenames:
            fpath = os.path.join(rel, fname) if rel != "/" else f"/{fname}"
            all_paths.append(fpath)
            full = os.path.join(dirpath, fname)
            if os.path.islink(full):
                symlink_count += 1
            else:
                file_count += 1

        path_count = len(all_paths)
        if path_count % 100000 == 0 and path_count > 0:
            log(f"  {path_count:,} paths collected ({dir_count:,} dirs, "
                f"{file_count:,} files, {symlink_count:,} symlinks)...")

        # Safety valve — /proc can be enormous
        if path_count > 5_000_000:
            log("  Hit 5M path limit, stopping walk")
            break

    log(f"Total paths: {len(all_paths):,}")
    log(f"  Directories: {dir_count:,}")
    log(f"  Files: {file_count:,}")
    log(f"  Symlinks: {symlink_count:,}")

    # Tokenize and build vocabulary
    all_tokenized = [path_to_tokens(p) for p in all_paths]
    all_tokenized = [t for t in all_tokenized if len(t) >= 2]  # need at least 2 components

    # Token statistics
    token_counts = Counter()
    lengths = []
    for tokens in all_tokenized:
        token_counts.update(tokens)
        lengths.append(len(tokens))

    log(f"Tokenized paths: {len(all_tokenized):,} (with ≥2 components)")
    log(f"Unique tokens: {len(token_counts):,}")
    log(f"Avg path length: {np.mean(lengths):.1f} components")
    log(f"Max path length: {max(lengths)} components")
    log(f"Top 30 tokens: {', '.join(f'{w}({c})' for w, c in token_counts.most_common(30))}")

    # Save paths
    with open(PATHS_FILE, "w") as f:
        json.dump({"paths": all_tokenized}, f)
    log(f"Paths saved to {PATHS_FILE}")

    # Build vocabulary
    build_vocab(all_tokenized, token_counts)

    return all_tokenized


def build_vocab(all_tokenized, token_counts):
    """Build vocabulary from collected paths."""
    # Filter by min count
    filtered = [(w, c) for w, c in token_counts.most_common() if c >= MIN_COUNT]
    if len(filtered) > MAX_VOCAB:
        filtered = filtered[:MAX_VOCAB]

    word2id = {w: i for i, (w, _) in enumerate(filtered)}
    counts = [c for _, c in filtered]
    total_count = sum(counts)

    vocab_data = {
        "word2id": word2id,
        "counts": counts,
        "total_count": total_count,
    }
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab_data, f)

    log(f"Vocabulary: {len(word2id):,} tokens (min_count={MIN_COUNT})")
    log(f"Top 20: {', '.join(w for w, _ in filtered[:20])}")
    log(f"Saved to {VOCAB_PATH}")


# ---------------------------------------------------------------------------
# Vocabulary (reused from V28)
# ---------------------------------------------------------------------------

class Vocabulary:
    """Path component vocabulary with frequency counts."""

    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.counts = []
        self.total_count = 0

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
        log(f"  Vocabulary loaded: {len(self.word2id):,} tokens from {path}")

    def encode(self, tokens):
        """Convert token list to id list, skipping unknown tokens."""
        return [self.word2id[t] for t in tokens if t in self.word2id]

    def decode(self, ids):
        return [self.id2word.get(i, "<unk>") for i in ids]

    def __len__(self):
        return len(self.word2id)

    def __contains__(self, token):
        return token in self.word2id


# ---------------------------------------------------------------------------
# Word2vec Model (identical to V28)
# ---------------------------------------------------------------------------

class SkipGram(nn.Module):
    """Skip-gram with negative sampling."""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)

        initrange = 0.5 / embed_dim
        nn.init.uniform_(self.target_embeddings.weight, -initrange, initrange)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, target_ids, context_ids, neg_ids):
        target_emb = self.target_embeddings(target_ids)
        context_emb = self.context_embeddings(context_ids)
        neg_emb = self.context_embeddings(neg_ids)

        pos_score = (target_emb * context_emb).sum(dim=-1)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(neg_emb, target_emb.unsqueeze(-1)).squeeze(-1)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=-1)

        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_embeddings(self):
        return self.target_embeddings.weight.detach()


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

class PathPairDataset:
    """Streams (target, context) skip-gram pairs from filesystem paths."""

    def __init__(self, paths_file, vocab, window_size=WINDOW_SIZE,
                 subsample_threshold=SUBSAMPLE_THRESHOLD):
        self.vocab = vocab
        self.window_size = window_size

        # Load tokenized paths
        log(f"Loading paths from {paths_file}...")
        with open(paths_file) as f:
            data = json.load(f)
        raw_paths = data["paths"]

        # Encode all paths to IDs
        self.paths = []
        for tokens in raw_paths:
            ids = vocab.encode(tokens)
            if len(ids) >= 2:
                self.paths.append(ids)
        log(f"  {len(self.paths):,} paths with ≥2 known tokens")

        # Subsampling probabilities
        self._subsample_probs = np.ones(len(vocab), dtype=np.float32)
        for i, count in enumerate(vocab.counts):
            freq = count / max(vocab.total_count, 1)
            if freq > 0:
                self._subsample_probs[i] = min(1.0,
                    (math.sqrt(freq / subsample_threshold) + 1) *
                    (subsample_threshold / freq))

        # Negative sampling table
        freqs = np.array(vocab.counts, dtype=np.float64)
        freqs = np.power(freqs, 0.75)
        self._neg_probs = freqs / freqs.sum()
        self._neg_table_size = 10_000_000
        self._neg_table = torch.from_numpy(
            np.random.choice(len(vocab), size=self._neg_table_size, p=self._neg_probs)
        ).long()
        self._neg_idx = 0

        # Pair buffer
        self._pair_buf = []
        self._MIN_BUF = 100000
        self._path_idx = 0

        log(f"PathPairDataset: {len(self.paths):,} paths, "
            f"vocab={len(vocab):,}, window={window_size}")

    def _extract_pairs(self, token_ids):
        """Extract (target, context) skip-gram pairs from a path."""
        pairs = []
        n = len(token_ids)
        for i in range(n):
            w = random.randint(1, self.window_size)
            for j in range(max(0, i - w), min(n, i + w + 1)):
                if j != i:
                    # Subsample
                    if random.random() < self._subsample_probs[token_ids[i]]:
                        pairs.append((token_ids[i], token_ids[j]))
        return pairs

    def _refill_buffer(self):
        while len(self._pair_buf) < self._MIN_BUF:
            path = self.paths[self._path_idx % len(self.paths)]
            self._path_idx += 1
            if self._path_idx % len(self.paths) == 0:
                random.shuffle(self.paths)
            self._pair_buf.extend(self._extract_pairs(path))
        random.shuffle(self._pair_buf)

    def sample_negatives(self, batch_size, num_neg):
        total_needed = batch_size * num_neg
        if self._neg_idx + total_needed > self._neg_table_size:
            self._neg_idx = 0
        neg = self._neg_table[self._neg_idx:self._neg_idx + total_needed]
        self._neg_idx += total_needed
        return neg.view(batch_size, num_neg)

    def get_batch(self, batch_size):
        while len(self._pair_buf) < batch_size:
            self._refill_buffer()
        pairs = [self._pair_buf.pop() for _ in range(batch_size)]
        targets = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        contexts = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        negatives = self.sample_negatives(batch_size, NEG_SAMPLES)
        return targets, contexts, negatives


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Path analogies: structural relationships we expect to find
PATH_ANALOGY_TESTS = [
    # usr/X structure mirrors
    ("usr", "bin", "usr", "lib"),       # usr:bin :: usr:lib — not a real analogy
    # Functional analogies
    ("bin", "usr", "sbin", "usr"),      # bin is to usr as sbin is to usr
]

# These are the interesting ones — component relationships
COMPONENT_SIMILARITY_TESTS = [
    # Should be similar (same functional role)
    ("bin", "sbin", "similar"),
    ("lib", "lib64", "similar"),
    ("doc", "man", "similar"),
    ("include", "src", "similar"),
    ("log", "cache", "similar"),
    ("conf", "config", "similar"),
    ("init.d", "rc.d", "similar"),
    ("share", "common", "similar"),
    # Should be different
    ("bin", "doc", "different"),
    ("lib", "log", "different"),
    ("proc", "usr", "different"),
    ("dev", "share", "different"),
    ("sys", "man", "different"),
    ("etc", "boot", "different"),
]


def evaluate_embeddings(model, vocab, step):
    """Evaluate path component embeddings."""
    emb = model.get_embeddings().to("cpu")
    emb_norm = F.normalize(emb, p=2, dim=-1)

    def get_vec(token):
        if token in vocab:
            return emb_norm[vocab.word2id[token]]
        return None

    def nearest(vec, exclude=None, k=10):
        sims = emb_norm @ vec
        if exclude:
            for w in exclude:
                if w in vocab:
                    sims[vocab.word2id[w]] = -1
        topk = sims.topk(k)
        results = []
        for idx, sim in zip(topk.indices, topk.values):
            token = vocab.id2word.get(idx.item(), "?")
            results.append((token, sim.item()))
        return results

    # Nearest neighbors for key filesystem components
    log(f"  --- Nearest neighbors ---")
    key_tokens = ["bin", "lib", "usr", "etc", "var", "share", "doc",
                  "proc", "sys", "dev", "log", "cache", "config",
                  "python3", "x86_64-linux-gnu", "systemd", "apache2",
                  "include", "man", "fonts", "locale"]
    for token in key_tokens:
        v = get_vec(token)
        if v is not None:
            top = nearest(v, exclude=[token], k=8)
            nns = ", ".join(f"{w}({s:.2f})" for w, s in top)
            log(f"    {token}: {nns}")

    # Similarities
    log(f"  --- Component similarities ---")
    sim_scores = {"similar": [], "different": []}
    for w1, w2, label in COMPONENT_SIMILARITY_TESTS:
        v1, v2 = get_vec(w1), get_vec(w2)
        if v1 is None or v2 is None:
            continue
        sim = (v1 * v2).sum().item()
        sim_scores[label].append(sim)
        log(f"    {w1} ↔ {w2}: {sim:.3f} ({label})")
    avg_sim = np.mean(sim_scores["similar"]) if sim_scores["similar"] else 0
    avg_diff = np.mean(sim_scores["different"]) if sim_scores["different"] else 0
    sim_gap = avg_sim - avg_diff
    log(f"  Similar avg: {avg_sim:.3f} | Different avg: {avg_diff:.3f} | Gap: {sim_gap:+.3f}")

    # Vector arithmetic: path reconstruction
    log(f"  --- Path arithmetic ---")
    arithmetic_tests = [
        # If bin and sbin are related, and lib is like bin...
        ("bin", "lib", "sbin", "?"),  # bin:lib :: sbin:? (expect lib64 or similar)
        ("usr", "bin", "var", "?"),   # usr:bin :: var:? (expect log or spool)
        ("share", "doc", "lib", "?"), # share:doc :: lib:? (expect include or src)
    ]
    for a, b, c, _ in arithmetic_tests:
        va, vb, vc = get_vec(a), get_vec(b), get_vec(c)
        if any(v is None for v in [va, vb, vc]):
            continue
        query = F.normalize(vb - va + vc, p=2, dim=-1)
        top = nearest(query, exclude=[a, b, c], k=5)
        results = ", ".join(f"{w}({s:.2f})" for w, s in top)
        log(f"    {a}:{b} :: {c}:? → {results}")

    return {
        "sim_gap": sim_gap,
        "avg_similar": avg_sim,
        "avg_different": avg_diff,
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "embed_dim": EMBED_DIM,
    }
    path = os.path.join(CHECKPOINT_DIR, f"step_{step:06d}.pt")
    torch.save(ckpt, path)
    latest = os.path.join(CHECKPOINT_DIR, "latest.pt")
    torch.save(ckpt, latest)
    log(f"  Checkpoint saved: {path}")


def load_checkpoint(model, optimizer):
    latest = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if not os.path.exists(latest):
        return 0
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step = ckpt["step"]
    log(f"Resumed V32: step {step}")
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
# Training
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    log("=" * 70)
    log("FLM V32 — Path2vec Skip-Gram (Filesystem Paths)")
    log("=" * 70)

    # Load vocabulary
    vocab = Vocabulary()
    if not os.path.exists(VOCAB_PATH):
        log("ERROR: No vocabulary found. Run --collect first.")
        return
    vocab.load(VOCAB_PATH)

    vocab_size = len(vocab)
    log(f"  Embed dim: {EMBED_DIM}")
    log(f"  Vocab size: {vocab_size:,}")
    log(f"  Window: {WINDOW_SIZE}, Neg samples: {NEG_SAMPLES}")
    log(f"  Batch size: {BATCH_SIZE}")
    log(f"  Total steps: {TOTAL_STEPS:,}")

    # Model
    model = SkipGram(vocab_size, EMBED_DIM).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    log(f"  Parameters: {params:,} ({params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=PEAK_LR)

    # Load checkpoint
    start_step = 0
    if not args.fresh:
        start_step = load_checkpoint(model, optimizer)

    log(f"  LR: {PEAK_LR} -> {MIN_LR} (cosine) | Steps: {start_step} -> {TOTAL_STEPS}")

    # Data
    dataset = PathPairDataset(PATHS_FILE, vocab)
    log("-" * 70)

    # Training
    model.train()
    running_loss = 0.0
    loss_count = 0
    start_time = time.time()

    # Signal handler
    stop_flag = [False]
    def handle_signal(sig, frame):
        log(f"Signal {sig} received, saving and exiting...")
        stop_flag[0] = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    for step in range(start_step, TOTAL_STEPS):
        if stop_flag[0]:
            save_checkpoint(model, optimizer, step)
            break

        current_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        targets, contexts, negatives = dataset.get_batch(BATCH_SIZE)
        targets = targets.to(DEVICE)
        contexts = contexts.to(DEVICE)
        negatives = negatives.to(DEVICE)

        loss = model(targets, contexts, negatives)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loss_count += 1

        if (step + 1) % LOG_EVERY == 0:
            avg_loss = running_loss / loss_count
            elapsed = time.time() - start_time
            sps = (step + 1 - start_step) / max(elapsed, 1)
            pct = (step + 1) / TOTAL_STEPS * 100

            log(f"step {step+1:>7d} [V32] | loss={avg_loss:.4f} | "
                f"lr {current_lr:.2e} | {pct:.1f}% | {sps:.1f} step/s")

            log_metrics(step + 1, {
                "loss": avg_loss,
                "lr": current_lr,
            })

            running_loss = 0.0
            loss_count = 0

        if (step + 1) % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                eval_metrics = evaluate_embeddings(model, vocab, step + 1)
                log_metrics(step + 1, eval_metrics)
            model.train()

        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, step + 1)

    if not stop_flag[0]:
        save_checkpoint(model, optimizer, TOTAL_STEPS)

    log("Training complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V32 — Path2vec")
    parser.add_argument("--setup-chroot", action="store_true",
                        help="Phase 1: Build Debian chroot with packages")
    parser.add_argument("--collect", action="store_true",
                        help="Phase 2: Collect paths and build vocabulary")
    parser.add_argument("--fresh", action="store_true",
                        help="Phase 3: Train from scratch")
    parser.add_argument("--resume", action="store_true",
                        help="Phase 3: Resume training")
    parser.add_argument("--all", action="store_true",
                        help="Run all phases")
    parser.add_argument("--unmount", action="store_true",
                        help="Unmount virtual filesystems from chroot")
    args = parser.parse_args()

    if args.all:
        args.setup_chroot = True
        args.collect = True
        args.fresh = True

    if args.setup_chroot:
        setup_chroot()

    if args.collect:
        collect_paths()

    if args.fresh or args.resume:
        train(args)

    if args.unmount:
        unmount_virtual_fs()
