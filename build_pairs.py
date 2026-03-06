#!/usr/bin/env python3
"""
Build paraphrase pair datasets for V4 encoder training.
========================================================

Downloads DFSG-compliant paraphrase and translation pair datasets,
processes them into a uniform JSONL format for train_encoder.py.

Sources:
  - PAWS (Apache 2.0) — hard paraphrase pairs from word scrambling
  - QQP (Quora Question Pairs) — question paraphrases
  - MRPC (Microsoft Research Paraphrase Corpus) — news paraphrases
  - EuroParl (Public Domain) — cross-lingual translation pairs
  - WikiMatrix (CC-BY-SA) — mined translation pairs

Output: data/pairs/*.jsonl — one file per source + eval splits
Format: {"text_a": "...", "text_b": "...", "label": 1/0, "type": "...", "source": "..."}

Usage:
    python build_pairs.py                    # build all
    python build_pairs.py --source paws      # build single source
    python build_pairs.py --list             # list sources and status
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

DATA_DIR = Path("data/pairs")
EVAL_FRACTION = 0.05  # 5% held out for eval


def log(msg):
    print(f"[build_pairs] {msg}", flush=True)


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def write_pair(f, text_a, text_b, label, pair_type, source):
    """Write a single pair to JSONL."""
    text_a = text_a.strip()
    text_b = text_b.strip()
    if len(text_a) < 5 or len(text_b) < 5:
        return False
    doc = {
        "text_a": text_a,
        "text_b": text_b,
        "label": label,
        "type": pair_type,
        "source": source,
    }
    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return True


def count_lines(path):
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for _ in f:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Source: PAWS (Paraphrase Adversaries from Word Scrambling)
# ---------------------------------------------------------------------------

def build_paws():
    """Download PAWS — hard paraphrase pairs. Apache 2.0."""
    out_path = DATA_DIR / "paws.jsonl"
    eval_path = DATA_DIR / "eval_paws.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"PAWS already exists ({count_lines(out_path):,} pairs), skipping.")
        return

    log("Building PAWS...")
    from datasets import load_dataset

    ds = load_dataset("paws", "labeled_final")

    count = 0
    eval_count = 0
    with open(out_path, "w") as f, open(eval_path, "w") as ef:
        for split in ["train", "validation", "test"]:
            if split not in ds:
                continue
            for sample in ds[split]:
                text_a = sample["sentence1"]
                text_b = sample["sentence2"]
                label = sample["label"]
                pair_type = "hard_negative" if label == 0 else "paraphrase"

                if random.random() < EVAL_FRACTION:
                    if write_pair(ef, text_a, text_b, label, pair_type, "paws"):
                        eval_count += 1
                else:
                    if write_pair(f, text_a, text_b, label, pair_type, "paws"):
                        count += 1

    log(f"  PAWS: {count:,} train + {eval_count:,} eval")


# ---------------------------------------------------------------------------
# Source: QQP (Quora Question Pairs)
# ---------------------------------------------------------------------------

def build_qqp():
    """Download QQP — question paraphrases."""
    out_path = DATA_DIR / "qqp.jsonl"
    eval_path = DATA_DIR / "eval_qqp.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"QQP already exists ({count_lines(out_path):,} pairs), skipping.")
        return

    log("Building QQP...")
    from datasets import load_dataset

    ds = load_dataset("glue", "qqp")

    count = 0
    eval_count = 0
    with open(out_path, "w") as f, open(eval_path, "w") as ef:
        for split in ["train", "validation"]:
            if split not in ds:
                continue
            for sample in ds[split]:
                text_a = sample["question1"]
                text_b = sample["question2"]
                label = sample["label"]
                pair_type = "paraphrase" if label == 1 else "non_paraphrase"

                if random.random() < EVAL_FRACTION:
                    if write_pair(ef, text_a, text_b, label, pair_type, "qqp"):
                        eval_count += 1
                else:
                    if write_pair(f, text_a, text_b, label, pair_type, "qqp"):
                        count += 1

    log(f"  QQP: {count:,} train + {eval_count:,} eval")


# ---------------------------------------------------------------------------
# Source: MRPC (Microsoft Research Paraphrase Corpus)
# ---------------------------------------------------------------------------

def build_mrpc():
    """Download MRPC — news paraphrases."""
    out_path = DATA_DIR / "mrpc.jsonl"
    eval_path = DATA_DIR / "eval_mrpc.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"MRPC already exists ({count_lines(out_path):,} pairs), skipping.")
        return

    log("Building MRPC...")
    from datasets import load_dataset

    ds = load_dataset("glue", "mrpc")

    count = 0
    eval_count = 0
    with open(out_path, "w") as f, open(eval_path, "w") as ef:
        for split in ["train", "validation", "test"]:
            if split not in ds:
                continue
            for sample in ds[split]:
                text_a = sample["sentence1"]
                text_b = sample["sentence2"]
                label = sample["label"]
                pair_type = "paraphrase" if label == 1 else "non_paraphrase"

                if random.random() < EVAL_FRACTION:
                    if write_pair(ef, text_a, text_b, label, pair_type, "mrpc"):
                        eval_count += 1
                else:
                    if write_pair(f, text_a, text_b, label, pair_type, "mrpc"):
                        count += 1

    log(f"  MRPC: {count:,} train + {eval_count:,} eval")


# ---------------------------------------------------------------------------
# Source: EuroParl (cross-lingual pairs)
# ---------------------------------------------------------------------------

def build_europarl():
    """Download EuroParl EN-FR — cross-lingual pairs. Public Domain."""
    out_path = DATA_DIR / "europarl.jsonl"
    eval_path = DATA_DIR / "eval_europarl.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"EuroParl already exists ({count_lines(out_path):,} pairs), skipping.")
        return

    log("Building EuroParl (EN-FR)...")
    from datasets import load_dataset

    # Try multiple EuroParl dataset names
    ds = None
    for name in ["europarl_bilingual", "Helsinki-NLP/europarl"]:
        try:
            ds = load_dataset(name, "en-fr", split="train", streaming=True)
            log(f"  Loaded from {name}")
            break
        except Exception:
            continue

    if ds is None:
        log("  WARNING: Could not load EuroParl, skipping.")
        return

    count = 0
    eval_count = 0
    max_pairs = 200_000  # Cap to keep dataset balanced

    with open(out_path, "w") as f, open(eval_path, "w") as ef:
        for sample in ds:
            translation = sample.get("translation", {})
            text_en = translation.get("en", "")
            text_fr = translation.get("fr", "")

            if len(text_en) < 10 or len(text_fr) < 10:
                continue
            if len(text_en) > 500:  # Skip very long parliamentary speeches
                continue

            if random.random() < EVAL_FRACTION:
                if write_pair(ef, text_en, text_fr, 1, "crosslingual", "europarl"):
                    eval_count += 1
            else:
                if write_pair(f, text_en, text_fr, 1, "crosslingual", "europarl"):
                    count += 1

            if count + eval_count >= max_pairs:
                break

    log(f"  EuroParl: {count:,} train + {eval_count:,} eval")


# ---------------------------------------------------------------------------
# Source: WikiMatrix (cross-lingual mined pairs)
# ---------------------------------------------------------------------------

def build_wikimatrix():
    """Download WikiMatrix EN-FR — mined cross-lingual pairs. CC-BY-SA."""
    out_path = DATA_DIR / "wikimatrix.jsonl"
    eval_path = DATA_DIR / "eval_wikimatrix.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"WikiMatrix already exists ({count_lines(out_path):,} pairs), skipping.")
        return

    log("Building WikiMatrix (EN-FR)...")
    from datasets import load_dataset

    try:
        ds = load_dataset("wiki_matrix", "en-fr", split="train", streaming=True)
    except Exception as e:
        log(f"  WARNING: Could not load WikiMatrix: {e}")
        return

    count = 0
    eval_count = 0
    max_pairs = 200_000

    with open(out_path, "w") as f, open(eval_path, "w") as ef:
        for sample in ds:
            # WikiMatrix has score, src, tgt
            score = sample.get("score", 0)
            if score < 1.05:  # Quality threshold
                continue

            text_a = sample.get("src", sample.get("sentence1", ""))
            text_b = sample.get("tgt", sample.get("sentence2", ""))

            if len(text_a) < 10 or len(text_b) < 10:
                continue

            if random.random() < EVAL_FRACTION:
                if write_pair(ef, text_a, text_b, 1, "crosslingual", "wikimatrix"):
                    eval_count += 1
            else:
                if write_pair(f, text_a, text_b, 1, "crosslingual", "wikimatrix"):
                    count += 1

            if count + eval_count >= max_pairs:
                break

    log(f"  WikiMatrix: {count:,} train + {eval_count:,} eval")


# ---------------------------------------------------------------------------
# Status and main
# ---------------------------------------------------------------------------

SOURCES = {
    "paws":       ("PAWS (hard paraphrases)", "Apache-2.0", build_paws),
    "qqp":        ("QQP (question pairs)",    "CC",         build_qqp),
    "mrpc":       ("MRPC (news paraphrases)", "Permissive", build_mrpc),
    "europarl":   ("EuroParl EN-FR",          "Public Domain", build_europarl),
    "wikimatrix": ("WikiMatrix EN-FR",        "CC-BY-SA",   build_wikimatrix),
}


def show_status():
    log("Pair dataset status:")
    log(f"  {'Source':<30s} {'License':<15s} {'Train':>10s} {'Eval':>10s}")
    log("  " + "-" * 70)

    total_train = 0
    total_eval = 0
    for key, (name, license_tag, _) in SOURCES.items():
        train_path = DATA_DIR / f"{key}.jsonl"
        eval_path = DATA_DIR / f"eval_{key}.jsonl"
        train_n = count_lines(train_path) if train_path.exists() else 0
        eval_n = count_lines(eval_path) if eval_path.exists() else 0
        total_train += train_n
        total_eval += eval_n
        status = "READY" if train_n > 0 else "MISSING"
        log(f"  {name:<30s} {license_tag:<15s} {train_n:>10,d} {eval_n:>10,d}  {status}")

    log("  " + "-" * 70)
    log(f"  {'TOTAL':<30s} {'':<15s} {total_train:>10,d} {total_eval:>10,d}")


def main():
    parser = argparse.ArgumentParser(description="Build paraphrase pair datasets")
    parser.add_argument("--source", type=str, help="Build single source")
    parser.add_argument("--list", action="store_true", help="List status")
    args = parser.parse_args()

    ensure_dirs()

    if args.list:
        show_status()
        return

    if args.source:
        if args.source not in SOURCES:
            log(f"Unknown source: {args.source}")
            log(f"Available: {', '.join(SOURCES.keys())}")
            sys.exit(1)
        name, _, build_fn = SOURCES[args.source]
        log(f"Building {name}...")
        build_fn()
        show_status()
        return

    log("Building all pair datasets...")
    for key, (name, _, build_fn) in SOURCES.items():
        log(f"\n{'=' * 60}")
        log(f"Source: {name}")
        try:
            build_fn()
        except Exception as e:
            log(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    log(f"\n{'=' * 60}")
    show_status()


if __name__ == "__main__":
    main()
