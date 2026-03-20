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
    """Download EuroParl EN-FR — cross-lingual pairs. Public Domain.

    Creates both positive (matching translations) and negative (mismatched)
    cross-lingual pairs so the model can't cheat with "EN+FR = same meaning".
    """
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

    # First pass: collect all pairs into memory
    all_en = []
    all_fr = []
    max_collect = 200_000

    for sample in ds:
        translation = sample.get("translation", {})
        text_en = translation.get("en", "").strip()
        text_fr = translation.get("fr", "").strip()

        if len(text_en) < 10 or len(text_fr) < 10:
            continue
        if len(text_en) > 500:
            continue

        all_en.append(text_en)
        all_fr.append(text_fr)

        if len(all_en) >= max_collect:
            break

    log(f"  Collected {len(all_en):,} EN-FR pairs")

    # Create negative pairs by shuffling the French side
    fr_shuffled = list(range(len(all_fr)))
    random.shuffle(fr_shuffled)
    # Ensure no pair maps to itself
    for i in range(len(fr_shuffled)):
        if fr_shuffled[i] == i:
            swap = (i + 1) % len(fr_shuffled)
            fr_shuffled[i], fr_shuffled[swap] = fr_shuffled[swap], fr_shuffled[i]

    pos_count = 0
    neg_count = 0
    eval_pos = 0
    eval_neg = 0

    with open(out_path, "w") as f, open(eval_path, "w") as ef:
        for i in range(len(all_en)):
            is_eval = random.random() < EVAL_FRACTION
            target = ef if is_eval else f

            # Positive: matching translation
            if write_pair(target, all_en[i], all_fr[i], 1, "crosslingual", "europarl"):
                if is_eval:
                    eval_pos += 1
                else:
                    pos_count += 1

            # Negative: mismatched translation (1:1 ratio with positives)
            j = fr_shuffled[i]
            if write_pair(target, all_en[i], all_fr[j], 0, "crosslingual_neg", "europarl"):
                if is_eval:
                    eval_neg += 1
                else:
                    neg_count += 1

    log(f"  EuroParl: {pos_count:,} pos + {neg_count:,} neg train, "
        f"{eval_pos:,} pos + {eval_neg:,} neg eval")


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
# Source: STS-B (Semantic Textual Similarity Benchmark)
# ---------------------------------------------------------------------------

def build_stsb():
    """Download STS-B — graded similarity scores 0-5. CC-BY-SA.

    Unlike binary paraphrase data, STS-B provides continuous similarity
    scores that teach the model graded semantic distance.
    """
    out_path = DATA_DIR / "stsb.jsonl"
    eval_path = DATA_DIR / "eval_stsb.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"STS-B already exists ({count_lines(out_path):,} pairs), skipping.")
        return

    log("Building STS-B...")
    from datasets import load_dataset

    ds = load_dataset("glue", "stsb")

    count = 0
    eval_count = 0
    with open(out_path, "w") as f, open(eval_path, "w") as ef:
        for split in ["train", "validation", "test"]:
            if split not in ds:
                continue
            for sample in ds[split]:
                text_a = sample["sentence1"]
                text_b = sample["sentence2"]
                score = sample["label"]  # float 0-5

                # Normalize to 0-1 for the model
                sim_target = score / 5.0

                doc = {
                    "text_a": text_a.strip(),
                    "text_b": text_b.strip(),
                    "label": 1 if sim_target > 0.6 else 0,
                    "sim_score": round(sim_target, 4),
                    "type": "graded",
                    "source": "stsb",
                }

                if len(doc["text_a"]) < 5 or len(doc["text_b"]) < 5:
                    continue

                target = ef if random.random() < EVAL_FRACTION else f
                target.write(json.dumps(doc, ensure_ascii=False) + "\n")
                if target == ef:
                    eval_count += 1
                else:
                    count += 1

    log(f"  STS-B: {count:,} train + {eval_count:,} eval")


# ---------------------------------------------------------------------------
# Source: NLI (SNLI + MultiNLI)
# ---------------------------------------------------------------------------

def build_nli():
    """Download SNLI + MultiNLI — entailment/contradiction pairs.

    Entailment pairs → positive (same meaning direction)
    Contradiction pairs → hard negative (explicitly opposite)
    Neutral pairs → weak negative (unrelated)
    """
    out_path = DATA_DIR / "nli.jsonl"
    eval_path = DATA_DIR / "eval_nli.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"NLI already exists ({count_lines(out_path):,} pairs), skipping.")
        return

    log("Building NLI (SNLI + MultiNLI)...")
    from datasets import load_dataset

    count = 0
    eval_count = 0
    label_counts = {"entailment": 0, "contradiction": 0, "neutral": 0}

    with open(out_path, "w") as f, open(eval_path, "w") as ef:
        for ds_name in ["snli", "multi_nli"]:
            log(f"  Loading {ds_name}...")
            try:
                ds = load_dataset(ds_name)
            except Exception as e:
                log(f"  WARNING: Could not load {ds_name}: {e}")
                continue

            for split in ["train", "validation", "validation_matched"]:
                if split not in ds:
                    continue
                for sample in ds[split]:
                    nli_label = sample["label"]  # 0=entailment, 1=neutral, 2=contradiction
                    if nli_label == -1:  # skip unlabeled
                        continue

                    text_a = sample["premise"]
                    text_b = sample["hypothesis"]

                    if len(text_a.strip()) < 5 or len(text_b.strip()) < 5:
                        continue

                    if nli_label == 0:
                        pair_type = "entailment"
                        label = 1
                    elif nli_label == 2:
                        pair_type = "contradiction"
                        label = 0
                    else:
                        pair_type = "neutral"
                        label = 0

                    doc = {
                        "text_a": text_a.strip(),
                        "text_b": text_b.strip(),
                        "label": label,
                        "type": pair_type,
                        "source": ds_name,
                    }

                    target = ef if random.random() < EVAL_FRACTION else f
                    target.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    if target == ef:
                        eval_count += 1
                    else:
                        count += 1
                    label_counts[pair_type] = label_counts.get(pair_type, 0) + 1

    log(f"  NLI: {count:,} train + {eval_count:,} eval")
    log(f"  Breakdown: {label_counts}")


# ---------------------------------------------------------------------------
# Status and main
# ---------------------------------------------------------------------------

SOURCES = {
    "paws":       ("PAWS (hard paraphrases)", "Apache-2.0", build_paws),
    "qqp":        ("QQP (question pairs)",    "CC",         build_qqp),
    "mrpc":       ("MRPC (news paraphrases)", "Permissive", build_mrpc),
    "europarl":   ("EuroParl EN-FR",          "Public Domain", build_europarl),
    "wikimatrix": ("WikiMatrix EN-FR",        "CC-BY-SA",   build_wikimatrix),
    "stsb":       ("STS-B (graded sim)",      "CC-BY-SA",   build_stsb),
    "nli":        ("SNLI+MultiNLI (entail)",  "CC-BY-SA",   build_nli),
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
