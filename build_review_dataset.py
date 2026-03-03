#!/usr/bin/env python3
"""
build_review_dataset.py — Build a review-ready SFT dataset with metadata
========================================================================

Takes the per-source filtered JSONL files from generate_sft_v2.py and creates
a review-organized dataset where each sample has:
  - Unique ID
  - Source and license info
  - Processing/filtering notes
  - Quality signals (computed from text)
  - Confidence tier (auto_approve / needs_review / auto_reject)

Output structure:
  data/sft/review/
    manifest.json            — Stats, source info, review progress
    auto_approved.jsonl      — High-confidence, ready for training
    auto_rejected.jsonl      — Low-confidence, discarded (for reference)
    needs_review/            — Batches organized by source for agent review
      se_unix/batch_001.jsonl
      se_askubuntu/batch_001.jsonl
      ...
      ubuntu/batch_001.jsonl
    reviewed/                — Agents write output here

Usage:
    python build_review_dataset.py              # Build full review dataset
    python build_review_dataset.py --stats      # Print current review status
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
SFT_DIR = PROJECT_DIR / "data" / "sft"
REVIEW_DIR = SFT_DIR / "review"

# Per-source input files (from generate_sft_v2.py)
SOURCE_FILES = {
    "oasst2": SFT_DIR / "oasst2_filtered.jsonl",
    "dolly": SFT_DIR / "dolly_filtered.jsonl",
    "stackexchange": SFT_DIR / "stackexchange_chat.jsonl",
    "ubuntu": SFT_DIR / "ubuntu_irc.jsonl",
}

# License info per source dataset
LICENSES = {
    "oasst2": "Apache-2.0",
    "dolly": "CC-BY-SA-3.0",
    "stackexchange": "CC-BY-SA-4.0",
    "ubuntu": "MIT",
}

# Processing notes per source
PROCESSING_NOTES = {
    "oasst2": (
        "OASST2 human conversations. Filtered: AI identity leaks removed "
        "(Open Assistant, LAION, 'I am an AI' patterns). "
        "~50% got flm system prompt injected. Multi-turn preserved."
    ),
    "dolly": (
        "Databricks Dolly 15K — written by Databricks employees. "
        "Filtered: responses < 60 chars removed, AI identity leaks removed. "
        "Instruction+context → user message, response → assistant. "
        "~30% got flm system prompt injected."
    ),
    "stackexchange": (
        "Stack Exchange Q/A pairs from 2.9M pretrain posts. "
        "Paired via keyword-matching heuristic (inverted index per site, "
        "title keyword overlap scoring). Filters: min_keyword_overlap=3 "
        "(4 for non-Linux sites), title_coverage >= 0.35, "
        "min_answer_chars=100, max_conv_chars=4000. "
        "WARNING: ~26% mismatch rate found in spot-check. "
        "Review needed to verify Q/A pairs actually match."
    ),
    "ubuntu": (
        "Ubuntu IRC dialogue corpus — real support conversations. "
        "Filtered: min 3 turns, min 35 assistant words, max 40 turns, "
        "must contain technical keywords. Consecutive same-speaker lines merged. "
        "Quality varies — some conversations are coherent troubleshooting, "
        "others are fragmented IRC chat."
    ),
}

# Batch sizes for review
SE_BATCH_SIZE = 50     # Smaller batches — SE needs careful Q/A match review
UBUNTU_BATCH_SIZE = 100  # Larger — Ubuntu quality is more about coherence

# Stopwords for keyword extraction (same as generate_sft_v2.py)
STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "about", "what",
    "which", "who", "whom", "this", "that", "these", "those", "it", "its",
    "my", "your", "his", "her", "our", "their", "me", "him", "us",
    "them", "i", "you", "he", "she", "we", "they", "any", "many", "much",
    "also", "still", "already", "even", "well", "really", "quite", "get",
    "got", "like", "know", "want", "think", "make", "use", "using",
    "one", "two", "way", "work", "try", "new", "first", "last", "good",
    "best", "right", "don", "doesn", "didn", "won", "wouldn", "couldn",
    "shouldn", "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def extract_keywords(text, max_chars=300):
    """Extract significant keywords from text."""
    text = text[:max_chars].lower()
    words = set(re.findall(r"\b[a-z][a-z0-9_./-]{2,}\b", text))
    return words - STOPWORDS


def parse_conversation(text):
    """Parse a conversation text into parts.

    Returns dict with:
      question_title: first line of user message (often the title for SE)
      question_body: rest of user message
      answer_text: all assistant text concatenated
      turn_count: number of turns
      has_system: whether it has a system prompt
    """
    parts = {
        "question_title": "",
        "question_body": "",
        "answer_text": "",
        "user_text": "",
        "turn_count": 0,
        "has_system": "<|system|>" in text,
    }

    # Split into role segments
    segments = re.split(r"<\|(?:system|user|assistant)\|>\n?", text)
    roles = re.findall(r"<\|(system|user|assistant)\|>", text)

    user_parts = []
    asst_parts = []

    for role, content in zip(roles, segments[1:]):  # skip empty first split
        content = content.strip()
        if role == "user":
            user_parts.append(content)
            parts["turn_count"] += 1
        elif role == "assistant":
            asst_parts.append(content)
            parts["turn_count"] += 1

    # First user message — split into title and body for SE
    if user_parts:
        first_user = user_parts[0]
        lines = first_user.split("\n\n", 1)
        parts["question_title"] = lines[0].strip()
        parts["question_body"] = lines[1].strip() if len(lines) > 1 else ""
        parts["user_text"] = " ".join(user_parts)

    parts["answer_text"] = " ".join(asst_parts)

    return parts


def compute_se_quality(text):
    """Compute quality signals for a Stack Exchange Q/A pair.

    Returns dict with quality metrics used for tiering.
    """
    parsed = parse_conversation(text)

    title_keywords = extract_keywords(parsed["question_title"])
    answer_keywords = extract_keywords(parsed["answer_text"], max_chars=500)

    overlap = title_keywords & answer_keywords
    overlap_count = len(overlap)
    title_kw_count = len(title_keywords)
    coverage = overlap_count / title_kw_count if title_kw_count > 0 else 0.0

    return {
        "keyword_overlap": overlap_count,
        "title_keyword_count": title_kw_count,
        "title_coverage": round(coverage, 3),
        "question_chars": len(parsed["user_text"]),
        "answer_chars": len(parsed["answer_text"]),
        "turn_count": parsed["turn_count"],
        "has_system": parsed["has_system"],
    }


def compute_ubuntu_quality(text):
    """Compute quality signals for a Ubuntu dialogue."""
    parsed = parse_conversation(text)

    answer_words = len(parsed["answer_text"].split())
    user_words = len(parsed["user_text"].split())

    # Check for technical substance
    tech_pattern = re.compile(
        r"\b(sudo|apt|dpkg|bash|terminal|command|install|package|"
        r"error|fail|kernel|driver|grub|boot|partition|mount|"
        r"network|wifi|dns|permission|chmod|config|"
        r"process|service|systemd|update|upgrade|ssh|port|"
        r"log|dmesg|python|gcc|compile|ubuntu|debian|linux)\b",
        re.I,
    )
    tech_matches = len(tech_pattern.findall(parsed["user_text"] + " " + parsed["answer_text"]))

    return {
        "turn_count": parsed["turn_count"],
        "answer_words": answer_words,
        "user_words": user_words,
        "tech_keyword_hits": tech_matches,
        "has_system": parsed["has_system"],
        "total_chars": len(text),
    }


def compute_basic_quality(text):
    """Basic quality signals for OASST2/Dolly."""
    parsed = parse_conversation(text)
    return {
        "turn_count": parsed["turn_count"],
        "answer_chars": len(parsed["answer_text"]),
        "question_chars": len(parsed["user_text"]),
        "has_system": parsed["has_system"],
        "total_chars": len(text),
    }


def tier_se_sample(quality):
    """Assign confidence tier for a Stack Exchange sample.

    Calibrated from manual spot-checks:
      - overlap=3 + low coverage: ~80% mismatched (common keywords like "server")
      - overlap>=4 + coverage>=0.55: ~90%+ correctly matched
      - overlap>=5: almost always correct

    Returns: 'auto_approve', 'needs_review', or 'auto_reject'
    """
    overlap = quality["keyword_overlap"]
    coverage = quality["title_coverage"]

    # Auto-reject: weakest matches (overlap=3 with poor coverage)
    if overlap <= 3 and coverage < 0.55:
        return "auto_reject"

    # Auto-approve: strong evidence of correct match
    if overlap >= 5:
        return "auto_approve"
    if overlap >= 4 and coverage >= 0.55:
        return "auto_approve"

    # Needs review: overlap=3 with decent coverage, or overlap=4 with low coverage
    return "needs_review"


def tier_ubuntu_sample(quality):
    """Assign confidence tier for a Ubuntu dialogue.

    Returns: 'auto_approve', 'needs_review', or 'auto_reject'
    """
    turns = quality["turn_count"]
    a_words = quality["answer_words"]
    tech_hits = quality["tech_keyword_hits"]

    # Auto-reject: no technical substance
    if tech_hits == 0:
        return "auto_reject"
    if a_words < 25:
        return "auto_reject"

    # Auto-approve: substantive technical conversations
    if turns >= 4 and a_words >= 50 and tech_hits >= 2:
        return "auto_approve"

    return "needs_review"


def write_batches(samples, batch_dir, batch_size):
    """Write samples into numbered batch files.

    Returns list of batch file paths.
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    batches = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        batch_num = i // batch_size + 1
        batch_path = batch_dir / f"batch_{batch_num:04d}.jsonl"
        with open(batch_path, "w") as f:
            for item in batch:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        batches.append(str(batch_path.relative_to(REVIEW_DIR)))
    return batches


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def build_review_dataset():
    """Build the full review-ready dataset."""
    log("=" * 60)
    log("Building Review Dataset")
    log("=" * 60)

    # Create output directories
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    (REVIEW_DIR / "needs_review").mkdir(exist_ok=True)
    (REVIEW_DIR / "reviewed").mkdir(exist_ok=True)

    auto_approved = []
    auto_rejected = []
    needs_review_by_source = defaultdict(list)  # source_subdir -> samples
    stats = {
        "sources": {},
        "totals": {
            "auto_approved": 0,
            "auto_rejected": 0,
            "needs_review": 0,
            "total": 0,
        },
        "batches": {},
    }

    sample_id = 0

    # ----- Process OASST2 -----
    log("\nProcessing OASST2...")
    oasst_path = SOURCE_FILES["oasst2"]
    if oasst_path.exists():
        count = 0
        with open(oasst_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                sample_id += 1
                quality = compute_basic_quality(entry["text"])

                sample = {
                    "id": f"oasst2_{sample_id:06d}",
                    "text": entry["text"],
                    "source": entry.get("source", "oasst2"),
                    "source_dataset": "oasst2",
                    "license": LICENSES["oasst2"],
                    "processing": PROCESSING_NOTES["oasst2"],
                    "quality": quality,
                    "tier": "auto_approve",
                    "status": "approved",
                }
                auto_approved.append(sample)
                count += 1

        log(f"  OASST2: {count:,} samples → all auto-approved (curated dataset)")
        stats["sources"]["oasst2"] = {
            "total": count,
            "auto_approved": count,
            "needs_review": 0,
            "auto_rejected": 0,
        }
    else:
        log(f"  SKIP: {oasst_path} not found")

    # ----- Process Dolly -----
    log("\nProcessing Dolly...")
    dolly_path = SOURCE_FILES["dolly"]
    if dolly_path.exists():
        count = 0
        with open(dolly_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                sample_id += 1
                quality = compute_basic_quality(entry["text"])

                sample = {
                    "id": f"dolly_{sample_id:06d}",
                    "text": entry["text"],
                    "source": entry.get("source", "dolly"),
                    "source_dataset": "dolly",
                    "license": LICENSES["dolly"],
                    "processing": PROCESSING_NOTES["dolly"],
                    "quality": quality,
                    "tier": "auto_approve",
                    "status": "approved",
                }
                auto_approved.append(sample)
                count += 1

        log(f"  Dolly: {count:,} samples → all auto-approved (curated dataset)")
        stats["sources"]["dolly"] = {
            "total": count,
            "auto_approved": count,
            "needs_review": 0,
            "auto_rejected": 0,
        }
    else:
        log(f"  SKIP: {dolly_path} not found")

    # ----- Process Stack Exchange -----
    log("\nProcessing Stack Exchange...")
    se_path = SOURCE_FILES["stackexchange"]
    if se_path.exists():
        se_total = 0
        se_approved = 0
        se_rejected = 0
        se_review = 0
        t0 = time.time()

        with open(se_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                se_total += 1
                sample_id += 1

                quality = compute_se_quality(entry["text"])
                tier = tier_se_sample(quality)

                source_tag = entry.get("source", "se_unknown")
                # Group into review subdirs by site
                # se_unix, se_askubuntu, se_superuser, se_serverfault → own dirs
                # everything else → se_other
                if source_tag in ("se_unix", "se_askubuntu", "se_superuser",
                                  "se_serverfault"):
                    review_subdir = source_tag
                else:
                    review_subdir = "se_other"

                sample = {
                    "id": f"se_{sample_id:06d}",
                    "text": entry["text"],
                    "source": source_tag,
                    "source_dataset": "stackexchange",
                    "license": LICENSES["stackexchange"],
                    "processing": PROCESSING_NOTES["stackexchange"],
                    "quality": quality,
                    "tier": tier,
                    "status": "approved" if tier == "auto_approve" else
                              "rejected" if tier == "auto_reject" else "pending",
                }

                if tier == "auto_approve":
                    auto_approved.append(sample)
                    se_approved += 1
                elif tier == "auto_reject":
                    auto_rejected.append(sample)
                    se_rejected += 1
                else:
                    needs_review_by_source[review_subdir].append(sample)
                    se_review += 1

                if se_total % 50_000 == 0:
                    elapsed = time.time() - t0
                    log(f"  SE: {se_total:,} processed ({elapsed:.0f}s) — "
                        f"approve: {se_approved:,}, reject: {se_rejected:,}, "
                        f"review: {se_review:,}")

        log(f"  SE total: {se_total:,}")
        log(f"    Auto-approved: {se_approved:,} ({se_approved/se_total*100:.1f}%)")
        log(f"    Auto-rejected: {se_rejected:,} ({se_rejected/se_total*100:.1f}%)")
        log(f"    Needs review:  {se_review:,} ({se_review/se_total*100:.1f}%)")
        stats["sources"]["stackexchange"] = {
            "total": se_total,
            "auto_approved": se_approved,
            "needs_review": se_review,
            "auto_rejected": se_rejected,
        }
    else:
        log(f"  SKIP: {se_path} not found")

    # ----- Process Ubuntu -----
    log("\nProcessing Ubuntu...")
    ubuntu_path = SOURCE_FILES["ubuntu"]
    if ubuntu_path.exists():
        ub_total = 0
        ub_approved = 0
        ub_rejected = 0
        ub_review = 0

        with open(ubuntu_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                ub_total += 1
                sample_id += 1

                quality = compute_ubuntu_quality(entry["text"])
                tier = tier_ubuntu_sample(quality)

                sample = {
                    "id": f"ubuntu_{sample_id:06d}",
                    "text": entry["text"],
                    "source": entry.get("source", "ubuntu_dialogue"),
                    "source_dataset": "ubuntu",
                    "license": LICENSES["ubuntu"],
                    "processing": PROCESSING_NOTES["ubuntu"],
                    "quality": quality,
                    "tier": tier,
                    "status": "approved" if tier == "auto_approve" else
                              "rejected" if tier == "auto_reject" else "pending",
                }

                if tier == "auto_approve":
                    auto_approved.append(sample)
                    ub_approved += 1
                elif tier == "auto_reject":
                    auto_rejected.append(sample)
                    ub_rejected += 1
                else:
                    needs_review_by_source["ubuntu"].append(sample)
                    ub_review += 1

        log(f"  Ubuntu total: {ub_total:,}")
        log(f"    Auto-approved: {ub_approved:,} ({ub_approved/ub_total*100:.1f}%)")
        log(f"    Auto-rejected: {ub_rejected:,} ({ub_rejected/ub_total*100:.1f}%)")
        log(f"    Needs review:  {ub_review:,} ({ub_review/ub_total*100:.1f}%)")
        stats["sources"]["ubuntu"] = {
            "total": ub_total,
            "auto_approved": ub_approved,
            "needs_review": ub_review,
            "auto_rejected": ub_rejected,
        }
    else:
        log(f"  SKIP: {ubuntu_path} not found")

    # ----- Write outputs -----
    log("\n" + "=" * 60)
    log("Writing review dataset")
    log("=" * 60)

    # Auto-approved
    approved_path = REVIEW_DIR / "auto_approved.jsonl"
    with open(approved_path, "w") as f:
        for item in auto_approved:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log(f"  Auto-approved: {len(auto_approved):,} → {approved_path.name}")

    # Auto-rejected
    rejected_path = REVIEW_DIR / "auto_rejected.jsonl"
    with open(rejected_path, "w") as f:
        for item in auto_rejected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    log(f"  Auto-rejected: {len(auto_rejected):,} → {rejected_path.name}")

    # Needs-review batches
    all_batch_paths = {}
    total_review = 0
    for subdir, samples in sorted(needs_review_by_source.items()):
        batch_size = SE_BATCH_SIZE if subdir.startswith("se_") else UBUNTU_BATCH_SIZE
        batch_dir = REVIEW_DIR / "needs_review" / subdir
        batch_paths = write_batches(samples, batch_dir, batch_size)
        all_batch_paths[subdir] = batch_paths
        total_review += len(samples)
        log(f"  {subdir}: {len(samples):,} samples → {len(batch_paths)} batches")

    stats["totals"] = {
        "auto_approved": len(auto_approved),
        "auto_rejected": len(auto_rejected),
        "needs_review": total_review,
        "total": len(auto_approved) + len(auto_rejected) + total_review,
    }
    stats["batches"] = {
        subdir: {"count": len(paths), "paths": paths}
        for subdir, paths in all_batch_paths.items()
    }
    stats["batch_sizes"] = {
        "se": SE_BATCH_SIZE,
        "ubuntu": UBUNTU_BATCH_SIZE,
    }

    # Write manifest
    manifest_path = REVIEW_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(stats, f, indent=2)
    log(f"\n  Manifest: {manifest_path.name}")

    # ----- Summary -----
    log("\n" + "=" * 60)
    log("Review Dataset Summary")
    log("=" * 60)
    log(f"  Total samples:   {stats['totals']['total']:,}")
    log(f"  Auto-approved:   {stats['totals']['auto_approved']:,} "
        f"({stats['totals']['auto_approved']/stats['totals']['total']*100:.1f}%)")
    log(f"  Auto-rejected:   {stats['totals']['auto_rejected']:,} "
        f"({stats['totals']['auto_rejected']/stats['totals']['total']*100:.1f}%)")
    log(f"  Needs review:    {stats['totals']['needs_review']:,} "
        f"({stats['totals']['needs_review']/stats['totals']['total']*100:.1f}%)")
    total_batches = sum(len(paths) for paths in all_batch_paths.values())
    log(f"  Review batches:  {total_batches}")
    log(f"\n  Output: {REVIEW_DIR}")

    return stats


def print_review_status():
    """Print current review progress."""
    manifest_path = REVIEW_DIR / "manifest.json"
    if not manifest_path.exists():
        print("No review dataset found. Run: python build_review_dataset.py")
        return

    with open(manifest_path) as f:
        stats = json.load(f)

    print("=" * 60)
    print("Review Dataset Status")
    print("=" * 60)
    print(f"  Total samples:   {stats['totals']['total']:,}")
    print(f"  Auto-approved:   {stats['totals']['auto_approved']:,}")
    print(f"  Auto-rejected:   {stats['totals']['auto_rejected']:,}")
    print(f"  Needs review:    {stats['totals']['needs_review']:,}")

    # Check reviewed batches
    reviewed_dir = REVIEW_DIR / "reviewed"
    if reviewed_dir.exists():
        reviewed_files = list(reviewed_dir.rglob("*.jsonl"))
        reviewed_count = 0
        approved_in_review = 0
        rejected_in_review = 0
        tweaked_in_review = 0
        for rf in reviewed_files:
            with open(rf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    reviewed_count += 1
                    status = entry.get("status", "")
                    if status == "approved":
                        approved_in_review += 1
                    elif status == "rejected":
                        rejected_in_review += 1
                    elif status == "tweaked":
                        tweaked_in_review += 1

        total_batches = sum(
            b["count"] for b in stats.get("batches", {}).values()
        )
        print(f"\n  Review progress:")
        print(f"    Reviewed files: {len(reviewed_files)} / {total_batches} batches")
        print(f"    Reviewed samples: {reviewed_count:,}")
        print(f"      Approved: {approved_in_review:,}")
        print(f"      Rejected: {rejected_in_review:,}")
        print(f"      Tweaked:  {tweaked_in_review:,}")
    else:
        print(f"\n  No reviews yet. Review batches are in: {REVIEW_DIR / 'needs_review'}")

    print(f"\n  Per-source breakdown:")
    for source, info in stats.get("sources", {}).items():
        print(f"    {source}: {info['total']:,} total — "
              f"{info['auto_approved']:,} approved, "
              f"{info['needs_review']:,} review, "
              f"{info['auto_rejected']:,} rejected")


def main():
    parser = argparse.ArgumentParser(
        description="Build review-ready SFT dataset with metadata"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print current review status only",
    )
    args = parser.parse_args()

    if args.stats:
        print_review_status()
    else:
        t0 = time.time()
        build_review_dataset()
        elapsed = time.time() - t0
        log(f"\nTotal time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
