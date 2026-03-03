#!/usr/bin/env python3
"""
compile_reviews.py — Compile reviewed SFT data into final training JSONL
========================================================================

Reads:
  - data/sft/review/auto_approved.jsonl (high-confidence samples)
  - data/sft/review/needs_review/<subdir>/batch_*.jsonl (needs-review samples)
  - data/sft/review/flagged_ids.json (IDs flagged for rejection by review agents)

Outputs:
  - data/sft/flm_sft_v2_reviewed.jsonl (final training data)

Pipeline:
  1. Auto-approved samples are included directly
  2. Needs-review samples are included UNLESS their ID is in flagged_ids.json
  3. Deduplication by MD5 hash of text
  4. Shuffle with fixed seed

Usage:
    python scripts/compile_reviews.py              # Compile all reviewed data
    python scripts/compile_reviews.py --stats      # Show review progress
    python scripts/compile_reviews.py --dry-run    # Show what would be compiled
"""

import json
import os
import re
import random
import hashlib
import argparse
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
REVIEW_DIR = PROJECT_DIR / "data" / "sft" / "review"
OUTPUT_FILE = PROJECT_DIR / "data" / "sft" / "flm_sft_v2_reviewed.jsonl"


def load_flagged_ids():
    """Load set of IDs flagged for rejection by review agents."""
    flagged_path = REVIEW_DIR / "flagged_ids.json"
    if flagged_path.exists():
        with open(flagged_path) as f:
            return set(json.load(f))
    return set()


def load_manifest():
    """Load manifest to know valid batch counts per subdir."""
    manifest_path = REVIEW_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return None


def extract_sample(entry):
    """Extract training-format sample from a review entry."""
    text = entry.get("text", "")
    source = entry.get("source", "unknown")
    return {"text": text, "source": source}


def compile_all(dry_run=False):
    """Compile all approved samples into final training data."""
    all_samples = []
    source_counts = Counter()
    flagged_ids = load_flagged_ids()
    manifest = load_manifest()

    print(f"Flagged IDs loaded: {len(flagged_ids):,}")

    # 1. Load auto-approved samples
    auto_path = REVIEW_DIR / "auto_approved.jsonl"
    auto_count = 0
    if auto_path.exists():
        with open(auto_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                auto_count += 1
                sample = extract_sample(entry)
                all_samples.append(sample)
                source_counts[sample["source"]] += 1
        print(f"Auto-approved: {auto_count:,} samples")
    else:
        print(f"WARNING: {auto_path} not found")

    # 2. Load needs-review samples (exclude flagged IDs)
    needs_review_dir = REVIEW_DIR / "needs_review"
    nr_total = 0
    nr_kept = 0
    nr_flagged = 0

    if needs_review_dir.exists() and manifest:
        # Only read valid batches per manifest
        valid_counts = {
            subdir: info["count"]
            for subdir, info in manifest.get("batches", {}).items()
        }

        for subdir in sorted(needs_review_dir.iterdir()):
            if not subdir.is_dir():
                continue
            max_batch = valid_counts.get(subdir.name, 0)

            for batch_file in sorted(subdir.glob("batch_*.jsonl")):
                # Only process valid batches
                m = re.search(r"batch_(\d+)\.jsonl", batch_file.name)
                if m and int(m.group(1)) > max_batch:
                    continue  # stale file, skip

                with open(batch_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        nr_total += 1
                        sample_id = entry.get("id", "")

                        if sample_id in flagged_ids:
                            nr_flagged += 1
                            continue

                        sample = extract_sample(entry)
                        all_samples.append(sample)
                        source_counts[sample["source"]] += 1
                        nr_kept += 1

        print(f"Needs-review: {nr_total:,} total, {nr_kept:,} kept, {nr_flagged:,} flagged")
    else:
        print("No needs-review data found")

    # 3. Deduplicate by text hash
    seen = set()
    deduped = []
    for item in all_samples:
        h = hashlib.md5(item["text"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(item)

    dupes = len(all_samples) - len(deduped)
    print(f"\nDeduplication: removed {dupes:,} duplicates")

    # 4. Shuffle
    random.seed(42)
    random.shuffle(deduped)

    # 5. Summary
    print(f"\n{'='*60}")
    print(f"Final dataset: {len(deduped):,} samples")
    print(f"{'='*60}")

    # Recount after dedup
    final_source_counts = Counter()
    for item in deduped:
        final_source_counts[item["source"]] += 1

    print(f"\nSource breakdown:")
    for source, count in final_source_counts.most_common(25):
        pct = count / len(deduped) * 100
        print(f"  {source}: {count:,} ({pct:.1f}%)")

    # 6. Write output
    if not dry_run:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            for item in deduped:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\nWrote: {OUTPUT_FILE}")
        print(f"Size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"\n[DRY RUN] Would write {len(deduped):,} samples to {OUTPUT_FILE}")

    return deduped


def show_stats():
    """Show current review progress without compiling."""
    manifest = load_manifest()
    if not manifest:
        print("No review dataset found. Run: python build_review_dataset.py")
        return

    flagged_ids = load_flagged_ids()

    print("=" * 60)
    print("Review Dataset Status")
    print("=" * 60)

    totals = manifest["totals"]
    print(f"  Total samples:   {totals['total']:,}")
    print(f"  Auto-approved:   {totals['auto_approved']:,}")
    print(f"  Auto-rejected:   {totals['auto_rejected']:,}")
    print(f"  Needs review:    {totals['needs_review']:,}")
    print(f"  Flagged by agents: {len(flagged_ids):,}")

    # Count decision files
    decisions_dir = REVIEW_DIR / "decisions"
    if decisions_dir.exists():
        decision_files = list(decisions_dir.glob("*.txt"))
        non_empty = sum(1 for f in decision_files if f.stat().st_size > 0)
        print(f"\n  Decision files: {len(decision_files)} ({non_empty} non-empty)")

    # Estimate final size
    est = totals["auto_approved"] + totals["needs_review"] - len(flagged_ids)
    print(f"\n  Estimated final size (before dedup): {est:,}")

    # Per-source breakdown
    print(f"\n  Per-source:")
    for source, info in manifest["sources"].items():
        print(f"    {source}: {info['total']:,} total "
              f"({info['auto_approved']:,} approved, "
              f"{info['needs_review']:,} review, "
              f"{info['auto_rejected']:,} rejected)")


def main():
    parser = argparse.ArgumentParser(
        description="Compile reviewed SFT data into training JSONL"
    )
    parser.add_argument("--stats", action="store_true", help="Show review progress")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        compile_all(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
