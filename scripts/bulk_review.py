#!/usr/bin/env python3
"""
bulk_review.py — Efficient bulk review for SFT data
=====================================================

Creates compact review documents and applies agent decisions.

Workflow:
  1. Create compact review files:
     python scripts/bulk_review.py create [--chunk-size 500]

  2. Agent reviews each compact file and writes a decisions file

  3. Apply decisions back to batch files:
     python scripts/bulk_review.py apply

Compact format (one line per sample):
  ID | Q: <question title, max 100ch> | A: <answer start, max 120ch>

Decision format (one line per sample):
  ID:APPROVE  or  ID:REJECT  or  ID:TWEAK
"""

import json
import re
import os
import argparse
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_DIR = Path(__file__).resolve().parent.parent
REVIEW_DIR = PROJECT_DIR / "data" / "sft" / "review"
COMPACT_DIR = REVIEW_DIR / "compact"
DECISIONS_DIR = REVIEW_DIR / "decisions"


def create_compact_files(chunk_size=500):
    """Create compact review documents from needs_review batches."""
    needs_dir = REVIEW_DIR / "needs_review"
    if not needs_dir.exists():
        print("No needs_review directory found")
        return

    COMPACT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all samples from all batches, grouped by source subdir
    all_samples = []
    for subdir in sorted(needs_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Skip already-reviewed batches
        reviewed_subdir = REVIEW_DIR / "reviewed" / subdir.name
        reviewed_batches = set()
        if reviewed_subdir.exists():
            reviewed_batches = {f.name for f in reviewed_subdir.glob("*.jsonl")}

        for batch_file in sorted(subdir.glob("*.jsonl")):
            if batch_file.name in reviewed_batches:
                continue
            with open(batch_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    # Extract compact info
                    text = entry["text"]
                    parts = text.split("<|assistant|>\n", 1)
                    q_part = parts[0].replace("<|user|>\n", "").replace("<|system|>\n", "").strip()
                    a_part = parts[1].strip() if len(parts) > 1 else ""

                    # Question: first line (title), truncated
                    q_title = q_part.split("\n")[0][:100].strip()
                    # Answer: first 250 chars for better context
                    a_start = a_part[:250].replace("\n", " ").strip()

                    all_samples.append({
                        "id": entry["id"],
                        "source": entry.get("source", "?"),
                        "source_subdir": subdir.name,
                        "batch_file": str(batch_file.relative_to(REVIEW_DIR)),
                        "q_title": q_title,
                        "a_start": a_start,
                    })

    print(f"Total unreviewed samples: {len(all_samples):,}")

    # Write compact files in chunks
    chunk_files = []
    for i in range(0, len(all_samples), chunk_size):
        chunk = all_samples[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        chunk_path = COMPACT_DIR / f"chunk_{chunk_num:04d}.txt"

        with open(chunk_path, "w") as f:
            # Header
            sources = Counter(s["source_subdir"] for s in chunk)
            source_str = ", ".join(f"{k}:{v}" for k, v in sources.most_common(5))
            f.write(f"# Chunk {chunk_num} | {len(chunk)} samples | {source_str}\n")
            f.write(f"# For each: APPROVE if answer addresses the question, REJECT if mismatch\n")
            f.write("#\n")

            for s in chunk:
                f.write(f"{s['id']} | Q: {s['q_title']} | A: {s['a_start']}\n")

        chunk_files.append(chunk_path)

    # Also write a mapping file so we can trace IDs back to batch files
    mapping_path = COMPACT_DIR / "id_to_batch.json"
    id_to_batch = {s["id"]: s["batch_file"] for s in all_samples}
    with open(mapping_path, "w") as f:
        json.dump(id_to_batch, f)

    print(f"Created {len(chunk_files)} compact files in {COMPACT_DIR}")
    print(f"Chunk size: {chunk_size} samples each")
    print(f"Mapping: {mapping_path}")

    return chunk_files


def apply_decisions():
    """Apply decisions from decision files back to batch files."""
    if not DECISIONS_DIR.exists():
        print(f"No decisions directory found at {DECISIONS_DIR}")
        return

    # Load ID-to-batch mapping
    mapping_path = COMPACT_DIR / "id_to_batch.json"
    if not mapping_path.exists():
        print(f"No mapping file found at {mapping_path}")
        return

    with open(mapping_path) as f:
        id_to_batch = json.load(f)

    # Parse all decision files
    decisions = {}  # id -> status
    for df in sorted(DECISIONS_DIR.glob("*.txt")):
        with open(df) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Parse "ID:APPROVE" or "ID:REJECT" or "ID: APPROVE" etc
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue
                sid = parts[0].strip()
                status_raw = parts[1].strip().upper()
                if "APPROVE" in status_raw:
                    decisions[sid] = "approved"
                elif "REJECT" in status_raw:
                    decisions[sid] = "rejected"
                elif "TWEAK" in status_raw:
                    decisions[sid] = "tweaked"

    print(f"Loaded {len(decisions):,} decisions")
    status_counts = Counter(decisions.values())
    for status, count in status_counts.most_common():
        print(f"  {status}: {count:,}")

    # Group decisions by batch file
    batch_decisions = defaultdict(dict)  # batch_path -> {id: status}
    for sid, status in decisions.items():
        batch_path = id_to_batch.get(sid)
        if batch_path:
            batch_decisions[batch_path][sid] = status

    # Apply to each batch
    batches_updated = 0
    for batch_rel, bdecs in sorted(batch_decisions.items()):
        batch_path = REVIEW_DIR / batch_rel
        if not batch_path.exists():
            continue

        # Read batch
        samples = []
        with open(batch_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        # Apply decisions
        for sample in samples:
            sid = sample.get("id", "")
            if sid in bdecs:
                sample["status"] = bdecs[sid]

        # Write to reviewed directory
        batch_rel_path = Path(batch_rel)
        # needs_review/se_unix/batch_0001.jsonl -> reviewed/se_unix/batch_0001.jsonl
        reviewed_path = REVIEW_DIR / "reviewed" / batch_rel_path.relative_to("needs_review")
        reviewed_path.parent.mkdir(parents=True, exist_ok=True)

        with open(reviewed_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        batches_updated += 1

    print(f"\nUpdated {batches_updated} batch files in reviewed/")


def status():
    """Show compact review status."""
    if not COMPACT_DIR.exists():
        print("No compact files created yet. Run: python scripts/bulk_review.py create")
        return

    chunks = list(COMPACT_DIR.glob("chunk_*.txt"))
    decided = list(DECISIONS_DIR.glob("*.txt")) if DECISIONS_DIR.exists() else []

    print(f"Compact chunks: {len(chunks)}")
    print(f"Decision files:  {len(decided)}")

    if decided:
        total_decisions = 0
        status_counts = Counter()
        for df in decided:
            with open(df) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        total_decisions += 1
                        raw = parts[1].strip().upper()
                        if "APPROVE" in raw:
                            status_counts["approved"] += 1
                        elif "REJECT" in raw:
                            status_counts["rejected"] += 1
        print(f"Total decisions: {total_decisions:,}")
        for s, c in status_counts.most_common():
            print(f"  {s}: {c:,}")


def main():
    parser = argparse.ArgumentParser(description="Bulk SFT review tools")
    parser.add_argument("command", choices=["create", "apply", "status"],
                        help="create: make compact files, apply: apply decisions, status: show progress")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Samples per compact file (default: 500)")
    args = parser.parse_args()

    if args.command == "create":
        create_compact_files(args.chunk_size)
    elif args.command == "apply":
        apply_decisions()
    elif args.command == "status":
        status()


if __name__ == "__main__":
    main()
