#!/usr/bin/env python3
"""
review_batch.py — Review a batch of SFT samples
=================================================

Used by Claude Code agents to review batches of SFT training data.
Reads a batch file, evaluates each sample, writes reviewed output.

For each sample, the agent should determine:
  - "approved"  — Q/A match is correct, text is coherent and useful
  - "rejected"  — Q/A mismatch, incoherent, or low quality
  - "tweaked"   — Approved but with minor text fixes (store in tweaked_text)

Usage (by agent):
    python scripts/review_batch.py <batch_file> [--output <output_file>]

If --output is not specified, writes to data/sft/review/reviewed/<same_subpath>.

The agent READS the batch file, EVALUATES each sample using this tool's output,
then calls this script with decisions to record them.

For automated agent use, pipe decisions as JSON to stdin:
    echo '[{"id":"se_017000","status":"approved"},{"id":"se_017001","status":"rejected","notes":"mismatch"}]' | \
    python scripts/review_batch.py data/sft/review/needs_review/se_unix/batch_0001.jsonl --from-stdin
"""

import json
import sys
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
REVIEW_DIR = PROJECT_DIR / "data" / "sft" / "review"


def load_batch(batch_path):
    """Load all samples from a batch file."""
    samples = []
    with open(batch_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def show_batch(batch_path):
    """Display batch samples for review."""
    samples = load_batch(batch_path)
    print(f"Batch: {batch_path}")
    print(f"Samples: {len(samples)}")
    print(f"Source: {samples[0].get('source', '?') if samples else '?'}")
    print("=" * 70)

    for i, sample in enumerate(samples):
        text = sample["text"]
        quality = sample.get("quality", {})
        source = sample.get("source", "?")

        # Split into Q/A for display
        parts = text.split("<|assistant|>\n", 1)
        question = parts[0].replace("<|system|>\n", "[SYS] ").replace("<|user|>\n", "")
        answer = parts[1] if len(parts) > 1 else "?"

        print(f"\n--- [{i+1}/{len(samples)}] id={sample.get('id','?')} source={source} ---")

        # Quality signals
        if quality:
            q_str = ", ".join(f"{k}={v}" for k, v in quality.items())
            print(f"Quality: {q_str}")

        print(f"\nQ: {question[:500]}")
        if len(question) > 500:
            print(f"   ... ({len(question)} chars total)")
        print(f"\nA: {answer[:500]}")
        if len(answer) > 500:
            print(f"   ... ({len(answer)} chars total)")
        print()

    return samples


def apply_decisions(batch_path, decisions, output_path=None):
    """Apply review decisions to a batch and write output.

    Args:
        batch_path: path to the input batch file
        decisions: list of dicts with 'id' and 'status' (and optional 'notes', 'tweaked_text')
        output_path: where to write reviewed output (default: reviewed/<subpath>)
    """
    samples = load_batch(batch_path)

    # Build decision lookup
    decision_map = {d["id"]: d for d in decisions}

    # Determine output path
    if output_path is None:
        # Mirror the needs_review path under reviewed/
        batch_rel = Path(batch_path).resolve().relative_to(REVIEW_DIR / "needs_review")
        output_path = REVIEW_DIR / "reviewed" / batch_rel

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reviewed = []
    for sample in samples:
        sid = sample.get("id", "")
        decision = decision_map.get(sid, {})

        if decision:
            sample["status"] = decision.get("status", "pending")
            if "notes" in decision:
                sample["review_notes"] = decision["notes"]
            if "tweaked_text" in decision:
                sample["tweaked_text"] = decision["tweaked_text"]
        # else: keep original status (pending)

        reviewed.append(sample)

    with open(output_path, "w") as f:
        for item in reviewed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    approved = sum(1 for s in reviewed if s.get("status") == "approved")
    rejected = sum(1 for s in reviewed if s.get("status") == "rejected")
    tweaked = sum(1 for s in reviewed if s.get("status") == "tweaked")
    pending = sum(1 for s in reviewed if s.get("status") == "pending")

    print(f"Wrote {len(reviewed)} samples to {output_path}")
    print(f"  Approved: {approved}, Rejected: {rejected}, Tweaked: {tweaked}, Pending: {pending}")

    return reviewed


def main():
    parser = argparse.ArgumentParser(description="Review a batch of SFT samples")
    parser.add_argument("batch", help="Path to batch JSONL file")
    parser.add_argument("--output", help="Output path for reviewed batch")
    parser.add_argument(
        "--show", action="store_true",
        help="Just display the batch for review (no decisions)",
    )
    parser.add_argument(
        "--from-stdin", action="store_true",
        help="Read decisions from stdin as JSON array",
    )
    args = parser.parse_args()

    if args.show:
        show_batch(args.batch)
        return

    if args.from_stdin:
        decisions_json = sys.stdin.read()
        decisions = json.loads(decisions_json)
        apply_decisions(args.batch, decisions, args.output)
        return

    # Default: show batch
    show_batch(args.batch)


if __name__ == "__main__":
    main()
