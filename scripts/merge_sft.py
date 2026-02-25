#!/usr/bin/env python3
"""
Validate, merge, shuffle, and report stats for flm SFT data files.

Steps:
  1. Validate each JSONL file (valid JSON, required fields, chat tags)
  2. Strip invalid lines in-place
  3. Merge all valid conversations into flm_combined.jsonl
  4. Shuffle with fixed seed (42) for reproducibility
  5. Print detailed stats
"""

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
SFT_DIR = REPO_ROOT / "data" / "sft"

# Files to include (order matches the spec)
SOURCE_FILES = [
    "flm_identity.jsonl",
    "flm_general.jsonl",
    "flm_os_admin.jsonl",
    "flm_packaging.jsonl",
    "flm_troubleshoot.jsonl",
    "flm_troubleshoot_v2.jsonl",
    "flm_troubleshoot_v3.jsonl",
    "flm_linux_admin.jsonl",
    "flm_easter_eggs.jsonl",
    "flm_personality.jsonl",
    "flm_system_aware.jsonl",
    "flm_multiturn_1.jsonl",
    "flm_multiturn_2.jsonl",
    # V2 additions — gap coverage
    "flm_networking_v2.jsonl",
    "flm_systemd.jsonl",
    "flm_filesystems.jsonl",
    "flm_security.jsonl",
    "flm_shell_scripting.jsonl",
    "flm_processes.jsonl",
    "flm_text_processing.jsonl",
    "flm_backup.jsonl",
    "flm_webservers.jsonl",
    "flm_logs.jsonl",
    "flm_performance.jsonl",
    "flm_debian_v2.jsonl",
    "flm_containers.jsonl",
    "flm_git.jsonl",
    "flm_users.jsonl",
    "flm_databases.jsonl",
]

OUTPUT_FILE = SFT_DIR / "flm_combined.jsonl"
SHUFFLE_SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────────

def validate_line(line: str) -> tuple[dict | None, str | None]:
    """Validate a single JSONL line.

    Returns (parsed_obj, None) on success or (None, reason) on failure.
    """
    line = line.strip()
    if not line:
        return None, "empty line"

    # 1. Valid JSON?
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"

    # 2. Required fields?
    if "text" not in obj:
        return None, "missing 'text' field"
    if "source" not in obj:
        return None, "missing 'source' field"

    text = obj["text"]

    # 3. Chat tags present?
    if "<|user|>" not in text:
        return None, "missing <|user|> tag"
    if "<|assistant|>" not in text:
        return None, "missing <|assistant|> tag"

    return obj, None


def validate_and_clean_file(filepath: Path) -> tuple[list[dict], int, int, list[str]]:
    """Validate a JSONL file, remove invalid lines, return valid entries.

    Returns (valid_entries, total_valid, total_invalid, error_reasons).
    """
    valid_entries = []
    invalid_count = 0
    error_reasons = []

    with open(filepath, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    valid_lines = []
    for i, line in enumerate(raw_lines, start=1):
        obj, reason = validate_line(line)
        if obj is not None:
            valid_entries.append(obj)
            valid_lines.append(line if line.endswith("\n") else line + "\n")
        else:
            if reason != "empty line":
                invalid_count += 1
                error_reasons.append(f"  line {i}: {reason}")

    # Write back only valid lines (strip bad lines)
    if invalid_count > 0:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(valid_lines)

    return valid_entries, len(valid_entries), invalid_count, error_reasons


def count_turns(text: str) -> int:
    """Count the number of user/assistant exchange pairs."""
    return len(re.findall(r"<\|user\|>", text))


def has_system_prompt(text: str) -> bool:
    return "<|system|>" in text


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("flm SFT Data Validator & Merger")
    print("=" * 72)

    # ── Step 1: Validate ─────────────────────────────────────────────────
    print("\n--- Step 1: Validate all JSONL files ---\n")

    all_entries = []
    total_valid_global = 0
    total_invalid_global = 0

    for fname in SOURCE_FILES:
        fpath = SFT_DIR / fname
        if not fpath.exists():
            print(f"  WARNING: {fname} not found, skipping.")
            continue

        entries, n_valid, n_invalid, reasons = validate_and_clean_file(fpath)
        all_entries.extend(entries)
        total_valid_global += n_valid
        total_invalid_global += n_invalid

        status = "OK" if n_invalid == 0 else "CLEANED"
        print(f"  {fname:<35s}  valid={n_valid:>5d}  invalid={n_invalid:>3d}  [{status}]")
        for r in reasons:
            print(f"    {r}")

    print(f"\n  TOTAL across all files:  valid={total_valid_global}  invalid={total_invalid_global}")

    # ── Step 2: Merge ────────────────────────────────────────────────────
    print("\n--- Step 2: Merge into combined file ---\n")

    print(f"  Writing {len(all_entries)} conversations to {OUTPUT_FILE.name} ...")

    # ── Step 3: Shuffle ──────────────────────────────────────────────────
    print(f"\n--- Step 3: Shuffle (seed={SHUFFLE_SEED}) ---\n")

    random.seed(SHUFFLE_SEED)
    random.shuffle(all_entries)
    print(f"  Shuffled {len(all_entries)} entries.")

    # Write combined file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  Wrote {OUTPUT_FILE}")

    # ── Step 4: Report stats ─────────────────────────────────────────────
    print("\n--- Step 4: Final statistics ---\n")

    total = len(all_entries)
    print(f"  Total conversations: {total}\n")

    # Breakdown by source
    source_counts = Counter(e["source"] for e in all_entries)
    print("  Breakdown by source:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src:<30s} {cnt:>5d}  ({100*cnt/total:.1f}%)")

    # System prompts
    n_with_sys = sum(1 for e in all_entries if has_system_prompt(e["text"]))
    n_without_sys = total - n_with_sys
    print(f"\n  With system prompt:    {n_with_sys:>5d}  ({100*n_with_sys/total:.1f}%)")
    print(f"  Without system prompt: {n_without_sys:>5d}  ({100*n_without_sys/total:.1f}%)")

    # Multi-turn
    n_multi = sum(1 for e in all_entries if count_turns(e["text"]) > 1)
    n_single = total - n_multi
    print(f"\n  Multi-turn (>1 exchange): {n_multi:>5d}  ({100*n_multi/total:.1f}%)")
    print(f"  Single-turn:             {n_single:>5d}  ({100*n_single/total:.1f}%)")

    # Average text length per source
    source_lengths = defaultdict(list)
    for e in all_entries:
        source_lengths[e["source"]].append(len(e["text"]))

    print("\n  Average text length (chars) per source:")
    for src in sorted(source_lengths.keys()):
        lengths = source_lengths[src]
        avg = sum(lengths) / len(lengths)
        mn = min(lengths)
        mx = max(lengths)
        print(f"    {src:<30s}  avg={avg:>7.0f}  min={mn:>5d}  max={mx:>6d}")

    # Duplicate detection (exact text match)
    text_counts = Counter(e["text"] for e in all_entries)
    duplicates = {t: c for t, c in text_counts.items() if c > 1}
    if duplicates:
        print(f"\n  Duplicate texts found: {len(duplicates)} unique texts appearing more than once")
        total_dup_lines = sum(c for c in duplicates.values()) - len(duplicates)
        print(f"  Total extra duplicate lines: {total_dup_lines}")
        print(f"  Duplicate details (showing first 80 chars of text):")
        for text, count in sorted(duplicates.items(), key=lambda x: -x[1]):
            snippet = text.replace("\n", "\\n")[:80]
            # Find sources for this duplicate
            dup_sources = [e["source"] for e in all_entries if e["text"] == text]
            print(f"    [{count}x] sources={set(dup_sources)}  \"{snippet}...\"")
    else:
        print(f"\n  Duplicate texts: none found")

    print("\n" + "=" * 72)
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
