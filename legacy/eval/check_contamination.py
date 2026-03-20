#!/usr/bin/env python3
"""Check for contamination between LinuxBench questions and SFT training data.

Verifies no overlap between eval/linux_bench.json and data/sft/*.jsonl using:
1. Exact substring matching (full question text in training data)
2. N-gram overlap (8-gram sequences from questions appearing in training data)
3. Question stem matching (question without choices in training conversations)
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def load_sft_texts(sft_dir: str) -> list[str]:
    """Load all text content from SFT JSONL files."""
    texts = []
    sft_path = Path(sft_dir)
    if not sft_path.exists():
        print(f"Warning: SFT directory {sft_dir} not found")
        return texts

    for jsonl_file in sorted(sft_path.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    text = entry.get("text", "")
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    continue

    return texts


def load_benchmark(bench_path: str) -> list[dict]:
    """Load benchmark questions from JSON."""
    with open(bench_path) as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def get_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Extract word-level n-grams from text."""
    words = text.split()
    if len(words) < n:
        return set()
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}


def check_exact_substring(question_text: str, training_texts: list[str]) -> list[str]:
    """Check if the question text appears as an exact substring in any training text."""
    matches = []
    norm_q = normalize_text(question_text)
    # Only check if the question is long enough to be meaningful
    if len(norm_q) < 20:
        return matches

    for text in training_texts:
        norm_t = normalize_text(text)
        if norm_q in norm_t:
            # Return a snippet for context
            idx = norm_t.find(norm_q)
            start = max(0, idx - 50)
            end = min(len(norm_t), idx + len(norm_q) + 50)
            matches.append(f"...{norm_t[start:end]}...")

    return matches


def check_ngram_overlap(question_text: str, training_ngrams: set[tuple[str, ...]], n: int = 8) -> list[tuple[str, ...]]:
    """Check if any n-grams from the question appear in training data."""
    q_ngrams = get_ngrams(normalize_text(question_text), n)
    overlapping = q_ngrams & training_ngrams
    return list(overlapping)


def check_question_stem(question_stem: str, training_texts: list[str]) -> list[str]:
    """Check if the question stem (without choices) appears in training data."""
    matches = []
    norm_stem = normalize_text(question_stem)
    if len(norm_stem) < 30:
        return matches

    for text in training_texts:
        norm_t = normalize_text(text)
        if norm_stem in norm_t:
            idx = norm_t.find(norm_stem)
            start = max(0, idx - 50)
            end = min(len(norm_t), idx + len(norm_stem) + 50)
            matches.append(f"...{norm_t[start:end]}...")

    return matches


def main():
    # Resolve paths relative to project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    bench_path = script_dir / "linux_bench.json"
    sft_dir = project_root / "data" / "sft"

    if not bench_path.exists():
        print(f"Error: Benchmark file not found: {bench_path}")
        sys.exit(1)

    print("Loading SFT training data...")
    training_texts = load_sft_texts(str(sft_dir))
    print(f"  Loaded {len(training_texts)} training conversations from {sft_dir}")

    print("Loading benchmark questions...")
    questions = load_benchmark(str(bench_path))
    print(f"  Loaded {len(questions)} questions from {bench_path}")

    # Pre-compute training n-grams for efficiency
    print("Building training 8-gram index...")
    training_ngrams = set()
    for text in training_texts:
        training_ngrams |= get_ngrams(normalize_text(text), 8)
    print(f"  Built index with {len(training_ngrams):,} unique 8-grams")

    # Check each question
    flagged = []
    clean = 0

    print("\nChecking questions for contamination...")
    for q in questions:
        qid = q["id"]
        question_text = q["question"]
        choices_text = " ".join(q["choices"].values())
        full_text = f"{question_text} {choices_text}"
        issues = []

        # Check 1: Exact substring of full question+choices
        exact_matches = check_exact_substring(full_text, training_texts)
        if exact_matches:
            issues.append(("exact_full", exact_matches[:3]))

        # Check 2: Exact substring of question stem only
        stem_matches = check_question_stem(question_text, training_texts)
        if stem_matches:
            issues.append(("exact_stem", stem_matches[:3]))

        # Check 3: 8-gram overlap
        ngram_matches = check_ngram_overlap(full_text, training_ngrams, 8)
        if ngram_matches:
            issues.append(("8gram", [" ".join(ng) for ng in ngram_matches[:5]]))

        if issues:
            flagged.append({"id": qid, "question": question_text, "issues": issues})
        else:
            clean += 1

        # Progress indicator
        if qid % 50 == 0:
            print(f"  Checked {qid}/{len(questions)} questions...")

    # Summary
    print("\n" + "=" * 60)
    print("CONTAMINATION CHECK RESULTS")
    print("=" * 60)
    print(f"Total questions:   {len(questions)}")
    print(f"Clean questions:   {clean}")
    print(f"Flagged questions: {len(flagged)}")
    print()

    if flagged:
        print("FLAGGED QUESTIONS (review manually):")
        print("-" * 60)
        for f in flagged:
            print(f"\n  ID {f['id']}: {f['question'][:80]}...")
            for issue_type, matches in f["issues"]:
                print(f"    [{issue_type}]:")
                for m in matches:
                    print(f"      {m[:120]}")
        print()
        print(f"ACTION REQUIRED: {len(flagged)} questions need review/replacement.")
        sys.exit(1)
    else:
        print("All questions are clean. No contamination detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
