"""Evaluate word2vec embeddings on the Google analogy test set."""

import json
import sys
import torch
import numpy as np
from collections import defaultdict
import time

def main():
    # Accept checkpoint path as CLI argument
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        ckpt_path = "checkpoints/word2vec_v28/latest.pt"

    print("=" * 70)
    print(f"Word2Vec — Google Analogy Test Evaluation")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 70)

    # Load vocab (shared across all word2vec versions)
    print("\nLoading vocab...")
    with open("checkpoints/word2vec_v28/vocab.json") as f:
        vocab_data = json.load(f)
    word2id = vocab_data["word2id"]
    vocab_size = len(word2id)
    print(f"  Vocab size: {vocab_size:,}")

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"  Training step: {ckpt.get('step', 'unknown'):,}")
    print(f"  Embedding dim: {ckpt.get('embed_dim', 'unknown')}")

    # Extract target embeddings and normalize
    embeddings = ckpt["model_state_dict"]["target_embeddings.weight"].numpy()  # (V, D)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embeddings / norms  # (V, D)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Parse questions-words.txt
    print("\nParsing analogy questions...")
    categories = []
    current_cat = None
    questions = []

    # Track semantic vs syntactic split
    # The standard split: first 5 categories are semantic, rest are syntactic
    semantic_categories = set()
    syntactic_categories = set()
    cat_index = 0

    with open("data/questions-words.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                current_cat = line[2:]  # strip ": "
                categories.append(current_cat)
                cat_index = len(categories) - 1
                # First 5 categories are semantic, rest are syntactic
                if cat_index < 5:
                    semantic_categories.add(current_cat)
                else:
                    syntactic_categories.add(current_cat)
            else:
                words = line.lower().split()
                if len(words) == 4:
                    questions.append((current_cat, words[0], words[1], words[2], words[3]))

    print(f"  Total questions: {len(questions):,}")
    print(f"  Categories: {len(categories)}")
    print(f"  Semantic categories ({len(semantic_categories)}): {sorted(semantic_categories)}")
    print(f"  Syntactic categories ({len(syntactic_categories)}): {sorted(syntactic_categories)}")

    # Evaluate
    print("\nEvaluating analogies...")
    t0 = time.time()

    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    cat_covered = defaultdict(int)
    total_correct = 0
    total_covered = 0
    total_questions = 0

    for i, (cat, w1, w2, w3, w4) in enumerate(questions):
        total_questions += 1
        cat_total[cat] += 1

        # Check all words in vocab
        if w1 not in word2id or w2 not in word2id or w3 not in word2id or w4 not in word2id:
            continue

        total_covered += 1
        cat_covered[cat] += 1

        id1, id2, id3, id4 = word2id[w1], word2id[w2], word2id[w3], word2id[w4]

        # Compute analogy vector: w2 - w1 + w3
        vec = normed[id2] - normed[id1] + normed[id3]
        # Normalize query
        vec = vec / (np.linalg.norm(vec) + 1e-8)

        # Cosine similarity with all words
        sims = normed @ vec  # (V,)

        # Exclude input words
        sims[id1] = -2.0
        sims[id2] = -2.0
        sims[id3] = -2.0

        pred_id = np.argmax(sims)

        if pred_id == id4:
            total_correct += 1
            cat_correct[cat] += 1

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {i+1:,}/{len(questions):,} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    coverage = total_covered / total_questions * 100 if total_questions else 0
    accuracy = total_correct / total_covered * 100 if total_covered else 0
    print(f"\nOverall accuracy:  {total_correct:,} / {total_covered:,} = {accuracy:.2f}%")
    print(f"Coverage:          {total_covered:,} / {total_questions:,} = {coverage:.2f}%")

    # Per-category
    print(f"\n{'Category':<35} {'Correct':>8} {'Covered':>8} {'Total':>8} {'Acc%':>8} {'Cov%':>8}")
    print("-" * 70)

    sem_correct = 0
    sem_covered = 0
    sem_total = 0
    syn_correct = 0
    syn_covered = 0
    syn_total = 0

    for cat in categories:
        c = cat_correct[cat]
        cov = cat_covered[cat]
        t = cat_total[cat]
        acc = c / cov * 100 if cov else 0
        cov_pct = cov / t * 100 if t else 0
        print(f"  {cat:<33} {c:>8,} {cov:>8,} {t:>8,} {acc:>7.2f}% {cov_pct:>7.2f}%")

        if cat in semantic_categories:
            sem_correct += c
            sem_covered += cov
            sem_total += t
        else:
            syn_correct += c
            syn_covered += cov
            syn_total += t

    print("-" * 70)

    # Semantic vs Syntactic
    sem_acc = sem_correct / sem_covered * 100 if sem_covered else 0
    syn_acc = syn_correct / syn_covered * 100 if syn_covered else 0
    sem_cov = sem_covered / sem_total * 100 if sem_total else 0
    syn_cov = syn_covered / syn_total * 100 if syn_total else 0

    print(f"\n{'Split':<35} {'Correct':>8} {'Covered':>8} {'Total':>8} {'Acc%':>8} {'Cov%':>8}")
    print("-" * 70)
    print(f"  {'Semantic':<33} {sem_correct:>8,} {sem_covered:>8,} {sem_total:>8,} {sem_acc:>7.2f}% {sem_cov:>7.2f}%")
    print(f"  {'Syntactic':<33} {syn_correct:>8,} {syn_covered:>8,} {syn_total:>8,} {syn_acc:>7.2f}% {syn_cov:>7.2f}%")
    print(f"  {'Total':<33} {total_correct:>8,} {total_covered:>8,} {total_questions:>8,} {accuracy:>7.2f}% {coverage:>7.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
