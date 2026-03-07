#!/usr/bin/env python3
"""
Probe the concept space geometry of the current encoder.

Tests whether semantic relationships form consistent directions:
- Do "cat" and "dog" cluster as animals?
- Does "big cat" - "cat" ≈ "big dog" - "dog"? (size direction)
- Does "king" - "man" ≈ "queen" - "woman"? (royalty direction)

Usage:
    python probe_concepts.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from encoder_model import EncoderConfig, SemanticEncoder

CHECKPOINT = "checkpoints/encoder_v4/latest.pt"


def load_model():
    path = Path(CHECKPOINT)
    if not path.exists():
        # Try to find any checkpoint
        ckpt_dir = Path("checkpoints/encoder_v4")
        candidates = sorted(ckpt_dir.glob("step_*.pt")) if ckpt_dir.exists() else []
        if candidates:
            path = candidates[-1]
        else:
            print("No checkpoint found. Run with current training model.")
            return None, None

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = EncoderConfig(**ckpt["config"])
    model = SemanticEncoder(config)
    # Strip _orig_mod. prefix from torch.compile checkpoints
    state = ckpt["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    step = ckpt.get("step", "?")
    print(f"Loaded checkpoint: step {step}")
    return model, tokenizer


@torch.no_grad()
def encode(model, tokenizer, text):
    enc = tokenizer(text, max_length=128, padding=True,
                    truncation=True, return_tensors="pt")
    vec = model(enc["input_ids"], enc["attention_mask"])
    return vec.cpu().numpy()[0]


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def probe_clusters(model, tokenizer):
    """Check if related concepts cluster together."""
    print("\n" + "=" * 60)
    print("CONCEPT CLUSTERING")
    print("=" * 60)

    groups = {
        "Animals": ["a cat", "a dog", "a bird", "a fish", "a horse"],
        "Vehicles": ["a car", "a truck", "a bus", "a bicycle", "a train"],
        "Food": ["an apple", "bread", "cheese", "rice", "soup"],
        "Emotions": ["I am happy", "I am sad", "I am angry", "I am scared", "I am excited"],
        "Colors": ["the red ball", "the blue ball", "the green ball", "the yellow ball", "the white ball"],
        "Sizes": ["a big house", "a small house", "a tiny house", "a huge house", "a large house"],
    }

    # Encode all
    group_vecs = {}
    for name, phrases in groups.items():
        vecs = [encode(model, tokenizer, p) for p in phrases]
        group_vecs[name] = (phrases, vecs)

    # Within-group similarity (should be HIGH)
    # Between-group similarity (should be LOW)
    print(f"\n{'Group':<12s} {'Within':>8s} {'Between':>8s} {'Ratio':>8s}")
    print("-" * 40)

    group_names = list(groups.keys())
    for name in group_names:
        phrases, vecs = group_vecs[name]

        # Within-group: average pairwise similarity
        within_sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                within_sims.append(cosine(vecs[i], vecs[j]))
        within = np.mean(within_sims)

        # Between-group: average sim to other groups' centroids
        centroid = np.mean(vecs, axis=0)
        between_sims = []
        for other_name in group_names:
            if other_name == name:
                continue
            other_centroid = np.mean(group_vecs[other_name][1], axis=0)
            between_sims.append(cosine(centroid, other_centroid))
        between = np.mean(between_sims)

        ratio = within / max(between, 0.01)
        print(f"{name:<12s} {within:>8.3f} {between:>8.3f} {ratio:>8.2f}")


def probe_analogies(model, tokenizer):
    """Check if semantic directions are consistent (king-man+woman≈queen)."""
    print("\n" + "=" * 60)
    print("CONCEPT ANALOGIES (a - b + c ≈ d?)")
    print("=" * 60)

    analogies = [
        # (a, b, c, expected_d)
        ("king", "man", "woman", "queen"),
        ("a big cat", "a cat", "a dog", "a big dog"),
        ("a small cat", "a cat", "a dog", "a small dog"),
        ("a fast car", "a car", "a boat", "a fast boat"),
        ("he is happy", "he", "she", "she is happy"),
        ("the cat sat on the mat", "cat", "dog", "the dog sat on the mat"),
        ("I ate breakfast", "breakfast", "dinner", "I ate dinner"),
        ("a hot day", "hot", "cold", "a cold day"),
    ]

    print(f"\n{'a':<25s} {'b':<12s} {'c':<12s} {'expect d':<20s} {'cos(result,d)':>12s}")
    print("-" * 85)

    for a, b, c, d in analogies:
        va = encode(model, tokenizer, a)
        vb = encode(model, tokenizer, b)
        vc = encode(model, tokenizer, c)
        vd = encode(model, tokenizer, d)

        # a - b + c should ≈ d
        result = va - vb + vc
        result = result / np.linalg.norm(result)

        sim = cosine(result, vd)
        direct_sim = cosine(va, vd)  # baseline: just how similar a and d are directly

        marker = " ✓" if sim > 0.7 else ""
        print(f"{a:<25s} {b:<12s} {c:<12s} {d:<20s} {sim:>8.3f} (direct={direct_sim:.3f}){marker}")


def probe_directions(model, tokenizer):
    """Check if modifying concepts changes vectors in consistent directions."""
    print("\n" + "=" * 60)
    print("CONCEPT DIRECTIONS")
    print("Do 'big X' - 'X' vectors point the same way for different X?")
    print("=" * 60)

    modifications = {
        "Size (big)": [
            ("a cat", "a big cat"),
            ("a dog", "a big dog"),
            ("a house", "a big house"),
            ("a car", "a big car"),
        ],
        "Negation (not)": [
            ("I am happy", "I am not happy"),
            ("the cat is alive", "the cat is not alive"),
            ("it is raining", "it is not raining"),
            ("he can swim", "he cannot swim"),
        ],
        "Tense (past)": [
            ("I run", "I ran"),
            ("I eat", "I ate"),
            ("I swim", "I swam"),
            ("I write", "I wrote"),
        ],
        "Plural": [
            ("a cat", "many cats"),
            ("a dog", "many dogs"),
            ("a car", "many cars"),
            ("a tree", "many trees"),
        ],
    }

    for mod_name, pairs in modifications.items():
        deltas = []
        for base, modified in pairs:
            vb = encode(model, tokenizer, base)
            vm = encode(model, tokenizer, modified)
            delta = vm - vb
            delta = delta / (np.linalg.norm(delta) + 1e-8)
            deltas.append(delta)

        # Check consistency: how similar are the delta directions?
        consistencies = []
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                consistencies.append(cosine(deltas[i], deltas[j]))

        mean_consistency = np.mean(consistencies)
        print(f"\n  {mod_name}: direction consistency = {mean_consistency:.3f}")
        print(f"    (1.0 = perfectly consistent direction, 0.0 = random)")
        for (base, modified), delta in zip(pairs, deltas):
            sim = cosine(encode(model, tokenizer, base),
                         encode(model, tokenizer, modified))
            print(f"      {base:>20s} → {modified:<20s}  sim={sim:.3f}")


def probe_similarity_matrix(model, tokenizer):
    """Print a similarity matrix of diverse concepts."""
    print("\n" + "=" * 60)
    print("CONCEPT SIMILARITY MATRIX")
    print("=" * 60)

    concepts = [
        "a cat",
        "a dog",
        "a big cat",
        "a big dog",
        "a car",
        "I am happy",
        "I am sad",
        "le chat",          # French: the cat
        "the stock market",
    ]

    vecs = [encode(model, tokenizer, c) for c in concepts]

    # Print header
    short = [c[:10] for c in concepts]
    print(f"\n{'':>14s}", end="")
    for s in short:
        print(f"{s:>11s}", end="")
    print()

    for i, c in enumerate(concepts):
        print(f"{short[i]:>14s}", end="")
        for j in range(len(concepts)):
            sim = cosine(vecs[i], vecs[j])
            print(f"{sim:>11.3f}", end="")
        print()


def main():
    model, tokenizer = load_model()
    if model is None:
        return

    probe_similarity_matrix(model, tokenizer)
    probe_clusters(model, tokenizer)
    probe_directions(model, tokenizer)
    probe_analogies(model, tokenizer)


if __name__ == "__main__":
    main()
