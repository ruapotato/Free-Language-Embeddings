#!/usr/bin/env python3
"""Probe concept geometry of pretrained sentence encoders."""

import numpy as np
import gc


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_probes(model_name):
    from sentence_transformers import SentenceTransformer
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name}")
    print("=" * 70)
    model = SentenceTransformer(model_name, device="cpu")

    def encode(text):
        return model.encode(text, normalize_embeddings=True)

    # CLUSTERING
    print("\nCLUSTERING:")
    groups = {
        "Animals": ["a cat", "a dog", "a bird", "a fish"],
        "Vehicles": ["a car", "a truck", "a bus", "a train"],
        "Emotions": ["I am happy", "I am sad", "I am angry", "I am scared"],
    }
    group_vecs = {n: [encode(p) for p in ps] for n, ps in groups.items()}
    for name, vecs in group_vecs.items():
        within = np.mean([cos(vecs[i], vecs[j])
                          for i in range(len(vecs)) for j in range(i + 1, len(vecs))])
        centroid = np.mean(vecs, axis=0)
        other_sims = []
        for o in groups:
            if o != name:
                other_sims.append(cos(centroid, np.mean(group_vecs[o], axis=0)))
        between = np.mean(other_sims)
        print(f"  {name:<12s} within={within:.3f} between={between:.3f} "
              f"ratio={within / max(between, 0.01):.2f}")

    # DIRECTIONS
    print("\nDIRECTION CONSISTENCY:")
    direction_tests = {
        "Size": [("a cat", "a big cat"), ("a dog", "a big dog"),
                 ("a house", "a big house"), ("a car", "a big car")],
        "Negation": [("I am happy", "I am not happy"),
                     ("the cat is alive", "the cat is not alive"),
                     ("it is raining", "it is not raining"),
                     ("he can swim", "he cannot swim")],
        "Tense": [("I run", "I ran"), ("I eat", "I ate"),
                  ("I swim", "I swam"), ("I write", "I wrote")],
    }
    for dir_name, pairs in direction_tests.items():
        deltas = []
        for base, mod in pairs:
            d = encode(mod) - encode(base)
            d = d / (np.linalg.norm(d) + 1e-8)
            deltas.append(d)
        sims = [cos(deltas[i], deltas[j])
                for i in range(len(deltas)) for j in range(i + 1, len(deltas))]
        print(f"  {dir_name:<12s} {np.mean(sims):.3f}")

    # ANALOGIES
    print("\nANALOGIES (a - b + c ~ d):")
    analogies = [
        ("a big cat", "a cat", "a dog", "a big dog"),
        ("he is happy", "he", "she", "she is happy"),
        ("a hot day", "hot", "cold", "a cold day"),
        ("I ate breakfast", "breakfast", "dinner", "I ate dinner"),
    ]
    for a, b, c, d in analogies:
        va, vb, vc, vd = encode(a), encode(b), encode(c), encode(d)
        r = va - vb + vc
        r = r / (np.linalg.norm(r) + 1e-8)
        print(f"  {a:<20s} - {b:<12s} + {c:<8s} ~ {d:<20s} sim={cos(r, vd):.3f}")

    # KEY PAIRS
    print("\nKEY PAIRS:")
    key_pairs = [
        ("a cat", "a dog", "similar animals"),
        ("a cat", "a car", "surface similar"),
        ("the cat sat on the mat", "le chat etait assis sur le tapis", "cross-lingual"),
        ("the cat sat on the mat", "the stock market crashed", "unrelated"),
        ("the dog bit the man", "the man bit the dog", "word order"),
        ("I am happy", "I am sad", "opposite emotions"),
    ]
    for a, b, label in key_pairs:
        sim = cos(encode(a), encode(b))
        print(f"  {sim:+.3f}  {a:<35s} <-> {b:<35s} ({label})")

    del model
    gc.collect()


if __name__ == "__main__":
    import sys
    models = sys.argv[1:] if len(sys.argv) > 1 else [
        "sentence-transformers/all-MiniLM-L6-v2",
    ]
    for m in models:
        run_probes(m)
