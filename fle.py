#!/usr/bin/env python3
"""Free Language Embeddings — load and query V34 word vectors.

Usage:
    python fle.py                        # interactive mode
    python fle.py king - man + woman     # single query
    python fle.py --similar cat          # nearest neighbors

Requires: fle_v34.npz (download from GitHub releases)
"""

import numpy as np
import sys
import os

EMBEDDINGS_FILE = os.path.join(os.path.dirname(__file__), "fle_v34.npz")


class FLE:
    """Free Language Embeddings — 100K words, 300d, V34 dynamic masking word2vec."""

    def __init__(self, path=EMBEDDINGS_FILE):
        data = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"]  # (100000, 300) float32
        self.words = list(data["words"])
        self.word2id = {w: i for i, w in enumerate(self.words)}
        self._normed = None

    @property
    def normed(self):
        if self._normed is None:
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self._normed = self.embeddings / np.maximum(norms, 1e-8)
        return self._normed

    def __contains__(self, word):
        return word in self.word2id

    def __getitem__(self, word):
        return self.embeddings[self.word2id[word]]

    def similar(self, word, n=10):
        """Find n most similar words."""
        if word not in self.word2id:
            return []
        vec = self.normed[self.word2id[word]]
        sims = self.normed @ vec
        sims[self.word2id[word]] = -1
        top = np.argsort(-sims)[:n]
        return [(self.words[i], float(sims[i])) for i in top]

    def analogy(self, a, b, c, n=5):
        """a is to b as c is to ? (b - a + c)"""
        for w in [a, b, c]:
            if w not in self.word2id:
                return []
        vec = self.normed[self.word2id[b]] - self.normed[self.word2id[a]] + self.normed[self.word2id[c]]
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        sims = self.normed @ vec
        for w in [a, b, c]:
            sims[self.word2id[w]] = -1
        top = np.argsort(-sims)[:n]
        return [(self.words[i], float(sims[i])) for i in top]

    def similarity(self, a, b):
        """Cosine similarity between two words."""
        if a not in self.word2id or b not in self.word2id:
            return None
        return float(self.normed[self.word2id[a]] @ self.normed[self.word2id[b]])

    def query(self, expression):
        """Evaluate a vector arithmetic expression like 'king - man + woman'."""
        tokens = expression.strip().split()
        if not tokens:
            return []

        vec = np.zeros(self.embeddings.shape[1])
        sign = 1.0
        used = set()
        for token in tokens:
            if token == '+':
                sign = 1.0
            elif token == '-':
                sign = -1.0
            elif token in self.word2id:
                vec += sign * self.normed[self.word2id[token]]
                used.add(token)
                sign = 1.0
            else:
                return [(f"'{token}' not in vocabulary", 0.0)]

        vec = vec / (np.linalg.norm(vec) + 1e-8)
        sims = self.normed @ vec
        for w in used:
            sims[self.word2id[w]] = -1
        top = np.argsort(-sims)[:10]
        return [(self.words[i], float(sims[i])) for i in top]


def main():
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Error: {EMBEDDINGS_FILE} not found.")
        print("Download from: https://github.com/ruapotato/Free-Language-Embeddings/releases")
        sys.exit(1)

    fle = FLE()
    print(f"Loaded {len(fle.words):,} words, {fle.embeddings.shape[1]}d")

    # CLI mode
    if len(sys.argv) > 1:
        if sys.argv[1] == "--similar":
            word = sys.argv[2] if len(sys.argv) > 2 else "cat"
            for w, s in fle.similar(word, 15):
                print(f"  {w:<20} {s:.4f}")
        else:
            expr = " ".join(sys.argv[1:])
            print(f"  {expr}")
            for w, s in fle.query(expr):
                print(f"  → {w:<20} {s:.4f}")
        return

    # Interactive mode
    print("\nExamples:")
    print("  king - man + woman")
    print("  similar cat")
    print("  paris - france + germany")
    print()

    while True:
        try:
            line = input("fle> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("similar "):
            word = line.split()[1]
            results = fle.similar(word, 15)
            if not results:
                print(f"  '{word}' not in vocabulary")
            for w, s in results:
                print(f"  {w:<20} {s:.4f}")
        else:
            for w, s in fle.query(line):
                print(f"  {w:<20} {s:.4f}")


if __name__ == "__main__":
    main()
