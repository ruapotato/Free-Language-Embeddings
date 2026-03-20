#!/usr/bin/env python3
"""Exp 1: Articulatory bottleneck autoencoder.

Word → V34 embedding (300d) → encoder → 9d bottleneck → decoder → reconstruct 300d

The 9 bottleneck dimensions are sigmoid [0,1] — mouth shape parameters.
After training, render through Pink Trombone for alien language.

Usage:
    python experiments/exp_1_articulatory/train.py
"""

import json
import torch
import torch.nn as nn
import numpy as np
import os

CHECKPOINT = "checkpoints/word2vec_v34/latest.pt"
VOCAB_PATH = "checkpoints/word2vec_v28/vocab.json"
BOTTLENECK_DIM = 9
HIDDEN_DIM = 256
EPOCHS = 5000
BATCH_SIZE = 4096
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "experiments/exp_1_articulatory"

BOTTLENECK_NAMES = [
    "voicing", "tenseness", "place", "manner",
    "nasality", "lip_rounding", "vowel_height", "vowel_backness",
    "sibilance",
]


class ArticulatoryAutoencoder(nn.Module):
    def __init__(self, embed_dim=300, hidden_dim=HIDDEN_DIM, bottleneck_dim=BOTTLENECK_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, bottleneck_dim),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def main():
    print("=" * 60)
    print("Exp 1: Articulatory Bottleneck Autoencoder")
    print("=" * 60)

    # Load embeddings
    print(f"\nLoading embeddings from {CHECKPOINT}...")
    cp = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    embeddings = cp["model_state_dict"]["target_embeddings.weight"].numpy()

    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    id2word = {v: k for k, v in vocab["word2id"].items()}

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normed = embeddings / np.maximum(norms, 1e-8)

    X = torch.tensor(embeddings_normed, dtype=torch.float32).to(DEVICE)
    N, D = X.shape
    print(f"  {N:,} words, {D}d, bottleneck={BOTTLENECK_DIM}d")
    print(f"  Device: {DEVICE}, epochs: {EPOCHS}")

    model = ArticulatoryAutoencoder(embed_dim=D).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=1e-5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # Train
    print(f"\nTraining...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        total_loss = 0
        total_recon = 0
        total_spread = 0
        n_batches = 0

        for i in range(0, N, BATCH_SIZE):
            batch = X[perm[i:i + BATCH_SIZE]]
            recon, z = model(batch)

            recon_loss = nn.functional.mse_loss(recon, batch)

            # Spread loss: penalize low variance per dimension
            # Forces each bottleneck dim to actually differentiate words
            z_var = z.var(dim=0)  # variance per dim across batch
            spread_loss = torch.mean(torch.clamp(0.04 - z_var, min=0))  # want var > 0.04

            loss = recon_loss + 0.5 * spread_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_spread += spread_loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 500 == 0 or epoch == 1 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                recon_all, z_all = model(X)
                cos_sim = nn.functional.cosine_similarity(recon_all, X, dim=1).mean().item()
                z_std = z_all.std(dim=0).cpu().numpy()
                z_min = z_all.min(dim=0).values.cpu().numpy()
                z_max = z_all.max(dim=0).values.cpu().numpy()

            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:>5d}/{EPOCHS} | recon={total_recon/n_batches:.6f} | "
                  f"spread={total_spread/n_batches:.6f} | cos={cos_sim:.4f} | lr={lr_now:.1e}")
            print(f"         std=[{', '.join(f'{v:.3f}' for v in z_std)}]")
            print(f"         min=[{', '.join(f'{v:.3f}' for v in z_min)}]")
            print(f"         max=[{', '.join(f'{v:.3f}' for v in z_max)}]")

    # Final eval
    print("\n" + "=" * 60)
    print("Final results")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        recon_all, z_all = model(X)
        cos_sim = nn.functional.cosine_similarity(recon_all, X, dim=1).mean().item()
    print(f"  Reconstruction cosine similarity: {cos_sim:.4f}")

    z_np = z_all.cpu().numpy()
    print(f"\n  Bottleneck stats:")
    print(f"  {'Param':<18} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6} {'Range':>6}")
    print(f"  {'-'*54}")
    for i, name in enumerate(BOTTLENECK_NAMES):
        col = z_np[:, i]
        print(f"  {name:<18} {col.mean():>6.3f} {col.std():>6.3f} "
              f"{col.min():>6.3f} {col.max():>6.3f} {col.max()-col.min():>6.3f}")

    # Check semantic preservation
    w2i = vocab["word2id"]
    pairs = [
        ("cat", "dog"), ("king", "queen"), ("man", "woman"),
        ("hot", "cold"), ("red", "blue"), ("mother", "father"),
        ("cat", "king"), ("love", "water"), ("red", "walk"),
    ]
    print(f"\n  Semantic preservation:")
    print(f"  {'Pair':<22} {'V34 cos':>8} {'Bottleneck L2':>14}")
    print(f"  {'-'*48}")
    for w1, w2 in pairs:
        if w1 not in w2i or w2 not in w2i:
            continue
        v_cos = float(nn.functional.cosine_similarity(X[w2i[w1]:w2i[w1]+1], X[w2i[w2]:w2i[w2]+1]).item())
        z1 = z_np[w2i[w1]]
        z2 = z_np[w2i[w2]]
        z_dist = np.linalg.norm(z1 - z2)
        print(f"  {w1+'/'+w2:<22} {v_cos:>8.3f} {z_dist:>14.3f}")

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "bottleneck_names": BOTTLENECK_NAMES,
        "bottleneck_dim": BOTTLENECK_DIM,
        "embed_dim": D,
        "hidden_dim": HIDDEN_DIM,
        "cos_sim": cos_sim,
    }, os.path.join(OUT_DIR, "model.pt"))

    # Export test words
    test_words = [
        "hello", "world", "cat", "dog", "love", "hate",
        "big", "small", "hot", "cold", "king", "queen",
        "man", "woman", "water", "fire", "yes", "no",
        "happy", "sad", "run", "walk", "eat", "drink",
        "red", "blue", "green", "black", "white",
        "one", "two", "three", "four", "five",
        "mother", "father", "child", "baby",
        "sun", "moon", "star", "earth",
    ]

    print(f"\n  Test words:")
    print(f"  {'Word':<12} {' '.join(f'{n[:5]:>6}' for n in BOTTLENECK_NAMES)}")
    print(f"  {'-'*70}")

    export_data = {}
    for word in test_words:
        if word not in w2i:
            continue
        wid = w2i[word]
        z_vals = z_np[wid]
        print(f"  {word:<12} {' '.join(f'{v:>6.3f}' for v in z_vals)}")
        export_data[word] = z_vals.tolist()

    with open(os.path.join(OUT_DIR, "bottleneck_values.json"), "w") as f:
        json.dump({"param_names": BOTTLENECK_NAMES, "words": export_data}, f, indent=2)
    print(f"\n  Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
