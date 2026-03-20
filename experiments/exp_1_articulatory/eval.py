#!/usr/bin/env python3
"""Evaluate articulatory bottleneck autoencoder reconstruction quality."""

import json
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, "experiments/exp_1_articulatory")
from train import ArticulatoryAutoencoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # Load model
    cp = torch.load("experiments/exp_1_articulatory/model.pt", map_location="cpu", weights_only=False)
    model = ArticulatoryAutoencoder(
        embed_dim=cp["embed_dim"], hidden_dim=cp["hidden_dim"],
        bottleneck_dim=cp["bottleneck_dim"]
    ).to(DEVICE)
    model.load_state_dict(cp["model_state_dict"])
    model.eval()

    # Load embeddings
    ecp = torch.load("checkpoints/word2vec_v34/latest.pt", map_location="cpu", weights_only=False)
    embeddings = ecp["model_state_dict"]["target_embeddings.weight"].numpy()
    with open("checkpoints/word2vec_v28/vocab.json") as f:
        vocab = json.load(f)
    w2i = vocab["word2id"]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    X = torch.tensor(embeddings / np.maximum(norms, 1e-8), dtype=torch.float32).to(DEVICE)
    N, D = X.shape

    with torch.no_grad():
        recon, z = model(X)
        cos_sim = nn.functional.cosine_similarity(recon, X, dim=1)
        mse = nn.functional.mse_loss(recon, X).item()

    print("=" * 65)
    print("ARTICULATORY BOTTLENECK — FINAL EVALUATION")
    print("=" * 65)
    print(f"  Vocab size:        {N:,} words")
    print(f"  Input dim:         {D}")
    print(f"  Bottleneck dim:    {cp['bottleneck_dim']}")
    print(f"  Compression ratio: {D / cp['bottleneck_dim']:.1f}x")
    print()

    # --- Reconstruction quality ---
    print("RECONSTRUCTION QUALITY")
    print("-" * 45)
    print(f"  Mean cosine sim:   {cos_sim.mean().item():.4f}")
    print(f"  Median cosine sim: {cos_sim.median().item():.4f}")
    print(f"  Std cosine sim:    {cos_sim.std().item():.4f}")
    print(f"  Min cosine sim:    {cos_sim.min().item():.4f}")
    print(f"  Max cosine sim:    {cos_sim.max().item():.4f}")
    print(f"  MSE:               {mse:.6f}")
    for pct in [5, 25, 75, 95]:
        val = torch.quantile(cos_sim, pct / 100).item()
        print(f"  P{pct:<2} cosine sim:    {val:.4f}")

    # --- Bottleneck utilization ---
    z_np = z.cpu().numpy()
    names = cp["bottleneck_names"]
    print()
    print("BOTTLENECK UTILIZATION")
    print("-" * 65)
    print(f"  {'Param':<16} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6} {'Range':>6} {'Entropy':>8}")
    print(f"  {'-' * 58}")
    for i, name in enumerate(names):
        col = z_np[:, i]
        hist, _ = np.histogram(col, bins=20, range=(0, 1))
        p = hist / hist.sum()
        p = p[p > 0]
        entropy = -np.sum(p * np.log2(p))
        print(f"  {name:<16} {col.mean():>6.3f} {col.std():>6.3f} "
              f"{col.min():>6.3f} {col.max():>6.3f} {col.max()-col.min():>6.3f} {entropy:>8.3f}")
    print(f"  (max entropy for 20 bins = {np.log2(20):.3f})")

    # --- Dimension correlations ---
    corr = np.corrcoef(z_np.T)
    print()
    print("BOTTLENECK DIMENSION CORRELATIONS (abs > 0.3)")
    print("-" * 45)
    found = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(corr[i, j]) > 0.3:
                print(f"  {names[i]}/{names[j]}: {corr[i,j]:.3f}")
                found = True
    if not found:
        print("  None — all dims fairly independent")

    # --- Nearest neighbor agreement (batched on GPU) ---
    print()
    print("SEMANTIC PRESERVATION — NEAREST NEIGHBOR AGREEMENT")
    print("-" * 45)
    X_norm = X / X.norm(dim=1, keepdim=True)
    z_norm = z / z.norm(dim=1, keepdim=True)

    for k in [1, 5, 10, 50]:
        agreement = 0.0
        batch = 1000
        for start in range(0, N, batch):
            end = min(start + batch, N)
            bs = end - start
            idx = torch.arange(bs, device=DEVICE)
            gidx = torch.arange(start, end, device=DEVICE)

            sim_e = X_norm[start:end] @ X_norm.T
            sim_e[idx, gidx] = -1
            top_e = sim_e.topk(k, dim=1).indices

            sim_b = z_norm[start:end] @ z_norm.T
            sim_b[idx, gidx] = -1
            top_b = sim_b.topk(k, dim=1).indices

            for row in range(bs):
                se = set(top_e[row].cpu().tolist())
                sb = set(top_b[row].cpu().tolist())
                agreement += len(se & sb) / k
        agreement /= N
        print(f"  Top-{k:<3} neighbor overlap: {agreement:.3f} ({agreement*100:.1f}%)")

    # --- Specific semantic pairs ---
    print()
    print("SEMANTIC PAIR DISTANCES")
    print("-" * 60)
    pairs = [
        ("cat", "dog"), ("king", "queen"), ("man", "woman"), ("hot", "cold"),
        ("red", "blue"), ("mother", "father"), ("happy", "sad"), ("eat", "drink"),
        ("sun", "moon"), ("one", "two"),
        ("cat", "king"), ("love", "water"), ("red", "walk"),
    ]
    print(f"  {'Pair':<20} {'Embed cos':>10} {'Bottle L2':>10} {'Bottle cos':>11}")
    print(f"  {'-' * 55}")
    for w1, w2 in pairs:
        if w1 not in w2i or w2 not in w2i:
            continue
        i1, i2 = w2i[w1], w2i[w2]
        e_cos = nn.functional.cosine_similarity(X[i1:i1+1], X[i2:i2+1]).item()
        b_l2 = float(np.linalg.norm(z_np[i1] - z_np[i2]))
        b_cos = nn.functional.cosine_similarity(z[i1:i1+1], z[i2:i2+1]).item()
        print(f"  {w1+'/'+w2:<20} {e_cos:>10.3f} {b_l2:>10.3f} {b_cos:>11.3f}")


if __name__ == "__main__":
    main()
