#!/usr/bin/env python3
"""
Worm State Tokenization Experiment
====================================
After meaningful training, analyze the worm's 302-dimensional state space:

1. Run inference on a diverse corpus, recording worm states at each token
2. K-means clustering at various vocab sizes (16, 32, 64, 128, 256, 512)
3. Visualize state transition graphs
4. Measure reconstruction error at each vocab size
5. Compare model quality with/without state quantization

Usage:
    python tokenize_worm_states.py                              # collect + cluster
    python tokenize_worm_states.py --checkpoint path/to/ckpt    # specific checkpoint
    python tokenize_worm_states.py --states-only                # just collect states
    python tokenize_worm_states.py --cluster-only               # just cluster existing
"""

import os
import sys
import json
import time
import math
import argparse
import numpy as np
import torch
from pathlib import Path

from model import HamnerConfig
from worm_sidecar import WormSidecarModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_INFERENCE_SEQUENCES = 500
SEQUENCE_LENGTH = 256
BATCH_SIZE = 8
VOCAB_SIZES = [16, 32, 64, 128, 256, 512]
STATES_OUTPUT = "logs/worm_states.npz"
CLUSTER_OUTPUT = "logs/worm_state_clusters.npz"
PLOT_DIR = "logs/plots"


def log(msg):
    print(f"[tokenize_worm] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Phase 1: Collect worm states from inference
# ---------------------------------------------------------------------------

def collect_worm_states(checkpoint_path, device="cuda"):
    """Run inference on random data and collect 302-dim worm states."""
    log(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = HamnerConfig(**ckpt["config"])
    model = WormSidecarModel(config, substeps=50).to(device)

    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")

    # Load a small amount of real data for inference
    log("Loading FineWeb-Edu for inference...")
    try:
        from datasets import load_dataset
        stream = iter(load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT",
            split="train", streaming=True,
        ))
        # Skip ahead
        for _ in range(10000):
            next(stream)
        use_real_data = True
    except Exception:
        log("Could not load FineWeb, using random tokens")
        use_real_data = False

    all_states = []
    all_tokens = []
    total_tokens = 0

    log(f"Collecting states from {NUM_INFERENCE_SEQUENCES} sequences "
        f"(seq_len={SEQUENCE_LENGTH})...")

    n_batches = NUM_INFERENCE_SEQUENCES // BATCH_SIZE

    with torch.no_grad():
        for batch_idx in range(n_batches):
            if use_real_data:
                input_ids_list = []
                for _ in range(BATCH_SIZE):
                    buf = []
                    while len(buf) < SEQUENCE_LENGTH:
                        sample = next(stream)
                        text = sample.get("text", "")
                        if len(text.strip()) < 50:
                            continue
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        buf.extend(tokens)
                    input_ids_list.append(
                        torch.tensor(buf[:SEQUENCE_LENGTH], dtype=torch.long)
                    )
                input_ids = torch.stack(input_ids_list).to(device)
            else:
                input_ids = torch.randint(
                    0, config.vocab_size, (BATCH_SIZE, SEQUENCE_LENGTH),
                    device=device
                )

            # Extract layer 5 hidden states by running layers 0-5
            x = model.model.embed_tokens(input_ids)
            causal_mask = torch.triu(
                torch.full((SEQUENCE_LENGTH, SEQUENCE_LENGTH),
                           float("-inf"), device=device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            layer5 = None
            for i, block in enumerate(model.model.layers):
                x, _ = block(x, model.model.rope_cos, model.model.rope_sin,
                             causal_mask)
                if i == model.TAP_LAYER:
                    layer5 = x.clone()
                    break

            if layer5 is None:
                continue

            # Run worm token by token, collecting states
            model.worm.reset_state(BATCH_SIZE, device)
            for t in range(SEQUENCE_LENGTH):
                token_hidden = layer5[:, t, :]
                sensory_input = model.tap_projection(token_hidden)
                model.worm.inject_sensory(sensory_input)
                model.worm.step()
                V = model.worm.get_full_state()  # (batch, 302)
                all_states.append(V.cpu().numpy())
                all_tokens.append(input_ids[:, t].cpu().numpy())

            total_tokens += BATCH_SIZE * SEQUENCE_LENGTH

            if (batch_idx + 1) % 10 == 0:
                log(f"  Batch {batch_idx+1}/{n_batches} "
                    f"({total_tokens:,} tokens collected)")

    # Stack all states
    states = np.concatenate(all_states, axis=0)  # (total_tokens, 302)
    tokens = np.concatenate(all_tokens, axis=0)   # (total_tokens,)

    log(f"Collected {states.shape[0]:,} state vectors of dim {states.shape[1]}")

    # Save
    os.makedirs(os.path.dirname(STATES_OUTPUT), exist_ok=True)
    np.savez_compressed(STATES_OUTPUT, states=states, tokens=tokens)
    log(f"Saved: {STATES_OUTPUT}")

    return states, tokens


# ---------------------------------------------------------------------------
# Phase 2: K-means clustering
# ---------------------------------------------------------------------------

def cluster_states(states, vocab_sizes=None):
    """Apply K-means clustering at multiple vocab sizes."""
    if vocab_sizes is None:
        vocab_sizes = VOCAB_SIZES

    from sklearn.cluster import MiniBatchKMeans

    results = {}
    log(f"Clustering {states.shape[0]:,} states into vocab sizes: {vocab_sizes}")

    for k in vocab_sizes:
        log(f"  K-means k={k}...")
        t0 = time.time()
        kmeans = MiniBatchKMeans(
            n_clusters=k, random_state=42, batch_size=4096,
            max_iter=300, n_init=3,
        )
        labels = kmeans.fit_predict(states)
        centers = kmeans.cluster_centers_

        # Reconstruction error
        reconstructed = centers[labels]
        mse = np.mean((states - reconstructed) ** 2)
        rmse = np.sqrt(mse)

        # Cluster statistics
        unique, counts = np.unique(labels, return_counts=True)
        entropy = -np.sum((counts / len(labels)) * np.log2(counts / len(labels) + 1e-10))
        max_entropy = np.log2(k)

        elapsed = time.time() - t0
        log(f"    RMSE={rmse:.4f} | Entropy={entropy:.2f}/{max_entropy:.2f} | "
            f"{elapsed:.1f}s")

        results[k] = {
            "labels": labels,
            "centers": centers,
            "inertia": kmeans.inertia_,
            "mse": mse,
            "rmse": rmse,
            "entropy": entropy,
            "max_entropy": max_entropy,
            "counts": counts,
        }

    # Save cluster results
    save_data = {}
    for k, r in results.items():
        save_data[f"labels_{k}"] = r["labels"]
        save_data[f"centers_{k}"] = r["centers"]
        save_data[f"counts_{k}"] = r["counts"]
    np.savez_compressed(CLUSTER_OUTPUT, **save_data)
    log(f"Saved: {CLUSTER_OUTPUT}")

    return results


# ---------------------------------------------------------------------------
# Phase 3: Visualization
# ---------------------------------------------------------------------------

def plot_tokenization_analysis(states, cluster_results, tokens=None):
    """Generate visualization of the state tokenization analysis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not available, skipping plots")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(18, 14))

    # ── 1. Reconstruction error vs vocab size ──
    ax1 = fig.add_subplot(2, 3, 1)
    ks = sorted(cluster_results.keys())
    rmses = [cluster_results[k]["rmse"] for k in ks]
    ax1.plot(ks, rmses, "o-", color="#E91E63", linewidth=2, markersize=8)
    ax1.set_xlabel("Vocab Size")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Reconstruction Error vs Vocab Size")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)
    for k, r in zip(ks, rmses):
        ax1.annotate(f"{r:.3f}", (k, r), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8)

    # ── 2. Entropy / utilization ──
    ax2 = fig.add_subplot(2, 3, 2)
    entropies = [cluster_results[k]["entropy"] for k in ks]
    max_entropies = [cluster_results[k]["max_entropy"] for k in ks]
    utilization = [e / m for e, m in zip(entropies, max_entropies)]
    ax2.bar(range(len(ks)), utilization, color="#4CAF50", alpha=0.7)
    ax2.set_xticks(range(len(ks)))
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_xlabel("Vocab Size")
    ax2.set_ylabel("Entropy / Max Entropy")
    ax2.set_title("Cluster Utilization")
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="y")

    # ── 3. 2D PCA of states colored by cluster (k=32) ──
    ax3 = fig.add_subplot(2, 3, 3)
    try:
        from sklearn.decomposition import PCA
        # Subsample for visualization
        n_show = min(5000, states.shape[0])
        idx = np.random.choice(states.shape[0], n_show, replace=False)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(states[idx])

        k_show = 32 if 32 in cluster_results else ks[0]
        labels = cluster_results[k_show]["labels"][idx]
        scatter = ax3.scatter(coords[:, 0], coords[:, 1], c=labels, s=1,
                              alpha=0.5, cmap="tab20")
        ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax3.set_title(f"PCA of Worm States (k={k_show})")
    except ImportError:
        ax3.text(0.5, 0.5, "sklearn needed for PCA", ha="center", va="center",
                 transform=ax3.transAxes)

    # ── 4. Cluster size distribution (k=64) ──
    ax4 = fig.add_subplot(2, 3, 4)
    k_dist = 64 if 64 in cluster_results else ks[-1]
    counts = cluster_results[k_dist]["counts"]
    ax4.bar(range(len(counts)), np.sort(counts)[::-1], color="#2196F3", alpha=0.7)
    ax4.set_xlabel("Cluster Rank")
    ax4.set_ylabel("Count")
    ax4.set_title(f"Cluster Size Distribution (k={k_dist})")
    ax4.grid(True, alpha=0.3, axis="y")

    # ── 5. State transition frequency (k=32) ──
    ax5 = fig.add_subplot(2, 3, 5)
    k_trans = 32 if 32 in cluster_results else ks[0]
    labels_seq = cluster_results[k_trans]["labels"]
    # Count transitions
    trans_matrix = np.zeros((k_trans, k_trans))
    for i in range(len(labels_seq) - 1):
        trans_matrix[labels_seq[i], labels_seq[i + 1]] += 1
    # Normalize rows
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = trans_matrix / row_sums

    im = ax5.imshow(trans_prob, cmap="YlOrRd", interpolation="nearest")
    ax5.set_xlabel("Next State")
    ax5.set_ylabel("Current State")
    ax5.set_title(f"State Transition Probabilities (k={k_trans})")
    plt.colorbar(im, ax=ax5, label="Probability")

    # ── 6. Stats panel ──
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    stats = "Worm State Tokenization\n"
    stats += "=" * 40 + "\n"
    stats += f"Total states:  {states.shape[0]:,}\n"
    stats += f"State dim:     {states.shape[1]}\n\n"
    stats += f"{'Vocab':>6s} {'RMSE':>8s} {'Entropy':>8s} {'Util':>6s}\n"
    stats += "-" * 32 + "\n"
    for k in ks:
        r = cluster_results[k]
        stats += (f"{k:>6d} {r['rmse']:>8.4f} "
                  f"{r['entropy']:>7.2f} {r['entropy']/r['max_entropy']:>5.1%}\n")

    # Self-information rate
    k_ref = 64 if 64 in cluster_results else ks[-1]
    labels_ref = cluster_results[k_ref]["labels"]
    unique_transitions = len(set(zip(labels_ref[:-1], labels_ref[1:])))
    possible_transitions = k_ref * k_ref
    stats += f"\nTransition coverage (k={k_ref}):\n"
    stats += f"  {unique_transitions:,} / {possible_transitions:,} "
    stats += f"({unique_transitions/possible_transitions:.1%})\n"

    ax6.text(0.02, 0.98, stats, transform=ax6.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                       edgecolor="#BDBDBD"))

    plt.suptitle("Worm State Tokenization Analysis", fontsize=16,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "worm_state_tokenization.png"),
                dpi=150, bbox_inches="tight")
    log(f"Saved: {PLOT_DIR}/worm_state_tokenization.png")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Worm state tokenization experiment")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to worm sidecar checkpoint")
    parser.add_argument("--states-only", action="store_true",
                        help="Only collect states, skip clustering")
    parser.add_argument("--cluster-only", action="store_true",
                        help="Only cluster existing states")
    parser.add_argument("--sequences", type=int, default=NUM_INFERENCE_SEQUENCES,
                        help="Number of sequences for state collection")
    parser.add_argument("--seq-len", type=int, default=SEQUENCE_LENGTH,
                        help="Sequence length for state collection")
    args = parser.parse_args()

    global NUM_INFERENCE_SEQUENCES, SEQUENCE_LENGTH
    NUM_INFERENCE_SEQUENCES = args.sequences
    SEQUENCE_LENGTH = args.seq_len

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = [
            "checkpoints/worm_sidecar/latest.pt",
            "checkpoints/worm_sidecar/best.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break
        if ckpt_path is None:
            print("No worm sidecar checkpoint found. Train first:")
            print("  python train_worm_sidecar.py --steps 10000")
            sys.exit(1)

    if args.cluster_only:
        # Load existing states
        if not os.path.exists(STATES_OUTPUT):
            print(f"No states file found at {STATES_OUTPUT}")
            sys.exit(1)
        log(f"Loading states from {STATES_OUTPUT}")
        data = np.load(STATES_OUTPUT)
        states = data["states"]
        tokens = data.get("tokens", None)
    else:
        # Collect states
        states, tokens = collect_worm_states(ckpt_path, device)

    if args.states_only:
        log("States collected. Run with --cluster-only to cluster.")
        return

    # Cluster
    cluster_results = cluster_states(states)

    # Visualize
    plot_tokenization_analysis(states, cluster_results, tokens)

    log("Done!")


if __name__ == "__main__":
    main()
