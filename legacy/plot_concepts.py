#!/usr/bin/env python3
"""
Visualize concept space geometry with UMAP.

Usage:
    python plot_concepts.py                          # latest checkpoint, static
    python plot_concepts.py --checkpoint step_020000.pt
    python plot_concepts.py --animate                # video across all checkpoints
    python plot_concepts.py --animate --fps 24

Animation caches encoded vectors in logs/concept_cache/. On re-run,
only new checkpoints are encoded — UMAP + video are rebuilt from cache.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT_DIR = Path("checkpoints/concept_v5")
OUTPUT_DIR = Path("logs/plots")
CACHE_DIR = Path("logs/concept_cache")
FRAME_DIR = Path("logs/umap_frames")

# ── Sentence data ──────────────────────────────────────────────────────

GROUPS = {
    "Animals": {
        "color": "#2ecc71",
        "texts": [
            "the cat sat on the mat",
            "a cat is sitting on the rug",
            "the dog ran across the yard",
            "a bird flew over the house",
            "the fish swam in the pond",
            "the horse galloped through the field",
            "a rabbit hid under the bush",
            "the snake slithered through the grass",
            "my dog loves to play fetch",
            "the parrot can talk",
        ],
    },
    "Emotions": {
        "color": "#e74c3c",
        "texts": [
            "I am very happy today",
            "she feels joyful and excited",
            "he is sad and lonely",
            "they were angry about it",
            "I feel scared and anxious",
            "she was filled with pride",
            "he felt deeply ashamed",
            "I am so grateful for this",
            "they were bored out of their minds",
            "she felt a wave of jealousy",
        ],
    },
    "Weather": {
        "color": "#3498db",
        "texts": [
            "it is raining outside",
            "the rain is pouring down",
            "the sun is shining brightly",
            "it is a cold winter day",
            "the snow is falling gently",
            "there is a thunderstorm coming",
            "the wind is blowing hard",
            "it is a hot summer afternoon",
            "the fog rolled in this morning",
            "a rainbow appeared after the storm",
        ],
    },
    "Food": {
        "color": "#f39c12",
        "texts": [
            "I ate breakfast this morning",
            "she cooked dinner for us",
            "the pizza was delicious",
            "he drank a cup of coffee",
            "they had lunch at noon",
            "the soup was too salty",
            "she baked a chocolate cake",
            "I ordered a sandwich and fries",
            "the steak was perfectly cooked",
            "he made fresh orange juice",
        ],
    },
    "Work": {
        "color": "#9b59b6",
        "texts": [
            "I have a meeting today",
            "she finished her report",
            "he is working on the project",
            "they discussed the budget",
            "the deadline is tomorrow",
            "she got promoted last week",
            "he sent the email to his boss",
            "the team is behind schedule",
            "I need to update the spreadsheet",
            "they hired three new employees",
        ],
    },
    "Sports": {
        "color": "#1abc9c",
        "texts": [
            "he kicked the ball into the goal",
            "she won the tennis match",
            "the team lost the championship",
            "he ran a marathon last weekend",
            "she swam fifty laps in the pool",
            "the basketball game was exciting",
            "he hit a home run",
            "the referee blew the whistle",
            "she scored the winning point",
            "they practiced every morning",
        ],
    },
    "Travel": {
        "color": "#e67e22",
        "texts": [
            "we flew to paris last summer",
            "the train arrived on time",
            "she booked a hotel near the beach",
            "he drove across the country",
            "the flight was delayed by two hours",
            "we took a boat to the island",
            "the airport was very crowded",
            "she packed her suitcase the night before",
            "they visited three countries in one trip",
            "I need to renew my passport",
        ],
    },
    "Health": {
        "color": "#e91e63",
        "texts": [
            "she went to the doctor yesterday",
            "he broke his arm playing football",
            "I have a terrible headache",
            "the medicine made me feel better",
            "she runs every morning to stay fit",
            "he needs to get more sleep",
            "the hospital was very busy",
            "I caught a cold last week",
            "she is allergic to peanuts",
            "he went for his annual checkup",
        ],
    },
    "Technology": {
        "color": "#00bcd4",
        "texts": [
            "the computer crashed again",
            "she wrote a python program",
            "he bought a new smartphone",
            "the internet is down",
            "they launched a new website",
            "the robot can walk on two legs",
            "she updated the software",
            "the printer is out of ink",
            "he built a machine learning model",
            "the server needs more memory",
        ],
    },
    "Nature": {
        "color": "#8bc34a",
        "texts": [
            "the mountain was covered in snow",
            "a river flows through the valley",
            "the forest is full of tall trees",
            "the ocean waves crashed on the shore",
            "the desert is hot and dry",
            "flowers bloomed in the garden",
            "the volcano erupted last night",
            "a waterfall cascaded down the cliff",
            "the lake was perfectly still",
            "the stars were bright in the sky",
        ],
    },
    "Education": {
        "color": "#ff9800",
        "texts": [
            "she studied for the exam all night",
            "the teacher explained the lesson",
            "he graduated from college last year",
            "they read the textbook chapter",
            "the students took a quiz",
            "she wrote an essay about history",
            "he learned to speak french",
            "the library was quiet",
            "they had a science experiment",
            "the professor gave a long lecture",
        ],
    },
    "Family": {
        "color": "#ff5722",
        "texts": [
            "my mother cooked a big meal",
            "his father taught him to ride a bike",
            "the baby started walking today",
            "she visited her grandparents",
            "the kids played in the backyard",
            "my brother is older than me",
            "she hugged her daughter",
            "the family gathered for the holiday",
            "he called his sister on the phone",
            "they adopted a puppy for the children",
        ],
    },
    "Music": {
        "color": "#673ab7",
        "texts": [
            "she played the piano beautifully",
            "he sang a song at the concert",
            "the band performed on stage",
            "I listened to music all day",
            "she learned to play the guitar",
            "the drums were very loud",
            "he composed a new symphony",
            "the crowd cheered after the song",
            "she has a beautiful singing voice",
            "they danced to the rhythm",
        ],
    },
}

SPECIAL_PAIRS = {
    "Word Order": {
        "color": "#ff1744",
        "marker": "D",
        "pairs": [
            ("the dog bit the man", "the man bit the dog"),
            ("alice likes bob", "bob likes alice"),
            ("she gave him a book", "he gave her a book"),
            ("the cat chased the mouse", "the mouse chased the cat"),
            ("the teacher praised the student", "the student praised the teacher"),
        ],
    },
    "Paraphrase": {
        "color": "#00e676",
        "marker": "s",
        "pairs": [
            ("the king died", "the monarch passed away"),
            ("the massive cat stepped on the rug",
             "there was a rug that a massive cat stepped on"),
            ("he is very smart", "he is quite intelligent"),
            ("she began to cry", "she started crying"),
            ("the car is fast", "the automobile is quick"),
        ],
    },
    "Unrelated": {
        "color": "#78909c",
        "marker": "^",
        "pairs": [
            ("the cat sat on the mat", "the stock market crashed today"),
            ("I am happy", "the car needs gas"),
            ("she played the piano", "the volcano erupted"),
            ("he ate a sandwich", "the equation has no solution"),
        ],
    },
}


def get_all_texts():
    """Return (cluster_texts, text_to_group, pair_texts) with stable ordering."""
    cluster_texts = []
    text_to_group = {}
    for gname, gdata in GROUPS.items():
        for t in gdata["texts"]:
            cluster_texts.append(t)
            text_to_group[t] = gname

    pair_texts = []
    seen = set()
    for pdata in SPECIAL_PAIRS.values():
        for a, b in pdata["pairs"]:
            if a not in seen:
                pair_texts.append(a)
                seen.add(a)
            if b not in seen:
                pair_texts.append(b)
                seen.add(b)

    return cluster_texts, text_to_group, pair_texts


# ── Model helpers ──────────────────────────────────────────────────────

def load_model(checkpoint_path):
    from concept_model import ConceptAutoencoder, ConceptConfig
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    step = ckpt.get("step", 0)
    return model, step


def encode_texts(model, tokenizer, texts):
    enc = tokenizer(texts, max_length=128, padding=True,
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        concepts = model.encode(enc["input_ids"], enc["attention_mask"])
    return concepts.view(concepts.shape[0], -1).numpy()


# ── Rendering ──────────────────────────────────────────────────────────

def render_frame(ax, cluster_emb, pair_emb, pair_vecs, cluster_texts,
                 text_to_group, pair_texts, step, show_labels=True):
    """Draw one frame of the concept space onto ax."""
    pair_text_to_idx = {t: i for i, t in enumerate(pair_texts)}

    for gname, gdata in GROUPS.items():
        mask = [i for i, t in enumerate(cluster_texts) if text_to_group[t] == gname]
        pts = cluster_emb[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=gdata["color"], s=60, alpha=0.8,
                   label=gname, edgecolors="white", linewidths=0.3, zorder=5)
        if show_labels:
            for j, idx in enumerate(mask):
                txt = cluster_texts[idx]
                if len(txt) > 30:
                    txt = txt[:28] + ".."
                ax.annotate(txt, (pts[j, 0], pts[j, 1]),
                            fontsize=4.5, color="#aaaaaa", alpha=0.6,
                            xytext=(4, 4), textcoords="offset points")

    for pname, pdata in SPECIAL_PAIRS.items():
        first = True
        for a, b in pdata["pairs"]:
            ia, ib = pair_text_to_idx[a], pair_text_to_idx[b]
            pa, pb = pair_emb[ia], pair_emb[ib]
            va, vb = pair_vecs[ia], pair_vecs[ib]
            cos_sim = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)

            ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                    color=pdata["color"], linewidth=1.5, alpha=0.6, zorder=4)
            label = f"{pname} (cos)" if first else None
            ax.scatter([pa[0], pb[0]], [pa[1], pb[1]],
                       c=pdata["color"], s=80, alpha=0.9,
                       marker=pdata["marker"], edgecolors="white",
                       linewidths=0.6, label=label, zorder=6)

            mid = (pa + pb) / 2
            ax.annotate(f"{cos_sim:.2f}", (mid[0], mid[1]),
                        fontsize=6, color=pdata["color"], fontweight="bold",
                        ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.1",
                                  facecolor="#0d1117", alpha=0.8,
                                  edgecolor="none"))
            if show_labels:
                for pt, txt in [(pa, a), (pb, b)]:
                    if len(txt) > 30:
                        txt = txt[:28] + ".."
                    ax.annotate(txt, (pt[0], pt[1]),
                                fontsize=4.5, color="#aaaaaa", alpha=0.6,
                                xytext=(4, -7), textcoords="offset points")
            first = False

    ax.set_xlabel("UMAP 1", color="#555", fontsize=9)
    ax.set_ylabel("UMAP 2", color="#555", fontsize=9)
    ax.tick_params(colors="#444", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#2a2a2a")
    ax.grid(True, alpha=0.05, color="white")
    ax.set_title(f"Concept Autoencoder — Concept Space (Step {step:,})",
                 fontsize=14, color="white", pad=12, fontweight="bold")
    ax.legend(fontsize=7, markerscale=1.0, loc="upper right",
              facecolor="#161b22", edgecolor="#333",
              labelcolor="white", framealpha=0.9, ncol=2)


# ── Static plot ────────────────────────────────────────────────────────

def plot_static(ckpt_path, output_path):
    from transformers import AutoTokenizer
    import umap

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"Loading {ckpt_path}...")
    model, step = load_model(ckpt_path)

    cluster_texts, text_to_group, pair_texts = get_all_texts()
    all_input = cluster_texts + pair_texts

    print(f"Encoding {len(all_input)} sentences...")
    vecs = encode_texts(model, tokenizer, all_input)
    n_cluster = len(cluster_texts)

    print(f"Fitting UMAP on {vecs.shape[0]} vectors...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                        min_dist=0.25, metric="cosine")
    emb = reducer.fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    render_frame(ax, emb[:n_cluster], emb[n_cluster:], vecs[n_cluster:],
                 cluster_texts, text_to_group, pair_texts, step)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")


# ── Animation with caching ─────────────────────────────────────────────

def ease_in_out(t):
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


def encode_with_cache(tokenizer):
    """Encode all texts at each checkpoint. Cache results as .npz files.
    Only encodes checkpoints that don't have a cached file yet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cluster_texts, _, pair_texts = get_all_texts()
    all_input = cluster_texts + pair_texts

    ckpt_files = sorted(CKPT_DIR.glob("step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))

    snapshots = []
    for ckpt_path in ckpt_files:
        step_num = int(ckpt_path.stem.split("_")[1])
        cache_file = CACHE_DIR / f"vecs_{step_num:06d}.npz"

        if cache_file.exists():
            data = np.load(cache_file)
            vecs = data["vecs"]
            print(f"  {ckpt_path.name} — cached ({cache_file.name})")
        else:
            print(f"  {ckpt_path.name} — encoding {len(all_input)} sentences...")
            model, _ = load_model(ckpt_path)
            vecs = encode_texts(model, tokenizer, all_input)
            np.savez_compressed(cache_file, vecs=vecs, step=step_num)
            del model
            gc.collect()

        snapshots.append((step_num, vecs))

    return snapshots


def build_animation(fps=24, output_path=None):
    from transformers import AutoTokenizer
    import umap
    import cv2

    if output_path is None:
        output_path = str(OUTPUT_DIR / "concept_evolution.mp4")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    cluster_texts, text_to_group, pair_texts = get_all_texts()
    n_cluster = len(cluster_texts)
    n_total = n_cluster + len(pair_texts)

    # Encode (with cache)
    print("Encoding checkpoints...")
    snapshots = encode_with_cache(tokenizer)
    if len(snapshots) < 2:
        print(f"Only {len(snapshots)} checkpoints — need at least 2.")
        return

    # Global UMAP (always refit — fast enough and needed if new points added)
    all_vecs = np.concatenate([v for _, v in snapshots], axis=0)
    print(f"Fitting global UMAP on {all_vecs.shape[0]} vectors...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                        min_dist=0.25, metric="cosine")
    all_emb = reducer.fit_transform(all_vecs)

    # Split back per snapshot and center each
    frame_data = []
    idx = 0
    for step, vecs in snapshots:
        emb = all_emb[idx:idx + n_total]
        idx += n_total
        centroid = emb.mean(axis=0)
        frame_data.append((step, vecs, emb - centroid))

    # Stable axis limits
    all_centered = np.concatenate([e for _, _, e in frame_data])
    pad = 0.1
    mn, mx = all_centered.min(axis=0), all_centered.max(axis=0)
    rng = mx - mn
    max_rng = max(rng) * (1 + 2 * pad)
    ctr = (mn + mx) / 2
    xlim = (ctr[0] - max_rng / 2, ctr[0] + max_rng / 2)
    ylim = (ctr[1] - max_rng / 2, ctr[1] + max_rng / 2)

    # Build interpolated frames
    frames_per_transition = max(30, fps * 2)
    hold_frames = fps

    interp = []
    for ki in range(len(frame_data)):
        step_k, vecs_k, emb_k = frame_data[ki]
        for _ in range(hold_frames):
            interp.append((step_k, vecs_k, emb_k))
        if ki < len(frame_data) - 1:
            step_next, vecs_next, emb_next = frame_data[ki + 1]
            for f in range(frames_per_transition):
                t = ease_in_out(f / frames_per_transition)
                step_i = int(step_k + t * (step_next - step_k))
                vecs_i = vecs_k + t * (vecs_next - vecs_k)
                emb_i = emb_k + t * (emb_next - emb_k)
                interp.append((step_i, vecs_i, emb_i))
    # Extra hold on last
    for _ in range(hold_frames * 2):
        step_l, vecs_l, emb_l = frame_data[-1]
        interp.append((step_l, vecs_l, emb_l))

    total = len(interp)
    duration = total / fps
    print(f"Rendering {total} frames ({duration:.1f}s at {fps}fps)...")

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    for fi, (step_i, vecs_i, emb_i) in enumerate(interp):
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        render_frame(ax, emb_i[:n_cluster], emb_i[n_cluster:], vecs_i[n_cluster:],
                     cluster_texts, text_to_group, pair_texts,
                     step_i, show_labels=False)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        plt.tight_layout()
        path = FRAME_DIR / f"frame_{fi:05d}.png"
        plt.savefig(path, dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        frame_paths.append(path)

        if (fi + 1) % 50 == 0 or fi == total - 1:
            print(f"  {fi + 1}/{total}")

    # Stitch video
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()

    print(f"Video: {output_path} ({duration:.1f}s, {fps}fps)")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize concept space")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--animate", action="store_true",
                        help="Build video across all checkpoints")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    if args.animate:
        out = args.output or str(OUTPUT_DIR / "concept_evolution.mp4")
        build_animation(fps=args.fps, output_path=out)
    else:
        if args.checkpoint:
            ckpt_path = CKPT_DIR / args.checkpoint
        else:
            ckpt_path = CKPT_DIR / "latest.pt"
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        out = args.output or str(OUTPUT_DIR / "concept_clusters.png")
        plot_static(ckpt_path, out)


if __name__ == "__main__":
    main()
