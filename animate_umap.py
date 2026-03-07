#!/usr/bin/env python3
"""
Animate concept space evolution from tracked vectors.

Loads saved vectors from logs/tracking/, fits ONE aligned UMAP across
all timesteps so points have consistent positions, then renders a smooth
tweened animation. Center-of-mass is stabilized so the view doesn't drift.
More frames allocated to transitions with more movement.

Usage:
    python animate_umap.py              # build video
    python animate_umap.py --fps 30     # set frame rate
    python animate_umap.py --watch      # rebuild when new vectors appear
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

TRACK_DIR = Path("logs/tracking")
OUTPUT_DIR = Path("logs/plots")
OUTPUT_VIDEO = OUTPUT_DIR / "concept_space_evolution.mp4"
FRAME_DIR = Path("logs/umap_frames")


def load_all_snapshots():
    files = sorted(TRACK_DIR.glob("step_*.npz"))
    if not files:
        return [], []
    snapshots = []
    for f in files:
        data = np.load(f)
        snapshots.append((int(data["step"]), data["vecs_a"], data["vecs_b"]))
    meta_path = TRACK_DIR / "metadata.json"
    metadata = []
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    return snapshots, metadata


def ease_in_out_cubic(t):
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


def build_animation(fps=30, total_duration=None):
    """
    total_duration: target video length in seconds (excluding holds).
                    If None, auto-scales based on snapshot count.
    """
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import cv2
    except ImportError as e:
        print(f"Missing: {e}")
        return False

    snapshots, metadata = load_all_snapshots()
    if len(snapshots) < 2:
        print(f"Only {len(snapshots)} snapshots — need at least 2.")
        return False

    n_pairs = len(snapshots[0][1])
    pair_types = [m.get("type", "unknown") for m in metadata] if metadata else ["unknown"] * n_pairs
    n_diag = min(6, n_pairs)
    diag_start = n_pairs - n_diag

    print(f"Loaded {len(snapshots)} snapshots, {n_pairs} tracked pairs")

    # ── Fit one global UMAP ──────────────────────────────────────────
    all_vecs = np.concatenate([np.concatenate([va, vb]) for _, va, vb in snapshots])
    print(f"Fitting UMAP on {all_vecs.shape[0]} vectors...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                        min_dist=0.2, spread=1.5)
    all_emb = reducer.fit_transform(all_vecs)

    # Split per snapshot
    idx = 0
    raw_frames = []
    for step, va, vb in snapshots:
        n = len(va)
        ea = all_emb[idx:idx + n]
        idx += n
        eb = all_emb[idx:idx + n]
        idx += n
        raw_frames.append((step, ea, eb))

    # ── Center-of-mass stabilization ─────────────────────────────────
    # Compute centroid of all points at each snapshot, then subtract
    # so the cloud stays centered at origin
    frame_data = []
    for step, ea, eb in raw_frames:
        all_pts = np.concatenate([ea, eb], axis=0)
        centroid = all_pts.mean(axis=0)
        frame_data.append((step, ea - centroid, eb - centroid))

    # ── Compute axis limits from centered data ───────────────────────
    all_centered = np.concatenate([np.concatenate([ea, eb]) for _, ea, eb in frame_data])
    pad_frac = 0.1
    ax_min = all_centered.min(axis=0)
    ax_max = all_centered.max(axis=0)
    ax_range = ax_max - ax_min
    # Make square
    max_range = max(ax_range) * (1 + 2 * pad_frac)
    center = (ax_min + ax_max) / 2
    xlim = (center[0] - max_range / 2, center[0] + max_range / 2)
    ylim = (center[1] - max_range / 2, center[1] + max_range / 2)

    # ── Dynamic frame allocation based on movement ───────────────────
    # Compute total point movement between consecutive snapshots
    movements = []
    for i in range(len(frame_data) - 1):
        _, ea1, eb1 = frame_data[i]
        _, ea2, eb2 = frame_data[i + 1]
        # RMS displacement across all points
        delta_a = np.sqrt(((ea2 - ea1) ** 2).sum(axis=1)).mean()
        delta_b = np.sqrt(((eb2 - eb1) ** 2).sum(axis=1)).mean()
        movements.append((delta_a + delta_b) / 2)

    total_movement = sum(movements) if movements else 1.0

    # Target duration per transition proportional to movement
    if total_duration is None:
        total_duration = max(6, len(frame_data) * 2)

    hold_first_sec = 1.5
    hold_last_sec = 2.5
    transition_budget = total_duration  # seconds for transitions

    interp_frames = []

    # Hold first frame
    step0, ea0, eb0 = frame_data[0]
    for _ in range(int(fps * hold_first_sec)):
        interp_frames.append((step0, ea0, eb0))

    # Transitions with dynamic frame counts
    for i in range(len(frame_data) - 1):
        step_a, ea_s, eb_s = frame_data[i]
        step_b, ea_e, eb_e = frame_data[i + 1]

        # Fraction of budget for this transition
        frac = movements[i] / total_movement if total_movement > 0 else 1.0 / len(movements)
        n_frames = max(10, int(fps * transition_budget * frac))

        for f in range(n_frames):
            t_raw = f / n_frames
            t = ease_in_out_cubic(t_raw)
            ea = ea_s + t * (ea_e - ea_s)
            eb = eb_s + t * (eb_e - eb_s)
            interp_step = int(step_a + t_raw * (step_b - step_a))
            interp_frames.append((interp_step, ea, eb))

    # Hold last
    step_last, ea_last, eb_last = frame_data[-1]
    for _ in range(int(fps * hold_last_sec)):
        interp_frames.append((step_last, ea_last, eb_last))

    total_frames = len(interp_frames)
    vid_duration = total_frames / fps
    print(f"Rendering {total_frames} frames ({vid_duration:.1f}s at {fps}fps)")
    print(f"  Movement per transition: {['%.2f' % m for m in movements]}")

    # ── Color scheme ─────────────────────────────────────────────────
    type_colors = {
        "paraphrase":          np.array([0.18, 0.80, 0.35]),
        "non_paraphrase":      np.array([0.90, 0.20, 0.20]),
        "crosslingual":        np.array([0.20, 0.55, 1.00]),
        "hard_negative":       np.array([1.00, 0.55, 0.05]),
        "concept_cluster":     np.array([0.95, 0.85, 0.20]),
        "concept_cross":       np.array([0.75, 0.20, 0.75]),
        "concept_size":        np.array([0.40, 0.90, 0.90]),
        "concept_emotion":     np.array([1.00, 0.40, 0.70]),
        "concept_analogy":     np.array([0.60, 1.00, 0.40]),
        "entailment":          np.array([0.30, 0.70, 0.50]),
        "contradiction":       np.array([0.85, 0.30, 0.30]),
        "neutral":             np.array([0.50, 0.50, 0.50]),
        "crosslingual_neg":    np.array([0.15, 0.35, 0.70]),
        "unknown":             np.array([0.53, 0.53, 0.53]),
    }
    type_labels = {
        "paraphrase": "Paraphrase (same meaning)",
        "non_paraphrase": "Non-paraphrase (different meaning)",
        "crosslingual": "Cross-lingual (EN↔FR)",
        "hard_negative": "Hard negative (same words, diff meaning)",
        "concept_cluster": "Concept cluster (cat↔dog)",
        "concept_cross": "Cross-category (cat↔car)",
        "concept_size": "Size direction (X↔big X)",
        "concept_emotion": "Emotion (happy↔sad)",
        "concept_analogy": "Analogy (big cat↔big dog)",
    }

    def compute_stats(ea, eb):
        stats = {}
        for ptype in type_colors:
            mask = [i for i, t in enumerate(pair_types) if t == ptype]
            if mask:
                idx_arr = np.array(mask)
                dists = np.sqrt(((ea[idx_arr] - eb[idx_arr]) ** 2).sum(axis=1))
                stats[ptype] = dists.mean()
        return stats

    # ── Render frames ────────────────────────────────────────────────
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    last_step_shown = frame_data[-1][0]

    for frame_i, (step, ea, eb) in enumerate(interp_frames):
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        # Lines
        for ptype, color in type_colors.items():
            mask = [i for i, t in enumerate(pair_types) if t == ptype and i < diag_start]
            if not mask:
                continue
            idx_arr = np.array(mask)
            segments = np.stack([
                np.stack([ea[idx_arr, 0], ea[idx_arr, 1]], axis=1),
                np.stack([eb[idx_arr, 0], eb[idx_arr, 1]], axis=1),
            ], axis=1)
            dists = np.sqrt(((ea[idx_arr] - eb[idx_arr]) ** 2).sum(axis=1))
            max_d = max(ax_range) * 0.4
            alphas = np.clip(0.06 + 0.3 * (dists / max_d), 0.04, 0.35)
            rgba = np.zeros((len(mask), 4))
            rgba[:, :3] = color
            rgba[:, 3] = alphas
            lc = LineCollection(segments, colors=rgba, linewidths=0.5)
            ax.add_collection(lc)

        # Points
        drawn = set()
        for ptype, color in type_colors.items():
            mask = [i for i, t in enumerate(pair_types) if t == ptype and i < diag_start]
            if not mask:
                continue
            idx_arr = np.array(mask)
            label = type_labels.get(ptype) if ptype not in drawn else None
            drawn.add(ptype)
            ax.scatter(ea[idx_arr, 0], ea[idx_arr, 1],
                       c=[color], s=12, alpha=0.65, marker="o",
                       label=label, edgecolors="none")
            ax.scatter(eb[idx_arr, 0], eb[idx_arr, 1],
                       c=[color], s=12, alpha=0.65, marker="o",
                       edgecolors="none")

        # Diagnostic stars
        for i in range(diag_start, min(diag_start + 4, n_pairs)):
            ax.scatter(ea[i, 0], ea[i, 1], c="white", s=250,
                       marker="o", alpha=0.06, zorder=8)
            ax.scatter(eb[i, 0], eb[i, 1], c="white", s=250,
                       marker="o", alpha=0.06, zorder=8)
            ax.scatter(ea[i, 0], ea[i, 1], c="white", s=90,
                       marker="*", zorder=10, edgecolors="#222", linewidths=0.3)
            ax.scatter(eb[i, 0], eb[i, 1], c="white", s=90,
                       marker="*", zorder=10, edgecolors="#222", linewidths=0.3)
            dist = np.sqrt((ea[i, 0] - eb[i, 0])**2 + (ea[i, 1] - eb[i, 1])**2)
            ax.plot([ea[i, 0], eb[i, 0]], [ea[i, 1], eb[i, 1]],
                    color="white", linewidth=1.5,
                    alpha=min(0.9, 0.2 + dist * 0.1), zorder=9)
            if metadata and i < len(metadata):
                short = metadata[i]["text_a"][:38]
                mid_x = (ea[i, 0] + eb[i, 0]) / 2
                mid_y = (ea[i, 1] + eb[i, 1]) / 2
                ax.annotate(short, (mid_x, mid_y), fontsize=5.5,
                            color="#bbbbbb", alpha=0.8,
                            bbox=dict(boxstyle="round,pad=0.15",
                                      facecolor="#0d1117", alpha=0.7,
                                      edgecolor="none"))

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("UMAP 1", color="#555", fontsize=9)
        ax.set_ylabel("UMAP 2", color="#555", fontsize=9)
        ax.tick_params(colors="#444", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#2a2a2a")
        ax.grid(True, alpha=0.05, color="white")

        leg = ax.legend(fontsize=8.5, markerscale=1.8, loc="upper right",
                        facecolor="#161b22", edgecolor="#333",
                        labelcolor="white", framealpha=0.9)

        pct = step / max(last_step_shown, 1) * 100
        ax.set_title(
            f"flm V4 — Concept Space Evolution\n"
            f"Step {step:,}  ({pct:.0f}%)",
            fontsize=14, color="white", pad=12, fontweight="bold")

        # Stats
        stats = compute_stats(ea, eb)
        lines = ["Avg pair distance:"]
        stat_order = [
            "paraphrase", "crosslingual", "hard_negative", "non_paraphrase",
            "concept_cluster", "concept_cross", "concept_size", "concept_analogy",
        ]
        for pt in stat_order:
            if pt in stats:
                name = pt.replace("_", " ").title()[:16]
                d = stats[pt]
                bar_len = int(min(20, d * 3))
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"{name:<16s} {bar} {d:.2f}")
        ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes,
                fontsize=7, color="#999", fontfamily="monospace",
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d1117",
                          alpha=0.85, edgecolor="#2a2a2a"))

        plt.tight_layout()
        path = FRAME_DIR / f"frame_{frame_i:05d}.png"
        plt.savefig(path, dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        frame_paths.append(path)

        if (frame_i + 1) % 50 == 0 or frame_i == total_frames - 1:
            print(f"  {frame_i + 1}/{total_frames}")

    # ── Stitch ───────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (w, h))
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()

    print(f"\nVideo: {OUTPUT_VIDEO} ({vid_duration:.1f}s, {fps}fps)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Animate concept space")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=None,
                        help="Target transition duration in seconds (auto if omitted)")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()

    if args.watch:
        print(f"Watching every {args.interval}s...")
        last_count = 0
        while True:
            try:
                count = len(list(TRACK_DIR.glob("step_*.npz")))
                if count > last_count and count >= 2:
                    print(f"\n{count} snapshots, rebuilding...")
                    build_animation(fps=args.fps, total_duration=args.duration)
                    last_count = count
                time.sleep(args.interval)
            except KeyboardInterrupt:
                break
    else:
        build_animation(fps=args.fps, total_duration=args.duration)


if __name__ == "__main__":
    main()
