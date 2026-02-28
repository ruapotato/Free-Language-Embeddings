#!/usr/bin/env python3
"""
Plot training dashboard for flm V2.

Usage:
    python plot_training.py              # save dashboard to logs/plots/
    python plot_training.py --live       # auto-refresh every 60s
    python plot_training.py --show       # show interactively instead of saving
"""

import os
import sys
import csv
import json
import argparse
import math
import time
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    print("matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)

# --- Config ---
METRICS_FILE = "logs/pretrain_v2_metrics.csv"
SAMPLES_FILE = "logs/pretrain_v2_samples.jsonl"
V1_METRICS_FILE = "logs/pretrain_v4_base_metrics.csv"
PLOT_DIR = "logs/plots"

TARGET_STEPS = 610_000
BATCH_SIZE = 8
SEQ_LEN = 2048
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN
TARGET_TOKENS_B = TARGET_STEPS * TOKENS_PER_STEP / 1e9  # ~10.0B

STAGE_BOUNDARIES = [
    (0,       "S1: General 80%"),
    (305_000, "S2: Linux 20%"),
    (408_700, "S3: Linux 25%"),
    (506_300, "S4: Linux 30%"),
]
STAGE_COLORS = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350"]


def load_metrics():
    """Load CSV metrics, handling mixed header formats."""
    if not os.path.exists(METRICS_FILE):
        print(f"No metrics file: {METRICS_FILE}")
        return []

    # The CSV may have two header formats from different runs.
    # Old: timestamp,step,loss,perplexity,learning_rate,tokens_per_sec,tokens_total,tokens_billions,elapsed_hours,phase
    # New: timestamp,step,loss,perplexity,learning_rate,tokens_per_sec,tokens_total,tokens_billions,elapsed_hours,val_loss,linux_bench_acc,stage
    FIELDS_12 = ["timestamp", "step", "loss", "perplexity", "learning_rate",
                 "tokens_per_sec", "tokens_total", "tokens_billions",
                 "elapsed_hours", "val_loss", "linux_bench_acc", "stage"]

    all_rows = []
    with open(METRICS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("timestamp"):
                continue
            parts = line.split(",")
            try:
                row = {
                    "step": int(parts[1]),
                    "loss": float(parts[2]),
                    "perplexity": float(parts[3]),
                    "lr": float(parts[4]),
                    "tokens_per_sec": float(parts[5]),
                    "tokens_total": int(parts[6]),
                    "tokens_billions": float(parts[7]),
                    "elapsed_hours": float(parts[8]),
                }
                # Handle 12-column format (val_loss, linux_bench_acc, stage)
                if len(parts) >= 12:
                    row["val_loss"] = float(parts[9]) if parts[9] else None
                    row["linux_bench_acc"] = float(parts[10]) if parts[10] else None
                    row["stage"] = int(parts[11]) if parts[11] else 1
                else:
                    row["val_loss"] = None
                    row["linux_bench_acc"] = None
                    row["stage"] = 1
                all_rows.append(row)
            except (ValueError, IndexError):
                continue

    # Detect restarts (step decreases) and keep only the latest run
    last_restart = 0
    for i in range(1, len(all_rows)):
        if all_rows[i]["step"] < all_rows[i - 1]["step"]:
            last_restart = i
    if last_restart > 0:
        print(f"  Detected restart at index {last_restart}, using latest run "
              f"({len(all_rows) - last_restart} entries)")
    metrics = all_rows[last_restart:]

    # Deduplicate steps (keep last entry per step for val_loss/linux_bench rows)
    seen = {}
    for m in metrics:
        s = m["step"]
        if s not in seen:
            seen[s] = m
        else:
            # Merge: keep non-None val_loss/linux_bench from either
            if m.get("val_loss") is not None:
                seen[s]["val_loss"] = m["val_loss"]
            if m.get("linux_bench_acc") is not None:
                seen[s]["linux_bench_acc"] = m["linux_bench_acc"]
    return list(seen.values())


def load_v1_metrics():
    """Load V1 (164M) pretrain metrics for comparison."""
    if not os.path.exists(V1_METRICS_FILE):
        return []
    metrics = []
    with open(V1_METRICS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("timestamp"):
                continue
            parts = line.split(",")
            try:
                metrics.append({
                    "step": int(parts[1]),
                    "loss": float(parts[2]),
                    "tokens_billions": float(parts[7]),
                })
            except (ValueError, IndexError):
                continue
    return metrics


def load_samples():
    """Load sample generations."""
    if not os.path.exists(SAMPLES_FILE):
        return []
    samples = []
    with open(SAMPLES_FILE) as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def smooth(values, window=50):
    """Exponential moving average — no edge artifacts."""
    if len(values) < 2:
        return values
    alpha = 2.0 / (window + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def get_stage_color(step):
    """Return color for the current training stage."""
    stage_idx = 0
    for i, (boundary, _) in enumerate(STAGE_BOUNDARIES):
        if step >= boundary:
            stage_idx = i
    return STAGE_COLORS[stage_idx]


def plot_dashboard(metrics, v1_metrics=None, save=True, show=False):
    """Generate 6-panel training dashboard."""
    if not metrics:
        print("No metrics to plot.")
        return

    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    ppls = [m["perplexity"] for m in metrics]
    lrs = [m["lr"] for m in metrics]
    tps_list = [m["tokens_per_sec"] for m in metrics]
    tokens_b = [m["tokens_billions"] for m in metrics]

    # Val loss points
    val_steps = [m["step"] for m in metrics if m.get("val_loss") is not None]
    val_losses = [m["val_loss"] for m in metrics if m.get("val_loss") is not None]

    # LinuxBench points
    lb_steps = [m["step"] for m in metrics if m.get("linux_bench_acc") is not None]
    lb_accs = [m["linux_bench_acc"] * 100 for m in metrics if m.get("linux_bench_acc") is not None]

    cur = metrics[-1]

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#FAFAFA")

    # --- Panel 1: Training Loss ---
    ax1 = fig.add_subplot(2, 3, 1)
    # Color points by stage
    for i, (boundary, label) in enumerate(STAGE_BOUNDARIES):
        next_boundary = STAGE_BOUNDARIES[i + 1][0] if i + 1 < len(STAGE_BOUNDARIES) else TARGET_STEPS + 1
        mask = [(s >= boundary and s < next_boundary) for s in steps]
        s_filtered = [s for s, m in zip(steps, mask) if m]
        l_filtered = [l for l, m in zip(losses, mask) if m]
        if s_filtered:
            ax1.scatter(s_filtered, l_filtered, s=1, alpha=0.15, color=STAGE_COLORS[i])
    # Smoothed line
    if len(losses) > 20:
        ax1.plot(steps, smooth(losses), color="#D32F2F", linewidth=2, label="Smoothed")
    # Val loss
    if val_losses:
        ax1.scatter(val_steps, val_losses, color="#1565C0", s=30, zorder=5,
                    marker="D", label=f"Val loss ({val_losses[-1]:.3f})")
    # Stage boundaries
    for boundary, label in STAGE_BOUNDARIES[1:]:
        if boundary <= max(steps) * 1.5:
            ax1.axvline(x=boundary, color="gray", linestyle="--", alpha=0.4)
            ax1.text(boundary, max(losses) * 0.95, label, fontsize=7,
                     rotation=90, va="top", ha="right", alpha=0.6)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training Loss (current: {cur['loss']:.3f})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    # --- Panel 2: Loss vs Tokens (log-log scaling law) ---
    ax2 = fig.add_subplot(2, 3, 2)
    tb_filtered = [t for t in tokens_b if t > 0]
    l_filtered = losses[-len(tb_filtered):]
    if len(tb_filtered) > 20:
        ax2.plot(tb_filtered, smooth(l_filtered, min(50, len(l_filtered) // 2)),
                 color="#D32F2F", linewidth=2)
    else:
        ax2.plot(tb_filtered, l_filtered, color="#D32F2F", linewidth=2)
    ax2.set_xlabel("Tokens (B)")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss vs Tokens Processed")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3, which="both")
    # Mark target
    ax2.axvline(x=TARGET_TOKENS_B, color="green", linestyle="--", alpha=0.5)
    ax2.text(TARGET_TOKENS_B, max(l_filtered) * 0.9, f"Target\n{TARGET_TOKENS_B:.0f}B",
             fontsize=8, ha="left", color="green")

    # --- Panel 3: Throughput ---
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(steps, tps_list, s=1, alpha=0.15, color="#00BCD4")
    if len(tps_list) > 20:
        ax3.plot(steps, smooth(tps_list), color="#E91E63", linewidth=2)
    avg_tps = sum(tps_list[-100:]) / len(tps_list[-100:])
    ax3.axhline(y=avg_tps, color="gray", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Tokens/sec")
    ax3.set_title(f"Throughput (recent avg: {avg_tps:,.0f} tok/s)")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    # --- Panel 4: LinuxBench + Learning Rate ---
    ax4 = fig.add_subplot(2, 3, 4)
    if lb_accs:
        ax4.plot(lb_steps, lb_accs, "o-", color="#4CAF50", linewidth=2,
                 markersize=6, label="LinuxBench")
        ax4.axhline(y=25, color="gray", linestyle=":", alpha=0.5, label="Random (25%)")
        ax4.axhline(y=42, color="orange", linestyle="--", alpha=0.5, label="SmolLM-135M (42%)")
        ax4.set_ylabel("LinuxBench Accuracy (%)")
        ax4.set_title(f"LinuxBench (latest: {lb_accs[-1]:.1f}%)")
        ax4.legend(fontsize=8)
        ax4.set_ylim(0, max(60, max(lb_accs) + 10))
    else:
        # Show LR schedule instead if no LinuxBench data yet
        ax4.plot(steps, lrs, color="#9C27B0", linewidth=1.5)
        ax4.set_ylabel("Learning Rate")
        ax4.set_title("Learning Rate Schedule")
        ax4.ticklabel_format(axis="y", style="scientific", scilimits=(-4, -4))
    ax4.set_xlabel("Step")
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    # --- Panel 5: V1 vs V2 Comparison ---
    ax5 = fig.add_subplot(2, 3, 5)
    # V2 loss vs tokens
    tb_v2 = [t for t in tokens_b if t > 0]
    loss_v2 = losses[-len(tb_v2):]
    if len(tb_v2) > 20:
        ax5.plot(tb_v2, smooth(loss_v2, min(50, len(loss_v2) // 2)),
                 color="#D32F2F", linewidth=2, label="V2 (493M)")
    elif tb_v2:
        ax5.plot(tb_v2, loss_v2, color="#D32F2F", linewidth=2, label="V2 (493M)")
    # V1 loss vs tokens
    if v1_metrics:
        tb_v1 = [m["tokens_billions"] for m in v1_metrics if m["tokens_billions"] > 0]
        loss_v1 = [m["loss"] for m in v1_metrics if m["tokens_billions"] > 0]
        if len(tb_v1) > 20:
            ax5.plot(tb_v1, smooth(loss_v1, min(50, len(loss_v1) // 2)),
                     color="#78909C", linewidth=2, alpha=0.7, label="V1 (164M)")
        elif tb_v1:
            ax5.plot(tb_v1, loss_v1, color="#78909C", linewidth=2, alpha=0.7, label="V1 (164M)")
    ax5.set_xlabel("Tokens (B)")
    ax5.set_ylabel("Loss")
    ax5.set_title("V1 vs V2 — Loss by Tokens")
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)
    ax5.set_xlim(left=0)

    # --- Panel 6: Stats Summary ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    pct = cur["tokens_billions"] / TARGET_TOKENS_B * 100
    elapsed_h = cur["elapsed_hours"]

    stats = [
        "flm V2 Pretraining — 493M params",
        "─" * 38,
        f"Step:          {cur['step']:>10,} / {TARGET_STEPS:,}",
        f"Loss:          {cur['loss']:>10.4f}",
        f"Perplexity:    {cur['perplexity']:>10.1f}",
        f"Tokens:        {cur['tokens_billions']:>7.2f}B / {TARGET_TOKENS_B:.0f}B",
        f"Progress:      {pct:>9.1f}%",
        f"Throughput:    {avg_tps:>7,.0f} tok/s",
        f"Elapsed:       {elapsed_h:>7.1f}h ({elapsed_h/24:.1f} days)",
    ]

    if avg_tps > 0:
        remaining_tokens = (TARGET_TOKENS_B - cur["tokens_billions"]) * 1e9
        remaining_h = remaining_tokens / avg_tps / 3600
        stats.append(f"ETA:           {remaining_h:>7.0f}h ({remaining_h/24:.1f} days)")

    if val_losses:
        stats.append(f"Val Loss:      {val_losses[-1]:>10.4f}")
    if lb_accs:
        stats.append(f"LinuxBench:    {lb_accs[-1]:>9.1f}%")

    stats.append("─" * 38)

    # Current stage
    stage_name = STAGE_BOUNDARIES[0][1]
    for boundary, label in STAGE_BOUNDARIES:
        if cur["step"] >= boundary:
            stage_name = label
    stats.append(f"Stage:  {stage_name}")

    # Progress bar
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    stats.append(f"[{bar}] {pct:.1f}%")

    stats_text = "\n".join(stats)
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                       edgecolor="#BDBDBD"))

    plt.suptitle("flm V2 Training Dashboard", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        path = os.path.join(PLOT_DIR, "v2_dashboard.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="flm V2 training dashboard")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--live", action="store_true", help="Auto-refresh every 60s")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval for --live")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg")

    v1_metrics = load_v1_metrics()
    if v1_metrics:
        print(f"Loaded V1 metrics: {len(v1_metrics)} entries ({v1_metrics[-1]['tokens_billions']:.1f}B tokens)")

    if args.live:
        print(f"Live mode — refreshing every {args.interval}s. Ctrl+C to stop.")
        while True:
            try:
                metrics = load_metrics()
                if metrics:
                    plot_dashboard(metrics, v1_metrics=v1_metrics, save=True, show=False)
                    cur = metrics[-1]
                    pct = cur["tokens_billions"] / TARGET_TOKENS_B * 100
                    print(f"  Step {cur['step']:,} | Loss {cur['loss']:.3f} | "
                          f"{cur['tokens_billions']:.2f}B tokens ({pct:.1f}%)")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        metrics = load_metrics()
        plot_dashboard(metrics, v1_metrics=v1_metrics, save=not args.show, show=args.show)


if __name__ == "__main__":
    main()
