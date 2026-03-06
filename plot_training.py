#!/usr/bin/env python3
"""
Plot training dashboard for flm — overlays all 3 training attempts.

Usage:
    python plot_training.py              # save dashboard to logs/plots/
    python plot_training.py --live       # auto-refresh every 60s
    python plot_training.py --show       # show interactively instead of saving
"""

import os
import sys
import argparse
import time

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
PLOT_DIR = "logs/plots"

# All three training runs
RUNS = {
    "V1 (164M)": {
        "file": "logs/pretrain_v1_metrics.csv",
        "color": "#78909C",
        "params": "164M",
        "batch": 8,
        "seq": 2048,
    },
    "V2 (493M)": {
        "file": "logs/pretrain_v2_metrics.csv",
        "color": "#2196F3",
        "params": "493M",
        "batch": 8,
        "seq": 2048,
    },
    "V3 (135M)": {
        "file": "logs/pretrain_v3_metrics.csv",
        "color": "#D32F2F",
        "params": "135M",
        "batch": 16,
        "seq": 2048,
    },
}


def load_metrics(filepath):
    """Load CSV metrics, handling mixed header formats."""
    if not os.path.exists(filepath):
        return []

    all_rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("timestamp"):
                continue
            parts = line.split(",")
            try:
                row = {
                    "step": int(parts[1]),
                    "loss": float(parts[2]),
                    "tokens_billions": float(parts[7]),
                    "elapsed_hours": float(parts[8]),
                    "tokens_per_sec": float(parts[5]),
                    "lr": float(parts[4]),
                }
                # Val loss (column 9) if present
                if len(parts) >= 10 and parts[9]:
                    try:
                        row["val_loss"] = float(parts[9])
                    except ValueError:
                        row["val_loss"] = None
                else:
                    row["val_loss"] = None
                # LinuxBench (column 10) if present
                if len(parts) >= 11 and parts[10]:
                    try:
                        row["linux_bench_acc"] = float(parts[10])
                    except ValueError:
                        row["linux_bench_acc"] = None
                else:
                    row["linux_bench_acc"] = None
                all_rows.append(row)
            except (ValueError, IndexError):
                continue

    # Detect restarts (step decreases) and keep only the latest run
    last_restart = 0
    for i in range(1, len(all_rows)):
        if all_rows[i]["step"] < all_rows[i - 1]["step"]:
            last_restart = i
    metrics = all_rows[last_restart:]

    # Deduplicate steps (keep last entry per step, merge val/bench data)
    seen = {}
    for m in metrics:
        s = m["step"]
        if s not in seen:
            seen[s] = m
        else:
            if m.get("val_loss") is not None:
                seen[s]["val_loss"] = m["val_loss"]
            if m.get("linux_bench_acc") is not None:
                seen[s]["linux_bench_acc"] = m["linux_bench_acc"]
    return list(seen.values())


def smooth(values, window=50):
    """Exponential moving average."""
    if len(values) < 2:
        return values
    alpha = 2.0 / (window + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def plot_dashboard(all_data, save=True, show=False):
    """Generate 4-panel dashboard overlaying all training runs."""
    if not any(all_data.values()):
        print("No metrics to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#FAFAFA")

    ax_loss_step = axes[0, 0]
    ax_loss_tok = axes[0, 1]
    ax_throughput = axes[1, 0]
    ax_stats = axes[1, 1]

    # --- Panel 1: Loss vs Step ---
    for name, metrics in all_data.items():
        if not metrics:
            continue
        cfg = RUNS[name]
        steps = [m["step"] for m in metrics]
        losses = [m["loss"] for m in metrics]
        # Raw points
        ax_loss_step.scatter(steps, losses, s=1, alpha=0.08, color=cfg["color"])
        # Smoothed line
        if len(losses) > 20:
            ax_loss_step.plot(steps, smooth(losses), color=cfg["color"],
                              linewidth=2, label=name)
        else:
            ax_loss_step.plot(steps, losses, color=cfg["color"],
                              linewidth=2, label=name)

    ax_loss_step.set_xlabel("Step")
    ax_loss_step.set_ylabel("Loss")
    ax_loss_step.set_title("Training Loss vs Step")
    ax_loss_step.grid(True, alpha=0.3)
    ax_loss_step.legend(fontsize=10)
    ax_loss_step.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    # --- Panel 2: Loss vs Tokens (the real comparison) ---
    for name, metrics in all_data.items():
        if not metrics:
            continue
        cfg = RUNS[name]
        tb = [m["tokens_billions"] for m in metrics if m["tokens_billions"] > 0]
        lo = [m["loss"] for m in metrics if m["tokens_billions"] > 0]
        if not tb:
            continue
        # Smoothed line only
        if len(lo) > 20:
            ax_loss_tok.plot(tb, smooth(lo, min(50, len(lo) // 2)),
                             color=cfg["color"], linewidth=2.5, label=name)
        else:
            ax_loss_tok.plot(tb, lo, color=cfg["color"], linewidth=2.5, label=name)

    ax_loss_tok.set_xlabel("Tokens (B)")
    ax_loss_tok.set_ylabel("Loss")
    ax_loss_tok.set_title("Training Loss vs Tokens Processed")
    ax_loss_tok.grid(True, alpha=0.3)
    ax_loss_tok.legend(fontsize=10)
    ax_loss_tok.set_xlim(left=0)

    # --- Panel 3: Throughput ---
    for name, metrics in all_data.items():
        if not metrics:
            continue
        cfg = RUNS[name]
        steps = [m["step"] for m in metrics]
        tps = [m["tokens_per_sec"] for m in metrics]
        if len(tps) > 20:
            ax_throughput.plot(steps, smooth(tps), color=cfg["color"],
                               linewidth=2, label=name)
        else:
            ax_throughput.plot(steps, tps, color=cfg["color"],
                               linewidth=2, label=name)

    ax_throughput.set_xlabel("Step")
    ax_throughput.set_ylabel("Tokens/sec")
    ax_throughput.set_title("Throughput")
    ax_throughput.grid(True, alpha=0.3)
    ax_throughput.legend(fontsize=10)
    ax_throughput.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    # --- Panel 4: Stats Summary ---
    ax_stats.axis("off")

    lines = ["flm Training Comparison", "─" * 45]

    for name, metrics in all_data.items():
        if not metrics:
            lines.append(f"\n{name}: no data")
            continue
        cfg = RUNS[name]
        cur = metrics[-1]
        avg_tps = sum(m["tokens_per_sec"] for m in metrics[-100:]) / len(metrics[-100:])

        lines.append(f"\n{name} ({cfg['params']} params)")
        lines.append(f"  Steps:      {cur['step']:>10,}")
        lines.append(f"  Loss:       {cur['loss']:>10.4f}")
        lines.append(f"  Tokens:     {cur['tokens_billions']:>7.2f}B")
        lines.append(f"  Throughput: {avg_tps:>7,.0f} tok/s")
        lines.append(f"  Time:       {cur['elapsed_hours']:>7.1f}h")

        # Val loss
        val = [m for m in metrics if m.get("val_loss") is not None]
        if val:
            lines.append(f"  Val Loss:   {val[-1]['val_loss']:>10.4f}")

        # LinuxBench
        lb = [m for m in metrics if m.get("linux_bench_acc") is not None]
        if lb:
            lines.append(f"  LinuxBench: {lb[-1]['linux_bench_acc']*100:>9.1f}%")

    stats_text = "\n".join(lines)
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, verticalalignment="top", fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                            edgecolor="#BDBDBD"))

    plt.suptitle("flm Training Dashboard — V1 vs V2 vs V3",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        path = os.path.join(PLOT_DIR, "training_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="flm training dashboard")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--live", action="store_true", help="Auto-refresh every 60s")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg")

    def load_all():
        data = {}
        for name, cfg in RUNS.items():
            metrics = load_metrics(cfg["file"])
            data[name] = metrics
            if metrics:
                print(f"  {name}: {len(metrics)} entries, "
                      f"{metrics[-1]['tokens_billions']:.2f}B tokens, "
                      f"loss {metrics[-1]['loss']:.3f}")
            else:
                print(f"  {name}: no data")
        return data

    if args.live:
        print(f"Live mode — refreshing every {args.interval}s. Ctrl+C to stop.")
        while True:
            try:
                data = load_all()
                plot_dashboard(data, save=True, show=False)
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        data = load_all()
        plot_dashboard(data, save=not args.show, show=args.show)


if __name__ == "__main__":
    main()
