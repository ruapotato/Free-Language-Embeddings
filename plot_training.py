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

    # Detect full restarts (large step decreases) and keep only the latest run
    # Small step decreases (< 1000) are just checkpoint resume jitter, not restarts
    last_restart = 0
    for i in range(1, len(all_rows)):
        if all_rows[i]["step"] < all_rows[i - 1]["step"] - 1000:
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


def project_loss(metrics, target_tokens_b=10.0, min_points=30):
    """Project final loss using power-law fit: loss = a * tokens^(-b) + c.

    This models the typical LLM scaling curve where loss drops fast early
    then squeezes out diminishing gains — the last 10% of training still
    helps but less dramatically.

    Returns (projected_loss, fit_tokens, fit_losses) or None if not enough data.
    """
    if len(metrics) < min_points:
        return None

    # Use smoothed data from second half of training (more stable fit)
    halfway = len(metrics) // 2
    tb = [m["tokens_billions"] for m in metrics[halfway:] if m["tokens_billions"] > 0.001]
    lo = smooth([m["loss"] for m in metrics[halfway:] if m["tokens_billions"] > 0.001], 20)

    if len(tb) < 10:
        return None

    # Fit log(loss - c) = log(a) - b * log(tokens)
    # Estimate c (asymptotic floor) as slightly below current best
    import math
    min_loss = min(lo)
    # Try a few floor values and pick best fit
    best_fit = None
    best_err = float("inf")

    for c_frac in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        c = min_loss * c_frac
        try:
            log_t = [math.log(t) for t in tb]
            log_l = [math.log(l - c) for l in lo if l > c]
            if len(log_l) < len(log_t):
                log_t = log_t[:len(log_l)]

            # Simple linear regression on log-log
            n = len(log_t)
            sx = sum(log_t)
            sy = sum(log_l)
            sxx = sum(x * x for x in log_t)
            sxy = sum(x * y for x, y in zip(log_t, log_l))
            denom = n * sxx - sx * sx
            if abs(denom) < 1e-12:
                continue
            b = -(n * sxy - sx * sy) / denom  # negative slope = positive b
            log_a = (sy + b * sx) / n  # since log_l = log_a - b*log_t
            a = math.exp(log_a)

            # Check fit quality
            err = sum((a * t ** (-b) + c - l) ** 2 for t, l in zip(tb, lo))
            if err < best_err and b > 0:
                best_err = err
                best_fit = (a, b, c)
        except (ValueError, ZeroDivisionError):
            continue

    if best_fit is None:
        return None

    a, b, c = best_fit

    # Generate projection curve from current position to target
    cur_tok = tb[-1]
    n_points = 50
    proj_tokens = [cur_tok + (target_tokens_b - cur_tok) * i / n_points
                   for i in range(n_points + 1)]
    proj_losses = [a * t ** (-b) + c for t in proj_tokens]

    projected_final = a * target_tokens_b ** (-b) + c

    return projected_final, proj_tokens, proj_losses


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
    projections = {}  # name -> projected_final_loss
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

        # Add projection for V3
        if name == "V3 (135M)":
            proj = project_loss(metrics, target_tokens_b=10.0)
            if proj:
                proj_loss, proj_tb, proj_lo = proj
                ax_loss_tok.plot(proj_tb, proj_lo, color=cfg["color"],
                                 linewidth=2.5, linestyle="--", alpha=0.7,
                                 label=f"V3 projected → {proj_loss:.2f}")
                ax_loss_tok.scatter([10.0], [proj_loss], color=cfg["color"],
                                    s=150, marker="*", zorder=10,
                                    edgecolors="black", linewidths=0.5)
                projections[name] = proj_loss

    ax_loss_tok.set_xlabel("Tokens (B)")
    ax_loss_tok.set_ylabel("Loss")
    ax_loss_tok.set_title("Training Loss vs Tokens Processed")
    ax_loss_tok.grid(True, alpha=0.3)
    ax_loss_tok.legend(fontsize=10)
    ax_loss_tok.set_xlim(left=0, right=11)

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

    # --- Panel 4: Stats Summary (apples-to-apples by step AND token) ---
    ax_stats.axis("off")

    v3_metrics = all_data.get("V3 (135M)", [])
    v3_cur = v3_metrics[-1] if v3_metrics else None

    def find_nearest(metrics, key, target):
        """Find entry closest to target value for given key."""
        if not metrics:
            return None
        best = None
        for m in metrics:
            if m[key] <= target * 1.05:
                best = m
        return best

    def fmt(val, fmt_str=".3f"):
        return f"{val:{fmt_str}}" if val is not None else "—"

    def get_loss(entry):
        return entry["loss"] if entry else None

    names = ["V1 (164M)", "V2 (493M)", "V3 (135M)"]
    header = f"  {'':>12s} {'V1(164M)':>10s} {'V2(493M)':>10s} {'V3(135M)':>10s}"
    sep = f"  {'─'*44}"

    lines = ["flm Training Comparison", "═" * 46]

    if v3_cur and v3_cur["tokens_billions"] > 0.001:
        # --- Compare at same TOKENS (the fair comparison) ---
        v3_tok = v3_cur["tokens_billions"]
        lines.append(f"\n  @ {v3_tok:.3f}B tokens")
        lines.append(header)
        lines.append(sep)

        at_tok = [
            find_nearest(all_data.get(n, []), "tokens_billions", v3_tok)
            for n in names
        ]
        vals = [fmt(get_loss(a)) for a in at_tok]
        lines.append(f"  {'Loss':>12s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")
        vals = [f"{a['step']:>9,d}" if a else "        —" for a in at_tok]
        lines.append(f"  {'Step':>12s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")
        vals = [fmt(a["elapsed_hours"], ".1f") + "h" if a else "    —" for a in at_tok]
        lines.append(f"  {'Time':>12s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")

    # --- V3 Projection ---
    if "V3 (135M)" in projections:
        proj = projections["V3 (135M)"]
        lines.append(f"\n  V3 projected @ 10B: loss {proj:.2f}")

    # --- Overall totals ---
    lines.append(f"\n  Overall (latest)")
    lines.append(header)
    lines.append(sep)

    row_data = {
        "Tokens": lambda m: f"{m[-1]['tokens_billions']:.2f}B",
        "Loss": lambda m: f"{m[-1]['loss']:.4f}",
        "Time": lambda m: f"{m[-1]['elapsed_hours']:.1f}h",
    }
    for label, fn in row_data.items():
        vals = []
        for n in names:
            m = all_data.get(n, [])
            vals.append(fn(m) if m else "—")
        lines.append(f"  {label:>12s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")

    stats_text = "\n".join(lines)
    ax_stats.text(0.02, 0.98, stats_text, transform=ax_stats.transAxes,
                  fontsize=8.5, verticalalignment="top", fontfamily="monospace",
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
