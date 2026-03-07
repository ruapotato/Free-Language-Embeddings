#!/usr/bin/env python3
"""
Plot training dashboard for flm.

Mode 1 (default): Concept autoencoder training dashboard (latest run)
Mode 2 (--compare): Side-by-side V1 vs V2 comparison
Mode 3 (--legacy): V1/V2/V3 pretrain comparison

Usage:
    python plot_training.py              # Latest concept autoencoder dashboard
    python plot_training.py --compare    # V1 vs V2 side-by-side
    python plot_training.py --live       # auto-refresh every 60s
    python plot_training.py --legacy     # V1/V2/V3 pretrain comparison
"""

import os
import re
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

PLOT_DIR = "logs/plots"


# =========================================================================
# Concept Autoencoder Dashboard
# =========================================================================

CONCEPT_RUNS = {
    "v1": {"log": "logs/concept_v1.log", "metrics": "logs/concept_v1_metrics.csv"},
    "v2": {"log": "logs/concept_v2.log", "metrics": "logs/concept_v2_metrics.csv"},
    "v3": {"log": "logs/concept_v3.log", "metrics": "logs/concept_v3_metrics.csv"},
}


def load_concept_step_data(log_path):
    """Parse per-step data from the training log."""
    if not os.path.exists(log_path):
        return []
    rows = []
    with open(log_path) as f:
        for line in f:
            if "step" in line and "loss" in line and "recon=" in line:
                try:
                    parts = line.split("|")
                    step = int(parts[0].split("step")[1].strip())
                    total_loss = float(parts[1].split("loss")[1].split("(")[0].strip())
                    detail = parts[1].split("(")[1].split(")")[0]
                    recon = float(re.search(r"recon=([\d.]+)", detail).group(1))
                    # V2 format: para=/neg=/wo=  V3 format: nce=/wo=/decorr=
                    para_m = re.search(r"para=([\d.]+)", detail)
                    nce_m = re.search(r"nce=([\d.]+)", detail)
                    para = float(nce_m.group(1)) if nce_m else (float(para_m.group(1)) if para_m else 0.0)
                    neg_m = re.search(r"neg=([\d.]+)", detail)
                    neg = float(neg_m.group(1)) if neg_m else 0.0
                    wo_m = re.search(r"wo=([\d.]+)", detail)
                    wo = float(wo_m.group(1)) if wo_m else 0.0
                    decorr_m = re.search(r"decorr=([\d.]+)", detail)
                    decorr = float(decorr_m.group(1)) if decorr_m else 0.0
                    sim_part = parts[2]
                    p_sim = float(re.search(r"p_sim=([\d.]+)", sim_part).group(1))
                    n_sim = float(re.search(r"n_sim=([\d.]+)", sim_part).group(1))
                    wo_sim_m = re.search(r"wo_sim=([\d.]+)", sim_part)
                    wo_sim = float(wo_sim_m.group(1)) if wo_sim_m else 0.0
                    rows.append({
                        "step": step, "total_loss": total_loss,
                        "recon_loss": recon, "para_loss": para, "neg_loss": neg,
                        "wo_loss": wo, "decorr_loss": decorr,
                        "p_sim": p_sim, "n_sim": n_sim, "wo_sim": wo_sim,
                    })
                except (ValueError, IndexError, AttributeError):
                    continue
    return rows


def load_concept_eval_data(log_path):
    """Parse EVAL diagnostic data from the training log."""
    if not os.path.exists(log_path):
        return []
    rows = []
    last_step = 0
    current_eval = None
    with open(log_path) as f:
        for line in f:
            if "step" in line and "loss" in line and "recon=" in line:
                try:
                    last_step = int(line.split("|")[0].split("step")[1].strip())
                except (ValueError, IndexError):
                    pass
            if "EVAL:" in line:
                current_eval = {"step": last_step}
                try:
                    for m in re.finditer(r"(\w+_sim)=([-\d.]+)", line):
                        current_eval[m.group(1)] = float(m.group(2))
                    for m in re.finditer(r"(rank\d+)=(\d+)", line):
                        current_eval[m.group(1)] = int(m.group(2))
                except (ValueError, IndexError):
                    pass
            if current_eval and re.search(r"[+-]\d\.\d{4}\s+\[", line):
                try:
                    m = re.search(r"([+-]?\d\.\d+)\s+\[(\w+)\s*\]", line)
                    if m:
                        sim = float(m.group(1))
                        ptype = m.group(2)
                        key = f"pair_{ptype}_{len([k for k in current_eval if k.startswith(f'pair_{ptype}')])}"
                        current_eval[key] = sim
                except (ValueError, IndexError):
                    pass
            if current_eval and ("--- RECON" in line or
                                 ("step" in line and "loss" in line and "recon=" in line
                                  and current_eval["step"] != last_step)):
                rows.append(current_eval)
                current_eval = None
    if current_eval:
        rows.append(current_eval)
    return rows


def load_concept_csv(csv_path):
    """Load metrics CSV for elapsed time. Handles both V2 and V3 formats."""
    if not os.path.exists(csv_path):
        return []
    rows = []
    has_eff_rank = False
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("timestamp"):
                has_eff_rank = "eff_rank" in line
                continue
            parts = line.split(",")
            try:
                if has_eff_rank:
                    # V3: timestamp,step,recon_loss,pos_sim,neg_sim,wo_sim,eff_rank,lr,elapsed_hours
                    rows.append({
                        "step": int(parts[1]),
                        "recon_loss": float(parts[2]),
                        "para_sim": float(parts[3]),
                        "neg_sim": float(parts[4]),
                        "word_order_sim": float(parts[5]),
                        "eff_rank": int(parts[6]),
                        "lr": float(parts[7]),
                        "elapsed_hours": float(parts[8]),
                    })
                else:
                    # V2: timestamp,step,recon_loss,pos_sim,neg_sim,word_order_sim,lr,elapsed_hours
                    rows.append({
                        "step": int(parts[1]),
                        "recon_loss": float(parts[2]),
                        "para_sim": float(parts[3]),
                        "neg_sim": float(parts[4]),
                        "word_order_sim": float(parts[5]),
                        "lr": float(parts[6]),
                        "elapsed_hours": float(parts[7]),
                    })
            except (ValueError, IndexError):
                continue
    return rows


def smooth(values, window=50):
    if len(values) < 2:
        return values
    alpha = 2.0 / (window + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def fmt_step(x, _):
    return f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"


def plot_concept_dashboard(run="v2", save=True, show=False):
    cfg = CONCEPT_RUNS[run]
    step_data = load_concept_step_data(cfg["log"])
    eval_data = load_concept_eval_data(cfg["log"])
    csv_data = load_concept_csv(cfg["metrics"])

    if not step_data:
        print(f"No concept {run} training data to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.patch.set_facecolor("#FAFAFA")

    ax_recon = axes[0, 0]
    ax_components = axes[0, 1]
    ax_batch_sim = axes[0, 2]
    ax_diag = axes[1, 0]
    ax_wordorder = axes[1, 1]
    ax_stats = axes[1, 2]

    C_RECON = "#D32F2F"
    C_PARA_LOSS = "#2E7D32"
    C_NEG_LOSS = "#1565C0"
    C_WO_LOSS = "#FF6F00"
    C_TOTAL = "#424242"
    C_PSIM = "#2E7D32"
    C_NSIM = "#C62828"
    C_WOSIM = "#FF6F00"
    C_WO = "#D32F2F"
    C_BIND = "#FF6F00"
    C_PARA_SIM = "#2E7D32"
    C_UNREL = "#1565C0"

    steps = [d["step"] for d in step_data]

    # Panel 1: Reconstruction Loss
    recon_vals = [d["recon_loss"] for d in step_data]
    ax_recon.scatter(steps, recon_vals, s=2, alpha=0.15, color=C_RECON)
    if len(recon_vals) > 10:
        ax_recon.plot(steps, smooth(recon_vals, min(30, len(recon_vals) // 3)),
                      color=C_RECON, linewidth=2.5, label="Recon loss")
    ax_recon.set_xlabel("Step")
    ax_recon.set_ylabel("Loss")
    ax_recon.set_title("Reconstruction Loss")
    ax_recon.grid(True, alpha=0.3)
    ax_recon.legend(fontsize=10)
    ax_recon.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # Panel 2: Component Losses
    para_vals = [d["para_loss"] for d in step_data]
    neg_vals = [d["neg_loss"] for d in step_data]
    wo_vals = [d["wo_loss"] for d in step_data]
    decorr_vals = [d["decorr_loss"] for d in step_data]
    is_v3 = any(v > 0 for v in decorr_vals)
    if len(para_vals) > 10:
        sw = min(30, len(para_vals) // 3)
        ax_components.plot(steps, smooth(para_vals, sw),
                           color=C_PARA_LOSS, linewidth=2,
                           label="NCE loss" if is_v3 else "Para loss")
        if not is_v3:
            ax_components.plot(steps, smooth(neg_vals, sw),
                               color=C_NEG_LOSS, linewidth=2, label="Neg loss")
        if any(v > 0 for v in wo_vals):
            ax_components.plot(steps, smooth(wo_vals, sw),
                               color=C_WO_LOSS, linewidth=2, label="WO loss")
        if is_v3:
            ax_components.plot(steps, smooth(decorr_vals, sw),
                               color="#7B1FA2", linewidth=2, label="Decorr loss")
    ax_components.set_xlabel("Step")
    ax_components.set_ylabel("Loss")
    ax_components.set_title("Geometry Losses")
    ax_components.grid(True, alpha=0.3)
    ax_components.legend(fontsize=9)
    ax_components.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # Panel 3: Batch Similarities
    p_sims = [d["p_sim"] for d in step_data]
    n_sims = [d["n_sim"] for d in step_data]
    wo_sims_batch = [d["wo_sim"] for d in step_data]
    if len(p_sims) > 10:
        sw = min(30, len(p_sims) // 3)
        ax_batch_sim.plot(steps, smooth(p_sims, sw),
                          color=C_PSIM, linewidth=2.5, label="Pos batch sim")
        ax_batch_sim.plot(steps, smooth(n_sims, sw),
                          color=C_NSIM, linewidth=2.5, label="Neg batch sim")
        if any(v > 0 for v in wo_sims_batch):
            ax_batch_sim.plot(steps, smooth(wo_sims_batch, sw),
                              color=C_WOSIM, linewidth=2.5, label="WO batch sim")
    ax_batch_sim.axhline(y=0.9, color=C_PSIM, linestyle="--", alpha=0.4, linewidth=1)
    ax_batch_sim.axhline(y=0.3, color=C_NSIM, linestyle="--", alpha=0.4, linewidth=1)
    ax_batch_sim.set_xlabel("Step")
    ax_batch_sim.set_ylabel("Cosine Similarity")
    ax_batch_sim.set_title("Batch Similarities (dashed = targets)")
    ax_batch_sim.set_ylim(-0.1, 1.05)
    ax_batch_sim.grid(True, alpha=0.3)
    ax_batch_sim.legend(fontsize=9)
    ax_batch_sim.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # Panel 4: Diagnostic Similarities
    if eval_data:
        e_steps = [d["step"] for d in eval_data]
        ax_diag.plot(e_steps, [d.get("para_sim", 0) for d in eval_data],
                     color=C_PARA_SIM, linewidth=2.5, marker="o", markersize=4,
                     label="Paraphrase sim")
        ax_diag.plot(e_steps, [d.get("unrelated_sim", 0) for d in eval_data],
                     color=C_UNREL, linewidth=2.5, marker="s", markersize=4,
                     label="Unrelated sim")
        ax_diag.plot(e_steps, [d.get("word_order_sim", 0) for d in eval_data],
                     color=C_WO, linewidth=2.5, marker="^", markersize=4,
                     label="Word order sim")
        ax_diag.axhline(y=0.9, color=C_PARA_SIM, linestyle="--", alpha=0.3, linewidth=1)
        ax_diag.axhline(y=0.1, color=C_UNREL, linestyle="--", alpha=0.3, linewidth=1)
        ax_diag.axhline(y=0.3, color=C_WO, linestyle="--", alpha=0.3, linewidth=1)
    ax_diag.set_xlabel("Step")
    ax_diag.set_ylabel("Cosine Similarity")
    ax_diag.set_title("Diagnostic Similarities (dashed = targets)")
    ax_diag.set_ylim(-0.5, 1.05)
    ax_diag.grid(True, alpha=0.3)
    ax_diag.legend(fontsize=9)
    ax_diag.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # Panel 5: Effective Rank (V3) or Word Order Detail (V2)
    has_rank = eval_data and any(d.get("rank90") for d in eval_data)
    if has_rank:
        e_steps = [d["step"] for d in eval_data]
        rank90 = [d.get("rank90", 0) for d in eval_data]
        rank95 = [d.get("rank95", 0) for d in eval_data]
        ax_wordorder.plot(e_steps, rank90, color="#1565C0", linewidth=2.5,
                          marker="o", markersize=4, label="Rank 90%")
        ax_wordorder.plot(e_steps, rank95, color="#2E7D32", linewidth=2.5,
                          marker="s", markersize=4, label="Rank 95%")
        ax_wordorder.axhline(y=1024, color="gray", linestyle="--", alpha=0.3,
                             linewidth=1, label="Max (1024)")
        ax_wordorder.set_xlabel("Step")
        ax_wordorder.set_ylabel("Effective Rank (PCA dims)")
        ax_wordorder.set_title("Effective Rank (higher = better)")
        ax_wordorder.grid(True, alpha=0.3)
        ax_wordorder.legend(fontsize=9)
        ax_wordorder.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))
    else:
        if eval_data:
            e_steps = [d["step"] for d in eval_data]
            pairs = {
                "dog/man bit": ("pair_word_order_0", C_WO, "o"),
                "alice/bob likes": ("pair_word_order_1", "#E91E63", "s"),
                "she/he gave": ("pair_word_order_2", "#FF9800", "^"),
                "purple binding": ("pair_binding_0", C_BIND, "D"),
            }
            for label, (key, color, marker) in pairs.items():
                vals = [d.get(key) for d in eval_data]
                valid = [(s, v) for s, v in zip(e_steps, vals) if v is not None]
                if valid:
                    ax_wordorder.plot([s for s, _ in valid], [v for _, v in valid],
                                      color=color, linewidth=2, marker=marker,
                                      markersize=4, label=label)
            ax_wordorder.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3,
                                 linewidth=1, label="Goal: below 0.5")
        ax_wordorder.set_xlabel("Step")
        ax_wordorder.set_ylabel("Cosine Similarity")
        ax_wordorder.set_title("Word Order & Binding (lower = better)")
        ax_wordorder.set_ylim(-0.1, 1.05)
        ax_wordorder.grid(True, alpha=0.3)
        ax_wordorder.legend(fontsize=7, loc="lower left")
        ax_wordorder.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # Panel 6: Stats
    ax_stats.axis("off")
    lines = [f"flm V4 -- Concept Autoencoder ({run})", "=" * 44]
    if step_data:
        latest = step_data[-1]
        lines += [f"", f"  Step:              {latest['step']:>10,d} / 200,000",
                  f"  Recon loss:        {latest['recon_loss']:>10.3f}",
                  f"  Total loss:        {latest['total_loss']:>10.3f}"]
    if csv_data:
        lc = csv_data[-1]
        lines.append(f"  Time:              {lc['elapsed_hours']:>10.1f}h")
        if lc["step"] > 0 and lc["elapsed_hours"] > 0:
            remaining = (200_000 - lc["step"]) / (lc["step"] / lc["elapsed_hours"])
            lines.append(f"  ETA:               {remaining:>10.1f}h")
    if eval_data:
        le = eval_data[-1]
        lines += [f"", f"  {'Metric':<24s} {'Value':>8s} {'Target':>8s}",
                  f"  {'_' * 44}"]
        for display, key, target in [("Paraphrase sim", "para_sim", ">0.90"),
                                      ("Unrelated sim", "unrelated_sim", "<0.10"),
                                      ("Word order sim", "word_order_sim", "<0.30")]:
            lines.append(f"  {display:<24s} {le.get(key, 0):>+8.3f} {target:>8s}")
        if le.get("rank90"):
            lines += [f"", f"  Effective Rank:"]
            lines.append(f"    {'rank90':<20s} {le['rank90']:>6d} / 1024")
            if le.get("rank95"):
                lines.append(f"    {'rank95':<20s} {le['rank95']:>6d} / 1024")
        lines += [f"", f"  Word Order Detail:"]
        for display, key in [("dog/man bit", "pair_word_order_0"),
                              ("alice/bob likes", "pair_word_order_1"),
                              ("she/he gave", "pair_word_order_2"),
                              ("purple binding", "pair_binding_0")]:
            val = le.get(key)
            if val is not None:
                lines.append(f"    {display:<20s} {val:>+.3f}")
    ax_stats.text(0.02, 0.98, "\n".join(lines), transform=ax_stats.transAxes,
                  fontsize=8, verticalalignment="top", fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                            edgecolor="#BDBDBD"))

    plt.suptitle(f"flm V4 -- Concept Autoencoder Training ({run})",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        path = os.path.join(PLOT_DIR, f"concept_{run}_dashboard.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close()


def plot_comparison(save=True, show=False):
    """Side-by-side comparison of V1 and V2 concept training."""
    v1_step = load_concept_step_data(CONCEPT_RUNS["v1"]["log"])
    v2_step = load_concept_step_data(CONCEPT_RUNS["v2"]["log"])
    v1_eval = load_concept_eval_data(CONCEPT_RUNS["v1"]["log"])
    v2_eval = load_concept_eval_data(CONCEPT_RUNS["v2"]["log"])

    if not v1_step and not v2_step:
        print("No data for comparison.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.patch.set_facecolor("#FAFAFA")

    C_V1 = "#1565C0"  # blue
    C_V2 = "#D32F2F"  # red

    def get_max_step():
        steps = []
        if v1_step:
            steps.append(v1_step[-1]["step"])
        if v2_step:
            steps.append(v2_step[-1]["step"])
        return max(steps) if steps else 10000

    # Panel 1: Recon Loss comparison
    ax = axes[0, 0]
    for data, color, label in [(v1_step, C_V1, "V1"), (v2_step, C_V2, "V2")]:
        if data:
            s = [d["step"] for d in data]
            v = [d["recon_loss"] for d in data]
            if len(v) > 10:
                ax.plot(s, smooth(v, min(30, len(v) // 3)),
                        color=color, linewidth=2.5, label=f"{label} recon")
    ax.set_title("Reconstruction Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # Panel 2: Total Loss comparison
    ax = axes[0, 1]
    for data, color, label in [(v1_step, C_V1, "V1"), (v2_step, C_V2, "V2")]:
        if data:
            s = [d["step"] for d in data]
            v = [d["total_loss"] for d in data]
            if len(v) > 10:
                ax.plot(s, smooth(v, min(30, len(v) // 3)),
                        color=color, linewidth=2.5, label=f"{label} total")
    ax.set_title("Total Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # Panel 3: Neg batch sim comparison
    ax = axes[0, 2]
    for data, color, label in [(v1_step, C_V1, "V1"), (v2_step, C_V2, "V2")]:
        if data:
            s = [d["step"] for d in data]
            v = [d["n_sim"] for d in data]
            if len(v) > 10:
                ax.plot(s, smooth(v, min(30, len(v) // 3)),
                        color=color, linewidth=2.5, label=f"{label} neg sim")
    ax.axhline(y=0.3, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_title("Negative Batch Similarity")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # Panel 4: Word order sim (EVAL) comparison
    ax = axes[1, 0]
    for edata, color, label in [(v1_eval, C_V1, "V1"), (v2_eval, C_V2, "V2")]:
        if edata:
            s = [d["step"] for d in edata]
            v = [d.get("word_order_sim", 1.0) for d in edata]
            ax.plot(s, v, color=color, linewidth=2.5, marker="o", markersize=4,
                    label=f"{label} word order")
    ax.axhline(y=0.3, color="gray", linestyle="--", alpha=0.4, linewidth=1,
               label="Target (<0.3)")
    ax.set_title("Word Order Similarity (lower = better)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # Panel 5: Paraphrase vs Unrelated (EVAL)
    ax = axes[1, 1]
    for edata, color, label in [(v1_eval, C_V1, "V1"), (v2_eval, C_V2, "V2")]:
        if edata:
            s = [d["step"] for d in edata]
            p = [d.get("para_sim", 0) for d in edata]
            u = [d.get("unrelated_sim", 0) for d in edata]
            ax.plot(s, p, color=color, linewidth=2.5, marker="o", markersize=4,
                    label=f"{label} paraphrase", linestyle="-")
            ax.plot(s, u, color=color, linewidth=2, marker="s", markersize=3,
                    label=f"{label} unrelated", linestyle="--", alpha=0.7)
    ax.set_title("Paraphrase vs Unrelated Similarity")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(-0.5, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="center right")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    # Panel 6: dog/man detail comparison
    ax = axes[1, 2]
    for edata, color, label in [(v1_eval, C_V1, "V1"), (v2_eval, C_V2, "V2")]:
        if edata:
            s = [d["step"] for d in edata]
            for pair_label, key, ls in [("dog/man", "pair_word_order_0", "-"),
                                         ("alice/bob", "pair_word_order_1", "--"),
                                         ("binding", "pair_binding_0", ":")]:
                vals = [d.get(key) for d in edata]
                valid = [(st, v) for st, v in zip(s, vals) if v is not None]
                if valid:
                    ax.plot([st for st, _ in valid], [v for _, v in valid],
                            color=color, linewidth=2, linestyle=ls,
                            marker="o", markersize=3,
                            label=f"{label} {pair_label}")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    ax.set_title("Individual Pair Detail")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="lower left", ncol=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_step))

    plt.suptitle("flm V4 -- Concept Autoencoder: V1 vs V2 Comparison",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        path = os.path.join(PLOT_DIR, "concept_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close()


# =========================================================================
# Legacy V1/V2/V3 Pretrain Dashboard
# =========================================================================

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


def load_pretrain_metrics(filepath):
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
                if len(parts) >= 10 and parts[9]:
                    try:
                        row["val_loss"] = float(parts[9])
                    except ValueError:
                        row["val_loss"] = None
                else:
                    row["val_loss"] = None
                all_rows.append(row)
            except (ValueError, IndexError):
                continue

    last_restart = 0
    for i in range(1, len(all_rows)):
        if all_rows[i]["step"] < all_rows[i - 1]["step"] - 1000:
            last_restart = i
    metrics = all_rows[last_restart:]

    seen = {}
    for m in metrics:
        s = m["step"]
        if s not in seen:
            seen[s] = m
        else:
            if m.get("val_loss") is not None:
                seen[s]["val_loss"] = m["val_loss"]
    return list(seen.values())


def plot_legacy_dashboard(save=True, show=False):
    all_data = {}
    for name, cfg in RUNS.items():
        metrics = load_pretrain_metrics(cfg["file"])
        all_data[name] = metrics
        if metrics:
            print(f"  {name}: {len(metrics)} entries, "
                  f"{metrics[-1]['tokens_billions']:.2f}B tokens, "
                  f"loss {metrics[-1]['loss']:.3f}")
        else:
            print(f"  {name}: no data")

    if not any(all_data.values()):
        print("No metrics to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#FAFAFA")

    ax_loss_step = axes[0, 0]
    ax_loss_tok = axes[0, 1]
    ax_throughput = axes[1, 0]
    ax_stats = axes[1, 1]

    for name, metrics in all_data.items():
        if not metrics:
            continue
        cfg = RUNS[name]
        steps = [m["step"] for m in metrics]
        losses = [m["loss"] for m in metrics]
        ax_loss_step.scatter(steps, losses, s=1, alpha=0.08, color=cfg["color"])
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

    for name, metrics in all_data.items():
        if not metrics:
            continue
        cfg = RUNS[name]
        tb = [m["tokens_billions"] for m in metrics if m["tokens_billions"] > 0]
        lo = [m["loss"] for m in metrics if m["tokens_billions"] > 0]
        if not tb:
            continue
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

    ax_stats.axis("off")
    names = ["V1 (164M)", "V2 (493M)", "V3 (135M)"]
    header = f"  {'':>12s} {'V1(164M)':>10s} {'V2(493M)':>10s} {'V3(135M)':>10s}"
    sep = f"  {'_'*44}"
    stat_lines = ["flm Pretrain Comparison (Legacy)", "=" * 46, "",
                  "  Overall (latest)", header, sep]

    row_data = {
        "Tokens": lambda m: f"{m[-1]['tokens_billions']:.2f}B",
        "Loss": lambda m: f"{m[-1]['loss']:.4f}",
        "Time": lambda m: f"{m[-1]['elapsed_hours']:.1f}h",
    }
    for label, fn in row_data.items():
        vals = []
        for n in names:
            m = all_data.get(n, [])
            vals.append(fn(m) if m else "-")
        stat_lines.append(f"  {label:>12s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")

    stats_text = "\n".join(stat_lines)
    ax_stats.text(0.02, 0.98, stats_text, transform=ax_stats.transAxes,
                  fontsize=8.5, verticalalignment="top", fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                            edgecolor="#BDBDBD"))

    plt.suptitle("flm Training Dashboard - V1 vs V2 vs V3 (Legacy)",
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


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="flm training dashboard")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--live", action="store_true", help="Auto-refresh every 60s")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval")
    parser.add_argument("--legacy", action="store_true", help="V1/V2/V3 pretrain comparison")
    parser.add_argument("--compare", action="store_true", help="V1 vs V2 concept comparison")
    parser.add_argument("--run", type=str, default=None, help="Which run to plot (v1, v2, v3; default: latest with data)")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg")

    if args.legacy:
        plot_legacy_dashboard(save=not args.show, show=args.show)
        return

    if args.compare:
        plot_comparison(save=not args.show, show=args.show)
        return

    # Auto-detect latest run with data
    run = args.run
    if run is None:
        for r in ["v3", "v2", "v1"]:
            if os.path.exists(CONCEPT_RUNS[r]["log"]):
                run = r
                break
        if run is None:
            run = "v3"

    if args.live:
        print(f"Live mode - refreshing every {args.interval}s. Ctrl+C to stop.")
        while True:
            try:
                plot_concept_dashboard(run=run, save=True, show=False)
                if os.path.exists(CONCEPT_RUNS["v1"]["log"]) and os.path.exists(CONCEPT_RUNS["v2"]["log"]):
                    plot_comparison(save=True, show=False)
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        plot_concept_dashboard(run=run, save=not args.show, show=args.show)


if __name__ == "__main__":
    main()
