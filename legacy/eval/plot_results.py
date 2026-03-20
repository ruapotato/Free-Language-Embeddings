#!/usr/bin/env python3
"""Comparison plots for multi-model benchmark evaluation.

Reads eval/results/*.json and generates:
  1. benchmark_comparison.png  — grouped bar chart (main "money plot")
  2. benchmark_radar.png       — radar/spider chart
  3. linuxbench_categories.png — per-category breakdown for LinuxBench
  4. flm_vs_smollm.png         — delta chart (flm − SmolLM)

Usage:
    python eval/plot_results.py              # show plots
    python eval/plot_results.py --save       # save to eval/results/plots/
    python eval/plot_results.py --save --no-show
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Style constants (matching plot_training.py conventions)
# ---------------------------------------------------------------------------

DPI = 150
GRID_ALPHA = 0.3

MODEL_COLORS = {
    "flm":          "#F44336",  # red — protagonist
    "SmolLM-135M":  "#4CAF50",  # green
    "GPT-2":        "#9E9E9E",  # gray
    "Pythia-160M":  "#2196F3",  # blue
    "SmolLM-360M":  "#FF9800",  # orange — upper bound
}

# Map result file model names → display names
MODEL_DISPLAY = {
    "gpt2":                     "GPT-2",
    "HuggingFaceTB_SmolLM-135M": "SmolLM-135M",
    "EleutherAI_pythia-160m":   "Pythia-160M",
    "HuggingFaceTB_SmolLM-360M": "SmolLM-360M",
}

BENCH_ORDER = ["linux_bench", "arc_easy", "hellaswag", "piqa", "winogrande", "boolq"]
BENCH_DISPLAY = {
    "linux_bench": "LinuxBench",
    "arc_easy":    "ARC-Easy",
    "hellaswag":   "HellaSwag",
    "piqa":        "PIQA",
    "winogrande":  "WinoGrande",
    "boolq":       "BoolQ",
}


def _display_name(model_key: str) -> str:
    """Convert result file model key to display name."""
    if model_key.startswith("flm"):
        return "flm"
    return MODEL_DISPLAY.get(model_key, model_key)


def _color(display_name: str) -> str:
    return MODEL_COLORS.get(display_name, "#607D8B")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_dir: str) -> dict[str, dict]:
    """Load all result JSONs, keep latest per model. Returns {display_name: data}."""
    rdir = Path(results_dir)
    if not rdir.exists():
        print(f"Results dir not found: {results_dir}")
        sys.exit(1)

    raw = {}
    for f in sorted(rdir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        model_key = data.get("model", "unknown")
        raw[model_key] = data  # last (sorted) wins

    # Convert to display names
    results = {}
    for key, data in raw.items():
        dname = _display_name(key)
        results[dname] = data
    return results


# ---------------------------------------------------------------------------
# Plot 1: Grouped bar chart
# ---------------------------------------------------------------------------

def plot_benchmark_comparison(results: dict, save_path: str | None = None):
    """Grouped bar chart — one cluster per benchmark, one bar per model."""
    # Determine which benchmarks have data
    all_benchmarks = set()
    for data in results.values():
        all_benchmarks.update(data.get("benchmarks", {}).keys())
    bench_list = [b for b in BENCH_ORDER if b in all_benchmarks]
    model_list = [m for m in results if any(
        b in results[m].get("benchmarks", {}) for b in bench_list)]

    if not bench_list or not model_list:
        print("No benchmark data to plot.")
        return

    n_bench = len(bench_list)
    n_models = len(model_list)
    bar_width = 0.8 / n_models
    x = np.arange(n_bench)

    fig, ax = plt.subplots(figsize=(max(10, n_bench * 2), 6))

    for i, model in enumerate(model_list):
        accs = []
        for b in bench_list:
            bd = results[model].get("benchmarks", {}).get(b)
            accs.append(bd["accuracy"] * 100 if bd else 0)
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, accs, bar_width * 0.9,
                       label=model, color=_color(model), zorder=3)
        # Value labels
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{acc:.1f}", ha="center", va="bottom", fontsize=7)

    # Random baselines (dashed lines per benchmark)
    from eval.benchmarks import BENCHMARKS
    for j, b in enumerate(bench_list):
        base = BENCHMARKS[b]["baseline"] * 100
        ax.plot([x[j] - 0.5, x[j] + 0.5], [base, base],
                color="black", linestyle="--", linewidth=1, alpha=0.5, zorder=2)
        if j == 0:
            ax.plot([], [], color="black", linestyle="--", linewidth=1,
                    alpha=0.5, label="Random baseline")

    ax.set_xticks(x)
    ax.set_xticklabels([BENCH_DISPLAY.get(b, b) for b in bench_list], fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Multi-Model Benchmark Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=GRID_ALPHA, zorder=0)
    ax.set_ylim(0, min(100, ax.get_ylim()[1] * 1.15))
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Radar chart
# ---------------------------------------------------------------------------

def plot_radar(results: dict, save_path: str | None = None):
    """Radar/spider chart — each model is a polygon, axes are benchmarks."""
    all_benchmarks = set()
    for data in results.values():
        all_benchmarks.update(data.get("benchmarks", {}).keys())
    bench_list = [b for b in BENCH_ORDER if b in all_benchmarks]
    model_list = list(results.keys())

    if len(bench_list) < 3:
        print("Radar chart needs >= 3 benchmarks, skipping.")
        return None

    n = len(bench_list)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in model_list:
        values = []
        for b in bench_list:
            bd = results[model].get("benchmarks", {}).get(b)
            values.append(bd["accuracy"] * 100 if bd else 0)
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model,
                color=_color(model), markersize=5)
        ax.fill(angles, values, alpha=0.1, color=_color(model))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([BENCH_DISPLAY.get(b, b) for b in bench_list], fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Benchmark Profile", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(alpha=GRID_ALPHA)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 3: LinuxBench category breakdown
# ---------------------------------------------------------------------------

def plot_linuxbench_categories(results: dict, save_path: str | None = None):
    """Per-category grouped bars for LinuxBench across models."""
    # Collect models that have linux_bench data with per_category
    models_with_lb = {}
    for model, data in results.items():
        lb = data.get("benchmarks", {}).get("linux_bench", {})
        cats = lb.get("per_category", {})
        if cats:
            models_with_lb[model] = cats

    if not models_with_lb:
        print("No LinuxBench per-category data, skipping category plot.")
        return None

    # Union of all categories
    all_cats = sorted(set(c for cats in models_with_lb.values() for c in cats))
    model_list = list(models_with_lb.keys())
    n_cats = len(all_cats)
    n_models = len(model_list)
    bar_width = 0.8 / n_models
    x = np.arange(n_cats)

    fig, ax = plt.subplots(figsize=(max(10, n_cats * 1.5), 6))

    for i, model in enumerate(model_list):
        accs = []
        for cat in all_cats:
            cd = models_with_lb[model].get(cat, {"correct": 0, "total": 0})
            acc = cd["correct"] / cd["total"] * 100 if cd["total"] > 0 else 0
            accs.append(acc)
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, accs, bar_width * 0.9,
               label=model, color=_color(model), zorder=3)

    ax.axhline(y=25, color="black", linestyle="--", linewidth=1, alpha=0.5,
               label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(all_cats, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("LinuxBench — Per-Category Breakdown", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=GRID_ALPHA, zorder=0)
    ax.set_ylim(0, 100)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Plot 4: Delta chart (flm vs SmolLM)
# ---------------------------------------------------------------------------

def plot_delta(results: dict, save_path: str | None = None):
    """Horizontal bars: accuracy difference (flm − SmolLM-135M) per benchmark."""
    if "flm" not in results:
        print("No flm results found, skipping delta plot.")
        return None

    # Find SmolLM-135M (or fall back to any SmolLM)
    smol = None
    for name in ["SmolLM-135M", "SmolLM-360M"]:
        if name in results:
            smol = name
            break
    if smol is None:
        print("No SmolLM results found, skipping delta plot.")
        return None

    all_benchmarks = set()
    for m in ["flm", smol]:
        all_benchmarks.update(results[m].get("benchmarks", {}).keys())
    bench_list = [b for b in BENCH_ORDER if b in all_benchmarks]

    if not bench_list:
        return None

    deltas = []
    bench_labels = []
    for b in bench_list:
        flm_acc = results["flm"].get("benchmarks", {}).get(b, {}).get("accuracy", 0) * 100
        smol_acc = results[smol].get("benchmarks", {}).get(b, {}).get("accuracy", 0) * 100
        deltas.append(flm_acc - smol_acc)
        bench_labels.append(BENCH_DISPLAY.get(b, b))

    fig, ax = plt.subplots(figsize=(8, max(4, len(bench_list) * 0.8)))
    y = np.arange(len(bench_list))
    colors = ["#F44336" if d >= 0 else "#4CAF50" for d in deltas]
    bars = ax.barh(y, deltas, color=colors, zorder=3, height=0.6)

    # Value labels
    for bar, d in zip(bars, deltas):
        x_pos = bar.get_width()
        ha = "left" if d >= 0 else "right"
        offset = 0.3 if d >= 0 else -0.3
        ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                f"{d:+.1f}%", ha=ha, va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(bench_labels, fontsize=10)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Accuracy Difference (percentage points)", fontsize=11)
    ax.set_title(f"flm vs {smol}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=GRID_ALPHA, zorder=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot multi-model benchmark results")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory with result JSONs (default: eval/results/)")
    parser.add_argument("--save", action="store_true", help="Save plots to disk")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.results_dir is None:
        args.results_dir = str(script_dir / "results")

    plot_dir = str(Path(args.results_dir) / "plots")

    results = load_all_results(args.results_dir)
    if not results:
        print("No results to plot.")
        return

    print(f"Found results for: {', '.join(results.keys())}")

    if args.save:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

    if not args.save and args.no_show:
        matplotlib.use("Agg")

    # Generate all plots
    save = lambda name: str(Path(plot_dir) / name) if args.save else None

    plot_benchmark_comparison(results, save("benchmark_comparison.png"))
    plot_radar(results, save("benchmark_radar.png"))
    plot_linuxbench_categories(results, save("linuxbench_categories.png"))
    plot_delta(results, save("flm_vs_smollm.png"))

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
