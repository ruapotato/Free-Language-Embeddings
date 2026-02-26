#!/bin/bash
# Run all comparison models on all benchmarks.
#
# Usage:
#   bash eval/run_all_models.sh              # all models, all benchmarks
#   bash eval/run_all_models.sh --limit 200  # cap per-benchmark questions
#
# NOTE: GPU-intensive. Run when GPU is free (not during training).

set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_ARGS="${*}"

echo "=== Multi-Model Benchmark Evaluation ==="
echo "Extra args: ${EXTRA_ARGS:-none}"
echo

# HuggingFace comparison models
for model in "gpt2" "HuggingFaceTB/SmolLM-135M" "EleutherAI/pythia-160m"; do
    echo "--- Evaluating: ${model} ---"
    python eval/run_eval.py --hf "${model}" --bench all ${EXTRA_ARGS}
    echo
done

# Optional upper-bound model (uncomment to include)
# echo "--- Evaluating: SmolLM-360M (upper bound) ---"
# python eval/run_eval.py --hf "HuggingFaceTB/SmolLM-360M" --bench all ${EXTRA_ARGS}

# flm checkpoints (uncomment when training completes)
# echo "--- Evaluating: flm (SFT) ---"
# python eval/run_eval.py --model checkpoints/sft/best.pt --bench all ${EXTRA_ARGS}

# Print comparison table
echo "=== Comparison Table ==="
python eval/run_eval.py --compare

# Generate plots
echo "=== Generating Plots ==="
python eval/plot_results.py --save --no-show

echo "Done. Results in eval/results/, plots in eval/results/plots/"
