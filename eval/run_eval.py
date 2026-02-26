#!/usr/bin/env python3
"""Multi-benchmark evaluation harness for flm and HuggingFace models.

Supports token-MCQ scoring (LinuxBench, ARC-Easy, BoolQ) and
completion-MCQ scoring (HellaSwag, PIQA, WinoGrande).

Usage:
    python eval/run_eval.py --hf gpt2 --bench linux_bench
    python eval/run_eval.py --hf gpt2 --bench all --limit 200
    python eval/run_eval.py --model checkpoints/sft/best.pt --bench linux_bench,arc_easy
    python eval/run_eval.py --compare   # print comparison table from results/
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow imports from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from eval.benchmarks import (
    BENCHMARKS,
    load_benchmark,
    get_benchmark_meta,
    resolve_bench_names,
)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

class FlmEvaluator:
    """Evaluator for local flm (Hamner) model checkpoints."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device

        from model import HamnerModel, HamnerConfig
        from transformers import AutoTokenizer

        print(f"Loading flm checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        config = HamnerConfig(**ckpt["config"])
        config.gradient_checkpointing = False
        self.model = HamnerModel(config).to(device)

        state_dict = ckpt["model_state_dict"]
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(cleaned, strict=True)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_len = config.max_seq_len if hasattr(config, "max_seq_len") else 1024

        step = ckpt.get("step", "?")
        loss = ckpt.get("avg_loss", "?")
        training_type = ckpt.get("training_type", "?")
        print(f"  Config: {config.hidden_size}h, {config.num_layers}L, "
              f"~{config.total_params_estimate()/1e6:.1f}M params")
        print(f"  Step: {step}, Loss: {loss}, Type: {training_type}")

        self.model_name = f"flm_{Path(checkpoint_path).stem}"

    def _truncate(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] > self.max_len:
            input_ids = input_ids[:, -self.max_len:]
        return input_ids

    @torch.no_grad()
    def get_answer_logprobs(self, prompt: str, token_labels: list[str]) -> dict[str, float]:
        """Get log-probabilities for answer tokens at the last position."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_ids = self._truncate(input_ids)

        outputs = self.model(input_ids)
        last_logits = outputs["logits"][:, -1, :]  # [1, vocab_size]
        log_probs = F.log_softmax(last_logits, dim=-1)

        result = {}
        for label in token_labels:
            token_id = self.tokenizer.encode(label, add_special_tokens=False)[0]
            result[label] = log_probs[0, token_id].item()
        return result

    @torch.no_grad()
    def get_completion_logprob(self, prompt: str, completion: str) -> float:
        """Mean log-prob of completion tokens given prompt context."""
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        if not completion_ids:
            return float("-inf")

        all_ids = torch.tensor([prompt_ids + completion_ids], device=self.device)
        all_ids = self._truncate(all_ids)

        outputs = self.model(all_ids)
        logits = outputs["logits"]  # [1, seq_len, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)

        # Score only the completion tokens
        n_prompt = len(prompt_ids)
        # After truncation, adjust n_prompt
        n_prompt = max(0, n_prompt - max(0, len(prompt_ids) + len(completion_ids) - all_ids.shape[1]))
        n_completion = all_ids.shape[1] - n_prompt

        total = 0.0
        for i in range(n_completion):
            pos = n_prompt + i  # position of completion token
            if pos == 0:
                continue
            token_id = all_ids[0, pos].item()
            total += log_probs[0, pos - 1, token_id].item()
        return total / max(n_completion, 1)


class HFEvaluator:
    """Evaluator for HuggingFace models."""

    def __init__(self, model_name: str, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        print(f"Loading HuggingFace model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        self.model.eval()

        self.max_len = getattr(self.model.config, "max_position_embeddings", 1024)

        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"  {params:.1f}M parameters")

        self.model_name = model_name.replace("/", "_")

    def _truncate(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] > self.max_len:
            input_ids = input_ids[:, -self.max_len:]
        return input_ids

    @torch.no_grad()
    def get_answer_logprobs(self, prompt: str, token_labels: list[str]) -> dict[str, float]:
        """Get log-probabilities for answer tokens at the last position."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_ids = self._truncate(input_ids)

        outputs = self.model(input_ids)
        last_logits = outputs.logits[:, -1, :]
        log_probs = F.log_softmax(last_logits, dim=-1)

        result = {}
        for label in token_labels:
            token_id = self.tokenizer.encode(label, add_special_tokens=False)[0]
            result[label] = log_probs[0, token_id].item()
        return result

    @torch.no_grad()
    def get_completion_logprob(self, prompt: str, completion: str) -> float:
        """Mean log-prob of completion tokens given prompt context."""
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        if not completion_ids:
            return float("-inf")

        all_ids = torch.tensor([prompt_ids + completion_ids], device=self.device)
        all_ids = self._truncate(all_ids)

        outputs = self.model(all_ids)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        n_prompt = len(prompt_ids)
        n_prompt = max(0, n_prompt - max(0, len(prompt_ids) + len(completion_ids) - all_ids.shape[1]))
        n_completion = all_ids.shape[1] - n_prompt

        total = 0.0
        for i in range(n_completion):
            pos = n_prompt + i
            if pos == 0:
                continue
            token_id = all_ids[0, pos].item()
            total += log_probs[0, pos - 1, token_id].item()
        return total / max(n_completion, 1)


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def eval_question(evaluator, question: dict) -> dict:
    """Evaluate a single question using the appropriate scoring method."""
    scoring = question["scoring"]

    if scoring == "token":
        labels = question.get("token_labels", ["A", "B", "C", "D"][:len(question["choices"])])
        logprobs = evaluator.get_answer_logprobs(question["prompt"], labels)
        predicted_label = max(logprobs, key=logprobs.get)
        predicted_idx = labels.index(predicted_label)
        return {
            "predicted_idx": predicted_idx,
            "correct": predicted_idx == question["answer_idx"],
            "scores": logprobs,
        }
    else:  # completion
        scores = {}
        for i, choice in enumerate(question["choices"]):
            scores[i] = evaluator.get_completion_logprob(question["prompt"], choice)
        predicted_idx = max(scores, key=scores.get)
        return {
            "predicted_idx": predicted_idx,
            "correct": predicted_idx == question["answer_idx"],
            "scores": {str(k): v for k, v in scores.items()},
        }


def run_benchmark(evaluator, bench_name: str, questions: list[dict]) -> dict:
    """Run a single benchmark and return results dict."""
    meta = get_benchmark_meta(bench_name)
    results = {
        "benchmark": bench_name,
        "benchmark_name": meta["name"],
        "scoring": meta["scoring"],
        "baseline": meta["baseline"],
        "total": len(questions),
        "correct": 0,
        "per_question": [],
        "per_category": defaultdict(lambda: {"correct": 0, "total": 0}),
    }

    start = time.time()

    for i, q in enumerate(questions):
        outcome = eval_question(evaluator, q)

        if outcome["correct"]:
            results["correct"] += 1

        results["per_question"].append({
            "id": q["id"],
            "correct": outcome["correct"],
            "predicted_idx": outcome["predicted_idx"],
            "answer_idx": q["answer_idx"],
            "scores": outcome["scores"],
            "category": q.get("category"),
        })

        cat = q.get("category")
        if cat:
            results["per_category"][cat]["total"] += 1
            if outcome["correct"]:
                results["per_category"][cat]["correct"] += 1

        # Progress
        if (i + 1) % 50 == 0 or i == 0 or i == len(questions) - 1:
            acc = results["correct"] / (i + 1) * 100
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(questions) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:4d}/{len(questions)}] {bench_name}: "
                  f"acc={acc:5.1f}% ({rate:.1f} q/s, ETA {eta:.0f}s)")

    elapsed = time.time() - start
    results["elapsed_seconds"] = round(elapsed, 1)
    results["accuracy"] = results["correct"] / results["total"]
    results["per_category"] = dict(results["per_category"])

    return results


def print_benchmark_report(bench_results: dict):
    """Print results for a single benchmark."""
    name = bench_results["benchmark_name"]
    acc = bench_results["accuracy"] * 100
    base = bench_results["baseline"] * 100
    n = bench_results["total"]
    correct = bench_results["correct"]
    elapsed = bench_results["elapsed_seconds"]

    print(f"\n  {name}: {correct}/{n} ({acc:.1f}%) "
          f"[baseline: {base:.0f}%] ({elapsed:.1f}s)")

    # Per-category breakdown if available
    cats = bench_results.get("per_category", {})
    if cats:
        for cat in sorted(cats):
            d = cats[cat]
            cat_acc = d["correct"] / d["total"] * 100 if d["total"] > 0 else 0
            print(f"    {cat:<25} {d['correct']:>3}/{d['total']:<3} ({cat_acc:.1f}%)")


# ---------------------------------------------------------------------------
# Multi-benchmark runner
# ---------------------------------------------------------------------------

def run_multi_benchmark(evaluator, bench_names: list[str], limit: int | None = None) -> dict:
    """Run multiple benchmarks, return combined results."""
    combined = {
        "model": evaluator.model_name,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {},
    }

    print(f"\nEvaluating {evaluator.model_name} on {len(bench_names)} benchmark(s)...")

    for bench_name in bench_names:
        print(f"\n--- {bench_name} ---")
        questions = load_benchmark(bench_name, limit=limit)
        bench_results = run_benchmark(evaluator, bench_name, questions)
        print_benchmark_report(bench_results)
        combined["benchmarks"][bench_name] = bench_results

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {evaluator.model_name}")
    print(f"{'=' * 60}")
    print(f"{'Benchmark':<20} {'Accuracy':>10} {'Baseline':>10} {'Delta':>10}")
    print("-" * 52)
    for bname, br in combined["benchmarks"].items():
        acc = br["accuracy"] * 100
        base = br["baseline"] * 100
        delta = acc - base
        print(f"{br['benchmark_name']:<20} {acc:>9.1f}% {base:>9.0f}% {delta:>+9.1f}%")
    print()

    return combined


# ---------------------------------------------------------------------------
# Comparison table from saved results
# ---------------------------------------------------------------------------

def print_comparison_table(results_dir: str):
    """Load all results JSONs and print a comparison table."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"No results directory found at {results_dir}")
        return

    # Collect latest result per model
    model_results = {}
    for f in sorted(results_path.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        model = data.get("model", "unknown")
        # Keep latest per model (files are sorted, so last wins)
        model_results[model] = data

    if not model_results:
        print("No result files found.")
        return

    # Collect all benchmark names across all models
    all_benchmarks = set()
    for data in model_results.values():
        all_benchmarks.update(data.get("benchmarks", {}).keys())
    bench_order = ["linux_bench", "arc_easy", "hellaswag", "piqa", "winogrande", "boolq"]
    bench_list = [b for b in bench_order if b in all_benchmarks]

    # Pretty names
    bench_names = {b: BENCHMARKS[b]["name"] for b in bench_list}

    # Header
    header = f"{'Model':<25}"
    for b in bench_list:
        header += f" {bench_names[b]:>12}"
    print(f"\n{'=' * (25 + 13 * len(bench_list))}")
    print("MODEL COMPARISON")
    print(f"{'=' * (25 + 13 * len(bench_list))}")
    print(header)
    print("-" * (25 + 13 * len(bench_list)))

    # Rows
    for model in sorted(model_results):
        data = model_results[model]
        row = f"{model:<25}"
        for b in bench_list:
            bench_data = data.get("benchmarks", {}).get(b)
            if bench_data:
                acc = bench_data["accuracy"] * 100
                row += f" {acc:>11.1f}%"
            else:
                row += f" {'—':>12}"
        print(row)

    # Baseline row
    row = f"{'random baseline':<25}"
    for b in bench_list:
        base = BENCHMARKS[b]["baseline"] * 100
        row += f" {base:>11.0f}%"
    print("-" * (25 + 13 * len(bench_list)))
    print(row)
    print()


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: dict, output_dir: str) -> str:
    """Save combined results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results['model']}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Strip per_question data to keep files manageable
    save_data = {
        "model": results["model"],
        "timestamp": results["timestamp"],
        "benchmarks": {},
    }
    for bname, bdata in results["benchmarks"].items():
        save_data["benchmarks"][bname] = {
            "benchmark": bdata["benchmark"],
            "benchmark_name": bdata["benchmark_name"],
            "scoring": bdata["scoring"],
            "baseline": bdata["baseline"],
            "total": bdata["total"],
            "correct": bdata["correct"],
            "accuracy": bdata["accuracy"],
            "elapsed_seconds": bdata["elapsed_seconds"],
            "per_category": bdata.get("per_category", {}),
        }

    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-benchmark evaluation harness")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", type=str, help="Path to flm checkpoint (.pt)")
    group.add_argument("--hf", type=str, help="HuggingFace model name (e.g., gpt2)")

    parser.add_argument("--bench", type=str, default="linux_bench",
                        help="Benchmark(s): linux_bench, arc_easy, hellaswag, piqa, "
                             "winogrande, boolq, general, all, or comma-separated")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max questions per benchmark (useful for large benchmarks)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for results (default: eval/results/)")
    parser.add_argument("--compare", action="store_true",
                        help="Print comparison table from saved results and exit")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.output_dir is None:
        args.output_dir = str(script_dir / "results")

    # Compare mode
    if args.compare:
        print_comparison_table(args.output_dir)
        return

    # Need a model for evaluation
    if not args.model and not args.hf:
        parser.error("Either --model or --hf is required (unless using --compare)")

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Create evaluator
    if args.model:
        evaluator = FlmEvaluator(args.model, device=args.device)
    else:
        evaluator = HFEvaluator(args.hf, device=args.device)

    # Resolve benchmarks
    bench_names = resolve_bench_names(args.bench)

    # Run
    results = run_multi_benchmark(evaluator, bench_names, limit=args.limit)

    # Save
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
