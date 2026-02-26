#!/usr/bin/env python3
"""Benchmark registry and loaders for multi-model evaluation.

Each loader returns a list of normalized question dicts:
    {"id": str, "prompt": str, "choices": list[str], "answer_idx": int,
     "scoring": "token"|"completion", "category": str|None}

Token MCQ: score by log-prob of a single answer token (A/B/C/D or Yes/No).
Completion MCQ: score by mean log-prob of each completion given the prompt.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "linux_bench": {
        "name": "LinuxBench",
        "scoring": "token",
        "num_choices": 4,
        "baseline": 0.25,
        "loader": "_load_linux_bench",
    },
    "arc_easy": {
        "name": "ARC-Easy",
        "scoring": "token",
        "num_choices": 4,
        "baseline": 0.25,
        "loader": "_load_arc_easy",
    },
    "hellaswag": {
        "name": "HellaSwag",
        "scoring": "completion",
        "num_choices": 4,
        "baseline": 0.25,
        "loader": "_load_hellaswag",
    },
    "piqa": {
        "name": "PIQA",
        "scoring": "completion",
        "num_choices": 2,
        "baseline": 0.50,
        "loader": "_load_piqa",
    },
    "winogrande": {
        "name": "WinoGrande",
        "scoring": "completion",
        "num_choices": 2,
        "baseline": 0.50,
        "loader": "_load_winogrande",
    },
    "boolq": {
        "name": "BoolQ",
        "scoring": "token",
        "num_choices": 2,
        "baseline": 0.50,
        "loader": "_load_boolq",
    },
}

# Convenience groups
BENCH_GROUPS = {
    "all": list(BENCHMARKS.keys()),
    "general": ["arc_easy", "hellaswag", "piqa", "winogrande", "boolq"],
}

LETTERS = ["A", "B", "C", "D", "E"]


def load_benchmark(name: str, limit: int | None = None) -> list[dict]:
    """Load a benchmark by name, returning normalized questions."""
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARKS.keys())}")
    meta = BENCHMARKS[name]
    loader_fn = globals()[meta["loader"]]
    questions = loader_fn()
    if limit and limit < len(questions):
        questions = questions[:limit]
    print(f"Loaded {len(questions)} questions from {meta['name']} "
          f"(scoring={meta['scoring']}, baseline={meta['baseline']:.0%})")
    return questions


def get_benchmark_meta(name: str) -> dict:
    """Get metadata for a benchmark."""
    return BENCHMARKS[name]


def resolve_bench_names(bench_arg: str) -> list[str]:
    """Resolve --bench argument to a list of benchmark names.

    Accepts: 'linux_bench', 'all', 'general', or comma-separated names.
    """
    if bench_arg in BENCH_GROUPS:
        return BENCH_GROUPS[bench_arg]
    names = [b.strip() for b in bench_arg.split(",")]
    for n in names:
        if n not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {n}. "
                             f"Available: {list(BENCHMARKS.keys())} + {list(BENCH_GROUPS.keys())}")
    return names


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_linux_bench() -> list[dict]:
    """Load LinuxBench from local JSON."""
    bench_path = Path(__file__).resolve().parent / "linux_bench.json"
    with open(bench_path) as f:
        raw = json.load(f)

    questions = []
    for q in raw:
        letters = sorted(q["choices"].keys())  # A, B, C, D
        choices = [q["choices"][l] for l in letters]
        answer_idx = letters.index(q["answer"])
        prompt = (
            f"Question: {q['question']}\n"
            + "\n".join(f"{l}. {q['choices'][l]}" for l in letters)
            + "\nAnswer:"
        )
        questions.append({
            "id": f"linux_{q['id']}",
            "prompt": prompt,
            "choices": choices,
            "answer_idx": answer_idx,
            "scoring": "token",
            "token_labels": letters,
            "category": q.get("category"),
            "difficulty": q.get("difficulty"),
        })
    return questions


def _load_arc_easy() -> list[dict]:
    """Load ARC-Easy from HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")

    questions = []
    for i, row in enumerate(ds):
        labels = row["choices"]["label"]   # e.g. ['A','B','C','D'] or ['1','2','3','4']
        texts = row["choices"]["text"]
        answer_key = row["answerKey"]

        # Normalize labels to A/B/C/D
        if answer_key in labels:
            answer_idx = labels.index(answer_key)
        else:
            # Some ARC rows use '1','2','3','4' as labels
            answer_idx = int(answer_key) - 1

        prompt = f"Question: {row['question']}\n"
        for j, (l, t) in enumerate(zip(LETTERS[:len(texts)], texts)):
            prompt += f"{l}. {t}\n"
        prompt += "Answer:"

        questions.append({
            "id": f"arc_easy_{i}",
            "prompt": prompt,
            "choices": texts,
            "answer_idx": answer_idx,
            "scoring": "token",
            "token_labels": LETTERS[:len(texts)],
            "category": None,
        })
    return questions


def _load_hellaswag() -> list[dict]:
    """Load HellaSwag from HuggingFace datasets.

    Completion-scored: prompt is the context, choices are the endings.
    """
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")

    questions = []
    for i, row in enumerate(ds):
        ctx = row["ctx"]
        endings = row["endings"]  # list of 4 strings
        answer_idx = int(row["label"])

        questions.append({
            "id": f"hellaswag_{i}",
            "prompt": ctx,
            "choices": endings,
            "answer_idx": answer_idx,
            "scoring": "completion",
            "category": row.get("activity_label"),
        })
    return questions


def _load_piqa() -> list[dict]:
    """Load PIQA (Physical Intuition QA) from HuggingFace datasets.

    Completion-scored: prompt is the goal, choices are sol1/sol2.
    """
    from datasets import load_dataset
    ds = load_dataset("lighteval/piqa", split="validation")

    questions = []
    for i, row in enumerate(ds):
        questions.append({
            "id": f"piqa_{i}",
            "prompt": row["goal"],
            "choices": [row["sol1"], row["sol2"]],
            "answer_idx": row["label"],
            "scoring": "completion",
            "category": None,
        })
    return questions


def _load_winogrande() -> list[dict]:
    """Load WinoGrande from HuggingFace datasets.

    Completion-scored: prompt is sentence with _ blank, choices fill the blank.
    """
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")

    questions = []
    for i, row in enumerate(ds):
        sentence = row["sentence"]  # contains '_' placeholder
        option1 = row["option1"]
        option2 = row["option2"]
        answer_idx = int(row["answer"]) - 1  # 1-indexed -> 0-indexed

        # Build two completions by substituting the blank
        choices = [
            sentence.replace("_", option1),
            sentence.replace("_", option2),
        ]

        questions.append({
            "id": f"winogrande_{i}",
            "prompt": sentence,
            "choices": choices,
            "answer_idx": answer_idx,
            "scoring": "completion",
            "category": None,
        })
    return questions


def _load_boolq() -> list[dict]:
    """Load BoolQ from HuggingFace datasets.

    Token-scored: model picks Yes or No.
    """
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="validation")

    questions = []
    for i, row in enumerate(ds):
        passage = row["passage"]
        question_text = row["question"]
        answer = row["answer"]  # True/False

        prompt = (
            f"{passage}\n\n"
            f"Question: {question_text}\n"
            f"A. Yes\n"
            f"B. No\n"
            f"Answer:"
        )

        questions.append({
            "id": f"boolq_{i}",
            "prompt": prompt,
            "choices": ["Yes", "No"],
            "answer_idx": 0 if answer else 1,
            "scoring": "token",
            "token_labels": ["A", "B"],
            "category": None,
        })
    return questions
