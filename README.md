# flm — The Free Language Model

A fully free AI assistant trained from scratch on a single RTX 3090. 493M parameters, every dataset DFSG-compliant, every weight reproducible. Built to be the first AI model you can `apt install` from Debian main.

**Free as in freedom** — the name is a direct reference to the Free Software Foundation's philosophy that software freedom is a matter of liberty, not price. flm proves that a fully free, fully reproducible language model can exist and be useful.

## What flm Is

- **An OS-aware assistant** that becomes an expert on whatever operating system it's built on
- **493M parameters** — trained on a single GPU from random weights
- **100% DFSG-compliant data** — no Common Crawl, no AI-generated content, every source explicitly licensed
- **Fully reproducible** — one person, one RTX 3090, from zero to working model
- **GPL-3.0** for weights, training code, and all custom data

flm is NOT a general-purpose chatbot competing with GPT-4. It's a focused, knowledgeable OS assistant that happens to be fully free software.

## Architecture

493M parameter dense transformer (V2), scaled up from the V1 164M winner of a [tournament of 10 competing designs](#tournament-architecture-search):

| Parameter | Value |
|-----------|-------|
| Hidden size | 1280 |
| Layers | 26 |
| Attention heads | 16 (4 KV, GQA 4:1) |
| MLP | Dense SwiGLU |
| Intermediate size | 3456 |
| Parameters | 493M |
| Sequence length | 2048 |
| Tokenizer | cosmo2-tokenizer (49,152 vocab) |

Key components: RMSNorm, SwiGLU, RoPE, GQA, fp16 mixed precision, torch.compile, gradient checkpointing. Fits on RTX 3090 at batch 8-10 with ~18 GB peak VRAM.

## Training Pipeline

3-phase pipeline inspired by [SmolLM2](https://arxiv.org/abs/2502.02737), trained on a single RTX 3090 with exclusively DFSG-compliant, human-written data.

### Phase 1: Pretraining (~10B tokens, multi-stage data mix)

One continuous pretraining run with the data mix evolving across 4 internal stages, following SmolLM2's curriculum approach. Linux/Unix knowledge is included from step 1.

```bash
python build_dataset.py      # download + process all 11 DFSG sources (~40 GB)
python train_pretrain.py --fresh   # ~15 days on RTX 3090
```

**Hyperparameters**: LR 5e-4, WSD schedule (warmup-stable-decay), AdamW (betas 0.9/0.95), weight decay 0.1, seq length 2048, batch 8, gradient checkpointing ON, torch.compile. 610K steps, ~7,800 tok/s on RTX 3090.

**Internal stages (evolving data mix):**

| Stage | Steps | General Text | Linux/Unix Docs | Code |
|-------|-------|-------------|-----------------|------|
| 1 | 0–305K | 80% | 10% | 10% |
| 2 | 305K–409K | 65% | 20% | 15% |
| 3 | 409K–506K | 55% | 25% | 20% |
| 4 | 506K–610K | 45% | 30% | 25% |

### Phase 2: Supervised Fine-Tuning

Teach flm who it is and how to behave as a helpful OS assistant.

```bash
python train_sft.py
```

- **Data**: OASST2 (human-written) + custom flm identity/Linux data + Ubuntu Dialogue Corpus
- **LR**: 3e-4, 2 epochs (SmolLM2 recipe)
- **Output**: `checkpoints/sft/best.pt`

### Phase 3: DPO Alignment

Direct Preference Optimization using human preference data.

```bash
python train_dpo.py
```

- **Data**: Audited preference pairs (CC-BY-4.0)
- **LR**: 1e-6, beta 0.5, 2 epochs (SmolLM2 recipe)
- **Output**: `checkpoints/dpo/best.pt` — the final model

## Training Data — 100% DFSG-Compliant, 100% Human-Written

~12B tokens across 11 sources, all explicitly licensed for redistribution and derivative works. No Common Crawl derivatives (FineWeb, DCLM, C4, etc.) — web scraping doesn't respect page-level licenses. No AI-generated content — model output provenance is always ambiguous.

### General Knowledge (~11.7B tokens)

| Source | License | Docs | Est. Tokens | Description |
|--------|---------|------|-------------|-------------|
| [Wikipedia English](https://dumps.wikimedia.org/) | CC-BY-SA | 5.9M articles | ~6.0B | General knowledge anchor |
| [Project Gutenberg](https://www.gutenberg.org/) | Public Domain | 38K books | ~4.7B | Language quality, long-form text |
| [Stack Exchange](https://archive.org/details/stackexchange) (all sites, score>=3) | CC-BY-SA | 2.9M posts | ~1.0B | Technical Q&A, reasoning |

### Linux/Unix Specialization (~237M tokens)

| Source | License | Docs | Est. Tokens | Description |
|--------|---------|------|-------------|-------------|
| [RFC documents](https://www.rfc-editor.org/) | IETF | 9,710 | ~164M | Networking standards |
| Debian man pages (all packages) | GPL/BSD | 8,336 | ~30M | Command reference |
| [GNU Info manuals](https://www.gnu.org/manual/) | GFDL | 46 | ~14M | Core tool documentation |
| Linux kernel docs | GPL-2.0 | 4,721 | ~10M | Kernel internals |
| [TLDP](https://tldp.org/) guides | GFDL | 422 | ~10M | Classic Linux instruction |
| [Arch Wiki](https://wiki.archlinux.org/) | GFDL | 2,109 | ~9M | Practical configuration |

### Code (~128M tokens)

| Source | License | Files | Est. Tokens | Description |
|--------|---------|-------|-------------|-------------|
| Linux kernel source | GPL-2.0 | 16,692 | ~91M | Systems code |
| Curated FOSS repos (coreutils, openssh, git, curl, nginx, etc.) | MIT/GPL/Apache | 9,698 | ~37M | Code understanding |

### Post-Training

| Source | License | Stage | Description |
|--------|---------|-------|-------------|
| [OpenAssistant OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache-2.0 | SFT | Human-written conversations |
| Custom flm data | GPL-3.0 | SFT | Identity, packaging, sysadmin help |
| [Ubuntu Dialogue Corpus](https://huggingface.co/datasets/ubuntu-dialogs-corpus/ubuntu_dialogs_corpus) | Apache-2.0 | SFT | Linux IRC troubleshooting |
| Audited preference data | CC-BY-4.0 | DPO | Human-annotated preferences |

### Why no Common Crawl?

Datasets like FineWeb-Edu, DCLM, C4, and RedPajama are filtered versions of Common Crawl. Common Crawl scrapes the web without per-page license verification. Most web pages are "all rights reserved" by default. For Debian's DFSG standard — which requires explicit permission to redistribute and create derivative works — this is insufficient. We use only sources where the license explicitly grants these rights.

### Why no AI-generated content?

Datasets like COSMOPEDIA (Mixtral-generated) and SmolTalk subsets (GPT-4, Llama) have ambiguous provenance. Even when the generating model has a permissive license, its training data may not. OpenAI's ToS explicitly prohibits using outputs to train competing models. For Debian's bar, the chain of provenance needs to be clean all the way down.

## Reproduce from Scratch

```bash
# 1. Setup
git clone https://github.com/ruapotato/chat_hamner.git
cd chat_hamner
pip install -r requirements.txt

# 2. Build dataset (downloads + processes all 11 DFSG sources, ~40 GB)
python build_dataset.py

# 3. Phase 1: Pretrain (~15 days on RTX 3090)
python train_pretrain.py --fresh

# 4. Phase 2: SFT
python train_sft.py

# 5. Phase 3: DPO alignment
python train_dpo.py

# 6. Chat!
python chat.py
```

**Prerequisites**: NVIDIA GPU with 24GB+ VRAM, 32GB+ RAM, Python 3.10+, CUDA 12.x, ~100GB disk for data + checkpoints.

## Evaluation

LinuxBench is evaluated periodically during pretraining to track Linux knowledge acquisition.

### Benchmarks

| Benchmark | Source | Size | Scoring | Baseline |
|-----------|--------|------|---------|----------|
| LinuxBench | Custom (315 MCQ) | 315 | Token MCQ | 25% |
| ARC-Easy | `allenai/ai2_arc` | 2,376 | Token MCQ | 25% |
| HellaSwag | `Rowan/hellaswag` | 10,042 | Completion MCQ | 25% |
| PIQA | `lighteval/piqa` | 1,838 | Completion MCQ | 50% |
| WinoGrande | `allenai/winogrande` | 1,267 | Completion MCQ | 50% |
| BoolQ | `google/boolq` | 3,270 | Token MCQ | 50% |

### V1 Results (164M, for reference)

| Benchmark | flm V1 (best stage) | SmolLM-135M | Target V2 |
|-----------|---------------------|-------------|-----------|
| LinuxBench | 19.0% | 42.0% | >35% |
| ARC-Easy | 25.3% | 31.0% | >30% |
| HellaSwag | 31.0% | 36.0% | >35% |
| PIQA | 63.6% | 74.0% | >65% |
| WinoGrande | 49.0% | 50.0% | >52% |
| BoolQ | 62.0% | 33.0%* | >62% |

*SmolLM BoolQ tested on 100 samples only.

## V1 History

V1 (164M params) completed all 5 training stages but underperformed SmolLM-135M on most benchmarks. Key issues:
- Pretraining used Common Crawl derivatives (FineWeb-Edu, DCLM) — not truly DFSG-compliant
- Linux knowledge never materialized (LinuxBench 19% vs SmolLM's 42%)
- 5-stage pipeline caused cumulative degradation (PIQA: 63.6% → 57.1%)
- SFT data contained identity contamination from OASST2 and SmolTalk

V1 checkpoints preserved in `checkpoints/archive_v1/`.

## Tournament Architecture Search (V1)

We trained **10 different architectures** in a 3-round elimination tournament to pick the best design:

| # | Variant | Params | Key Idea | Result |
|---|---------|--------|----------|--------|
| 1 | Dense small | 75.5M | Small baseline | Eliminated R2 |
| 2 | **Dense medium** | **163.6M** | **Standard GQA** | **WINNER** |
| 3 | DiffAttn small | 81.8M | Noise-canceling attention | Eliminated R2 |
| 4 | DiffAttn medium | 179.4M | Medium + differential | Eliminated R3 |
| 5 | MoE small | 185.7M (100M active) | Sparse routing | Eliminated R2 |
| 6 | MoE medium | 516M (~180M active) | Large sparse | Eliminated R3 |
| 7 | Hybrid small | 190.4M | DiffAttn + MoE | Eliminated R1 |
| 8 | Hybrid medium | 528.6M | Full hybrid | OOM |
| 9 | Deep narrow | 70.8M | 32 layers, narrow | Eliminated R3 |
| 10 | Wide shallow | 121.7M | 8 layers, wide | OOM |

V2 scales the winner (Dense GQA) from 164M → 493M: wider (1280 vs 768), deeper (26 vs 20 layers), same architecture.

## Key Lessons Learned

1. **Data provenance matters more than data volume** — FineWeb-Edu gave us 10B tokens but Linux knowledge never materialized. Quality, domain-specific data beats generic web scrapes.
2. **Fewer training stages = less degradation** — V1's 5 stages eroded PIQA by 6.5 points. V2 uses 3 phases (SmolLM2 approach).
3. **Audit everything for contamination** — OASST2 had Open Assistant identity leaks. SmolTalk had GPT-4 outputs. License labels can be misleading.
4. **Common Crawl is not DFSG-compliant** — Web pages are "all rights reserved" by default. Scraping without consent doesn't create a license.
5. **Linux knowledge needs Linux data in pretraining** — Adding domain data only in later stages doesn't work. The model needs it from step 1.

## Project Structure

```
flm/
├── model.py                  # Core transformer architecture (493M)
├── chat.py                   # Interactive CLI chat
├── build_dataset.py          # Download + process all 11 DFSG data sources
├── train_pretrain.py         # Phase 1: Multi-stage pretraining
├── train_sft.py              # Phase 2: Supervised fine-tuning
├── train_dpo.py              # Phase 3: DPO alignment
├── eval/
│   ├── run_eval.py           # Multi-benchmark evaluation harness
│   ├── benchmarks.py         # Benchmark registry + loaders (6 benchmarks)
│   ├── linux_bench.json      # LinuxBench: 315 Linux MCQ questions
│   └── results/              # Evaluation results + plots
├── data/                     # Processed training data (built by build_dataset.py)
├── checkpoints/              # Model checkpoints
│   └── archive_v1/           # V1 (164M) checkpoints for reference
└── logs/                     # Training logs, metrics, sample generations
```

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.

Built by David Hamner with help from Claude.
