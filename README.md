# flm — The Free Language Model

> **Status: Active development (V3).** Redesigned from scratch with SmolLM-135M's proven architecture and Common Pile data. Starting with a 10B token test run to validate coherent output before scaling up.

A fully free AI assistant trained from scratch on a single RTX 3090. 135M parameters, every dataset DFSG-compliant, every weight reproducible. Built to be the first AI model you can `apt install` from Debian main.

**Free as in freedom** — the name is a direct reference to the Free Software Foundation's philosophy that software freedom is a matter of liberty, not price. flm proves that a fully free, fully reproducible language model can exist and be useful.

## What flm Is

- **An OS-aware assistant** that becomes an expert on whatever operating system it's built on
- **135M parameters** (SmolLM-135M architecture) — trained on a single GPU from random weights
- **100% DFSG-compliant data** — Common Pile (safe subset) + curated Linux data, no Common Crawl, no AI-generated content
- **Fully reproducible** — one person, one RTX 3090, from zero to working model
- **GPL-3.0** for weights, training code, and all custom data

flm is NOT a general-purpose chatbot competing with GPT-4. It's a focused, knowledgeable OS assistant that happens to be fully free software.

## Architecture

135M parameter dense transformer (V3), using SmolLM-135M's exact architecture — deep and narrow (MobileLLM insight):

| Parameter | Value |
|-----------|-------|
| Hidden size | 576 |
| Layers | 30 |
| Attention heads | 9 (3 KV, GQA 3:1) |
| Head dim | 64 |
| MLP | Dense SwiGLU |
| Intermediate size | 1536 |
| Parameters | ~135M |
| Sequence length | 2048 |
| Tokenizer | cosmo2-tokenizer (49,152 vocab) |
| Tied embeddings | Yes |

Key components: RMSNorm, SwiGLU, RoPE, GQA, fp16 mixed precision, torch.compile, gradient checkpointing.

### Why This Architecture?

V2 (493M, 26 layers × 1280 hidden) failed because we couldn't feed it enough tokens — 3.5B tokens for a 493M model is severely undertrained (Chinchilla says ~10B minimum). SmolLM-135M at 10B tokens is **~74× Chinchilla-optimal**, which is enough for coherent text and real benchmark scores.

## Training Pipeline

3-phase pipeline trained on a single RTX 3090 with exclusively DFSG-compliant, human-written data.

### Phase 1: Pretraining (10B tokens, ~6 days on RTX 3090)

Single flat data mix — no staged ramp for the 10B test run.

```bash
python build_dataset.py      # download + process all data sources
python train_pretrain.py --fresh   # ~6 days on RTX 3090
```

**Hyperparameters**: LR 1e-3, WSD schedule (warmup-stable-decay), AdamW (betas 0.9/0.95), weight decay 0.1, seq length 2048, batch 32, gradient checkpointing ON, torch.compile. ~152K steps, ~20K tok/s on RTX 3090.

**Data mix (flat):**

| Category | Sources | Target Tokens | % |
|----------|---------|---------------|---|
| General text | Wikipedia + Gutenberg (Common Pile) | ~5B | 50% |
| Technical Q&A | StackExchange (Common Pile) | ~2B | 20% |
| Code | The Stack V2 FOSS (Common Pile) | ~1B | 10% |
| Scientific | peS2o + arXiv (Common Pile) | ~1B | 10% |
| Linux specialization | Existing V2 Linux data | ~365M | 3.6% |
| Government/Legal | USGPO (Common Pile) | ~635M | 6.4% |

### Phase 2: Supervised Fine-Tuning

Teach flm who it is and how to behave as a helpful OS assistant.

```bash
python train_sft.py
```

- **Data**: 216,968 human-written samples from 4 DFSG sources (already built)
- **LR**: 3e-4, 2 epochs
- **Output**: `checkpoints/sft/best.pt`

### Phase 3: DPO Alignment

Direct Preference Optimization using human preference data.

```bash
python train_dpo.py
```

- **Data**: Audited preference pairs (CC-BY-4.0)
- **LR**: 1e-6, beta 0.5, 2 epochs
- **Output**: `checkpoints/dpo/best.pt` — the final model

## Training Data — 100% DFSG-Compliant, 100% Human-Written

~12B tokens across 14 sources, all explicitly licensed for redistribution and derivative works. No Common Crawl derivatives. No AI-generated content.

The build pipeline (`build_dataset.py`) can source data from Common Pile or direct downloads — currently using a mix of both. Future rebuilds will use Common Pile throughout for better deduplication.

### General Text (~10.3B tokens, 50% mix weight)

| Source | License | Size on Disk | Description |
|--------|---------|-------------|-------------|
| [Wikipedia English](https://dumps.wikimedia.org/) | CC-BY-SA | 19 GB | 5.9M articles, direct HF download |
| [Project Gutenberg](https://www.gutenberg.org/) | Public Domain | 14 GB | ~35K books, HF download |

### Technical Q&A (~1B tokens, 20% mix weight)

| Source | License | Size on Disk | Description |
|--------|---------|-------------|-------------|
| [Stack Exchange](https://archive.org/details/stackexchange) (all sites, score>=3) | CC-BY-SA | 3.2 GB | 2.9M posts, archive.org dumps |

### Code (~50M tokens, 10% mix weight)

| Source | License | Size on Disk | Description |
|--------|---------|-------------|-------------|
| Curated FOSS repos (coreutils, git, curl, etc.) | MIT/GPL/Apache | 162 MB | ~15K files from 27 repos |

### Scientific (~750M tokens, 10% mix weight)

| Source | License | Size on Disk | Description |
|--------|---------|-------------|-------------|
| [peS2o](https://huggingface.co/datasets/common-pile/peS2o) | CC (per-paper) | 1.2 GB | Scientific papers (Common Pile) |
| [arXiv papers](https://huggingface.co/datasets/common-pile/arxiv_papers) | CC (per-paper) | 1.3 GB | Scientific preprints (Common Pile) |

### Linux/Unix Specialization (~365M tokens, 3.6% mix weight)

| Source | License | Size on Disk | Description |
|--------|---------|-------------|-------------|
| [RFC documents](https://www.rfc-editor.org/) | IETF | 521 MB | 9,716 networking standards |
| Linux kernel source | GPL-2.0 | 289 MB | 16,690 files (key subsystems) |
| Debian man pages | GPL/BSD | 96 MB | 8,339 pages |
| [GNU Info manuals](https://www.gnu.org/manual/) | GFDL | 46 MB | 46 manuals |
| Linux kernel docs | GPL-2.0 | 34 MB | 4,721 documentation files |
| [TLDP](https://tldp.org/) guides | GFDL | 32 MB | 422 HOWTOs and guides |
| [Arch Wiki](https://wiki.archlinux.org/) | GFDL | 29 MB | 2,109 articles |

### Government/Legal (~390M tokens, 6.4% mix weight)

| Source | License | Size on Disk | Description |
|--------|---------|-------------|-------------|
| [USGPO](https://huggingface.co/datasets/common-pile/usgpo) | Public Domain | 1.3 GB | US government documents (Common Pile) |

### Why no Common Crawl?

Datasets like FineWeb-Edu, DCLM, C4, and RedPajama are filtered versions of Common Crawl. Common Crawl scrapes the web without per-page license verification. Most web pages are "all rights reserved" by default. For Debian's DFSG standard — which requires explicit permission to redistribute and create derivative works — this is insufficient.

### Why no AI-generated content?

Datasets like COSMOPEDIA (Mixtral-generated) and SmolTalk subsets (GPT-4, Llama) have ambiguous provenance. Even when the generating model has a permissive license, its training data may not. For Debian's bar, the chain of provenance needs to be clean all the way down.

## Success Criteria at 10B Tokens

| Metric | Target | V2 at 3.5B (493M) | V1 best (164M) |
|--------|--------|--------------------|-----------------|
| Loss | < 3.5 | 2.70 | — |
| Generated text | Coherent sentences | Word salad | — |
| LinuxBench | > 25% | — | 19% |
| PIQA | > 60% | — | 63.6% |
| ARC-Easy | > 28% | — | 25.3% |

Key test: If generated text at 10B tokens is coherent (not word salad), the architecture and data are working. Scale up to 100-200B tokens.

## Reproduce from Scratch

```bash
# 1. Setup
git clone https://github.com/ruapotato/chat_hamner.git
cd chat_hamner
pip install -r requirements.txt

# 2. Build dataset (downloads Common Pile sources + Linux data)
python build_dataset.py

# 3. Phase 1: Pretrain (~6 days on RTX 3090 for 10B tokens)
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

## Version History

### V3 (current) — SmolLM-135M Architecture, Common Pile Data

- 135M params (SmolLM-135M exact: 30 layers × 576 hidden, GQA 3:1)
- Common Pile data (safe subset) + existing Linux data
- 10B token test run first, then scale to 100-200B if coherent

### V2 (mothballed) — 493M Dense Transformer

- 493M params (26 layers × 1280 hidden)
- 12B tokens from 11 hand-curated sources
- Reached step 211K/610K (3.5B tokens), loss 2.70 but text still incoherent
- **Problem**: Way too few tokens for model size (needed ~500B+)

### V1 (archived) — Tournament of 10 Architectures

- 164M winner (Dense GQA, 20 layers × 1024 hidden)
- Used Common Crawl derivatives (not truly DFSG-compliant)
- 5-stage pipeline caused cumulative degradation (PIQA: 63.6% → 57.1%)
- LinuxBench 19% vs SmolLM-135M's 42%

## Key Lessons Learned

1. **Model size must match data budget** — V2's 493M model needed ~500B+ tokens but we only had ~12B. SmolLM-135M at 10B tokens is 74× Chinchilla-optimal.
2. **Deep & narrow beats wide & shallow** at small scale — SmolLM-135M (30×576) outperforms larger but shallower models (MobileLLM insight).
3. **Data provenance matters more than volume** — Common Pile provides pre-deduplicated, well-licensed data from diverse sources.
4. **Audit everything for contamination** — OASST2 had identity leaks, SmolTalk had GPT-4 outputs, CCCC had unreliable license detection.
5. **Common Crawl is not DFSG-compliant** — Web pages are "all rights reserved" by default.

## Project Structure

```
flm/
├── model.py                  # Core transformer architecture (135M, SmolLM-135M config)
├── chat.py                   # Interactive CLI chat
├── build_dataset.py          # Download + process Common Pile + Linux data
├── train_pretrain.py         # Phase 1: Pretraining (10B tokens, flat mix)
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
