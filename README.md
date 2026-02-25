# flm — The Free Language Model

A fully free AI assistant trained from scratch on a single RTX 3090. 164M parameters, every dataset DFSG-compliant, every weight reproducible. Built to be the first AI model you can `apt install` from Debian main.

**Free as in freedom** — the name is a direct reference to the Free Software Foundation's philosophy that software freedom is a matter of liberty, not price. flm proves that a fully free, fully reproducible language model can exist and be useful.

## What flm Is

- **An OS-aware assistant** that becomes an expert on whatever operating system it's built on
- **164M parameters** — small, honest about its limits, focused on being useful
- **Trained on a single GPU** from random weights — no pretrained models, no distillation
- **Fully reproducible** — one person, one RTX 3090, from zero to working model
- **GPL-3.0** for weights, training code, and all custom data

flm is NOT a general-purpose chatbot competing with GPT-4. It's a focused, knowledgeable OS assistant that happens to be fully free software.

## Architecture

164M parameter dense transformer, selected through a [tournament of 10 competing designs](#tournament-architecture-search):

| Parameter | Value |
|-----------|-------|
| Hidden size | 768 |
| Layers | 20 |
| Attention heads | 12 (4 KV, GQA 3:1) |
| MLP | Dense SwiGLU |
| Intermediate size | 2048 |
| Parameters | 163.6M |
| Sequence length | 1024 |
| Tokenizer | cosmo2-tokenizer (49,152 vocab) |

Key components: RMSNorm, SwiGLU, RoPE, GQA, fp16 mixed precision, torch.compile, gradient checkpointing.

## Training Pipeline

Inspired by [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): multi-stage curriculum on a single RTX 3090 with exclusively DFSG-compliant data.

### Stage 1: Base Pretraining (~4.5 days)

Foundational English comprehension, general knowledge, basic reasoning.

```bash
python train_pretrain.py --fresh
```

- **Data**: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 60% + [DCLM](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) 40%
- **Steps**: 400,000 (~10B tokens)
- **LR**: 2e-4 with Warmup-Stable-Decay schedule
- **Output**: `checkpoints/pretrain_v4/latest.pt`

### Stage 2: Annealing (~32 hours)

SmolLM2-style annealing with curated high-quality data including OS-specific knowledge. Decaying LR settles the model while injecting domain expertise.

```bash
python train_pretrain.py --stage anneal
```

- **Base**: Stage 1 checkpoint
- **Data**: Curated mix of documentation, code, math, troubleshooting, and general retention:

| Source | Share | License | Description |
|--------|-------|---------|-------------|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 20% | ODC-BY | High-quality educational web text |
| [Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | 10% | Apache-2.0 | Synthetic textbooks |
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | 10% | CC-BY-SA-4.0 | Factual knowledge |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | 10% | ODC-BY | Math reasoning |
| [Ubuntu Dialogue Corpus](https://huggingface.co/datasets/ubuntu-dialogs-corpus/ubuntu_dialogs_corpus) | 10% | CC-BY | 1M Linux troubleshooting conversations |
| Debian documentation | 20% | DFSG-free | Man pages, Policy Manual, packaging metadata |
| Debian source code | 10% | DFSG-free | dpkg, apt, systemd, coreutils (Shell/Python/C/Make) |
| [DCLM](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | 10% | ODC-BY | General web text retention |

- **Steps**: ~122,000 (~3B tokens)
- **Output**: `checkpoints/pretrain_v4_anneal/latest.pt`

### Stage 3: Chat Pretraining (~11 hours)

Teach conversational structure with OS-relevant dialogue patterns.

```bash
python train_chat_pretrain.py
```

- **Base**: Stage 2 checkpoint
- **Data**: [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) 30% + Synthetic OS Q&A 30% + FineWeb-Edu 15% + DCLM 10% + Mailing list conversations 10% + Synthetic tasks 5%
- **Steps**: ~40,000 (~1B tokens)
- **Output**: `checkpoints/chat_pretrain/latest.pt`

### Stage 4: Supervised Fine-Tuning (~3 hours)

Teach flm who it is and how to behave as a helpful OS assistant.

```bash
python prepare_sft_data.py
python train_sft.py
```

- **Base**: Stage 3 checkpoint
- **Data**: 12,000-22,000 conversations across identity, OS administration, packaging, debugging, and general helpfulness
- **Epochs**: 3 with early stopping
- **Output**: `checkpoints/sft/best.pt`

### Stage 5: DPO Alignment (~1-2 hours)

Direct Preference Optimization using human preference data.

```bash
python train_dpo.py
```

- **Base**: Stage 4 checkpoint
- **Data**: [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) (~7-10k preference pairs, CC-BY-4.0)
- **Output**: `checkpoints/dpo/best.pt` — the final model

## Training Data

Every dataset is DFSG-compliant. No NonCommercial clauses. No OpenAI-generated synthetic data. No unclear provenance.

| Dataset | License | Stage | Description |
|---------|---------|-------|-------------|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | ODC-BY | 1, 2, 3 | Educational web text |
| [DCLM](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | ODC-BY | 1, 2, 3 | General web text |
| [Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | Apache-2.0 | 2 | Synthetic textbooks |
| [Wikipedia (en)](https://huggingface.co/datasets/wikimedia/wikipedia) | CC-BY-SA-4.0 | 2 | Encyclopedic knowledge |
| [Ubuntu Dialogue Corpus](https://huggingface.co/datasets/ubuntu-dialogs-corpus/ubuntu_dialogs_corpus) | CC-BY | 2, 3 | 1M Linux troubleshooting conversations |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | ODC-BY | 2 | Math reasoning |
| Debian documentation | DFSG-free (various) | 2 | Man pages, Policy Manual, packaging metadata, copyright |
| Debian source code | DFSG-free (various) | 2 | dpkg, apt, systemd, coreutils (Shell/Python/C/Make) |
| [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | Apache-2.0 | 3, 4 | Conversational structure |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | CC-BY-4.0 | 5 | Human-annotated preferences |
| [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache-2.0 | 3, 4 | Human-written conversations |
| Custom SFT data (flm identity, OS Q&A) | GPL-3.0 | 4 | Identity, packaging, sysadmin help |
| Synthetic tasks | GPL-3.0 | 3 | Arithmetic, sorting, reasoning |

## Reproduce from Scratch

```bash
# 1. Setup
git clone https://github.com/ruapotato/chat_hamner.git
cd chat_hamner
pip install -r requirements.txt

# 2. Prepare OS-specific data (detects OS from /etc/os-release)
python prepare_debian_data.py     # documentation, man pages, packaging metadata

# 3. Prepare other training data
python prepare_sft_data.py        # downloads SmolTalk, generates flm SFT data
python prepare_dpo_data.py        # downloads HelpSteer2 preference pairs

# 4. Stage 1: Base pretrain (~4.5 days on RTX 3090)
python train_pretrain.py --fresh

# 5. Stage 2: Anneal with OS knowledge + curated mix (~32 hours)
python train_pretrain.py --stage anneal

# 6. Stage 3: Chat pretrain (~11 hours)
python train_chat_pretrain.py

# 7. Stage 4: SFT — teach flm who it is (~3 hours)
python train_sft.py

# 8. Stage 5: DPO alignment (~1-2 hours)
python train_dpo.py

# 9. Chat!
python chat.py
```

**Prerequisites**: NVIDIA GPU with 24GB+ VRAM, 32GB+ RAM, Python 3.10+, CUDA 12.x, ~50GB disk.

## The Bug That Changed Everything

The original pretraining ran for 122k steps (~2 days, 4.4B tokens) but **never produced coherent text**.

**Root cause**: A double-shift label bug. The model's `forward()` shifts labels internally, but every training script ALSO pre-shifted — so the model learned to predict **2 tokens ahead** instead of 1.

```python
# BUG — every training script had this:
input_ids = tokens[:-1]   # shifted once in data prep
labels = tokens[1:]       # shifted again → model predicts 2-ahead!

# FIX — model.forward() handles the shift internally:
input_ids = tokens[:seq_len]
labels = tokens[:seq_len]  # same tensor, no pre-shifting
```

## Key Lessons Learned

1. **SFT data must be diverse** — Tech-only conversations produce a model that can only talk about programming. Diversity is essential.
2. **Chat pretraining bridges the gap** — Going straight from base pretrain to SFT doesn't work well. An intermediate dialogue-format stage teaches conversational structure.
3. **SFT overfits fast** — Best results at epoch 2-3. By epoch 9+, the model memorizes response templates.
4. **Math requires exhaustive examples** — The model memorizes number pairs, not arithmetic. ALL multiplication tables 2-12 were necessary.
5. **Documentation > source code** — For an OS assistant, man pages and packaging metadata are far more valuable per token than bulk source code.

## Tournament Architecture Search

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

## Project Structure

```
flm/
├── model.py                  # Core transformer architecture (164M)
├── chat.py                   # Interactive CLI chat
├── train_pretrain.py         # Stages 1-2: Base pretraining + annealing
├── train_chat_pretrain.py    # Stage 3: Chat pretraining (dialogue mix)
├── train_sft.py              # Stage 4: Supervised fine-tuning
├── train_dpo.py              # Stage 5: DPO alignment
├── prepare_debian_data.py    # Collect OS documentation, man pages, packaging metadata
├── prepare_sft_data.py       # Download/prepare SFT data
├── prepare_dpo_data.py       # Download/prepare HelpSteer2 DPO pairs
├── generate_sft_data.py      # Generate flm identity + OS Q&A SFT data
├── synthetic_tasks.py        # Synthetic task generators
├── plot_training.py          # Training metrics visualization
├── data/
│   ├── os_specific/          # Debian docs, man pages, packaging metadata
│   ├── ubuntu_dialogue.jsonl # Ubuntu IRC troubleshooting conversations
│   └── sft/                  # SFT conversation data
├── checkpoints/              # Model checkpoints for each stage
└── logs/                     # Training logs, metrics, sample generations
```

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.

Built by David Hamner with help from Claude.
