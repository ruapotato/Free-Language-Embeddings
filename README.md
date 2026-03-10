# flm — The Free Language Model

> **Status: Active training (Concept Autoencoder V13).** Dual-decoder architecture compresses English into geometric concept vectors, then reconstructs both English and French from the same bottleneck — forcing language-independent meaning encoding.

A fully free AI project trained from scratch on a single RTX 3090. Every dataset DFSG-compliant, every weight reproducible. Built to be the first AI model you can `apt install` from Debian main.

**Free as in freedom** — the name is a direct reference to the Free Software Foundation's philosophy that software freedom is a matter of liberty, not price.

## The Goal

Build a geometric language representation — a 1024-dimensional space where meaning has shape. Sentences map to points, similar meanings cluster together, semantic relationships form consistent directions, and word-order changes that alter meaning are distinguishable. This representation becomes the input layer to a small language model (SLM) that reasons over geometry instead of raw tokens.

Right now the project is focused entirely on stage 1: getting the concept autoencoder to produce a space with good clustering, consistent directions, and real semantic structure. Stage 2 (the SLM that reasons in concept space) comes later.

## Concept Autoencoder — Current Architecture (V13)

### The Idea

Compress English into a 1024-dim geometric bottleneck, then decode it back to both English and French. The dual-decoder forces language-independent encoding — the French decoder can't rely on English surface tokens, so the concept space must encode actual meaning, not just token patterns.

### Why Dual-Decoder?

V10-V12 achieved good reconstruction but the concept space was essentially a bag-of-words token matcher — "the dog bit the man" and "the man bit the dog" had 0.97 cosine similarity. Adding French translation forces the bottleneck to encode meaning because:
- French has different word order (adjectives after nouns, etc.)
- Different vocabulary means no token-level shortcuts
- The concept space must be language-independent to serve both decoders

### Architecture (~82.8M params)

```
Encoder (EN) → 32×32 Bottleneck → EN Decoder (reconstruction)
                                 → FR Decoder (translation)
```

| Component | Details |
|-----------|---------|
| Encoder | 6 layers, 384 hidden, 6 heads, SwiGLU FFN |
| Bottleneck | 32 learned queries cross-attend to encoder, project to 32-dim each |
| EN Decoder | 6 parallel decoder blocks, cross-attends to concept stack |
| FR Decoder | 6 parallel decoder blocks, separate weights, cross-attends to same concepts |
| Concept space | 32 slots × 32 dims = 1024-dim representation |
| EN Tokenizer | BERT base uncased (30,522 vocab) |
| FR Tokenizer | CamemBERT (32,005 vocab) |
| Max sequence | 128 tokens |
| Positional encoding | RoPE |
| Normalization | RMSNorm |

### Training Data (DFSG-compliant)

3.2M English↔French translation pairs from three sources:

| Dataset | License | Pairs | Source |
|---------|---------|-------|--------|
| Europarl v7 EN-FR | Public domain | ~190K | European Parliament proceedings |
| Tatoeba EN-FR | CC-BY 2.0 | ~275K | Community-contributed translations |
| WikiMatrix EN-FR | CC-BY-SA | ~2.7M | Mined parallel sentences from Wikipedia |

### Training Setup

| Parameter | Value |
|-----------|-------|
| Batch size | 48 |
| Peak LR | 2e-4 (cosine decay) |
| Total steps | 600K |
| Loss | recon_loss + λ_fr × translation_loss |
| λ_fr | 1.0 |

### Training Progress

**[Training Dashboard (live charts)](https://ruapotato.github.io/chat_hamner/dashboard.html)** — interactive Chart.js dashboard with full reference guide, updated every push.

**Local live monitor:** `python web_dashboard.py` then open http://localhost:8501 (auto-refreshes every 30s)

### Geometry Probes

| Probe | What it measures |
|-------|-----------------|
| Analogy | Parallelogram completion (king-man+woman≈queen) |
| Cluster gap | Separation between same-meaning and different-meaning clusters |
| Direction consistency | Whether semantic directions are consistent across the space |
| Word-order sensitivity | Can the space distinguish "dog bit man" from "man bit dog"? |
| Effective rank | Dimensionality utilization (rank90, rank95) |

### Quick Start

```bash
# 1. Build translation pair datasets
python build_pairs.py

# 2. Train V13 dual-decoder concept autoencoder
python train_v13.py --fresh         # from scratch

# 3. Training dashboard (auto-detects latest version)
python web_dashboard.py
```

## Version History

### Concept Autoencoder V13 (current) — Dual-Decoder EN↔FR
- Dual decoder: EN reconstruction + FR translation from shared concept bottleneck
- Forces language-independent meaning encoding (FR can't exploit EN surface tokens)
- 82.8M params (54M shared encoder/bottleneck/EN-decoder + 28M FR decoder)
- 3.2M DFSG-compliant EN↔FR pairs (Europarl + Tatoeba + WikiMatrix)
- Non-autoregressive parallel decoder for both EN and FR
- V12 post-mortem: good reconstruction but bag-of-words — word order sensitivity too high (0.987)

### Concept Autoencoder V12 (archived) — Pure Reconstruction
- Same architecture as V10, trained longer
- Excellent reconstruction (96% token accuracy, 84% exact match)
- But concept space was a bag-of-words token matcher, not a meaning space

### Concept Autoencoder V11 (archived) — Pure Reconstruction on Diverse Data
- 600K steps on diverse English text
- Good reconstruction, geometry probes showed some structure

### Concept Autoencoder V10 (archived) — Parallel Decoder
- Replaced autoregressive decoder with non-autoregressive parallel decoder
- 6 parallel decoder blocks with learned position queries
- Cross-attention to concept vectors, all positions decoded simultaneously

### Concept Autoencoder V9 (archived) — Recon-Gated Slot Specialization
- Recon-gated geometry: geometry losses suppressed until reconstruction works
- Dynamic phases driven by recon quality, not fixed step counts
- Hard repulsion loss: targets worst-offending pairs for clustering gap

### Concept Autoencoder V8 (archived) — Staged Slot Specialization
- Three training phases: Foundation → Slot Focus → Joint Fine-tune
- Per-slot gradient scaling in Phase 2

### Concept Autoencoder V7 (archived) — Flat Cosine + Margin + Real Paraphrase
- Flat cosine similarity, margin losses, per-slot paraphrase loss on real data
- Gradient taper, batch repulsion loss

### Concept Autoencoder V6 (archived) — Detached Geometry
- Detached encoder/decoder training, per-slot classifiers
- 32/32 slot assignments correct within 1,000 steps

### Concept Autoencoder V5 (archived) — 32-Slot Supervised Concepts
- 54.3M params, 32×32 concept bottleneck, supervised slot isolation

### Concept Autoencoder V4 (archived) — Hard Negatives + Rank Pushing
### Concept Autoencoder V3 (archived) — Decorrelation Focus
### Concept Autoencoder V2 (archived) — Scheduled Weights + Word-Order
### Concept Autoencoder V1 (archived) — Baseline

### V3 (stopped) — SmolLM-135M, Common Pile Data
### V2 (mothballed) — 493M Dense Transformer
### V1 (archived) — Tournament of 10 Architectures

## Key Lessons Learned

1. **Next-token prediction at small scale needs enormous data** — 100B+ tokens for coherent output from a 135M model.
2. **Bottleneck forces information encoding** — reconstruction loss ensures the concept vectors actually capture meaning.
3. **Bag-of-words problem** — pure reconstruction doesn't force word-order sensitivity or deep meaning; a model can achieve 95%+ token accuracy with surface-level encoding.
4. **Dual-decoder forces meaning** — adding a cross-lingual decoder prevents surface-token shortcuts and forces the bottleneck to encode language-independent semantics.
5. **Multi-task loss imbalance** — easier tasks (same-language reconstruction) dominate shared representations; cross-lingual translation is slower but shapes better geometry.
6. **DFSG-compliant data is sufficient** — 3.2M parallel pairs from public domain and Creative Commons sources.

## Project Structure

```
flm/
├── concept_model.py          # Concept autoencoder (V13: 82.8M, dual decoder)
├── train_v13.py              # V13 training (dual-decoder EN→concepts→EN+FR)
├── train_v12.py              # V12 training (archived)
├── train_v11.py              # V11 training (archived)
├── train_v9.py               # V9 training (archived)
├── web_dashboard.py          # Live web dashboard (auto-refresh)
├── probe_geometry.py         # Probe concept space geometry
├── build_pairs.py            # Download DFSG paraphrase pair datasets
├── data/
│   ├── pairs/                # Translation pairs, paraphrases, etc.
│   └── concept_axes/         # Synthetic slot isolation data (32 slots)
├── checkpoints/              # Model checkpoints (gitignored)
├── docs/
│   └── dashboard.html        # Static dashboard export for GitHub Pages
└── logs/                     # Training logs and metrics CSVs
```

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.

Built by David Hamner with help from Claude.
