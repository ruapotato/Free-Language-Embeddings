# flm — The Free Language Model

> **Status: Active training (Concept Autoencoder V9).** Training a concept autoencoder that compresses language into geometric concept vectors — a bottleneck where meaning determines position, not surface form.

A fully free AI project trained from scratch on a single RTX 3090. Every dataset DFSG-compliant, every weight reproducible. Built to be the first AI model you can `apt install` from Debian main.

**Free as in freedom** — the name is a direct reference to the Free Software Foundation's philosophy that software freedom is a matter of liberty, not price.

## Concept Autoencoder — Current Architecture

### The Idea

Instead of predicting tokens, encode language into a compressed geometric space where:
- **Each slot encodes a specific concept family** (size, color, tense, sentiment, etc.)
- **Paraphrases** map to nearby vectors (same meaning = close)
- **Unrelated sentences** map far apart
- **Word-order changes** that alter meaning are distinguishable

A stage 2 model (future) can then reason purely in concept space, never touching raw language.

### Architecture (~54.3M params)

Encoder (bidirectional) -> 32x32 Bottleneck -> Decoder (autoregressive)

| Component | Details |
|-----------|---------|
| Encoder | 6 layers, 384 hidden, 6 heads, SwiGLU FFN |
| Bottleneck | 32 learned queries cross-attend to encoder, project to 32-dim each |
| Decoder | 6 layers, 384 hidden, 6 heads, cross-attends to concept stack |
| Concept space | 32 slots x 32 dims = 1024-dim representation |
| Tokenizer | BERT base uncased (30,522 vocab) |
| Max sequence | 128 tokens |
| Positional encoding | RoPE |
| Normalization | RMSNorm |

### The 32 Concept Slots

Each slot owns one semantic family. Synthetic training data teaches the model which slot encodes what:

| Slot | Concept | Slot | Concept |
|------|---------|------|---------|
| 0 | Subject/Entity | 16 | Distance/Proximity |
| 1 | Object/Patient | 17 | Tense/Aspect |
| 2 | Animacy/Gender | 18 | Duration/Frequency |
| 3 | Age/Life Stage | 19 | Time Reference |
| 4 | Size/Scale | 20 | Number/Amount |
| 5 | Color/Brightness | 21 | Degree/Comparison |
| 6 | Shape/Form | 22 | Core Sentiment |
| 7 | Material/Texture | 23 | Specific Emotion |
| 8 | Weight/Density | 24 | Arousal/Energy |
| 9 | Temperature/Weather | 25 | Quality/Value |
| 10 | Action Type | 26 | Difficulty/Importance |
| 11 | Manner/Intensity | 27 | Negation/Truth |
| 12 | Speed/Completion | 28 | Certainty/Obligation |
| 13 | Direction/Path | 29 | Causation/Condition |
| 14 | Location/Scene | 30 | Formality/Register |
| 15 | Spatial Relations | 31 | Speech Act/Intent |

### Training Losses (V9)

V9 uses V8's 15 losses with **recon-gated geometry** and **hard repulsion** for clustering.

V8 post-mortem: slot structure excellent (31/32 assigned, iso=0.554) but reconstruction was garbage (gibberish output despite 0.55 teacher-forced CE). Clustering gap still tiny (+0.018). Geometry pushed vectors around but decoder couldn't use them — concept space may not have encoded real information.

V9 fixes with recon-gated dynamic phases:

**Phase 1 — Recon Focus** (until recon < 0.1): Geometry losses suppressed by a gate (0→1 as recon drops from 0.5→0.1). Decoder gets 3x weight boost. Full gradient taper (1.0) through bottleneck so encoder learns representations the decoder can use.

**Phase 2 — Slot Focus** (32 x 500 steps, starts when recon is good): Cycles through each slot with per-slot gradient scaling (focus=1.0, others=0.05). Starts automatically when the recon gate opens.

**Phase 3 — Joint Fine-tune** (remaining steps): All slots at full gradient, balanced refinement.

**Hard repulsion** (V9): Targets the top-k most similar non-matching pairs per sample, not just the batch mean. This focuses gradient on the worst offenders — the pairs that keep the clustering gap small.

**Recon gate**: If reconstruction drifts bad during Phase 2/3, the gate automatically reduces geometry influence. Ensures concept vectors always encode meaningful information.

**Per-slot stats**: Dashboard shows per-slot isolation bar chart, metric glossary, and geo gate on recon chart.

### Training Data (DFSG-compliant)

| Dataset | License | Pairs | Use |
|---------|---------|-------|-----|
| Synthetic concept axes | Generated | 2.15M | Slot isolation + classification |
| ParaNMT | CC-BY | ~5M | Paraphrase pairs |
| PAWS | Apache 2.0 | 108K | Hard paraphrase pairs |
| QQP | CC | 400K | Question paraphrases |
| Tatoeba | CC-BY | 350K | Cross-lingual pairs |

### Training Progress (Concept Autoencoder V9)

V9 uses recon-gated slot specialization with dynamic phases.

**Live Dashboard:** `python web_dashboard.py` then open http://localhost:8501

### Quick Start

```bash
# 1. Generate synthetic concept axis data
python generate_concept_data.py

# 2. Build paraphrase pair datasets
python build_pairs.py

# 3. Train concept autoencoder
python train_v9.py --fresh         # from scratch

# 4. Training dashboard (auto-detects latest version)
python web_dashboard.py
```

## Version History

### Concept Autoencoder V9 (current) — Recon-Gated Slot Specialization
- Recon-gated geometry: geometry losses suppressed until reconstruction works (recon < 0.1)
- Dynamic phases driven by recon quality, not fixed step counts
- Recon boost (3x decoder weight) during Phase 1 for faster convergence
- Hard repulsion loss: targets worst-offending pairs for clustering gap
- Full gradient taper (1.0) through bottleneck in Phase 1
- Dashboard: metric glossary, geo gate chart, hard repulsion tracking
- V8 post-mortem: slots worked (31/32) but reconstruction was garbage, clustering gap still tiny

### Concept Autoencoder V8 (archived) — Staged Slot Specialization
- Three training phases: Foundation → Slot Focus → Joint Fine-tune
- Per-slot gradient scaling (soft guidance, not hard freezing) in Phase 2
- Per-slot isolation stats on dashboard (bar chart with assignment status)
- Stronger margin targets: para>0.92, neg<0.2, WO<0.4, repulsion<0.05
- Much higher margin weights (5-8x vs V7's 1-1.5)
- Lower NCE temperature (0.04 vs V7's 0.07) for sharper discrimination
- V7 post-mortem: slot structure worked (28/32) but global similarity compressed into narrow band

### Concept Autoencoder V7 (archived) — Flat Cosine + Margin + Real Paraphrase
- Flat cosine similarity replaces mean-of-per-slot-cosines (eliminates dead-slot floor)
- Margin losses: explicit absolute similarity targets (para>0.85, neg<0.3, WO<0.5)
- Per-slot paraphrase loss on REAL data (bridges synthetic→real gap)
- Gradient taper: 100% at decoder, 30% at bottleneck, 5% at early encoder (replaces blunt 10% RECON_LEAK)
- Batch repulsion loss prevents clustering collapse (pushes random text sim < 0.3)
- 14 losses total: 12 encoder geometry + 2 decoder reconstruction

### Concept Autoencoder V6 (archived) — Detached Geometry
- Detached encoder/decoder training: geometry gradients to encoder only, recon to decoder (10% leak)
- Per-slot classifiers: auxiliary heads classify concept_value from slot vectors
- Per-slot contrastive: shapes within-slot geometry (same value → close, different → far)
- Cross-reconstruction: encode A, decode toward paraphrase B
- 32/32 slot assignments correct within 1,000 steps (V5 had ~3/10)
- Batch similarities flat at p_sim=0.69, n_sim=0.006 — dead-slot similarity floor

### Concept Autoencoder V5 (archived) — 32-Slot Supervised Concepts
- 54.3M param encoder-decoder with 32x32 concept bottleneck
- Each slot assigned a specific concept family (size, color, tense, sentiment, etc.)
- 2.15M synthetic sentence pairs for slot isolation training
- Slot isolation loss teaches which slot encodes which concept
- Good reconstruction and rank, but slots didn't learn assigned concepts

### Concept Autoencoder V4 (archived) — Hard Negatives + Rank Pushing
- 8x128 bottleneck, hard negative InfoNCE, spectral spread / per-dim variance
- Rank plateaued at ~76-83 despite multiple rank-pushing approaches
- Demonstrated that unsupervised rank expansion has limits

### Concept Autoencoder V3 (archived) — Decorrelation Focus
- Added slot decorrelation and spectral spread losses
- Rank plateaued at 57-58; geometry probing showed lookup-table behavior

### Concept Autoencoder V2 (archived) — Scheduled Weights + Word-Order
- Minimal 2-token swap word-order contrastive loss
- Hard phase-based loss scheduling

### Concept Autoencoder V1 (archived) — Baseline
- Same architecture, no word-order loss, static weights

### V3 (stopped) — SmolLM-135M, Common Pile Data
- 135M params, reached loss 2.67 at 1.23B tokens

### V2 (mothballed) — 493M Dense Transformer
- 493M params, reached loss 2.70 at 3.5B tokens

### V1 (archived) — Tournament of 10 Architectures
- 164M winner, 9.8B tokens, Common Crawl (not DFSG-compliant)

## Key Lessons Learned

1. **Next-token prediction at small scale needs enormous data** — 100B+ tokens for coherent output from a 135M model.
2. **Bottleneck forces information encoding** — reconstruction loss ensures the concept vectors actually capture meaning, not just cluster statistics.
3. **Unsupervised rank expansion plateaus** — spectral spread (SVD sample cap), per-dim variance (scale shortcut), decorrelation (slot-level only) all plateau around rank 60-83. Supervised slot assignment may break through.
4. **Hard negatives beat random negatives** — NLI contradictions and PAWS adversarial pairs force finer semantic discrimination.
5. **Smooth weight scheduling beats hard phases** — continuous `1/(1+recon_loss)` ramp avoids phase-boundary instability.
6. **Full-shuffle word-order is too easy** — minimal 2-token swap provides sustained gradient.

## Project Structure

```
flm/
├── concept_model.py          # Concept autoencoder (54.3M, encoder-decoder)
├── train_v9.py               # V9 training (recon-gated slot specialization)
├── train_v8.py               # V8 training (archived)
├── train_v7.py               # V7 training (archived)
├── train_v6.py               # V6 training (archived)
├── train_concept.py          # V5 training (archived)
├── generate_concept_data.py  # Synthetic concept axis dataset generator
├── plot_concepts.py          # UMAP concept space visualization
├── plot_training.py          # Training dashboard (static PNG)
├── web_dashboard.py          # Live web dashboard (auto-refresh)
├── probe_geometry.py         # Probe concept space geometry
├── probe_concepts.py         # Interactive concept probing
├── build_pairs.py            # Download DFSG paraphrase pair datasets
├── data/
│   ├── pairs/                # Paraphrase, hard negative, STS pairs
│   └── concept_axes/         # Synthetic slot isolation data (32 slots)
├── checkpoints/              # Model checkpoints (gitignored)
└── logs/                     # Training logs and plots
    ├── concept_v5.log        # Current training log
    └── plots/                # Generated dashboards
```

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.

Built by David Hamner with help from Claude.
