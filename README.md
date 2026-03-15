# flm — The Free Language Model

> **Status: V28/V29 — Word2vec experiments.** Exploring embedding geometry with classic skip-gram word2vec. V28 (300d) scores 43.9% on the Google analogy benchmark. V29 (3d) produces a directly-visualizable word globe — 0% on analogies but striking categorical structure.

A fully free AI project trained from scratch on a single RTX 3090. Every dataset DFSG-compliant, every weight reproducible. Built to be the first AI model you can `apt install` from Debian main.

**Free as in freedom** — the name is a direct reference to the Free Software Foundation's philosophy that software freedom is a matter of liberty, not price.

## The Big Idea

Traditional LLMs predict one token at a time. This project takes a different approach: **think in sentences, not tokens.**

### The Sentence2vec Insight

Word2vec learns word embeddings with beautiful geometric properties (king - man + woman = queen) from a simple task: predict a word from its neighbors. The geometry isn't explicitly trained — it *emerges* from the prediction task.

V25 applies the same principle one level up: **predict a sentence from its neighbors.** The encoder learns to organize sentence vectors so that next-sentence prediction works, which forces meaningful geometric structure into the concept space.

### Architecture

```
         Encoder + Bottleneck          Sentence Predictor         Decoder
┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│                          │    │                          │    │                          │
│ tokens → Encoder →       │    │  Causal Transformer      │    │       Concept vector →   │
│          Bottleneck →  ──┼──→ │  (predicts next sentence ┼──→ │       Parallel Decoder → │
│          concept vector  │    │   concept vector)        │    │       tokens             │
│                          │    │                          │    │                          │
└──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
              ↑                                                            │
              └────────────── reconstruction loss ─────────────────────────┘
                              (keeps bottleneck faithful)
```

**Everything trains jointly.** Both reconstruction and prediction losses backpropagate through the encoder. The encoder gets two signals:
1. **Reconstruction**: "your concept vectors must contain enough info to reconstruct the sentence"
2. **Prediction**: "organize your concept vectors so the next sentence is predictable from context"

Together, these force concept vectors that are both faithful to meaning AND geometrically structured.

### How Inference Works

A prompt like *"The sky is blue. Birds fly south in winter."* becomes:

1. Split into sentences: `["The sky is blue.", "Birds fly south in winter."]`
2. Each sentence → encoder → one concept vector each
3. Predictor receives 2 concept vectors, predicts concept vector #3
4. Concept vector #3 → decoder → output sentence
5. Encode output → feed back to predictor → predict next → repeat

### Why This Matters

- **No frozen phases.** Unlike the old two-phase approach, the encoder/predictor/decoder all train together. The concept space geometry is shaped by prediction from step 1.
- **Parallel token decoding.** The non-autoregressive decoder renders concept vectors to text in a single pass.
- **Prediction shapes geometry.** Just as word2vec's prediction task creates arithmetic-friendly word vectors, next-sentence prediction creates structured sentence vectors.
- **The predictor IS the language model.** No separate Phase 2 needed — after training, the predictor generates text by iteratively predicting next-sentence concept vectors.
- **Decoder-backprop loss.** Prediction loss uses token-level cross-entropy through the decoder (not MSE on vectors), following Meta's LCM finding that MSE produces blurry predictions.

Related work: Meta's Large Concept Model (LCM) and SONAR-LLM validate this direction. Our key difference is joint training — they freeze the sentence encoder first, we train everything together.

## Current Architecture (V25)

### Encoder/Decoder (from V24)

```
tokens → [4L Transformer Encoder] → [Cross-Attention Bottleneck] → concept vector → [4L Parallel Decoder] → tokens
```

- **Encoder**: 4-layer transformer, 256 hidden, 4 heads
- **Concept Bottleneck**: 64 learned queries compress encoder output into fixed-size vector (64 x 16 = 1024-dim)
- **Concept Whitening**: Cholesky whitening for disentangled dimensions
- **Parallel Decoder**: Non-autoregressive — all output tokens predicted simultaneously

### Sentence Predictor

```
concept vectors [sent 1, sent 2, ..., sent N] → [8L Causal Transformer] → predicted concept vector [sent N+1]
```

- **Input**: sequence of 1024-dim concept vectors (one per sentence)
- **Architecture**: 8-layer causal transformer with RoPE, SwiGLU, GQA
- **Output**: predicted next-sentence concept vector
- **Loss**: predicted vector → decoder → token-level cross-entropy against actual next sentence

### V25 Config

| Parameter | Value |
|-----------|-------|
| Total Parameters | 58.4M |
| Encoder/Decoder | 25.9M (4L, 256d, 4 heads, 1024 FFN) |
| Predictor | 32.5M (8L, 512d, 8 heads/4 KV, 2048 FFN) |
| Bottleneck | 64 concepts x 16 dims = 1024-dim |
| Tokenizer | bert-base-uncased (30,522 vocab) |
| Max tokens/sentence | 64 |
| Sentence window | 8 sentences |
| Batch size | 16 |
| Training data | Consecutive sentences from documents |
| Losses | 0.5 * reconstruction CE + prediction CE |

### Training Progress

**[Training Dashboard](https://ruapotato.github.io/chat_hamner/dashboard.html)** — interactive charts, updated every push.

**Local live monitor:** `python web_dashboard.py` then open http://localhost:8501

**Current status (298K / 600K steps, ~50%):**
- Reconstruction loss: 0.14 (nearly reconstructing full sentences)
- Prediction loss: 2.59 (steadily dropping)
- Next-sentence predictions show thematic coherence (e.g., water/temperature prompts → water/heat predictions)

**Concept space geometry at 298K steps:**
- Paraphrases cluster well (avg 0.90 cosine similarity)
- Semantic topic grouping emerging (within-group vs between-group gap: +0.35)
- Negation sensitivity developing ("he loves her" vs "he does not love her" = 0.04 cosine sim)
- Currently organizing primarily by surface features (length, punctuation, structure); semantic organization expected to strengthen as prediction loss drops
- Effective rank: 19 dims explain 90% of variance (space is well-spread, not collapsed)

## Version History

### V29 (current) — 3D Word2vec
- Pure 3D word2vec skip-gram: every word is a real (x, y, z) coordinate on a unit sphere
- 0.6M params, 100K whole-word vocabulary, 500K steps
- **[Interactive 3D Visualization](https://ruapotato.github.io/chat_hamner/probe_v29_3d.html)** — drag to orbit, hover for words, search by name
- Google Analogy Benchmark: **0.0%** — too many words packed onto a sphere for vector arithmetic to resolve
- Similarity gap: **+1.074** (excellent broad category separation: king/banana = -0.65, cat/computer = -0.86)
- Tense direction consistency: **0.932** (nearly perfect — go→went aligns with run→ran, see→saw, etc.)
- The model self-organized into a ~2D belt on the sphere (X-axis variance collapsed to 0.18 vs 0.55 for Y/Z)
- Key finding: 3 dimensions capture semantic *regions* (tech, nature, emotions occupy distinct areas) but cannot resolve individual words within a region

### V28 — Word2vec Skip-Gram (300d)
- Classic word2vec skip-gram with negative sampling, 300d, 100K whole-word vocabulary
- 60M params, 500K steps on OpenWebText subset (~2B tokens)
- **[Embedding Probe](https://ruapotato.github.io/chat_hamner/probe_w2v.html)** — analogies, vector arithmetic, t-SNE, benchmark comparison
- Google Analogy Benchmark: **43.9%** (semantic 27.3%, syntactic 52.7%, 80% coverage)
- Published comparison: word2vec (Google News) 61%, GloVe 75%, FastText 78% — gap explained by smaller training corpus
- king - man + woman = queen works (0.737 cosine), paris - france + germany = berlin (0.679)
- Custom analogies: 23/26 (88%)
- Key finding: whole-word vocabulary is critical — WordPiece subword tokenization completely breaks word2vec geometry

### V27 (archived) — Contrastive Autoencoder (SimCSE + VICReg)
- Added SimCSE contrastive loss with VICReg variance/covariance regularization to V24 autoencoder
- Fixed anisotropy (0.97→0.42) but failed to create real semantic structure
- Key finding: SimCSE with dropout augmentation only teaches noise invariance, not semantic similarity

### V26 (archived) — Masked Sentence Modeling
- Sentence-level masked prediction experiment

### V25 (archived) — Unified Sentence LM
- 58.4M params total, joint encoder/decoder + sentence predictor
- Sentence2vec approach: prediction task shapes concept space geometry
- Decoder-backprop loss (token-level CE, not MSE on vectors)
- Trains on consecutive sentences from documents
- At 50% training: surface features dominate geometry, semantic structure emerging

### V24 (archived) — Sentence Compressor
- 25.9M params, 4L encoder/decoder, 1024-dim bottleneck, max 64 tokens
- 100% exact match on test sentences at 250K steps
- Excellent reconstruction but flat concept space geometry (no useful arithmetic/interpolation)
- Key finding: reconstruction alone creates "swiss cheese" space — perfect compression but meaningless structure between encodings

### V22 (archived) — Scaled English-Only
- 248M params, 12L encoder, 8L decoder, bert-base-uncased
- Plateaued at ~6.0 — same padding mask bug as V21

### V21 (archived) — 4L Decoder Experiment
- Asymptoted at ~6.48, led to scaling attempts

### V13 (archived) — Dual-Decoder EN-FR
- 82.8M params, English reconstruction + French translation
- Forced language-independent encoding via cross-lingual decoder

### V10-V12 (archived) — Pure Reconstruction
- Good reconstruction (96% token accuracy) but bag-of-words concept space

### V1-V9 (archived) — Geometry Experiments
- Various approaches: supervised slots, decorrelation, margin losses, contrastive learning, WordNet structures, NLI graded losses
- Key finding: explicit geometry losses either destroy reconstruction or overfit to synthetic templates

## Key Lessons Learned

1. **Prediction creates geometry, reconstruction doesn't.** V24 achieved perfect reconstruction but flat geometry. Word2vec's insight applies: prediction tasks force meaningful structure.
2. **Joint training beats frozen phases.** The old plan (freeze encoder, then train LM) means the encoder never learns from the prediction task. Joint training lets prediction shape the concept space from step 1.
3. **Decoder-backprop beats MSE.** Meta's LCM found that MSE loss on concept vectors produces blurry/averaged predictions. Token-level CE through the decoder gives sharp gradients.
4. **Non-autoregressive decoding works.** Parallel decoders can reconstruct text from concept vectors — no sequential generation needed.
5. **Padding masks matter enormously.** The cross-attention bottleneck must properly mask padding tokens. This was the difference between 6.0 plateau and 0.01 loss.
6. **Small models can compress well.** 25.9M params outperforms 248M when the architecture is correct. Bottleneck capacity matters more than raw model size.
7. **Explicit geometry losses are a dead end.** 24 versions of experiments showed that contrastive, margin, decorrelation, and structural losses either break reconstruction or overfit to templates. The prediction task is what creates geometry naturally.

## Project Structure

```
flm/
├── train_v29.py              # V29: 3D word2vec skip-gram
├── train_v28.py              # V28: 300d word2vec skip-gram
├── probe_w2v.py              # V28 embedding probe + benchmark visualization
├── eval_analogy.py           # Google analogy benchmark runner
├── concept_model.py          # Model definitions (encoder, bottleneck, decoder, whitening)
├── train_v25.py              # V25: unified sentence compressor + LM training (archived)
├── train_v24.py              # V24: sentence compressor only (archived)
├── model.py                  # Transformer backbone (HamnerBlock, GQA, RoPE, SwiGLU)
├── web_dashboard.py          # Live training dashboard
├── docs/
│   ├── dashboard.html        # Static dashboard snapshot (GitHub Pages)
│   ├── probe_w2v.html        # V28 embedding geometry report
│   └── probe_v29_3d.html     # V29 interactive 3D word globe
├── data/
│   └── questions-words.txt   # Google analogy benchmark (19.5K questions)
├── checkpoints/              # Model checkpoints (gitignored)
└── logs/                     # Training metrics CSVs
```

## Quick Start

```bash
# Train V25 (unified sentence LM) from scratch
python train_v25.py --fresh

# Or initialize encoder/decoder from a V24 checkpoint
python train_v25.py --v24-init checkpoints/concept_v24/step_250000.pt

# Monitor training
python web_dashboard.py        # http://localhost:8501
```

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.

Built by David Hamner with help from Claude.
