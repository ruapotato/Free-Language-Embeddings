# flm — The Free Language Model

> **Status: Active training (Concept Autoencoder V24).** A sentence compressor that learns to compress arbitrary-length text into a dense concept vector and perfectly reconstruct it — the foundation for an LM that thinks in sentences, not tokens.

A fully free AI project trained from scratch on a single RTX 3090. Every dataset DFSG-compliant, every weight reproducible. Built to be the first AI model you can `apt install` from Debian main.

**Free as in freedom** — the name is a direct reference to the Free Software Foundation's philosophy that software freedom is a matter of liberty, not price.

## The Big Idea

Traditional LLMs predict one token at a time. This project takes a different approach: build two models that work together so the language model **never touches tokens at all**.

### The Two-Model Architecture

```
         Model 1 (frozen)                    Model 2                    Model 1 (frozen)
┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│                          │    │                          │    │                          │
│ tokens → Encoder →       │    │                          │    │       → Bottleneck →     │
│          Bottleneck →  ──┼──→ │  Concept-Space LM  ──────┼──→ │         Parallel    →    │
│          concept vectors │    │  (sentence predictor)    │    │         Decoder → tokens  │
│                          │    │                          │    │                          │
└──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
```

**Model 1 — Sentence Compressor** (Phase 1, in progress): A non-autoregressive encoder-decoder that compresses text into fixed-size concept vectors and reconstructs it perfectly. Once trained, its weights are **frozen** and it becomes a fixed translation layer between token-space and concept-space.

**Model 2 — Concept-Space LM** (Phase 2, planned): An autoregressive language model that operates **entirely in concept space**. It predicts the next *sentence* (as a concept vector), not the next *token*. Model 1 is strapped onto the front and back as frozen bookends — encoding input text to concepts on the way in, decoding predicted concepts back to text on the way out.

### How Inference Works

A prompt like *"The sky is blue. Birds fly south in winter."* becomes:

1. Split into sentences: `["The sky is blue.", "Birds fly south in winter."]`
2. Each sentence → frozen encoder → one concept vector each
3. LM receives 2 concept vectors, predicts concept vector #3
4. Concept vector #3 → frozen decoder → output sentence

**The LM never sees tokens.** It operates entirely on concept vectors, one per sentence.

### Why This Matters

- **The LM never sees tokens.** It thinks in sentence-level meaning representations. Each prediction step covers an entire sentence worth of semantics.
- **Parallel token decoding.** Model 1's non-autoregressive decoder renders concept vectors to text in a single pass — no sequential token-by-token generation.
- **Forced semantic compression.** The tight bottleneck can't encode surface-level token patterns. The concept space must capture actual meaning to achieve perfect reconstruction.
- **Hierarchical generation.** The LM plans at the sentence/concept level for long-range coherence, then Model 1 handles the low-level rendering.
- **Training data = individual sentences.** Model 1 trains on one sentence per sample (not random document chunks) because that's exactly how it will be used at inference time.

Related work (SONAR-LLM, Latent Reasoning via Sentence Embedding Prediction) validates this general direction but uses autoregressive decoders. Our parallel decoder should be faster.

## Phase 1: Sentence Compressor — Current Architecture (V24)

### How It Works

```
tokens → [Transformer Encoder] → [Cross-Attention Bottleneck] → concept vector → [Parallel Decoder] → tokens
```

- **Encoder**: 4-layer transformer processes input tokens
- **Concept Bottleneck**: 64 learned cross-attention queries compress the encoder output into a fixed-size concept vector (64 vectors x 16 dims = 1024-dim)
- **Concept Whitening**: ZCA whitening + learned rotation for disentangled concept dimensions
- **Parallel Decoder**: Non-autoregressive — all output tokens predicted simultaneously from the concept vector

The decoder is **not autoregressive**. Every output token is predicted in parallel from the concept representation alone. This forces the bottleneck to fully capture the sentence's meaning.

### V24 Config

| Parameter | Value |
|-----------|-------|
| Parameters | 25.9M |
| Encoder | 4 layers, 256 hidden, 4 heads, 1024 FFN |
| Decoder | 4 layers, 256 hidden, 4 heads, 1024 FFN |
| Bottleneck | 64 concepts x 16 dims = 1024-dim |
| Tokenizer | bert-base-uncased (30,522 vocab, English) |
| Max sequence | 64 tokens |
| Batch size | 256 |
| Throughput | ~134K tok/s |
| Training data | Individual sentences (not document chunks) |

### Current Results

96% token accuracy, 62% exact match at 79K steps — loss 0.03 and dropping. The model reconstructs most sentences perfectly from concept vectors alone.

### Key Breakthroughs

1. **Padding mask fix**: V21/V22 plateaued at loss ~6.0 because the concept bottleneck's cross-attention attended to padding tokens. Fixing this unlocked real learning.
2. **Sentence-level training**: Each sample is one sentence — because at inference time, a prompt gets split into sentences and each becomes one concept vector for the LM. Training on random document chunks was meaningless for this use case.
3. **Cholesky whitening**: ZCA whitening via eigendecomposition (`eigh`) kept crashing on ill-conditioned covariance matrices. Replaced with Cholesky-based whitening which is numerically stable.

### Training Progress

**[Training Dashboard](https://ruapotato.github.io/chat_hamner_v2/dashboard.html)** — interactive charts, updated every push.

**Local live monitor:** `python web_dashboard.py` then open http://localhost:8501

## Phase 2: Concept-Space LM (In Development)

Once Model 1 achieves near-perfect reconstruction (loss near 0, high exact-match accuracy):

1. **Freeze** Model 1's encoder, bottleneck, and decoder weights
2. **Build Model 2**: a block-causal transformer LM that operates directly on concept slots
3. **Wire it up**: `text → [frozen encoder+bottleneck] → concept slots → [Model 2 LM] → predicted slots → [frozen bottleneck+decoder] → text`

### Block-Causal Architecture

The key insight: don't flatten the 64 concept slots into a single vector. Keep them as individual positions so they can **attend to each other** — like words in a sentence.

```
Sentence 1:  [slot_1 slot_2 ... slot_64]  ←→ bidirectional within block
Sentence 2:  [slot_1 slot_2 ... slot_64]  ←→ bidirectional, causal from sent 1
Sentence 3:  [slot_1 slot_2 ... slot_64]  ←→ bidirectional, causal from sent 1-2
```

- **Within a sentence**: concept slots attend bidirectionally (slots refine each other)
- **Across sentences**: causal attention (autoregressive at the sentence level)
- **Position encoding**: slot-level RoPE (repeating per sentence) + learned sentence embeddings

### Why Block-Causal

The LM outputs 64 concept slots for the next sentence and feeds them **directly back in** as input. No decode-to-text, no re-encode. V24's encoder/decoder only touch the edges:

```
[prompt text] → V24 encode → concept slots → LM → LM → LM → concept slots → V24 decode → [output text]
                                                ↑___________|
                                             (slots feed back directly,
                                              no V24 roundtrip needed)
```

### Concept LM Config

| Parameter | Value |
|-----------|-------|
| Parameters | 63M |
| Layers | 16 |
| Hidden | 512 |
| Heads | 8 (4 KV) |
| FFN | 2048 (SwiGLU) |
| Slots per sentence | 64 x 16d |
| Max sentences | 32 |
| Loss | MSE on predicted vs actual concept slots |

## Version History

### V24 (current) — Sentence Compressor
- 25.9M params, 4L encoder/decoder, 1024-dim bottleneck, max 64 tokens
- Trains on individual sentences (not document chunks)
- Fixed padding mask bug, switched to Cholesky whitening for stability
- 96% token accuracy, 62% exact match at 79K steps

### V22 (archived) — Scaled English-Only
- 248M params, 12L encoder, 8L decoder, bert-base-uncased
- Plateaued at ~6.0 — same padding mask bug as V21

### V21 (archived) — 4L Decoder Experiment
- Asymptoted at ~6.48, led to scaling attempts

### V13 (archived) — Dual-Decoder EN↔FR
- 82.8M params, English reconstruction + French translation
- Forced language-independent encoding via cross-lingual decoder

### V10-V12 (archived) — Pure Reconstruction
- Good reconstruction (96% token accuracy) but bag-of-words concept space

### V1-V9 (archived) — Geometry Experiments
- Various approaches to concept space structure: supervised slots, decorrelation, margin losses, staged training

## Key Lessons Learned

1. **Non-autoregressive decoding works** — parallel decoders can reconstruct text from concept vectors, no sequential generation needed.
2. **Padding masks matter enormously** — the cross-attention bottleneck must properly mask padding tokens or concept vectors get corrupted. This was the difference between 6.0 plateau and 0.16 loss.
3. **Small models can compress well** — 25.9M params outperforms 248M when the architecture bug is fixed. The bottleneck capacity matters more than raw model size.
4. **Geometry may not need explicit losses** — V24 dropped all geometry losses and just optimizes reconstruction. Whether the resulting concept space has useful geometric properties will be tested when we build Model 2.
5. **Contrastive learning from scratch doesn't work** — SimCSE-style training trivially achieves near-zero loss because random encoders already produce distinguishable embeddings for different texts.

## Project Structure

```
flm/
├── concept_model.py          # Model definitions (encoder, bottleneck, decoder, whitening)
├── train_v24.py              # Phase 1: sentence compressor training
├── train_concept_lm.py       # Phase 2: block-causal concept-space LM training
├── model.py                  # Transformer backbone (HamnerBlock, GQA, RoPE, SwiGLU)
├── web_dashboard.py          # Live training dashboard
├── docs/
│   └── dashboard.html        # Static dashboard snapshot (GitHub Pages)
├── checkpoints/              # Model checkpoints (gitignored)
└── logs/                     # Training metrics CSVs
```

## Quick Start

```bash
# Phase 1: Train V24 sentence compressor
python train_v24.py --fresh

# Phase 2: Train concept-space LM (requires trained V24 checkpoint)
python train_concept_lm.py --fresh

# Monitor training
python web_dashboard.py        # http://localhost:8501
```

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.

Built by David Hamner with help from Claude.
