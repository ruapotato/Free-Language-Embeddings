# flm — The Free Language Model

> **Status: Active development (V4).** Fundamental architectural pivot — instead of scaling next-token prediction, V4 explores encoding language into unambiguous concept geometry. A vector space where meaning determines position, not surface form.

A fully free AI project trained from scratch on a single RTX 3090. Every dataset DFSG-compliant, every weight reproducible. Built to be the first AI model you can `apt install` from Debian main.

**Free as in freedom** — the name is a direct reference to the Free Software Foundation's philosophy that software freedom is a matter of liberty, not price.

## V4 — Concept Vector Architecture

### The Problem with Next-Token Prediction at Small Scale

V1 through V3 all tried to train decoder-only transformers on next-token prediction. Even with the right architecture (V3's SmolLM-135M) and good data (Common Pile + Linux), the model produces grammatical but semantically incoherent text at 1.2B tokens. Usable output likely requires ~100B+ tokens — months on a single RTX 3090.

The deeper issue: current language models encode language ambiguously. These two sentences:
- "the massive cat stepped on the rug"
- "there was a rug that a massive cat stepped on"

...produce different internal representations despite being semantically identical. Surface form, syntax, and word order all pollute the representation. A small model wastes most of its capacity encoding these irrelevant variations.

### The Concept Vector Approach

V4 explores a three-model pipeline:

1. **Model 1 — Semantic Encoder** (current focus): Text in, concept vector out. Paraphrases and translations of the same idea produce identical vectors. Trained with contrastive loss on paraphrase pairs.

2. **Model 2 — Concept Predictor** (future): Predicts sequences of concept vectors — reasons purely in concept space, never sees raw language.

3. **Model 3 — Text Renderer** (future): Renders concept vectors back into fluent text.

V4 begins with Model 1 only. Models 2 and 3 come later once the geometry is verified to work.

### Model 1 — Semantic Encoder (~31M params)

Bidirectional transformer encoder → L2-normalized concept vector.

| Parameter | Value |
|-----------|-------|
| Hidden size | 384 |
| Layers | 8 |
| Attention heads | 6 (head dim 64) |
| FFN | SwiGLU, 1536 intermediate |
| Max sequence | 128 tokens |
| Output | 512-dim unit vector |
| Parameters | ~31M |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Pooling | Mean over tokens |

**Training**: Contrastive only (NT-Xent / SimCLR loss). No next-token prediction, no MLM. The entire training signal is: paraphrase pairs produce identical vectors, non-paraphrase pairs produce distant vectors.

**Data**: DFSG-compliant paraphrase and translation pairs:
- PAWS (Apache 2.0) — hard paraphrase pairs from word scrambling
- QQP (CC) — 400K question paraphrase pairs
- MRPC (Permissive) — news paraphrases
- EuroParl (Public Domain) — EN-FR translation pairs
- WikiMatrix (CC-BY-SA) — mined cross-lingual pairs

### Success Criteria

| Metric | Target | Description |
|--------|--------|-------------|
| Paraphrase cosine sim | >0.95 | Same meaning → same vector |
| Non-paraphrase cosine sim | <0.30 | Different meaning → distant vector |
| Separation ratio | >3.0 | para_sim / nonpara_sim |
| Cross-lingual alignment | >0.90 | EN and FR of same sentence → same vector |
| Hard negative separation | <0.40 | Similar surface form, different meaning → distant |

### Quick Start

```bash
# 1. Build paraphrase pair datasets
python build_pairs.py

# 2. Train semantic encoder
python train_encoder.py --fresh

# 3. Evaluate geometry
python train_encoder.py --eval-only
```

## Version History

### V4 (current) — Concept Vector Architecture
- 31M param bidirectional encoder, contrastive training
- Fundamental pivot from next-token prediction to semantic geometry
- Goal: encode meaning into unambiguous vector space

### V3 (stopped) — SmolLM-135M, Common Pile Data
- 135M params (SmolLM-135M exact: 30 layers × 576 hidden)
- Reached step 37,550, 1.23B tokens, loss 2.67
- Beat V1 and V2 per-token but text still incoherent at 12% through training
- Estimates suggest ~100B tokens needed for usable output — months on single GPU
- Checkpoints preserved in `checkpoints/archive_v3/`

### V2 (mothballed) — 493M Dense Transformer
- 493M params (26 layers × 1280 hidden)
- Reached step 211K, 3.5B tokens, loss 2.70 but text still incoherent
- Problem: Way too few tokens for model size (needed ~500B+)

### V1 (archived) — Tournament of 10 Architectures
- 164M winner (Dense GQA, 20 layers × 1024 hidden)
- Trained on 9.8B tokens, loss 3.14
- Used Common Crawl derivatives (not truly DFSG-compliant)

## Key Lessons Learned

1. **Next-token prediction at small scale needs enormous data** — V3 (135M) needs ~100B tokens for coherent output. That's months on one GPU.
2. **Model size must match data budget** — V2's 493M model needed ~500B+ tokens.
3. **Deep & narrow beats wide & shallow** at small scale (MobileLLM insight).
4. **Loss ≠ coherence** — V3 reached loss 2.67 (better than V1's final 3.14) but still produced word salad.
5. **Common Crawl is not DFSG-compliant** — web pages are "all rights reserved" by default.

## Project Structure

```
flm/
├── encoder_model.py          # V4: Semantic encoder (31M, bidirectional)
├── train_encoder.py          # V4: Contrastive training
├── build_pairs.py            # V4: Download paraphrase pair datasets
├── model.py                  # V1-V3: Decoder-only transformer
├── train_pretrain.py         # V1-V3: Next-token pretraining
├── build_dataset.py          # V1-V3: Pretrain data pipeline
├── train_sft.py              # SFT training
├── train_dpo.py              # DPO alignment
├── plot_training.py          # Training dashboard (V1/V2/V3 comparison)
├── eval/                     # Benchmarks
├── data/                     # Training data
├── checkpoints/              # Model checkpoints
│   ├── archive_v1/           # V1 checkpoints
│   └── archive_v3/           # V3 checkpoints
└── logs/                     # Training logs and plots
```

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.

Built by David Hamner with help from Claude.
