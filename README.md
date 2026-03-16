# flm — The Free Language Model

An open experiment in building AI from scratch using only DFSG-compliant data, training on a single RTX 3090. Every dataset free, every weight reproducible, every decision documented.

The goal: build the first AI model you could `apt install` from Debian main.

**Free as in freedom** — the name is a direct reference to the Free Software Foundation's philosophy that software freedom is a matter of liberty, not price.

## What's Happening Now

**V33 — Mixed Skip-Gram + CBOW word2vec** (COMPLETE — **59.2% Google analogies**, up from V28's 43.9%)

Dual-objective word2vec on shared embeddings hit 59.2% on the Google analogy benchmark, competitive with the original word2vec's 61% despite training on ~2B tokens vs 6B. Syntactic accuracy (65.4%) actually exceeds the original's 59%.

Next up: **V34 — Dynamic masking** to push past 60%.

**[Training Dashboard](https://ruapotato.github.io/chat_hamner/dashboard.html)** | **Local:** `python web_dashboard.py` → http://localhost:8501

## Interactive Demos

- **[Word Embedding Probe](https://ruapotato.github.io/chat_hamner/probe_w2v.html)** — V28: analogies, vector arithmetic, t-SNE, benchmark comparison
- **[3D Filesystem Embeddings](https://ruapotato.github.io/chat_hamner/probe_v32_paths_3d.html)** — V32: the entire Debian filesystem as a point cloud, 17 color-coded categories
- **[V33 Embedding Probe](https://ruapotato.github.io/chat_hamner/probe_v33_3d.html)** — V33: t-SNE, analogies (59.2% Google), cluster analysis, benchmark comparison
- **[3D Word Globe](https://ruapotato.github.io/chat_hamner/probe_v29_3d.html)** — V29: 100K words compressed to 3 dimensions on a sphere

## The Experiment So Far

This project has gone through 33 versions across several research directions. Each taught something about how structure emerges (or doesn't) in learned representations.

| Phase | Versions | What we tried | What we learned |
|-------|----------|---------------|-----------------|
| Geometry from scratch | [V1-V12](docs/history_geometry.md) | Contrastive, margin, decorrelation, WordNet, NLI losses | Explicit geometry losses are a dead end — they destroy reconstruction or overfit |
| Sentence compression | [V13-V24](docs/history_sentences.md) | Autoencoders, dual-decoder, scaling, cross-lingual, bottleneck tuning | Perfect reconstruction is easy; meaningful *structure between* encodings is hard |
| Prediction creates geometry | [V25-V27](docs/history_prediction.md) | Joint encoder/predictor/decoder, sentence2vec, contrastive fine-tuning | Prediction tasks force geometric structure naturally — word2vec's core insight scales up |
| Word & path embeddings | [V28-V33](docs/history_embeddings.md) | Word2vec (300d, 3D), path2vec on Debian filesystem, mixed SG+CBOW | V33 hits 59.2% analogies — competitive with word2vec (61% on 3x more data); dual signal gives richer geometry |

### Key Lessons

1. **Prediction creates geometry, reconstruction doesn't.** Perfect autoencoding gives "swiss cheese" space — meaningless between encodings.
2. **Joint training beats frozen phases.** Let the prediction task shape representations from step 1.
3. **Whole-word vocabulary matters.** Subword tokenization breaks word2vec geometry completely.
4. **Dual objectives create richer geometry.** V33's mixed SG+CBOW reaches 59.2% analogies — competitive with published word2vec (61% on 3x more data).
5. **Small models can compress well.** 25.9M params outperforms 248M when the architecture is right.

## Quick Start

```bash
# Train V33 (mixed word2vec) from scratch
python train_v33.py --fresh

# Monitor training
python web_dashboard.py        # http://localhost:8501

# Run Google analogy benchmark
python eval_analogy.py checkpoints/word2vec_v33/latest.pt
```

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.

Built by David Hamner with help from Claude.
