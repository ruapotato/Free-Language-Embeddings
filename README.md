# flm — The Free Language Model

**66.5% on Google analogies with 1/3 the data.** A single RTX 3090, ~2B tokens of DFSG-compliant text, and a dynamic masking word2vec that beats the original (61% on 6B tokens) by 5.5 points.

Free as in freedom — every dataset DFSG-compliant, every weight reproducible, every decision documented. The goal: the first AI model you could `apt install` from Debian main.

## Results

| Model | Data | Dim | Google Analogies |
|-------|------|-----|-----------------|
| **V34 (ours)** | **~2B tokens** | **300d** | **66.5%** |
| word2vec (Mikolov 2013) | 6B tokens | 300d | 61.0% |
| GloVe (small) | 6B tokens | 300d | 71.0% |
| Google word2vec | 6B tokens | 300d | 72.7% |
| GloVe (Pennington 2014) | 840B tokens | 300d | 75.6% |
| FastText (Bojanowski 2017) | 16B tokens | 300d | 77.0% |

V34 breakdown: semantic 61.4%, syntactic 69.2%. Comparatives 91.7%, plurals 86.8%, capitals 82.6%.

## Quick Start

```bash
# Train V34 from scratch (~2M steps, ~24h on RTX 3090)
python train_v34.py --fresh

# Monitor training
python web_dashboard.py        # http://localhost:8501

# Run Google analogy benchmark (GPU-accelerated, <1s)
python eval_analogy.py checkpoints/word2vec_v34/latest.pt

# Generate embedding spectrogram
python generate_spectrogram.py  # → docs/spectrogram.html

# Generate 3D semantic direction explorer
python generate_semantic_3d.py  # → docs/semantic_3d.html
```

## Interactive Demos

- **[Embedding Spectrogram](https://ruapotato.github.io/chat_hamner/spectrogram.html)** — PCA waves, sine fits, cosine surfaces across V28/V33/V34/Google
- **[3D Semantic Directions](https://ruapotato.github.io/chat_hamner/semantic_3d.html)** — See how semantic axes (size, temperature, time) align in the learned geometry
- **[Training Dashboard](https://ruapotato.github.io/chat_hamner/dashboard.html)** — Live training metrics and loss curves
- **[Alien Language](experiments/exp_1_articulatory/render.html)** — V34 embeddings compressed to 9 articulatory dimensions, rendered as speech

## Experiments

### Exp 1: Articulatory Bottleneck

Compresses 300d embeddings through a 9-dimensional bottleneck (voicing, tenseness, place, manner, nasality, lip rounding, vowel height, vowel backness, sibilance) and reconstructs. The bottleneck values map to mouth shape parameters and render through Web Audio synthesis as "alien language" — semantically similar words sound alike.

```bash
python experiments/exp_1_articulatory/train.py   # Train autoencoder
python experiments/exp_1_articulatory/eval.py    # Evaluate reconstruction
# Open experiments/exp_1_articulatory/render.html in browser to hear it
```

## How We Got Here

This started as a language model experiment, evolved into geometric structure research, and became what it is now: a study of how prediction tasks create meaningful geometry in learned representations.

34 versions across four phases:

| Phase | Versions | What happened |
|-------|----------|---------------|
| Geometry from scratch | V1-V12 | Tried to force structure with contrastive/margin/decorrelation losses. Dead end — explicit geometry destroys reconstruction. |
| Sentence compression | V13-V24 | Autoencoders, dual-decoder, cross-lingual. Perfect reconstruction is easy; meaningful structure *between* encodings is hard. |
| Prediction creates geometry | V25-V27 | The breakthrough: prediction tasks force geometric structure naturally. Word2vec's core insight. |
| Word embeddings | V28-V34 | Skip-gram → mixed SG+CBOW → dynamic masking. Each step unlocked more structure. V34 crystallizes during cosine LR decay. |

Key lessons:
1. **Prediction creates geometry, reconstruction doesn't.** Autoencoding gives "swiss cheese" space.
2. **Dynamic masking unlocks crystallization.** V34 did nothing for 50% of training, then geometry exploded as LR dropped.
3. **Small data can win.** 2B tokens beats 6B when the training signal is right.

Full history: [geometry](docs/history_geometry.md) | [sentences](docs/history_sentences.md) | [prediction](docs/history_prediction.md) | [embeddings](docs/history_embeddings.md)

## Repository Structure

```
train_v34.py              # V34 dynamic masking word2vec training
eval_analogy.py           # Google analogy benchmark (GPU)
generate_spectrogram.py   # Embedding spectrogram visualization
generate_semantic_3d.py   # 3D semantic direction explorer
web_dashboard.py          # Training dashboard (Streamlit)
experiments/              # Active experiments (exp_1_articulatory)
docs/                     # Generated HTML visualizations
logs/                     # Training metrics (CSV)
legacy/                   # V1-V33 training scripts, old probes, utilities
```

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.

Built by David Hamner with help from Claude.
