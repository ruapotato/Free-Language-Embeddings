---
language:
- en
license: gpl-3.0
tags:
- word-embeddings
- word2vec
- embeddings
- nlp
- free-software
- dfsg
datasets:
- wikimedia/wikipedia
- pg19
metrics:
- accuracy
model-index:
- name: fle-v34
  results:
  - task:
      type: word-analogy
      name: Word Analogy
    dataset:
      type: custom
      name: Google Analogy Test Set
    metrics:
    - type: accuracy
      value: 66.5
      name: Overall Accuracy
    - type: accuracy
      value: 61.4
      name: Semantic Accuracy
    - type: accuracy
      value: 69.2
      name: Syntactic Accuracy
library_name: numpy
pipeline_tag: feature-extraction
---

# Free Language Embeddings (V34)

300-dimensional word vectors trained from scratch on ~2B tokens of freely-licensed text using a single RTX 3090.

**66.5% on Google analogies** — beating the original word2vec (61% on 6B tokens) by 5.5 points with 1/3 the data.

## Model Details

| | |
|---|---|
| **Architecture** | Dynamic masking word2vec skip-gram |
| **Dimensions** | 300 |
| **Vocabulary** | 100,000 whole words |
| **Training data** | ~2B tokens, all [DFSG-compliant](https://wiki.debian.org/DFSGLicenses) (see below) |
| **Training hardware** | Single NVIDIA RTX 3090 |
| **Training time** | ~4 days (2M steps) |
| **License** | GPL-3.0 |
| **Parameters** | 60M (30M target + 30M context embeddings) |

### Training Data

All training data meets the [Debian Free Software Guidelines](https://wiki.debian.org/DFSGLicenses) for redistribution, modification, and use. No web scrapes, no proprietary datasets.

| Source | Weight | License |
|--------|--------|---------|
| Wikipedia | 30% | CC BY-SA 3.0 |
| Project Gutenberg | 20% | Public domain |
| arXiv | 20% | Various open access |
| Stack Exchange | 16% | CC BY-SA 4.0 |
| US Government Publishing Office | 10% | Public domain (US gov) |
| RFCs | 2.5% | IETF Trust |
| Linux kernel docs, Arch Wiki, TLDP, GNU manuals, man pages | 1.5% | GPL/GFDL |

## Benchmark Results

| Model | Data | Google Analogies |
|-------|------|-----------------|
| **fle V34 (this model)** | **~2B tokens** | **66.5%** |
| word2vec (Mikolov 2013) | 6B tokens | 61.0% |
| GloVe (small) | 6B tokens | 71.0% |
| Google word2vec | 6B tokens | 72.7% |
| GloVe (Pennington 2014) | 840B tokens | 75.6% |
| FastText (Bojanowski 2017) | 16B tokens | 77.0% |

Breakdown: semantic 61.4%, syntactic 69.2%. Comparatives 91.7%, plurals 86.8%, capitals 82.6%.

## Quick Start

```bash
# Download
pip install huggingface_hub numpy
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('hackersgame/Free_Language_Embeddings', 'fle_v34.npz', local_dir='.')
hf_hub_download('hackersgame/Free_Language_Embeddings', 'fle.py', local_dir='.')
"

# Use
python fle.py king - man + woman
python fle.py --similar cat
python fle.py   # interactive mode
```

### Python API

```python
from fle import FLE

fle = FLE()                                  # loads fle_v34.npz
vec = fle["cat"]                             # 300d numpy array
fle.similar("cat", n=10)                     # nearest neighbors
fle.analogy("king", "man", "woman")          # king:man :: woman:?
fle.similarity("cat", "dog")                 # cosine similarity
fle.query("king - man + woman")              # vector arithmetic
```

## Examples

```
$ python fle.py king - man + woman
  → queen                0.7387
  → princess             0.6781
  → monarch              0.5546

$ python fle.py paris - france + germany
  → berlin               0.8209
  → vienna               0.7862
  → munich               0.7850

$ python fle.py --similar cat
  kitten               0.7168
  cats                  0.6849
  tabby                 0.6572
  dog                   0.5919

$ python fle.py ubuntu - debian + redhat
  centos               0.6261
  linux                0.6016
  rhel                 0.5949

$ python fle.py brain
  cerebral             0.6665
  cerebellum           0.6022
  nerves               0.5748
```

## What Makes This Different

- **Free as in freedom.** Every dataset is DFSG-compliant. Every weight is reproducible. GPL-3.0 licensed. The goal: word embeddings you could `apt install` from Debian main.
- **Dynamic masking.** Randomly masks context positions during training, forcing the model to extract signal from partial views. The result: geometry that crystallizes during cosine LR decay — analogies jump from 1.2% to 66.5% in the second half of training.
- **Whole-word vocabulary.** No subword tokenization. Subwords break word2vec geometry completely — they don't carry enough meaning individually for co-occurrence statistics to produce useful structure.

## Training

Trained with cosine learning rate schedule (3e-4 → 1e-6). The training curve shows a striking crystallization pattern: near-zero analogy accuracy for the first 50% of training, then rapid emergence of geometric structure as the learning rate decays.

Full training code and visualizations: [github.com/ruapotato/Free-Language-Embeddings](https://github.com/ruapotato/Free-Language-Embeddings)

## Interactive Visualizations

- [Embedding Spectrogram](https://ruapotato.github.io/Free-Language-Embeddings/spectrogram.html) — PCA waves, sine fits, cosine surfaces
- [3D Semantic Directions](https://ruapotato.github.io/Free-Language-Embeddings/semantic_3d.html) — See how semantic axes align in the learned geometry
- [Training Dashboard](https://ruapotato.github.io/Free-Language-Embeddings/dashboard.html) — Loss curves and training metrics

## Citation

```bibtex
@misc{hamner2026fle,
  title={Free Language Embeddings: Dynamic Masking Word2Vec on DFSG-Compliant Data},
  author={David Hamner},
  year={2026},
  url={https://github.com/ruapotato/Free-Language-Embeddings}
}
```

## License

GPL-3.0 — See [LICENSE](https://github.com/ruapotato/Free-Language-Embeddings/blob/main/LICENSE) for details.

Built by David Hamner.
