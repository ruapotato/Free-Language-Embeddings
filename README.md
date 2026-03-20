# fle — Free Language Embeddings

300-dimensional word vectors trained from scratch on ~2B tokens of DFSG-compliant text using a single RTX 3090. The model is a dynamic masking variant of word2vec skip-gram with 100K whole-word vocabulary, trained for 2M steps (~24 hours).

**66.5% on Google analogies** — beating the original word2vec (61% on 6B tokens) by 5.5 points with 1/3 the data.

Free as in freedom — every dataset DFSG-compliant, every weight reproducible, every decision documented. The goal: the first word embeddings you could `apt install` from Debian main.

This project started as a language model experiment, spent 24 versions discovering that prediction tasks — not reconstruction — create geometric structure in vector spaces, then focused on pushing word embeddings as far as possible on free data.

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

## Download and Use

```bash
# 1. Clone the repo
git clone https://github.com/ruapotato/Free-Language-Embeddings.git
cd Free-Language-Embeddings

# 2. Download the embeddings (~107 MB)
wget https://github.com/ruapotato/Free-Language-Embeddings/releases/download/v34/fle_v34.npz

# 3. Query (no dependencies beyond numpy)
python fle.py --similar cat
python fle.py king - man + woman
python fle.py paris - france + germany

# 4. Interactive mode
python fle.py
```

### Examples

```
$ python fle.py king - man + woman
  king - man + woman
  → queen                0.7387
  → princess             0.6781
  → monarch              0.5546

$ python fle.py paris - france + germany
  paris - france + germany
  → berlin               0.8209
  → vienna               0.7862
  → munich               0.7850

$ python fle.py --similar cat
  kitten               0.7168
  cats                  0.6849
  tabby                 0.6572
  dog                   0.5919

$ python fle.py linux - gnu
  asus                 0.4434
  vmware               0.3976
  booting              0.3964
  usb                  0.3883
  redhat               0.3830

$ python fle.py ubuntu - debian + redhat
  centos               0.6261
  linux                0.6016
  rhel                 0.5949
  vmware               0.5784

$ python fle.py frogs
  toads                0.8035
  frog                 0.7257
  tadpoles             0.6941
  newts                0.6939
  lizards              0.6608

$ python fle.py brain
  cerebral             0.6665
  cerebellum           0.6022
  nerves               0.5748
  cerebrum             0.5736
```

Use in your own code:

```python
from fle import FLE

fle = FLE()                                  # loads fle_v34.npz
vec = fle["cat"]                             # 300d numpy array
fle.similar("cat", n=10)                     # nearest neighbors
fle.analogy("king", "man", "woman")          # king:man :: woman:?
fle.similarity("cat", "dog")                 # cosine similarity
fle.query("king - man + woman")              # vector arithmetic
```

## The Progression

### Phase 1: Trying to force geometry (V1-V12)

Started with the goal of building a free language model. Tried contrastive losses, margin losses, decorrelation penalties, WordNet supervision, NLI-based objectives — anything to force meaningful geometric structure into learned representations.

**Dead end.** Explicit geometry losses either destroy reconstruction quality or overfit to the specific structure you impose. You can't bolt geometry onto representations after the fact.

### Phase 2: Sentence compression (V13-V24)

Pivoted to autoencoders. Dual-decoder architectures, cross-lingual training, bottleneck tuning, scaling experiments from 25M to 248M parameters.

**Key discovery:** Perfect reconstruction is easy. But the space *between* encodings is meaningless — "swiss cheese" manifolds where interpolation produces garbage. Reconstruction doesn't need structure, so the model doesn't learn any.

### Phase 3: Prediction creates geometry (V25-V27)

The breakthrough. Joint encoder/predictor/decoder architectures where the model must *predict* missing context, not just reconstruct input. This is word2vec's core insight: prediction forces the model to organize representations so that similar things are near each other, because they predict the same contexts.

### Phase 4: Word embeddings (V28-V34)

Dropped the LM framing entirely and focused on studying how prediction tasks create geometric structure in embeddings.

**V28 — Skip-gram baseline** (43.9% analogies). Classic word2vec, 300d, 100K whole-word vocab on ~2B tokens. Established that whole-word vocabulary is critical — subword tokenization completely breaks the geometry. king - man + woman = queen works (0.737 cosine). [Interactive probe](https://ruapotato.github.io/Free-Language-Embeddings/probe_w2v.html)

**V29 — 3D word2vec** (0% analogies, but perfect broad separation). Every word is just an (x,y,z) coordinate on a unit sphere. Can't resolve individual words within a region, but tech/nature/emotions self-organize into distinct areas. Proved: you need extra dimensions for fine-grained structure, but broad semantic regions emerge even in 3D. [Interactive globe](https://ruapotato.github.io/Free-Language-Embeddings/probe_v29_3d.html)

**V32 — Filesystem embeddings.** Applied skip-gram to Debian filesystem paths. 199K paths from a full trixie chroot, treating path components as "words." `apache2` ↔ `nginx` (0.60), `systemd` ↔ `udev` (0.67), `__pycache__` → `__init__.py` (0.90). Skip-gram captures "what lives near what" in any tree structure, but hierarchy doesn't produce clean vector arithmetic. [3D filesystem](https://ruapotato.github.io/Free-Language-Embeddings/probe_v32_paths_3d.html)

**V33 — Mixed SG+CBOW** (59.2% analogies). Two views of the same data create richer geometry. Alternating skip-gram and CBOW on shared embeddings jumps +15.3% over V28. Syntactic accuracy (65.4%) exceeds original word2vec. Effective rank doubles from 21 to 49 dimensions — dual signal fills more of the space. [3D probe](https://ruapotato.github.io/Free-Language-Embeddings/probe_v33_3d.html)

**V34 — Dynamic masking** (66.5% analogies). The current model. Randomly masks context positions during training, forcing the model to extract signal from partial views. The training curve is remarkable: nothing happens for 50% of training, then geometry crystallizes as the cosine LR decays — analogies jump from 1.2% to 66.5% in the second half. Beats published word2vec (61% on 3x more data).

### Key Lessons

1. **Prediction creates geometry, reconstruction doesn't.** Autoencoding gives "swiss cheese" space — meaningless between encodings.
2. **Whole-word vocabulary is critical.** Subword tokenization breaks word2vec geometry completely.
3. **Dual objectives create richer geometry.** Two complementary gradients on shared embeddings produce substantially richer structure.
4. **Dynamic masking unlocks crystallization.** V34 did nothing for 50% of training, then geometry exploded as LR dropped.
5. **Small data can win.** 2B tokens beats 6B when the training signal is right.

Full history: [geometry](docs/history_geometry.md) | [sentences](docs/history_sentences.md) | [prediction](docs/history_prediction.md) | [embeddings](docs/history_embeddings.md)

## Train From Scratch

```bash
pip install torch numpy tqdm rich streamlit

# Train V34 from scratch (~2M steps, ~4 days on RTX 3090)
python train_v34.py --fresh

# Monitor training
python web_dashboard.py        # http://localhost:8501

# Run Google analogy benchmark (GPU-accelerated, <1s)
python eval_analogy.py checkpoints/word2vec_v34/latest.pt

# Export embeddings for inference
python -c "
import torch, json, numpy as np
cp = torch.load('checkpoints/word2vec_v34/latest.pt', map_location='cpu', weights_only=False)
vocab = json.load(open('checkpoints/word2vec_v28/vocab.json'))
words = [''] * len(vocab['word2id'])
for w, i in vocab['word2id'].items(): words[i] = w
np.savez_compressed('fle_v34.npz', embeddings=cp['model_state_dict']['target_embeddings.weight'].numpy(), words=np.array(words, dtype=object))
"

# Generate visualizations
python generate_spectrogram.py  # → docs/spectrogram.html
python generate_semantic_3d.py  # → docs/semantic_3d.html
```

## Interactive Demos

- **[Embedding Spectrogram](https://ruapotato.github.io/Free-Language-Embeddings/spectrogram.html)** — PCA waves, sine fits, cosine surfaces across V28/V33/V34/Google
- **[3D Semantic Directions](https://ruapotato.github.io/Free-Language-Embeddings/semantic_3d.html)** — See how semantic axes (size, temperature, time) align in the learned geometry
- **[Training Dashboard](https://ruapotato.github.io/Free-Language-Embeddings/dashboard.html)** — Live training metrics and loss curves


## Experiments

### Exp 1: Articulatory Bottleneck

Compresses 300d embeddings through a 9-dimensional bottleneck (voicing, tenseness, place, manner, nasality, lip rounding, vowel height, vowel backness, sibilance) and reconstructs. The bottleneck values map to mouth shape parameters and render through Web Audio synthesis as "alien language" — semantically similar words sound alike.

```bash
python experiments/exp_1_articulatory/train.py   # Train autoencoder
python experiments/exp_1_articulatory/eval.py    # Evaluate reconstruction
# Open experiments/exp_1_articulatory/render.html in browser to hear it
```

## Repository Structure

```
fle.py                    # Load and query embeddings (no GPU needed)
fle_v34.npz               # Pre-trained embeddings (download from releases)
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
