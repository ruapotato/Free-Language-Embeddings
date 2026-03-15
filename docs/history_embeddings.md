# V28-V33: Word & Path Embeddings

Shifted focus to word-level embeddings to study how geometric structure emerges from prediction tasks in a simpler, faster-to-train setting. This phase produced the project's strongest quantitative results and some fun visualizations.

## V28 — Word2vec Skip-Gram (300d)

Classic word2vec skip-gram with negative sampling. The baseline for everything that followed.

- 60M params, 300d embeddings, 100K whole-word vocabulary, 500K steps
- Trained on OpenWebText subset (~2B tokens)
- **[Interactive Embedding Probe](https://ruapotato.github.io/chat_hamner/probe_w2v.html)**

Results:
- **Google Analogy Benchmark: 43.9%** (semantic 27.3%, syntactic 52.7%, 80% coverage)
- Published comparison: word2vec (Google News) 61%, GloVe 75%, FastText 78% — gap explained by smaller training corpus
- king - man + woman = queen works (0.737 cosine)
- paris - france + germany = berlin (0.679)
- Custom analogies: 23/26 (88%)

**Key finding**: whole-word vocabulary is critical. An earlier attempt with WordPiece subword tokenization completely broke the geometry — subwords don't carry enough meaning individually for word2vec's co-occurrence statistics to produce useful structure.

## V29 — 3D Word2vec

What if every word is just an (x, y, z) coordinate on a unit sphere?

- 0.6M params, 100K vocabulary, 500K steps
- **[Interactive 3D Globe](https://ruapotato.github.io/chat_hamner/probe_v29_3d.html)** — drag to orbit, hover for words, search by name

Results:
- Google analogies: **0.0%** — too many words packed onto a sphere
- But similarity gap: **+1.074** (excellent broad separation: king/banana = -0.65)
- Tense direction consistency: **0.932** (go→went aligns with run→ran, see→saw)
- Self-organized into a ~2D belt (X-axis variance collapsed)

**Key finding**: 3 dimensions capture semantic *regions* (tech, nature, emotions occupy distinct areas) but can't resolve individual words within a region. You need the extra dimensions for fine-grained structure.

## V30 (archived) — Word2vec LM

- Autoregressive LM using frozen V28 word2vec embeddings as input/output space
- 55.6M params (25.6M trainable), 8-layer causal transformer
- Stopped early — a standard word-level LM doesn't leverage embedding geometry interestingly

## V31 (archived) — Phrase-level BPE Word2vec

- Attempted aggressive word-level BPE to compress sentences to ~3 phrase tokens, then train skip-gram
- First merges were correct ("of the", "in the", "to the") but the Python merge loop was too slow (~2s per merge, needs thousands)

## V32 — Path2vec: Filesystem Embeddings

What does a Debian installation look like as a vector space? Treat each filesystem path as a "sentence" of path components and train skip-gram.

- Data: full Debian trixie chroot with 1,752 packages installed, including `/proc`, `/sys`, `/dev`
- 199K paths, 14,241 unique path components, avg 6.1 components per path
- 8.5M params, 300d, 1M steps, trained in ~5 hours
- `build_debian_dataset.sh` — fully reproducible: builds the chroot, installs packages, collects paths
- **[Interactive 3D Visualization](https://ruapotato.github.io/chat_hamner/probe_v32_paths_3d.html)**

Results:
- **Similarity gap: +0.101** | **Effective rank: 48.1**
- Strong structural clustering: `apache2` ↔ `nginx` (0.60), `systemd` ↔ `udev` (0.67), `man1` ↔ `man8` (0.83)
- Python ecosystem captured: `__pycache__` → `__init__.py` (0.90), `python3` → `dist-packages` (0.63)
- Vector arithmetic is noisy — filesystem paths have strict hierarchy, not the free co-occurrence that makes word2vec analogies work

**Key finding**: skip-gram captures "what lives near what" in the tree, but hierarchical structure doesn't map cleanly to vector arithmetic. Analogy requires symmetric co-occurrence.

## V33 (current, training) — Mixed Skip-Gram + CBOW

Can two views of the same data create richer geometry? Alternating skip-gram and CBOW steps on shared embeddings. A/B comparison against V28.

- Reuses V28's 100K vocab for fair comparison
- 60M params, 1M steps (2x V28), 17.7 step/s
- Data pipeline: numpy ring buffers with background reader thread. A 50ms `time.sleep()` to yield the GIL was the difference between 1.8 and 17.7 step/s — the reader thread's Python-heavy tokenization was starving the GPU.
- At 80K steps: no geometry yet, matching V28's trajectory (also 0% at 80K)

Hypothesis: dual training signals create richer geometry, similar to how V25's joint reconstruction + prediction gave two complementary gradients.
