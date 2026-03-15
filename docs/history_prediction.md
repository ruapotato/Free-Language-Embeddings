# V25-V27: Prediction Creates Geometry

The word2vec insight: prediction tasks create geometric structure for free. Word2vec never explicitly trains king - man + woman = queen — the geometry *emerges* from predicting neighboring words. V25 applied this one level up.

## V25 — Unified Sentence LM

The big architectural idea: **predict a sentence from its neighbors**, jointly training encoder, predictor, and decoder end-to-end.

```
tokens → [4L Encoder] → [Bottleneck] → concept vector → [8L Causal Predictor] → predicted vector → [4L Decoder] → tokens
```

- 58.4M params total (25.9M encoder/decoder + 32.5M predictor)
- Two training signals through the encoder:
  1. **Reconstruction**: concept vectors must faithfully represent the sentence
  2. **Prediction**: concept vectors must be organized so the next sentence is predictable
- Decoder-backprop loss (token-level CE, not MSE) following Meta's LCM finding that MSE produces blurry predictions

At 50% training (298K / 600K steps):
- Reconstruction loss: 0.14 (near-perfect sentence reconstruction)
- Prediction loss: 2.59 (steadily dropping)
- Paraphrases clustered (0.90 cosine similarity)
- Semantic topic grouping emerging (within vs between group gap: +0.35)
- Surface features still dominated geometry, but semantic structure was growing

Related work: Meta's Large Concept Model (LCM) and SONAR-LLM. Key difference: they freeze the sentence encoder first, we trained everything jointly.

## V26 — Masked Sentence Modeling

Sentence-level masked prediction experiment. Archived — the masked objective didn't produce as clean a training signal as causal prediction.

## V27 — Contrastive Autoencoder (SimCSE + VICReg)

- Added SimCSE contrastive loss with VICReg variance/covariance regularization to V24's autoencoder
- Fixed anisotropy (0.97 → 0.42) but failed to create real semantic structure
- **Key finding**: SimCSE with dropout augmentation only teaches noise invariance, not semantic similarity. The model learns "these two dropout-perturbed views of the same sentence should be close" but that doesn't generalize to "these two sentences that mean similar things should be close."
