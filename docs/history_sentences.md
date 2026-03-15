# V13-V24: Sentence Compression

After the geometry experiments failed, the focus shifted to building a strong sentence autoencoder — compress a sentence into a fixed-size vector and reconstruct it perfectly. The idea was to nail reconstruction first, then add structure later.

## V13 — Dual-Decoder EN-FR

- 82.8M params, English reconstruction + French translation decoder
- Forced language-independent encoding by requiring the same bottleneck to produce both languages
- Interesting idea but the translation quality was poor and the concept space was still flat

## V21 — 4L Decoder Experiment

- Asymptoted at loss ~6.48
- Led to the hypothesis that scaling would help

## V22 — Scaled English-Only

- 248M params, 12L encoder, 8L decoder, bert-base-uncased tokenizer
- Plateaued at ~6.0 — turned out to be a padding mask bug, not a scale problem
- **Lesson**: 248M params with a bug loses to 25.9M params without one

## V24 — Sentence Compressor (the breakthrough)

- 25.9M params, 4L encoder/decoder, 1024-dim bottleneck (64 concepts x 16 dims), max 64 tokens
- Cross-attention bottleneck with proper padding masks
- **100% exact match** on test sentences at 250K steps — perfect reconstruction

But the concept space was "swiss cheese":
- Perfect compression, but meaningless structure between encodings
- Nearby vectors didn't mean similar sentences
- Vector arithmetic produced nonsense
- Interpolation gave garbage

**Key finding**: reconstruction alone creates islands of perfect fidelity in a sea of nothing. The space *between* encodings is never trained, so it's random. This is why you need a prediction task — it forces the model to care about the relationships between encodings, not just the encodings themselves.
