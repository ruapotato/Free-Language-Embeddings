# V1-V12: Learning Geometry from Scratch

The first 12 versions explored whether you could *explicitly* train geometric structure into sentence representations. Every approach failed in the same way: either reconstruction quality collapsed, or the geometry overfit to synthetic templates.

## V1-V9 — Geometry Experiments

A parade of ideas for forcing structure into the concept space:

- **Supervised slots**: assign meaning to specific embedding dimensions
- **Decorrelation losses**: push dimensions to be independent
- **Margin losses**: pull similar sentences together, push different ones apart
- **Contrastive learning**: SimCLR-style with various augmentations
- **WordNet structure**: use the taxonomy as a geometric target
- **NLI graded losses**: use entailment/contradiction/neutral as distance targets

None of these worked. The losses either fought reconstruction (making the model worse at its primary task) or collapsed to trivial solutions (everything mapped to a small region, templates memorized).

**Key finding**: you can't bolt geometry onto an autoencoder. The geometry has to emerge from the task itself.

## V10-V12 — Pure Reconstruction

Stripped back to just reconstruction to establish a baseline.

- Good reconstruction (96% token accuracy)
- But the concept space was a bag-of-words — no meaningful structure between encodings
- Similar sentences weren't nearby, analogies didn't work, interpolation produced garbage

This confirmed the problem: autoencoders learn to compress, not to organize. The space between encodings is meaningless because nothing in the training signal cares about it.
