#!/usr/bin/env python3
"""
Probe V24 concept space — reconstruction, interpolation, arithmetic, nearest neighbors.
Usage: python probe_v24.py [checkpoint_path]
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from concept_model import ConceptAutoencoderV24, ConceptConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoderV24(config).to(DEVICE).eval()
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    step = ckpt["step"]
    loss = ckpt.get("loss", 0)
    em = ckpt.get("exact_match_ema", 0)
    print(f"Loaded step {step:,} | loss={loss:.4f} | em_ema={em:.3f}")
    print(f"  {config.enc_layers}L enc, {config.dec_layers}L dec, "
          f"{config.num_concepts}x{config.concept_dim}={config.num_concepts*config.concept_dim}d bottleneck")
    return model, tokenizer, config


@torch.no_grad()
def encode(model, tokenizer, text):
    """Encode text -> concept vector (num_concepts, concept_dim)."""
    enc = tokenizer([text], max_length=64, padding=True, truncation=True,
                    return_tensors="pt").to(DEVICE)
    concepts = model.encode(enc["input_ids"], enc["attention_mask"])
    return concepts[0], enc  # (K, D), tokenizer output


@torch.no_grad()
def decode(model, tokenizer, concepts, seq_len=64):
    """Decode concept vector -> text."""
    if concepts.dim() == 2:
        concepts = concepts.unsqueeze(0)  # (1, K, D)
    # Create dummy attention mask for decoding
    mask = torch.ones(1, seq_len, dtype=torch.long, device=DEVICE)
    logits = model.decode(concepts, seq_len, mask)
    tokens = logits.argmax(dim=-1)[0]
    # Strip padding and special tokens
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text


@torch.no_grad()
def encode_flat(model, tokenizer, text):
    """Encode text -> flat normalized vector for similarity."""
    c, _ = encode(model, tokenizer, text)
    flat = c.view(1, -1)
    return F.normalize(flat, p=2, dim=-1)[0]


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def test_reconstruction(model, tokenizer):
    section("RECONSTRUCTION")
    sentences = [
        "the cat sat on the mat",
        "the dog bit the man",
        "she runs every morning before breakfast",
        "artificial intelligence will change the world",
        "the quick brown fox jumps over the lazy dog near the river",
        "def fibonacci ( n ) : return n if n < 2 else fibonacci ( n - 1 ) + fibonacci ( n - 2 )",
        "water boils at one hundred degrees celsius at sea level",
        "the president gave a speech about climate change yesterday",
        "please pass the salt and pepper",
        "i have never seen such a beautiful sunset in my entire life",
    ]
    exact = 0
    for s in sentences:
        c, enc = encode(model, tokenizer, s)
        out = decode(model, tokenizer, c, enc["input_ids"].shape[1])
        match = (out.strip() == s.strip())
        exact += int(match)
        status = "OK " if match else "DIFF"
        print(f"  [{status}] {s}")
        if not match:
            print(f"       -> {out}")
    print(f"\n  Exact match: {exact}/{len(sentences)} ({exact/len(sentences):.0%})")


def test_interpolation(model, tokenizer):
    section("INTERPOLATION (concept space)")
    pairs = [
        ("the cat is sleeping", "the dog is running"),
        ("it is very cold outside", "it is very hot outside"),
        ("she is happy", "she is sad"),
        ("the food was delicious", "the food was terrible"),
    ]
    for a_text, b_text in pairs:
        ca, _ = encode(model, tokenizer, a_text)
        cb, _ = encode(model, tokenizer, b_text)
        print(f"\n  '{a_text}' <-> '{b_text}'")
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            c_interp = (1 - alpha) * ca + alpha * cb
            out = decode(model, tokenizer, c_interp)
            print(f"    {alpha:.2f}: {out}")


def test_arithmetic(model, tokenizer):
    section("SEMANTIC ARITHMETIC")
    # king - man + woman = queen style
    tests = [
        ("the cat is sleeping", "the cat", "the dog", "expect: the dog is sleeping"),
        ("he is happy", "he", "she", "expect: she is happy"),
        ("the water is cold", "cold", "hot", "expect: the water is hot"),
        ("the big red car", "red", "blue", "expect: the big blue car"),
    ]
    for a, b, c, expect in tests:
        va = encode_flat(model, tokenizer, a)
        vb = encode_flat(model, tokenizer, b)
        vc = encode_flat(model, tokenizer, c)
        # a - b + c
        v_result = va - vb + vc
        # We need the full-shape concepts to decode, so this is approximate
        ca, _ = encode(model, tokenizer, a)
        cb, _ = encode(model, tokenizer, b)
        cc, _ = encode(model, tokenizer, c)
        c_result = ca - cb + cc
        out = decode(model, tokenizer, c_result)
        print(f"  '{a}' - '{b}' + '{c}'")
        print(f"    = {out}  ({expect})")


def test_similarity(model, tokenizer):
    section("SIMILARITY (cosine in concept space)")
    groups = {
        "animals": ["the cat is sleeping", "the dog is running", "birds fly south in winter"],
        "weather": ["it is raining outside", "the weather is sunny today", "a storm is coming"],
        "food": ["the pizza was delicious", "she cooked pasta for dinner", "i love chocolate cake"],
    }
    # Encode all
    all_texts = []
    all_vecs = []
    all_labels = []
    for label, texts in groups.items():
        for t in texts:
            v = encode_flat(model, tokenizer, t)
            all_texts.append(t)
            all_vecs.append(v)
            all_labels.append(label)

    vecs = torch.stack(all_vecs)
    sim_matrix = vecs @ vecs.T

    # Print within-group vs between-group
    within, between = [], []
    for i in range(len(all_texts)):
        for j in range(i+1, len(all_texts)):
            s = sim_matrix[i, j].item()
            if all_labels[i] == all_labels[j]:
                within.append(s)
            else:
                between.append(s)

    print(f"  Within-group avg similarity:  {np.mean(within):.3f}")
    print(f"  Between-group avg similarity: {np.mean(between):.3f}")
    print(f"  Gap: {np.mean(within) - np.mean(between):+.3f}")

    print("\n  Nearest neighbors:")
    for i, t in enumerate(all_texts):
        sims = [(sim_matrix[i, j].item(), all_texts[j]) for j in range(len(all_texts)) if j != i]
        sims.sort(reverse=True)
        nn = sims[0]
        print(f"    [{all_labels[i]}] '{t}'")
        print(f"      -> NN: '{nn[1]}' (sim={nn[0]:.3f})")


def test_negation_sensitivity(model, tokenizer):
    section("NEGATION / WORD ORDER SENSITIVITY")
    pairs = [
        ("the dog bit the man", "the man bit the dog"),
        ("alice likes bob", "bob likes alice"),
        ("she is happy", "she is not happy"),
        ("he won the game", "he lost the game"),
        ("the cat chased the mouse", "the mouse chased the cat"),
    ]
    for a, b in pairs:
        va = encode_flat(model, tokenizer, a)
        vb = encode_flat(model, tokenizer, b)
        sim = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
        print(f"  sim={sim:.3f}  '{a}' vs '{b}'")


def test_effective_rank(model, tokenizer):
    section("EFFECTIVE RANK (concept space dimensionality)")
    sentences = [
        "the cat sat on the mat", "birds fly south in winter",
        "she is reading a book", "the car drove down the highway",
        "water boils at one hundred degrees", "the president signed the bill",
        "he played guitar at the concert", "the flowers bloomed in spring",
        "they built a house on the hill", "the teacher explained the lesson",
        "rain fell heavily all afternoon", "the ship sailed across the ocean",
        "she painted a beautiful landscape", "the doctor examined the patient",
        "children played in the park", "the algorithm runs in linear time",
        "the moon orbits the earth", "he wrote a letter to his friend",
        "the restaurant was crowded tonight", "snow covered the mountain peaks",
        "she solved the math problem quickly", "the train arrived on time",
        "they celebrated their anniversary", "the wind blew through the trees",
        "the scientist published her findings", "he fixed the broken window",
        "the river flows to the sea", "she danced gracefully on stage",
        "the company announced record profits", "the baby laughed at the funny face",
    ]
    vecs = []
    for s in sentences:
        v = encode_flat(model, tokenizer, s)
        vecs.append(v.cpu().numpy())
    vecs = np.array(vecs)

    U, S, Vt = np.linalg.svd(vecs, full_matrices=False)
    S_norm = S / S.sum()
    cumsum = np.cumsum(S_norm)

    rank90 = int(np.searchsorted(cumsum, 0.90) + 1)
    rank95 = int(np.searchsorted(cumsum, 0.95) + 1)
    rank99 = int(np.searchsorted(cumsum, 0.99) + 1)

    print(f"  Vectors: {vecs.shape[0]} x {vecs.shape[1]}")
    print(f"  Rank to explain 90% variance: {rank90}")
    print(f"  Rank to explain 95% variance: {rank95}")
    print(f"  Rank to explain 99% variance: {rank99}")
    print(f"  Top-5 singular values: {S[:5].round(3)}")
    print(f"  Top-5 cumulative:      {(cumsum[:5]*100).round(1)}%")


def test_dimension_tweaking(model, tokenizer):
    section("DIMENSION TWEAKING (what do individual dims encode?)")
    sentences = [
        "the cat sat on the mat",
        "she is very happy today",
        "the big red car drove fast",
    ]
    for text in sentences:
        c, enc = encode(model, tokenizer, text)
        seq_len = enc["input_ids"].shape[1]
        original = decode(model, tokenizer, c, seq_len)
        print(f"\n  Original: '{text}'")
        print(f"  Decoded:  '{original}'")

        # Find the most impactful dimensions by tweaking each
        # c is (K, D) = (64, 16)
        K, D = c.shape
        impacts = []
        for k in range(K):
            for d in range(D):
                c_mod = c.clone()
                c_mod[k, d] += 2.0  # nudge up
                out_up = decode(model, tokenizer, c_mod, seq_len)
                c_mod[k, d] -= 4.0  # nudge down (original - 2.0)
                out_down = decode(model, tokenizer, c_mod, seq_len)
                if out_up != original or out_down != original:
                    impacts.append((k, d, out_up, out_down))

        if impacts:
            print(f"  Sensitive dims ({len(impacts)} of {K*D}):")
            for k, d, up, down in impacts[:15]:  # show top 15
                print(f"    slot[{k}][{d}] +2: '{up}'")
                print(f"    slot[{k}][{d}] -2: '{down}'")
        else:
            print(f"  No dimension changes affected output with ±2.0 nudge")

        # Also try bigger nudges on a few dims
        print(f"\n  Large nudges (±5.0) on first 5 slots:")
        for k in range(min(5, K)):
            c_mod = c.clone()
            c_mod[k, :] += 5.0  # shift entire slot
            out = decode(model, tokenizer, c_mod, seq_len)
            if out != original:
                print(f"    slot[{k}] +5.0 all dims: '{out}'")
            c_mod = c.clone()
            c_mod[k, :] -= 5.0
            out = decode(model, tokenizer, c_mod, seq_len)
            if out != original:
                print(f"    slot[{k}] -5.0 all dims: '{out}'")


def test_random_decode(model, tokenizer):
    section("RANDOM CONCEPT VECTORS (what does random noise decode to?)")
    for i in range(10):
        c_rand = torch.randn(1, 64, 16, device=DEVICE) * 1.0
        out = decode(model, tokenizer, c_rand[0])
        print(f"  Random {i}: '{out}'")

    print("\n  Zero vector:")
    c_zero = torch.zeros(1, 64, 16, device=DEVICE)
    print(f"  -> '{decode(model, tokenizer, c_zero[0])}'")

    print("\n  Ones vector:")
    c_ones = torch.ones(1, 64, 16, device=DEVICE)
    print(f"  -> '{decode(model, tokenizer, c_ones[0])}'")


if __name__ == "__main__":
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/concept_v24/latest.pt"
    model, tokenizer, config = load_model(ckpt_path)

    test_reconstruction(model, tokenizer)
    test_negation_sensitivity(model, tokenizer)
    test_similarity(model, tokenizer)
    test_arithmetic(model, tokenizer)
    test_interpolation(model, tokenizer)
    test_dimension_tweaking(model, tokenizer)
    test_random_decode(model, tokenizer)
    test_effective_rank(model, tokenizer)

    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}")
