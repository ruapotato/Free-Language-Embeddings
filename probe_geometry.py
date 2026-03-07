#!/usr/bin/env python3
"""Probe the geometry of the concept space."""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizerFast
from concept_model import ConceptAutoencoder, ConceptConfig

CKPT = "checkpoints/concept_v4/latest.pt"
DEVICE = "cpu"

def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def encode(model, tokenizer, texts):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        concepts = model.encode(enc["input_ids"], enc["attention_mask"])
    flat = concepts.view(concepts.shape[0], -1)
    return F.normalize(flat, p=2, dim=-1), concepts

def cosine(a, b):
    return F.cosine_similarity(a, b).item()

def main():
    print("Loading model...")
    model = load_model()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # ================================================================
    # 1. Semantic Clustering — within-group vs between-group similarity
    # ================================================================
    print("\n" + "="*60)
    print("1. SEMANTIC CLUSTERING")
    print("="*60)

    groups = {
        "animals": ["the cat sat on the mat", "a dog ran in the park", "the bird flew over the tree",
                     "fish swim in the ocean", "the horse galloped across the field"],
        "weather": ["it is raining heavily today", "the sun is shining brightly", "snow covered the ground",
                     "a storm is approaching fast", "the wind blew the leaves away"],
        "food": ["she cooked a delicious pasta", "the pizza was freshly baked", "he ate a bowl of rice",
                  "the soup was too salty", "they ordered sushi for dinner"],
        "emotions": ["she was very happy today", "he felt sad and lonely", "the news made them angry",
                      "they were excited about the trip", "she felt anxious before the exam"],
        "technology": ["the computer crashed again", "she updated her phone software", "the internet was slow",
                       "he wrote a python program", "the server went down last night"],
    }

    group_vecs = {}
    for name, sents in groups.items():
        vecs, _ = encode(model, tokenizer, sents)
        group_vecs[name] = vecs

    # Within-group similarity
    print("\nWithin-group avg cosine similarity:")
    within_sims = []
    for name, vecs in group_vecs.items():
        sims = []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                sims.append(cosine(vecs[i:i+1], vecs[j:j+1]))
        avg = np.mean(sims)
        within_sims.append(avg)
        print(f"  {name:>12s}: {avg:.3f}  (range {min(sims):.3f} - {max(sims):.3f})")
    print(f"  {'AVERAGE':>12s}: {np.mean(within_sims):.3f}")

    # Between-group similarity
    print("\nBetween-group avg cosine similarity:")
    between_sims = []
    group_names = list(group_vecs.keys())
    for i in range(len(group_names)):
        for j in range(i+1, len(group_names)):
            va = group_vecs[group_names[i]]
            vb = group_vecs[group_names[j]]
            sims = []
            for a in range(len(va)):
                for b in range(len(vb)):
                    sims.append(cosine(va[a:a+1], vb[b:b+1]))
            avg = np.mean(sims)
            between_sims.append(avg)
            print(f"  {group_names[i]:>12s} vs {group_names[j]:<12s}: {avg:.3f}")
    print(f"  {'AVERAGE':>12s}: {np.mean(between_sims):.3f}")

    gap = np.mean(within_sims) - np.mean(between_sims)
    print(f"\n  CLUSTERING GAP: {gap:+.3f} (within - between, want > 0)")

    # ================================================================
    # 2. Direction Consistency — do attributes form consistent directions?
    # ================================================================
    print("\n" + "="*60)
    print("2. DIRECTION CONSISTENCY")
    print("="*60)

    directions = {
        "size": [
            ("the big cat", "the cat"),
            ("the big dog", "the dog"),
            ("the big house", "the house"),
            ("a large tree", "a tree"),
        ],
        "negation": [
            ("the cat is not here", "the cat is here"),
            ("she did not run", "she did run"),
            ("he is not happy", "he is happy"),
            ("they are not coming", "they are coming"),
        ],
        "tense": [
            ("she ran quickly", "she runs quickly"),
            ("he walked home", "he walks home"),
            ("they played outside", "they play outside"),
            ("it rained all day", "it rains all day"),
        ],
        "sentiment": [
            ("the movie was great", "the movie was terrible"),
            ("she loves the food", "she hates the food"),
            ("a wonderful day", "a horrible day"),
            ("he is kind", "he is cruel"),
        ],
        "plurality": [
            ("the cats sat", "the cat sat"),
            ("the dogs ran", "the dog ran"),
            ("the birds flew", "the bird flew"),
            ("the cars drove", "the car drove"),
        ],
    }

    for attr, pairs in directions.items():
        deltas = []
        for pos, neg in pairs:
            v_pos, _ = encode(model, tokenizer, [pos])
            v_neg, _ = encode(model, tokenizer, [neg])
            delta = v_pos - v_neg
            delta = F.normalize(delta, p=2, dim=-1)
            deltas.append(delta)

        # Pairwise cosine between direction vectors
        consistencies = []
        for i in range(len(deltas)):
            for j in range(i+1, len(deltas)):
                consistencies.append(cosine(deltas[i], deltas[j]))
        avg = np.mean(consistencies)
        print(f"  {attr:>12s}: avg consistency = {avg:.3f}  "
              f"(range {min(consistencies):.3f} - {max(consistencies):.3f})")

    # ================================================================
    # 3. Analogy Test — a - b + c ≈ d?
    # ================================================================
    print("\n" + "="*60)
    print("3. ANALOGY TEST (a - b + c ≈ d?)")
    print("="*60)

    analogies = [
        ("she ran", "she runs", "he walked", "he walks", "tense transfer"),
        ("big cat", "cat", "big dog", "dog", "size transfer"),
        ("king", "man", "queen", "woman", "gender royalty"),
        ("he is happy", "he is sad", "she is happy", "she is sad", "emotion transfer"),
        ("the cats", "the cat", "the dogs", "the dog", "plural transfer"),
    ]

    for a, b, c, expected_d, label in analogies:
        va, _ = encode(model, tokenizer, [a])
        vb, _ = encode(model, tokenizer, [b])
        vc, _ = encode(model, tokenizer, [c])
        vd, _ = encode(model, tokenizer, [expected_d])

        predicted = F.normalize(va - vb + vc, p=2, dim=-1)
        sim = cosine(predicted, vd)
        print(f"  {label:>20s}: {a} - {b} + {c} ≈ {expected_d}?  sim={sim:.3f}")

    # ================================================================
    # 4. Slot Analysis — what does each slot capture?
    # ================================================================
    print("\n" + "="*60)
    print("4. SLOT ANALYSIS")
    print("="*60)

    test_sents = [
        "the cat sat on the mat",
        "the dog sat on the mat",
        "the cat ran on the mat",
        "a huge cat sat on the mat",
        "the cat sat on the mat yesterday",
    ]
    _, concepts = encode(model, tokenizer, test_sents)
    # concepts: (5, 8, 128)

    print("\nPer-slot cosine similarity between test sentences:")
    print(f"  {'':>5s}", end="")
    for i in range(len(test_sents)):
        print(f"  sent{i}", end="")
    print()

    # Compare slot-by-slot: which slots change between similar sentences?
    base = concepts[0]  # "the cat sat on the mat"
    print(f"\n  Slot changes vs base: '{test_sents[0]}'")
    for j, sent in enumerate(test_sents[1:], 1):
        diffs = []
        for s in range(8):
            slot_sim = F.cosine_similarity(
                base[s:s+1], concepts[j, s:s+1], dim=-1
            ).item()
            diffs.append(slot_sim)
        changed = [i for i, d in enumerate(diffs) if d < 0.95]
        print(f"  vs '{test_sents[j]}'")
        print(f"    slot sims: [{', '.join(f'{d:.2f}' for d in diffs)}]")
        print(f"    changed slots (<0.95): {changed if changed else 'none'}")

    # ================================================================
    # 5. Effective Rank (confirm from vectors directly)
    # ================================================================
    print("\n" + "="*60)
    print("5. EFFECTIVE RANK (measured from 500 sentences)")
    print("="*60)

    all_sents = []
    for sents in groups.values():
        all_sents.extend(sents)
    # Add more diverse sentences
    extras = [
        "the president gave a speech", "she plays piano beautifully",
        "the train arrived on time", "he forgot his wallet at home",
        "the garden was full of flowers", "she read the book in one day",
        "the mountain was covered in snow", "they celebrated their anniversary",
        "the experiment failed completely", "she won the championship",
        "the baby cried all night", "he fixed the broken window",
        "the river flows to the sea", "she painted a beautiful portrait",
        "the fire spread quickly through the building",
        "he whispered a secret to her", "the team won the final game",
        "she discovered a new species", "the market crashed suddenly",
        "he climbed the tallest mountain",
    ]
    all_sents.extend(extras)

    vecs, _ = encode(model, tokenizer, all_sents)
    vecs_np = vecs.numpy()

    # Center
    vecs_np = vecs_np - vecs_np.mean(axis=0)

    # SVD
    _, s, _ = np.linalg.svd(vecs_np, full_matrices=False)
    var = s ** 2
    var_ratio = var / var.sum()
    cumvar = np.cumsum(var_ratio)

    rank90 = np.searchsorted(cumvar, 0.90) + 1
    rank95 = np.searchsorted(cumvar, 0.95) + 1
    rank99 = np.searchsorted(cumvar, 0.99) + 1

    print(f"  Total dims: 1024")
    print(f"  Samples: {len(all_sents)}")
    print(f"  rank90 = {rank90}")
    print(f"  rank95 = {rank95}")
    print(f"  rank99 = {rank99}")
    print(f"\n  Top 20 singular values: {', '.join(f'{v:.2f}' for v in s[:20])}")
    print(f"  Variance in top 10 dims: {cumvar[9]:.1%}")
    print(f"  Variance in top 20 dims: {cumvar[19]:.1%}")
    print(f"  Variance in top 50 dims: {cumvar[49]:.1%}")

    # ================================================================
    # 6. Reconstruction quality spot-check
    # ================================================================
    print("\n" + "="*60)
    print("6. RECONSTRUCTION SPOT-CHECK")
    print("="*60)

    recon_tests = [
        "the cat sat on the mat",
        "she runs every morning before breakfast",
        "the dog bit the man",
        "the man bit the dog",
        "artificial intelligence will change the world",
    ]
    enc = tokenizer(recon_tests, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        logits, _ = model(enc["input_ids"], enc["attention_mask"])
    preds = logits.argmax(dim=-1)
    for i, sent in enumerate(recon_tests):
        decoded = tokenizer.decode(preds[i], skip_special_tokens=True)
        print(f"  IN:  {sent}")
        print(f"  OUT: {decoded}")
        print()


if __name__ == "__main__":
    main()
