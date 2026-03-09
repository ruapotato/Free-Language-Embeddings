#!/usr/bin/env python3
"""Probe the geometry of the concept space — 32 slots x 32 dims."""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizerFast
from concept_model import ConceptAutoencoder, ConceptConfig, flat_similarity_matrix

CKPT = "checkpoints/concept_v9/latest.pt"
DEVICE = "cpu"

SLOT_NAMES = {
    0: "subject", 1: "object", 2: "animacy", 3: "age",
    4: "size", 5: "color", 6: "shape", 7: "material",
    8: "weight", 9: "temperature", 10: "action_type", 11: "manner",
    12: "speed", 13: "direction", 14: "location", 15: "spatial",
    16: "distance", 17: "tense", 18: "duration", 19: "time_ref",
    20: "number", 21: "degree", 22: "sentiment", 23: "emotion",
    24: "arousal", 25: "quality", 26: "difficulty", 27: "negation",
    28: "certainty", 29: "causation", 30: "formality", 31: "speech_act",
}


def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    model = ConceptAutoencoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def encode(model, tokenizer, texts):
    """Returns (flat_normalized [B, 1024], concepts [B, 32, 32])."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        concepts = model.encode(enc["input_ids"], enc["attention_mask"])
    flat = concepts.view(concepts.shape[0], -1)
    return F.normalize(flat, p=2, dim=-1), concepts


def slot_cosine(a, b):
    """Flat cosine similarity between two single-example concept tensors."""
    return flat_similarity_matrix(a, b).item()


def flat_cosine(a, b):
    return F.cosine_similarity(a, b).item()


def main():
    print("Loading model...")
    model = load_model()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # ================================================================
    # 1. Semantic Clustering — within-group vs between-group
    # ================================================================
    print("\n" + "=" * 60)
    print("1. SEMANTIC CLUSTERING")
    print("=" * 60)

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

    group_concepts = {}
    for name, sents in groups.items():
        _, concepts = encode(model, tokenizer, sents)
        group_concepts[name] = concepts

    print("\nWithin-group avg slot-aware similarity:")
    within_sims = []
    for name, concepts in group_concepts.items():
        sims = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                sims.append(slot_cosine(concepts[i:i+1], concepts[j:j+1]))
        avg = np.mean(sims)
        within_sims.append(avg)
        print(f"  {name:>12s}: {avg:.3f}  (range {min(sims):.3f} - {max(sims):.3f})")
    print(f"  {'AVERAGE':>12s}: {np.mean(within_sims):.3f}")

    print("\nBetween-group avg slot-aware similarity:")
    between_sims = []
    group_names = list(group_concepts.keys())
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            ca = group_concepts[group_names[i]]
            cb = group_concepts[group_names[j]]
            sims = []
            for a in range(len(ca)):
                for b in range(len(cb)):
                    sims.append(slot_cosine(ca[a:a+1], cb[b:b+1]))
            avg = np.mean(sims)
            between_sims.append(avg)
    print(f"  {'AVERAGE':>12s}: {np.mean(between_sims):.3f}")
    gap = np.mean(within_sims) - np.mean(between_sims)
    print(f"\n  CLUSTERING GAP: {gap:+.3f} (within - between, want > 0)")

    # ================================================================
    # 2. Direction Consistency — do concept changes form consistent directions?
    # ================================================================
    print("\n" + "=" * 60)
    print("2. DIRECTION CONSISTENCY (per-slot deltas)")
    print("=" * 60)

    directions = {
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
        "size": [
            ("the big cat", "the small cat"),
            ("a large dog", "a small dog"),
            ("the huge house", "the tiny house"),
            ("a massive tree", "a little tree"),
        ],
    }

    for attr, pairs in directions.items():
        # Flat-vector direction consistency
        flat_deltas = []
        # Per-slot direction consistency
        slot_deltas = []
        for pos, neg in pairs:
            v_pos, c_pos = encode(model, tokenizer, [pos])
            v_neg, c_neg = encode(model, tokenizer, [neg])
            flat_delta = F.normalize(v_pos - v_neg, p=2, dim=-1)
            flat_deltas.append(flat_delta)
            slot_delta = c_pos - c_neg  # (1, 32, 32)
            slot_deltas.append(slot_delta)

        # Flat consistency
        flat_cons = []
        for i in range(len(flat_deltas)):
            for j in range(i + 1, len(flat_deltas)):
                flat_cons.append(flat_cosine(flat_deltas[i], flat_deltas[j]))

        # Find which slots change most for this attribute
        avg_slot_change = torch.stack(slot_deltas).squeeze(1).abs().mean(dim=0).mean(dim=-1)  # (32,)
        top_slots = avg_slot_change.topk(5)
        slot_info = ", ".join(f"{SLOT_NAMES[idx.item()]}({avg_slot_change[idx].item():.3f})"
                              for idx in top_slots.indices)

        print(f"  {attr:>12s}: flat_consistency={np.mean(flat_cons):.3f}  "
              f"(range {min(flat_cons):.3f}-{max(flat_cons):.3f})")
        print(f"               top changing slots: {slot_info}")

    # ================================================================
    # 3. Slot Isolation — does changing one concept only affect its slot?
    # ================================================================
    print("\n" + "=" * 60)
    print("3. SLOT ISOLATION (single-concept changes)")
    print("=" * 60)

    isolation_tests = [
        # (sent_a, sent_b, expected_changing_concept, description)
        ("the big cat sat", "the small cat sat", "size", "size change"),
        ("the red car drove fast", "the blue car drove fast", "color", "color change"),
        ("she ran to the store", "she walked to the store", "action_type", "action change"),
        ("the cat is here", "the cat is not here", "negation", "negation flip"),
        ("she ran yesterday", "she runs today", "tense", "tense change"),
        ("the happy man smiled", "the sad man smiled", "emotion", "emotion change"),
        ("three cats played", "seven cats played", "number", "number change"),
        ("it is certainly true", "it might be true", "certainty", "certainty change"),
        ("the movie was great", "the movie was terrible", "sentiment", "sentiment change"),
        ("he walked quickly", "he walked slowly", "manner", "manner change"),
    ]

    for sent_a, sent_b, expected_slot_name, desc in isolation_tests:
        _, c_a = encode(model, tokenizer, [sent_a])
        _, c_b = encode(model, tokenizer, [sent_b])

        # Per-slot cosine similarity
        per_slot_sim = []
        for s in range(32):
            sim = F.cosine_similarity(c_a[0, s:s+1], c_b[0, s:s+1], dim=-1).item()
            per_slot_sim.append(sim)

        # Find most-changed slots
        per_slot_sim = np.array(per_slot_sim)
        changed_mask = per_slot_sim < 0.90
        changed_slots = np.where(changed_mask)[0]
        unchanged_mean = per_slot_sim[~changed_mask].mean() if (~changed_mask).any() else 0

        # Check if expected slot is among the most changed
        expected_idx = [k for k, v in SLOT_NAMES.items() if v == expected_slot_name][0]
        expected_sim = per_slot_sim[expected_idx]
        min_slot = np.argmin(per_slot_sim)

        status = "OK" if expected_idx == min_slot or expected_sim < 0.85 else "??"
        changed_names = [SLOT_NAMES[i] for i in changed_slots[:5]]

        print(f"  [{status}] {desc:>18s}: expected={expected_slot_name}(sim={expected_sim:.2f}), "
              f"most_changed={SLOT_NAMES[min_slot]}(sim={per_slot_sim[min_slot]:.2f}), "
              f"unchanged_mean={unchanged_mean:.3f}")
        if len(changed_slots) > 1:
            print(f"      all changed (<0.90): {changed_names}")

    # ================================================================
    # 4. Analogy Test — a - b + c ≈ d?
    # ================================================================
    print("\n" + "=" * 60)
    print("4. ANALOGY TEST (a - b + c ≈ d?)")
    print("=" * 60)

    analogies = [
        ("she ran", "she runs", "he walked", "he walks", "tense transfer"),
        ("big cat", "cat", "big dog", "dog", "size transfer"),
        ("he is happy", "he is sad", "she is happy", "she is sad", "emotion transfer"),
        ("the cats", "the cat", "the dogs", "the dog", "plural transfer"),
        ("she is not here", "she is here", "he is not there", "he is there", "negation transfer"),
        ("the movie was great", "the movie was terrible",
         "the food was great", "the food was terrible", "sentiment transfer"),
    ]

    for a, b, c, expected_d, label in analogies:
        va, ca = encode(model, tokenizer, [a])
        vb, cb = encode(model, tokenizer, [b])
        vc, cc = encode(model, tokenizer, [c])
        vd, cd = encode(model, tokenizer, [expected_d])

        # Flat-space analogy
        predicted_flat = F.normalize(va - vb + vc, p=2, dim=-1)
        flat_sim = flat_cosine(predicted_flat, vd)

        # Slot-space analogy
        predicted_concepts = ca - cb + cc  # (1, 32, 32)
        slot_sim = slot_cosine(predicted_concepts, cd)

        print(f"  {label:>20s}: flat={flat_sim:.3f}  slot_aware={slot_sim:.3f}  "
              f"| {a} - {b} + {c} ≈ {expected_d}")

    # ================================================================
    # 5. Effective Rank
    # ================================================================
    print("\n" + "=" * 60)
    print("5. EFFECTIVE RANK")
    print("=" * 60)

    all_sents = []
    for sents in groups.values():
        all_sents.extend(sents)
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

    vecs, concepts_all = encode(model, tokenizer, all_sents)

    # Full-space rank
    vecs_np = vecs.numpy()
    vecs_np = vecs_np - vecs_np.mean(axis=0)
    _, s, _ = np.linalg.svd(vecs_np, full_matrices=False)
    var = s ** 2
    cumvar = np.cumsum(var / var.sum())
    rank90 = np.searchsorted(cumvar, 0.90) + 1
    rank95 = np.searchsorted(cumvar, 0.95) + 1
    rank99 = np.searchsorted(cumvar, 0.99) + 1
    print(f"  Full space (1024 dims, {len(all_sents)} samples):")
    print(f"    rank90={rank90}  rank95={rank95}  rank99={rank99}")
    print(f"    Top 10 singular values: {', '.join(f'{v:.2f}' for v in s[:10])}")

    # Per-slot rank
    print(f"\n  Per-slot effective rank (32 dims each):")
    slot_ranks = []
    for slot in range(32):
        slot_vecs = concepts_all[:, slot, :].numpy()
        slot_vecs = slot_vecs - slot_vecs.mean(axis=0)
        _, ss, _ = np.linalg.svd(slot_vecs, full_matrices=False)
        svar = ss ** 2
        scumvar = np.cumsum(svar / svar.sum())
        sr90 = np.searchsorted(scumvar, 0.90) + 1
        slot_ranks.append(sr90)
    # Print in a compact grid
    for row in range(4):
        parts = []
        for col in range(8):
            idx = row * 8 + col
            name = SLOT_NAMES[idx]
            parts.append(f"{name[:6]:>6s}:{slot_ranks[idx]:2d}")
        print(f"    {' | '.join(parts)}")

    # ================================================================
    # 6. Reconstruction Spot-check
    # ================================================================
    print("\n" + "=" * 60)
    print("6. RECONSTRUCTION SPOT-CHECK")
    print("=" * 60)

    recon_tests = [
        "the cat sat on the mat",
        "she runs every morning before breakfast",
        "the dog bit the man",
        "the man bit the dog",
        "artificial intelligence will change the world",
        "three big red cars drove quickly north",
        "he certainly did not enjoy the terrible movie yesterday",
    ]
    for sent in recon_tests:
        enc = tokenizer(sent, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            concepts = model.encode(enc["input_ids"], enc["attention_mask"])
            bos_id = tokenizer.cls_token_id or 101
            generated = [bos_id]
            for _ in range(len(enc["input_ids"][0]) + 10):
                dec_input = torch.tensor([generated])
                logits = model.decode(dec_input, concepts)
                next_token = logits[0, -1].argmax().item()
                if next_token == tokenizer.sep_token_id:
                    break
                generated.append(next_token)
        decoded = tokenizer.decode(generated[1:], skip_special_tokens=True)
        print(f"  IN:  {sent}")
        print(f"  OUT: {decoded}")
        print()

    # ================================================================
    # 7. Word Order Sensitivity
    # ================================================================
    print("=" * 60)
    print("7. WORD ORDER SENSITIVITY")
    print("=" * 60)

    wo_pairs = [
        ("the dog bit the man", "the man bit the dog"),
        ("alice likes bob", "bob likes alice"),
        ("she gave him a book", "he gave her a book"),
        ("the teacher praised the student", "the student praised the teacher"),
        ("the cat chased the mouse", "the mouse chased the cat"),
    ]

    for a, b in wo_pairs:
        _, ca = encode(model, tokenizer, [a])
        _, cb = encode(model, tokenizer, [b])
        s_sim = slot_cosine(ca, cb)

        # Which slots differ most?
        per_slot = []
        for s in range(32):
            sim = F.cosine_similarity(ca[0, s:s+1], cb[0, s:s+1], dim=-1).item()
            per_slot.append(sim)
        per_slot = np.array(per_slot)
        changed = np.where(per_slot < 0.90)[0]
        changed_names = [f"{SLOT_NAMES[i]}({per_slot[i]:.2f})" for i in changed[:6]]

        print(f"  slot_sim={s_sim:.3f}  '{a}' vs '{b}'")
        if changed_names:
            print(f"    changed slots: {', '.join(changed_names)}")
        else:
            print(f"    no slots changed <0.90 (min={per_slot.min():.2f} at {SLOT_NAMES[per_slot.argmin()]})")


if __name__ == "__main__":
    main()
