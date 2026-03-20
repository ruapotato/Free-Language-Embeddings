#!/usr/bin/env python3
"""Probe the geometry of the concept space — 32 slots x 32 dims.

V11+: measures emergent geometry (no forced slot assignments).
Tests analogies, clustering, direction consistency, reconstruction on diverse inputs.

Usage:
    python probe_geometry.py                                    # auto-detect latest
    python probe_geometry.py checkpoints/concept_v11/step_600000.pt
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizerFast
from concept_model import (ConceptAutoencoder, ConceptAutoencoderV10,
                           ConceptAutoencoderV13, ConceptAutoencoderV14,
                           ConceptConfig, flat_similarity_matrix)

import os
CKPT = None
for candidate in ["checkpoints/concept_v15/latest.pt",
                   "checkpoints/concept_v14/latest.pt",
                   "checkpoints/concept_v13/latest.pt",
                   "checkpoints/concept_v11/latest.pt",
                   "checkpoints/concept_v11/step_600000.pt",
                   "checkpoints/concept_v10/latest.pt",
                   "checkpoints/concept_v9/latest.pt"]:
    if os.path.exists(candidate):
        CKPT = candidate
        break
if len(sys.argv) > 1:
    CKPT = sys.argv[1]

DEVICE = "cpu"


def load_model():
    if not CKPT:
        raise FileNotFoundError("No checkpoint found")
    print(f"  Checkpoint: {CKPT}")
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    config = ConceptConfig(**ckpt["config"])
    version = ckpt.get("version", "")
    keys_str = str(list(ckpt["model_state_dict"].keys()))
    if version == "v14" or "es_dec_layers" in keys_str:
        print("  Model: ConceptAutoencoderV14 (Hydra — 5 decoder heads)")
        model = ConceptAutoencoderV14(config)
    elif version == "v13" or "fr_dec_layers.0.self_attn.q_proj.weight" in ckpt["model_state_dict"]:
        print("  Model: ConceptAutoencoderV13 (dual decoder EN+FR)")
        model = ConceptAutoencoderV13(config)
    elif version == "v10" or "par_dec_layers" in keys_str:
        print("  Model: ConceptAutoencoderV10 (parallel decoder)")
        model = ConceptAutoencoderV10(config)
    else:
        print("  Model: ConceptAutoencoder (autoregressive decoder)")
        model = ConceptAutoencoder(config)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    step = ckpt.get("step", "?")
    em_ema = ckpt.get("em_ema", "?")
    print(f"  Step: {step}  EM EMA: {em_ema}")
    return model


def encode(model, tokenizer, texts):
    """Returns (flat_normalized [B, 1024], concepts [B, 32, 32])."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        concepts = model.encode(enc["input_ids"], enc["attention_mask"])
    flat = concepts.view(concepts.shape[0], -1)
    return F.normalize(flat, p=2, dim=-1), concepts


def flat_cosine(a, b):
    return F.cosine_similarity(a, b).item()


def slot_cosine(a, b):
    return flat_similarity_matrix(a, b).item()


def reconstruct_batch(model, tokenizer, texts):
    """Encode and decode a batch of texts, return list of decoded strings.

    For parallel decoders (V10, V13): decode one-at-a-time to avoid seq_len
    padding mismatch — the decoder is very sensitive to position query length.
    """
    is_v10 = isinstance(model, (ConceptAutoencoderV10, ConceptAutoencoderV13, ConceptAutoencoderV14))
    results = []
    with torch.no_grad():
        if is_v10:
            for text in texts:
                enc = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
                concepts = model.encode(enc["input_ids"], enc["attention_mask"])
                logits = model.decode_parallel(concepts, seq_len=enc["input_ids"].shape[1])
                predicted = logits.argmax(dim=-1)
                mask = enc["attention_mask"][0].bool()
                pred = predicted[0][mask]
                results.append(tokenizer.decode(pred, skip_special_tokens=True))
        else:
            enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            concepts = model.encode(enc["input_ids"], enc["attention_mask"])
            for i in range(len(texts)):
                bos_id = tokenizer.cls_token_id or 101
                generated = [bos_id]
                c = concepts[i:i+1]
                for _ in range(len(enc["input_ids"][i]) + 10):
                    dec_input = torch.tensor([generated])
                    logits = model.decode(dec_input, c)
                    next_token = logits[0, -1].argmax().item()
                    if next_token == tokenizer.sep_token_id:
                        break
                    generated.append(next_token)
                results.append(tokenizer.decode(generated[1:], skip_special_tokens=True))
    return results


def word_overlap(a, b):
    """Token-level overlap percentage."""
    wa, wb = a.lower().split(), b.lower().split()
    if not wa:
        return 0.0
    matches = sum(1 for i, w in enumerate(wa) if i < len(wb) and wb[i] == w)
    return matches / len(wa)


def main():
    print("Loading model...")
    model = load_model()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    scores = {}

    # ================================================================
    # 1. ANALOGY TEST — a - b + c ≈ d?
    # ================================================================
    print("\n" + "=" * 60)
    print("1. ANALOGY TEST (a - b + c ≈ d?)")
    print("=" * 60)

    analogies = [
        # Tense
        ("she ran", "she runs", "he walked", "he walks", "tense: ran/runs → walked/walks"),
        ("they played outside", "they play outside", "she danced well", "she dances well", "tense: played/play → danced/dances"),
        # Sentiment
        ("the movie was great", "the movie was terrible", "the food was great", "the food was terrible", "sentiment: great/terrible (movie→food)"),
        ("she loves the city", "she hates the city", "he loves the job", "he hates the job", "sentiment: loves/hates (city→job)"),
        # Negation
        ("she is not here", "she is here", "he is not there", "he is there", "negation: not here/here → not there/there"),
        ("they did not win", "they did win", "she did not leave", "she did leave", "negation: did not/did (win→leave)"),
        # Plurality
        ("the cats", "the cat", "the dogs", "the dog", "plural: cats/cat → dogs/dog"),
        ("the birds flew", "the bird flew", "the cars drove", "the car drove", "plural: birds/bird → cars/car"),
        # Subject swap
        ("he is happy", "she is happy", "he is tired", "she is tired", "subject: he/she (happy→tired)"),
        ("alice likes bob", "bob likes alice", "the cat chased the mouse", "the mouse chased the cat", "role swap: subject↔object"),
        # Size
        ("big cat", "cat", "big dog", "dog", "size: big+cat/cat → big+dog/dog"),
        ("the huge house", "the tiny house", "the huge tree", "the tiny tree", "size: huge/tiny (house→tree)"),
        # Action
        ("she ran to the store", "she walked to the store", "he ran to the park", "he walked to the park", "action: ran/walked (store→park)"),
        # Formality / register
        ("the experiment yielded results", "the test gave results", "the analysis yielded data", "the analysis gave data", "formality: yielded/gave"),
    ]

    analogy_scores = []
    for a, b, c, expected_d, label in analogies:
        va, _ = encode(model, tokenizer, [a])
        vb, _ = encode(model, tokenizer, [b])
        vc, _ = encode(model, tokenizer, [c])
        vd, _ = encode(model, tokenizer, [expected_d])

        predicted_flat = F.normalize(va - vb + vc, p=2, dim=-1)
        sim = flat_cosine(predicted_flat, vd)
        analogy_scores.append(sim)

        quality = "★" if sim >= 0.9 else "●" if sim >= 0.7 else "○"
        print(f"  {quality} {sim:.3f}  {label}")
        print(f"          {a} - {b} + {c} ≈ {expected_d}")

    avg = np.mean(analogy_scores)
    scores["analogy_avg"] = avg
    print(f"\n  AVERAGE: {avg:.3f}  (★ ≥0.9, ● ≥0.7, ○ <0.7)")
    print(f"  ≥0.9: {sum(1 for s in analogy_scores if s >= 0.9)}/{len(analogy_scores)}")
    print(f"  ≥0.7: {sum(1 for s in analogy_scores if s >= 0.7)}/{len(analogy_scores)}")

    # ================================================================
    # 2. SEMANTIC CLUSTERING
    # ================================================================
    print("\n" + "=" * 60)
    print("2. SEMANTIC CLUSTERING")
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

    within_sims = []
    for name, concepts in group_concepts.items():
        sims = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                sims.append(slot_cosine(concepts[i:i+1], concepts[j:j+1]))
        avg = np.mean(sims)
        within_sims.append(avg)
        print(f"  {name:>12s}: {avg:.3f}  (range {min(sims):.3f} - {max(sims):.3f})")
    print(f"  {'WITHIN AVG':>12s}: {np.mean(within_sims):.3f}")

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
            between_sims.append(np.mean(sims))
    print(f"  {'BETWEEN AVG':>12s}: {np.mean(between_sims):.3f}")

    gap = np.mean(within_sims) - np.mean(between_sims)
    scores["clustering_gap"] = gap
    print(f"\n  CLUSTERING GAP: {gap:+.4f}  (want > 0, ideally > 0.05)")

    # ================================================================
    # 3. DIRECTION CONSISTENCY (emergent — no slot name assumptions)
    # ================================================================
    print("\n" + "=" * 60)
    print("3. DIRECTION CONSISTENCY (emergent)")
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

    dir_scores = []
    for attr, pairs in directions.items():
        flat_deltas = []
        for pos, neg in pairs:
            v_pos, _ = encode(model, tokenizer, [pos])
            v_neg, _ = encode(model, tokenizer, [neg])
            flat_delta = F.normalize(v_pos - v_neg, p=2, dim=-1)
            flat_deltas.append(flat_delta)

        # Pairwise consistency of direction vectors
        flat_cons = []
        for i in range(len(flat_deltas)):
            for j in range(i + 1, len(flat_deltas)):
                flat_cons.append(flat_cosine(flat_deltas[i], flat_deltas[j]))

        avg_con = np.mean(flat_cons)
        dir_scores.append(avg_con)
        quality = "★" if avg_con >= 0.5 else "●" if avg_con >= 0.2 else "○"
        print(f"  {quality} {attr:>12s}: {avg_con:.3f}  (range {min(flat_cons):.3f} - {max(flat_cons):.3f})")

    avg_dir = np.mean(dir_scores)
    scores["direction_consistency"] = avg_dir
    print(f"\n  AVERAGE: {avg_dir:.3f}  (★ ≥0.5, ● ≥0.2, ○ <0.2)")

    # ================================================================
    # 4. RECONSTRUCTION — diverse inputs (prose, code, math, logic)
    # ================================================================
    print("\n" + "=" * 60)
    print("4. RECONSTRUCTION (diverse inputs)")
    print("=" * 60)

    recon_tests = [
        # Short prose
        ("the cat sat on the mat", "short prose"),
        ("she runs every morning before breakfast", "medium prose"),
        ("the dog bit the man", "short prose"),
        ("the man bit the dog", "short prose (word order)"),
        # Longer prose
        ("he certainly did not enjoy the terrible movie yesterday", "long prose"),
        ("alice likes bob but bob does not like alice", "logic/names"),
        ("the quick brown fox jumps over the lazy dog near the river", "long prose"),
        ("three big red cars drove quickly north", "adjective stack"),
        ("the purple man licked the sucker", "unusual content"),
        # Code
        ("def fibonacci ( n ) : return n if n < 2 else fibonacci ( n - 1 ) + fibonacci ( n - 2 )", "code: recursion"),
        ("for i in range ( 10 ) : print ( i * 2 )", "code: loop"),
        ("x = [ i for i in range ( n ) if i % 2 == 0 ]", "code: list comp"),
        ("import os ; path = os . path . join ( dir , file )", "code: import"),
        # Math
        ("the derivative of x squared plus three x equals two x plus three", "math: calculus"),
        ("the sum of one plus two plus three equals six", "math: arithmetic"),
        ("if x is greater than zero then x squared is positive", "math: logic"),
        # Logic
        ("if all dogs are animals and all animals breathe then all dogs breathe", "logic: syllogism"),
        ("either it rains or it does not rain", "logic: excluded middle"),
        ("if she studies then she passes the exam", "logic: conditional"),
        # Technical
        ("the tcp handshake requires three packets syn synack and ack", "technical: networking"),
        ("malloc allocates memory on the heap and returns a pointer", "technical: systems"),
    ]

    texts = [t for t, _ in recon_tests]
    categories = [c for _, c in recon_tests]
    decoded_all = reconstruct_batch(model, tokenizer, texts)

    perfect = 0
    high = 0  # >=90%
    total = len(recon_tests)
    recon_scores = []

    for text, category, decoded in zip(texts, categories, decoded_all):
        overlap = word_overlap(text, decoded)
        recon_scores.append(overlap)

        if overlap >= 1.0:
            perfect += 1
            marker = "  ✓"
        elif overlap >= 0.9:
            high += 1
            marker = f"  ~ {overlap:.0%}"
        else:
            marker = f"  ✗ {overlap:.0%}"

        if overlap < 1.0:
            print(f"  [{category}]{marker}")
            print(f"    IN:  {text}")
            print(f"    OUT: {decoded}")
        else:
            print(f"  [{category}]{marker}")

    avg_recon = np.mean(recon_scores)
    scores["recon_avg"] = avg_recon
    scores["recon_perfect"] = perfect
    print(f"\n  PERFECT: {perfect}/{total} ({perfect/total:.0%})")
    print(f"  ≥90%:   {perfect + high}/{total} ({(perfect + high)/total:.0%})")
    print(f"  AVG OVERLAP: {avg_recon:.3f}")

    # ================================================================
    # 4b. FR TRANSLATION (V13/V14)
    # ================================================================
    if isinstance(model, (ConceptAutoencoderV13, ConceptAutoencoderV14)):
        print("\n" + "=" * 60)
        print("4b. FR TRANSLATION")
        print("=" * 60)
        from transformers import CamembertTokenizerFast
        fr_tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")

        fr_tests = [
            ("The vote will take place tomorrow", "Le vote aura lieu demain"),
            ("I would like to thank the Commission", "Je voudrais remercier la Commission"),
            ("The situation is very serious", "La situation est très grave"),
            ("We need to find a solution", "Nous devons trouver une solution"),
            ("The cat sat on the mat", "Le chat s'est assis sur le tapis"),
            ("She runs every morning", "Elle court chaque matin"),
            ("He is not happy today", "Il n'est pas content aujourd'hui"),
            ("The big red car drove quickly", "La grande voiture rouge a roulé vite"),
        ]

        decode_fr = model.decode_fr if isinstance(model, ConceptAutoencoderV14) else model.decode_parallel_fr
        for en, ref_fr in fr_tests:
            enc_en = tokenizer([en], padding=True, truncation=True, max_length=128, return_tensors="pt")
            ref_enc = fr_tokenizer([ref_fr], padding=True, truncation=True, max_length=128, return_tensors="pt")
            fr_seq_len = ref_enc["input_ids"].shape[1]
            with torch.no_grad():
                concepts = model.encode(enc_en["input_ids"], enc_en["attention_mask"])
                fr_logits = decode_fr(concepts, seq_len=fr_seq_len)
                fr_predicted = fr_logits.argmax(dim=-1)
                decoded_fr = fr_tokenizer.decode(fr_predicted[0], skip_special_tokens=True)
            print(f"  EN:  {en}")
            print(f"  FR:  {decoded_fr}")
            print(f"  ref: {ref_fr}")
            print()

    # ================================================================
    # 4c. ES TRANSLATION (V14 only)
    # ================================================================
    if isinstance(model, ConceptAutoencoderV14):
        print("\n" + "=" * 60)
        print("4c. ES TRANSLATION")
        print("=" * 60)
        from transformers import AutoTokenizer
        es_tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

        es_tests = [
            ("The vote will take place tomorrow", "La votación tendrá lugar mañana"),
            ("The situation is very serious", "La situación es muy grave"),
            ("The cat sat on the mat", "El gato se sentó en la alfombra"),
            ("She runs every morning", "Ella corre cada mañana"),
            ("He is not happy today", "Él no está contento hoy"),
            ("We need to find a solution", "Necesitamos encontrar una solución"),
        ]

        for en, ref_es in es_tests:
            enc_en = tokenizer([en], padding=True, truncation=True, max_length=128, return_tensors="pt")
            ref_enc = es_tokenizer([ref_es], padding=True, truncation=True, max_length=128, return_tensors="pt")
            es_seq_len = ref_enc["input_ids"].shape[1]
            with torch.no_grad():
                concepts = model.encode(enc_en["input_ids"], enc_en["attention_mask"])
                es_logits = model.decode_es(concepts, seq_len=es_seq_len)
                es_predicted = es_logits.argmax(dim=-1)
                decoded_es = es_tokenizer.decode(es_predicted[0], skip_special_tokens=True)
            print(f"  EN:  {en}")
            print(f"  ES:  {decoded_es}")
            print(f"  ref: {ref_es}")
            print()

    # ================================================================
    # 4d. SEMANTIC PARSE (V14 only)
    # ================================================================
    if isinstance(model, ConceptAutoencoderV14):
        print("\n" + "=" * 60)
        print("4d. SEMANTIC PARSE")
        print("=" * 60)

        parse_tests = [
            ("the dog bit the man", "subject : the dog | action : bite | object : the man"),
            ("she runs every morning", "subject : she | action : run | location : every morning"),
            ("he did not enjoy the movie", "subject : he | action : enjoy | negation : true | object : the movie"),
            ("the cat chased the mouse quickly", "subject : the cat | action : chase | object : the mouse | manner : quickly"),
            ("alice likes bob", "subject : alice | action : like | object : bob"),
            ("the big red car drove north", "subject : the big red car | action : drive | direction : north"),
        ]

        for en, ref_parse in parse_tests:
            enc_en = tokenizer([en], padding=True, truncation=True, max_length=128, return_tensors="pt")
            ref_enc = tokenizer([ref_parse], padding=True, truncation=True, max_length=128, return_tensors="pt")
            parse_seq_len = ref_enc["input_ids"].shape[1]
            with torch.no_grad():
                concepts = model.encode(enc_en["input_ids"], enc_en["attention_mask"])
                parse_logits = model.decode_parse(concepts, seq_len=parse_seq_len)
                parse_predicted = parse_logits.argmax(dim=-1)
                decoded_parse = tokenizer.decode(parse_predicted[0], skip_special_tokens=True)
            print(f"  EN:    {en}")
            print(f"  Parse: {decoded_parse}")
            print(f"  ref:   {ref_parse}")
            print()

    # ================================================================
    # 5. WORD ORDER SENSITIVITY
    # ================================================================
    print("\n" + "=" * 60)
    print("5. WORD ORDER SENSITIVITY")
    print("=" * 60)

    wo_pairs = [
        ("the dog bit the man", "the man bit the dog"),
        ("alice likes bob", "bob likes alice"),
        ("she gave him a book", "he gave her a book"),
        ("the teacher praised the student", "the student praised the teacher"),
        ("the cat chased the mouse", "the mouse chased the cat"),
    ]

    wo_sims = []
    for a, b in wo_pairs:
        va, _ = encode(model, tokenizer, [a])
        vb, _ = encode(model, tokenizer, [b])
        sim = flat_cosine(va, vb)
        wo_sims.append(sim)
        # Lower = better differentiation
        quality = "★" if sim < 0.85 else "●" if sim < 0.95 else "○"
        print(f"  {quality} {sim:.3f}  '{a}' vs '{b}'")

    avg_wo = np.mean(wo_sims)
    scores["word_order_avg_sim"] = avg_wo
    print(f"\n  AVG SIMILARITY: {avg_wo:.3f}  (lower = better differentiation)")
    print(f"  (★ <0.85 = distinct, ● <0.95 = some distinction, ○ ≥0.95 = too similar)")

    # ================================================================
    # 6. EFFECTIVE RANK
    # ================================================================
    print("\n" + "=" * 60)
    print("6. EFFECTIVE RANK")
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

    vecs, _ = encode(model, tokenizer, all_sents)
    vecs_np = vecs.numpy()
    vecs_np = vecs_np - vecs_np.mean(axis=0)
    _, s, _ = np.linalg.svd(vecs_np, full_matrices=False)
    var = s ** 2
    cumvar = np.cumsum(var / var.sum())
    rank90 = int(np.searchsorted(cumvar, 0.90) + 1)
    rank95 = int(np.searchsorted(cumvar, 0.95) + 1)
    rank99 = int(np.searchsorted(cumvar, 0.99) + 1)
    scores["rank90"] = rank90
    scores["rank95"] = rank95

    print(f"  Full space (1024 dims, {len(all_sents)} samples):")
    print(f"    rank90={rank90}  rank95={rank95}  rank99={rank99}")
    print(f"    Top 10 singular values: {', '.join(f'{v:.2f}' for v in s[:10])}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Analogy avg:          {scores['analogy_avg']:.3f}  (want > 0.8)")
    print(f"  Clustering gap:       {scores['clustering_gap']:+.4f}  (want > 0.05)")
    print(f"  Direction consistency: {scores['direction_consistency']:.3f}  (want > 0.3)")
    print(f"  Recon perfect:        {scores['recon_perfect']}/{total}  ({scores['recon_perfect']/total:.0%})")
    print(f"  Recon avg overlap:    {scores['recon_avg']:.3f}")
    print(f"  Word order sim:       {scores['word_order_avg_sim']:.3f}  (want < 0.85)")
    print(f"  Effective rank:       {scores['rank90']}/{scores['rank95']} (90%/95%)")


if __name__ == "__main__":
    main()
