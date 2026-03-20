#!/usr/bin/env python3
"""
Discover what clusters the V25 concept space has learned.
Encodes a large diverse sentence set, runs k-means and shows what groups together.
Usage: python probe_clusters.py [checkpoint_path] [--k 10]
"""

import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from transformers import AutoTokenizer
from concept_model import ConceptAutoencoderV24, ConceptConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Diverse sentence pool — many topics, lengths, structures
SENTENCES = [
    # Animals
    "the cat is sleeping on the couch.",
    "the dog ran across the yard.",
    "birds fly south in winter.",
    "the horse galloped across the field.",
    "fish swim in the ocean.",
    "the rabbit hid under the bush.",
    "wolves hunt in packs at night.",
    "the eagle soared above the mountains.",
    "dolphins are very intelligent animals.",
    "the snake slithered through the grass.",

    # Weather
    "it is raining outside today.",
    "the weather is sunny and warm.",
    "a storm is coming tonight.",
    "snow is falling from the sky.",
    "the wind is blowing hard.",
    "thunder rumbled in the distance.",
    "the fog was thick this morning.",
    "it was a beautiful clear day.",
    "hail damaged the cars in the lot.",
    "the temperature dropped below freezing.",

    # Food / cooking
    "the pizza was delicious.",
    "she cooked pasta for dinner.",
    "i love chocolate cake.",
    "the soup was too salty.",
    "he ordered a hamburger.",
    "the bread was fresh from the oven.",
    "she baked cookies for the party.",
    "the restaurant served excellent seafood.",
    "he grilled steaks on the barbecue.",
    "the salad had too much dressing.",

    # Science / nature
    "the earth orbits the sun.",
    "atoms are made of protons and neutrons.",
    "gravity pulls objects toward the ground.",
    "water freezes at zero degrees.",
    "light travels faster than sound.",
    "the moon causes ocean tides.",
    "plants convert sunlight into energy.",
    "dna contains the genetic code.",
    "volcanoes erupt when pressure builds.",
    "the universe is constantly expanding.",

    # Emotions / feelings
    "she was very happy today.",
    "he felt sad and lonely.",
    "they were angry about the decision.",
    "the children were excited.",
    "she felt nervous before the test.",
    "he was proud of his daughter.",
    "the news made her anxious.",
    "they celebrated with great joy.",
    "he was disappointed by the result.",
    "she felt grateful for the help.",

    # Work / office
    "the meeting starts at nine.",
    "she finished the report on time.",
    "he got a promotion last week.",
    "the deadline is tomorrow morning.",
    "they hired three new employees.",
    "the boss approved the project.",
    "she sent the email to the team.",
    "the office closes at five.",
    "he quit his job yesterday.",
    "the company announced layoffs.",

    # Sports
    "the team won the championship.",
    "he scored three goals in the match.",
    "she ran a marathon last weekend.",
    "the game was tied in overtime.",
    "he hit a home run in the ninth.",
    "the swimmer broke the world record.",
    "they lost the final game.",
    "the coach was fired after the season.",
    "she won the gold medal.",
    "the crowd cheered loudly.",

    # History / politics
    "the president signed the new law.",
    "the war lasted four years.",
    "the empire fell in the fifth century.",
    "the revolution changed everything.",
    "the treaty was signed in paris.",
    "the election results surprised everyone.",
    "the king ruled for thirty years.",
    "the colony declared independence.",
    "parliament voted on the new bill.",
    "the protest drew thousands of people.",

    # Technology
    "the computer crashed again.",
    "she updated the software.",
    "the internet connection is slow.",
    "he built a website from scratch.",
    "the phone battery is dead.",
    "the robot assembled the parts.",
    "the algorithm runs in linear time.",
    "she wrote the code in python.",
    "the server went down at midnight.",
    "they launched the new app.",

    # Daily life / mundane
    "he woke up early this morning.",
    "she drove to the store.",
    "the kids went to school.",
    "he took the bus to work.",
    "she washed the dishes after dinner.",
    "they watched television all evening.",
    "he walked the dog before breakfast.",
    "she picked up the children from school.",
    "he mowed the lawn on saturday.",
    "they went to bed early.",

    # Travel / places
    "the flight was delayed two hours.",
    "she visited paris last summer.",
    "the hotel had a beautiful view.",
    "they drove across the country.",
    "the train arrived on time.",
    "he climbed the mountain in three days.",
    "the beach was crowded.",
    "she took a boat across the lake.",
    "the museum was closed on monday.",
    "they camped in the national park.",

    # Health / body
    "the doctor examined the patient.",
    "she broke her arm last week.",
    "he has a terrible headache.",
    "the surgery was successful.",
    "she takes medicine every morning.",
    "the hospital was very busy.",
    "he needs to lose weight.",
    "the dentist cleaned his teeth.",
    "she felt dizzy and sat down.",
    "the vaccine prevented the disease.",

    # Education
    "the teacher explained the lesson.",
    "she passed the exam with high marks.",
    "the students studied all night.",
    "he graduated from the university.",
    "the professor gave a long lecture.",
    "she wrote her thesis in six months.",
    "the school was built in nineteen fifty.",
    "he learned to play the piano.",
    "the library had thousands of books.",
    "she got a scholarship to college.",

    # Short sentences
    "he ran.", "she smiled.", "it rained.", "they left.", "we won.",
    "stop.", "help.", "go away.", "come here.", "sit down.",

    # Questions
    "what time is it?",
    "where did he go?",
    "how does this work?",
    "why is the sky blue?",
    "who wrote this book?",
    "can you help me?",
    "did they win the game?",
    "when does the movie start?",
    "how much does it cost?",
    "what happened yesterday?",

    # Exclamations
    "that is amazing!",
    "what a beautiful day!",
    "i cannot believe it!",
    "this is terrible!",
    "how wonderful!",
    "watch out!",
    "congratulations!",
    "oh no!",
    "fantastic!",
    "incredible!",

    # Negative statements
    "he did not go.",
    "she is not happy.",
    "they never arrived.",
    "it does not work.",
    "we cannot stay.",
    "nobody came to the party.",
    "nothing happened.",
    "she has no money.",
    "he never eats breakfast.",
    "there is no hope.",

    # Numbers / math
    "two plus two equals four.",
    "the population is seven billion.",
    "the distance is three hundred miles.",
    "the price dropped by ten percent.",
    "he scored ninety five on the test.",
    "the building has fifty floors.",
    "she worked twelve hours a day.",
    "the recipe calls for two cups of flour.",
    "the temperature is twenty degrees.",
    "he waited for three hours.",

    # Literature / narrative style
    "once upon a time there was a king.",
    "the old man sat by the fire.",
    "she whispered his name in the dark.",
    "the ship sailed into the unknown.",
    "he turned the page and continued reading.",
    "the castle stood on a hill above the town.",
    "the stranger appeared at the door.",
    "she held the letter with trembling hands.",
    "the forest was silent and still.",
    "he knew this was the end.",
]


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "v24_state_dict" in ckpt:
        config = ConceptConfig(**ckpt["v24_config"])
        model = ConceptAutoencoderV24(config).to(DEVICE).eval()
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["v24_state_dict"].items()}
        model.load_state_dict(state, strict=True)
        step = ckpt["step"]
    else:
        config = ConceptConfig(**ckpt["config"])
        model = ConceptAutoencoderV24(config).to(DEVICE).eval()
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state, strict=True)
        step = ckpt["step"]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer, step


@torch.no_grad()
def encode_all(model, tokenizer, sentences, batch_size=32):
    """Encode all sentences -> flat normalized vectors."""
    all_vecs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        enc = tokenizer(batch, max_length=64, padding=True, truncation=True,
                        return_tensors="pt").to(DEVICE)
        concepts = model.encode(enc["input_ids"], enc["attention_mask"])
        flat = concepts.view(concepts.shape[0], -1)
        normed = F.normalize(flat, p=2, dim=-1)
        all_vecs.append(normed.cpu().numpy())
    return np.concatenate(all_vecs, axis=0)


def kmeans(vecs, k, n_iter=100):
    """Simple k-means on normalized vectors (cosine distance via dot product)."""
    n, d = vecs.shape
    # Init: k-means++
    centers = np.zeros((k, d))
    idx = np.random.randint(n)
    centers[0] = vecs[idx]
    for c in range(1, k):
        # Distance to nearest center
        sims = vecs @ centers[:c].T  # (n, c)
        dists = 1.0 - sims.max(axis=1)
        dists = np.maximum(dists, 0)
        probs = dists / (dists.sum() + 1e-10)
        idx = np.random.choice(n, p=probs)
        centers[c] = vecs[idx]

    for _ in range(n_iter):
        # Assign
        sims = vecs @ centers.T  # (n, k)
        labels = sims.argmax(axis=1)
        # Update
        new_centers = np.zeros_like(centers)
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                new_centers[c] = vecs[mask].mean(axis=0)
                new_centers[c] /= (np.linalg.norm(new_centers[c]) + 1e-10)
            else:
                new_centers[c] = centers[c]
        centers = new_centers

    # Final assign
    sims = vecs @ centers.T
    labels = sims.argmax(axis=1)
    return labels, centers


def find_best_k(vecs, k_range=range(3, 25)):
    """Use silhouette-like score to find best k."""
    best_k, best_score = 3, -1

    for k in k_range:
        labels, centers = kmeans(vecs, k)
        counts = Counter(labels)

        # Skip if any cluster is too small or too dominant
        sizes = [counts[c] for c in range(k)]
        if min(sizes) < 2 or max(sizes) > len(vecs) * 0.6:
            continue

        # Average within-cluster similarity - average between-cluster similarity
        within, between = [], []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                s = vecs[i] @ vecs[j]
                if labels[i] == labels[j]:
                    within.append(s)
                else:
                    between.append(s)

        if within and between:
            score = np.mean(within) - np.mean(between)
            if score > best_score:
                best_score = score
                best_k = k

    return best_k, best_score


def print_clusters(sentences, vecs, labels, centers, k):
    """Print each cluster with its members, sorted by distance to center."""
    print(f"\n{'='*70}")
    print(f"  {k} CLUSTERS FOUND")
    print(f"{'='*70}")

    for c in range(k):
        mask = labels == c
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue

        # Sort by similarity to center (most representative first)
        sims_to_center = vecs[idxs] @ centers[c]
        order = np.argsort(-sims_to_center)

        print(f"\n  --- Cluster {c} ({len(idxs)} sentences) ---")

        # Show the most central sentence as the "theme"
        central_idx = idxs[order[0]]
        print(f"  Theme: \"{sentences[central_idx]}\"")
        print()

        # Show all members
        for rank, o in enumerate(order):
            idx = idxs[o]
            sim = sims_to_center[o]
            print(f"    {sim:.3f}  \"{sentences[idx]}\"")

    # Inter-cluster similarity matrix
    print(f"\n{'='*70}")
    print(f"  INTER-CLUSTER SIMILARITY")
    print(f"{'='*70}")
    header = "         " + "".join(f"  C{c:<5d}" for c in range(k))
    print(header)
    for ci in range(k):
        row = f"  C{ci:<3d}   "
        for cj in range(k):
            s = centers[ci] @ centers[cj]
            row += f"  {s:+.3f}"
        print(row)


def analyze_cluster_properties(sentences, vecs, labels, k):
    """Try to figure out what each cluster is keying on."""
    print(f"\n{'='*70}")
    print(f"  CLUSTER PROPERTY ANALYSIS")
    print(f"{'='*70}")

    for c in range(k):
        idxs = np.where(labels == c)[0]
        if len(idxs) < 2:
            continue

        cluster_sents = [sentences[i] for i in idxs]

        # Length stats
        lengths = [len(s.split()) for s in cluster_sents]
        avg_len = np.mean(lengths)

        # Punctuation
        ends_period = sum(1 for s in cluster_sents if s.endswith(".")) / len(cluster_sents)
        ends_question = sum(1 for s in cluster_sents if s.endswith("?")) / len(cluster_sents)
        ends_excl = sum(1 for s in cluster_sents if s.endswith("!")) / len(cluster_sents)

        # Common words (excluding stopwords)
        stopwords = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'in', 'on', 'at',
                     'to', 'of', 'and', 'for', 'it', 'he', 'she', 'they', 'his', 'her',
                     'this', 'that', 'with', 'from', 'by', 'be', 'has', 'had', 'have',
                     'not', 'but', 'or', 'if', 'as', 'no', 'so', 'we', 'do', 'did'}
        words = Counter()
        for s in cluster_sents:
            for w in s.lower().replace(".", "").replace("?", "").replace("!", "").split():
                if w not in stopwords and len(w) > 1:
                    words[w] += 1
        top_words = words.most_common(8)

        # Has negation?
        neg_count = sum(1 for s in cluster_sents
                       if any(w in s.lower() for w in ['not', 'never', 'nobody', 'nothing', 'cannot', "n't", 'no ']))

        print(f"\n  Cluster {c} ({len(idxs)} sents):")
        print(f"    Avg length: {avg_len:.1f} words")
        print(f"    Punctuation: {ends_period:.0%} period, {ends_question:.0%} question, {ends_excl:.0%} exclamation")
        print(f"    Negations: {neg_count}/{len(idxs)}")
        print(f"    Top words: {', '.join(f'{w}({n})' for w, n in top_words)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/concept_v25/latest.pt")
    parser.add_argument("--k", type=int, default=0, help="Number of clusters (0 = auto)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    model, tokenizer, step = load_model(args.checkpoint)
    print(f"Step {step:,} | {len(SENTENCES)} sentences")

    print("Encoding sentences...")
    vecs = encode_all(model, tokenizer, SENTENCES)
    print(f"Vectors: {vecs.shape}")

    if args.k > 0:
        k = args.k
    else:
        print("Finding best k...")
        k, score = find_best_k(vecs)
        print(f"Best k={k} (gap score={score:.3f})")

    labels, centers = kmeans(vecs, k)
    print_clusters(SENTENCES, vecs, labels, centers, k)
    analyze_cluster_properties(SENTENCES, vecs, labels, k)

    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}")
