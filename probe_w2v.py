#!/usr/bin/env python3
"""
Probe and visualize the V28 word2vec embeddings.

Generates interactive HTML visualizations:
  - t-SNE / PCA 2D scatter of word clusters
  - Analogy arithmetic explorer
  - Nearest neighbor explorer
  - Direction analysis (gender, tense, size, sentiment)
  - Semantic similarity heatmaps

Usage:
    python probe_w2v.py [checkpoint_path] [--output probe_w2v.html]
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Load model + vocab
# ---------------------------------------------------------------------------

def load_w2v(ckpt_path, vocab_path=None):
    """Load word2vec embeddings and vocabulary."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Get embeddings
    emb = ckpt["model_state_dict"]["target_embeddings.weight"]
    emb_norm = F.normalize(emb, p=2, dim=-1)

    # Load vocab
    if vocab_path is None:
        vocab_path = os.path.join(os.path.dirname(ckpt_path), "vocab.json")
    with open(vocab_path) as f:
        vdata = json.load(f)

    word2id = vdata["word2id"]
    id2word = {int(i): w for w, i in word2id.items()}
    counts = vdata["counts"] if isinstance(vdata["counts"], list) else \
             [vdata["counts"][str(i)] for i in range(len(word2id))]

    step = ckpt.get("step", 0)
    embed_dim = emb.shape[1]

    print(f"Loaded: {len(word2id):,} words, {embed_dim}d, step {step:,}")
    return emb, emb_norm, word2id, id2word, counts, step


def get_vec(word, emb_norm, word2id):
    if word in word2id:
        return emb_norm[word2id[word]]
    return None


def nearest(vec, emb_norm, id2word, exclude=None, k=10):
    sims = emb_norm @ vec
    if exclude:
        for idx in exclude:
            sims[idx] = -1
    topk = sims.topk(k)
    return [(id2word[i.item()], s.item()) for i, s in zip(topk.indices, topk.values)]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def run_analogies(emb_norm, word2id, id2word):
    """Extended analogy tests."""
    tests = [
        # Royalty
        ("king", "man", "woman", "queen"),
        ("king", "queen", "man", "woman"),
        ("prince", "man", "woman", "princess"),
        # Gender
        ("man", "woman", "boy", "girl"),
        ("father", "mother", "son", "daughter"),
        ("husband", "wife", "brother", "sister"),
        ("he", "she", "his", "her"),
        # Comparative
        ("big", "bigger", "small", "smaller"),
        ("good", "better", "bad", "worse"),
        ("slow", "slower", "fast", "faster"),
        ("tall", "taller", "short", "shorter"),
        # Superlative
        ("good", "best", "bad", "worst"),
        ("big", "biggest", "small", "smallest"),
        # Tense
        ("go", "went", "come", "came"),
        ("see", "saw", "hear", "heard"),
        ("run", "ran", "swim", "swam"),
        ("eat", "ate", "drink", "drank"),
        ("take", "took", "give", "gave"),
        # Geography
        ("france", "paris", "germany", "berlin"),
        ("france", "paris", "italy", "rome"),
        ("japan", "tokyo", "china", "beijing"),
        ("france", "paris", "england", "london"),
        ("france", "french", "spain", "spanish"),
        ("france", "french", "germany", "german"),
        # Plurals
        ("car", "cars", "dog", "dogs"),
        ("child", "children", "man", "men"),
    ]

    results = []
    correct = 0
    total = 0
    for a, b, c, expected in tests:
        va, vb, vc, ve = [get_vec(w, emb_norm, word2id) for w in [a, b, c, expected]]
        if any(v is None for v in [va, vb, vc, ve]):
            results.append((a, b, c, expected, "OOV", [], False))
            continue
        query = F.normalize(vc + vb - va, p=2, dim=-1)
        exclude = [word2id[w] for w in [a, b, c] if w in word2id]
        top = nearest(query, emb_norm, id2word, exclude=exclude, k=5)
        hit = expected in [w for w, _ in top]
        total += 1
        if hit:
            correct += 1
        results.append((a, b, c, expected, top[0][0], top, hit))

    return results, correct, total


def run_directions(emb_norm, word2id):
    """Analyze directional consistency across word pairs."""
    direction_sets = {
        "Gender (M→F)": [
            ("king", "queen"), ("man", "woman"), ("boy", "girl"),
            ("father", "mother"), ("brother", "sister"), ("he", "she"),
            ("husband", "wife"), ("son", "daughter"), ("uncle", "aunt"),
            ("prince", "princess"), ("actor", "actress"),
        ],
        "Tense (present→past)": [
            ("go", "went"), ("run", "ran"), ("see", "saw"),
            ("come", "came"), ("eat", "ate"), ("take", "took"),
            ("give", "gave"), ("know", "knew"), ("think", "thought"),
            ("buy", "bought"), ("find", "found"),
        ],
        "Singular→Plural": [
            ("car", "cars"), ("dog", "dogs"), ("cat", "cats"),
            ("house", "houses"), ("tree", "trees"), ("city", "cities"),
            ("book", "books"), ("year", "years"), ("day", "days"),
        ],
        "Positive→Negative": [
            ("happy", "sad"), ("good", "bad"), ("love", "hate"),
            ("beautiful", "ugly"), ("rich", "poor"), ("strong", "weak"),
            ("fast", "slow"), ("hot", "cold"), ("light", "dark"),
        ],
        "Country→Capital": [
            ("france", "paris"), ("germany", "berlin"), ("italy", "rome"),
            ("japan", "tokyo"), ("spain", "madrid"), ("england", "london"),
            ("russia", "moscow"), ("china", "beijing"),
        ],
        "Country→Language": [
            ("france", "french"), ("germany", "german"), ("spain", "spanish"),
            ("italy", "italian"), ("japan", "japanese"), ("china", "chinese"),
            ("russia", "russian"), ("england", "english"),
        ],
    }

    results = {}
    for name, pairs in direction_sets.items():
        vecs = []
        valid_pairs = []
        for w1, w2 in pairs:
            v1, v2 = get_vec(w1, emb_norm, word2id), get_vec(w2, emb_norm, word2id)
            if v1 is not None and v2 is not None:
                vecs.append(F.normalize(v2 - v1, p=2, dim=-1))
                valid_pairs.append((w1, w2))

        if len(vecs) >= 2:
            # Pairwise cosine similarity of direction vectors
            sims = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    sims.append((vecs[i] * vecs[j]).sum().item())
            avg_consistency = np.mean(sims)
            # Average direction
            avg_dir = F.normalize(torch.stack(vecs).mean(dim=0), p=2, dim=-1)
            results[name] = {
                "consistency": avg_consistency,
                "pairs": valid_pairs,
                "n_pairs": len(valid_pairs),
                "avg_direction": avg_dir,
            }

    return results


def run_clusters(emb_norm, word2id, id2word):
    """Find semantic clusters for curated word groups."""
    groups = {
        "Animals": ["dog", "cat", "horse", "fish", "bird", "wolf", "bear",
                     "lion", "tiger", "elephant", "rabbit", "snake", "eagle"],
        "Countries": ["france", "germany", "england", "spain", "italy", "japan",
                       "china", "russia", "india", "brazil", "canada", "australia"],
        "Colors": ["red", "blue", "green", "yellow", "black", "white",
                    "purple", "orange", "brown", "pink", "gray"],
        "Emotions": ["happy", "sad", "angry", "afraid", "surprised", "love",
                      "hate", "joy", "fear", "hope", "grief", "pride"],
        "Body parts": ["head", "hand", "foot", "arm", "leg", "eye",
                        "ear", "nose", "mouth", "heart", "brain", "bone"],
        "Professions": ["doctor", "teacher", "lawyer", "engineer", "scientist",
                         "artist", "soldier", "farmer", "priest", "judge"],
        "Food": ["bread", "cheese", "meat", "fish", "rice", "fruit",
                  "cake", "soup", "milk", "butter", "salt", "sugar"],
        "Weather": ["rain", "snow", "wind", "storm", "sun", "cloud",
                     "thunder", "fog", "frost", "ice"],
        "Music": ["song", "music", "piano", "guitar", "drum", "violin",
                   "orchestra", "melody", "rhythm", "concert"],
        "Math": ["number", "equation", "formula", "theorem", "proof",
                  "algebra", "geometry", "calculus", "function", "variable"],
    }

    cluster_data = {}
    for name, words in groups.items():
        vecs = []
        valid_words = []
        for w in words:
            v = get_vec(w, emb_norm, word2id)
            if v is not None:
                vecs.append(v)
                valid_words.append(w)
        if len(vecs) >= 3:
            vecs_t = torch.stack(vecs)
            # Within-group similarity
            sim_matrix = vecs_t @ vecs_t.T
            mask = ~torch.eye(len(vecs), dtype=torch.bool)
            within_sim = sim_matrix[mask].mean().item()
            cluster_data[name] = {
                "words": valid_words,
                "within_sim": within_sim,
                "vecs": vecs_t,
            }

    # Between-group similarities
    group_names = list(cluster_data.keys())
    between_sims = {}
    for i, g1 in enumerate(group_names):
        for j, g2 in enumerate(group_names):
            if i < j:
                cross_sim = (cluster_data[g1]["vecs"] @ cluster_data[g2]["vecs"].T).mean().item()
                between_sims[(g1, g2)] = cross_sim

    return cluster_data, between_sims


def run_google_analogies(emb_norm, word2id, id2word, path="data/questions-words.txt"):
    """Run the standard Google analogy benchmark (~19,544 questions)."""
    import os
    if not os.path.exists(path):
        print(f"  Skipping Google analogies: {path} not found")
        return None

    # Parse questions
    categories = []
    current_cat = None
    questions = []
    cat_index = 0
    semantic_cats = set()
    syntactic_cats = set()

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                current_cat = line[2:]
                categories.append(current_cat)
                cat_index = len(categories) - 1
                if cat_index < 5:
                    semantic_cats.add(current_cat)
                else:
                    syntactic_cats.add(current_cat)
            else:
                words = line.lower().split()
                if len(words) == 4:
                    questions.append((current_cat, words[0], words[1], words[2], words[3]))

    # Evaluate
    emb_np = emb_norm.numpy()
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    cat_covered = defaultdict(int)

    for cat, w1, w2, w3, w4 in questions:
        cat_total[cat] += 1
        if w1 not in word2id or w2 not in word2id or w3 not in word2id or w4 not in word2id:
            continue
        cat_covered[cat] += 1
        id1, id2, id3, id4 = word2id[w1], word2id[w2], word2id[w3], word2id[w4]
        vec = emb_np[id2] - emb_np[id1] + emb_np[id3]
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        sims = emb_np @ vec
        sims[id1] = sims[id2] = sims[id3] = -2.0
        if np.argmax(sims) == id4:
            cat_correct[cat] += 1

    # Aggregate
    total_covered = sum(cat_covered.values())
    total_correct = sum(cat_correct.values())
    total_questions = len(questions)

    sem_c = sum(cat_correct[c] for c in semantic_cats)
    sem_cov = sum(cat_covered[c] for c in semantic_cats)
    sem_t = sum(cat_total[c] for c in semantic_cats)
    syn_c = sum(cat_correct[c] for c in syntactic_cats)
    syn_cov = sum(cat_covered[c] for c in syntactic_cats)
    syn_t = sum(cat_total[c] for c in syntactic_cats)

    per_cat = []
    for cat in categories:
        c = cat_correct[cat]
        cov = cat_covered[cat]
        t = cat_total[cat]
        acc = c / cov * 100 if cov else 0
        cov_pct = cov / t * 100 if t else 0
        kind = "semantic" if cat in semantic_cats else "syntactic"
        per_cat.append({"name": cat, "correct": c, "covered": cov, "total": t,
                        "accuracy": acc, "coverage": cov_pct, "kind": kind})

    return {
        "total_correct": total_correct, "total_covered": total_covered,
        "total_questions": total_questions,
        "accuracy": total_correct / total_covered * 100 if total_covered else 0,
        "coverage": total_covered / total_questions * 100,
        "sem_accuracy": sem_c / sem_cov * 100 if sem_cov else 0,
        "syn_accuracy": syn_c / syn_cov * 100 if syn_cov else 0,
        "sem_coverage": sem_cov / sem_t * 100 if sem_t else 0,
        "syn_coverage": syn_cov / syn_t * 100 if syn_t else 0,
        "per_category": per_cat,
    }


def run_arithmetic(emb_norm, word2id, id2word):
    """Fun arithmetic experiments."""
    experiments = [
        ("king - man + woman", ["king", "-man", "+woman"]),
        ("paris - france + germany", ["paris", "-france", "+germany"]),
        ("tokyo - japan + italy", ["tokyo", "-japan", "+italy"]),
        ("bigger - big + small", ["bigger", "-big", "+small"]),
        ("went - go + come", ["went", "-go", "+come"]),
        ("queen - woman + man", ["queen", "-woman", "+man"]),
        ("swimming - swim + run", ["swimming", "-swim", "+run"]),
        ("dogs - dog + cat", ["dogs", "-dog", "+cat"]),
        ("french - france + spain", ["french", "-france", "+spain"]),
        ("brother - man + woman", ["brother", "-man", "+woman"]),
        ("worst - bad + good", ["worst", "-bad", "+good"]),
        ("happy - good + bad", ["happy", "-good", "+bad"]),
    ]

    results = []
    for desc, terms in experiments:
        vec = None
        words_used = []
        valid = True
        for term in terms:
            sign = 1
            word = term
            if term.startswith("-"):
                sign = -1
                word = term[1:]
            elif term.startswith("+"):
                word = term[1:]
            v = get_vec(word, emb_norm, word2id)
            if v is None:
                valid = False
                break
            words_used.append(word)
            if vec is None:
                vec = sign * v
            else:
                vec = vec + sign * v

        if valid and vec is not None:
            query = F.normalize(vec, p=2, dim=-1)
            exclude = [word2id[w] for w in words_used if w in word2id]
            top = nearest(query, emb_norm, id2word, exclude=exclude, k=8)
            results.append((desc, top))
        else:
            results.append((desc, [("OOV", 0)]))

    return results


def compute_tsne(emb_norm, word2id, id2word, counts, n_words=500):
    """Compute t-SNE for top words."""
    from sklearn.manifold import TSNE

    # Get top N words by frequency (skip very common stopwords)
    stopwords = {'the', 'of', 'and', 'to', 'a', 'in', 'that', 'i', 'is', 'was',
                 'it', 'he', 'for', 'with', 'as', 'his', 'by', 'on', 'be', 'at',
                 'this', 'had', 'not', 'but', 'or', 'from', 'they', 'she', 'an',
                 'which', 'you', 'her', 'were', 'all', 'their', 'has', 'would',
                 'been', 'have', 'one', 'if', 'who', 'more', 'no', 'so', 'we',
                 'do', 'my', 'are', 'me', 'what', 'there', 'them', 'can', 'than',
                 'its', 'will', 'did', 'him', 'about', 'up', 'out', 'into'}

    # Curated words to always include
    curated = [
        # Royalty
        "king", "queen", "prince", "princess", "emperor",
        # Gender
        "man", "woman", "boy", "girl", "father", "mother", "son", "daughter",
        "brother", "sister", "husband", "wife",
        # Countries
        "france", "germany", "england", "spain", "italy", "japan", "china",
        "russia", "india", "brazil",
        # Capitals
        "paris", "berlin", "london", "rome", "tokyo", "moscow", "madrid",
        # Animals
        "dog", "cat", "horse", "fish", "bird", "wolf", "lion", "tiger", "bear",
        # Emotions
        "happy", "sad", "angry", "love", "hate", "fear", "joy",
        # Colors
        "red", "blue", "green", "yellow", "black", "white",
        # Size
        "big", "small", "large", "tiny", "huge",
        "bigger", "smaller", "biggest", "smallest",
        # Comparatives
        "good", "better", "best", "bad", "worse", "worst",
        "fast", "faster", "fastest", "slow", "slower",
        # Tense
        "go", "went", "gone", "run", "ran", "see", "saw",
        "come", "came", "eat", "ate", "take", "took",
        # Tech
        "computer", "internet", "software", "algorithm",
        # Professions
        "doctor", "teacher", "lawyer", "scientist", "soldier",
        # Food
        "bread", "cheese", "meat", "rice", "cake", "milk",
        # Nature
        "water", "fire", "earth", "sun", "moon", "star", "river", "mountain",
        "tree", "flower", "rain", "snow", "wind",
    ]

    # Build word list: curated + top frequent non-stopwords
    selected = []
    selected_set = set()
    for w in curated:
        if w in word2id and w not in selected_set:
            selected.append(w)
            selected_set.add(w)

    # Add frequent words
    word_freq = [(id2word[i], counts[i]) for i in range(len(counts))
                 if id2word.get(i, "") not in stopwords and id2word.get(i, "") not in selected_set]
    word_freq.sort(key=lambda x: -x[1])
    for w, _ in word_freq:
        if len(selected) >= n_words:
            break
        if w not in selected_set and len(w) > 2:
            selected.append(w)
            selected_set.add(w)

    # Get vectors
    vecs = torch.stack([emb_norm[word2id[w]] for w in selected]).numpy()

    # t-SNE
    print(f"Computing t-SNE for {len(selected)} words...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    coords = tsne.fit_transform(vecs)

    # Assign categories for coloring
    categories = {}
    cat_map = {
        "royalty": ["king", "queen", "prince", "princess", "emperor", "monarch", "throne", "crown", "royal"],
        "gender_m": ["man", "boy", "father", "son", "brother", "husband", "he", "him", "his", "uncle"],
        "gender_f": ["woman", "girl", "mother", "daughter", "sister", "wife", "she", "her", "aunt"],
        "country": ["france", "germany", "england", "spain", "italy", "japan", "china", "russia",
                     "india", "brazil", "canada", "australia", "america"],
        "capital": ["paris", "berlin", "london", "rome", "tokyo", "moscow", "madrid", "beijing"],
        "animal": ["dog", "cat", "horse", "fish", "bird", "wolf", "lion", "tiger", "bear",
                    "elephant", "rabbit", "snake", "eagle", "deer", "sheep", "cow"],
        "emotion": ["happy", "sad", "angry", "love", "hate", "fear", "joy", "hope",
                     "glad", "grief", "pride", "afraid"],
        "color": ["red", "blue", "green", "yellow", "black", "white", "purple",
                   "orange", "brown", "pink", "gray"],
        "nature": ["water", "fire", "earth", "sun", "moon", "star", "river", "mountain",
                    "tree", "flower", "rain", "snow", "wind", "sea", "ocean", "forest"],
        "food": ["bread", "cheese", "meat", "rice", "cake", "milk", "fruit", "fish",
                  "butter", "sugar", "salt", "soup", "wine", "beer"],
    }
    for cat, words in cat_map.items():
        for w in words:
            if w in selected_set:
                categories[w] = cat

    return selected, coords, categories


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def generate_html(step, analogies, analogy_correct, analogy_total,
                  directions, clusters, between_sims, arithmetic,
                  tsne_words, tsne_coords, tsne_cats, emb_norm, word2id, id2word,
                  google_benchmark=None, version_title=None):
    """Generate interactive HTML visualization."""
    vtitle = version_title or "V28"
    vshort = vtitle.split()[0] if vtitle else "V28"  # e.g. "V33" from "V33 Mixed SG+CBOW"

    # Build t-SNE data for chart
    cat_colors = {
        "royalty": "#FFD700", "gender_m": "#4169E1", "gender_f": "#FF69B4",
        "country": "#228B22", "capital": "#FF4500", "animal": "#8B4513",
        "emotion": "#DC143C", "color": "#9370DB", "nature": "#2E8B57",
        "food": "#FF8C00", "other": "#888888",
    }

    tsne_data = []
    for i, word in enumerate(tsne_words):
        cat = tsne_cats.get(word, "other")
        tsne_data.append({
            "x": float(tsne_coords[i, 0]),
            "y": float(tsne_coords[i, 1]),
            "word": word,
            "cat": cat,
            "color": cat_colors.get(cat, "#888888"),
        })

    # Analogy table HTML
    analogy_rows = ""
    for a, b, c, expected, got, top, hit in analogies:
        if got == "OOV":
            analogy_rows += f"<tr class='oov'><td>{a}:{b} :: {c}:?</td><td colspan='3'>OOV</td></tr>\n"
            continue
        top_str = ", ".join(f"{w}({s:.2f})" for w, s in top[:5])
        cls = "hit" if hit else "miss"
        analogy_rows += f"<tr class='{cls}'><td>{a}:{b} :: {c}:?</td><td>{expected}</td><td>{got}</td><td>{top_str}</td></tr>\n"

    # Direction table
    dir_rows = ""
    for name, data in directions.items():
        pairs_str = ", ".join(f"{a}→{b}" for a, b in data["pairs"][:6])
        dir_rows += f"<tr><td>{name}</td><td>{data['consistency']:.3f}</td><td>{data['n_pairs']}</td><td>{pairs_str}</td></tr>\n"

    # Cluster table
    cluster_rows = ""
    sorted_clusters = sorted(clusters.items(), key=lambda x: -x[1]["within_sim"])
    for name, data in sorted_clusters:
        words_str = ", ".join(data["words"][:10])
        cluster_rows += f"<tr><td>{name}</td><td>{data['within_sim']:.3f}</td><td>{len(data['words'])}</td><td>{words_str}</td></tr>\n"

    # Between-group similarity table
    group_names = sorted(clusters.keys())
    between_header = "<th></th>" + "".join(f"<th>{g[:8]}</th>" for g in group_names)
    between_rows = ""
    for g1 in group_names:
        row = f"<td><b>{g1[:12]}</b></td>"
        for g2 in group_names:
            if g1 == g2:
                sim = clusters[g1]["within_sim"]
                row += f"<td class='diag'>{sim:.2f}</td>"
            else:
                key = (min(g1, g2), max(g1, g2))
                sim = between_sims.get(key, between_sims.get((g2, g1), 0))
                row += f"<td>{sim:.2f}</td>"
        between_rows += f"<tr>{row}</tr>\n"

    # Arithmetic results
    arith_rows = ""
    for desc, top in arithmetic:
        top_str = ", ".join(f"{w}({s:.2f})" for w, s in top[:6])
        arith_rows += f"<tr><td>{desc}</td><td><b>{top[0][0]}</b></td><td>{top_str}</td></tr>\n"

    # Build benchmark section if available
    benchmark_html = ""
    benchmark_chart_js = ""
    if google_benchmark:
        gb = google_benchmark
        # Per-category rows
        bench_cat_rows = ""
        for cat in gb["per_category"]:
            cls = "sem" if cat["kind"] == "semantic" else "syn"
            bench_cat_rows += (f"<tr class='{cls}'><td>{cat['name']}</td>"
                              f"<td>{cat['correct']:,}/{cat['covered']:,}</td>"
                              f"<td>{cat['accuracy']:.1f}%</td>"
                              f"<td>{cat['coverage']:.0f}%</td>"
                              f"<td>{cat['kind']}</td></tr>\n")

        benchmark_html = f"""
<div class="section">
<h2>Google Analogy Benchmark (Standard Test)</h2>
<p>The standard word2vec evaluation: {gb['total_questions']:,} analogy questions across {len(gb['per_category'])} categories.
Format: A:B :: C:? — find D such that the A-to-B relationship mirrors C-to-D.</p>

<div class="stats">
  <div class="stat"><div class="value">{gb['accuracy']:.1f}%</div><div class="label">Overall Accuracy</div></div>
  <div class="stat"><div class="value">{gb['sem_accuracy']:.1f}%</div><div class="label">Semantic</div></div>
  <div class="stat"><div class="value">{gb['syn_accuracy']:.1f}%</div><div class="label">Syntactic</div></div>
  <div class="stat"><div class="value">{gb['coverage']:.0f}%</div><div class="label">Coverage</div></div>
  <div class="stat"><div class="value">{gb['total_correct']:,}/{gb['total_covered']:,}</div><div class="label">Correct/Covered</div></div>
</div>

<h3>Comparison with Published Models</h3>
<p>All models evaluated on the same Google analogy test set (questions-words.txt, ~19.5K questions).</p>
<div class="chart-container" style="max-width: 800px;">
  <canvas id="benchmark-chart" height="350"></canvas>
</div>
<table style="max-width: 800px; margin-top: 10px; font-size: 0.85em;">
<tr><th>Model</th><th>Corpus</th><th>Vocab</th><th>Dim</th><th>Overall</th><th>Semantic</th><th>Syntactic</th></tr>
<tr><td><b>{vshort} (ours)</b></td><td>OpenWebText subset</td><td>100K</td><td>300</td>
    <td><b>{gb['accuracy']:.1f}%</b></td><td>{gb['sem_accuracy']:.1f}%</td><td>{gb['syn_accuracy']:.1f}%</td></tr>
<tr><td>word2vec (Mikolov 2013)</td><td>Google News 100B</td><td>3M</td><td>300</td>
    <td>61.0%</td><td>~65%</td><td>~57%</td></tr>
<tr><td>GloVe (Pennington 2014)</td><td>Common Crawl 42B</td><td>1.9M</td><td>300</td>
    <td>75.0%</td><td>~81%</td><td>~70%</td></tr>
<tr><td>GloVe (Pennington 2014)</td><td>Wikipedia 6B</td><td>400K</td><td>300</td>
    <td>71.7%</td><td>~77%</td><td>~67%</td></tr>
<tr><td>FastText (Bojanowski 2017)</td><td>Wikipedia 16B</td><td>2.5M</td><td>300</td>
    <td>77.8%</td><td>~77%</td><td>~78%</td></tr>
</table>
<p style="font-size: 0.8em; color: #888; margin-top: 5px;">
Note: Published models use 10-100x more training data. {vshort} uses a small OpenWebText subset (~2B tokens).
Vocab coverage also matters — our 100K vocab covers {gb['coverage']:.0f}% of test questions vs near-100% for larger vocabs.</p>

<h3>Per-Category Breakdown</h3>
<div class="chart-container" style="max-width: 900px;">
  <canvas id="category-chart" height="300"></canvas>
</div>
<table style="font-size: 0.85em;">
<tr><th>Category</th><th>Score</th><th>Accuracy</th><th>Coverage</th><th>Type</th></tr>
{bench_cat_rows}
</table>
</div>
"""

        # Chart.js for benchmark comparison
        cat_names = [c["name"][:20] for c in gb["per_category"]]
        cat_accs = [round(c["accuracy"], 1) for c in gb["per_category"]]
        cat_colors_list = ["#4fc3f7" if c["kind"] == "semantic" else "#FFD700" for c in gb["per_category"]]

        benchmark_chart_js = f"""
// Benchmark comparison bar chart
const benchCtx = document.getElementById('benchmark-chart').getContext('2d');
new Chart(benchCtx, {{
    type: 'bar',
    data: {{
        labels: ['{vshort} (ours)', 'word2vec\\n(Google News)', 'GloVe\\n(Common Crawl)', 'GloVe\\n(Wiki 6B)', 'FastText\\n(Wiki 16B)'],
        datasets: [
            {{
                label: 'Overall',
                data: [{gb['accuracy']:.1f}, 61.0, 75.0, 71.7, 77.8],
                backgroundColor: ['#FFD700', '#4fc3f7', '#4fc3f7', '#4fc3f7', '#4fc3f7'],
                borderColor: ['#FFD700', '#4fc3f7', '#4fc3f7', '#4fc3f7', '#4fc3f7'],
                borderWidth: 1,
            }},
            {{
                label: 'Semantic',
                data: [{gb['sem_accuracy']:.1f}, 65, 81, 77, 77],
                backgroundColor: 'rgba(79, 195, 247, 0.3)',
                borderColor: '#4fc3f7',
                borderWidth: 1,
            }},
            {{
                label: 'Syntactic',
                data: [{gb['syn_accuracy']:.1f}, 57, 70, 67, 78],
                backgroundColor: 'rgba(255, 215, 0, 0.3)',
                borderColor: '#FFD700',
                borderWidth: 1,
            }}
        ]
    }},
    options: {{
        responsive: true,
        plugins: {{
            title: {{ display: true, text: 'Google Analogy Benchmark — Overall / Semantic / Syntactic', color: '#e0e0e0', font: {{ size: 14 }} }},
            legend: {{ labels: {{ color: '#ccc' }} }}
        }},
        scales: {{
            y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'Accuracy (%)', color: '#ccc' }},
                  ticks: {{ color: '#aaa' }}, grid: {{ color: '#333' }} }},
            x: {{ ticks: {{ color: '#ccc', font: {{ size: 11 }} }}, grid: {{ display: false }} }}
        }}
    }}
}});

// Per-category bar chart
const catCtx = document.getElementById('category-chart').getContext('2d');
new Chart(catCtx, {{
    type: 'bar',
    data: {{
        labels: {json.dumps(cat_names)},
        datasets: [{{
            label: 'Accuracy (%)',
            data: {json.dumps(cat_accs)},
            backgroundColor: {json.dumps(cat_colors_list)},
            borderWidth: 0,
        }}]
    }},
    options: {{
        responsive: true,
        indexAxis: 'y',
        plugins: {{
            title: {{ display: true, text: 'Per-Category Accuracy (blue=semantic, gold=syntactic)', color: '#e0e0e0' }},
            legend: {{ display: false }}
        }},
        scales: {{
            x: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'Accuracy (%)', color: '#ccc' }},
                  ticks: {{ color: '#aaa' }}, grid: {{ color: '#333' }} }},
            y: {{ ticks: {{ color: '#ccc', font: {{ size: 11 }} }}, grid: {{ display: false }} }}
        }}
    }}
}});
"""

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>Word2vec {vtitle} — Embedding Probe ({step:,} steps)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
h1 {{ color: #FFD700; }} h2 {{ color: #4fc3f7; border-bottom: 1px solid #333; padding-bottom: 5px; }}
h3 {{ color: #aaa; margin-top: 20px; }}
.stats {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
.stat {{ background: #16213e; padding: 15px 25px; border-radius: 8px; text-align: center; }}
.stat .value {{ font-size: 2em; font-weight: bold; color: #FFD700; }}
.stat .label {{ color: #888; font-size: 0.9em; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.9em; }}
th, td {{ padding: 6px 10px; border: 1px solid #333; text-align: left; }}
th {{ background: #16213e; color: #4fc3f7; }}
tr.hit {{ background: #1a3a1a; }} tr.miss {{ background: #3a1a1a; }}
tr.oov {{ color: #666; }}
tr.sem {{ background: rgba(79, 195, 247, 0.05); }}
tr.syn {{ background: rgba(255, 215, 0, 0.05); }}
td.diag {{ background: #16213e; font-weight: bold; }}
.chart-container {{ background: #16213e; border-radius: 8px; padding: 15px; margin: 15px 0; }}
canvas {{ max-height: 600px; }}
.section {{ margin: 30px 0; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }}
.legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 0.85em; }}
.legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
</style>
</head><body>
<h1>Word2vec {vtitle} — Embedding Geometry Probe</h1>
<p>300d, 100K whole-word vocabulary, trained {step:,} steps on OpenWebText subset (~2B tokens).</p>

<div class="stats">
  <div class="stat"><div class="value">{analogy_correct}/{analogy_total}</div><div class="label">Custom Analogies</div></div>
  <div class="stat"><div class="value">{analogy_correct/max(analogy_total,1)*100:.0f}%</div><div class="label">Custom Accuracy</div></div>
  {"<div class='stat'><div class='value'>" + f"{google_benchmark['accuracy']:.1f}%" + "</div><div class='label'>Google Benchmark</div></div>" if google_benchmark else ""}
  <div class="stat"><div class="value">{len(tsne_words):,}</div><div class="label">Vocab Visualized</div></div>
  <div class="stat"><div class="value">300d</div><div class="label">Embedding Dim</div></div>
</div>

{benchmark_html}

<div class="section">
<h2>t-SNE Visualization (Top {len(tsne_words)} Words)</h2>
<div class="legend">
  {"".join(f'<div class="legend-item"><div class="legend-dot" style="background:{c}"></div>{n}</div>' for n, c in cat_colors.items() if n != "other")}
</div>
<div class="chart-container">
  <canvas id="tsne-chart" height="600"></canvas>
</div>
</div>

<div class="section">
<h2>Vector Arithmetic</h2>
<table><tr><th>Expression</th><th>Result</th><th>Top Matches</th></tr>
{arith_rows}
</table>
</div>

<div class="section">
<h2>Analogy Tests ({analogy_correct}/{analogy_total} = {analogy_correct/max(analogy_total,1)*100:.0f}%)</h2>
<table><tr><th>Analogy</th><th>Expected</th><th>Got</th><th>Top 5</th></tr>
{analogy_rows}
</table>
</div>

<div class="section">
<h2>Directional Consistency</h2>
<p>How consistently word pairs share the same direction vector (1.0 = perfect, 0.0 = random).</p>
<table><tr><th>Direction</th><th>Consistency</th><th>Pairs</th><th>Examples</th></tr>
{dir_rows}
</table>
</div>

<div class="section">
<h2>Semantic Clusters</h2>
<table><tr><th>Category</th><th>Within-Sim</th><th>Words</th><th>Members</th></tr>
{cluster_rows}
</table>
</div>

<div class="section">
<h2>Inter-Group Similarity Matrix</h2>
<p>Diagonal = within-group similarity. Off-diagonal = between-group similarity.</p>
<table><tr>{between_header}</tr>
{between_rows}
</table>
</div>

<script>
const tsneData = {json.dumps(tsne_data)};

// t-SNE scatter plot
const ctx = document.getElementById('tsne-chart').getContext('2d');
new Chart(ctx, {{
    type: 'scatter',
    data: {{
        datasets: [{{
            data: tsneData.map(d => ({{x: d.x, y: d.y}})),
            backgroundColor: tsneData.map(d => d.color),
            pointRadius: tsneData.map(d => d.cat === 'other' ? 2 : 5),
            pointHoverRadius: 8,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            legend: {{ display: false }},
            tooltip: {{
                callbacks: {{
                    label: function(ctx) {{
                        const d = tsneData[ctx.dataIndex];
                        return d.word + ' (' + d.cat + ')';
                    }}
                }}
            }}
        }},
        scales: {{
            x: {{ display: false }},
            y: {{ display: false }}
        }}
    }},
    plugins: [{{
        afterDraw: function(chart) {{
            const ctx = chart.ctx;
            ctx.font = '11px sans-serif';
            ctx.fillStyle = '#ccc';
            ctx.textAlign = 'center';
            chart.data.datasets[0].data.forEach((point, i) => {{
                const d = tsneData[i];
                if (d.cat !== 'other') {{
                    const meta = chart.getDatasetMeta(0);
                    const pos = meta.data[i];
                    ctx.fillText(d.word, pos.x, pos.y - 8);
                }}
            }});
        }}
    }}]
}});

{benchmark_chart_js}
</script>
</body></html>"""

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/word2vec_v28/latest.pt")
    parser.add_argument("--vocab", default=None)
    parser.add_argument("--output", default="docs/probe_w2v.html")
    parser.add_argument("--n-tsne", type=int, default=500)
    parser.add_argument("--title", default=None, help="Version title, e.g. 'V33 Mixed SG+CBOW'")
    args = parser.parse_args()

    emb, emb_norm, word2id, id2word, counts, step = load_w2v(args.checkpoint, args.vocab)

    print("\n=== Analogies ===")
    analogies, a_correct, a_total = run_analogies(emb_norm, word2id, id2word)
    for a, b, c, expected, got, top, hit in analogies:
        mark = "✓" if hit else ("OOV" if got == "OOV" else "✗")
        print(f"  {a}:{b} :: {c}:? → {got} [expect: {expected}] {mark}")
    print(f"  Accuracy: {a_correct}/{a_total} ({a_correct/max(a_total,1)*100:.0f}%)")

    print("\n=== Directions ===")
    directions = run_directions(emb_norm, word2id)
    for name, data in directions.items():
        print(f"  {name}: consistency={data['consistency']:.3f} ({data['n_pairs']} pairs)")

    print("\n=== Clusters ===")
    clusters, between_sims = run_clusters(emb_norm, word2id, id2word)
    for name, data in sorted(clusters.items(), key=lambda x: -x[1]["within_sim"]):
        print(f"  {name}: within_sim={data['within_sim']:.3f} ({len(data['words'])} words)")

    print("\n=== Arithmetic ===")
    arithmetic = run_arithmetic(emb_norm, word2id, id2word)
    for desc, top in arithmetic:
        print(f"  {desc} → {top[0][0]} ({top[0][1]:.3f})")

    print("\n=== Google Analogy Benchmark ===")
    google_benchmark = run_google_analogies(emb_norm, word2id, id2word)
    if google_benchmark:
        print(f"  Overall: {google_benchmark['accuracy']:.1f}% ({google_benchmark['total_correct']}/{google_benchmark['total_covered']})")
        print(f"  Semantic: {google_benchmark['sem_accuracy']:.1f}%  Syntactic: {google_benchmark['syn_accuracy']:.1f}%")
        print(f"  Coverage: {google_benchmark['coverage']:.0f}%")

    print("\n=== t-SNE ===")
    tsne_words, tsne_coords, tsne_cats = compute_tsne(
        emb_norm, word2id, id2word, counts, n_words=args.n_tsne)

    print("\nGenerating HTML report...")
    html = generate_html(step, analogies, a_correct, a_total,
                         directions, clusters, between_sims, arithmetic,
                         tsne_words, tsne_coords, tsne_cats,
                         emb_norm, word2id, id2word,
                         google_benchmark=google_benchmark,
                         version_title=args.title)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Report saved: {args.output}")
