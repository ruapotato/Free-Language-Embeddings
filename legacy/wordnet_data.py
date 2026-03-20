#!/usr/bin/env python3
"""
WordNet-based data generators for V20 concept autoencoder.

Generates sentence pairs with known semantic relationships:
1. Noun hierarchy pairs: sentences differing by one noun, with WordNet path distance
2. Adjective axis pairs: sentences differing by one adjective (antonym/similar-to)
3. Verb troponym chains: sentences with verbs along specificity chains

All generators use template slot-filling with train/test vocabulary splits
to prevent memorization (lesson from V15/V18).

Usage:
    python wordnet_data.py              # Run probe: show sample data + statistics
    python wordnet_data.py --full       # Full probe with all generators
"""

import random
import itertools
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import nltk
from nltk.corpus import wordnet as wn

# Ensure WordNet data available
try:
    wn.synsets("dog")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


# ---------------------------------------------------------------------------
# Concrete noun vocabulary (things that work in simple sentence templates)
# ---------------------------------------------------------------------------
NOUNS = [
    # Animals
    "dog", "cat", "horse", "cow", "pig", "sheep", "chicken", "duck", "eagle",
    "hawk", "sparrow", "robin", "fish", "shark", "whale", "dolphin", "frog",
    "snake", "lizard", "turtle", "rabbit", "mouse", "rat", "bear", "wolf",
    "fox", "deer", "lion", "tiger", "elephant", "monkey", "gorilla", "ant",
    "bee", "butterfly", "spider",
    # Vehicles
    "car", "truck", "bus", "bicycle", "motorcycle", "boat", "ship", "airplane",
    "helicopter", "train",
    # Buildings/places
    "house", "building", "church", "school", "hospital", "store", "restaurant",
    "hotel", "castle", "bridge", "tower", "barn",
    # Furniture/objects
    "chair", "table", "bed", "desk", "lamp", "clock", "phone", "computer",
    "book", "pen", "cup", "plate", "bottle", "box", "bag", "key", "door",
    "window", "mirror", "picture",
    # Food
    "apple", "banana", "orange", "bread", "cake", "pizza", "sandwich",
    "cheese", "egg", "rice", "soup", "salad",
    # Nature
    "tree", "flower", "grass", "river", "lake", "mountain", "rock", "cloud",
    "rain", "snow", "sun", "moon", "star",
    # People
    "boy", "girl", "man", "woman", "child", "baby", "teacher", "doctor",
    "soldier", "king", "queen", "farmer", "student",
    # Body parts
    "hand", "foot", "head", "eye", "ear", "nose", "mouth", "arm", "leg",
    # Clothing
    "hat", "shirt", "shoe", "coat", "dress", "ring",
    # Tools/instruments
    "hammer", "knife", "sword", "gun", "drum", "guitar", "piano",
]

# Adjective antonym pairs (axis-defining)
ADJ_ANTONYMS = [
    ("big", "small"), ("large", "tiny"), ("tall", "short"), ("long", "brief"),
    ("hot", "cold"), ("warm", "cool"), ("fast", "slow"), ("quick", "sluggish"),
    ("happy", "sad"), ("joyful", "miserable"), ("cheerful", "gloomy"),
    ("good", "bad"), ("nice", "awful"), ("beautiful", "ugly"),
    ("strong", "weak"), ("hard", "soft"), ("heavy", "light"),
    ("old", "young"), ("new", "ancient"), ("modern", "outdated"),
    ("rich", "poor"), ("expensive", "cheap"),
    ("bright", "dark"), ("loud", "quiet"), ("clean", "dirty"),
    ("safe", "dangerous"), ("simple", "complex"), ("easy", "difficult"),
    ("wide", "narrow"), ("thick", "thin"), ("deep", "shallow"),
    ("wet", "dry"), ("smooth", "rough"), ("sharp", "dull"),
    ("brave", "cowardly"), ("kind", "cruel"), ("honest", "dishonest"),
    ("polite", "rude"), ("calm", "anxious"), ("proud", "humble"),
]

# Adjective similarity clusters (same direction, different intensity)
ADJ_INTENSITY = [
    ["warm", "hot", "scorching"],
    ["cool", "cold", "freezing"],
    ["big", "large", "enormous", "gigantic"],
    ["small", "tiny", "microscopic"],
    ["good", "great", "excellent", "superb"],
    ["bad", "terrible", "horrible", "atrocious"],
    ["happy", "delighted", "ecstatic"],
    ["sad", "miserable", "devastated"],
    ["fast", "rapid", "lightning"],
    ["slow", "sluggish", "glacial"],
    ["old", "ancient", "prehistoric"],
    ["young", "youthful", "juvenile"],
    ["loud", "noisy", "deafening"],
    ["quiet", "silent", "mute"],
    ["bright", "brilliant", "blinding"],
    ["dark", "dim", "pitch-black"],
]

# Verb troponym chains (general -> specific)
VERB_CHAINS = [
    ["move", "walk", "march"],
    ["move", "walk", "stroll"],
    ["move", "walk", "stride"],
    ["move", "run", "sprint"],
    ["move", "run", "jog"],
    ["move", "crawl", "slither"],
    ["move", "swim", "dive"],
    ["move", "fly", "soar"],
    ["look", "stare", "glare"],
    ["look", "glance", "peek"],
    ["look", "gaze", "scrutinize"],
    ["speak", "whisper", "murmur"],
    ["speak", "shout", "scream"],
    ["speak", "sing", "chant"],
    ["speak", "mumble", "stammer"],
    ["eat", "devour", "gorge"],
    ["eat", "nibble", "peck"],
    ["eat", "chew", "gnaw"],
    ["hit", "punch", "jab"],
    ["hit", "kick", "stamp"],
    ["hit", "slap", "smack"],
    ["cut", "slice", "dice"],
    ["cut", "chop", "hack"],
    ["hold", "grip", "clutch"],
    ["hold", "grasp", "seize"],
    ["laugh", "giggle", "cackle"],
    ["laugh", "chuckle", "snicker"],
    ["cry", "weep", "sob"],
    ["cry", "wail", "howl"],
    ["think", "ponder", "contemplate"],
    ["think", "wonder", "speculate"],
]

# Sentence templates for noun swaps
NOUN_TEMPLATES = [
    "the {noun} is on the table",
    "I saw a {noun} in the park",
    "the {noun} was very old",
    "she bought a new {noun}",
    "the {noun} fell on the ground",
    "there is a {noun} near the door",
    "he found a {noun} in the box",
    "the {noun} was broken",
    "a large {noun} sat in the corner",
    "the small {noun} was hidden",
    "we need a better {noun}",
    "the {noun} belonged to her",
    "they painted the {noun} blue",
    "the {noun} moved slowly",
    "a beautiful {noun} appeared",
]

# Templates for adjective swaps
ADJ_TEMPLATES = [
    "the {adj} house stood on the hill",
    "she wore a {adj} dress",
    "it was a {adj} day",
    "the {adj} man walked slowly",
    "we found a {adj} stone",
    "the {adj} river flowed past",
    "he told a {adj} story",
    "the {adj} child smiled",
    "a {adj} wind blew through the valley",
    "the {adj} car drove away",
    "she had a {adj} voice",
    "the {adj} dog barked loudly",
    "they lived in a {adj} town",
    "the {adj} bird sang at dawn",
    "he gave a {adj} answer",
]

# Templates for verb swaps (subject + verb + optional complement)
VERB_TEMPLATES = [
    "the man {verb}ed down the street",
    "she {verb}ed across the room",
    "the child {verb}ed in the garden",
    "they {verb}ed along the path",
    "he {verb}ed through the forest",
    "the woman {verb}ed toward the door",
    "the soldier {verb}ed over the hill",
    "the boy {verb}ed around the corner",
]

# Simple present tense templates (for verbs that don't conjugate well with -ed)
VERB_TEMPLATES_PRESENT = [
    "the man {verb}s down the street",
    "she {verb}s across the room",
    "the child {verb}s in the garden",
    "they {verb} along the path",
    "he {verb}s through the forest",
    "the woman {verb}s toward the door",
]

# Pre-conjugated verb forms to avoid bad auto-conjugation
VERB_PAST = {
    "move": "moved", "walk": "walked", "march": "marched", "stroll": "strolled",
    "stride": "strode", "run": "ran", "sprint": "sprinted", "jog": "jogged",
    "crawl": "crawled", "slither": "slithered", "swim": "swam", "dive": "dove",
    "fly": "flew", "soar": "soared", "look": "looked", "stare": "stared",
    "glare": "glared", "glance": "glanced", "peek": "peeked", "gaze": "gazed",
    "scrutinize": "scrutinized", "speak": "spoke", "whisper": "whispered",
    "murmur": "murmured", "shout": "shouted", "scream": "screamed",
    "sing": "sang", "chant": "chanted", "mumble": "mumbled", "stammer": "stammered",
    "eat": "ate", "devour": "devoured", "gorge": "gorged", "nibble": "nibbled",
    "peck": "pecked", "chew": "chewed", "gnaw": "gnawed", "hit": "hit",
    "punch": "punched", "jab": "jabbed", "kick": "kicked", "stamp": "stamped",
    "slap": "slapped", "smack": "smacked", "cut": "cut", "slice": "sliced",
    "dice": "diced", "chop": "chopped", "hack": "hacked", "hold": "held",
    "grip": "gripped", "clutch": "clutched", "grasp": "grasped", "seize": "seized",
    "laugh": "laughed", "giggle": "giggled", "cackle": "cackled",
    "chuckle": "chuckled", "snicker": "snickered", "cry": "cried",
    "weep": "wept", "sob": "sobbed", "wail": "wailed", "howl": "howled",
    "think": "thought", "ponder": "pondered", "contemplate": "contemplated",
    "wonder": "wondered", "speculate": "speculated",
}

# Past tense templates using pre-conjugated forms
VERB_TEMPLATES_PAST = [
    "the man {past} down the street",
    "she {past} across the room",
    "the child {past} in the garden",
    "they {past} along the path",
    "he {past} through the forest",
    "the woman {past} toward the door",
    "the soldier {past} over the hill",
    "the boy {past} around the corner",
]


# ---------------------------------------------------------------------------
# WordNet distance computation
# ---------------------------------------------------------------------------

def get_best_synset(word: str, pos=wn.NOUN) -> Optional[object]:
    """Get the most common synset for a word."""
    synsets = wn.synsets(word, pos=pos)
    return synsets[0] if synsets else None


def compute_noun_distances(nouns: List[str]) -> Dict[Tuple[str, str], float]:
    """Pre-compute WordNet path_similarity for all noun pairs."""
    synsets = {}
    for n in nouns:
        s = get_best_synset(n, wn.NOUN)
        if s:
            synsets[n] = s

    distances = {}
    valid_nouns = list(synsets.keys())
    for i, n1 in enumerate(valid_nouns):
        for n2 in valid_nouns[i + 1:]:
            sim = synsets[n1].path_similarity(synsets[n2])
            if sim is not None:
                distances[(n1, n2)] = sim
                distances[(n2, n1)] = sim
    return distances, valid_nouns


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

class WordNetDataGenerator:
    """
    Generates training data from WordNet relationships.

    Supports train/test splits to prevent memorization.
    Templates are split: even-indexed for train, odd-indexed for test.
    """

    def __init__(self, split: str = "train", seed: int = 42):
        assert split in ("train", "test")
        self.split = split
        self.rng = random.Random(seed if split == "test" else None)

        # Split templates
        if split == "train":
            self.noun_templates = NOUN_TEMPLATES[::2]  # even indices
            self.adj_templates = ADJ_TEMPLATES[::2]
            self.verb_templates = VERB_TEMPLATES_PAST[::2]
        else:
            self.noun_templates = NOUN_TEMPLATES[1::2]  # odd indices
            self.adj_templates = ADJ_TEMPLATES[1::2]
            self.verb_templates = VERB_TEMPLATES_PAST[1::2]

        # Pre-compute noun distances
        self.noun_distances, self.valid_nouns = compute_noun_distances(NOUNS)

        # Bucket noun pairs by distance tier
        self.noun_pairs_by_tier = self._bucket_noun_pairs()

        # Validate adjective pairs (both words must have WordNet entries)
        self.valid_adj_antonyms = [
            (a, b) for a, b in ADJ_ANTONYMS
            if get_best_synset(a, wn.ADJ) and get_best_synset(b, wn.ADJ)
        ]

        # Validate verb chains
        self.valid_verb_chains = VERB_CHAINS  # pre-curated, all should work

    def _bucket_noun_pairs(self) -> Dict[str, List[Tuple[str, str, float]]]:
        """Bucket noun pairs into distance tiers."""
        tiers = {
            "very_close": [],   # path_sim > 0.33 (same category)
            "close": [],        # 0.14 < path_sim <= 0.33
            "medium": [],       # 0.08 < path_sim <= 0.14
            "far": [],          # path_sim <= 0.08
        }
        for (n1, n2), sim in self.noun_distances.items():
            if n1 >= n2:
                continue  # avoid duplicates
            pair = (n1, n2, sim)
            if sim > 0.33:
                tiers["very_close"].append(pair)
            elif sim > 0.14:
                tiers["close"].append(pair)
            elif sim > 0.08:
                tiers["medium"].append(pair)
            else:
                tiers["far"].append(pair)
        return tiers

    def noun_hierarchy_batch(self, batch_size: int = 16) -> Dict:
        """
        Generate pairs of sentences differing by one noun, with WN distances.

        Returns:
            dict with keys: sentences_a, sentences_b, distances (float 0-1),
                           tier_labels (str)
        """
        sentences_a, sentences_b, distances, tiers = [], [], [], []
        tier_names = list(self.noun_pairs_by_tier.keys())

        for _ in range(batch_size):
            # Sample a tier (weighted toward informative middle tiers)
            tier = self.rng.choices(
                tier_names, weights=[0.25, 0.30, 0.25, 0.20]
            )[0]
            pairs = self.noun_pairs_by_tier[tier]
            if not pairs:
                tier = "close"
                pairs = self.noun_pairs_by_tier[tier]

            n1, n2, sim = self.rng.choice(pairs)
            template = self.rng.choice(self.noun_templates)

            sentences_a.append(template.format(noun=n1))
            sentences_b.append(template.format(noun=n2))
            distances.append(sim)
            tiers.append(tier)

        return {
            "sentences_a": sentences_a,
            "sentences_b": sentences_b,
            "distances": distances,
            "tiers": tiers,
        }

    def adj_axis_batch(self, batch_size: int = 16, contexts_per_axis: int = 4) -> Dict:
        """
        Generate groups of sentence pairs along adjective axes.

        For each axis (e.g., big/small), generates multiple contexts so the model
        learns that the DIRECTION is consistent regardless of context.

        Returns:
            dict with keys: groups (list of dicts, each with sentences_a, sentences_b, axis_label)
        """
        n_groups = batch_size // contexts_per_axis
        groups = []

        for _ in range(max(1, n_groups)):
            adj_a, adj_b = self.rng.choice(self.valid_adj_antonyms)
            templates = self.rng.sample(
                self.adj_templates,
                min(contexts_per_axis, len(self.adj_templates))
            )

            group = {
                "sentences_a": [t.format(adj=adj_a) for t in templates],
                "sentences_b": [t.format(adj=adj_b) for t in templates],
                "axis_label": f"{adj_a}/{adj_b}",
            }
            groups.append(group)

        return {"groups": groups}

    def adj_intensity_batch(self, batch_size: int = 16) -> Dict:
        """
        Generate pairs along intensity gradients (e.g., warm < hot < scorching).

        Returns ordered pairs where the distance should reflect intensity difference.
        """
        sentences_a, sentences_b, intensity_gaps = [], [], []

        for _ in range(batch_size):
            chain = self.rng.choice(ADJ_INTENSITY)
            if len(chain) < 2:
                continue
            # Pick two positions in the chain
            i = self.rng.randint(0, len(chain) - 2)
            j = self.rng.randint(i + 1, len(chain) - 1)

            template = self.rng.choice(self.adj_templates)
            sentences_a.append(template.format(adj=chain[i]))
            sentences_b.append(template.format(adj=chain[j]))
            # Gap normalized by chain length
            intensity_gaps.append((j - i) / (len(chain) - 1))

        return {
            "sentences_a": sentences_a,
            "sentences_b": sentences_b,
            "intensity_gaps": intensity_gaps,
        }

    def verb_troponym_batch(self, batch_size: int = 16, contexts_per_chain: int = 3) -> Dict:
        """
        Generate sentence groups along verb troponym chains.

        For each chain (e.g., move→run→sprint), generates sentences in multiple
        contexts. The model should learn:
        - Consistent direction (general→specific)
        - Graded distance (adjacent verbs closer than distant ones)

        Returns:
            dict with keys: chains (list of dicts with sentences_per_level, chain_verbs)
        """
        n_chains = batch_size // contexts_per_chain
        chains = []

        for _ in range(max(1, n_chains)):
            chain = self.rng.choice(self.valid_verb_chains)
            templates = self.rng.sample(
                self.verb_templates,
                min(contexts_per_chain, len(self.verb_templates))
            )

            chain_data = {
                "chain_verbs": chain,
                "sentences_per_level": [],
            }
            for verb in chain:
                past = VERB_PAST.get(verb, verb + "ed")
                chain_data["sentences_per_level"].append(
                    [t.format(past=past) for t in templates]
                )
            chains.append(chain_data)

        return {"chains": chains}

    def mixed_batch(self, batch_size: int = 32) -> Dict:
        """Generate a mixed batch with all data types for probing."""
        noun_bs = batch_size // 4
        adj_bs = batch_size // 4
        intensity_bs = batch_size // 4
        verb_bs = batch_size // 4

        return {
            "noun_hierarchy": self.noun_hierarchy_batch(noun_bs),
            "adj_axis": self.adj_axis_batch(adj_bs),
            "adj_intensity": self.adj_intensity_batch(intensity_bs),
            "verb_troponym": self.verb_troponym_batch(verb_bs),
        }


# ---------------------------------------------------------------------------
# Probe: verify data quality before training
# ---------------------------------------------------------------------------

def probe_data():
    """Print samples and statistics to verify data quality."""
    print("=" * 70)
    print("WORDNET DATA GENERATOR PROBE")
    print("=" * 70)

    gen_train = WordNetDataGenerator(split="train", seed=42)
    gen_test = WordNetDataGenerator(split="test", seed=42)

    # --- Noun hierarchy ---
    print("\n" + "=" * 70)
    print("1. NOUN HIERARCHY PAIRS")
    print("=" * 70)

    print(f"\nValid nouns: {len(gen_train.valid_nouns)}")
    print(f"Total noun pairs with WN distance: {len(gen_train.noun_distances) // 2}")
    for tier, pairs in gen_train.noun_pairs_by_tier.items():
        print(f"  {tier}: {len(pairs)} pairs")

    batch = gen_train.noun_hierarchy_batch(12)
    print("\nSample noun pairs (TRAIN templates):")
    for a, b, d, t in zip(batch["sentences_a"], batch["sentences_b"],
                           batch["distances"], batch["tiers"]):
        print(f"  [{t:>10}] sim={d:.3f}")
        print(f"    A: {a}")
        print(f"    B: {b}")

    batch_test = gen_test.noun_hierarchy_batch(4)
    print("\nSample noun pairs (TEST templates):")
    for a, b, d, t in zip(batch_test["sentences_a"], batch_test["sentences_b"],
                           batch_test["distances"], batch_test["tiers"]):
        print(f"  [{t:>10}] sim={d:.3f}")
        print(f"    A: {a}")
        print(f"    B: {b}")

    # --- Adjective axes ---
    print("\n" + "=" * 70)
    print("2. ADJECTIVE AXIS PAIRS (antonyms, consistent direction)")
    print("=" * 70)

    print(f"\nValid antonym pairs: {len(gen_train.valid_adj_antonyms)}")

    batch = gen_train.adj_axis_batch(16, contexts_per_axis=4)
    for group in batch["groups"]:
        print(f"\n  Axis: {group['axis_label']}")
        for a, b in zip(group["sentences_a"], group["sentences_b"]):
            print(f"    A: {a}")
            print(f"    B: {b}")

    # --- Adjective intensity ---
    print("\n" + "=" * 70)
    print("3. ADJECTIVE INTENSITY GRADIENTS")
    print("=" * 70)

    print(f"\nIntensity chains: {len(ADJ_INTENSITY)}")
    batch = gen_train.adj_intensity_batch(8)
    for a, b, gap in zip(batch["sentences_a"], batch["sentences_b"],
                          batch["intensity_gaps"]):
        print(f"  gap={gap:.2f}  A: {a}")
        print(f"             B: {b}")

    # --- Verb troponyms ---
    print("\n" + "=" * 70)
    print("4. VERB TROPONYM CHAINS")
    print("=" * 70)

    print(f"\nVerb chains: {len(gen_train.valid_verb_chains)}")
    batch = gen_train.verb_troponym_batch(9, contexts_per_chain=3)
    for chain_data in batch["chains"][:3]:
        print(f"\n  Chain: {' → '.join(chain_data['chain_verbs'])}")
        for level_idx, (verb, sents) in enumerate(
                zip(chain_data["chain_verbs"],
                    chain_data["sentences_per_level"])):
            print(f"    Level {level_idx} ({verb}):")
            for s in sents:
                print(f"      {s}")

    # --- Template split verification ---
    print("\n" + "=" * 70)
    print("5. TEMPLATE SPLIT VERIFICATION (no overlap)")
    print("=" * 70)

    train_noun_t = set(gen_train.noun_templates)
    test_noun_t = set(gen_test.noun_templates)
    overlap = train_noun_t & test_noun_t
    print(f"  Noun templates: train={len(train_noun_t)}, test={len(test_noun_t)}, overlap={len(overlap)}")

    train_adj_t = set(gen_train.adj_templates)
    test_adj_t = set(gen_test.adj_templates)
    overlap = train_adj_t & test_adj_t
    print(f"  Adj templates:  train={len(train_adj_t)}, test={len(test_adj_t)}, overlap={len(overlap)}")

    train_verb_t = set(gen_train.verb_templates)
    test_verb_t = set(gen_test.verb_templates)
    overlap = train_verb_t & test_verb_t
    print(f"  Verb templates: train={len(train_verb_t)}, test={len(test_verb_t)}, overlap={len(overlap)}")

    # --- Statistics ---
    print("\n" + "=" * 70)
    print("6. TRAINING DATA CAPACITY")
    print("=" * 70)

    n_nouns = len(gen_train.valid_nouns)
    n_noun_pairs = len(gen_train.noun_distances) // 2
    n_noun_templates = len(gen_train.noun_templates)
    print(f"  Noun: {n_nouns} nouns × {n_noun_pairs} pairs × {n_noun_templates} templates")
    print(f"    = {n_noun_pairs * n_noun_templates:,} unique noun-swap sentence pairs")

    n_adj_pairs = len(gen_train.valid_adj_antonyms)
    n_adj_templates = len(gen_train.adj_templates)
    print(f"  Adj antonyms: {n_adj_pairs} pairs × {n_adj_templates} templates")
    print(f"    = {n_adj_pairs * n_adj_templates:,} unique adj-swap sentence pairs")

    n_intensity = sum(len(c) * (len(c) - 1) // 2 for c in ADJ_INTENSITY)
    print(f"  Adj intensity: {n_intensity} ordered pairs × {n_adj_templates} templates")
    print(f"    = {n_intensity * n_adj_templates:,} unique intensity sentence pairs")

    n_verb_chains = len(VERB_CHAINS)
    n_verb_templates = len(gen_train.verb_templates)
    n_verb_pairs = sum(len(c) * (len(c) - 1) // 2 for c in VERB_CHAINS)
    print(f"  Verb: {n_verb_chains} chains, {n_verb_pairs} pairs × {n_verb_templates} templates")
    print(f"    = {n_verb_pairs * n_verb_templates:,} unique verb-swap sentence pairs")

    print(f"\n  NLI data: 913,300 labeled pairs (entail/neutral/contradiction)")
    total_wn = (n_noun_pairs + n_adj_pairs + n_intensity + n_verb_pairs) * n_noun_templates
    print(f"  Total WordNet-generated pairs: ~{total_wn:,}")
    print(f"  Combined: ~{total_wn + 913300:,} structured training pairs")

    print("\n" + "=" * 70)
    print("PROBE COMPLETE")
    print("=" * 70)


def probe_with_model():
    """
    Probe using actual model to verify signals push in right direction.
    Loads V19 checkpoint and measures whether WordNet-derived pairs
    show the expected similarity patterns.
    """
    import torch
    import torch.nn.functional as F
    import sys
    sys.path.insert(0, "/home/david/chat_hamner_v2")
    from transformers import AutoTokenizer
    from concept_model import ConceptAutoencoderV19, ConceptConfig

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load V19 as baseline
    ckpt_path = "/home/david/chat_hamner_v2/checkpoints/concept_v19/latest.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except FileNotFoundError:
        print("No V19 checkpoint found, skipping model probe.")
        return

    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        config = ConceptConfig(**cfg)
    else:
        config = cfg
    model = ConceptAutoencoderV19(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    step = ckpt.get("step", "?")
    print(f"\nLoaded V19 checkpoint at step {step}")

    tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def encode(texts):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to(DEVICE)
        with torch.no_grad():
            v = model.concept_vector(enc["input_ids"], enc["attention_mask"])
        return F.normalize(v, dim=-1)

    gen = WordNetDataGenerator(split="test", seed=42)

    # --- Test 1: Noun hierarchy should show distance gradient ---
    print("\n" + "=" * 70)
    print("MODEL PROBE: NOUN HIERARCHY (closer nouns = higher cosine sim?)")
    print("=" * 70)

    tier_sims = defaultdict(list)
    for _ in range(10):
        batch = gen.noun_hierarchy_batch(16)
        vecs_a = encode(batch["sentences_a"])
        vecs_b = encode(batch["sentences_b"])
        sims = F.cosine_similarity(vecs_a, vecs_b, dim=-1)
        for sim, tier, wn_dist in zip(sims, batch["tiers"], batch["distances"]):
            tier_sims[tier].append((sim.item(), wn_dist))

    print(f"\n  {'Tier':<12} {'Count':>5} {'Avg CosSim':>10} {'Avg WN Dist':>11} {'Expected':>10}")
    print(f"  {'-'*12} {'-'*5} {'-'*10} {'-'*11} {'-'*10}")
    for tier in ["very_close", "close", "medium", "far"]:
        if tier_sims[tier]:
            avg_sim = sum(s for s, _ in tier_sims[tier]) / len(tier_sims[tier])
            avg_dist = sum(d for _, d in tier_sims[tier]) / len(tier_sims[tier])
            expected = "HIGH" if tier == "very_close" else "MED-HIGH" if tier == "close" else "MED-LOW" if tier == "medium" else "LOW"
            print(f"  {tier:<12} {len(tier_sims[tier]):>5} {avg_sim:>10.3f} {avg_dist:>11.3f} {expected:>10}")

    # Check if ordering is correct
    tier_avgs = {}
    for tier in ["very_close", "close", "medium", "far"]:
        if tier_sims[tier]:
            tier_avgs[tier] = sum(s for s, _ in tier_sims[tier]) / len(tier_sims[tier])
    ordered = all(tier_avgs.get(a, 0) >= tier_avgs.get(b, 0)
                  for a, b in [("very_close", "close"), ("close", "medium"), ("medium", "far")])
    print(f"\n  Ordering correct (very_close > close > medium > far): {'YES ✓' if ordered else 'NO ✗'}")

    # --- Test 2: Adjective axes should show consistent direction ---
    print("\n" + "=" * 70)
    print("MODEL PROBE: ADJECTIVE AXES (consistent direction across contexts?)")
    print("=" * 70)

    batch = gen.adj_axis_batch(32, contexts_per_axis=4)
    for group in batch["groups"][:6]:
        vecs_a = encode(group["sentences_a"])
        vecs_b = encode(group["sentences_b"])
        # Difference vectors (the "direction" for this axis)
        deltas = F.normalize(vecs_a - vecs_b, dim=-1)
        # Pairwise cosine of difference vectors (should all be ~1.0 if consistent)
        n = deltas.shape[0]
        if n > 1:
            cos_matrix = F.cosine_similarity(
                deltas.unsqueeze(0).expand(n, -1, -1),
                deltas.unsqueeze(1).expand(-1, n, -1),
                dim=-1
            )
            # Upper triangle (exclude diagonal)
            mask = torch.triu(torch.ones(n, n, device=DEVICE), diagonal=1).bool()
            avg_consistency = cos_matrix[mask].mean().item()
            pair_sim = F.cosine_similarity(vecs_a, vecs_b, dim=-1).mean().item()
            print(f"  {group['axis_label']:<20} dir_consistency={avg_consistency:.3f}  pair_sim={pair_sim:.3f}  (n={n})")

    # --- Test 3: Verb troponyms should show direction + ordering ---
    print("\n" + "=" * 70)
    print("MODEL PROBE: VERB TROPONYM CHAINS (consistent direction?)")
    print("=" * 70)

    batch = gen.verb_troponym_batch(12, contexts_per_chain=3)
    for chain_data in batch["chains"][:4]:
        verbs = chain_data["chain_verbs"]
        # Encode all levels
        level_vecs = []
        for sents in chain_data["sentences_per_level"]:
            vecs = encode(sents)
            level_vecs.append(vecs.mean(dim=0))  # average across contexts

        # Check pairwise distances along chain
        print(f"\n  Chain: {' → '.join(verbs)}")
        for i in range(len(level_vecs)):
            for j in range(i + 1, len(level_vecs)):
                sim = F.cosine_similarity(
                    level_vecs[i].unsqueeze(0),
                    level_vecs[j].unsqueeze(0)
                ).item()
                print(f"    {verbs[i]} ↔ {verbs[j]}: sim={sim:.3f}  (gap={j-i})")

    # --- Test 4: NLI should show 3-tier separation ---
    print("\n" + "=" * 70)
    print("MODEL PROBE: NLI 3-TIER (entailment > neutral > contradiction?)")
    print("=" * 70)

    import json
    nli_by_type = defaultdict(list)
    with open("/home/david/chat_hamner_v2/data/pairs/nli.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 3000:
                break
            d = json.loads(line)
            nli_by_type[d["type"]].append((d["text_a"], d["text_b"]))

    type_sims = {}
    for nli_type in ["entailment", "neutral", "contradiction"]:
        pairs = nli_by_type[nli_type][:200]
        sims = []
        for batch_start in range(0, len(pairs), 16):
            batch_pairs = pairs[batch_start:batch_start + 16]
            texts_a = [p[0] for p in batch_pairs]
            texts_b = [p[1] for p in batch_pairs]
            va = encode(texts_a)
            vb = encode(texts_b)
            batch_sims = F.cosine_similarity(va, vb, dim=-1)
            sims.extend(batch_sims.tolist())
        avg = sum(sims) / len(sims)
        type_sims[nli_type] = avg
        print(f"  {nli_type:<15} avg_cosine_sim={avg:.3f}  (n={len(sims)})")

    ordered = type_sims["entailment"] > type_sims["neutral"] > type_sims["contradiction"]
    print(f"\n  Ordering correct (entail > neutral > contradict): {'YES ✓' if ordered else 'NO ✗'}")
    spread = type_sims["entailment"] - type_sims["contradiction"]
    print(f"  Spread (entail - contradict): {spread:.3f}")

    print("\n" + "=" * 70)
    print("MODEL PROBE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    probe_data()
    if "--model" in sys.argv or "--full" in sys.argv:
        probe_with_model()
