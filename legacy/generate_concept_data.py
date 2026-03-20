#!/usr/bin/env python3
"""
Synthetic dataset generator for training a concept autoencoder with 32 concept slots.

Generates sentence pairs where ONLY one concept axis varies, so the model learns
which slot should change for which kind of variation.

Output: JSONL files in data/concept_axes/ with fields:
  - slot: int (0-31)
  - base: str (base sentence)
  - variant: str (sentence with one concept changed)
  - concept_value: str (what the concept was changed to)
"""

import json
import os
import random
import itertools
from typing import List, Dict, Tuple, Any

# ============================================================================
# RICH WORD LISTS
# ============================================================================

PEOPLE = [
    "man", "woman", "child", "doctor", "teacher", "king", "queen", "soldier",
    "farmer", "artist", "musician", "scientist", "chef", "nurse", "pilot",
    "judge", "priest", "monk", "thief", "knight", "merchant", "sailor",
    "blacksmith", "shepherd", "librarian", "detective", "professor", "student",
    "baker", "carpenter", "architect", "engineer", "poet", "dancer", "singer",
    "painter", "photographer", "surgeon", "lawyer", "journalist", "author",
    "gardener", "fisherman", "miner", "tailor", "weaver", "potter", "butcher",
    "barber", "clerk", "mayor", "governor", "general", "captain", "princess",
    "prince", "emperor", "beggar", "stranger", "elder", "girl", "boy",
]

ANIMALS = [
    "cat", "dog", "horse", "eagle", "whale", "ant", "bear", "wolf", "fox",
    "deer", "rabbit", "snake", "lion", "tiger", "elephant", "owl", "hawk",
    "raven", "crow", "sparrow", "dolphin", "shark", "octopus", "spider",
    "beetle", "butterfly", "moth", "frog", "toad", "turtle", "lizard",
    "crocodile", "parrot", "penguin", "flamingo", "swan", "goose", "duck",
    "rooster", "hen", "pig", "cow", "sheep", "goat", "donkey", "camel",
    "monkey", "gorilla", "panther", "leopard", "cheetah", "gazelle",
]

OBJECTS = [
    "car", "table", "book", "phone", "sword", "bridge", "clock", "lamp",
    "chair", "door", "window", "mirror", "bottle", "cup", "plate", "bowl",
    "knife", "hammer", "bell", "drum", "flag", "crown", "ring", "coin",
    "key", "lock", "chain", "rope", "basket", "box", "chest", "barrel",
    "wagon", "ship", "boat", "candle", "torch", "lantern", "shield",
    "statue", "painting", "scroll", "letter", "map", "compass", "telescope",
    "wheel", "gate", "fence", "pillar", "fountain", "vase", "carpet",
]

PLACES = [
    "house", "castle", "forest", "ocean", "city", "mountain", "village",
    "garden", "temple", "church", "tower", "cave", "island", "valley",
    "river", "lake", "desert", "meadow", "swamp", "cliff", "harbor",
    "market", "palace", "fortress", "cottage", "barn", "mill", "mine",
    "prison", "library", "museum", "theater", "stadium", "park", "bridge",
    "tunnel", "alley", "plaza", "courtyard", "balcony", "rooftop",
]

ALL_NOUNS = PEOPLE + ANIMALS + OBJECTS + PLACES

CONCRETE_NOUNS = PEOPLE + ANIMALS + OBJECTS  # nouns that can "do things"
ENTITY_NOUNS = PEOPLE + ANIMALS  # animate nouns

VERBS_ACTION = [
    "walked", "ran", "jumped", "climbed", "swam", "flew", "crawled",
    "danced", "sang", "whispered", "shouted", "laughed", "cried",
    "painted", "built", "destroyed", "carried", "lifted", "pushed",
    "pulled", "threw", "caught", "dropped", "held", "grabbed", "touched",
    "watched", "listened", "searched", "found", "lost", "opened", "closed",
    "broke", "fixed", "created", "wrote", "read", "drew", "carved",
    "cooked", "ate", "drank", "slept", "woke", "stood", "sat", "fell",
    "fought", "chased", "escaped", "hid", "appeared", "vanished",
]

VERBS_PRESENT = [
    "walks", "runs", "jumps", "climbs", "swims", "flies", "crawls",
    "dances", "sings", "whispers", "shouts", "laughs", "cries",
    "paints", "builds", "destroys", "carries", "lifts", "pushes",
    "pulls", "throws", "catches", "drops", "holds", "grabs", "touches",
    "watches", "listens", "searches", "finds", "loses", "opens", "closes",
    "breaks", "fixes", "creates", "writes", "reads", "draws", "carves",
    "cooks", "eats", "drinks", "sleeps", "wakes", "stands", "sits", "falls",
    "fights", "chases", "escapes", "hides", "appears", "vanishes",
]

VERBS_BASE = [
    "walk", "run", "jump", "climb", "swim", "fly", "crawl",
    "dance", "sing", "whisper", "shout", "laugh", "cry",
    "paint", "build", "destroy", "carry", "lift", "push",
    "pull", "throw", "catch", "drop", "hold", "grab", "touch",
    "watch", "listen", "search", "find", "lose", "open", "close",
    "break", "fix", "create", "write", "read", "draw", "carve",
    "cook", "eat", "drink", "sleep", "wake", "stand", "sit", "fall",
    "fight", "chase", "escape", "hide", "appear", "vanish",
]

ADJECTIVES_GENERAL = [
    "old", "new", "strange", "beautiful", "broken", "ancient", "familiar",
    "forgotten", "mysterious", "ordinary", "remarkable", "humble", "proud",
    "silent", "noisy", "lonely", "crowded", "empty", "famous", "unknown",
]

ADVERBS_GENERAL = [
    "silently", "suddenly", "slowly", "carefully", "eagerly", "nervously",
    "happily", "sadly", "angrily", "calmly", "bravely", "quietly",
]

NAMES_MALE = ["John", "James", "Robert", "Thomas", "William", "Henry", "George",
              "Edward", "Charles", "Arthur", "Daniel", "Michael", "David", "Peter",
              "Marcus", "Leo", "Felix", "Oscar", "Hugo", "Ivan"]

NAMES_FEMALE = ["Mary", "Elizabeth", "Sarah", "Anna", "Margaret", "Catherine",
                "Alice", "Clara", "Emma", "Grace", "Helen", "Julia", "Laura",
                "Martha", "Rose", "Sofia", "Lily", "Diana", "Nora", "Iris"]

NAMES_ALL = NAMES_MALE + NAMES_FEMALE

# ============================================================================
# CONCEPT AXIS DEFINITIONS
# ============================================================================

SLOT_NAMES = {
    0: "subject_entity_type",
    1: "object_patient_identity",
    2: "animacy_gender",
    3: "age_life_stage",
    4: "size_scale",
    5: "color_brightness",
    6: "shape_form",
    7: "material_texture",
    8: "weight_density",
    9: "temperature_weather",
    10: "action_type",
    11: "action_manner_intensity",
    12: "speed_completion",
    13: "direction_path",
    14: "location_scene",
    15: "spatial_relations",
    16: "distance_proximity",
    17: "tense_aspect",
    18: "duration_frequency",
    19: "time_reference",
    20: "number_amount",
    21: "degree_comparison",
    22: "core_sentiment",
    23: "specific_emotion",
    24: "arousal_energy",
    25: "quality_value",
    26: "difficulty_importance",
    27: "negation_truth",
    28: "certainty_obligation",
    29: "causation_condition",
    30: "formality_register",
    31: "speech_act_intent",
}


# ============================================================================
# SLOT 0: Subject/Entity Type
# ============================================================================

SLOT0_ENTITY_TYPES = {
    "person": PEOPLE[:30],
    "animal": ANIMALS[:30],
    "object": OBJECTS[:30],
    "place": PLACES[:25],
}

SLOT0_TEMPLATES = [
    ("the {entity} stood in the doorway", None),
    ("a {entity} appeared at the edge of the road", None),
    ("the {entity} was the first thing she noticed", None),
    ("nobody expected to find a {entity} there", None),
    ("he stared at the {entity} for a long time", None),
    ("the {entity} had been there for years", None),
    ("she picked up the {entity} and examined it closely", None),
    ("there was a {entity} sitting in the middle of the room", None),
    ("the old {entity} reminded him of something from his childhood", None),
    ("a single {entity} remained after everything else was gone", None),
    ("they discovered a {entity} hidden behind the wall", None),
    ("the {entity} caught everyone's attention immediately", None),
    ("right in front of them was a {entity}", None),
    ("the {entity} seemed out of place in this setting", None),
    ("she had always wanted to see a {entity} like that", None),
    ("the {entity} was unlike anything they had encountered before", None),
    ("everyone gathered around the {entity} in amazement", None),
    ("the {entity} moved slightly in the wind", None),
    ("a beautiful {entity} sat at the center of the display", None),
    ("the {entity} was the only thing left standing", None),
]


def generate_slot0() -> List[Dict]:
    """Subject/Entity Type pairs."""
    pairs = []
    entity_types = list(SLOT0_ENTITY_TYPES.keys())

    for template, _ in SLOT0_TEMPLATES:
        for base_type in entity_types:
            base_entities = SLOT0_ENTITY_TYPES[base_type]
            for variant_type in entity_types:
                if variant_type == base_type:
                    continue
                variant_entities = SLOT0_ENTITY_TYPES[variant_type]
                for be in base_entities[:15]:
                    for ve in random.sample(variant_entities, min(8, len(variant_entities))):
                        pairs.append({
                            "slot": 0,
                            "base": template.format(entity=be),
                            "variant": template.format(entity=ve),
                            "concept_value": variant_type,
                        })
    return pairs


# ============================================================================
# SLOT 1: Object/Patient Identity
# ============================================================================

SLOT1_OBJECTS = OBJECTS + ANIMALS[:20] + PEOPLE[:20]

SLOT1_TEMPLATES = [
    "the {subj} picked up the {obj}",
    "she handed the {obj} to the {subj}",
    "the {subj} carefully examined the {obj}",
    "he placed the {obj} on the table",
    "the {subj} was carrying a {obj}",
    "they found the {obj} near the river",
    "the {subj} dropped the {obj} by accident",
    "she bought a {obj} from the old shop",
    "the {subj} needed a {obj} to finish the job",
    "he noticed the {obj} was missing",
    "the {subj} traded the {obj} for something else",
    "she wrapped the {obj} in a cloth",
    "they buried the {obj} under the tree",
    "the {subj} stole the {obj} in the night",
    "someone left a {obj} at the front gate",
    "the {subj} couldn't lift the {obj} alone",
    "she broke the {obj} without meaning to",
    "the {subj} tossed the {obj} across the room",
    "he polished the {obj} until it gleamed",
    "the {subj} hid the {obj} inside the drawer",
]


def generate_slot1() -> List[Dict]:
    """Object/Patient Identity pairs."""
    pairs = []
    subjects = random.sample(PEOPLE, 20)
    objects_list = SLOT1_OBJECTS

    for template in SLOT1_TEMPLATES:
        for subj in subjects:
            base_objs = random.sample(objects_list, min(20, len(objects_list)))
            for i, base_obj in enumerate(base_objs):
                variant_candidates = [o for o in objects_list if o != base_obj]
                for var_obj in random.sample(variant_candidates, min(8, len(variant_candidates))):
                    pairs.append({
                        "slot": 1,
                        "base": template.format(subj=subj, obj=base_obj),
                        "variant": template.format(subj=subj, obj=var_obj),
                        "concept_value": var_obj,
                    })
    return pairs


# ============================================================================
# SLOT 2: Animacy & Gender
# ============================================================================

SLOT2_GENDERED = {
    "male": {
        "pronoun_subj": "he", "pronoun_obj": "him", "possessive": "his",
        "nouns": ["man", "boy", "king", "prince", "father", "brother", "son",
                  "husband", "uncle", "grandfather", "lord", "gentleman", "monk"],
    },
    "female": {
        "pronoun_subj": "she", "pronoun_obj": "her", "possessive": "her",
        "nouns": ["woman", "girl", "queen", "princess", "mother", "sister",
                  "daughter", "wife", "aunt", "grandmother", "lady", "maiden", "nun"],
    },
    "neutral_living": {
        "pronoun_subj": "it", "pronoun_obj": "it", "possessive": "its",
        "nouns": ANIMALS[:25],
    },
    "neutral_nonliving": {
        "pronoun_subj": "it", "pronoun_obj": "it", "possessive": "its",
        "nouns": OBJECTS[:25],
    },
}

SLOT2_TEMPLATES_PRONOUN = [
    "{subj} walked through the garden and {subj} smiled",
    "{subj} opened the door and stepped inside",
    "everyone watched as {subj} crossed the bridge",
    "{subj} picked up the letter and read it carefully",
    "the crowd cheered when {subj} arrived",
    "{subj} sat by the fire and thought about the future",
    "{subj} turned around and noticed the shadow",
    "nobody expected {subj} to return so soon",
    "{subj} spoke softly to the crowd",
    "{subj} lifted the heavy stone with both hands",
    "they asked {obj} to stay a little longer",
    "the message was addressed to {obj}",
    "she handed the letter to {obj}",
    "everyone trusted {obj} completely",
    "{poss} voice echoed through the hall",
    "{poss} eyes were filled with determination",
    "the decision was entirely in {poss} hands",
]

SLOT2_TEMPLATES_NOUN = [
    "the {noun} stood at the edge of the cliff",
    "a {noun} appeared in the distance",
    "the old {noun} had seen better days",
    "they found the {noun} resting by the river",
    "the {noun} was the last one to leave",
    "a young {noun} approached the gate",
    "the {noun} carried a heavy burden",
    "nobody noticed the {noun} in the corner",
    "the {noun} had traveled far to reach this place",
    "a {noun} waited patiently by the road",
    "the {noun} watched the sunset in silence",
    "the {noun} emerged from the shadows",
    "a brave {noun} stepped forward to speak",
    "the tired {noun} collapsed onto the bed",
    "the {noun} refused to give up despite everything",
]


def generate_slot2() -> List[Dict]:
    """Animacy & Gender pairs."""
    pairs = []
    categories = list(SLOT2_GENDERED.keys())

    # Pronoun-based templates
    for template in SLOT2_TEMPLATES_PRONOUN:
        for base_cat in categories:
            base = SLOT2_GENDERED[base_cat]
            for var_cat in categories:
                if var_cat == base_cat:
                    continue
                var = SLOT2_GENDERED[var_cat]
                base_sent = template.format(
                    subj=base["pronoun_subj"], obj=base["pronoun_obj"], poss=base["possessive"]
                )
                var_sent = template.format(
                    subj=var["pronoun_subj"], obj=var["pronoun_obj"], poss=var["possessive"]
                )
                if base_sent != var_sent:
                    pairs.append({
                        "slot": 2,
                        "base": base_sent,
                        "variant": var_sent,
                        "concept_value": var_cat,
                    })

    # Noun-based templates
    for template in SLOT2_TEMPLATES_NOUN:
        for base_cat in categories:
            base_nouns = SLOT2_GENDERED[base_cat]["nouns"]
            for var_cat in categories:
                if var_cat == base_cat:
                    continue
                var_nouns = SLOT2_GENDERED[var_cat]["nouns"]
                for bn in base_nouns[:10]:
                    for vn in random.sample(var_nouns, min(6, len(var_nouns))):
                        pairs.append({
                            "slot": 2,
                            "base": template.format(noun=bn),
                            "variant": template.format(noun=vn),
                            "concept_value": var_cat,
                        })
    return pairs


# ============================================================================
# SLOT 3: Age & Life Stage
# ============================================================================

SLOT3_STAGES = {
    "baby": ["baby", "infant", "newborn", "toddler"],
    "child": ["child", "kid", "youngster", "little one", "schoolchild"],
    "teenager": ["teenager", "adolescent", "teen", "youth"],
    "young_adult": ["young man", "young woman", "young adult"],
    "adult": ["man", "woman", "adult", "grown-up"],
    "middle_aged": ["middle-aged man", "middle-aged woman"],
    "elderly": ["old man", "old woman", "elder", "elderly person", "grandmother", "grandfather"],
}

SLOT3_ADJ = {
    "baby": "newborn",
    "child": "young",
    "teenager": "teenage",
    "young_adult": "youthful",
    "adult": "grown",
    "middle_aged": "middle-aged",
    "elderly": "elderly",
}

SLOT3_TEMPLATES = [
    "the {noun} sat alone on the bench",
    "a {noun} walked slowly down the path",
    "the {noun} looked out the window with curiosity",
    "they found a {noun} wandering in the market",
    "the {noun} had a look of wonder on their face",
    "a {noun} stood waiting by the fountain",
    "nobody noticed the {noun} slip away from the group",
    "the {noun} paused to rest under the oak tree",
    "a {noun} approached the stranger with caution",
    "the {noun} carried a small bundle in their arms",
    "the {noun} was the first to arrive that morning",
    "a {noun} appeared at the window just before dawn",
    "the {noun} spoke with a voice full of experience",
    "a {noun} watched from across the street",
    "the {noun} remembered a time when things were different",
    "everyone turned to look at the {noun} who had just entered",
    "the {noun} stumbled but quickly regained balance",
    "a {noun} was spotted near the old well",
    "the {noun} smiled warmly and waved",
    "a quiet {noun} sat reading in the corner",
]

SLOT3_TEMPLATES_ADJ = [
    "the {adj} traveler rested by the river",
    "a {adj} figure emerged from the fog",
    "the {adj} stranger asked for directions",
    "a {adj} voice called out from the darkness",
    "the {adj} face showed signs of weariness",
    "a {adj} hand reached for the doorknob",
    "the {adj} visitor was greeted with suspicion",
    "a {adj} person stood at the crossroads",
    "the {adj} shepherd guided the flock home",
    "a {adj} apprentice began learning the trade",
]


def generate_slot3() -> List[Dict]:
    """Age & Life Stage pairs."""
    pairs = []
    stages = list(SLOT3_STAGES.keys())

    for template in SLOT3_TEMPLATES:
        for base_stage in stages:
            for var_stage in stages:
                if var_stage == base_stage:
                    continue
                for bn in SLOT3_STAGES[base_stage]:
                    for vn in SLOT3_STAGES[var_stage]:
                        pairs.append({
                            "slot": 3,
                            "base": template.format(noun=bn),
                            "variant": template.format(noun=vn),
                            "concept_value": var_stage,
                        })

    for template in SLOT3_TEMPLATES_ADJ:
        for base_stage in stages:
            for var_stage in stages:
                if var_stage == base_stage:
                    continue
                pairs.append({
                    "slot": 3,
                    "base": template.format(adj=SLOT3_ADJ[base_stage]),
                    "variant": template.format(adj=SLOT3_ADJ[var_stage]),
                    "concept_value": var_stage,
                })
    return pairs


# ============================================================================
# SLOT 4: Size & Scale
# ============================================================================

SLOT4_SIZES = [
    "microscopic", "tiny", "small", "little", "medium-sized", "large",
    "big", "huge", "enormous", "massive", "colossal", "gigantic",
    "towering", "vast", "immense", "miniature", "petite", "oversized",
]

SLOT4_TEMPLATES = [
    "the {size} {noun} sat in the middle of the room",
    "she noticed a {size} {noun} in the corner",
    "the {size} {noun} towered over everything nearby",
    "he had never seen such a {size} {noun} before",
    "a {size} {noun} blocked the entire path",
    "the {size} {noun} fit perfectly in the palm of a hand",
    "they carried the {size} {noun} through the gate",
    "a {size} {noun} appeared on the horizon",
    "the {size} {noun} was impossible to ignore",
    "she picked up the surprisingly {size} {noun}",
    "the {size} {noun} cast a long shadow across the ground",
    "a {size} {noun} was discovered in the attic",
    "the {size} {noun} dominated the landscape",
    "he struggled to move the {size} {noun} even an inch",
    "the {size} {noun} had been growing steadily for years",
    "it was a {size} {noun} unlike any other",
    "the {size} {noun} barely fit through the doorway",
    "a remarkably {size} {noun} caught everyone's eye",
    "the {size} {noun} made the others look ordinary",
    "someone had left a {size} {noun} on the table",
    "the {size} {noun} was the prize of the collection",
    "even from far away the {size} {noun} was visible",
    "the children stared at the {size} {noun} in awe",
    "a {size} {noun} rolled down the hillside",
    "the {size} {noun} swayed gently in the breeze",
]

SLOT4_NOUNS = (
    ANIMALS[:25] + OBJECTS[:25] +
    ["building", "tree", "rock", "crystal", "flower", "wave", "shadow",
     "creature", "stone", "boulder", "statue", "tower", "hill", "mushroom"]
)


def generate_slot4() -> List[Dict]:
    """Size & Scale pairs."""
    pairs = []
    for template in SLOT4_TEMPLATES:
        for noun in SLOT4_NOUNS:
            for base_size in SLOT4_SIZES:
                for var_size in SLOT4_SIZES:
                    if var_size == base_size:
                        continue
                    pairs.append({
                        "slot": 4,
                        "base": template.format(size=base_size, noun=noun),
                        "variant": template.format(size=var_size, noun=noun),
                        "concept_value": var_size,
                    })
    # This produces a LOT — subsample
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 5: Color & Brightness
# ============================================================================

SLOT5_COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "golden", "silver", "crimson", "scarlet",
    "azure", "emerald", "ivory", "dark", "bright", "pale", "vivid",
    "deep", "faded", "glowing", "shimmering",
]

SLOT5_TEMPLATES = [
    "the {color} {noun} gleamed in the light",
    "she wore a {color} dress to the ceremony",
    "the {color} {noun} stood out against the background",
    "a {color} {noun} lay on the table",
    "he painted the wall a {color} shade",
    "the {color} {noun} reminded her of home",
    "a {color} light flickered in the distance",
    "the {color} {noun} was the most striking feature",
    "she noticed the {color} {noun} right away",
    "the {color} flowers bloomed across the entire field",
    "a {color} bird perched on the branch above",
    "the sky turned a deep {color} at sunset",
    "the {color} {noun} was beautiful beyond words",
    "he chose the {color} one from the pile",
    "a streak of {color} paint marked the wall",
    "the {color} fabric felt smooth under her fingers",
    "the {color} {noun} reflected the morning sun",
    "a {color} shadow crept across the floor",
    "the {color} {noun} was impossible to miss",
    "she wrapped herself in the {color} blanket",
    "the {color} {noun} sparkled under the chandelier",
    "a {color} mist hung over the valley",
    "the room was bathed in {color} light",
    "he pulled out a {color} stone from his pocket",
    "the {color} {noun} was the finest in the shop",
]

SLOT5_NOUNS = [
    "stone", "gem", "crystal", "ribbon", "flag", "flower", "feather",
    "cloak", "banner", "scarf", "thread", "leaf", "petal", "shell",
    "tile", "glass", "curtain", "veil", "cape", "candle", "lantern",
    "jewel", "ornament", "bead", "pendant", "mask", "hood", "robe",
    "tapestry", "cloth", "ink", "flame", "orb", "sphere", "ring",
]


def generate_slot5() -> List[Dict]:
    """Color & Brightness pairs."""
    pairs = []
    for template in SLOT5_TEMPLATES:
        for noun in SLOT5_NOUNS:
            for base_color in SLOT5_COLORS:
                for var_color in SLOT5_COLORS:
                    if var_color == base_color:
                        continue
                    pairs.append({
                        "slot": 5,
                        "base": template.format(color=base_color, noun=noun),
                        "variant": template.format(color=var_color, noun=noun),
                        "concept_value": var_color,
                    })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 6: Shape & Form
# ============================================================================

SLOT6_SHAPES = [
    "round", "square", "long", "flat", "tall", "thin", "curved", "angular",
    "oval", "cylindrical", "triangular", "spiral", "jagged", "smooth",
    "pointed", "bulky", "slender", "twisted", "hollow", "blocky",
]

SLOT6_TEMPLATES = [
    "the {shape} {noun} rested on the shelf",
    "she traced the {shape} edge with her finger",
    "the {shape} {noun} caught his attention",
    "a {shape} {noun} lay half-buried in the sand",
    "the {shape} {noun} was clearly handmade",
    "he picked up the {shape} {noun} and turned it over",
    "the {shape} {noun} fit perfectly in the slot",
    "a {shape} shadow stretched across the wall",
    "the {shape} {noun} reminded her of a childhood toy",
    "they carved a {shape} {noun} from the wood",
    "the table held a {shape} {noun} made of clay",
    "she preferred the {shape} one to the others",
    "the {shape} {noun} rolled off the edge",
    "a {shape} hole had been cut in the wall",
    "the {shape} design was etched into the metal",
    "he admired the {shape} pattern on the ceiling",
    "a {shape} pillar supported the archway",
    "the {shape} {noun} was surprisingly aerodynamic",
    "the artist sculpted a {shape} figure from marble",
    "a {shape} window let in the morning light",
]

SLOT6_NOUNS = [
    "stone", "box", "table", "mirror", "crystal", "frame", "pillar",
    "tower", "bottle", "vase", "shield", "pendant", "bowl", "plate",
    "brick", "tile", "arch", "beam", "column", "rod", "disk", "block",
    "prism", "dome", "ring", "coin", "medallion", "sculpture", "mask",
]


def generate_slot6() -> List[Dict]:
    """Shape & Form pairs."""
    pairs = []
    for template in SLOT6_TEMPLATES:
        for noun in SLOT6_NOUNS:
            for base_shape in SLOT6_SHAPES:
                for var_shape in SLOT6_SHAPES:
                    if var_shape == base_shape:
                        continue
                    pairs.append({
                        "slot": 6,
                        "base": template.format(shape=base_shape, noun=noun),
                        "variant": template.format(shape=var_shape, noun=noun),
                        "concept_value": var_shape,
                    })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 7: Material & Texture
# ============================================================================

SLOT7_MATERIALS = [
    "wooden", "metal", "glass", "stone", "leather", "silk", "cotton",
    "woolen", "iron", "bronze", "copper", "golden", "silver", "clay",
    "ceramic", "marble", "granite", "bamboo", "paper", "plastic",
    "rubber", "velvet", "linen", "steel", "crystal",
]

SLOT7_TEXTURES = [
    "soft", "rough", "smooth", "silky", "coarse", "fuzzy", "gritty",
    "polished", "matte", "glossy", "bumpy", "grainy", "slippery",
    "textured", "woven",
]

SLOT7_ALL = SLOT7_MATERIALS + SLOT7_TEXTURES

SLOT7_TEMPLATES = [
    "the {mat} {noun} felt heavy in her hands",
    "he ran his fingers over the {mat} surface",
    "a {mat} {noun} sat on the mantelpiece",
    "the {mat} {noun} was warm to the touch",
    "she admired the {mat} finish on the {noun}",
    "the {mat} {noun} gleamed under the lamplight",
    "he preferred the {mat} one over the rest",
    "a {mat} {noun} hung from the ceiling",
    "the {mat} {noun} was a gift from abroad",
    "she could tell it was {mat} just by touching it",
    "the {mat} {noun} was centuries old",
    "a {mat} {noun} blocked the passage",
    "the {mat} {noun} reflected no light at all",
    "he carved the {mat} {noun} with great care",
    "the {mat} {noun} crumbled at the edges",
    "a beautifully crafted {mat} {noun} graced the hall",
    "the {mat} {noun} was stronger than it looked",
    "she wrapped herself in the {mat} blanket",
    "the {mat} door creaked as it opened",
    "a {mat} figurine stood on the windowsill",
]

SLOT7_NOUNS = [
    "box", "table", "chair", "door", "cup", "bowl", "plate", "ring",
    "sword", "shield", "statue", "figure", "frame", "panel", "tile",
    "column", "bench", "chest", "gate", "fence", "mask", "crown",
    "throne", "urn", "chalice", "pendant", "bracelet", "staff", "rod",
]


def generate_slot7() -> List[Dict]:
    """Material & Texture pairs."""
    pairs = []
    for template in SLOT7_TEMPLATES:
        for noun in SLOT7_NOUNS:
            for base_mat in SLOT7_ALL:
                for var_mat in SLOT7_ALL:
                    if var_mat == base_mat:
                        continue
                    pairs.append({
                        "slot": 7,
                        "base": template.format(mat=base_mat, noun=noun),
                        "variant": template.format(mat=var_mat, noun=noun),
                        "concept_value": var_mat,
                    })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 8: Weight & Density
# ============================================================================

SLOT8_WEIGHTS = [
    "light", "lightweight", "feather-light", "weightless", "airy",
    "heavy", "hefty", "weighty", "ponderous", "leaden",
    "dense", "solid", "compact", "bulky", "massive",
]

SLOT8_TEMPLATES = [
    "the {weight} {noun} was difficult to carry",
    "she lifted the surprisingly {weight} {noun}",
    "the {weight} package arrived this morning",
    "he struggled with the {weight} {noun} on his back",
    "the {weight} {noun} sank immediately",
    "a {weight} {noun} sat on the counter",
    "the {weight} {noun} could barely be moved",
    "she tossed the {weight} {noun} to her friend",
    "the {weight} {noun} floated on the water's surface",
    "he was surprised by how {weight} the {noun} felt",
    "the {weight} stone rolled downhill effortlessly",
    "a {weight} fog hung over the valley",
    "the {weight} armor slowed the knight considerably",
    "she carried the {weight} basket on her head",
    "the {weight} chains rattled as he moved",
    "a {weight} book lay open on the desk",
    "the {weight} load strained the horse's back",
    "the {weight} fabric draped gracefully over the chair",
    "he dropped the {weight} hammer with a thud",
    "the {weight} cloud drifted lazily overhead",
]

SLOT8_NOUNS = [
    "box", "stone", "bag", "chest", "bundle", "crate", "sack", "barrel",
    "trunk", "block", "sphere", "plate", "shield", "coin", "ingot",
    "rod", "chain", "anchor", "beam", "pillar", "tablet", "brick",
]


def generate_slot8() -> List[Dict]:
    """Weight & Density pairs."""
    pairs = []
    for template in SLOT8_TEMPLATES:
        for noun in SLOT8_NOUNS:
            for base_w in SLOT8_WEIGHTS:
                for var_w in SLOT8_WEIGHTS:
                    if var_w == base_w:
                        continue
                    pairs.append({
                        "slot": 8,
                        "base": template.format(weight=base_w, noun=noun),
                        "variant": template.format(weight=var_w, noun=noun),
                        "concept_value": var_w,
                    })
    random.shuffle(pairs)
    return pairs[:60000]


# ============================================================================
# SLOT 9: Temperature & Weather
# ============================================================================

SLOT9_TEMPS = [
    "freezing", "icy", "cold", "chilly", "cool", "mild", "warm",
    "hot", "scorching", "burning", "boiling", "sweltering", "tepid",
]

SLOT9_WEATHER = [
    "rainy", "sunny", "cloudy", "foggy", "stormy", "windy", "snowy",
    "misty", "humid", "dry", "overcast", "clear", "hazy", "drizzly",
]

SLOT9_TEMPLATES_TEMP = [
    "the {temp} air hit them as they stepped outside",
    "she wrapped her hands around the {temp} cup",
    "the {temp} water ran over his fingers",
    "a {temp} breeze swept through the valley",
    "the {temp} wind made them shiver",
    "he touched the {temp} metal surface cautiously",
    "the {temp} sand burned under their feet",
    "she could feel the {temp} stone through her shoes",
    "the {temp} night settled over the village",
    "a {temp} mist rose from the ground",
    "the {temp} rain soaked them to the bone",
    "he plunged his hand into the {temp} stream",
    "the {temp} floor chilled her bare feet",
    "a {temp} draft crept in through the cracks",
    "the {temp} sun beat down relentlessly",
    "she served a {temp} bowl of soup",
    "the {temp} morning greeted them with frost",
    "a {temp} gust rattled the windows",
    "the {temp} fire crackled in the hearth",
    "he felt the {temp} sweat on his brow",
]

SLOT9_TEMPLATES_WEATHER = [
    "it was a {weather} day when they set out",
    "the {weather} afternoon kept everyone indoors",
    "they walked through the {weather} streets",
    "the {weather} sky stretched endlessly above",
    "a {weather} morning greeted them at dawn",
    "the {weather} evening was perfect for a walk",
    "she stared out at the {weather} landscape",
    "the {weather} conditions made travel difficult",
    "they arrived on a {weather} night",
    "the {weather} weather lasted for three days",
    "a {weather} spell settled over the town",
    "he remembered the {weather} summers of his youth",
    "the {weather} season had just begun",
    "she loved the {weather} days best of all",
    "the {weather} climate suited them perfectly",
]


def generate_slot9() -> List[Dict]:
    """Temperature & Weather pairs."""
    pairs = []
    for template in SLOT9_TEMPLATES_TEMP:
        for base_t in SLOT9_TEMPS:
            for var_t in SLOT9_TEMPS:
                if var_t == base_t:
                    continue
                pairs.append({
                    "slot": 9,
                    "base": template.format(temp=base_t),
                    "variant": template.format(temp=var_t),
                    "concept_value": var_t,
                })
    for template in SLOT9_TEMPLATES_WEATHER:
        for base_w in SLOT9_WEATHER:
            for var_w in SLOT9_WEATHER:
                if var_w == base_w:
                    continue
                pairs.append({
                    "slot": 9,
                    "base": template.format(weather=base_w),
                    "variant": template.format(weather=var_w),
                    "concept_value": var_w,
                })
    random.shuffle(pairs)
    return pairs[:60000]


# ============================================================================
# SLOT 10: Action Type
# ============================================================================

SLOT10_ACTIONS = {
    "motion": [
        "{subj} walked across the field", "{subj} ran through the forest",
        "{subj} climbed the steep hill", "{subj} swam across the river",
        "{subj} jumped over the fence", "{subj} crawled under the table",
        "{subj} danced around the fire", "{subj} stumbled down the stairs",
        "{subj} marched along the road", "{subj} wandered through the streets",
    ],
    "speech": [
        "{subj} whispered a secret to the child", "{subj} shouted from the rooftop",
        "{subj} spoke calmly to the crowd", "{subj} muttered something under their breath",
        "{subj} announced the news to everyone", "{subj} sang a lullaby by the fire",
        "{subj} recited the poem from memory", "{subj} called out across the valley",
        "{subj} argued with the merchant", "{subj} pleaded for mercy",
    ],
    "creation": [
        "{subj} painted a portrait of the king", "{subj} built a shelter from branches",
        "{subj} wrote a letter to the governor", "{subj} carved a figure from oak",
        "{subj} composed a melody for the festival", "{subj} wove a tapestry of silk",
        "{subj} forged a blade in the smithy", "{subj} sculpted a statue from clay",
        "{subj} drew a map of the territory", "{subj} designed a bridge for the town",
    ],
    "destruction": [
        "{subj} smashed the vase against the wall", "{subj} tore the letter to pieces",
        "{subj} burned the old documents", "{subj} shattered the mirror with a stone",
        "{subj} demolished the crumbling wall", "{subj} crushed the clay pot underfoot",
        "{subj} snapped the branch in two", "{subj} destroyed the evidence completely",
        "{subj} knocked down the wooden fence", "{subj} ripped the fabric apart",
    ],
    "perception": [
        "{subj} watched the sunset from the cliff", "{subj} listened to the birds singing",
        "{subj} noticed a crack in the wall", "{subj} felt the cold wind on their face",
        "{subj} smelled smoke in the distance", "{subj} tasted the bitter medicine",
        "{subj} spotted a figure in the fog", "{subj} heard footsteps behind them",
        "{subj} observed the stars through a telescope", "{subj} sensed danger nearby",
    ],
    "thought": [
        "{subj} considered the options carefully", "{subj} remembered the promise they made",
        "{subj} imagined a world without borders", "{subj} wondered about the meaning of it all",
        "{subj} realized the truth at last", "{subj} decided to leave before dawn",
        "{subj} believed every word of the story", "{subj} doubted the stranger's motives",
        "{subj} planned the journey in advance", "{subj} understood the message immediately",
    ],
}


def generate_slot10() -> List[Dict]:
    """Action Type pairs."""
    pairs = []
    subjects = PEOPLE[:30]
    action_types = list(SLOT10_ACTIONS.keys())

    for subj in subjects:
        for base_type in action_types:
            base_sents = SLOT10_ACTIONS[base_type]
            for var_type in action_types:
                if var_type == base_type:
                    continue
                var_sents = SLOT10_ACTIONS[var_type]
                for bs in base_sents:
                    for vs in var_sents:
                        pairs.append({
                            "slot": 10,
                            "base": bs.format(subj="the " + subj),
                            "variant": vs.format(subj="the " + subj),
                            "concept_value": var_type,
                        })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 11: Action Manner & Intensity
# ============================================================================

SLOT11_MANNERS = [
    "gently", "softly", "carefully", "tenderly", "delicately",
    "forcefully", "violently", "fiercely", "roughly", "brutally",
    "carelessly", "recklessly", "hastily", "clumsily", "wildly",
    "gracefully", "elegantly", "skillfully", "calmly", "steadily",
    "desperately", "frantically", "methodically", "lazily", "eagerly",
]

SLOT11_TEMPLATES = [
    "the {subj} {manner} pushed the door open",
    "she {manner} placed the cup on the table",
    "the {subj} {manner} lifted the heavy stone",
    "he {manner} pulled the rope toward himself",
    "the {subj} {manner} swung the sword at the target",
    "she {manner} stirred the mixture in the pot",
    "the {subj} {manner} turned the pages of the book",
    "he {manner} pressed the button on the wall",
    "the {subj} {manner} knocked on the wooden door",
    "she {manner} brushed the dust from the surface",
    "the {subj} {manner} threw the ball across the yard",
    "he {manner} carved the wood into a figure",
    "the {subj} {manner} climbed the ladder to the roof",
    "she {manner} folded the cloth into a neat square",
    "the {subj} {manner} wrapped the gift in paper",
    "he {manner} broke the seal on the letter",
    "the {subj} {manner} lowered the basket into the well",
    "she {manner} drew the curtains across the window",
    "the {subj} {manner} guided the horse through the pass",
    "he {manner} set the lantern on the ground",
]


def generate_slot11() -> List[Dict]:
    """Action Manner & Intensity pairs."""
    pairs = []
    subjects = PEOPLE[:25]

    for template in SLOT11_TEMPLATES:
        for subj in subjects:
            for base_m in SLOT11_MANNERS:
                for var_m in SLOT11_MANNERS:
                    if var_m == base_m:
                        continue
                    pairs.append({
                        "slot": 11,
                        "base": template.format(subj=subj, manner=base_m),
                        "variant": template.format(subj=subj, manner=var_m),
                        "concept_value": var_m,
                    })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 12: Speed & Completion
# ============================================================================

SLOT12_SPEEDS = [
    "slowly", "gradually", "steadily", "quickly", "rapidly", "instantly",
    "suddenly", "leisurely", "briskly", "swiftly",
]

SLOT12_COMPLETION = {
    "starting": [
        "the {subj} began to {verb}",
        "the {subj} started {verb}ing",
        "the {subj} was about to {verb}",
        "the {subj} prepared to {verb}",
        "the {subj} was on the verge of {verb}ing",
    ],
    "ongoing": [
        "the {subj} was {verb}ing at that moment",
        "the {subj} kept {verb}ing without pause",
        "the {subj} continued to {verb}",
        "the {subj} was still {verb}ing",
        "the {subj} went on {verb}ing",
    ],
    "finished": [
        "the {subj} had already {verb}ed",
        "the {subj} finished {verb}ing",
        "the {subj} was done {verb}ing",
        "the {subj} had completed the {verb}ing",
        "the {subj} stopped {verb}ing at last",
    ],
}

SLOT12_VERBS = [
    "walk", "climb", "paint", "build", "cook", "read", "search",
    "clean", "pack", "carve", "weave", "plant", "harvest", "dig",
]

SLOT12_SPEED_TEMPLATES = [
    "the {subj} {speed} crossed the bridge",
    "she {speed} opened the heavy door",
    "the {subj} {speed} made their way through the crowd",
    "he {speed} finished the remaining work",
    "the {subj} {speed} approached the stranger",
    "she {speed} wrapped the package in brown paper",
    "the {subj} {speed} descended the winding staircase",
    "he {speed} scanned the document for errors",
    "the {subj} {speed} gathered their belongings",
    "she {speed} poured the water into the basin",
    "the {subj} {speed} climbed the hill in the rain",
    "he {speed} sorted through the pile of letters",
    "the {subj} {speed} counted the coins on the table",
    "she {speed} stitched the torn fabric back together",
    "the {subj} {speed} loaded the cart with supplies",
    "he {speed} ate the meal without tasting it",
    "the {subj} {speed} walked away from the argument",
    "she {speed} pulled the thread through the needle",
    "the {subj} {speed} turned the crank on the machine",
    "he {speed} flipped through the pages of the journal",
]


def generate_slot12() -> List[Dict]:
    """Speed & Completion pairs."""
    pairs = []
    subjects = PEOPLE[:20]

    # Speed variations
    for template in SLOT12_SPEED_TEMPLATES:
        for subj in subjects:
            for base_s in SLOT12_SPEEDS:
                for var_s in SLOT12_SPEEDS:
                    if var_s == base_s:
                        continue
                    pairs.append({
                        "slot": 12,
                        "base": template.format(subj=subj, speed=base_s),
                        "variant": template.format(subj=subj, speed=var_s),
                        "concept_value": var_s,
                    })

    # Completion variations
    phases = list(SLOT12_COMPLETION.keys())
    for subj in subjects:
        for verb in SLOT12_VERBS:
            for base_phase in phases:
                for var_phase in phases:
                    if var_phase == base_phase:
                        continue
                    for bt in SLOT12_COMPLETION[base_phase]:
                        for vt in SLOT12_COMPLETION[var_phase]:
                            pairs.append({
                                "slot": 12,
                                "base": bt.format(subj=subj, verb=verb),
                                "variant": vt.format(subj=subj, verb=verb),
                                "concept_value": var_phase,
                            })

    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 13: Direction & Path
# ============================================================================

SLOT13_DIRECTIONS = [
    "upward", "downward", "forward", "backward", "to the left", "to the right",
    "toward the gate", "away from the fire", "into the darkness", "out of the cave",
    "across the field", "through the tunnel", "around the corner", "over the wall",
    "along the river", "past the church", "beneath the bridge", "beyond the hill",
]

SLOT13_TEMPLATES = [
    "the {subj} moved {dir}",
    "she pointed {dir} and whispered",
    "the {subj} ran {dir} without looking back",
    "he threw the stone {dir}",
    "the {subj} glanced {dir} nervously",
    "she dragged the crate {dir}",
    "the {subj} pushed the cart {dir}",
    "he rolled the barrel {dir}",
    "the {subj} walked {dir} in silence",
    "she turned {dir} and vanished",
    "the {subj} climbed {dir} using the old rope",
    "he swam {dir} against the current",
    "the {subj} galloped {dir} at full speed",
    "she crawled {dir} to avoid being seen",
    "the {subj} leapt {dir} with surprising agility",
    "he stumbled {dir} in the dark",
    "the {subj} slid {dir} on the wet floor",
    "she marched {dir} with determination",
    "the {subj} floated {dir} on the gentle breeze",
    "he drifted {dir} without any clear destination",
]


def generate_slot13() -> List[Dict]:
    """Direction & Path pairs."""
    pairs = []
    subjects = PEOPLE[:20] + ANIMALS[:10]

    for template in SLOT13_TEMPLATES:
        for subj in subjects:
            for base_d in SLOT13_DIRECTIONS:
                for var_d in SLOT13_DIRECTIONS:
                    if var_d == base_d:
                        continue
                    pairs.append({
                        "slot": 13,
                        "base": template.format(subj=subj, dir=base_d),
                        "variant": template.format(subj=subj, dir=var_d),
                        "concept_value": var_d,
                    })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 14: Location & Scene
# ============================================================================

SLOT14_LOCATIONS = {
    "indoor": [
        "inside the old library", "in the kitchen", "in the attic",
        "in the basement", "in the grand hall", "inside the chapel",
        "in the bedroom", "in the workshop", "inside the tavern",
        "in the cellar",
    ],
    "outdoor": [
        "in the open field", "under the oak tree", "beside the road",
        "in the garden", "on the hilltop", "at the crossroads",
        "in the courtyard", "along the riverbank", "at the edge of the woods",
        "beneath the stars",
    ],
    "urban": [
        "in the crowded marketplace", "on the busy street", "in the town square",
        "at the city gate", "near the clock tower", "in the cobblestone alley",
        "at the harbor", "in the merchant quarter", "near the old fountain",
        "on the bridge over the canal",
    ],
    "rural": [
        "in the quiet village", "on the farm", "in the wheat field",
        "near the old barn", "beside the stream", "in the orchard",
        "on the dirt path", "near the windmill", "in the meadow",
        "at the edge of the pasture",
    ],
    "wilderness": [
        "deep in the forest", "on the mountain slope", "in the desert",
        "at the cave entrance", "in the swamp", "on the rocky shore",
        "in the dense jungle", "on the frozen tundra", "in the canyon",
        "at the waterfall",
    ],
    "underwater": [
        "beneath the waves", "on the ocean floor", "in the coral reef",
        "in the underwater cave", "among the kelp forest", "in the deep trench",
        "near the sunken ship", "in the tidal pool", "at the bottom of the lake",
        "in the flooded chamber",
    ],
}

SLOT14_TEMPLATES = [
    "the {subj} found shelter {loc}",
    "she discovered the artifact {loc}",
    "the {subj} waited patiently {loc}",
    "he made camp {loc} for the night",
    "the {subj} explored every corner {loc}",
    "she heard a strange sound {loc}",
    "the {subj} rested {loc} after the long journey",
    "he stumbled upon a clue {loc}",
    "the {subj} trained {loc} every morning",
    "she met the stranger {loc}",
    "the {subj} hid the treasure {loc}",
    "he fought the beast {loc}",
    "the {subj} prayed silently {loc}",
    "she read the old book {loc}",
    "the {subj} sang a quiet song {loc}",
    "he built a fire {loc}",
    "the {subj} stood guard {loc} all night",
    "she painted a picture {loc}",
    "the {subj} practiced swordplay {loc}",
    "he fell asleep {loc} without meaning to",
]


def generate_slot14() -> List[Dict]:
    """Location & Scene pairs."""
    pairs = []
    subjects = PEOPLE[:20]
    loc_types = list(SLOT14_LOCATIONS.keys())

    for template in SLOT14_TEMPLATES:
        for subj in subjects:
            for base_type in loc_types:
                for var_type in loc_types:
                    if var_type == base_type:
                        continue
                    for bl in SLOT14_LOCATIONS[base_type][:5]:
                        for vl in random.sample(SLOT14_LOCATIONS[var_type], min(3, len(SLOT14_LOCATIONS[var_type]))):
                            pairs.append({
                                "slot": 14,
                                "base": template.format(subj=subj, loc=bl),
                                "variant": template.format(subj=subj, loc=vl),
                                "concept_value": var_type,
                            })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 15: Spatial Relations
# ============================================================================

SLOT15_RELATIONS = [
    "above", "below", "inside", "outside", "beside", "between",
    "behind", "in front of", "beneath", "on top of", "next to",
    "underneath", "within", "across from", "near", "far from",
    "at the center of", "at the edge of", "surrounding", "opposite",
]

SLOT15_ANCHORS = [
    "the old oak tree", "the stone wall", "the wooden fence", "the church tower",
    "the garden gate", "the stone bridge", "the ancient well", "the market stall",
    "the broken fountain", "the iron gate", "the tall hedge", "the barn door",
    "the clock tower", "the stone archway", "the village square",
]

SLOT15_TEMPLATES = [
    "the {noun} was {rel} {anchor}",
    "she found the key {rel} {anchor}",
    "the {noun} stood {rel} {anchor}",
    "he noticed something {rel} {anchor}",
    "the {noun} had been placed {rel} {anchor}",
    "a shadow appeared {rel} {anchor}",
    "they built the shelter {rel} {anchor}",
    "the treasure was hidden {rel} {anchor}",
    "the {noun} waited silently {rel} {anchor}",
    "he spotted the footprints {rel} {anchor}",
    "the flowers grew {rel} {anchor}",
    "she hid {rel} {anchor} to avoid being seen",
    "the {noun} rested {rel} {anchor}",
    "something glinted {rel} {anchor}",
    "the camp was set up {rel} {anchor}",
    "a light flickered {rel} {anchor}",
    "the {noun} appeared {rel} {anchor} out of nowhere",
    "he left the message {rel} {anchor}",
    "the small creature sat {rel} {anchor}",
    "she placed the offering {rel} {anchor}",
]


def generate_slot15() -> List[Dict]:
    """Spatial Relations pairs."""
    pairs = []
    nouns = CONCRETE_NOUNS[:20]

    for template in SLOT15_TEMPLATES:
        for noun in nouns:
            for anchor in SLOT15_ANCHORS:
                for base_r in SLOT15_RELATIONS:
                    for var_r in SLOT15_RELATIONS:
                        if var_r == base_r:
                            continue
                        pairs.append({
                            "slot": 15,
                            "base": template.format(noun=noun, rel=base_r, anchor=anchor),
                            "variant": template.format(noun=noun, rel=var_r, anchor=anchor),
                            "concept_value": var_r,
                        })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 16: Distance & Proximity
# ============================================================================

SLOT16_DISTANCES = [
    "right here", "nearby", "close by", "just a few steps away",
    "a short walk away", "across the street", "down the road",
    "a mile away", "far in the distance", "on the distant horizon",
    "miles and miles away", "on the other side of the valley",
    "at the far end of the world", "barely visible in the distance",
    "just around the corner", "adjacent to this spot",
]

SLOT16_TEMPLATES = [
    "the {noun} was {dist}",
    "she could see the {noun} {dist}",
    "the village was {dist}",
    "he pointed to something {dist}",
    "the {noun} lay {dist}",
    "the sound came from {dist}",
    "they spotted a fire {dist}",
    "the mountains rose {dist}",
    "a tower stood {dist}",
    "the {noun} waited {dist}",
    "she heard footsteps from {dist}",
    "the {noun} was located {dist}",
    "he could barely make out the shape {dist}",
    "the forest began {dist}",
    "a river flowed {dist}",
    "the {noun} appeared {dist} through the haze",
    "a caravan was traveling {dist}",
    "the lighthouse blinked {dist}",
    "the {noun} gleamed {dist}",
    "she waved to someone standing {dist}",
]


def generate_slot16() -> List[Dict]:
    """Distance & Proximity pairs."""
    pairs = []
    nouns = ALL_NOUNS[:30]

    for template in SLOT16_TEMPLATES:
        for noun in nouns:
            for base_d in SLOT16_DISTANCES:
                for var_d in SLOT16_DISTANCES:
                    if var_d == base_d:
                        continue
                    pairs.append({
                        "slot": 16,
                        "base": template.format(noun=noun, dist=base_d),
                        "variant": template.format(noun=noun, dist=var_d),
                        "concept_value": var_d,
                    })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 17: Tense & Aspect
# ============================================================================

SLOT17_TENSES = {
    "past_simple": [
        "the {subj} walked through the gate",
        "she opened the letter and read it",
        "the {subj} climbed the tower at dawn",
        "he carried the basket to the market",
        "the {subj} painted the fence white",
        "she wrote a message on the wall",
        "the {subj} crossed the bridge in silence",
        "he found a coin beneath the floorboards",
        "the {subj} built a fire to stay warm",
        "she sang a song to calm the child",
    ],
    "present_simple": [
        "the {subj} walks through the gate",
        "she opens the letter and reads it",
        "the {subj} climbs the tower at dawn",
        "he carries the basket to the market",
        "the {subj} paints the fence white",
        "she writes a message on the wall",
        "the {subj} crosses the bridge in silence",
        "he finds a coin beneath the floorboards",
        "the {subj} builds a fire to stay warm",
        "she sings a song to calm the child",
    ],
    "future_simple": [
        "the {subj} will walk through the gate",
        "she will open the letter and read it",
        "the {subj} will climb the tower at dawn",
        "he will carry the basket to the market",
        "the {subj} will paint the fence white",
        "she will write a message on the wall",
        "the {subj} will cross the bridge in silence",
        "he will find a coin beneath the floorboards",
        "the {subj} will build a fire to stay warm",
        "she will sing a song to calm the child",
    ],
    "past_continuous": [
        "the {subj} was walking through the gate",
        "she was opening the letter and reading it",
        "the {subj} was climbing the tower at dawn",
        "he was carrying the basket to the market",
        "the {subj} was painting the fence white",
        "she was writing a message on the wall",
        "the {subj} was crossing the bridge in silence",
        "he was finding a coin beneath the floorboards",
        "the {subj} was building a fire to stay warm",
        "she was singing a song to calm the child",
    ],
    "present_perfect": [
        "the {subj} has walked through the gate",
        "she has opened the letter and read it",
        "the {subj} has climbed the tower at dawn",
        "he has carried the basket to the market",
        "the {subj} has painted the fence white",
        "she has written a message on the wall",
        "the {subj} has crossed the bridge in silence",
        "he has found a coin beneath the floorboards",
        "the {subj} has built a fire to stay warm",
        "she has sung a song to calm the child",
    ],
    "habitual": [
        "the {subj} always walks through the gate",
        "she usually opens the letter and reads it",
        "the {subj} often climbs the tower at dawn",
        "he regularly carries the basket to the market",
        "the {subj} frequently paints the fence white",
        "she habitually writes a message on the wall",
        "the {subj} routinely crosses the bridge in silence",
        "he typically finds a coin beneath the floorboards",
        "the {subj} normally builds a fire to stay warm",
        "she often sings a song to calm the child",
    ],
}


def generate_slot17() -> List[Dict]:
    """Tense & Aspect pairs."""
    pairs = []
    subjects = PEOPLE[:25]
    tenses = list(SLOT17_TENSES.keys())

    for subj in subjects:
        for base_tense in tenses:
            for var_tense in tenses:
                if var_tense == base_tense:
                    continue
                base_sents = SLOT17_TENSES[base_tense]
                var_sents = SLOT17_TENSES[var_tense]
                for i in range(len(base_sents)):
                    pairs.append({
                        "slot": 17,
                        "base": base_sents[i].format(subj=subj),
                        "variant": var_sents[i].format(subj=subj),
                        "concept_value": var_tense,
                    })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 18: Duration & Frequency
# ============================================================================

SLOT18_DURATIONS = [
    "for a brief moment", "for a few seconds", "for several minutes",
    "for half an hour", "for hours on end", "for the entire day",
    "for weeks at a time", "for months", "for years", "forever",
    "only an instant", "all night long",
]

SLOT18_FREQUENCIES = [
    "once", "twice", "a few times", "occasionally", "sometimes",
    "often", "frequently", "regularly", "always", "never",
    "every single day", "once in a while", "every now and then",
    "constantly", "rarely", "seldom",
]

SLOT18_TEMPLATES_DUR = [
    "the {subj} waited {dur}",
    "she stared at the painting {dur}",
    "the {subj} listened to the music {dur}",
    "he sat by the window {dur}",
    "the {subj} traveled {dur} before stopping",
    "she held the child's hand {dur}",
    "the {subj} searched the ruins {dur}",
    "he studied the map {dur}",
    "the {subj} practiced the skill {dur}",
    "she watched the horizon {dur}",
    "the {subj} slept {dur} without waking",
    "he pondered the question {dur}",
    "the {subj} guarded the entrance {dur}",
    "she mourned {dur} in silence",
    "the {subj} worked on the project {dur}",
]

SLOT18_TEMPLATES_FREQ = [
    "the {subj} {freq} visited the old temple",
    "she {freq} walked along the shore at dusk",
    "the {subj} {freq} thought about the past",
    "he {freq} returned to the place where it happened",
    "the {subj} {freq} told that story to the children",
    "she {freq} baked bread on Sunday mornings",
    "the {subj} {freq} looked up at the stars",
    "he {freq} wrote letters to his family",
    "the {subj} {freq} played music in the square",
    "she {freq} dreamed of a different life",
    "the {subj} {freq} helped the neighbors with the harvest",
    "he {freq} forgot to lock the door",
    "the {subj} {freq} missed the train by minutes",
    "she {freq} stopped to admire the flowers",
    "the {subj} {freq} checked the old clock on the wall",
]


def generate_slot18() -> List[Dict]:
    """Duration & Frequency pairs."""
    pairs = []
    subjects = PEOPLE[:20]

    for template in SLOT18_TEMPLATES_DUR:
        for subj in subjects:
            for base_d in SLOT18_DURATIONS:
                for var_d in SLOT18_DURATIONS:
                    if var_d == base_d:
                        continue
                    pairs.append({
                        "slot": 18,
                        "base": template.format(subj=subj, dur=base_d),
                        "variant": template.format(subj=subj, dur=var_d),
                        "concept_value": var_d,
                    })

    for template in SLOT18_TEMPLATES_FREQ:
        for subj in subjects:
            for base_f in SLOT18_FREQUENCIES:
                for var_f in SLOT18_FREQUENCIES:
                    if var_f == base_f:
                        continue
                    pairs.append({
                        "slot": 18,
                        "base": template.format(subj=subj, freq=base_f),
                        "variant": template.format(subj=subj, freq=var_f),
                        "concept_value": var_f,
                    })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 19: Time Reference
# ============================================================================

SLOT19_TIMES = [
    "at dawn", "in the early morning", "at sunrise", "at noon",
    "in the afternoon", "at dusk", "in the evening", "at sunset",
    "at midnight", "in the dead of night", "at twilight", "before first light",
]

SLOT19_SEASONS = [
    "in the spring", "during the summer", "in the autumn", "in the winter",
    "at the start of the rainy season", "during the harvest",
]

SLOT19_ERAS = [
    "in ancient times", "in the distant past", "centuries ago",
    "in the modern era", "in recent years", "long before anyone could remember",
    "in the age of kings", "during the golden age", "in a forgotten era",
]

SLOT19_ALL = SLOT19_TIMES + SLOT19_SEASONS + SLOT19_ERAS

SLOT19_TEMPLATES = [
    "{time} the {subj} set out on the journey",
    "the {subj} always woke {time}",
    "{time} the market came alive with noise",
    "the bell rang {time} without fail",
    "{time} the village was quiet and still",
    "the {subj} preferred to work {time}",
    "{time} the sky turned a shade of gold",
    "the guards changed shifts {time}",
    "{time} the wind picked up considerably",
    "the {subj} returned home {time}",
    "{time} the streets were nearly empty",
    "the ceremony took place {time}",
    "{time} the {subj} sat down to eat",
    "the music always played {time}",
    "{time} the river rose above its banks",
    "the {subj} finished the task {time}",
    "{time} the air smelled of wood smoke",
    "the {subj} went fishing {time}",
    "{time} shadows stretched long across the ground",
    "the {subj} told stories {time}",
]


def generate_slot19() -> List[Dict]:
    """Time Reference pairs."""
    pairs = []
    subjects = PEOPLE[:20]

    for template in SLOT19_TEMPLATES:
        for subj in subjects:
            for base_t in SLOT19_ALL:
                for var_t in SLOT19_ALL:
                    if var_t == base_t:
                        continue
                    pairs.append({
                        "slot": 19,
                        "base": template.format(subj=subj, time=base_t),
                        "variant": template.format(subj=subj, time=var_t),
                        "concept_value": var_t,
                    })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 20: Number & Amount
# ============================================================================

SLOT20_QUANTITIES = [
    "one", "two", "three", "a few", "several", "a dozen",
    "many", "hundreds of", "thousands of", "countless",
    "a pair of", "a handful of", "no", "all the",
]

SLOT20_TEMPLATES = [
    "{qty} {noun} stood at the entrance",
    "she counted {qty} {noun} on the shelf",
    "{qty} {noun} arrived before sunset",
    "there were {qty} {noun} in the basket",
    "he noticed {qty} {noun} scattered on the ground",
    "{qty} {noun} blocked the narrow path",
    "she carried {qty} {noun} in her arms",
    "{qty} {noun} floated down the river",
    "the table held {qty} {noun} of different sizes",
    "{qty} {noun} had been placed along the wall",
    "he found {qty} {noun} hidden in the chest",
    "{qty} {noun} appeared out of the mist",
    "she traded {qty} {noun} for a loaf of bread",
    "{qty} {noun} were left after the storm",
    "the merchant sold {qty} {noun} that morning",
    "{qty} {noun} circled overhead in the sky",
    "he wrapped {qty} {noun} in a cloth",
    "{qty} {noun} rested on the windowsill",
    "she bought {qty} {noun} from the traveling vendor",
    "{qty} {noun} tumbled down the hillside",
]

SLOT20_NOUNS = [
    "stones", "birds", "candles", "coins", "flowers", "letters", "swords",
    "bottles", "keys", "rings", "books", "apples", "fish", "horses",
    "soldiers", "children", "ships", "stars", "trees", "wolves",
    "lanterns", "barrels", "baskets", "cats", "flags", "shields",
]


def generate_slot20() -> List[Dict]:
    """Number & Amount pairs."""
    pairs = []
    for template in SLOT20_TEMPLATES:
        for noun in SLOT20_NOUNS:
            for base_q in SLOT20_QUANTITIES:
                for var_q in SLOT20_QUANTITIES:
                    if var_q == base_q:
                        continue
                    pairs.append({
                        "slot": 20,
                        "base": template.format(qty=base_q, noun=noun),
                        "variant": template.format(qty=var_q, noun=noun),
                        "concept_value": var_q,
                    })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 21: Degree & Comparison
# ============================================================================

SLOT21_DEGREES = [
    "slightly", "somewhat", "moderately", "fairly", "quite",
    "very", "extremely", "incredibly", "remarkably", "extraordinarily",
    "barely", "hardly", "almost entirely", "completely", "utterly",
]

SLOT21_TEMPLATES = [
    "the weather was {deg} cold that morning",
    "she was {deg} satisfied with the result",
    "the task proved {deg} difficult for the group",
    "he felt {deg} confident about the plan",
    "the road was {deg} dangerous at night",
    "the food tasted {deg} different from usual",
    "the room was {deg} dark when they entered",
    "she found the story {deg} convincing",
    "the landscape was {deg} beautiful in the spring",
    "he was {deg} surprised by the outcome",
    "the lake was {deg} calm before the storm",
    "the solution was {deg} simple once they understood",
    "the crowd was {deg} hostile toward the outsider",
    "the air was {deg} thick with smoke",
    "she was {deg} tired after the long walk",
    "the forest was {deg} quiet at that hour",
    "the news was {deg} troubling to everyone",
    "the performance was {deg} impressive by any standard",
    "the old house was {deg} damaged by the flood",
    "he was {deg} aware of the danger ahead",
]


def generate_slot21() -> List[Dict]:
    """Degree & Comparison pairs."""
    pairs = []
    for template in SLOT21_TEMPLATES:
        for base_d in SLOT21_DEGREES:
            for var_d in SLOT21_DEGREES:
                if var_d == base_d:
                    continue
                pairs.append({
                    "slot": 21,
                    "base": template.format(deg=base_d),
                    "variant": template.format(deg=var_d),
                    "concept_value": var_d,
                })
    return pairs


# ============================================================================
# SLOT 22: Core Sentiment
# ============================================================================

SLOT22_SENTIMENTS = {
    "positive": [
        "the {subj} smiled warmly at the crowd",
        "the {subj} felt a wave of joy wash over them",
        "it was a wonderful day for everyone involved",
        "the {subj} was grateful for the kind gesture",
        "everything turned out beautifully in the end",
        "the {subj} celebrated the good news with friends",
        "a sense of hope filled the room",
        "the {subj} was delighted by the surprise",
        "the view from the hilltop was breathtaking",
        "the {subj} embraced the opportunity with enthusiasm",
        "a warm feeling spread through the village",
        "the {subj} laughed with genuine happiness",
        "the garden was blooming with life and color",
        "the {subj} felt at peace with the world",
        "it was a blessing nobody expected",
    ],
    "negative": [
        "the {subj} frowned deeply at the news",
        "the {subj} felt a wave of dread wash over them",
        "it was a terrible day for everyone involved",
        "the {subj} was bitter about the unfair decision",
        "everything fell apart in the end",
        "the {subj} mourned the devastating loss alone",
        "a sense of despair filled the room",
        "the {subj} was crushed by the betrayal",
        "the view from the hilltop was bleak and desolate",
        "the {subj} rejected the idea with disgust",
        "a cold dread spread through the village",
        "the {subj} wept with genuine sorrow",
        "the garden was withered and dead",
        "the {subj} felt at odds with the world",
        "it was a tragedy nobody anticipated",
    ],
    "neutral": [
        "the {subj} observed the scene without expression",
        "the {subj} felt nothing particular about it",
        "it was an ordinary day like any other",
        "the {subj} acknowledged the fact without comment",
        "everything proceeded as expected",
        "the {subj} noted the information and moved on",
        "a sense of normalcy filled the room",
        "the {subj} accepted the news calmly",
        "the view from the hilltop was unremarkable",
        "the {subj} considered the matter objectively",
        "a quiet stillness settled over the village",
        "the {subj} nodded without visible emotion",
        "the garden was plain and untended",
        "the {subj} felt indifferent to the outcome",
        "it was an event of no particular significance",
    ],
}


def generate_slot22() -> List[Dict]:
    """Core Sentiment pairs."""
    pairs = []
    subjects = PEOPLE[:25]
    sentiments = list(SLOT22_SENTIMENTS.keys())

    for subj in subjects:
        for base_s in sentiments:
            for var_s in sentiments:
                if var_s == base_s:
                    continue
                for i in range(len(SLOT22_SENTIMENTS[base_s])):
                    pairs.append({
                        "slot": 22,
                        "base": SLOT22_SENTIMENTS[base_s][i].format(subj=subj),
                        "variant": SLOT22_SENTIMENTS[var_s][i].format(subj=subj),
                        "concept_value": var_s,
                    })
    return pairs


# ============================================================================
# SLOT 23: Specific Emotion
# ============================================================================

SLOT23_EMOTIONS = {
    "happy": ["smiled brightly", "beamed with joy", "laughed with delight", "grinned from ear to ear"],
    "sad": ["wept quietly", "sobbed into their hands", "sighed with sorrow", "hung their head in grief"],
    "angry": ["clenched their fists in rage", "shouted furiously", "slammed the door in anger", "glared with fury"],
    "afraid": ["trembled with fear", "backed away in terror", "froze in panic", "screamed in fright"],
    "surprised": ["gasped in astonishment", "stared in disbelief", "jumped back in shock", "blinked in amazement"],
    "disgusted": ["recoiled in disgust", "turned away in revulsion", "grimaced with distaste", "wrinkled their nose"],
    "proud": ["stood tall with pride", "puffed out their chest", "held their head high", "spoke with confidence"],
    "jealous": ["watched enviously", "burned with jealousy", "seethed with envy", "glowered with resentment"],
    "grateful": ["bowed in gratitude", "thanked them profusely", "expressed deep appreciation", "clasped their hands in thanks"],
    "lonely": ["sat alone in silence", "stared out the window with longing", "wandered the empty halls", "felt the ache of solitude"],
}

SLOT23_TEMPLATES = [
    "the {subj} {emotion_phrase}",
    "when they heard the news the {subj} {emotion_phrase}",
    "standing in the doorway the {subj} {emotion_phrase}",
    "after hearing the verdict the {subj} {emotion_phrase}",
    "upon seeing the letter the {subj} {emotion_phrase}",
    "the {subj} {emotion_phrase} as the sun set",
    "without warning the {subj} {emotion_phrase}",
    "at the ceremony the {subj} {emotion_phrase}",
    "looking at the old photograph the {subj} {emotion_phrase}",
    "the {subj} {emotion_phrase} before anyone else could react",
    "after the long silence the {subj} {emotion_phrase}",
    "in the middle of the crowd the {subj} {emotion_phrase}",
    "once the truth was revealed the {subj} {emotion_phrase}",
    "the {subj} {emotion_phrase} and everyone understood",
    "as the door opened the {subj} {emotion_phrase}",
    "the {subj} {emotion_phrase} at the sight of the gift",
    "returning home the {subj} {emotion_phrase}",
    "the {subj} {emotion_phrase} when no one was looking",
    "hearing the music the {subj} {emotion_phrase}",
    "the {subj} {emotion_phrase} despite trying to hide it",
]


def generate_slot23() -> List[Dict]:
    """Specific Emotion pairs."""
    pairs = []
    subjects = PEOPLE[:25]
    emotions = list(SLOT23_EMOTIONS.keys())

    for template in SLOT23_TEMPLATES:
        for subj in subjects:
            for base_e in emotions:
                for var_e in emotions:
                    if var_e == base_e:
                        continue
                    for bp in SLOT23_EMOTIONS[base_e]:
                        for vp in SLOT23_EMOTIONS[var_e]:
                            pairs.append({
                                "slot": 23,
                                "base": template.format(subj=subj, emotion_phrase=bp),
                                "variant": template.format(subj=subj, emotion_phrase=vp),
                                "concept_value": var_e,
                            })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 24: Arousal & Energy
# ============================================================================

SLOT24_AROUSAL = {
    "calm": ["sat peacefully", "breathed slowly and evenly", "rested with closed eyes", "remained perfectly still"],
    "relaxed": ["leaned back comfortably", "lounged in the shade", "stretched lazily", "yawned and settled in"],
    "alert": ["scanned the area carefully", "listened intently for sounds", "watched with sharp eyes", "stood at attention"],
    "excited": ["bounced on their toes", "clapped their hands eagerly", "spoke with breathless enthusiasm", "could barely contain themselves"],
    "frantic": ["paced back and forth rapidly", "scrambled in every direction", "shouted orders desperately", "ran around in a panic"],
    "peaceful": ["gazed at the still water", "hummed a quiet melody", "sat in contemplative silence", "let their mind drift gently"],
    "restless": ["fidgeted constantly", "tapped their foot impatiently", "shifted from side to side", "could not sit still for a moment"],
    "tense": ["gripped the armrest tightly", "clenched their jaw", "held their breath nervously", "stiffened at every noise"],
}

SLOT24_TEMPLATES = [
    "the {subj} {arousal_phrase}",
    "in the waiting room the {subj} {arousal_phrase}",
    "before the announcement the {subj} {arousal_phrase}",
    "during the meeting the {subj} {arousal_phrase}",
    "while everyone else talked the {subj} {arousal_phrase}",
    "throughout the evening the {subj} {arousal_phrase}",
    "as the hour approached the {subj} {arousal_phrase}",
    "sitting by the fire the {subj} {arousal_phrase}",
    "the {subj} {arousal_phrase} as they waited for the signal",
    "alone in the room the {subj} {arousal_phrase}",
    "after the long day the {subj} {arousal_phrase}",
    "the {subj} {arousal_phrase} while the others slept",
    "standing at the gate the {subj} {arousal_phrase}",
    "the {subj} {arousal_phrase} as the clock ticked on",
    "on the eve of the journey the {subj} {arousal_phrase}",
    "with nothing left to do the {subj} {arousal_phrase}",
    "upon hearing footsteps the {subj} {arousal_phrase}",
    "the {subj} {arousal_phrase} until the very end",
    "from the moment they arrived the {subj} {arousal_phrase}",
    "the {subj} {arousal_phrase} and everyone noticed",
]


def generate_slot24() -> List[Dict]:
    """Arousal & Energy pairs."""
    pairs = []
    subjects = PEOPLE[:25]
    arousal_types = list(SLOT24_AROUSAL.keys())

    for template in SLOT24_TEMPLATES:
        for subj in subjects:
            for base_a in arousal_types:
                for var_a in arousal_types:
                    if var_a == base_a:
                        continue
                    for bp in SLOT24_AROUSAL[base_a]:
                        for vp in SLOT24_AROUSAL[var_a]:
                            pairs.append({
                                "slot": 24,
                                "base": template.format(subj=subj, arousal_phrase=bp),
                                "variant": template.format(subj=subj, arousal_phrase=vp),
                                "concept_value": var_a,
                            })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 25: Quality & Value
# ============================================================================

SLOT25_QUALITIES = [
    "terrible", "awful", "poor", "mediocre", "decent", "acceptable",
    "good", "fine", "excellent", "outstanding", "superb", "perfect",
    "worthless", "flawed", "substandard", "remarkable", "exceptional",
    "magnificent", "dreadful", "adequate",
]

SLOT25_TEMPLATES = [
    "the {quality} craftsmanship was evident in every detail",
    "she considered the work to be {quality}",
    "the {quality} meal was served at the banquet",
    "he described the performance as {quality}",
    "the {quality} fabric was used for the king's robes",
    "she found the wine to be {quality}",
    "the {quality} sword was forged by the master smith",
    "he rated the lodging as {quality} at best",
    "the {quality} painting hung in the gallery for years",
    "she judged the proposal to be {quality}",
    "the {quality} harvest was a sign of the season",
    "he called the design {quality} without hesitation",
    "the {quality} reputation of the inn preceded it",
    "she deemed the translation {quality}",
    "the {quality} condition of the road slowed them down",
    "he found the taste to be {quality}",
    "the {quality} view from the terrace was well known",
    "she pronounced the effort {quality} and moved on",
    "the {quality} service left much to be desired",
    "he thought the new arrangement was {quality}",
]


def generate_slot25() -> List[Dict]:
    """Quality & Value pairs."""
    pairs = []
    for template in SLOT25_TEMPLATES:
        for base_q in SLOT25_QUALITIES:
            for var_q in SLOT25_QUALITIES:
                if var_q == base_q:
                    continue
                pairs.append({
                    "slot": 25,
                    "base": template.format(quality=base_q),
                    "variant": template.format(quality=var_q),
                    "concept_value": var_q,
                })
    return pairs


# ============================================================================
# SLOT 26: Difficulty & Importance
# ============================================================================

SLOT26_DIFFICULTY = [
    "easy", "simple", "straightforward", "moderate", "challenging",
    "difficult", "hard", "demanding", "arduous", "impossible",
    "effortless", "trivial", "grueling", "overwhelming",
]

SLOT26_IMPORTANCE = [
    "trivial", "minor", "unimportant", "routine", "significant",
    "important", "crucial", "critical", "vital", "essential",
    "urgent", "paramount", "negligible", "monumental",
]

SLOT26_TEMPLATES_DIFF = [
    "the task was {diff} for the apprentice",
    "she found the puzzle {diff} to solve",
    "the {diff} journey tested everyone's endurance",
    "he considered the problem {diff} but not impossible",
    "the {diff} terrain slowed the caravan to a halt",
    "she described the exam as {diff}",
    "the {diff} repair took most of the afternoon",
    "he admitted the negotiation was {diff}",
    "the {diff} climb required proper equipment",
    "she warned them that the route was {diff}",
    "the {diff} recipe required years of practice",
    "he found the language {diff} to learn at first",
    "the {diff} decision weighed on the council",
    "she approached the {diff} situation with caution",
    "the {diff} training regimen exhausted every recruit",
]

SLOT26_TEMPLATES_IMP = [
    "the {imp} matter was discussed at length",
    "she emphasized that the detail was {imp}",
    "the {imp} decision affected the entire kingdom",
    "he dismissed the issue as {imp}",
    "the {imp} meeting could not be postponed",
    "she considered the evidence {imp}",
    "the {imp} message had to be delivered at once",
    "he stressed that the rule was {imp}",
    "the {imp} discovery changed everything",
    "she treated the warning as {imp}",
    "the {imp} treaty was signed after months of debate",
    "he viewed the tradition as {imp} to their identity",
    "the {imp} fact was overlooked by most",
    "she marked the document as {imp}",
    "the {imp} lesson was taught to every student",
]


def generate_slot26() -> List[Dict]:
    """Difficulty & Importance pairs."""
    pairs = []
    for template in SLOT26_TEMPLATES_DIFF:
        for base_d in SLOT26_DIFFICULTY:
            for var_d in SLOT26_DIFFICULTY:
                if var_d == base_d:
                    continue
                pairs.append({
                    "slot": 26,
                    "base": template.format(diff=base_d),
                    "variant": template.format(diff=var_d),
                    "concept_value": var_d,
                })
    for template in SLOT26_TEMPLATES_IMP:
        for base_i in SLOT26_IMPORTANCE:
            for var_i in SLOT26_IMPORTANCE:
                if var_i == base_i:
                    continue
                pairs.append({
                    "slot": 26,
                    "base": template.format(imp=base_i),
                    "variant": template.format(imp=var_i),
                    "concept_value": var_i,
                })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 27: Negation & Truth
# ============================================================================

SLOT27_PAIRS = [
    # (affirmative, negated, concept_label)
    ("the {subj} was there", "the {subj} was not there", "negated"),
    ("she did finish the work", "she did not finish the work", "negated"),
    ("the {subj} believed the story", "the {subj} did not believe the story", "negated"),
    ("he could see the mountain", "he could not see the mountain", "negated"),
    ("the {subj} had the key", "the {subj} did not have the key", "negated"),
    ("she agreed with the plan", "she did not agree with the plan", "negated"),
    ("the {subj} knew the answer", "the {subj} did not know the answer", "negated"),
    ("he wanted to stay", "he did not want to stay", "negated"),
    ("the {subj} trusted the stranger", "the {subj} did not trust the stranger", "negated"),
    ("she expected to win", "she did not expect to win", "negated"),
    ("the door was locked", "the door was not locked", "negated"),
    ("he found the treasure", "he never found the treasure", "negated"),
    ("the {subj} remembered the promise", "the {subj} forgot the promise", "negated"),
    ("it was real", "it was fake", "fake"),
    ("the jewel was genuine", "the jewel was counterfeit", "fake"),
    ("the {subj} told the truth", "the {subj} told a lie", "false"),
    ("the story was true", "the story was false", "false"),
    ("the report was accurate", "the report was fabricated", "false"),
    ("the evidence was authentic", "the evidence was forged", "false"),
    ("the claim was valid", "the claim was baseless", "false"),
]

SLOT27_EXTRA_TEMPLATES_AFF = [
    "the {subj} did arrive on time",
    "she was indeed present at the meeting",
    "the {subj} truly understood the situation",
    "he certainly completed the assignment",
    "the {subj} really enjoyed the performance",
    "she actually found the missing piece",
    "the {subj} does care about the outcome",
    "he has seen the document before",
    "the {subj} will accept the terms",
    "she can solve this problem easily",
]

SLOT27_EXTRA_TEMPLATES_NEG = [
    "the {subj} did not arrive on time",
    "she was not present at the meeting",
    "the {subj} never understood the situation",
    "he certainly did not complete the assignment",
    "the {subj} did not enjoy the performance",
    "she never found the missing piece",
    "the {subj} does not care about the outcome",
    "he has never seen the document before",
    "the {subj} will not accept the terms",
    "she cannot solve this problem easily",
]


def generate_slot27() -> List[Dict]:
    """Negation & Truth pairs."""
    pairs = []
    subjects = PEOPLE[:30]

    for subj in subjects:
        # Fixed pair templates
        for aff, neg, label in SLOT27_PAIRS:
            pairs.append({
                "slot": 27,
                "base": aff.format(subj=subj),
                "variant": neg.format(subj=subj),
                "concept_value": label,
            })
            pairs.append({
                "slot": 27,
                "base": neg.format(subj=subj),
                "variant": aff.format(subj=subj),
                "concept_value": "affirmed",
            })

        # Extra templates
        for i in range(len(SLOT27_EXTRA_TEMPLATES_AFF)):
            pairs.append({
                "slot": 27,
                "base": SLOT27_EXTRA_TEMPLATES_AFF[i].format(subj=subj),
                "variant": SLOT27_EXTRA_TEMPLATES_NEG[i].format(subj=subj),
                "concept_value": "negated",
            })
            pairs.append({
                "slot": 27,
                "base": SLOT27_EXTRA_TEMPLATES_NEG[i].format(subj=subj),
                "variant": SLOT27_EXTRA_TEMPLATES_AFF[i].format(subj=subj),
                "concept_value": "affirmed",
            })
    return pairs


# ============================================================================
# SLOT 28: Certainty & Obligation
# ============================================================================

SLOT28_CERTAINTY = [
    "definitely", "certainly", "surely", "undoubtedly",
    "probably", "likely", "presumably",
    "possibly", "perhaps", "maybe", "conceivably",
    "unlikely to be", "impossibly",
]

SLOT28_OBLIGATION = [
    "must", "has to", "needs to", "is required to",
    "should", "ought to", "is supposed to",
    "may", "might", "could", "is allowed to",
    "cannot", "must not", "is forbidden to",
]

SLOT28_TEMPLATES_CERT = [
    "the {subj} will {cert} arrive before noon",
    "the treasure is {cert} hidden in the cave",
    "she {cert} knew the answer all along",
    "the plan will {cert} succeed this time",
    "the {subj} is {cert} the one responsible",
    "the storm will {cert} pass by morning",
    "the letter was {cert} written by the king",
    "the {subj} {cert} heard the noise last night",
    "the map {cert} leads to the right place",
    "the {subj} will {cert} return before winter",
    "the verdict was {cert} fair and just",
    "the witness {cert} saw everything that happened",
    "the {subj} {cert} left before the others",
    "the cure {cert} exists somewhere in the world",
    "the {subj} {cert} understood the warning",
    "she {cert} planned the whole thing in advance",
    "the outcome will {cert} be favorable",
    "the {subj} was {cert} innocent of the charge",
    "the bridge {cert} held despite the flood",
    "he {cert} noticed the change in her expression",
]

SLOT28_TEMPLATES_OBL = [
    "the {subj} {obl} finish the task today",
    "she {obl} report to the council at once",
    "the {subj} {obl} follow the rules carefully",
    "he {obl} pay the debt before the deadline",
    "the {subj} {obl} leave the premises immediately",
    "she {obl} inform the others of the decision",
    "the {subj} {obl} wait until further notice",
    "he {obl} speak with the elder first",
    "the {subj} {obl} present the evidence in court",
    "she {obl} return the borrowed item soon",
    "the {subj} {obl} attend the ceremony tomorrow",
    "he {obl} carry the message to the capital",
    "the {subj} {obl} protect the village at all costs",
    "she {obl} choose between the two offers",
    "the {subj} {obl} accept the consequences",
]


def generate_slot28() -> List[Dict]:
    """Certainty & Obligation pairs."""
    pairs = []
    subjects = PEOPLE[:20]

    for template in SLOT28_TEMPLATES_CERT:
        for subj in subjects:
            for base_c in SLOT28_CERTAINTY:
                for var_c in SLOT28_CERTAINTY:
                    if var_c == base_c:
                        continue
                    pairs.append({
                        "slot": 28,
                        "base": template.format(subj=subj, cert=base_c),
                        "variant": template.format(subj=subj, cert=var_c),
                        "concept_value": var_c,
                    })

    for template in SLOT28_TEMPLATES_OBL:
        for subj in subjects:
            for base_o in SLOT28_OBLIGATION:
                for var_o in SLOT28_OBLIGATION:
                    if var_o == base_o:
                        continue
                    pairs.append({
                        "slot": 28,
                        "base": template.format(subj=subj, obl=base_o),
                        "variant": template.format(subj=subj, obl=var_o),
                        "concept_value": var_o,
                    })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 29: Causation & Condition
# ============================================================================

SLOT29_CONNECTORS = {
    "because": "because",
    "therefore": "therefore",
    "despite": "despite that",
    "if": "if",
    "unless": "unless",
    "although": "although",
    "since": "since",
    "so_that": "so that",
    "even_though": "even though",
    "as_long_as": "as long as",
    "in_case": "in case",
    "provided_that": "provided that",
}

SLOT29_CLAUSES_A = [
    "the {subj} left early",
    "she finished the work",
    "the {subj} stayed behind",
    "he decided to help",
    "the {subj} changed the plan",
    "she refused the offer",
    "the {subj} spoke up",
    "he apologized publicly",
    "the {subj} joined the group",
    "she returned the favor",
    "the {subj} locked the door",
    "he sold the horse",
    "the {subj} planted the seeds",
    "she wrote the letter",
    "the {subj} hired a guide",
]

SLOT29_CLAUSES_B = [
    "the road was dangerous",
    "the weather improved",
    "the supplies ran low",
    "everyone agreed to the terms",
    "the deadline was approaching",
    "the rain finally stopped",
    "the king gave the order",
    "the bridge was repaired",
    "the harvest was plentiful",
    "the enemy retreated",
    "the treaty was signed",
    "the message arrived in time",
    "the river flooded its banks",
    "the rumors proved true",
    "the fire went out",
]


def generate_slot29() -> List[Dict]:
    """Causation & Condition pairs."""
    pairs = []
    subjects = PEOPLE[:15]
    connectors = list(SLOT29_CONNECTORS.items())

    for subj in subjects:
        for clause_a in SLOT29_CLAUSES_A:
            for clause_b in SLOT29_CLAUSES_B:
                a = clause_a.format(subj=subj)
                for base_key, base_conn in connectors:
                    for var_key, var_conn in connectors:
                        if var_key == base_key:
                            continue
                        pairs.append({
                            "slot": 29,
                            "base": f"{a} {base_conn} {clause_b}",
                            "variant": f"{a} {var_conn} {clause_b}",
                            "concept_value": var_key,
                        })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 30: Formality & Register
# ============================================================================

SLOT30_REGISTERS = {
    "slang": [
        "yo the {subj} totally nailed it",
        "the {subj} was like super stoked about it",
        "dude the {subj} just bounced outta there",
        "the {subj} was lowkey freaking out",
        "nah the {subj} ain't gonna do that",
        "the {subj} was all like whatever about it",
        "bet the {subj} crushed it no cap",
        "the {subj} straight up dipped without saying bye",
        "the {subj} was vibing hard all day",
        "fr the {subj} was goated at the task",
    ],
    "casual": [
        "the {subj} did a pretty good job honestly",
        "so the {subj} just went ahead and did it",
        "the {subj} was really happy about the whole thing",
        "well the {subj} decided to leave early",
        "the {subj} thought it was kind of weird",
        "turns out the {subj} knew all along",
        "the {subj} figured they would give it a shot",
        "the {subj} was cool with the arrangement",
        "the {subj} just showed up and got to work",
        "the {subj} had a great time at the event",
    ],
    "formal": [
        "the {subj} performed the task with commendable diligence",
        "the {subj} expressed satisfaction with the arrangement",
        "the {subj} chose to depart ahead of schedule",
        "it was determined that the {subj} had been informed",
        "the {subj} demonstrated considerable aptitude",
        "the {subj} conveyed their appreciation for the gesture",
        "the {subj} was observed departing the premises",
        "the {subj} presented the findings to the assembly",
        "the {subj} fulfilled the obligations without exception",
        "the {subj} acknowledged the significance of the decision",
    ],
    "academic": [
        "the {subj} exhibited a statistically significant improvement",
        "the {subj} demonstrated proficiency in the designated domain",
        "the {subj} was observed to exhibit the hypothesized behavior",
        "the {subj} contributed substantively to the body of literature",
        "the {subj} applied the theoretical framework systematically",
        "the {subj} operationalized the variable with precision",
        "the {subj} corroborated the preliminary findings",
        "the {subj} adhered to the established methodological protocol",
        "the {subj} yielded results consistent with prior research",
        "the {subj} articulated a nuanced interpretation of the data",
    ],
    "legal": [
        "the {subj} hereby acknowledges the terms set forth herein",
        "the {subj} shall comply with all applicable provisions",
        "the {subj} is bound by the conditions stipulated above",
        "the {subj} warrants that all representations are accurate",
        "the {subj} agrees to indemnify and hold harmless the parties",
        "the {subj} shall be liable for damages arising therefrom",
        "the {subj} is entitled to remedies as prescribed by law",
        "the {subj} covenants to perform the duties enumerated herein",
        "the {subj} affirms under penalty of perjury the following",
        "the {subj} petitions the court for relief from the obligations",
    ],
    "poetic": [
        "the {subj} wandered through the silver veil of twilight",
        "the {subj} danced upon the breath of the evening wind",
        "the {subj} wove a song from threads of starlight",
        "the {subj} carried the weight of unspoken sorrow",
        "the {subj} stood where the sea whispered to the shore",
        "the {subj} bloomed like a flower kissed by dawn",
        "the {subj} faded like a memory lost to time",
        "the {subj} spoke with a voice like flowing water",
        "the {subj} gazed upon the world through ancient eyes",
        "the {subj} walked a path of shadow and of light",
    ],
}


def generate_slot30() -> List[Dict]:
    """Formality & Register pairs."""
    pairs = []
    subjects = PEOPLE[:25]
    registers = list(SLOT30_REGISTERS.keys())

    for subj in subjects:
        for base_r in registers:
            for var_r in registers:
                if var_r == base_r:
                    continue
                base_sents = SLOT30_REGISTERS[base_r]
                var_sents = SLOT30_REGISTERS[var_r]
                for i in range(len(base_sents)):
                    pairs.append({
                        "slot": 30,
                        "base": base_sents[i].format(subj=subj),
                        "variant": var_sents[i].format(subj=subj),
                        "concept_value": var_r,
                    })
    return pairs


# ============================================================================
# SLOT 31: Speech Act & Intent
# ============================================================================

SLOT31_ACTS = {
    "statement": [
        "the {subj} left the village at dawn",
        "the bridge was built over a hundred years ago",
        "the harvest was better than expected this year",
        "the {subj} completed the assignment on time",
        "the market opens every morning at sunrise",
    ],
    "question": [
        "did the {subj} leave the village at dawn",
        "was the bridge built over a hundred years ago",
        "was the harvest better than expected this year",
        "did the {subj} complete the assignment on time",
        "does the market open every morning at sunrise",
    ],
    "command": [
        "leave the village at dawn {subj}",
        "build the bridge before the winter comes",
        "bring the harvest to the storehouse immediately",
        "complete the assignment on time {subj}",
        "open the market at sunrise without delay",
    ],
    "wish": [
        "if only the {subj} had left the village at dawn",
        "I wish the bridge had been built sooner",
        "if only the harvest had been better this year",
        "I wish the {subj} had completed the assignment",
        "if only the market would open earlier in the morning",
    ],
    "warning": [
        "beware the {subj} may leave the village at dawn",
        "the bridge could collapse if not repaired soon",
        "the harvest may fail if the drought continues",
        "the {subj} risks failing to complete the assignment",
        "the market may close permanently if sales decline",
    ],
    "promise": [
        "the {subj} promised to leave the village at dawn",
        "they swore the bridge would be finished by spring",
        "the {subj} vowed the harvest would be shared equally",
        "the {subj} guaranteed the assignment would be done",
        "the {subj} pledged to open the market every morning",
    ],
    "threat": [
        "the {subj} will be forced to leave the village",
        "the bridge will be destroyed if demands are not met",
        "the harvest will be confiscated by the authorities",
        "the {subj} will face consequences for the late assignment",
        "the market will be shut down if violations continue",
    ],
}

SLOT31_EXTRA_TEMPLATES = {
    "statement": [
        "the {subj} arrived before the others",
        "the door was closed when they got there",
        "the river runs through the center of the valley",
        "the fire burned all through the night",
        "the {subj} chose the longer path",
    ],
    "question": [
        "did the {subj} arrive before the others",
        "was the door closed when they got there",
        "does the river run through the center of the valley",
        "did the fire burn all through the night",
        "did the {subj} choose the longer path",
    ],
    "command": [
        "arrive before the others {subj}",
        "close the door before they get there",
        "follow the river through the valley",
        "keep the fire burning through the night",
        "choose the longer path and do not look back",
    ],
    "wish": [
        "if only the {subj} had arrived before the others",
        "I wish the door had been closed in time",
        "if only the river ran closer to the village",
        "I wish the fire had not burned so long",
        "if only the {subj} had chosen a different path",
    ],
    "warning": [
        "watch out the {subj} may arrive before you",
        "be careful the door might slam shut",
        "the river is dangerously high after the rain",
        "the fire could spread if not contained",
        "the {subj} might take the wrong path by mistake",
    ],
    "promise": [
        "the {subj} promised to arrive on time",
        "they assured everyone the door would remain closed",
        "the {subj} vowed to protect the river",
        "the {subj} swore to tend the fire all night",
        "the {subj} pledged to follow the right path",
    ],
    "threat": [
        "the {subj} will not be allowed to arrive late again",
        "the door will be sealed permanently",
        "the river will be diverted away from the town",
        "the fire will consume everything if unleashed",
        "the {subj} will be punished for straying from the path",
    ],
}


def generate_slot31() -> List[Dict]:
    """Speech Act & Intent pairs."""
    pairs = []
    subjects = PEOPLE[:25]
    acts = list(SLOT31_ACTS.keys())

    for subj in subjects:
        for base_act in acts:
            for var_act in acts:
                if var_act == base_act:
                    continue
                for i in range(len(SLOT31_ACTS[base_act])):
                    pairs.append({
                        "slot": 31,
                        "base": SLOT31_ACTS[base_act][i].format(subj=subj),
                        "variant": SLOT31_ACTS[var_act][i].format(subj=subj),
                        "concept_value": var_act,
                    })
                for i in range(len(SLOT31_EXTRA_TEMPLATES[base_act])):
                    pairs.append({
                        "slot": 31,
                        "base": SLOT31_EXTRA_TEMPLATES[base_act][i].format(subj=subj),
                        "variant": SLOT31_EXTRA_TEMPLATES[var_act][i].format(subj=subj),
                        "concept_value": var_act,
                    })
    return pairs


# ============================================================================
# MAIN GENERATOR
# ============================================================================

GENERATORS = {
    0: generate_slot0,
    1: generate_slot1,
    2: generate_slot2,
    3: generate_slot3,
    4: generate_slot4,
    5: generate_slot5,
    6: generate_slot6,
    7: generate_slot7,
    8: generate_slot8,
    9: generate_slot9,
    10: generate_slot10,
    11: generate_slot11,
    12: generate_slot12,
    13: generate_slot13,
    14: generate_slot14,
    15: generate_slot15,
    16: generate_slot16,
    17: generate_slot17,
    18: generate_slot18,
    19: generate_slot19,
    20: generate_slot20,
    21: generate_slot21,
    22: generate_slot22,
    23: generate_slot23,
    24: generate_slot24,
    25: generate_slot25,
    26: generate_slot26,
    27: generate_slot27,
    28: generate_slot28,
    29: generate_slot29,
    30: generate_slot30,
    31: generate_slot31,
}


def main():
    random.seed(42)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "concept_axes")
    os.makedirs(output_dir, exist_ok=True)

    all_pairs = []
    total_count = 0
    stats = {}

    print("=" * 70)
    print("CONCEPT AXES DATASET GENERATOR")
    print("=" * 70)
    print()

    for slot_id in range(32):
        slot_name = SLOT_NAMES[slot_id]
        print(f"Generating slot {slot_id:2d}: {slot_name}...", end=" ", flush=True)

        generator = GENERATORS[slot_id]
        pairs = generator()
        random.shuffle(pairs)

        # Write per-slot file
        slot_file = os.path.join(output_dir, f"slot_{slot_id:02d}_{slot_name}.jsonl")
        with open(slot_file, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        count = len(pairs)
        stats[slot_id] = {"name": slot_name, "count": count}
        all_pairs.extend(pairs)
        total_count += count
        print(f"{count:>8,} pairs  ->  {slot_file}")

    # Write combined file
    random.shuffle(all_pairs)
    combined_file = os.path.join(output_dir, "all_axes.jsonl")
    print(f"\nWriting combined file: {combined_file}")
    with open(combined_file, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Print summary
    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print()
    print(f"{'Slot':>6}  {'Name':<30}  {'Pairs':>10}")
    print("-" * 52)
    for slot_id in range(32):
        s = stats[slot_id]
        print(f"{slot_id:>6}  {s['name']:<30}  {s['count']:>10,}")
    print("-" * 52)
    print(f"{'TOTAL':>6}  {'':30}  {total_count:>10,}")
    print()
    print(f"Output directory: {output_dir}")
    print(f"Combined file:    {combined_file}")
    print(f"Total pairs:      {total_count:,}")
    print()


if __name__ == "__main__":
    main()
