#!/usr/bin/env python3
"""
Supplemental concept axis data generator.

Fixes BAD slots (10, 13, 22, 23, 29) with better isolation patterns
and boosts WEAK low-count slots (3, 9, 17, 26, 30, 31).

Outputs per-slot JSONL files that REPLACE the originals, then rebuilds all_axes.jsonl.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict

random.seed(42)

OUT_DIR = Path("data/concept_axes")

# Shared word lists (same as main generator)
PEOPLE = [
    "man", "woman", "boy", "girl", "child", "soldier", "farmer", "baker",
    "merchant", "knight", "queen", "prince", "princess", "priest", "monk",
    "scholar", "artist", "musician", "blacksmith", "carpenter", "tailor",
    "fisherman", "shepherd", "hunter", "sailor", "cook", "gardener",
    "teacher", "doctor", "nurse", "judge", "mayor", "captain", "servant",
    "thief", "beggar", "stranger", "traveler", "pilgrim", "hermit",
    "old man", "young woman", "tall boy", "little girl", "wise elder",
    "brave warrior", "kind healer", "tired worker", "shy student", "proud chief",
    "scientist", "professor", "librarian", "architect", "inventor",
    "philosopher", "poet", "painter", "sculptor", "weaver",
]

ANIMALS = [
    "cat", "dog", "horse", "bird", "fish", "wolf", "bear", "fox",
    "deer", "rabbit", "eagle", "hawk", "owl", "crow", "snake",
    "lion", "tiger", "elephant", "monkey", "goat", "sheep", "cow",
    "duck", "chicken", "frog", "turtle", "whale", "dolphin", "mouse", "rat",
]

OBJECTS = [
    "stone", "rope", "box", "barrel", "bucket", "basket", "sack",
    "blanket", "chair", "table", "ladder", "wheel", "log", "plank",
    "sword", "shield", "book", "lantern", "candle", "key",
    "hammer", "axe", "knife", "pot", "cup", "plate", "bowl",
    "ball", "stick", "bottle", "bag", "crate", "trunk", "pillow",
]

LOCATIONS = [
    "the garden", "the kitchen", "the field", "the forest",
    "the market", "the courtyard", "the stable", "the workshop",
    "the river", "the hilltop", "the road", "the bridge",
    "the cellar", "the attic", "the hall", "the tower",
]


# ============================================================================
# SLOT 10: Action Type — REDESIGNED
# Only the verb changes, everything else stays the same
# ============================================================================

SLOT10_VERB_GROUPS = {
    "motion": [
        "carried", "dragged", "pushed", "pulled", "lifted",
        "rolled", "tossed", "slid", "hauled", "lowered",
    ],
    "perception": [
        "examined", "inspected", "studied", "watched", "observed",
        "noticed", "spotted", "eyed", "surveyed", "scrutinized",
    ],
    "manipulation": [
        "grabbed", "held", "squeezed", "gripped", "clasped",
        "touched", "handled", "stroked", "poked", "tapped",
    ],
    "destruction": [
        "smashed", "broke", "shattered", "crushed", "cracked",
        "snapped", "split", "ripped", "tore", "wrecked",
    ],
    "creation": [
        "built", "assembled", "constructed", "crafted", "fashioned",
        "made", "formed", "shaped", "molded", "designed",
    ],
    "transfer": [
        "gave", "handed", "passed", "delivered", "offered",
        "presented", "returned", "sent", "tossed", "threw",
    ],
}

SLOT10_TEMPLATES_V2 = [
    "the {subj} {verb} the {obj}",
    "the {subj} {verb} the {obj} carefully",
    "the {subj} {verb} the {obj} without hesitation",
    "the {subj} {verb} the {obj} in the {loc}",
    "the {subj} quietly {verb} the {obj}",
    "the {subj} {verb} the {obj} and stepped back",
    "the {subj} {verb} the {obj} one more time",
    "suddenly the {subj} {verb} the {obj}",
    "the {subj} {verb} the {obj} before leaving",
    "the {subj} {verb} the {obj} with both hands",
    "the {subj} {verb} the old {obj}",
    "the {subj} {verb} the heavy {obj}",
    "the {subj} finally {verb} the {obj}",
    "the {subj} {verb} the {obj} as expected",
    "the {subj} {verb} the {obj} for the last time",
]


def generate_slot10_v2() -> List[Dict]:
    """Action Type — verb-only swap, shared object/subject/template."""
    pairs = []
    action_types = list(SLOT10_VERB_GROUPS.keys())
    subjects = PEOPLE[:30]
    objects = OBJECTS[:20]

    for template in SLOT10_TEMPLATES_V2:
        for subj in subjects:
            for obj in objects:
                for base_type in action_types:
                    base_verbs = SLOT10_VERB_GROUPS[base_type]
                    for var_type in action_types:
                        if var_type == base_type:
                            continue
                        var_verbs = SLOT10_VERB_GROUPS[var_type]
                        bv = random.choice(base_verbs)
                        vv = random.choice(var_verbs)
                        loc = random.choice(LOCATIONS)
                        pairs.append({
                            "slot": 10,
                            "base": template.format(subj=subj, verb=bv, obj=obj, loc=loc),
                            "variant": template.format(subj=subj, verb=vv, obj=obj, loc=loc),
                            "concept_value": var_type,
                        })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 13: Direction & Path — FIXED
# Pure directional words only, no landmark bleed
# ============================================================================

SLOT13_PURE_DIRECTIONS = [
    "upward", "downward", "forward", "backward",
    "left", "right", "sideways", "diagonally",
    "north", "south", "east", "west",
    "uphill", "downhill", "inward", "outward",
    "away", "closer", "aside", "overhead",
]

SLOT13_TEMPLATES_V2 = [
    "the {subj} moved {dir}",
    "the {subj} ran {dir} quickly",
    "the {subj} walked {dir} in silence",
    "she pushed the cart {dir}",
    "he threw the ball {dir}",
    "the {subj} glanced {dir}",
    "the {subj} pointed {dir}",
    "she leaned {dir} slightly",
    "the {subj} stepped {dir}",
    "he jumped {dir} suddenly",
    "the {subj} rolled {dir}",
    "she swam {dir} against the current",
    "the {subj} crawled {dir} slowly",
    "he reached {dir}",
    "the {subj} pulled the rope {dir}",
    "she shifted {dir} nervously",
    "the {subj} tumbled {dir}",
    "he slid {dir} on the ice",
    "the {subj} floated {dir} gently",
    "she turned {dir} and paused",
    "the {subj} flew {dir} at great speed",
    "he stumbled {dir} in the dark",
    "the {subj} drifted {dir}",
    "she lunged {dir} with force",
    "the {subj} veered {dir} sharply",
]


def generate_slot13_v2() -> List[Dict]:
    """Direction — pure directional words only."""
    pairs = []
    subjects = PEOPLE[:25] + ANIMALS[:10]

    for template in SLOT13_TEMPLATES_V2:
        for subj in subjects:
            for base_d in SLOT13_PURE_DIRECTIONS:
                for var_d in SLOT13_PURE_DIRECTIONS:
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
# SLOT 22: Core Sentiment — REDESIGNED
# Template-based with shared structure, only sentiment word swaps
# ============================================================================

SLOT22_POS_ADJS = [
    "wonderful", "fantastic", "beautiful", "amazing", "delightful",
    "excellent", "great", "lovely", "perfect", "outstanding",
    "pleasant", "joyful", "brilliant", "glorious", "magnificent",
    "superb", "splendid", "charming", "marvelous", "terrific",
]

SLOT22_NEG_ADJS = [
    "terrible", "awful", "horrible", "dreadful", "miserable",
    "disgusting", "appalling", "ghastly", "wretched", "abysmal",
    "painful", "dismal", "grim", "bleak", "devastating",
    "horrendous", "atrocious", "vile", "depressing", "tragic",
]

SLOT22_NEU_ADJS = [
    "ordinary", "unremarkable", "average", "typical", "expected",
    "standard", "routine", "normal", "usual", "mundane",
    "predictable", "conventional", "moderate", "acceptable", "adequate",
    "common", "regular", "uneventful", "straightforward", "plain",
]

SLOT22_TEMPLATES_V2 = [
    "the {subj} thought the day was {adj}",
    "the {subj} described the event as {adj}",
    "the {subj} found the meal {adj}",
    "the {subj} said the weather was {adj}",
    "the {subj} considered the outcome {adj}",
    "the {subj} felt the experience was {adj}",
    "the {subj} declared the performance {adj}",
    "the {subj} called the view {adj}",
    "the {subj} found the situation {adj}",
    "the {subj} thought the gift was {adj}",
    "the {subj} said the news was {adj}",
    "the {subj} called the journey {adj}",
    "the {subj} felt the morning was {adj}",
    "the {subj} thought the story was {adj}",
    "the {subj} described the sunset as {adj}",
    "the {subj} found the conversation {adj}",
    "the {subj} considered the plan {adj}",
    "the {subj} said the song was {adj}",
    "the {subj} called the idea {adj}",
    "the {subj} thought the work was {adj}",
    "the {subj} declared the result {adj}",
    "the {subj} found the ending {adj}",
    "the {subj} said the film was {adj}",
    "the {subj} described the party as {adj}",
    "the {subj} felt the silence was {adj}",
]

SLOT22_SENTIMENTS = {
    "positive": SLOT22_POS_ADJS,
    "negative": SLOT22_NEG_ADJS,
    "neutral": SLOT22_NEU_ADJS,
}


def generate_slot22_v2() -> List[Dict]:
    """Core Sentiment — adjective-only swap in shared template."""
    pairs = []
    subjects = PEOPLE[:40]
    sentiments = list(SLOT22_SENTIMENTS.keys())

    for template in SLOT22_TEMPLATES_V2:
        for subj in subjects:
            for base_s in sentiments:
                base_adjs = SLOT22_SENTIMENTS[base_s]
                for var_s in sentiments:
                    if var_s == base_s:
                        continue
                    var_adjs = SLOT22_SENTIMENTS[var_s]
                    # Pick matched pairs to maximize variety
                    for ba in base_adjs:
                        va = random.choice(var_adjs)
                        pairs.append({
                            "slot": 22,
                            "base": template.format(subj=subj, adj=ba),
                            "variant": template.format(subj=subj, adj=va),
                            "concept_value": var_s,
                        })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 23: Specific Emotion — REDESIGNED
# Single-word emotion adjective swap, minimal surface change
# ============================================================================

SLOT23_EMOTION_ADJS = {
    "happy": ["happy", "cheerful", "joyful", "elated", "content", "pleased", "ecstatic", "thrilled"],
    "sad": ["sad", "sorrowful", "gloomy", "heartbroken", "miserable", "melancholy", "downcast", "dejected"],
    "angry": ["angry", "furious", "livid", "enraged", "irate", "outraged", "wrathful", "seething"],
    "afraid": ["afraid", "scared", "terrified", "frightened", "anxious", "panicked", "alarmed", "petrified"],
    "surprised": ["surprised", "astonished", "stunned", "amazed", "startled", "bewildered", "dumbfounded", "flabbergasted"],
    "disgusted": ["disgusted", "repulsed", "revolted", "sickened", "appalled", "nauseated", "offended", "horrified"],
    "proud": ["proud", "confident", "triumphant", "dignified", "accomplished", "self-assured", "victorious", "honored"],
    "jealous": ["jealous", "envious", "resentful", "covetous", "bitter", "grudging", "possessive", "suspicious"],
    "grateful": ["grateful", "thankful", "appreciative", "indebted", "obliged", "touched", "moved", "humbled"],
    "lonely": ["lonely", "isolated", "abandoned", "forsaken", "desolate", "homesick", "solitary", "estranged"],
}

SLOT23_TEMPLATES_V2 = [
    "the {subj} felt {emotion}",
    "the {subj} was {emotion}",
    "the {subj} looked {emotion}",
    "the {subj} seemed {emotion}",
    "the {subj} became {emotion}",
    "the {subj} appeared {emotion}",
    "the {subj} was deeply {emotion}",
    "the {subj} was very {emotion}",
    "the {subj} felt {emotion} about the news",
    "the {subj} was {emotion} after the meeting",
    "the {subj} looked {emotion} when they heard",
    "the {subj} felt {emotion} all morning",
    "the {subj} seemed {emotion} during dinner",
    "the {subj} became {emotion} upon arrival",
    "the {subj} felt {emotion} about the decision",
    "the {subj} was {emotion} about the outcome",
    "the {subj} looked {emotion} in the photograph",
    "the {subj} felt {emotion} for the first time",
    "the {subj} was {emotion} without reason",
    "the {subj} seemed {emotion} at the ceremony",
    "the {subj} was clearly {emotion}",
    "the {subj} was visibly {emotion}",
    "the {subj} grew {emotion} over time",
    "the {subj} remained {emotion} throughout",
    "the {subj} felt increasingly {emotion}",
]


def generate_slot23_v2() -> List[Dict]:
    """Specific Emotion — adjective-only swap."""
    pairs = []
    subjects = PEOPLE[:35]
    emotions = list(SLOT23_EMOTION_ADJS.keys())

    for template in SLOT23_TEMPLATES_V2:
        for subj in subjects:
            for base_e in emotions:
                base_adjs = SLOT23_EMOTION_ADJS[base_e]
                for var_e in emotions:
                    if var_e == base_e:
                        continue
                    var_adjs = SLOT23_EMOTION_ADJS[var_e]
                    ba = random.choice(base_adjs)
                    va = random.choice(var_adjs)
                    pairs.append({
                        "slot": 23,
                        "base": template.format(subj=subj, emotion=ba),
                        "variant": template.format(subj=subj, emotion=va),
                        "concept_value": var_e,
                    })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 29: Causation & Condition — REDESIGNED
# Structurally distinct causal relationships, not just connector swaps
# ============================================================================

# Connector-swap approach: same two clauses, only the connector changes
# Fewer categories but more semantically distinct
SLOT29_CONNECTORS_V2 = {
    "cause": "because",
    "concession": "despite",
    "condition": "if",
    "prevention": "unless",
    "temporal": "after",
    "contrast": "but",
}

SLOT29_CLAUSES_A_V2 = [
    "the {subj} left early",
    "the {subj} stayed behind",
    "the {subj} changed the plan",
    "the {subj} spoke up",
    "the {subj} joined the group",
    "the {subj} locked the door",
    "the {subj} sold the horse",
    "the {subj} hired a guide",
    "the {subj} started running",
    "the {subj} packed the bags",
    "the {subj} lit the fire",
    "the {subj} closed the shop",
    "the {subj} moved away",
    "the {subj} asked for help",
    "the {subj} returned home",
    "the {subj} cooked dinner",
    "the {subj} read the letter",
    "the {subj} fixed the roof",
    "the {subj} opened the gate",
    "the {subj} told the truth",
]

SLOT29_CLAUSES_B_V2 = [
    "the road was dangerous",
    "the weather improved",
    "the supplies ran low",
    "the deadline was near",
    "the rain finally stopped",
    "the king gave the order",
    "the harvest was poor",
    "the enemy retreated",
    "the message arrived",
    "the river flooded",
    "the price went up",
    "the bridge was broken",
    "the food ran out",
    "the fire went out",
    "the storm was coming",
    "the noise grew louder",
    "the sun went down",
    "the crowd grew restless",
    "the door was locked",
    "the path was blocked",
]


def generate_slot29_v2() -> List[Dict]:
    """Causation — connector-only swap between same two clauses."""
    pairs = []
    subjects = PEOPLE[:25]
    connectors = list(SLOT29_CONNECTORS_V2.items())

    for subj in subjects:
        for clause_a in SLOT29_CLAUSES_A_V2:
            for clause_b in SLOT29_CLAUSES_B_V2:
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
# SLOT 3: Age/Life Stage — BOOST (was 13,660)
# ============================================================================

SLOT3_AGES = [
    "baby", "infant", "toddler", "young", "teenage", "adolescent",
    "adult", "middle-aged", "elderly", "ancient", "newborn", "juvenile",
    "youthful", "mature", "aging", "old", "little", "grown",
]

SLOT3_TEMPLATES_V2 = [
    "the {age} {subj} sat in the {loc}",
    "the {age} {subj} walked through the {loc}",
    "the {age} {subj} looked at the sky",
    "the {age} {subj} spoke to the crowd",
    "the {age} {subj} waited by the door",
    "the {age} {subj} ate the bread",
    "the {age} {subj} rested under the tree",
    "the {age} {subj} held the {obj}",
    "the {age} {subj} stood in the rain",
    "the {age} {subj} laughed at the joke",
    "the {age} {subj} carried the {obj} home",
    "the {age} {subj} watched the sunset",
    "the {age} {subj} climbed the hill",
    "the {age} {subj} read the old book",
    "the {age} {subj} sang a quiet song",
    "a {age} {subj} entered the room",
    "a {age} {subj} found the {obj}",
    "a {age} {subj} called for help",
    "the {age} {subj} slept by the fire",
    "the {age} {subj} worked in the field",
]


def generate_slot3_v2() -> List[Dict]:
    """Age — adjective-only swap."""
    pairs = []
    subjects = PEOPLE[:30]
    objects = OBJECTS[:15]

    for template in SLOT3_TEMPLATES_V2:
        for subj in subjects:
            for base_age in SLOT3_AGES:
                for var_age in SLOT3_AGES:
                    if var_age == base_age:
                        continue
                    obj = random.choice(objects)
                    loc = random.choice(["garden", "kitchen", "forest", "market",
                                        "courtyard", "workshop", "field", "hall"])
                    pairs.append({
                        "slot": 3,
                        "base": template.format(age=base_age, subj=subj, obj=obj, loc=loc),
                        "variant": template.format(age=var_age, subj=subj, obj=obj, loc=loc),
                        "concept_value": var_age,
                    })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 9: Temperature/Weather — BOOST (was 5,850)
# ============================================================================

SLOT9_TEMPS = [
    "freezing", "cold", "chilly", "cool", "mild", "warm",
    "hot", "scorching", "boiling", "icy", "frosty", "balmy",
    "sweltering", "tepid", "crisp", "biting", "searing", "pleasant",
]

SLOT9_TEMPLATES_V2 = [
    "the {subj} stepped into the {temp} air",
    "the {temp} wind blew through the village",
    "it was a {temp} morning in the valley",
    "the {subj} shivered in the {temp} weather",
    "the {temp} sun beat down on the road",
    "the {subj} felt the {temp} breeze",
    "the {temp} water splashed against the rocks",
    "the {subj} walked through the {temp} rain",
    "it was unusually {temp} that day",
    "the {temp} night settled over the town",
    "the {subj} endured the {temp} conditions",
    "the {temp} fog rolled in from the sea",
    "the ground was {temp} beneath their feet",
    "the {subj} drank the {temp} tea",
    "the room felt {temp} and still",
    "the {temp} season had arrived at last",
    "the {subj} sat by the {temp} fire",
    "the {temp} stream flowed down the mountain",
    "the {subj} breathed the {temp} evening air",
    "everything felt {temp} and quiet",
]


def generate_slot9_v2() -> List[Dict]:
    """Temperature — adjective-only swap."""
    pairs = []
    subjects = PEOPLE[:30]

    for template in SLOT9_TEMPLATES_V2:
        for subj in subjects:
            for base_t in SLOT9_TEMPS:
                for var_t in SLOT9_TEMPS:
                    if var_t == base_t:
                        continue
                    pairs.append({
                        "slot": 9,
                        "base": template.format(subj=subj, temp=base_t),
                        "variant": template.format(subj=subj, temp=var_t),
                        "concept_value": var_t,
                    })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 17: Tense/Aspect — BOOST (was 7,500)
# ============================================================================

SLOT17_TENSE_FORMS = {
    "past": {
        "walk": "walked", "run": "ran", "eat": "ate", "see": "saw",
        "give": "gave", "take": "took", "make": "made", "say": "said",
        "go": "went", "come": "came", "find": "found", "know": "knew",
        "think": "thought", "tell": "told", "write": "wrote", "read": "read",
        "sing": "sang", "drink": "drank", "sleep": "slept", "sit": "sat",
    },
    "present": {
        "walk": "walks", "run": "runs", "eat": "eats", "see": "sees",
        "give": "gives", "take": "takes", "make": "makes", "say": "says",
        "go": "goes", "come": "comes", "find": "finds", "know": "knows",
        "think": "thinks", "tell": "tells", "write": "writes", "read": "reads",
        "sing": "sings", "drink": "drinks", "sleep": "sleeps", "sit": "sits",
    },
    "future": {
        "walk": "will walk", "run": "will run", "eat": "will eat", "see": "will see",
        "give": "will give", "take": "will take", "make": "will make", "say": "will say",
        "go": "will go", "come": "will come", "find": "will find", "know": "will know",
        "think": "will think", "tell": "will tell", "write": "will write", "read": "will read",
        "sing": "will sing", "drink": "will drink", "sleep": "will sleep", "sit": "will sit",
    },
    "progressive": {
        "walk": "is walking", "run": "is running", "eat": "is eating", "see": "is seeing",
        "give": "is giving", "take": "is taking", "make": "is making", "say": "is saying",
        "go": "is going", "come": "is coming", "find": "is finding", "know": "is knowing",
        "think": "is thinking", "tell": "is telling", "write": "is writing", "read": "is reading",
        "sing": "is singing", "drink": "is drinking", "sleep": "is sleeping", "sit": "is sitting",
    },
}

SLOT17_TEMPLATES_V2 = [
    "the {subj} {verb} every day",
    "the {subj} {verb} in the morning",
    "the {subj} {verb} by the river",
    "the {subj} {verb} at the market",
    "the {subj} {verb} after dinner",
    "the {subj} {verb} alone in the room",
    "the {subj} {verb} with the others",
    "the {subj} {verb} before sunrise",
    "the {subj} {verb} in the old house",
    "the {subj} {verb} near the bridge",
    "the {subj} {verb} until dusk",
    "the {subj} {verb} from time to time",
    "the {subj} {verb} when the bell rings",
    "the {subj} {verb} on the hilltop",
    "the {subj} {verb} without anyone noticing",
]


def generate_slot17_v2() -> List[Dict]:
    """Tense — verb form swap only."""
    pairs = []
    subjects = PEOPLE[:30]
    tenses = list(SLOT17_TENSE_FORMS.keys())
    verbs = list(SLOT17_TENSE_FORMS["past"].keys())

    for template in SLOT17_TEMPLATES_V2:
        for subj in subjects:
            for verb_key in verbs:
                for base_tense in tenses:
                    for var_tense in tenses:
                        if var_tense == base_tense:
                            continue
                        bv = SLOT17_TENSE_FORMS[base_tense][verb_key]
                        vv = SLOT17_TENSE_FORMS[var_tense][verb_key]
                        pairs.append({
                            "slot": 17,
                            "base": template.format(subj=subj, verb=bv),
                            "variant": template.format(subj=subj, verb=vv),
                            "concept_value": var_tense,
                        })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 26: Difficulty/Importance — BOOST (was 5,460)
# ============================================================================

SLOT26_DIFFICULTY = [
    "easy", "simple", "effortless", "straightforward", "trivial",
    "hard", "difficult", "challenging", "demanding", "grueling",
    "impossible", "daunting", "overwhelming", "exhausting", "arduous",
    "manageable", "moderate", "reasonable", "doable", "feasible",
]

SLOT26_IMPORTANCE = [
    "important", "crucial", "vital", "essential", "critical",
    "minor", "trivial", "insignificant", "negligible", "irrelevant",
    "urgent", "pressing", "paramount", "necessary", "fundamental",
    "optional", "secondary", "peripheral", "supplementary", "marginal",
]

SLOT26_TEMPLATES_DIFF = [
    "the {subj} found the task {adj}",
    "the {subj} said the work was {adj}",
    "the {subj} thought the problem was {adj}",
    "the {subj} considered the challenge {adj}",
    "the {subj} described the test as {adj}",
    "the {subj} felt the assignment was {adj}",
    "the {subj} called the puzzle {adj}",
    "the {subj} judged the exam {adj}",
    "the {subj} rated the exercise {adj}",
    "the {subj} declared the mission {adj}",
    "the {subj} found the climb {adj}",
    "the {subj} said the recipe was {adj}",
    "the {subj} called the repair {adj}",
    "the {subj} thought the journey was {adj}",
    "the {subj} considered the negotiation {adj}",
]

SLOT26_TEMPLATES_IMP = [
    "the {subj} thought the matter was {adj}",
    "the {subj} said the issue was {adj}",
    "the {subj} considered the topic {adj}",
    "the {subj} declared the decision {adj}",
    "the {subj} felt the question was {adj}",
    "the {subj} called the detail {adj}",
    "the {subj} judged the point {adj}",
    "the {subj} rated the concern {adj}",
    "the {subj} thought the meeting was {adj}",
    "the {subj} described the event as {adj}",
]


def generate_slot26_v2() -> List[Dict]:
    """Difficulty/Importance — adjective-only swap."""
    pairs = []
    subjects = PEOPLE[:30]

    # Difficulty pairs
    for template in SLOT26_TEMPLATES_DIFF:
        for subj in subjects:
            for base_adj in SLOT26_DIFFICULTY:
                for var_adj in SLOT26_DIFFICULTY:
                    if var_adj == base_adj:
                        continue
                    pairs.append({
                        "slot": 26,
                        "base": template.format(subj=subj, adj=base_adj),
                        "variant": template.format(subj=subj, adj=var_adj),
                        "concept_value": var_adj,
                    })

    # Importance pairs
    for template in SLOT26_TEMPLATES_IMP:
        for subj in subjects:
            for base_adj in SLOT26_IMPORTANCE:
                for var_adj in SLOT26_IMPORTANCE:
                    if var_adj == base_adj:
                        continue
                    pairs.append({
                        "slot": 26,
                        "base": template.format(subj=subj, adj=base_adj),
                        "variant": template.format(subj=subj, adj=var_adj),
                        "concept_value": var_adj,
                    })

    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 30: Formality/Register — BOOST (was 7,500)
# Keep the existing approach (whole-sentence register) but add
# adjective-based formality within shared templates too
# ============================================================================

# Formality uses verb-choice and framing to signal register
# Same subject + action, different register of expression

SLOT30_ACTION_SETS = {
    "leave": {
        "slang": ["{subj} bounced", "{subj} dipped out", "{subj} peaced out", "{subj} split"],
        "casual": ["{subj} left", "{subj} headed out", "{subj} took off", "{subj} went home"],
        "formal": ["{subj} departed", "{subj} excused themselves", "{subj} withdrew", "{subj} took their leave"],
        "academic": ["{subj} vacated the premises", "{subj} concluded their visit", "{subj} terminated the engagement", "{subj} ceased attendance"],
    },
    "agree": {
        "slang": ["{subj} was like yeah totally", "{subj} said bet", "{subj} was down with it", "{subj} said for sure"],
        "casual": ["{subj} agreed", "{subj} said yes", "{subj} went along with it", "{subj} was fine with it"],
        "formal": ["{subj} expressed agreement", "{subj} concurred", "{subj} endorsed the proposal", "{subj} gave their assent"],
        "academic": ["{subj} affirmed the proposition", "{subj} corroborated the assertion", "{subj} validated the hypothesis", "{subj} substantiated the claim"],
    },
    "complain": {
        "slang": ["{subj} was salty about it", "{subj} said that's wack", "{subj} was mad heated", "{subj} threw shade"],
        "casual": ["{subj} complained about it", "{subj} wasn't happy about it", "{subj} grumbled a bit", "{subj} said it wasn't fair"],
        "formal": ["{subj} voiced their displeasure", "{subj} lodged a complaint", "{subj} expressed dissatisfaction", "{subj} raised an objection"],
        "academic": ["{subj} articulated their grievance", "{subj} documented the deficiency", "{subj} noted the inadequacy", "{subj} identified the shortcoming"],
    },
    "succeed": {
        "slang": ["{subj} crushed it", "{subj} nailed it", "{subj} absolutely killed it", "{subj} was goated"],
        "casual": ["{subj} did great", "{subj} succeeded", "{subj} pulled it off", "{subj} got it done"],
        "formal": ["{subj} achieved the objective", "{subj} accomplished the goal", "{subj} attained success", "{subj} fulfilled the mandate"],
        "academic": ["{subj} demonstrated exemplary performance", "{subj} yielded optimal outcomes", "{subj} exceeded established benchmarks", "{subj} realized the anticipated results"],
    },
    "fail": {
        "slang": ["{subj} totally bombed", "{subj} choked hard", "{subj} blew it big time", "{subj} face planted"],
        "casual": ["{subj} failed", "{subj} messed up", "{subj} didn't make it", "{subj} fell short"],
        "formal": ["{subj} did not succeed", "{subj} fell short of expectations", "{subj} failed to meet the standard", "{subj} was unable to achieve the goal"],
        "academic": ["{subj} failed to meet criteria", "{subj} produced suboptimal results", "{subj} did not attain the threshold", "{subj} underperformed relative to projections"],
    },
    "think": {
        "slang": ["{subj} figured it was whatever", "{subj} reckoned it was chill", "{subj} was like hmm okay", "{subj} thought it was lowkey true"],
        "casual": ["{subj} thought about it", "{subj} figured it out", "{subj} had an idea", "{subj} guessed it was true"],
        "formal": ["{subj} considered the matter", "{subj} reflected upon it", "{subj} deliberated carefully", "{subj} contemplated the situation"],
        "academic": ["{subj} analyzed the proposition", "{subj} evaluated the evidence", "{subj} synthesized the findings", "{subj} formulated a hypothesis"],
    },
    "help": {
        "slang": ["{subj} had their back", "{subj} helped out real quick", "{subj} came through clutch", "{subj} hooked them up"],
        "casual": ["{subj} helped out", "{subj} gave a hand", "{subj} pitched in", "{subj} lent some help"],
        "formal": ["{subj} provided assistance", "{subj} offered their support", "{subj} rendered aid", "{subj} extended their help"],
        "academic": ["{subj} facilitated the process", "{subj} contributed substantively", "{subj} administered the intervention", "{subj} implemented supportive measures"],
    },
    "explain": {
        "slang": ["{subj} broke it down for them", "{subj} spilled the tea", "{subj} laid it all out", "{subj} gave them the lowdown"],
        "casual": ["{subj} explained it", "{subj} told them about it", "{subj} went over the details", "{subj} made it clear"],
        "formal": ["{subj} provided an explanation", "{subj} clarified the situation", "{subj} elucidated the matter", "{subj} presented the information"],
        "academic": ["{subj} expounded upon the topic", "{subj} delineated the parameters", "{subj} explicated the framework", "{subj} systematically described the methodology"],
    },
    "refuse": {
        "slang": ["{subj} said nah fam", "{subj} was like nope", "{subj} straight up said no way", "{subj} wasn't having it"],
        "casual": ["{subj} said no thanks", "{subj} turned it down", "{subj} passed on it", "{subj} decided against it"],
        "formal": ["{subj} respectfully declined", "{subj} politely refused", "{subj} expressed their reluctance", "{subj} courteously rejected the offer"],
        "academic": ["{subj} declined the proposition", "{subj} rejected the overture", "{subj} demurred on the matter", "{subj} withheld their consent"],
    },
    "apologize": {
        "slang": ["{subj} was like my bad", "{subj} said sorry yo", "{subj} felt real bad about it", "{subj} was all like oops"],
        "casual": ["{subj} said sorry", "{subj} apologized", "{subj} felt bad about it", "{subj} said they were sorry"],
        "formal": ["{subj} offered an apology", "{subj} expressed their regret", "{subj} conveyed their remorse", "{subj} asked for forgiveness"],
        "academic": ["{subj} acknowledged the error", "{subj} took responsibility for the oversight", "{subj} issued a formal retraction", "{subj} conceded the methodological lapse"],
    },
    "laugh": {
        "slang": ["{subj} was dying laughing", "{subj} was weak from laughing", "{subj} lost it completely", "{subj} was cracking up"],
        "casual": ["{subj} laughed out loud", "{subj} had a good laugh", "{subj} couldn't stop laughing", "{subj} chuckled a bit"],
        "formal": ["{subj} shared a polite laugh", "{subj} was visibly amused", "{subj} expressed their amusement", "{subj} appreciated the humor"],
        "academic": ["{subj} exhibited a humorous response", "{subj} demonstrated amusement", "{subj} responded with levity", "{subj} acknowledged the comedic element"],
    },
    "worry": {
        "slang": ["{subj} was stressing hard", "{subj} was buggin out", "{subj} was lowkey freaking", "{subj} was mad anxious"],
        "casual": ["{subj} was worried", "{subj} felt anxious about it", "{subj} was nervous", "{subj} kept worrying"],
        "formal": ["{subj} expressed concern", "{subj} was apprehensive", "{subj} harbored reservations", "{subj} conveyed their unease"],
        "academic": ["{subj} demonstrated elevated anxiety", "{subj} exhibited signs of concern", "{subj} manifested apprehension", "{subj} reported heightened distress"],
    },
}

SLOT30_CONTEXTS = [
    "", " at the meeting", " in the morning", " after the announcement",
    " during the event", " when asked", " to the group", " before leaving",
    " at the ceremony", " without hesitation", " at the party", " after work",
    " in the office", " on the way home", " during lunch",
]


def generate_slot30_v2() -> List[Dict]:
    """Formality — same action, different register."""
    pairs = []
    subjects = ["the " + p for p in PEOPLE[:25]]
    registers = ["slang", "casual", "formal", "academic"]

    for action, reg_dict in SLOT30_ACTION_SETS.items():
        for subj in subjects:
            for ctx in SLOT30_CONTEXTS:
                for base_r in registers:
                    for var_r in registers:
                        if var_r == base_r:
                            continue
                        bt = random.choice(reg_dict[base_r])
                        vt = random.choice(reg_dict[var_r])
                        pairs.append({
                            "slot": 30,
                            "base": bt.format(subj=subj) + ctx,
                            "variant": vt.format(subj=subj) + ctx,
                            "concept_value": var_r,
                        })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# SLOT 31: Speech Act/Intent — BOOST (was 10,500)
# ============================================================================

SLOT31_ACTS = {
    "question": [
        "the {subj} asked where the {obj} was",
        "the {subj} wondered if the door was open",
        "the {subj} inquired about the weather",
        "did the {subj} find the answer",
        "the {subj} asked what happened next",
        "the {subj} questioned the decision",
        "the {subj} wanted to know the truth",
    ],
    "command": [
        "the {subj} ordered everyone to stop",
        "the {subj} told them to bring the {obj}",
        "the {subj} demanded silence immediately",
        "the {subj} instructed them to wait",
        "the {subj} commanded them to leave",
        "the {subj} directed them to the exit",
        "the {subj} insisted they finish the work",
    ],
    "statement": [
        "the {subj} said the {obj} was on the table",
        "the {subj} explained the situation clearly",
        "the {subj} announced the results to all",
        "the {subj} stated the facts plainly",
        "the {subj} declared the matter closed",
        "the {subj} reported what they had seen",
        "the {subj} described the event in detail",
    ],
    "request": [
        "the {subj} asked them to pass the {obj}",
        "the {subj} requested a moment of quiet",
        "the {subj} begged for another chance",
        "the {subj} pleaded for more time",
        "the {subj} appealed for help",
        "the {subj} implored them to listen",
        "the {subj} urged them to reconsider",
    ],
    "promise": [
        "the {subj} promised to return the {obj}",
        "the {subj} swore to protect the village",
        "the {subj} vowed to finish the task",
        "the {subj} pledged to help everyone",
        "the {subj} guaranteed the outcome",
        "the {subj} committed to the plan",
        "the {subj} assured them it would work",
    ],
    "warning": [
        "the {subj} warned them about the {obj}",
        "the {subj} cautioned against the danger",
        "the {subj} alerted everyone to the risk",
        "the {subj} threatened consequences",
        "the {subj} signaled the approaching storm",
        "the {subj} advised them to be careful",
        "the {subj} notified them of the problem",
    ],
    "apology": [
        "the {subj} apologized for the mistake",
        "the {subj} expressed regret for the incident",
        "the {subj} said they were sorry about the {obj}",
        "the {subj} asked for forgiveness",
        "the {subj} admitted the error was theirs",
        "the {subj} took responsibility for the failure",
        "the {subj} sought to make amends",
    ],
    "greeting": [
        "the {subj} greeted everyone warmly",
        "the {subj} welcomed the visitors inside",
        "the {subj} said hello to the stranger",
        "the {subj} introduced themselves politely",
        "the {subj} waved and smiled at the group",
        "the {subj} nodded in acknowledgment",
        "the {subj} bowed before the audience",
    ],
}


def generate_slot31_v2() -> List[Dict]:
    """Speech Act — intent-type swap."""
    pairs = []
    subjects = PEOPLE[:35]
    acts = list(SLOT31_ACTS.keys())
    objects = OBJECTS[:20]

    for subj in subjects:
        for obj in objects:
            for base_act in acts:
                base_sents = SLOT31_ACTS[base_act]
                for var_act in acts:
                    if var_act == base_act:
                        continue
                    var_sents = SLOT31_ACTS[var_act]
                    # Generate multiple pairs per combo
                    for bt in base_sents:
                        vt = random.choice(var_sents)
                        pairs.append({
                            "slot": 31,
                            "base": bt.format(subj=subj, obj=obj),
                            "variant": vt.format(subj=subj, obj=obj),
                            "concept_value": var_act,
                        })
    random.shuffle(pairs)
    return pairs[:80000]


# ============================================================================
# Main
# ============================================================================

GENERATORS = {
    # BAD slots — redesigned
    10: ("action_type", generate_slot10_v2),
    13: ("direction_path", generate_slot13_v2),
    22: ("core_sentiment", generate_slot22_v2),
    23: ("specific_emotion", generate_slot23_v2),
    29: ("causation_condition", generate_slot29_v2),
    # WEAK low-count slots — boosted
    3: ("age_life_stage", generate_slot3_v2),
    9: ("temperature_weather", generate_slot9_v2),
    17: ("tense_aspect", generate_slot17_v2),
    26: ("difficulty_importance", generate_slot26_v2),
    30: ("formality_register", generate_slot30_v2),
    31: ("speech_act_intent", generate_slot31_v2),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    for slot_id in sorted(GENERATORS.keys()):
        name, gen_fn = GENERATORS[slot_id]
        print(f"Generating slot {slot_id:2d} ({name})...", end=" ", flush=True)
        pairs = gen_fn()
        count = len(pairs)
        total += count

        # Write per-slot file (replaces existing)
        fname = OUT_DIR / f"slot_{slot_id:02d}_{name}.jsonl"
        with open(fname, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"{count:,} pairs -> {fname}")

    # Rebuild all_axes.jsonl from ALL slot files
    print(f"\nRebuilding all_axes.jsonl...")
    all_pairs = []
    for fname in sorted(OUT_DIR.glob("slot_*.jsonl")):
        with open(fname) as f:
            for line in f:
                all_pairs.append(line)
    random.shuffle(all_pairs)

    combined = OUT_DIR / "all_axes.jsonl"
    with open(combined, "w") as f:
        for line in all_pairs:
            f.write(line)

    print(f"Total: {len(all_pairs):,} pairs in {combined}")
    print(f"  (generated {total:,} new pairs for {len(GENERATORS)} slots)")

    # Per-slot summary
    print("\nPer-slot counts:")
    slot_counts = {}
    for fname in sorted(OUT_DIR.glob("slot_*.jsonl")):
        with open(fname) as f:
            c = sum(1 for _ in f)
        slot_counts[fname.name] = c
        print(f"  {fname.name}: {c:,}")


if __name__ == "__main__":
    main()
