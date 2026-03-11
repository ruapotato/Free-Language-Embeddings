"""
Programmatic geometry data generation for V16+.

Generates diverse training/test data for geometry losses using templates
with slot filling. Train and test vocabularies are strictly separated
so geometry evaluation tests genuine generalization, not memorization.

Key design:
- Each template type uses 3+ independent slots with 30+ entries each
- Multiple structural patterns per type (not just one template)
- Train/test split by vocabulary, same semantic transformations
- Combinatorial explosion ensures minimal repetition across 600K steps
"""

import random
from typing import List, Tuple, Dict, Optional


# ===========================================================================
# Vocabulary with strict train/test splits
# ===========================================================================

VOCAB = {
    # --- Persons / agents ---
    "person": {
        "train": [
            "the cat", "the dog", "the bird", "alice", "bob", "the teacher",
            "the doctor", "the child", "the wolf", "the king", "the boy",
            "the girl", "the mother", "the father", "john", "mary",
            "the student", "the chef", "the pilot", "the farmer",
            "the soldier", "the dancer", "the painter", "the singer",
            "the writer", "the driver", "the player", "the guard",
            "the clerk", "the judge", "the sailor", "the monk",
        ],
        "test": [
            "the rabbit", "the horse", "the fish", "carol", "dave", "the nurse",
            "the engineer", "the baby", "the fox", "the queen", "the man",
            "the woman", "the brother", "the sister", "tom", "sarah",
            "the professor", "the baker", "the captain", "the gardener",
            "the knight", "the sculptor", "the poet", "the drummer",
            "the author", "the rider", "the coach", "the warden",
            "the mayor", "the priest", "the pirate", "the hermit",
        ],
    },

    # --- Transitive verbs (past tense) ---
    "action_transitive": {
        "train": [
            "chased", "bit", "liked", "praised", "pushed", "watched",
            "called", "hit", "helped", "followed", "trusted", "blamed",
            "admired", "feared", "served", "thanked", "ignored", "found",
            "noticed", "greeted", "defeated", "saved", "tested", "taught",
        ],
        "test": [
            "grabbed", "kicked", "loved", "scolded", "pulled", "observed",
            "phoned", "struck", "aided", "trailed", "doubted", "accused",
            "respected", "dreaded", "assisted", "applauded", "avoided", "spotted",
            "recognized", "welcomed", "conquered", "rescued", "examined", "trained",
        ],
    },

    # --- Intransitive verbs (past/present pairs) ---
    "verb_tense_pair": {
        "train": [
            ("ran", "runs"), ("walked", "walks"), ("danced", "dances"),
            ("played", "plays"), ("slept", "sleeps"), ("laughed", "laughs"),
            ("cried", "cries"), ("worked", "works"), ("cooked", "cooks"),
            ("waited", "waits"), ("listened", "listens"), ("shouted", "shouts"),
            ("whispered", "whispers"), ("studied", "studies"), ("traveled", "travels"),
            ("painted", "paints"), ("sang", "sings"), ("swam", "swims"),
            ("climbed", "climbs"), ("rested", "rests"),
        ],
        "test": [
            ("jumped", "jumps"), ("crawled", "crawls"), ("skated", "skates"),
            ("jogged", "jogs"), ("napped", "naps"), ("giggled", "giggles"),
            ("wept", "weeps"), ("typed", "types"), ("baked", "bakes"),
            ("lingered", "lingers"), ("watched", "watches"), ("yelled", "yells"),
            ("mumbled", "mumbles"), ("practiced", "practices"), ("wandered", "wanders"),
            ("sketched", "sketches"), ("hummed", "hums"), ("dove", "dives"),
            ("hiked", "hikes"), ("dozed", "dozes"),
        ],
    },

    # --- Plurality pairs (plural, singular) ---
    "noun_plurality_pair": {
        "train": [
            ("cats", "cat"), ("dogs", "dog"), ("birds", "bird"),
            ("cars", "car"), ("trees", "tree"), ("books", "book"),
            ("boys", "boy"), ("girls", "girl"), ("houses", "house"),
            ("ships", "ship"), ("stars", "star"), ("songs", "song"),
            ("stones", "stone"), ("rivers", "river"), ("clouds", "cloud"),
            ("kings", "king"), ("swords", "sword"), ("chairs", "chair"),
            ("tables", "table"), ("bells", "bell"),
        ],
        "test": [
            ("horses", "horse"), ("foxes", "fox"), ("rabbits", "rabbit"),
            ("trucks", "truck"), ("flowers", "flower"), ("letters", "letter"),
            ("brothers", "brother"), ("sisters", "sister"), ("towers", "tower"),
            ("boats", "boat"), ("moons", "moon"), ("poems", "poem"),
            ("rocks", "rock"), ("lakes", "lake"), ("storms", "storm"),
            ("queens", "queen"), ("shields", "shield"), ("beds", "bed"),
            ("lamps", "lamp"), ("drums", "drum"),
        ],
    },

    # --- Size adjective pairs (big, small) ---
    "adj_size_pair": {
        "train": [
            ("big", "small"), ("huge", "tiny"), ("large", "little"),
            ("massive", "miniature"), ("tall", "short"), ("wide", "narrow"),
            ("thick", "thin"), ("heavy", "light"), ("deep", "shallow"),
            ("vast", "cramped"),
        ],
        "test": [
            ("enormous", "microscopic"), ("giant", "petite"), ("grand", "compact"),
            ("immense", "minute"), ("towering", "squat"), ("broad", "slim"),
            ("dense", "sparse"), ("bulky", "feathery"), ("profound", "superficial"),
            ("expansive", "confined"),
        ],
    },

    # --- Sentiment adjective pairs (positive, negative) ---
    "adj_sentiment_pair": {
        "train": [
            ("great", "terrible"), ("wonderful", "horrible"), ("good", "bad"),
            ("beautiful", "ugly"), ("happy", "sad"), ("kind", "cruel"),
            ("brave", "cowardly"), ("clever", "foolish"), ("calm", "anxious"),
            ("friendly", "hostile"), ("brilliant", "dull"), ("gentle", "harsh"),
        ],
        "test": [
            ("fantastic", "dreadful"), ("superb", "awful"), ("excellent", "poor"),
            ("gorgeous", "hideous"), ("joyful", "miserable"), ("generous", "selfish"),
            ("courageous", "timid"), ("wise", "ignorant"), ("peaceful", "restless"),
            ("warm", "cold"), ("magnificent", "mediocre"), ("tender", "rough"),
        ],
    },

    # --- General adjectives (for subject swaps, negation) ---
    "adjective": {
        "train": [
            "happy", "tired", "hungry", "angry", "tall", "fast",
            "strong", "quiet", "busy", "proud", "ready", "careful",
            "honest", "patient", "polite", "certain", "curious", "healthy",
            "awake", "afraid", "alone", "sorry", "lucky", "eager",
        ],
        "test": [
            "joyful", "exhausted", "starving", "furious", "lean", "swift",
            "mighty", "silent", "occupied", "humble", "prepared", "cautious",
            "truthful", "tolerant", "courteous", "confident", "inquisitive", "fit",
            "alert", "nervous", "isolated", "grateful", "fortunate", "keen",
        ],
    },

    # --- Objects for ditransitive/give constructions ---
    "object": {
        "train": [
            "a book", "a cake", "a letter", "coffee", "a secret",
            "a coin", "a flower", "a toy", "a map", "a key",
            "a hat", "a ring", "a sword", "a gift", "a blanket",
            "some bread", "a ticket", "a painting", "a note", "a tool",
        ],
        "test": [
            "a scroll", "a pie", "a message", "water", "a story",
            "a medal", "a leaf", "a doll", "a compass", "a lock",
            "a scarf", "a pendant", "a shield", "a prize", "a cloak",
            "some fruit", "a pass", "a drawing", "a card", "a lantern",
        ],
    },

    # --- Locations ---
    "location": {
        "train": [
            "in the park", "at home", "on the street", "by the river",
            "in the kitchen", "at the store", "on the hill", "near the church",
            "in the garden", "at the office", "on the bridge", "by the fire",
            "in the forest", "at the market", "on the roof", "near the tower",
        ],
        "test": [
            "in the meadow", "at school", "on the trail", "by the lake",
            "in the cellar", "at the inn", "on the cliff", "near the castle",
            "in the valley", "at the harbor", "on the deck", "by the fountain",
            "in the cave", "at the plaza", "on the shore", "near the ruins",
        ],
    },

    # --- Time expressions ---
    "time": {
        "train": [
            "yesterday", "last week", "this morning", "every day",
            "at dawn", "before lunch", "after dinner", "in the evening",
            "on monday", "last summer", "all night", "during the storm",
        ],
        "test": [
            "tonight", "last month", "this afternoon", "on weekends",
            "at dusk", "before class", "after work", "in the night",
            "on friday", "last winter", "all day", "during the rain",
        ],
    },

    # --- Manner adverbs ---
    "manner": {
        "train": [
            "quickly", "slowly", "carefully", "loudly", "quietly",
            "happily", "sadly", "eagerly", "gently", "bravely",
            "proudly", "nervously", "gracefully", "fiercely", "patiently",
        ],
        "test": [
            "swiftly", "gradually", "cautiously", "noisily", "softly",
            "cheerfully", "gloomily", "anxiously", "tenderly", "boldly",
            "humbly", "restlessly", "elegantly", "wildly", "calmly",
        ],
    },

    # --- Nouns for standalone use ---
    "noun": {
        "train": [
            "cat", "dog", "house", "tree", "car", "book", "river",
            "mountain", "ship", "sword", "castle", "bridge", "storm",
            "garden", "market", "forest", "tower", "crown", "fire", "stone",
        ],
        "test": [
            "horse", "fox", "cottage", "flower", "truck", "letter", "lake",
            "volcano", "boat", "shield", "fortress", "tunnel", "flood",
            "meadow", "harbor", "jungle", "lighthouse", "throne", "ice", "crystal",
        ],
    },
}

# ===========================================================================
# Cluster sentences by semantic category
# Each category has templates that produce coherent within-group sentences
# ===========================================================================

CLUSTER_VOCAB = {
    "animals": {
        "train": {
            "animals": ["the cat", "a dog", "the bird", "the wolf", "the bear",
                        "a deer", "the eagle", "a snake", "the lion", "a turtle"],
            "actions": ["ran through", "sat in", "flew over", "hid behind",
                        "swam across", "hunted near", "slept under", "climbed up"],
            "places": ["the field", "the forest", "the river", "the mountain",
                       "the meadow", "the tall grass", "the old tree", "the dark cave"],
        },
        "test": {
            "animals": ["the horse", "a rabbit", "the owl", "the fox", "the moose",
                        "a frog", "the hawk", "a lizard", "the tiger", "a crab"],
            "actions": ["galloped across", "rested in", "soared above", "crept behind",
                        "waded through", "prowled near", "dozed beneath", "scrambled over"],
            "places": ["the valley", "the jungle", "the pond", "the hillside",
                       "the clearing", "the thick brush", "the fallen log", "the rocky den"],
        },
    },
    "weather": {
        "train": {
            "descriptions": ["it was raining", "the sun was shining", "snow covered the ground",
                             "a storm approached", "the wind blew hard", "fog rolled in",
                             "thunder rumbled", "lightning flashed", "hail fell", "frost formed"],
            "modifiers": ["heavily today", "brightly this morning", "all night long",
                          "from the west", "across the plains", "at dawn",
                          "in the distance", "across the sky", "on the roof", "on every surface"],
        },
        "test": {
            "descriptions": ["it was drizzling", "the clouds parted", "ice coated the roads",
                             "a hurricane neared", "the breeze picked up", "mist settled",
                             "thunder cracked", "lightning struck", "sleet pelted", "dew gathered"],
            "modifiers": ["softly tonight", "warmly this afternoon", "all day long",
                          "from the north", "over the hills", "at dusk",
                          "beyond the horizon", "through the clouds", "on the windows", "on every branch"],
        },
    },
    "food": {
        "train": {
            "people": ["she", "he", "the chef", "the mother", "they",
                       "the baker", "the boy", "grandmother", "the cook", "the girl"],
            "actions": ["cooked", "baked", "ate", "prepared", "served",
                        "tasted", "stirred", "sliced", "grilled", "roasted"],
            "foods": ["a delicious pasta", "fresh bread", "a bowl of soup",
                      "a chocolate cake", "a warm pie", "grilled chicken",
                      "a fruit salad", "rice and beans", "a hearty stew", "fresh cookies"],
        },
        "test": {
            "people": ["the woman", "the man", "the sous chef", "the father", "we",
                       "the pastry chef", "the teenager", "grandfather", "the apprentice", "the aunt"],
            "actions": ["fried", "steamed", "devoured", "assembled", "presented",
                        "sampled", "whisked", "diced", "broiled", "smoked"],
            "foods": ["a spicy curry", "warm rolls", "a cup of chili",
                      "a lemon tart", "a savory quiche", "seared salmon",
                      "a garden salad", "noodles and broth", "a thick gumbo", "fresh muffins"],
        },
    },
    "emotions": {
        "train": {
            "people": ["she was", "he felt", "they were", "the child was",
                       "the teacher felt", "the boy seemed", "alice looked",
                       "the soldier appeared", "the dancer was", "the old man felt"],
            "emotions": ["very happy today", "sad and lonely", "angry about it",
                         "excited and eager", "anxious before the test", "proud of the work",
                         "grateful for help", "confused by the news", "hopeful about tomorrow",
                         "peaceful and calm"],
        },
        "test": {
            "people": ["the woman was", "the man felt", "we were", "the baby was",
                       "the professor felt", "the girl seemed", "carol looked",
                       "the knight appeared", "the poet was", "the young woman felt"],
            "emotions": ["deeply content now", "gloomy and withdrawn", "furious about it",
                         "thrilled and restless", "nervous before the match", "humble after the win",
                         "thankful for kindness", "puzzled by the message", "optimistic about the future",
                         "serene and relaxed"],
        },
    },
    "technology": {
        "train": {
            "subjects": ["the computer", "the server", "the program", "the network",
                         "the database", "the website", "the app", "the system",
                         "the code", "the software"],
            "predicates": ["crashed unexpectedly", "ran slowly today", "was updated last night",
                           "failed to respond", "processed the request", "stored the data",
                           "displayed an error", "connected to the network",
                           "compiled without errors", "handled the load well"],
        },
        "test": {
            "subjects": ["the laptop", "the router", "the script", "the firewall",
                         "the cache", "the portal", "the plugin", "the platform",
                         "the module", "the firmware"],
            "predicates": ["froze without warning", "loaded slowly yesterday", "was patched this morning",
                           "refused to boot", "executed the command", "backed up the files",
                           "showed a warning", "synced with the cloud",
                           "deployed successfully", "managed the traffic well"],
        },
    },
    "sports": {
        "train": {
            "people": ["he", "she", "the team", "the player", "the coach",
                       "the runner", "the goalkeeper", "the boxer", "the swimmer", "the cyclist"],
            "actions": ["kicked the ball hard", "ran the marathon", "won the game",
                        "scored in the last minute", "trained for hours", "broke the record",
                        "blocked the shot", "crossed the finish line",
                        "lifted the trophy", "sprinted down the track"],
        },
        "test": {
            "people": ["the woman", "the man", "the squad", "the athlete", "the trainer",
                       "the sprinter", "the pitcher", "the wrestler", "the diver", "the skater"],
            "actions": ["threw the javelin far", "finished the race", "claimed the title",
                        "tied it in overtime", "practiced all week", "set a new record",
                        "saved the penalty", "reached the summit",
                        "hoisted the medal", "dashed across the field"],
        },
    },
    "travel": {
        "train": {
            "people": ["they", "she", "he", "the family", "the tourists",
                       "the explorer", "the pilgrim", "the merchant", "the sailor", "we"],
            "actions": ["flew to", "drove across", "sailed to", "hiked through",
                        "took a train to", "journeyed to", "wandered through",
                        "arrived at", "departed from", "explored"],
            "destinations": ["paris last week", "the countryside", "a tropical island",
                             "the ancient ruins", "the capital city", "the northern coast",
                             "the desert", "the mountain village", "the busy port", "the old quarter"],
        },
        "test": {
            "people": ["the couple", "the woman", "the man", "the group", "the backpackers",
                       "the adventurer", "the nomad", "the trader", "the captain", "the friends"],
            "actions": ["traveled to", "rode through", "cruised to", "trekked across",
                        "caught a bus to", "ventured to", "strolled through",
                        "reached", "left from", "discovered"],
            "destinations": ["tokyo last month", "the highlands", "a remote island",
                             "the buried temple", "the harbor town", "the eastern shore",
                             "the tundra", "the fishing village", "the crowded dock", "the hidden alley"],
        },
    },
    "music": {
        "train": {
            "people": ["she", "he", "the band", "the orchestra", "the singer",
                       "the pianist", "the guitarist", "the drummer", "the composer", "the choir"],
            "actions": ["played a beautiful melody", "performed on stage", "sang a love song",
                        "composed a new piece", "practiced for hours", "recorded an album",
                        "tuned the instrument", "conducted the symphony",
                        "strummed a gentle chord", "hummed a familiar tune"],
        },
        "test": {
            "people": ["the woman", "the man", "the trio", "the ensemble", "the vocalist",
                       "the cellist", "the bassist", "the percussionist", "the arranger", "the quartet"],
            "actions": ["played a haunting riff", "debuted at the hall", "sang a folk ballad",
                        "wrote a new arrangement", "rehearsed all evening", "released a single",
                        "adjusted the strings", "led the orchestra",
                        "plucked a soft note", "whistled a catchy phrase"],
        },
    },
}


# ===========================================================================
# Generator class
# ===========================================================================

class GeometryDataGenerator:
    """Generates diverse geometry training/test data from templates.

    Usage:
        train_gen = GeometryDataGenerator(split="train")
        test_gen = GeometryDataGenerator(split="test", seed=42)

        # Each call returns different examples
        origs, swaps = train_gen.word_order_batch(16)
        a, b, c, d = train_gen.analogy_batch(6)
    """

    def __init__(self, split: str = "train", seed: Optional[int] = None):
        assert split in ("train", "test"), f"split must be 'train' or 'test', got {split}"
        self.split = split
        self.rng = random.Random(seed)

    def _pick(self, key: str, n: int = 1) -> list:
        """Pick n random distinct items from VOCAB[key][split]."""
        pool = VOCAB[key][self.split]
        if n >= len(pool):
            return list(pool)
        return self.rng.sample(pool, n)

    def _pick_one(self, key: str):
        return VOCAB[key][self.split][self.rng.randint(0, len(VOCAB[key][self.split]) - 1)]

    def _pick_pair(self, key: str):
        """Pick a random pair from a paired vocab (e.g., verb_tense_pair)."""
        pool = VOCAB[key][self.split]
        return pool[self.rng.randint(0, len(pool) - 1)]

    # -------------------------------------------------------------------
    # Word Order: generate (original, swapped) pairs
    # -------------------------------------------------------------------

    def word_order_batch(self, batch_size: int = 16) -> Tuple[List[str], List[str]]:
        """Generate word-order swap pairs using multiple templates."""
        origs, swaps = [], []
        for _ in range(batch_size):
            template = self.rng.randint(0, 4)
            if template == 0:
                # Simple transitive: "A verb B" vs "B verb A"
                a, b = self._pick("person", 2)
                v = self._pick_one("action_transitive")
                origs.append(f"{a} {v} {b}")
                swaps.append(f"{b} {v} {a}")
            elif template == 1:
                # Ditransitive: "A gave B obj" vs "B gave A obj"
                a, b = self._pick("person", 2)
                obj = self._pick_one("object")
                origs.append(f"{a} gave {b} {obj}")
                swaps.append(f"{b} gave {a} {obj}")
            elif template == 2:
                # "A told B + location" vs "B told A + location"
                a, b = self._pick("person", 2)
                loc = self._pick_one("location")
                origs.append(f"{a} met {b} {loc}")
                swaps.append(f"{b} met {a} {loc}")
            elif template == 3:
                # "A verb B manner" vs "B verb A manner"
                a, b = self._pick("person", 2)
                v = self._pick_one("action_transitive")
                m = self._pick_one("manner")
                origs.append(f"{a} {v} {b} {m}")
                swaps.append(f"{b} {v} {a} {m}")
            else:
                # "A served B obj time" vs "B served A obj time"
                a, b = self._pick("person", 2)
                obj = self._pick_one("object")
                t = self._pick_one("time")
                origs.append(f"{a} brought {b} {obj} {t}")
                swaps.append(f"{b} brought {a} {obj} {t}")

        return origs, swaps

    # -------------------------------------------------------------------
    # Analogies: generate (a, b, c, d) quads — a:b :: c:d
    # -------------------------------------------------------------------

    def analogy_batch(self, batch_size: int = 6) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Generate analogy quads using diverse templates."""
        a_texts, b_texts, c_texts, d_texts = [], [], [], []

        for _ in range(batch_size):
            template = self.rng.randint(0, 7)

            if template == 0:
                # Tense: "person past" : "person present" :: "person2 past" : "person2 present"
                p1, p2 = self._pick("person", 2)
                past1, pres1 = self._pick_pair("verb_tense_pair")
                past2, pres2 = self._pick_pair("verb_tense_pair")
                a_texts.append(f"{p1} {past1}")
                b_texts.append(f"{p1} {pres1}")
                c_texts.append(f"{p2} {past2}")
                d_texts.append(f"{p2} {pres2}")

            elif template == 1:
                # Sentiment: "the noun was pos" : "the noun was neg" :: same with noun2
                n1, n2 = self._pick("noun", 2)
                pos1, neg1 = self._pick_pair("adj_sentiment_pair")
                pos2, neg2 = self._pick_pair("adj_sentiment_pair")
                a_texts.append(f"the {n1} was {pos1}")
                b_texts.append(f"the {n1} was {neg1}")
                c_texts.append(f"the {n2} was {pos2}")
                d_texts.append(f"the {n2} was {neg2}")

            elif template == 2:
                # Negation: "person is not adj" : "person is adj" :: person2
                p1, p2 = self._pick("person", 2)
                adj1 = self._pick_one("adjective")
                adj2 = self._pick_one("adjective")
                a_texts.append(f"{p1} is not {adj1}")
                b_texts.append(f"{p1} is {adj1}")
                c_texts.append(f"{p2} is not {adj2}")
                d_texts.append(f"{p2} is {adj2}")

            elif template == 3:
                # Plurality: "the Xs" : "the X" :: "the Ys" : "the Y"
                pl1, sg1 = self._pick_pair("noun_plurality_pair")
                pl2, sg2 = self._pick_pair("noun_plurality_pair")
                a_texts.append(f"the {pl1}")
                b_texts.append(f"the {sg1}")
                c_texts.append(f"the {pl2}")
                d_texts.append(f"the {sg2}")

            elif template == 4:
                # Size: "the big noun" : "the small noun" :: noun2
                n1, n2 = self._pick("noun", 2)
                big1, small1 = self._pick_pair("adj_size_pair")
                big2, small2 = self._pick_pair("adj_size_pair")
                a_texts.append(f"the {big1} {n1}")
                b_texts.append(f"the {small1} {n1}")
                c_texts.append(f"the {big2} {n2}")
                d_texts.append(f"the {small2} {n2}")

            elif template == 5:
                # Subject swap: "he is adj" : "she is adj" :: "he is adj2" : "she is adj2"
                adj1, adj2 = self._pick("adjective", 2)
                a_texts.append(f"he is {adj1}")
                b_texts.append(f"she is {adj1}")
                c_texts.append(f"he is {adj2}")
                d_texts.append(f"she is {adj2}")

            elif template == 6:
                # Location: "person verb loc1" : "person verb loc2" :: "person2 verb loc1" : "person2 verb loc2"
                p1, p2 = self._pick("person", 2)
                past, _ = self._pick_pair("verb_tense_pair")
                loc1, loc2 = self._pick("location", 2)
                a_texts.append(f"{p1} {past} {loc1}")
                b_texts.append(f"{p1} {past} {loc2}")
                c_texts.append(f"{p2} {past} {loc1}")
                d_texts.append(f"{p2} {past} {loc2}")

            else:
                # Manner: "person verb manner1" : "person verb manner2" :: person2
                p1, p2 = self._pick("person", 2)
                past, _ = self._pick_pair("verb_tense_pair")
                m1, m2 = self._pick("manner", 2)
                a_texts.append(f"{p1} {past} {m1}")
                b_texts.append(f"{p1} {past} {m2}")
                c_texts.append(f"{p2} {past} {m1}")
                d_texts.append(f"{p2} {past} {m2}")

        return a_texts, b_texts, c_texts, d_texts

    # -------------------------------------------------------------------
    # Direction pairs: grouped by attribute (same transformation)
    # -------------------------------------------------------------------

    def direction_batch(self, n_pairs_per_attr: int = 3) -> Dict[str, List[Tuple[str, str]]]:
        """Generate direction pairs grouped by semantic attribute."""
        result = {}

        # Negation: "person is not adj" vs "person is adj"
        pairs = []
        for _ in range(n_pairs_per_attr):
            p = self._pick_one("person")
            adj = self._pick_one("adjective")
            pairs.append((f"{p} is not {adj}", f"{p} is {adj}"))
        result["negation"] = pairs

        # Tense: "person past" vs "person present"
        pairs = []
        for _ in range(n_pairs_per_attr):
            p = self._pick_one("person")
            past, pres = self._pick_pair("verb_tense_pair")
            pairs.append((f"{p} {past}", f"{p} {pres}"))
        result["tense"] = pairs

        # Sentiment: "the noun was pos" vs "the noun was neg"
        pairs = []
        for _ in range(n_pairs_per_attr):
            n = self._pick_one("noun")
            pos, neg = self._pick_pair("adj_sentiment_pair")
            pairs.append((f"the {n} was {pos}", f"the {n} was {neg}"))
        result["sentiment"] = pairs

        # Plurality: "the Xs" vs "the X"
        pairs = []
        for _ in range(n_pairs_per_attr):
            pl, sg = self._pick_pair("noun_plurality_pair")
            past, _ = self._pick_pair("verb_tense_pair")
            pairs.append((f"the {pl} {past}", f"the {sg} {past}"))
        result["plurality"] = pairs

        # Size: "the big noun" vs "the small noun"
        pairs = []
        for _ in range(n_pairs_per_attr):
            n = self._pick_one("noun")
            big, small = self._pick_pair("adj_size_pair")
            pairs.append((f"the {big} {n}", f"the {small} {n}"))
        result["size"] = pairs

        # Manner: "person verb manner1" vs "person verb manner2"
        pairs = []
        for _ in range(n_pairs_per_attr):
            p = self._pick_one("person")
            past, _ = self._pick_pair("verb_tense_pair")
            m1, m2 = self._pick("manner", 2)
            pairs.append((f"{p} {past} {m1}", f"{p} {past} {m2}"))
        result["manner"] = pairs

        # Location: "person verb loc1" vs "person verb loc2"
        pairs = []
        for _ in range(n_pairs_per_attr):
            p = self._pick_one("person")
            past, _ = self._pick_pair("verb_tense_pair")
            loc1, loc2 = self._pick("location", 2)
            pairs.append((f"{p} {past} {loc1}", f"{p} {past} {loc2}"))
        result["location"] = pairs

        return result

    # -------------------------------------------------------------------
    # Cluster sentences: groups of semantically similar sentences
    # -------------------------------------------------------------------

    def cluster_batch(self, n_groups: int = 3, n_per_group: int = 3) -> Dict[str, List[str]]:
        """Generate groups of semantically related sentences."""
        available = list(CLUSTER_VOCAB.keys())
        selected = self.rng.sample(available, min(n_groups, len(available)))
        result = {}

        for group_name in selected:
            vocab = CLUSTER_VOCAB[group_name][self.split]
            sents = []
            for _ in range(n_per_group):
                sents.append(self._generate_cluster_sentence(group_name, vocab))
            result[group_name] = sents

        return result

    def _generate_cluster_sentence(self, group: str, vocab: dict) -> str:
        """Generate one sentence for a cluster group."""
        if group == "animals":
            animal = self.rng.choice(vocab["animals"])
            action = self.rng.choice(vocab["actions"])
            place = self.rng.choice(vocab["places"])
            return f"{animal} {action} {place}"

        elif group == "weather":
            desc = self.rng.choice(vocab["descriptions"])
            mod = self.rng.choice(vocab["modifiers"])
            return f"{desc} {mod}"

        elif group == "food":
            person = self.rng.choice(vocab["people"])
            action = self.rng.choice(vocab["actions"])
            food = self.rng.choice(vocab["foods"])
            return f"{person} {action} {food}"

        elif group == "emotions":
            person = self.rng.choice(vocab["people"])
            emotion = self.rng.choice(vocab["emotions"])
            return f"{person} {emotion}"

        elif group == "technology":
            subj = self.rng.choice(vocab["subjects"])
            pred = self.rng.choice(vocab["predicates"])
            return f"{subj} {pred}"

        elif group == "sports":
            person = self.rng.choice(vocab["people"])
            action = self.rng.choice(vocab["actions"])
            return f"{person} {action}"

        elif group == "travel":
            person = self.rng.choice(vocab["people"])
            action = self.rng.choice(vocab["actions"])
            dest = self.rng.choice(vocab["destinations"])
            return f"{person} {action} {dest}"

        elif group == "music":
            person = self.rng.choice(vocab["people"])
            action = self.rng.choice(vocab["actions"])
            return f"{person} {action}"

        else:
            raise ValueError(f"Unknown cluster group: {group}")

    # -------------------------------------------------------------------
    # Diverse unrelated sentences (for batch repulsion)
    # -------------------------------------------------------------------

    def diverse_sentences(self, batch_size: int = 32) -> List[str]:
        """Generate diverse unrelated sentences for repulsion losses."""
        sents = []
        templates = [
            lambda: f"{self._pick_one('person')} {self._pick_one('action_transitive')} {self._pick_one('person')}",
            lambda: f"{self._pick_one('person')} {self._pick_pair('verb_tense_pair')[0]} {self._pick_one('location')}",
            lambda: f"the {self._pick_one('noun')} was {self._pick_one('adjective')}",
            lambda: f"{self._pick_one('person')} is {self._pick_one('adjective')}",
            lambda: f"the {self._pick_pair('adj_size_pair')[0]} {self._pick_one('noun')} {self._pick_pair('verb_tense_pair')[0]} {self._pick_one('manner')}",
            lambda: f"{self._pick_one('person')} gave {self._pick_one('person')} {self._pick_one('object')} {self._pick_one('time')}",
        ]
        for _ in range(batch_size):
            t = self.rng.choice(templates)
            sents.append(t())
        return sents


# ===========================================================================
# Verification: ensure no overlap between train and test vocabularies
# ===========================================================================

def verify_splits():
    """Assert zero overlap between train and test vocabularies."""
    for key, splits in VOCAB.items():
        train_set = set()
        test_set = set()
        for item in splits["train"]:
            if isinstance(item, tuple):
                train_set.update(item)
            else:
                train_set.add(item)
        for item in splits["test"]:
            if isinstance(item, tuple):
                test_set.update(item)
            else:
                test_set.add(item)
        overlap = train_set & test_set
        assert not overlap, f"VOCAB[{key}] has train/test overlap: {overlap}"

    for group, splits in CLUSTER_VOCAB.items():
        for vocab_key in splits["train"]:
            train_items = set(splits["train"][vocab_key])
            test_items = set(splits["test"][vocab_key])
            overlap = train_items & test_items
            assert not overlap, f"CLUSTER_VOCAB[{group}][{vocab_key}] overlap: {overlap}"

    print("All train/test splits verified: zero overlap.")


if __name__ == "__main__":
    verify_splits()

    # Demo: show some generated examples
    gen = GeometryDataGenerator(split="train", seed=123)

    print("\n=== Word Order Pairs ===")
    origs, swaps = gen.word_order_batch(5)
    for o, s in zip(origs, swaps):
        print(f"  {o}  vs  {s}")

    print("\n=== Analogy Quads ===")
    a, b, c, d = gen.analogy_batch(5)
    for i in range(5):
        print(f"  {a[i]} : {b[i]} :: {c[i]} : {d[i]}")

    print("\n=== Direction Pairs ===")
    dirs = gen.direction_batch(2)
    for attr, pairs in dirs.items():
        print(f"  [{attr}]")
        for p1, p2 in pairs:
            print(f"    {p1}  vs  {p2}")

    print("\n=== Cluster Sentences ===")
    clusters = gen.cluster_batch(3, 3)
    for group, sents in clusters.items():
        print(f"  [{group}]")
        for s in sents:
            print(f"    {s}")

    print("\n=== Test split (different vocabulary) ===")
    test_gen = GeometryDataGenerator(split="test", seed=123)
    origs, swaps = test_gen.word_order_batch(3)
    for o, s in zip(origs, swaps):
        print(f"  {o}  vs  {s}")
