"""Class taxonomy for ImageGenie weak labeling (FR-3).

Single source of truth mapping the final class roster to its label sources. Two
passes back the labels (see `ml/ml.md#class-list-approach`):

* ``CLASS_TO_LVIS_CATEGORIES`` — curated fine LVIS categories folded into each class. Clean
  seed + gold set; `ml/build_class_list.py` unions their object UIDs to measure
  per-class support and lock the roster against the >=300 bar.
* ``SKETCHFAB_CATEGORY_TO_CLASSES`` — the coarse Sketchfab top-level `categories`
  field as a pre-filter/disambiguator for the full-corpus weak labels (FR-3
  pass 2). Each category maps to the *set* of roster classes plausibly under it;
  tag/title keywords (next commit) pick within the set. `ml/weak_label.py`
  applies it.

LVIS strings must match ``objaverse.load_lvis_annotations()`` keys **exactly**;
quirks are preserved on purpose (e.g. ``crab_(animal)``,
``monitor_(computer_equipment) computer_monitor``). `build_class_list.py`
validates every string against the live keys and fails loudly on a typo.

Curation is deliberate, not a keyword sweep: a broad grep pulls in homographs
(``bowl``/``bowling_ball``/``bowler_hat`` are not animals; ``spear`` and
``steak_knife`` are not food; ``table_lamp`` is a lamp, not a table). Only
visually-coherent members are listed; notable exclusions are noted inline.
"""

from __future__ import annotations

# class -> curated LVIS categories merged into it. Order is by count (descending)
# for readability only; the build script treats each list as a set.
CLASS_TO_LVIS_CATEGORIES: dict[str, list[str]] = {
    # Seating. Excludes typewriter (keyword-sweep false positive).
    "chair": [
        "chair", "armchair", "bench", "sofa", "stool", "pew_(church_bench)",
        "sofa_bed", "loveseat", "wheelchair", "step_stool", "music_stool",
        "footstool", "recliner", "ottoman", "deck_chair", "rocking_chair",
        "highchair", "folding_chair",
    ],
    # Tables/desks. Excludes table_lamp (-> lamp) and tablecloth (textile).
    "table": [
        "table", "desk", "dining_table", "coffee_table", "kitchen_table",
        "pool_table", "table-tennis_table",
    ],
    # Road motor vehicles. Two-wheelers included for volume; rail (freight_car,
    # train) excluded to keep the class visually coherent as "car".
    "car": [
        "pickup_truck", "car_(automobile)", "race_car", "police_cruiser",
        "motorcycle", "jeep", "tractor_(farm_equipment)", "motor_scooter",
        "bus_(vehicle)", "truck", "school_bus", "trailer_truck",
        "convertible_(automobile)", "camper_(vehicle)", "minivan", "cab_(taxi)",
        "ambulance", "garbage_truck", "tow_truck",
    ],
    "aircraft": [
        "fighter_jet", "helicopter", "airplane", "drone", "blimp", "jet_plane",
        "seaplane",
    ],
    # Live creatures. Excludes teddy_bear (-> figure) and the bowl/bowling/
    # bowler homograph cluster.
    "animal": [
        "owl", "lion", "rabbit", "elephant", "crab_(animal)", "shark", "pony",
        "wolf", "frog", "penguin", "butterfly", "kitten", "turtle", "horse",
        "giraffe", "puppy", "cat", "dog", "bird", "fish", "snake", "monkey",
        "lizard", "duckling", "bat_(animal)", "tiger", "cow", "eagle", "deer",
        "chicken_(animal)", "goldfish", "octopus_(animal)", "dolphin", "bear",
        "starfish", "seahorse",
    ],
    # Human/creature figures and figurines.
    "figure": [
        "snowman", "figurine", "teddy_bear", "sculpture", "mascot",
        "statue_(sculpture)", "puppet", "doll", "rag_doll",
    ],
    # Edibles. Vegetables (cauliflower, eggplant) live here, not under plant.
    # Excludes spear/steak_knife (-> weapon).
    "food": [
        "doughnut", "banana", "mushroom", "pumpkin", "apple", "pizza",
        "cupcake", "hamburger", "chocolate_cake", "bread", "pineapple", "pear",
        "tomato", "strawberry", "wedding_cake", "sandwich", "patty_(food)",
        "carrot", "birthday_cake", "cake", "potato", "orange_(fruit)", "sushi",
        "edible_corn", "lemon", "pancake", "sweet_potato", "cornbread",
        "pita_(bread)", "mashed_potato", "crisp_(potato_chip)", "cauliflower",
        "eggplant", "fish_(food)",
    ],
    # Decorative plants/trees + potted arrangements.
    "plant": [
        "Christmas_tree", "flowerpot", "flower_arrangement", "sunflower",
        "carnation", "window_box_(for_plants)", "sugarcane_(plant)",
    ],
    # Consumer electronics. Excludes telephone_booth/telephone_pole (fixtures).
    "electronics": [
        "monitor_(computer_equipment) computer_monitor", "telephone",
        "television_set", "computer_keyboard", "earphone", "radio_receiver",
        "microphone", "boom_microphone", "remote_control",
        "mouse_(computer_equipment)", "laptop_computer", "calculator",
        "speaker_(stero_equipment)", "television_camera",
        "router_(computer_equipment)", "cellular_telephone", "printer",
        "camera",
    ],
    # Arms + armor. Excludes steak_knife (cutlery, -> food).
    "weapon": [
        "shield", "armor", "sword", "rifle", "machine_gun", "pistol", "gun",
        "knife", "projectile_(weapon)", "pocketknife", "spear", "dagger",
        "bow_(weapon)",
    ],
    # Light fixtures. candle_holder included; bare candle excluded (consumable).
    "lamp": [
        "chandelier", "lampshade", "lamppost", "oil_lamp", "table_lamp",
        "lantern", "candle_holder", "streetlight",
    ],
    # Structures. LVIS is object-centric so this is thin here; full volume comes
    # from the Sketchfab `architecture` category in pass 2.
    "building": [
        "windmill", "fireplace", "telephone_booth", "dollhouse", "birdhouse",
        "clock_tower", "water_tower", "houseboat",
    ],
}


# Sketchfab top-level category -> candidate roster classes it can contain (pass 2
# pre-filter + disambiguator). A category maps to the *set* of classes plausibly
# under it; single-class categories yield a label directly, multi-class ones are
# resolved by tag/title keywords (next commit). Categories too abstract or mixed
# to imply any roster class are deliberately omitted — objects whose only
# category is unmapped get no category-gated label. Omitted (and why):
# art-abstract, cultural-heritage-history, science-technology, places-travel,
# fashion-style, sports-fitness, music, news-politics, people-less scenes.
SKETCHFAB_CATEGORY_TO_CLASSES: dict[str, list[str]] = {
    "animals-pets": ["animal"],
    "food-drink": ["food"],
    "weapons-military": ["weapon"],
    "electronics-gadgets": ["electronics"],
    "architecture": ["building"],
    "people": ["figure"],
    # Multi-candidate — need keyword disambiguation:
    "cars-vehicles": ["car", "aircraft"],          # "jet"/"plane" -> aircraft
    "furniture-home": ["chair", "table", "lamp"],   # "lamp"/"desk"/"sofa" split
    "characters-creatures": ["figure", "animal"],   # creature vs. humanoid figure
    "nature-plants": ["plant", "food"],             # produce (apple, mushroom) -> food
}


# class -> indicative tag/title keywords (single lowercase tokens), for resolving a
# multi-candidate category to one class (stage 2). The resolver scores each *in-set*
# candidate by how many of its keywords appear in the object's tokenized tags+title
# and picks a clear winner, so the category gate already rules out cross-category
# homographs ("jaguar" in cars-vehicles only scores car/aircraft, never animal).
# Only the classes reachable via a multi-candidate category are needed here; the
# rest are added if/when out-of-scope objects are rescued by keyword.
CLASS_TO_KEYWORDS: dict[str, list[str]] = {
    "chair": ["chair", "armchair", "sofa", "couch", "seat", "stool", "bench",
              "recliner", "loveseat", "ottoman"],
    "table": ["table", "desk", "nightstand", "dining", "workbench", "countertop"],
    # Bare "light" omitted — too broad (matches lighting/highlight/etc.).
    "lamp": ["lamp", "lampshade", "chandelier", "lantern", "sconce", "luminaire",
             "lamppost", "streetlight"],
    "car": ["car", "truck", "vehicle", "van", "bus", "jeep", "sedan", "suv",
            "automobile", "motorcycle", "scooter", "wagon", "ambulance",
            "tractor", "tank"],
    "aircraft": ["aircraft", "airplane", "plane", "jet", "helicopter", "chopper",
                 "drone", "glider", "biplane", "seaplane", "spaceship", "rocket"],
    "figure": ["character", "figure", "figurine", "statue", "doll", "robot",
               "person", "human", "man", "woman", "girl", "boy", "soldier",
               "warrior", "knight", "hero", "zombie", "skeleton", "mannequin"],
    "animal": ["animal", "creature", "monster", "dragon", "pokemon", "beast",
               "cat", "dog", "horse", "bird", "fish", "snake", "dinosaur",
               "wolf", "bear", "lion", "tiger"],
    # food vs. plant split the `nature-plants` category: produce/fungi are food,
    # flora is plant. Keys mirror the LVIS food/plant merges (edible corn/pumpkin/
    # mushroom are food, not plant).
    "food": ["food", "apple", "banana", "orange", "pear", "strawberry",
             "pineapple", "lemon", "tomato", "potato", "carrot", "corn",
             "pumpkin", "mushroom", "fruit", "vegetable", "bread", "cake",
             "cupcake", "pizza", "burger", "hamburger", "sandwich", "sushi",
             "doughnut", "donut", "egg", "pie"],
    "plant": ["plant", "tree", "flower", "leaf", "cactus", "palm", "fern",
              "bonsai", "succulent", "shrub", "bush", "foliage", "sunflower",
              "rose", "tulip", "houseplant", "ivy", "bamboo", "grass"],
    # Structures only. Excludes generic/element words (monument, fountain) that
    # pull in statues (-> figure); those must stay unconfirmed rather than wrong.
    "building": ["building", "house", "tower", "castle", "church", "cathedral",
                 "temple", "mosque", "bridge", "skyscraper", "barn", "cabin",
                 "cottage", "hut", "lighthouse", "windmill", "fortress", "palace",
                 "mansion", "warehouse", "factory", "dome", "fort", "shed",
                 "stadium", "pyramid", "villa", "apartment", "hall"],
    # weapon/electronics get no keyword disambiguation from their (single-candidate)
    # categories; these lists exist so out-of-scope objects can be rescued. Kept
    # conservative for the gateless rescue: no "mouse" (-> animal), "drone"
    # (-> aircraft), or "tank" (-> car).
    "weapon": ["weapon", "sword", "gun", "rifle", "pistol", "knife", "dagger",
               "spear", "blade", "cannon", "shield", "armor", "firearm",
               "revolver", "shotgun", "grenade", "missile", "katana", "machete",
               "crossbow", "halberd", "warhammer", "axe", "mace"],
    "electronics": ["electronics", "computer", "laptop", "phone", "smartphone",
                    "keyboard", "monitor", "camera", "television", "radio",
                    "speaker", "headphone", "earphone", "microphone", "printer",
                    "router", "console", "tablet", "calculator", "telephone",
                    "motherboard", "circuit", "charger", "gadget"],
}


# Classes that must NOT be auto-labeled from a single category alone — a keyword
# must confirm, or the object is left ambiguous. `building`'s only category source
# is Sketchfab's `architecture`, a grab-bag that also holds benches, statues, and
# streetlights (see ml.md#sketchfab-weak-labeling), so require a building keyword.
CONFIRM_REQUIRED_CLASSES: set[str] = {"building"}
