"""GitHub-style random session codenames (``Adjective-Noun``)."""
from __future__ import annotations

import random
from typing import Optional, Set

ADJECTIVES = (
    "Adaptable", "Agile", "Amazing", "Amber", "Ancient", "Arctic", "Astral", "Atomic",
    "Aurora", "Autumn", "Balanced", "Bold", "Brave", "Bright", "Calm", "Candid",
    "Celestial", "Clever", "Cosmic", "Crystal", "Curious", "Daring", "Dawn", "Dazzling",
    "Deep", "Divine", "Electric", "Elegant", "Emerald", "Epic", "Fearless", "Floral",
    "Flying", "Forest", "Gentle", "Glowing", "Golden", "Grand", "Hidden", "Humble",
    "Infinite", "Jolly", "Jovial", "Keen", "Kind", "Lively", "Lucky", "Lunar",
    "Magic", "Marble", "Mighty", "Misty", "Modern", "Mystic", "Noble", "Northern",
    "Oceanic", "Patient", "Peaceful", "Polar", "Prime", "Proud", "Quantum", "Quiet",
    "Radiant", "Rapid", "Robust", "Royal", "Rustic", "Sacred", "Scarlet", "Secret",
    "Serene", "Shadow", "Sharp", "Silent", "Silver", "Solar", "Solid", "Steady",
    "Stellar", "Stormy", "Subtle", "Sunny", "Swift", "Tender", "Timeless", "Tranquil",
    "True", "Valiant", "Vast", "Velvet", "Verdant", "Vivid", "Warm", "Wild",
    "Wise", "Witty", "Zen",
)

NOUNS = (
    "Acorn", "Aurora", "Badger", "Beacon", "Blossom", "Breeze", "Brook", "Canyon",
    "Cedar", "Cloud", "Comet", "Coral", "Crane", "Crown", "Dawn", "Dolphin",
    "Dragon", "Dream", "Eagle", "Echo", "Ember", "Falcon", "Fjord", "Flame",
    "Forest", "Fox", "Garden", "Glacier", "Harbor", "Hawk", "Heron", "Horizon",
    "Island", "Jaguar", "Journey", "Lagoon", "Lantern", "Legend", "Lotus", "Lynx",
    "Maple", "Meadow", "Meteor", "Mirror", "Moon", "Moss", "Mountain", "Nebula",
    "Nova", "Oak", "Octopus", "Orbit", "Otter", "Owl", "Panda", "Panther",
    "Pearl", "Phoenix", "Pine", "Planet", "Prairie", "Quartz", "Quest", "Rain",
    "Raven", "River", "Robin", "Rocket", "Sage", "Saturn", "Shadow", "Sierra",
    "Spark", "Sparrow", "Spirit", "Spruce", "Star", "Stone", "Storm", "Summit",
    "Sun", "Swan", "Thunder", "Tiger", "Torch", "Tower", "Trail", "Valley",
    "Violet", "Voyage", "Wave", "Willow", "Wolf", "Wonder", "Zephyr",
)


def generate_session_id(existing: Optional[Set[str]] = None, rng: Optional[random.Random] = None) -> str:
    """Return a unique codename such as ``Quantum-Octopus``."""
    taken = existing or set()
    randomizer = rng or random.Random()
    for _ in range(200):
        candidate = f"{randomizer.choice(ADJECTIVES)}-{randomizer.choice(NOUNS)}"
        if candidate not in taken:
            return candidate
    # Extremely unlikely fallback when the word space is exhausted.
    while True:
        candidate = f"{randomizer.choice(ADJECTIVES)}-{randomizer.choice(NOUNS)}-{randomizer.randint(100, 999)}"
        if candidate not in taken:
            return candidate
