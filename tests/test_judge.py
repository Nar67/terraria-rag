import pytest

from judge import quote_in

SOURCES = [{"text": "\n".join([
    "Guide : Crafting a Shellphone > Goblin Tech",
    "- Stopwatch: Buy from Traveling Merchant (see above for tips).",
    "- DPS Meter: Buy from Traveling Merchant (see above for tips).",
    "- Metal Detector: Create a long tunnel in an Ice biome.",
    "Once you are satisfied, head to the underworld. Kill a Voodoo Demon to get a Guide Voodoo Doll.",
    "- Defeat Skeletron in order to access the Dungeon enemies and drop the Tally Counter.",
])}]


@pytest.mark.parametrize("quote, found", [
    # real support, quoted exactly, condensed from a list, or lightly paraphrased
    ("Kill a Voodoo Demon to get a Guide Voodoo Doll.", True),
    ("Goblin Tech: Stopwatch, DPS Meter, Metal Detector", True),
    ("Stopwatch: Buy from Traveling Merchant (see above for tips).\nDPS Meter: Buy from Traveling Merchant (see above for tips).", True),
    ("Tally Counter: Drop from Dungeon enemies (post-Skeletron)", True),
    # invented or misplaced support must fail
    ("Kill a Voodoo Demon in the Jungle Temple", False),
    ("Crafting the True Excalibur requires the Excalibur and 24 Chlorophyte Bars.", False),
    ("Stopwatch costs 5 Gold from the Traveling Merchant.", False),  # a number not in the excerpts
    ("", False),
])
def test_quote_in(quote, found):
    assert quote_in(quote, SOURCES) is found
