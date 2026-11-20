"""facts.py on small HTML snippets copied from the real wiki markup
(King Slime, Mothron, Eye of Cthulhu, Wooden Arrow, Recipes/Work Bench)."""
import pytest
from bs4 import BeautifulSoup

from facts import MODES, mode_values, modes_of, parse_page


@pytest.mark.parametrize("classes, modes", [
    (["m-normal", "m-journey"], {"classic"}),
    (["m-expert-master"], {"expert", "master"}),
    (["m-normal-expert-master"], set(MODES)),
    (["m-all"], set(MODES)),
    (["mode-exclusive", "expert"], set()),  # styling class, not a mode marker
])
def test_modes_of(classes, modes):
    assert modes_of(classes) == modes


@pytest.mark.parametrize("cell, values", [
    # one value per mode, the Expert/Master ones wrapped in links (Mothron Wings)
    ('<td><span class="mode-content"><span class="m-normal m-journey">5%</span>'
     '<span class="m-expert-master"><a href="/wiki/Expert_Mode"><span class="mode-exclusive expert">9.75%</span></a>'
     '</span></span></td>', {"classic": "5%", "expert": "9.75%", "master": "9.75%"}),
    # no mode markers: same everywhere
    ("<td>10%</td>", {"classic": "10%", "expert": "10%", "master": "10%"}),
    # phase thresholds stay together per mode (Eye of Cthulhu, second form)
    ('<td><span class="m-normal">1400/2800</span><span class="m-expert">2366/3640</span>'
     '<span class="m-master">3016/4641</span></td>', {"classic": "1400/2800", "expert": "2366/3640", "master": "3016/4641"}),
])
def test_mode_values(cell, values):
    assert mode_values(BeautifulSoup(cell, "lxml").td) == values


NPC = """<div class="infobox npc modesbox"><div class="title">Mothron Hardmode exclusive</div>
<div class="section statistics"><table class="stat">
 <tr><th>Type</th><td><div class="tags"><span class="tag"><a>Boss</a></span></div></td></tr>
 <tr><th>Max Life</th><td><span class="npcstat"><span class="m-normal m-journey">6000</span>
   <span class="m-expert">12000</span><span class="m-master">18000</span></span></td></tr>
 <tr><th>Defense</th><td><span class="npcstat"><span class="m-all">50</span></span></td></tr>
</table></div>
<ul class="drops items">
 <li><strong>Item</strong><strong>Rate</strong></li>
 <li class="m-normal"><div><span class="i"><a title="Broken Hero Sword">Broken Hero Sword</a></span></div><div>25%</div></li>
 <li class="groupstart caption m-normal">One of the following 2 items will be dropped<sup class="reference">[1]</sup></li>
 <li class="m-normal group"><div><span class="i"><a title="Moth Wings">Moth Wings</a></span></div><div>50%</div></li>
 <li class="groupend m-normal"></li>
 <li><div><span class="i"><a title="Luminite">Luminite</a> (70–90)</span></div><div>100%</div></li>
 <li class="lootstart m-expert-master"></li>
 <li class="m-expert-master loot"><div><span class="i"><a title="Gravity Globe">Gravity Globe</a></span></div><div>100%</div></li>
 <li class="m-master"><div><span class="i"><a title="Badger's Hat">Badger's Hat</a> (when defeated on the same day)</span></div><div>100%</div></li>
</ul></div>"""


def test_parse_npc():
    npc = parse_page(NPC, "Mothron")["npcs"][0]
    assert npc["name"] == "Mothron" and npc["hardmode_only"]
    assert npc["tags"]["Type"] == ["Boss"]
    assert npc["stats"]["Max Life"] == {"classic": "6000", "expert": "12000", "master": "18000"}
    assert npc["stats"]["Defense"] == {m: "50" for m in MODES}
    drops = {d["item"]: d for d in npc["drops"]}
    assert drops["Broken Hero Sword"]["rate"] == {"classic": "25%"}       # Classic only
    assert drops["Moth Wings"]["group"].startswith("One of the following 2")  # footnote [1] removed
    assert drops["Luminite"]["quantity"] == "70–90" and drops["Luminite"]["group"] is None
    assert drops["Gravity Globe"]["from_treasure_bag"]
    assert drops["Badger's Hat"]["condition"] == "when defeated on the same day"
    assert drops["Badger's Hat"]["rate"] == {"master": "100%"}


RECIPES = """<table class="recipes">
 <tr><th class="result">Result</th><th class="ingredients">Ingredients</th><th class="station">Crafting station</th></tr>
 <tr><td class="result" rowspan="2"><span class="eico"><a title="Desktop version">only:</a></span>
   <span class="i"><a title="Wooden Arrow">Wooden Arrow</a></span><span class="am">25</span></td>
   <td class="ingredients"><ul><li><span class="i"><a title="Wood">Wood</a></span></li>
   <li><span class="i"><a title="Stone Block">Stone Block</a></span></li></ul></td>
   <td class="station" rowspan="2">Work Bench</td></tr>
 <tr><td class="ingredients"><ul><li><span class="i"><a title="Lens">Lens</a></span><span class="am">6</span></li></ul></td></tr>
</table>"""


def test_parse_recipes_with_rowspan_and_version_note():
    recipes = parse_page(RECIPES, "Wooden_Arrow")["recipes"]
    assert [(r["result"], r["amount"], r["station"]) for r in recipes] == [("Wooden Arrow", "25", "Work Bench")] * 2
    assert recipes[1]["ingredients"] == [{"item": "Lens", "amount": "6"}]  # second row inherits result + station


def test_station_from_listing_page():
    table = """<table class="recipes"><tr><th>Result</th><th>Ingredients</th></tr>
      <tr><td class="result"><span class="i"><a title="Chair">Chair</a></span></td>
      <td class="ingredients"><ul><li><span class="i"><a title="Wood">Wood</a></span><span class="am">4</span></li></ul></td></tr></table>"""
    assert parse_page(table, "Recipes%2FWork_Bench")["recipes"][0]["station"] == "Work Bench"


import json  # noqa: E402

from facts import FactBook, asked_mode  # noqa: E402

BOSS = {"name": "King Slime", "page": "King_Slime", "url": "u", "hardmode_only": False,
        "tags": {"Type": ["Boss"]}, "coins": {},
        "stats": {"Max Life": {"classic": "2000", "expert": "2800", "master": "3570"}, "Hurt": {}},
        "drops": [{"item": "Solidifier", "quantity": "1", "condition": None, "rate": {"classic": "100%"},
                   "group": None, "from_treasure_bag": False},
                  {"item": "Royal Gel", "quantity": "1", "condition": None, "rate": {"expert": "100%", "master": "100%"},
                   "group": None, "from_treasure_bag": True}]}
GUIDE_COPY = {**BOSS, "page": "Guide:King_Slime_strategies"}
RECIPE = {"result": "Suspicious Looking Eye", "amount": "1", "ingredients": [{"item": "Lens", "amount": "6"}],
          "station": "Demon Altar or Crimson Altar", "versions": None, "page": "Lens", "url": "u"}


@pytest.fixture
def book(tmp_path):
    path = tmp_path / "facts.json"
    path.write_text(json.dumps({"npcs": [BOSS, GUIDE_COPY], "recipes": [RECIPE]}))
    return FactBook(path)


@pytest.mark.parametrize("question, ids", [
    ("What does King Slime drop in Expert Mode?", ["facts:drops:King Slime"]),
    ("what does kingslime drop", ["facts:drops:King Slime"]),           # written as one word
    ("How much health does King Slime have?", ["facts:stats:King Slime"]),
    ("How do I make a Suspicious Looking Eye?", ["facts:recipe:Suspicious Looking Eye"]),
    ("How do I craft Suspicious Looking Eyes?", ["facts:recipe:Suspicious Looking Eye"]),  # plural
    ("How do I beat King Slime?", []),                                   # no drops/stats/crafting word
    ("What does the Wall of Flesh drop?", []),                           # no facts for it
])
def test_for_question(book, question, ids):
    assert [s["id"] for s in book.for_question(question)] == ids


def test_fact_text(book):
    drops = book.drops("King Slime", "expert")["text"]
    assert "Royal Gel 100% [in the Treasure Bag]" in drops and "Solidifier" not in drops  # Classic only
    assert drops.count("King Slime drops") == 1                    # the guide's copy of the infobox is skipped
    stats = book.stats("king slime")["text"]
    assert "2000 / 2800 / 3570 (Classic / Expert / Master)" in stats and "Hurt" not in stats
    assert "Lens ×6 at Demon Altar or Crimson Altar" in book.recipe("Suspicious Looking Eye")["text"]


@pytest.mark.parametrize("question, mode", [
    ("King Slime drops in Expert Mode", "expert"), ("master mode health", "master"),
    ("expert or master?", None), ("what does it drop", None),
])
def test_asked_mode(question, mode):
    assert asked_mode(question) == mode


from facts import parse_conditions  # noqa: E402

PAGE_MD = """# Nurse

Source: https://terraria.wiki.gg/wiki/Nurse
License: CC BY-NC-SA 4.0

Infobox: Nurse
[Statistics]
Type: NPC
Environment: Valid house
- Some drop: 100%

The Nurse is an NPC who will move in once the following criteria have been met:
- There is an available vacant house.
- The player's maximum health is greater than 100.

## Spawn conditions
- Only after the Merchant is present.

## Notes
- She will not move in if the house is in an evil biome.
- Healing costs 1 Copper per health.
"""


def test_parse_conditions():
    c = parse_conditions(PAGE_MD, "Nurse")
    assert "will move in once" in c["intro"] and "greater than 100" in c["intro"]  # the list after the intro
    assert "Some drop" not in c["intro"] and "Type: NPC" not in c["intro"]       # infobox lines left out
    assert [s["heading"] for s in c["sections"]] == ["Spawn conditions"]
    assert c["notes"] == ["- She will not move in if the house is in an evil biome."]  # healing tip left out


@pytest.mark.parametrize("question, wanted", [
    ("Where does King Slime spawn?", True),
    ("What comes after King Slime?", False),     # progression, not spawning
    ("How do I beat King Slime?", False),
])
def test_conditions_trigger(book, question, wanted):
    assert any(s["id"].startswith("facts:conditions") for s in book.for_question(question)) is wanted
