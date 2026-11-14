from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from extract import drop_lines, extract, tidy_parens

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def page(body):
    """A minimal wiki page around some article HTML."""
    return (f'<html><body><h1 id="firstHeading">Test page</h1>'
            f'<div class="mw-parser-output">{body}</div></body></html>')


def item(name):
    """The wiki's item link: icon plus name (span.i)."""
    return f'<span class="i"><a href="/wiki/{name}"><img alt="{name}"/></a><span><span>{name}</span></span></span>'


def icon(name):
    """An item link that shows only its icon, like debuff immunities."""
    return f'<span class="i"><a href="/wiki/{name}"><img alt="{name}"/></a></span>'


@pytest.mark.parametrize("html, expected", [
    # item links side by side used to run together ("Star CannonPhoenix Blaster")
    (f"<p>{item('Star Cannon')}{item('Phoenix Blaster')}</p>", "Star Cannon, Phoenix Blaster"),
    # per-difficulty values used to run together ("280036404641")
    ('<p>Max Life: <span class="mode-content"><span class="m-normal">2800</span>'
     '<span class="m-expert">3640</span><span class="m-master">4641</span></span></p>',
     "Max Life: Classic: 2800 / Expert: 3640 / Master: 4641"),
    # several icon-only items: only the first one used to keep its name
    (f"<p>Immune to: {icon('Bleeding')}{icon('Confused')}</p>", "Immune to: Bleeding, Confused"),
    # a numbering column must not make a row look like a Markdown heading
    ("<table><tr><th>#</th><th>Hand</th></tr><tr><td>1</td><td>Left</td></tr></table>", "No. | Hand"),
])
def test_extract_fixes(html, expected):
    assert expected in extract(page(html))[1]


def test_stray_translation_is_dropped():
    html = ('<p>Obtain a boss-summoning item.</p>'
            '<div>Có được <a href="/wiki/Consumables/vi">vật phẩm</a></div>')
    text = extract(page(html))[1]
    assert "Obtain a boss-summoning item." in text
    assert "Có được" not in text


@pytest.mark.parametrize("raw, tidy", [
    ("Badger's Hat(when defeated )", "Badger's Hat (when defeated)"),
    ("Treasure Bag ( Eye of Cthulhu )", "Treasure Bag (Eye of Cthulhu)"),
    ("Already (fine)", "Already (fine)"),
])
def test_tidy_parens(raw, tidy):
    assert tidy_parens(raw) == tidy


def test_drop_lines():
    ul = BeautifulSoup(
        '<ul class="drops items">'
        '<li><strong>Item</strong><strong>Rate</strong></li>'
        '<li class="groupstart caption">Only in Corrupt worlds</li>'
        '<li class="m-normal group"><div>Demonite Ore (30–90)</div><div>100%</div></li>'
        '<li class="m-expert-master"><div>Shield of Cthulhu</div><div>100%</div></li>'
        '</ul>', "lxml").ul
    assert drop_lines(ul) == [
        "Only in Corrupt worlds:",
        "  - Demonite Ore (30–90): 100% (Classic only)",
        "- Shield of Cthulhu: 100% (Expert/Master only)",
    ]


def test_sample_page_keeps_recipe():
    """Regression check on a real saved page: the Zenith recipe line survives intact."""
    title, text = extract((SAMPLES / "Zenith.html").read_text(encoding="utf-8"))
    assert title == "Zenith"
    assert ("Zenith | Terra Blade, Meowmere, Star Wrath, Influx Waver, The Horseman's Blade, Seedler, "
            "Starfury, Bee Keeper, Enchanted Sword, Copper Shortsword | Mythril Anvil or Orichalcum Anvil") in text
