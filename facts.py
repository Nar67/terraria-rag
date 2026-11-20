#!/usr/bin/env python3
"""Structured facts from the saved wiki HTML: NPC stats, drops and recipes.

Usage:
    uv run facts.py                      # build index/facts.json from html/
    uv run facts.py --page King_Slime    # print what one page parses into

Why: the slowest and most error-prone answers are the ones where the model
has to untangle tables with a column per difficulty (drops, health) or
several recipes (DEVLOG §30-31). The wiki's HTML marks all of that
explicitly, so code can read it exactly: each value carries the modes it
applies to (classes m-normal / m-expert / m-master / m-all and combinations
like m-expert-master). Tools built on these facts (phase 2) hand the model
small, already-resolved answers instead.

What is parsed (DEVLOG §31, phase 0 coverage):
- NPC infoboxes (`.infobox.npc`; several per page for forms and parts):
  name, Hardmode-only flag, type and environment tags, stats per mode
  (`span.npcstat`), drops (`ul.drops.items`: item, quantity, condition,
  rate per mode, "one of the following" groups, Treasure Bag contents),
  coin drops.
- Recipe tables (`table.recipes`): result and amount, ingredients and
  amounts, crafting station, version note. Cells spanning several rows
  (alternative recipes) are carried over.
Never the flattened text: "5% 9.75%" in one cell is Classic / Expert.
"""
import argparse
import gzip
import json
import logging
import re
import sys
from collections import Counter

from bs4 import BeautifulSoup

from config import HERE, INDEX_DIR, add_logging_args, setup_logging
from download import page_url

log = logging.getLogger("rag.facts")

HTML_DIR = HERE / "html"
FACTS = INDEX_DIR / "facts.json"
MODES = ("classic", "expert", "master")


def text(el):
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip() if el is not None else ""


def modes_of(classes):
    """The difficulty modes a wiki mode class applies to, or an empty set.

    m-normal = Classic; m-journey is ignored (Journey follows Classic here);
    m-all = every mode; "m-expert-master" = Expert and Master."""
    found = set()
    for c in classes or []:
        if c == "m-all":
            return set(MODES)
        if c.startswith("m-"):
            for part in c[2:].split("-"):
                if part == "normal":
                    found.add("classic")
                elif part in ("expert", "master"):
                    found.add(part)
    return found


def mode_values(cell):
    """{mode: text} for a cell whose value may differ per difficulty.

    Uses the outermost elements that carry a mode class; a cell without any
    has the same value in every mode."""
    values = {}
    for el in cell.find_all(True):
        modes = modes_of(el.get("class"))
        if not modes:
            continue
        if any(modes_of(p.get("class")) for p in el.parents if p is not cell and cell in p.parents):
            continue  # inside an outer mode element: already counted
        value = text(el)
        for m in modes:
            values.setdefault(m, value)
    if not values:
        value = text(cell)
        return {m: value for m in MODES} if value else {}
    return values


def link_name(el):
    """The item an item cell names ("Solidifier"), else its text.

    The item's own link sits in span.i; a version note ("only: Desktop
    version") may link elsewhere first."""
    a = el.select_one("span.i a[title]") or next(
        (a for a in el.find_all("a", title=True) if not a.find_parent(class_="eico")), None)
    return a["title"] if a else text(el)


def parse_stats(box):
    stats, tags = {}, {}
    for row in box.select("table.stat tr"):
        th, td = row.find("th"), row.find("td")
        if th is None or td is None:
            continue
        key = text(th)
        found_tags = [text(t) for t in td.select(".tags .tag")]
        if found_tags:
            tags[key] = found_tags
        else:  # mode spans may sit in span.npcstat or directly in the cell
            stats[key] = mode_values(td)
    return stats, tags


def parse_drops(box):
    """Item drops of one NPC infobox; [] if it has none."""
    drops, group = [], None
    for li in box.select("ul.drops.items > li"):
        classes = li.get("class") or []
        if li.find("strong"):  # the "Item (Quantity) | Rate" header
            continue
        if "caption" in classes:
            group = text(li)
            continue
        if "groupend" in classes:
            group = None
            continue
        if "lootstart" in classes or "lootend" in classes:
            continue
        cells = li.find_all("div", recursive=False)
        if not cells:
            continue
        label = text(cells[0])
        name = link_name(cells[0])
        extra = label[len(name):].strip() if label.startswith(name) else label
        quantity = re.match(r"\(([\d,]+(?:\s*[–-]\s*[\d,]+)?)\)", extra)
        condition = extra[quantity.end():].strip() if quantity else extra
        modes = modes_of(classes) or set(MODES)
        rates = mode_values(cells[1]) if len(cells) > 1 else {}
        drops.append({
            "item": name,
            "quantity": quantity.group(1) if quantity else "1",
            "condition": condition.strip("() ") or None,
            "rate": {m: rates.get(m) or next(iter(rates.values()), None) for m in MODES if m in modes},
            "group": group if "group" in classes else None,
            "from_treasure_bag": "loot" in classes,
        })
    return drops


def parse_npcs(soup, page):
    npcs = []
    for box in soup.select(".infobox.npc"):
        title = box.select_one(".title")
        name = text(title)
        hardmode = "Hardmode exclusive" in name
        name = name.replace("Hardmode exclusive", "").strip()
        stats, tags = parse_stats(box)
        coins = box.select_one("table.drops.money td")
        npcs.append({
            "name": name, "page": page, "url": page_url(page), "hardmode_only": hardmode,
            "tags": tags, "stats": stats, "drops": parse_drops(box),
            "coins": mode_values(coins) if coins else {},
        })
    return npcs


def amount(el):
    am = el.select_one(".am")
    return text(am) if am else "1"


def parse_recipes(soup, page):
    recipes = []
    for table in soup.select("table.recipes"):
        carried = {}  # column class -> (cell, rows left) for rowspan cells
        for row in table.select("tr"):
            cells = {next(iter(td.get("class") or ["?"])): td for td in row.find_all("td")}
            for col, (cell, left) in list(carried.items()):
                if col not in cells:
                    cells[col] = cell
                    carried[col] = (cell, left - 1)
                if carried[col][1] <= 0:
                    del carried[col]
            for col, cell in list(cells.items()):
                span = int(cell.get("rowspan") or 1)
                if span > 1 and col not in carried:
                    carried[col] = (cell, span - 1)
            if "result" not in cells or "ingredients" not in cells:
                continue
            result, station = cells["result"], cells.get("station")
            version = result.select_one(".eico")
            # "Recipes/Work Bench"-style listing pages have no station column:
            # the page itself is the station.
            station_name = text(station) if station is not None else \
                (page.removeprefix("Recipes%2F").replace("_", " ") if page.startswith("Recipes%2F") else "")
            recipes.append({
                "result": link_name(result), "amount": amount(result),
                "ingredients": [{"item": link_name(li), "amount": amount(li)}
                                for li in cells["ingredients"].select("li")],
                "station": station_name,
                "versions": version.get("title") if version else None,
                "page": page, "url": page_url(page),
            })
    return recipes


def parse_page(html, page):
    soup = BeautifulSoup(html, "lxml")
    for sup in soup.select("sup.reference"):  # [1] footnote markers
        sup.decompose()
    return {"npcs": parse_npcs(soup, page), "recipes": parse_recipes(soup, page)}


def build():
    files = sorted(HTML_DIR.glob("*.html.gz"))
    if not files:
        sys.exit(f"no saved HTML in {HTML_DIR} (download.py --keep-html)")
    facts, recipes = {"npcs": []}, {}
    seen = Counter()
    for n, path in enumerate(files, 1):
        page = path.name.removesuffix(".html.gz")
        parsed = parse_page(gzip.open(path, "rt", encoding="utf-8", errors="replace").read(), page)
        facts["npcs"] += parsed["npcs"]
        for r in parsed["recipes"]:
            # A recipe is listed on its result's page, on each ingredient's
            # "used in" table and on station pages: keep one copy, preferring
            # the result's own page (the best citation) and a known station.
            key = (r["result"], r["amount"], tuple((i["item"], i["amount"]) for i in r["ingredients"]))
            old = recipes.get(key)
            if old is None or (not old["station"] and r["station"]) or \
                    (r["page"].replace("_", " ") == r["result"] and old["page"].replace("_", " ") != r["result"]):
                recipes[key] = r if r["station"] or old is None else {**r, "station": old["station"]}
        seen["pages with NPC infobox"] += bool(parsed["npcs"])
        seen["pages with drops"] += any(n_["drops"] for n_ in parsed["npcs"])
        seen["pages with stats"] += any(n_["stats"] for n_ in parsed["npcs"])
        seen["pages with recipes"] += bool(parsed["recipes"])
        if n % 500 == 0:
            log.info(f"{n}/{len(files)} pages")
    facts["recipes"] = list(recipes.values())
    FACTS.parent.mkdir(parents=True, exist_ok=True)
    tmp = FACTS.with_suffix(".json.part")
    tmp.write_text(json.dumps(facts, ensure_ascii=False), encoding="utf-8")
    tmp.replace(FACTS)
    log.info(f"{len(facts['npcs'])} NPC infoboxes, {len(facts['recipes'])} recipes from {len(files)} pages -> {FACTS}")
    for key, value in seen.items():
        log.info(f"  {key}: {value}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", help="print the facts of one saved page (file name without .html.gz)")
    add_logging_args(ap)
    args = ap.parse_args()
    setup_logging(args)
    if args.page:
        path = HTML_DIR / f"{args.page}.html.gz"
        print(json.dumps(parse_page(gzip.open(path, "rt", encoding="utf-8").read(), args.page), ensure_ascii=False, indent=1))
    else:
        build()


if __name__ == "__main__":
    main()
