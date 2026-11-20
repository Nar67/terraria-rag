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

from config import HERE, INDEX_DIR, TEXT_DIR, add_logging_args, setup_logging
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


# Conditions (phase 3): what makes something spawn, appear, move in or start.
# Unlike stats and drops this is prose, spread over a page: the intro, a few
# sections, the transcluded "special spawning restrictions" list and some
# Notes bullets. They are read from the extracted text (text/*.md) and put
# together in one place, so the model gets one checklist-like source instead
# of picking the pieces out of several chunks.
CONDITION_HEADING = re.compile(r"spawn|summon(?!.*weapon)|condition|criteria|requirement|occurr|arriv|trigger|growing", re.I)
CONDITION_WORDS = re.compile(r"\b(only|not|never|must|requires?|cannot|can't|won't|unless|after|before|once|"
                             r"chance|until|at least|if)\b", re.I)
ABOUT_SPAWNING = re.compile(r"spawn|appear|move in|moves in|arriv|summon|occur|generat|lands?\b|start|trigger", re.I)
MAX_INTRO, MAX_SECTION, MAX_NOTES = 900, 1200, 8


def parse_conditions(md, page):
    """Condition text of one page, or None if it has none worth keeping."""
    lines = md.splitlines()
    heads = [(i, len(m.group(1)), m.group(2)) for i, l in enumerate(lines) if (m := re.match(r"(#{2,4}) (.*)", l))]
    first_head = heads[0][0] if heads else len(lines)

    # intro: the prose before the first heading, minus title, source and infobox lines
    intro, in_infobox = [], False
    for line in lines[:first_head]:
        if not line.strip():
            in_infobox = False
            continue
        if line.startswith(("# ", "Source:", "License:", "“")) or line.startswith("Infobox:"):
            in_infobox = line.startswith("Infobox:") or in_infobox
            continue
        if in_infobox or line.startswith("[") or re.match(r"[\w ()'/]+: ", line):
            continue
        intro.append(line.strip())  # including lists ("will move in once ...: - ...")
    intro_text = "\n".join(intro)[:MAX_INTRO]

    def body(start, level):
        end = next((i for i, lv, _ in heads if i > start and lv <= level), len(lines))
        return "\n".join(l for l in lines[start + 1:end] if l.strip())

    sections = [{"heading": h, "text": body(i, lv)[:MAX_SECTION]}
                for i, lv, h in heads if CONDITION_HEADING.search(h) and not h.lower().startswith("despawn")]
    special, notes = [], []
    for i, line in enumerate(lines):
        if re.search(r"special spawning restrictions|can only spawn if", line, re.I):
            special = [line.strip()] + [l.rstrip() for l in lines[i + 1:i + 14] if l.startswith("  ")]
            break
    for i, lv, h in heads:
        if h.lower() == "notes":
            notes = [l.strip() for l in body(i, lv).splitlines()
                     if l.startswith("- ") and CONDITION_WORDS.search(l) and ABOUT_SPAWNING.search(l)][:MAX_NOTES]
    if not (sections or special or notes or intro_text):
        return None
    return {"page": page, "url": page_url(page), "intro": intro_text, "sections": sections,
            "special_rules": "\n".join(special), "notes": notes}


# Bosses and progression (phase 4). The Bosses page lists them by stage in
# progression order; each boss page says when fighting it "is not required
# for game progression" and has an Aftermath section with what the first kill
# unlocks (the three mechanical bosses share theirs on "Mechanical bosses").
OPTIONAL = re.compile(r"[^.\n]*\bnot required for (?:game )?progression[^.\n]*\.", re.I)
MECHANICAL = {"The Twins", "The Destroyer", "Skeletron Prime"}
MAX_AFTERMATH = 26  # lines; the mechanical bosses list "one" and "all three defeated"


def parse_bosses(text_dir=TEXT_DIR):
    """[{"name", "stage", "order", "optional", "optional_quote", "aftermath", "page", "url"}]"""
    path = text_dir / "Bosses.md"
    if not path.exists():
        return []
    bosses, stage = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if m := re.match(r"## (.*)", line):
            stage = m.group(1) if "bosses" in m.group(1).lower() or "boss" in m.group(1).lower() else None
        elif (m := re.match(r"### (.*)", line)) and stage:
            bosses.append({"name": m.group(1).strip(), "stage": stage})
    for n, boss in enumerate(bosses):
        page = boss["name"].replace(" ", "_")
        md_path = text_dir / f"{page}.md"
        md = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        optional = OPTIONAL.search(md)
        aftermath_page = "Mechanical_bosses" if boss["name"] in MECHANICAL else page
        aftermath_md = (text_dir / f"{aftermath_page}.md")
        boss.update({
            "order": n, "page": page, "url": page_url(page),
            "optional": bool(optional), "optional_quote": optional.group(0).strip() if optional else None,
            "aftermath": _aftermath(aftermath_md.read_text(encoding="utf-8")) if aftermath_md.exists() else [],
        })
    return bosses


def _aftermath(md):
    """The Aftermath section's lines (bullets and sub-headings), or []."""
    lines = md.splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(r"## Aftermath", l)), None)
    if start is None:
        return []
    out = []
    for line in lines[start + 1:]:
        if re.match(r"## ", line):
            break
        if line.strip():
            out.append(re.sub(r"^#{3,4} ", "", line.rstrip()))
    return out[:MAX_AFTERMATH]


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
    # conditions for pages with an NPC infobox (creatures, town NPCs, bosses)
    # or with a condition-like section (events, Plantera's Bulb, ...)
    npc_pages = {n["page"] for n in facts["npcs"]}
    facts["conditions"] = {}
    for path in sorted(TEXT_DIR.glob("*.md")):
        page = path.stem
        md = path.read_text(encoding="utf-8")
        if page in npc_pages or any(CONDITION_HEADING.search(h) for h in re.findall(r"(?m)^#{2,4} (.*)", md)):
            entry = parse_conditions(md, page)
            if entry:
                facts["conditions"][page] = entry
    seen["pages with conditions"] = len(facts["conditions"])
    facts["bosses"] = parse_bosses()
    seen["bosses"] = len(facts["bosses"])
    FACTS.parent.mkdir(parents=True, exist_ok=True)
    tmp = FACTS.with_suffix(".json.part")
    tmp.write_text(json.dumps(facts, ensure_ascii=False), encoding="utf-8")
    tmp.replace(FACTS)
    log.info(f"{len(facts['npcs'])} NPC infoboxes, {len(facts['recipes'])} recipes from {len(files)} pages -> {FACTS}")
    for key, value in seen.items():
        log.info(f"  {key}: {value}")


# ---------------------------------------------------------------------------
# Lookups: facts.json -> small, already-resolved sources for the model.
# A fact source looks like a search result ({"id", "title", "sections",
# "kind": "facts", "url", "text"}), so it is numbered and cited like a chunk.

ASKS_DROPS = re.compile(r"\b(drops?|dropped|loot|treasure bag)\b", re.I)
ASKS_STATS = re.compile(r"\b(health|life|hp|damage|defen[cs]e|stats?|knockback)\b", re.I)
ASKS_RECIPE = re.compile(r"\b(craft\w*|recipes?|make|made|ingredients?)\b", re.I)
ASKS_CONDITIONS = re.compile(r"\b(spawn\w*|appear\w*|shows? up|show up|mov\w* in|arriv\w*|summon\w*|trigger\w*|"
                             r"start\w*|occur\w*|happen\w*)\b", re.I)
MAX_CONDITIONS = 2500  # characters of one conditions source
# "what comes / happens after X" is progression, not how X spawns
ASKS_BOSS_LIST = re.compile(r"\b(which|what|list|every|optional|required)\b[^.?]*\bbosses\b|\bbosses\b[^.?]*\b(optional|required|order|list)\b", re.I)
# "after beating X", "I've beaten X", "do I have to fight X to progress"
ASKS_PROGRESSION = re.compile(r"\bafter (beating|defeating|killing)\b|(?<!do )\b(i|we)('ve| have)? (beat|beaten|killed|defeated)\b|"
                              r"\b(have to|need to|must)\b[^.?]*\b(fight|beat|kill|defeat)\b|\bto progress\b", re.I)
# "... but no Plantera's Bulb spawns, why?" is troubleshooting, not progression
ASKS_WHY = re.compile(r"\bwhy\b|n't\b|\bdoes not\b|\bwill not\b", re.I)
ASKS_WHATS_NEXT = re.compile(r"\b(what (comes|happens|now|next)|next|after (i|you) (kill|beat|defeat))", re.I)
# page names that are also everyday words of the questions themselves
GENERIC_NAMES = {"spawn", "summon", "craft", "crafting", "drop", "drops", "loot", "recipe", "recipes",
                 "health", "life", "damage", "defense", "luck", "fishing", "house", "time"}
MAX_RECIPES = 6  # some items have many alternative recipes
MODE_WORDS = {"classic": r"\b(classic|normal)\b", "expert": r"\bexpert\b", "master": r"\bmaster\b"}


def asked_mode(question):
    """The difficulty a question asks about, if exactly one is named."""
    found = [m for m, pattern in MODE_WORDS.items() if re.search(pattern, question, re.I)]
    return found[0] if len(found) == 1 else None


class FactBook:
    """Look up NPC stats, drops and recipes in index/facts.json."""

    def __init__(self, path=FACTS):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run `uv run facts.py` first")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.npcs, self.recipes = {}, {}
        for npc in data["npcs"]:  # by page (all forms of a boss) and by infobox name
            if npc["page"].startswith("Guide:"):  # strategy guides repeat the boss infobox
                continue
            for name in {npc["page"].replace("_", " "), npc["name"]}:
                self.npcs.setdefault(name.lower(), []).append(npc)
        for recipe in data["recipes"]:
            self.recipes.setdefault(recipe["result"].lower(), []).append(recipe)
        self.conditions_by = {page.replace("_", " ").lower(): c for page, c in data.get("conditions", {}).items()}
        self.bosses = data.get("bosses", [])
        self.boss_by = {b["name"].lower(): b for b in self.bosses}
        if "the twins" in self.boss_by:  # the three share one page and one Aftermath
            self.boss_by.setdefault("mechanical bosses", {**self.boss_by["the twins"], "name": "Mechanical bosses",
                                                         "page": "Mechanical_bosses", "url": page_url("Mechanical_bosses")})
        names = set(self.npcs) | set(self.recipes)
        # also written as one word ("moonlord", "kingslime"), as players often do
        for table in (self.npcs, self.recipes, self.conditions_by):
            for name in list(table):
                if " " in name and len(name) > 8:
                    table.setdefault(name.replace(" ", ""), table[name])
        names = (set(self.npcs) | set(self.recipes) | set(self.conditions_by) | set(self.boss_by)) - GENERIC_NAMES
        # longest first, so "Eye of Cthulhu" wins over "The Eye of Cthulhu" parts etc.
        self.names = sorted(names, key=len, reverse=True)

    def names_in(self, question):
        """Known NPC/item names in the question, longest match first, no overlaps."""
        q, found, taken = question.lower(), [], []
        for name in self.names:
            # plural allowed: "Hellstone Bars", "Greater Healing Potions"
            for m in re.finditer(rf"(?<![\w']){re.escape(name)}(?:e?s)?(?![\w])", q):
                if not any(a < m.end() and m.start() < b for a, b in taken):
                    found.append(name)
                    taken.append(m.span())
                    break
        return found

    def drops(self, name, mode=None):
        boxes = [b for b in self.npcs.get(name.lower(), []) if b["drops"]]
        if not boxes:
            return None
        lines = []
        for box in _distinct(boxes):
            lines.append(f"{box['name']} drops ({_modes_label(mode)}):")
            for m in [mode] if mode else MODES:
                lines.append(f"- {m.capitalize()}: {_drop_list(box, m)}")
        return _source(boxes[0], "drops", "\n".join(lines))

    def stats(self, name, mode=None):
        boxes = [b for b in self.npcs.get(name.lower(), []) if b["stats"] or b["tags"]]
        if not boxes:
            return None
        lines = []
        for box in _distinct(boxes):
            tags = "; ".join(f"{k}: {', '.join(v)}" for k, v in box["tags"].items())
            head = f"{box['name']}{' (Hardmode only)' if box['hardmode_only'] else ''}"
            lines.append(f"{head}: {tags}" if tags else head)
            for key, values in box["stats"].items():
                if any(values.values()):  # sound/immunity rows are icons only
                    lines.append(f"- {key}: {_per_mode(values, mode)}")
        return _source(boxes[0], "stats", "\n".join(lines))

    def recipe(self, name):
        recipes = self.recipes.get(name.lower(), [])
        if not recipes:
            return None
        lines = [f"Recipes for {recipes[0]['result']}:"]
        for r in recipes[:MAX_RECIPES]:
            ingredients = ", ".join(f"{i['item']} ×{i['amount']}" for i in r["ingredients"])
            made = f"{r['result']} ×{r['amount']}" if r["amount"] != "1" else r["result"]
            where = r["station"] or "station not listed"
            note = f" ({r['versions']})" if r["versions"] else ""
            lines.append(f"- {made} = {ingredients} at {where}{note}")
        if len(recipes) > MAX_RECIPES:
            lines.append(f"- ... and {len(recipes) - MAX_RECIPES} more recipes")
        return _source(recipes[0], "recipe", "\n".join(lines), subject=recipes[0]["result"])

    def conditions(self, name):
        """What makes it spawn, appear, move in or start: infobox environment
        and Hardmode flag, the page intro, condition sections, the special
        spawning rules and condition-like Notes, in one source."""
        cond = self.conditions_by.get(name.lower())
        boxes = _distinct(self.npcs.get(name.lower(), []))
        if cond is None and not boxes:
            return None
        lines = []
        for box in boxes[:3]:  # forms and parts share their environment; a few are enough
            tags = "; ".join(f"{k}: {', '.join(v)}" for k, v in box["tags"].items())
            lines.append(f"{box['name']}{' (Hardmode only)' if box['hardmode_only'] else ''}: {tags}")
        if cond:
            if cond["intro"]:
                lines.append(cond["intro"])
            for section in cond["sections"]:
                lines.append(f"{section['heading']}:\n{section['text']}")
            if cond["special_rules"]:
                lines.append(f"Special spawning rules:\n{cond['special_rules']}")
            notes = [n for n in cond["notes"] if not cond["special_rules"] or n[2:40] not in cond["special_rules"]]
            if notes:
                lines.append("Notes:\n" + "\n".join(notes))
        entry = cond or boxes[0]
        return _source(entry, "conditions", "\n".join(lines)[:MAX_CONDITIONS])

    def boss_list(self, stage=None):
        """The bosses by stage in the wiki's progression order, each marked
        optional (quoting its page) or not. stage: "pre-hardmode",
        "hardmode", "event" or None for all."""
        if not self.bosses:
            return None
        lines = ["Bosses on the wiki's Bosses page, in progression order. \"optional\" = its page says "
                 "fighting it is not required for game progression; the others are on the main path."]
        for stage_name in dict.fromkeys(b["stage"] for b in self.bosses):
            if stage and not _stage_matches(stage_name, stage):
                continue
            items = []
            for b in (x for x in self.bosses if x["stage"] == stage_name):
                if "event" in stage_name.lower() or "seed" in stage_name.lower():
                    items.append(f"- {b['name']}")
                elif b["optional"]:
                    items.append(f"- {b['name']}: optional (\"{b['optional_quote']}\")")
                else:
                    items.append(f"- {b['name']}: on the main path")
            lines.append(f"{stage_name}:\n" + "\n".join(items))
        src = _source({"page": "Bosses", "url": page_url("Bosses")}, "boss list", "\n".join(lines))
        if stage:
            src["id"] += f" ({stage})"
        return src

    def progression(self, name):
        """What defeating a boss unlocks (its Aftermath) and which bosses come
        next on the wiki's list, marked optional or on the main path."""
        boss = self.boss_by.get(name.lower())
        if boss is None or "event" in boss["stage"].lower():
            return None
        status = "optional" if boss["optional"] else "on the main path"
        lines = [f"{boss['name']} ({boss['stage'].removesuffix(' bosses')}, {status})."]
        if boss["aftermath"]:
            lines.append("After the first kill (Aftermath):\n" + "\n".join(boss["aftermath"]))
        main = [b for b in self.bosses if "event" not in b["stage"].lower() and "seed" not in b["stage"].lower()]
        later = [b for b in main if b["order"] > boss["order"]]
        upcoming, required = [], 0
        for b in later:  # up to and including the next two bosses on the main path
            upcoming.append(f"- {b['name']} ({b['stage'].removesuffix(' bosses')}, "
                            f"{'optional' if b['optional'] else 'on the main path'})")
            required += not b["optional"] and b["name"] not in MECHANICAL | {"Brain of Cthulhu"}
            if required >= 2:
                break
        if upcoming:
            lines.append("Next on the wiki's boss list, in order:\n" + "\n".join(upcoming))
        return _source(boss, "progression", "\n".join(lines))

    def lookup(self, kind, name, mode=None):
        """One tool call: kind is "drops", "stats", "recipe", "conditions",
        "progression" or "bosses" (name = stage or "all")."""
        if kind == "bosses":
            return self.boss_list(None if name.lower() in ("", "all") else name)
        if kind in ("recipe", "conditions", "progression"):
            return getattr(self, kind)(name)
        return getattr(self, kind)(name, mode)

    def for_question(self, question):
        """Fact sources a question clearly asks for, without any model: a
        known name plus a drops / stats / crafting keyword."""
        mode, sources = asked_mode(question), []
        if ASKS_BOSS_LIST.search(question):
            q = question.lower()
            stage = "pre-hardmode" if re.search(r"pre[- ]?hard ?mode", q) else "hardmode" if "hardmode" in q.replace(" ", "") else None
            sources.append(self.boss_list(stage))
        if (ASKS_WHATS_NEXT.search(question) or ASKS_PROGRESSION.search(question)) and not ASKS_WHY.search(question):
            for name in self.names_in(question):
                sources.append(self.progression(name))
        for name in self.names_in(question):
            if ASKS_DROPS.search(question):
                sources.append(self.drops(name, mode))
            if ASKS_STATS.search(question):
                sources.append(self.stats(name, mode))
            if ASKS_RECIPE.search(question):
                sources.append(self.recipe(name))
            if ASKS_CONDITIONS.search(question) and not ASKS_WHATS_NEXT.search(question):
                sources.append(self.conditions(name))
        unique = {s["id"]: s for s in sources if s}
        return list(unique.values())


def open_factbook(enabled=True):
    """The FactBook, or None if switched off or not built yet (with a warning)."""
    if not enabled:
        return None
    try:
        return FactBook()
    except FileNotFoundError as e:
        log.warning(f"{e}; answering without fact sources")
        return None


def _stage_matches(stage_name, stage):
    name, stage = stage_name.lower(), stage.lower().replace(" ", "")
    if stage.startswith("pre"):
        return name.startswith("pre")
    if stage == "hardmode":
        return name.startswith("hardmode")
    return stage in name.replace(" ", "")


def _distinct(boxes):
    """Each infobox once (a box is indexed under its page and its own name)."""
    seen, out = set(), []
    for b in boxes:
        key = (b["page"], b["name"])
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out


def _modes_label(mode):
    return f"{mode.capitalize()} Mode" if mode else "per difficulty"


def _per_mode(values, mode):
    if mode:
        return values.get(mode, "?")
    distinct = {values.get(m) for m in MODES}
    if len(distinct) == 1:
        return values.get("classic") or next(iter(values.values()), "?")
    return " / ".join(values.get(m, "-") for m in MODES) + " (Classic / Expert / Master)"


def _drop_list(box, mode):
    parts, group = [], None
    for d in box["drops"]:
        rate = d["rate"].get(mode)
        if rate is None:
            continue
        item = d["item"] + (f" ({d['quantity']})" if d["quantity"] != "1" else "")
        item += f" {rate}"
        if d["condition"]:
            item += f" [{d['condition']}]"
        if d["from_treasure_bag"]:
            item += " [in the Treasure Bag]"
        if d["group"] and d["group"] != group:
            parts.append(f"{d['group']}: {item}")
        else:
            parts.append(item)
        group = d["group"]
    return "; ".join(parts) or "nothing"


def _source(entry, kind, body, subject=None):
    """A fact source. subject: what it is about (a recipe's result), if not the
    page it was read from ("Recipes/Work Bench" lists many results)."""
    page = entry["page"].replace("_", " ").replace("%2F", "/")
    subject = subject or page
    return {"id": f"facts:{kind}:{subject}", "title": page, "sections": [page, f"{subject}: {kind} (from the wiki)"],
            "kind": "facts", "url": entry["url"], "text": f"{subject} > {kind}\n{body}"}


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
