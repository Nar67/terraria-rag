#!/usr/bin/env python3
"""Extract the useful text from saved Terraria Wiki (terraria.wiki.gg) pages.

Usage:
    python3 extract.py page.html [more.html ...] [-o OUTDIR]

Writes one Markdown file per page: the title, the infobox as "key: value"
lines, headings, paragraphs, lists and tables (one row per line, cells
separated by " | "). Navigation boxes, images, the table of contents, edit
links, references and platform icons are dropped.
"""
import argparse
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

# Elements that carry no article content.
DROP_SELECTORS = [
    "script", "style", "noscript",
    ".ranger-navbox", ".navbox",            # navigation boxes at the bottom
    ".toc", "#toc",                         # table of contents
    ".mw-editsection",                      # [edit] links
    ".thumb", ".gallery", "figure",         # images and their captions
    ".reflist", "sup.reference", ".mw-references-wrap",
    ".hat-note",                            # "For other uses, see ..."
    ".message-box",                         # banners (stub, exclusive, ...)
    ".crafting-tree",                       # diagram that repeats the recipes
    ".eico",                                # platform icons: "(Desktop, Console ...)"
    ".mw-collapsible-toggle-placeholder",
    ".infobox .section.images",             # sprites
]

# Difficulty classes the wiki puts on values and drops that differ by game mode.
MODE_LABELS = {
    "m-normal": "Classic",
    "m-journey": "Journey",
    "m-expert": "Expert",
    "m-master": "Master",
    "m-expert-master": "Expert/Master",
    "m-normal-expert-master": "Classic/Expert/Master",
}

# Links to translated subpages ("Consumables/vi"); a block holding one is a
# stray translation on the English wiki.
LANGUAGE_SUBPAGE = re.compile(
    r"/(ar|bg|cs|da|de|el|es|fi|fr|hu|id|it|ja|ko|lt|nl|no|pl|pt|pt-br|ro|ru|sv|th|tr|uk|vi|zh|zh-hans|zh-hant)(\?|#|$)"
)

# Sections whose whole content is dropped (heading text, lower case).
DROP_SECTIONS = {"references", "footnotes", "gallery", "see also"}

BLOCK_TAGS = {"p", "ul", "ol", "dl", "table", "div", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre"}


def clean(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def inline_text(el):
    """Text of an element, with list items inside it joined by commas."""
    if isinstance(el, NavigableString):
        return str(el)
    parts = []
    for child in el.children:
        if isinstance(child, Tag) and child.name in ("ul", "ol"):
            items = [clean(inline_text(li)) for li in child.find_all("li", recursive=False)]
            parts.append(" " + ", ".join(i for i in items if i) + " ")
        elif isinstance(child, Tag) and child.name == "br":
            parts.append(" ")
        else:
            parts.append(inline_text(child))
    return re.sub(r"\s+", " ", "".join(parts))


def images_to_alt(root):
    """Replace an image by its alt text only when it is the only content of
    its cell (e.g. the rarity icon); otherwise the name is already written
    next to it and the image is dropped."""
    for img in root.find_all("img"):
        alt = (img.get("alt") or "").strip()
        holder = img.find_parent(["td", "th", "li", "p", "dd"])
        holder_text = holder.get_text(strip=True) if holder else ""
        if alt and holder is not None and not holder_text and "sprite" not in alt.lower():
            img.replace_with(alt)
        else:
            img.decompose()


def tidy_parens(text):
    """'Hat(when ... )' -> 'Hat (when ...)': spacing lost between adjacent spans."""
    text = re.sub(r"(\w)\(", r"\1 (", text)
    text = re.sub(r"\(\s+", "(", text)
    return re.sub(r"\s+([),.])", r"\1", text)


def mode_label(el):
    """'Classic', 'Expert/Master', ... from an element's m-* classes, or None."""
    classes = el.get("class") or []
    if "m-normal" in classes:  # "m-normal m-journey": Journey uses the Classic value
        return MODE_LABELS["m-normal"]
    return next((MODE_LABELS[c] for c in classes if c in MODE_LABELS), None)


def label_modes(root):
    """Write per-difficulty values out with their mode.

    The wiki shows one value per game mode as sibling spans (sometimes with a
    "/" span between them, sometimes with nothing), and its CSS hides all but
    one. As plain text they would run together ("145000217500277311"), so
    each becomes "Classic: 145000 / Expert: 217500 / Master: 277311"."""
    for box in root.select(".mode-content, .npcstat"):
        parts = []
        for child in box.find_all(recursive=False):
            label = mode_label(child)
            text = tidy_parens(clean(child.get_text(" ")))
            if label and text:
                parts.append(f"{label}: {text}")
        if parts:
            box.replace_with(" / ".join(parts))


def drop_stray_translations(root):
    for a in root.find_all("a", href=LANGUAGE_SUBPAGE):
        block = a.find_parent("div")
        if block is not None:
            block.decompose()


def drop_lines(ul):
    """An infobox drop list as '- Item (amount): rate (mode only)' lines."""
    lines = []
    for li in ul.find_all("li", recursive=False):
        classes = li.get("class") or []
        if li.find("strong"):  # the "Item (Quantity) | Rate" header
            continue
        if "caption" in classes:  # starts a group, e.g. "Only in Corrupt worlds"
            text = clean(li.get_text(" "))
            if text:
                lines.append(f"{text}:")
            continue
        cells = li.find_all("div", recursive=False)
        if not cells:
            continue
        item = tidy_parens(clean(inline_text(cells[0])))
        if not item:
            continue
        rate = clean(inline_text(cells[1])) if len(cells) > 1 else ""
        mode = mode_label(li)
        indent = "  " if "group" in classes else ""
        line = f"{indent}- {item}"
        if rate:
            line += f": {rate}"
        if mode:
            line += f" ({mode} only)"
        lines.append(line)
    return lines


def infobox_lines(box):
    lines = []
    title = box.find("div", class_="title", recursive=False)
    if title:
        lines.append(f"Infobox: {clean(title.get_text(' '))}")
    stack = box.select_one(".stack[title]")
    if stack:
        lines.append(stack["title"])
    for section in box.select(".section"):
        sec_title = section.find("div", class_="title", recursive=False)
        label = None
        if sec_title and sec_title.get_text(strip=True):
            label = f"[{clean(sec_title.get_text(' '))}]"
            sec_title.decompose()
        start = len(lines)
        for tr in section.select("tr"):
            th, td = tr.find("th"), tr.find("td")
            if th and td:
                tags = td.select(".tag")
                key = clean(inline_text(th))
                value = ", ".join(clean(t.get_text(" ")) for t in tags) if tags else clean(inline_text(td))
                if key and value:  # rows such as "Use:" hold only a sound player
                    lines.append(f"{key}: {value}")
            else:
                row = clean(inline_text(tr))
                if row:
                    lines.append(row)
        for li in section.select("ul.infobox-inline > li"):
            row = clean(inline_text(li))
            if row:
                lines.append(row)
        for ul in section.select("ul.drops"):
            lines.extend(drop_lines(ul))
        if label and len(lines) > start:  # no label for a section left empty
            lines.insert(start, label)
    return lines


def item_name(span):
    """Name of an item from the wiki's item span (span.i), without its amount."""
    span = span.__copy__()
    for am in span.select(".am"):
        am.decompose()
    return clean(span.get_text(" "))


def item_with_amount(el):
    """'Hallowed Bar ×12' from an element holding one item and an optional span.am."""
    item = el.select_one("span.i")
    name = item_name(item) if item else clean(el.get_text(" "))
    am = el.select_one(".am")
    amount = clean(am.get_text()) if am else ""
    return f"{name} ×{amount}" if amount else name


def version_note(td):
    note = td.select_one(".version-note")
    if not note:
        return ""
    versions = [a.get("title", "").replace(" version", "") for a in note.find_all("a") if a.get("title")]
    note.decompose()
    return f" ({', '.join(versions)} only)" if versions else ""


def is_recipe_table(table):
    """Crafting tables have Result and Ingredients headers; Shimmer tables
    share the 'recipes' class but use Output / Input."""
    first = table.find("tr")
    headers = {clean(th.get_text(" ")).lower() for th in first.find_all("th")} if first else set()
    return "result" in headers and "ingredients" in headers


def recipe_lines(table):
    """Recipe rows as 'Result ×n | Ingredient ×n, ... | Station or Station'.

    Shared cells (rowspan) are repeated on every row they cover, so each
    line is a complete recipe a script can read back."""
    lines = ["Result | Ingredients | Crafting station"]
    # Station pages leave the station column out and name it in the caption.
    caption = table.find("caption")
    default_station = clean(caption.get_text(" ")) if caption else ""
    carry = {}  # column class -> [text, rows left]
    for tr in table.find_all("tr"):
        if tr.find("th") and not tr.find("td"):
            continue
        row = {}
        for td in tr.find_all("td", recursive=False):
            col = next((c for c in ("result", "ingredients", "station") if c in (td.get("class") or [])), None)
            if col is None:
                continue
            if col == "result":
                note = version_note(td)
                text = item_with_amount(td) + note
            elif col == "ingredients":
                text = ", ".join(item_with_amount(li) for li in td.select("li"))
            else:
                stations = [item_name(s) for s in td.select("span.i")]
                text = " or ".join(stations) if stations else clean(td.get_text(" "))
            row[col] = text
            span = int(td.get("rowspan", 1) or 1)
            if span > 1:
                carry[col] = [text, span - 1]
        for col in ("result", "ingredients", "station"):
            if col not in row and col in carry and carry[col][1] > 0:
                row[col] = carry[col][0]
                carry[col][1] -= 1
        if not row.get("station") and default_station:
            row["station"] = default_station
        if row.get("result") or row.get("ingredients"):
            lines.append(" | ".join(row.get(c, "") for c in ("result", "ingredients", "station")))
    return lines


def table_lines(table):
    lines = []
    for tr in table.find_all("tr"):
        cells = [clean(inline_text(c)) for c in tr.find_all(["th", "td"], recursive=False)]
        cells = [c for c in cells if c]
        if cells and cells[0] == "#":  # a numbering column; keep rows from looking like headings
            cells[0] = "No."
        if cells:
            lines.append(" | ".join(cells))
    return lines


def list_lines(lst, depth=0):
    lines = []
    for li in lst.find_all(["li", "dt", "dd"], recursive=False):
        nested = li.find_all(["ul", "ol", "dl"], recursive=False)
        tables = li.find_all("table")
        for n in nested + tables:
            n.extract()
        text = clean(inline_text(li))
        if text:
            lines.append("  " * depth + "- " + text)
        for t in tables:  # a table inside a list item keeps its rows
            lines.extend("  " * (depth + 1) + row for row in table_lines(t))
        for n in nested:
            lines.extend(list_lines(n, depth + 1))
    return lines


def walk(container, out, state):
    for el in container.children:
        if isinstance(el, NavigableString):
            text = clean(str(el))
            if text and not state["skip"]:
                out.append(text)
            continue
        if not isinstance(el, Tag):
            continue
        name = el.name
        if name in ("h2", "h3", "h4", "h5", "h6"):
            heading = clean(el.get_text(" "))
            level = int(name[1])
            if level == 2:
                state["skip"] = heading.lower() in DROP_SECTIONS
            if not state["skip"]:
                out.append("\n" + "#" * level + " " + heading)
            continue
        if state["skip"]:
            continue
        classes = el.get("class") or []
        if "infobox" in classes:
            out.extend(infobox_lines(el))
            out.append("")
        elif name == "table" and "recipes" in classes and is_recipe_table(el):
            out.extend(recipe_lines(el))
            out.append("")
        elif name == "table":
            out.extend(table_lines(el))
            out.append("")
        elif name in ("ul", "ol", "dl"):
            out.extend(list_lines(el))
        elif name == "p":
            text = clean(inline_text(el))
            if text:
                out.append(text + "\n")
        elif name == "div" and any(c.name in BLOCK_TAGS for c in el.find_all(recursive=False)):
            walk(el, out, state)
        else:
            text = clean(inline_text(el))
            if text:
                out.append(text)


BULLET_ONLY = re.compile(r"^[•·∙●\s]+$")


def tidy(lines):
    """Drop bullet-only lines and headings whose section ended up empty."""
    lines = [l for l in lines if not BULLET_ONLY.match(l)]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    rows = text.split("\n")
    kept = []
    for i, row in enumerate(rows):
        m = re.match(r"^(#{2,6}) ", row)
        if m:
            level = len(m.group(1))
            nxt = next((r for r in rows[i + 1:] if r.strip()), None)
            nm = re.match(r"^(#{2,6}) ", nxt) if nxt else None
            if nxt is None or (nm and len(nm.group(1)) <= level):
                continue  # nothing under this heading before the next one
        kept.append(row)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept))


def extract(html):
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.select_one("#firstHeading")
    title = clean(title_el.get_text(" ")) if title_el else ""
    content = soup.select_one("div.mw-parser-output")
    if content is None:
        return title, ""
    for selector in DROP_SELECTORS:
        for el in content.select(selector):
            el.decompose()
    for comment in content.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    drop_stray_translations(content)
    label_modes(content)
    images_to_alt(content)
    out = [f"# {title}", ""]
    walk(content, out, {"skip": False})
    text = tidy("\n".join(out).split("\n"))
    return title, text.strip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("-o", "--outdir", type=Path, default=Path("text"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for f in args.files:
        title, text = extract(f.read_text(encoding="utf-8"))
        out = args.outdir / (f.stem + ".md")
        out.write_text(text, encoding="utf-8")
        print(f"{f.name}: {f.stat().st_size:,} B html -> {len(text.encode()):,} B text ({title})", file=sys.stderr)


if __name__ == "__main__":
    main()
