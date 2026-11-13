#!/usr/bin/env python3
"""Split the wiki pages in text/ into chunks for retrieval.

Usage:
    uv run chunk.py                          # text/*.md -> chunks.jsonl
    uv run chunk.py text/Zenith.md --show    # print one page's chunks instead

Every chunk fits the embedding model: counted with the bge-small-en-v1.5
tokenizer, [CLS] and [SEP] included, it is at most --max-tokens (512) long,
so the model never truncates it.

How a page is split:
- Each chunk starts with its context path, e.g. "Zenith > Crafting > Recipes",
  so a section still says what it is about on its own.
- A page, or a section, that fits whole stays one chunk. One that does not is
  split at its subheadings, then at blank lines, then at line breaks, then at
  sentence ends, and as a last resort at token boundaries.
- A table or infobox split into pieces repeats its first line (the column
  header, or "Infobox: <name>") in every piece.
- Neighbouring chunks of the same page are merged while one of them is under
  --min-tokens and the result still fits.
- "History" sections (the per-version changelog) are chunked on their own,
  never merged with other text, and tagged "kind": "history"; every other
  chunk is "kind": "content". Their "Now does X" / "Previously Y" lines read
  as current facts out of context, so search should leave them out unless
  the question is about versions or changes.
"""
import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tokenizers import Tokenizer

MODEL = "BAAI/bge-small-en-v1.5"
HERE = Path(__file__).resolve().parent

HEADING = re.compile(r"^(#{2,6})\s+(.*\S)\s*$")
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
BLANK_LINES = re.compile(r"\n\s*\n")

_tok = Tokenizer.from_pretrained(MODEL)
_tok.no_truncation()
_tok.no_padding()


def ntok(text):
    """Token count as the model sees it, including [CLS] and [SEP]."""
    return len(_tok.encode(text).ids)


@dataclass
class Node:
    path: list  # ["Zenith", "Crafting", "Recipes"]
    body: list = field(default_factory=list)  # lines before the first child
    children: list = field(default_factory=list)


@dataclass
class Chunk:
    sections: list  # context paths of the parts, e.g. ["Zenith > Notes"]
    text: str


def parse(md, fallback_title):
    """Turn a page into a tree of headings. Returns (root node, source url)."""
    lines = md.splitlines()
    title, url = fallback_title, ""
    if lines and lines[0].startswith("# "):
        title = lines.pop(0)[2:].strip()
    # The Source and License lines sit right under the title; keep them out of the text.
    while lines and (not lines[0].strip() or lines[0].startswith(("Source: ", "License: "))):
        line = lines.pop(0)
        if line.startswith("Source: "):
            url = line[len("Source: "):].strip()

    root = Node([title])
    stack = [(1, root)]
    for line in lines:
        m = HEADING.match(line)
        if m:
            level = len(m.group(1))
            while stack[-1][0] >= level:
                stack.pop()
            node = Node(stack[-1][1].path + [m.group(2)])
            stack[-1][1].children.append(node)
            stack.append((level, node))
        else:
            stack[-1][1].body.append(line)
    return root, url


def body_text(node):
    return BLANK_LINES.sub("\n\n", "\n".join(node.body)).strip()


def render(node):
    """The node's whole text, its subsections included under their headings."""
    parts = [body_text(node)]
    for child in node.children:
        inner = render(child)
        if inner:
            parts.append("#" * len(child.path) + " " + child.path[-1] + "\n" + inner)
    return "\n\n".join(p for p in parts if p)


def compose(header, body):
    return header + "\n" + body


def fits(header, body, limit):
    return ntok(compose(header, body)) <= limit


def pack(header, units, sep, limit):
    """Greedily join units (each of which fits) into as few pieces as fit."""
    pieces, current = [], []
    for unit in units:
        if current and not fits(header, sep.join(current + [unit]), limit):
            pieces.append(sep.join(current))
            current = []
        current.append(unit)
    if current:
        pieces.append(sep.join(current))
    return pieces


def hard_split(header, text, limit):
    """Cut text at token boundaries into pieces that fit. The last resort."""
    budget = limit - ntok(compose(header, ""))
    if budget < 16:
        raise ValueError(f"context path too long to leave room for text: {header!r}")
    offsets = _tok.encode(text, add_special_tokens=False).offsets
    pieces, i = [], 0
    while i < len(offsets):
        j = min(i + budget, len(offsets))
        while True:
            piece = text[offsets[i][0]:offsets[j - 1][1]].strip()
            # Re-tokenizing a slice can differ slightly at its edges, so check again.
            if fits(header, piece, limit) or j - i == 1:
                break
            j -= max(1, (j - i) // 20)
        if piece:
            pieces.append(piece)
        i = j
    return pieces


def split_line(header, line, limit):
    """Split one over-long line into pieces that fit: sentences first, then tokens."""
    if fits(header, line, limit):
        return [line]
    sentences = SENTENCE_END.split(line)
    if len(sentences) == 1:
        return hard_split(header, line, limit)
    units = []
    for sentence in sentences:
        units += split_line(header, sentence, limit)
    return pack(header, units, " ", limit)


def is_table(lines):
    return len(lines) > 1 and " | " in lines[0] and sum(" | " in l for l in lines) > 1


def split_block(header, block, limit):
    """Split one paragraph, list, table or infobox into pieces that fit."""
    if fits(header, block, limit):
        return [block]
    lines = block.split("\n")
    if len(lines) == 1:
        return split_line(header, block, limit)
    repeat = lines[0] if is_table(lines) or lines[0].startswith("Infobox:") else None
    if repeat and not fits(compose(header, repeat), "x" * 50, limit):
        repeat = None  # a header row that long leaves no room; split it like any text
    if repeat:
        row_header = compose(header, repeat)
        units = [u for line in lines[1:] for u in split_line(row_header, line, limit)]
        return [compose(repeat, p) for p in pack(row_header, units, "\n", limit)]
    units = [u for line in lines for u in split_line(header, line, limit)]
    return pack(header, units, "\n", limit)


def split_body(header, text, limit):
    units = [u for block in BLANK_LINES.split(text) for u in split_block(header, block, limit)]
    return pack(header, units, "\n\n", limit)


def chunk_node(node, limit):
    """Chunks for a node: whole if it fits, else its own text and each child apart."""
    header = " > ".join(node.path)
    whole = render(node)
    if not whole:
        return []
    if fits(header, whole, limit):
        return [Chunk([header], compose(header, whole))]
    chunks = []
    own = body_text(node)
    if own:
        chunks += [Chunk([header], compose(header, p)) for p in split_body(header, own, limit)]
    for child in node.children:
        chunks += chunk_node(child, limit)
    return chunks


def merge_small(chunks, limit, minimum):
    """Merge neighbours while one of them is small and the result still fits."""
    merged = []
    for chunk in chunks:
        if merged:
            prev = merged[-1]
            small = ntok(prev.text) < minimum or ntok(chunk.text) < minimum
            text = prev.text + "\n\n" + chunk.text
            if small and ntok(text) <= limit:
                merged[-1] = Chunk(prev.sections + chunk.sections, text)
                continue
        merged.append(chunk)
    return merged


def pull_sections(node, name):
    """Remove the subsections called `name` from the tree and return them."""
    pulled = [c for c in node.children if c.path[-1] == name]
    node.children = [c for c in node.children if c.path[-1] != name]
    for child in node.children:
        pulled += pull_sections(child, name)
    return pulled


def chunk_page(path, limit, minimum):
    root, url = parse(path.read_text(encoding="utf-8"), path.stem)
    history = pull_sections(root, "History")
    content_chunks = merge_small(chunk_node(root, limit), limit, minimum)
    history_chunks = merge_small([c for h in history for c in chunk_node(h, limit)], limit, minimum)
    tagged = [("content", c) for c in content_chunks] + [("history", c) for c in history_chunks]
    title = root.path[0]
    records = []
    for i, (kind, chunk) in enumerate(tagged):
        tokens = ntok(chunk.text)
        if tokens > limit:  # cannot happen if the splitting above is right
            raise AssertionError(f"{path.name} chunk {i} has {tokens} tokens")
        records.append({
            "id": f"{title}#{i}",
            "title": title,
            "kind": kind,
            "sections": chunk.sections,
            "url": url,
            "file": str(path.relative_to(HERE)) if path.is_relative_to(HERE) else str(path),
            "tokens": tokens,
            "text": chunk.text,
        })
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="pages to chunk (default: text/*.md)")
    ap.add_argument("-o", "--output", type=Path, default=HERE / "chunks.jsonl")
    ap.add_argument("--max-tokens", type=int, default=512, help="hard limit per chunk (default 512)")
    ap.add_argument("--min-tokens", type=int, default=100, help="merge chunks smaller than this (default 100)")
    ap.add_argument("--show", action="store_true", help="print the chunks instead of writing them")
    args = ap.parse_args()

    files = args.files or sorted((HERE / "text").glob("*.md"))
    if not files:
        sys.exit("no pages found; run download.py first")

    records = []
    for n, path in enumerate(files, 1):
        records += chunk_page(path.resolve(), args.max_tokens, args.min_tokens)
        if not args.show and n % 500 == 0:
            print(f"{n}/{len(files)} pages, {len(records)} chunks", file=sys.stderr)

    if args.show:
        for r in records:
            print(f"===== {r['id']} [{r['kind']}]  ({r['tokens']} tokens)  {' | '.join(r['sections'])}")
            print(r["text"])
            print()
    else:
        with args.output.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(records)} chunks from {len(files)} pages to {args.output.name}")

    sizes = sorted(r["tokens"] for r in records)
    print(
        f"tokens per chunk: min {sizes[0]}, median {statistics.median(sizes):.0f}, "
        f"p90 {sizes[int(len(sizes) * 0.9)]}, max {sizes[-1]} (limit {args.max_tokens})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
