#!/usr/bin/env python3
"""Build the search indexes from chunks.jsonl (made by chunk.py).

Usage:
    uv run index.py

Writes two indexes under index/:
- index/chroma: every chunk embedded with bge-small-en-v1.5 (384 numbers per
  chunk, no query prefix on passages) in a ChromaDB collection, with its title,
  kind, sections and url as metadata. Cosine distance.
- index/bm25: a BM25 keyword index over the same chunk texts.

Embedding runs on the CPU and is the slow part, about 15 minutes for the whole
wiki the first time. It is incremental: a chunk whose text is unchanged since
the last run is not embedded again, chunks that no longer exist are deleted,
and an interrupted run keeps the batches it finished. The BM25 index is
rebuilt in full each time; that takes seconds.
"""
import hashlib
import json
import sys
import time

import bm25s
from fastembed import TextEmbedding

from search import BM25_DIR, HERE, MODEL, bm25_tokenize, open_collection

CHUNKS = HERE / "chunks.jsonl"
BATCH = 256  # chunks embedded and stored per step


def text_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def metadata(r):
    # Chroma metadata values must be scalars, so the section list is joined.
    return {
        "title": r["title"],
        "kind": r["kind"],
        "sections": " | ".join(r["sections"]),
        "url": r["url"],
        "file": r["file"],
        "tokens": r["tokens"],
        "hash": text_hash(r["text"]),
    }


def build_vectors(records):
    collection = open_collection(create=True)
    stored = collection.get(include=["metadatas"])
    stored_hash = {cid: m.get("hash") for cid, m in zip(stored["ids"], stored["metadatas"])}

    wanted = {r["id"] for r in records}
    stale = [cid for cid in stored_hash if cid not in wanted]
    for i in range(0, len(stale), 5000):
        collection.delete(ids=stale[i:i + 5000])

    todo = [r for r in records if stored_hash.get(r["id"]) != text_hash(r["text"])]
    print(f"vectors: {len(records) - len(todo)} chunks up to date, {len(todo)} to embed, "
          f"{len(stale)} removed", file=sys.stderr)
    if not todo:
        return

    # Similar lengths in a batch mean less padding, so sort by length.
    todo.sort(key=lambda r: r["tokens"])
    model = TextEmbedding(MODEL)
    start = time.time()
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        vectors = [v.tolist() for v in model.embed([r["text"] for r in batch], batch_size=32)]
        collection.upsert(
            ids=[r["id"] for r in batch],
            embeddings=vectors,
            documents=[r["text"] for r in batch],
            metadatas=[metadata(r) for r in batch],
        )
        done = i + len(batch)
        rate = done / (time.time() - start)
        print(f"  {done}/{len(todo)} embedded, {rate:.0f}/s, about {(len(todo) - done) / rate / 60:.0f} min left",
              file=sys.stderr)


def build_bm25(records):
    retriever = bm25s.BM25()
    retriever.index(bm25_tokenize([r["text"] for r in records]), show_progress=False)
    retriever.save(str(BM25_DIR))
    (BM25_DIR / "chunk_ids.json").write_text(json.dumps({
        "ids": [r["id"] for r in records],
        "kinds": [r["kind"] for r in records],
    }))
    print(f"bm25: indexed {len(records)} chunks", file=sys.stderr)


def main():
    if not CHUNKS.exists():
        sys.exit("chunks.jsonl not found; run: uv run chunk.py")
    records = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        sys.exit("chunks.jsonl has duplicate ids")
    build_bm25(records)
    build_vectors(records)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
