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
import argparse
import hashlib
import json
import logging
import shutil
import sys
import time

import bm25s
from fastembed import TextEmbedding

from config import BM25_DIR, BM25_META, CHUNKS, EMBED_MODEL, add_logging_args, setup_logging
from search import bm25_tokenize, open_collection

BATCH = 256  # chunks embedded and stored per step

log = logging.getLogger("rag.index")


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


def build_vectors(records, build):
    collection = open_collection(create=True)
    # A fresh build stamp only once every vector below is stored.
    collection.modify(metadata={"model": EMBED_MODEL, "build": "incomplete"})
    stored = collection.get(include=["metadatas"])
    stored_hash = {cid: m.get("hash") for cid, m in zip(stored["ids"], stored["metadatas"])}

    wanted = {r["id"] for r in records}
    stale = [cid for cid in stored_hash if cid not in wanted]
    for i in range(0, len(stale), 5000):
        collection.delete(ids=stale[i:i + 5000])

    todo = [r for r in records if stored_hash.get(r["id"]) != text_hash(r["text"])]
    log.info(f"vectors: {len(records) - len(todo)} chunks up to date, {len(todo)} to embed, "
             f"{len(stale)} removed")
    if todo:
        embed(collection, todo)
    collection.modify(metadata={"model": EMBED_MODEL, "build": build})


def embed(collection, todo):
    # Similar lengths in a batch mean less padding, so sort by length.
    todo.sort(key=lambda r: r["tokens"])
    model = TextEmbedding(EMBED_MODEL)
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
        log.info(f"{done}/{len(todo)} embedded, {rate:.0f}/s, about {(len(todo) - done) / rate / 60:.0f} min left")


def build_bm25(records, build):
    retriever = bm25s.BM25()
    retriever.index(bm25_tokenize([r["text"] for r in records]), show_progress=False)
    # Save to a temporary folder and swap it in, so an interruption never
    # leaves a half-written index behind.
    tmp = BM25_DIR.with_name(BM25_DIR.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    retriever.save(str(tmp))
    (tmp / BM25_META.name).write_text(json.dumps({
        "build": build,
        "ids": [r["id"] for r in records],
        "kinds": [r["kind"] for r in records],
    }))
    if BM25_DIR.exists():
        shutil.rmtree(BM25_DIR)
    tmp.rename(BM25_DIR)
    log.info(f"bm25: indexed {len(records)} chunks")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_logging_args(ap)
    setup_logging(ap.parse_args())
    if not CHUNKS.exists():
        sys.exit("chunks.jsonl not found; run: uv run chunk.py")
    data = CHUNKS.read_bytes()
    records = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        sys.exit("chunks.jsonl has duplicate ids")
    # Both indexes get this stamp; search.py warns when they differ.
    build = hashlib.sha1(EMBED_MODEL.encode() + b"\0" + data).hexdigest()[:16]
    build_vectors(records, build)  # slow and resumable, so first
    build_bm25(records, build)
    log.info(f"done (build {build})")


if __name__ == "__main__":
    main()
