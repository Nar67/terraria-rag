#!/usr/bin/env python3
"""Search the wiki chunks: vector search, BM25 keyword search, or both merged.

Usage:
    uv run search.py "how do I summon the Eye of Cthulhu"
    uv run search.py "Zenith recipe" -k 5 --full     # print whole chunks
    uv run search.py "when was Zenith added" --mode vector
    uv run search.py "what changed in 1.4.4"         # History chunks included

Needs the indexes built by index.py.

- Vector search embeds the question with bge-small-en-v1.5, with the
  instruction prefix bge expects on queries (chunks are embedded without it),
  and finds the chunks whose vectors are closest by cosine similarity.
- BM25 scores chunks by the question's words (stemmed, stopwords removed).
  It catches exact names like "Suspicious Looking Eye" that vectors can blur.
- Hybrid (the default) merges both rankings with reciprocal rank fusion:
  a chunk scores sum(1 / (60 + rank)) over the lists it appears in.
- Chunks tagged "history" (per-version changelogs) are left out unless the
  question looks like it is about versions or changes, or --history is given.
"""
import argparse
import json
import re
import sys

import bm25s
import chromadb
import numpy as np
import Stemmer
from chromadb.config import Settings
from fastembed import TextEmbedding

from config import BM25_DIR, BM25_META, CHROMA_DIR, COLLECTION, EMBED_MODEL, QUERY_PREFIX

RRF_K = 60  # the usual constant; larger values flatten the gap between ranks

# Questions about versions or changes are the ones History chunks can answer.
HISTORY_WORDS = re.compile(
    r"\b(\d+\.\d+(\.\d+)*|history|patch(es)?|updates?|versions?|changed?|changes|"
    r"added|introduced|removed|nerf(ed)?|buff(ed)?|previously|used to|when was|when were)\b",
    re.I,
)


def wants_history(query):
    return bool(HISTORY_WORDS.search(query))


def bm25_tokenize(texts):
    return bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=False)


def open_collection(create=False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    if create:
        return client.get_or_create_collection(
            COLLECTION, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )
    return client.get_collection(COLLECTION, embedding_function=None)


class Searcher:
    def __init__(self):
        if not (CHROMA_DIR.exists() and BM25_META.exists()):
            sys.exit("no index found; run: uv run index.py")
        self.collection = open_collection()
        meta = json.loads(BM25_META.read_text())
        vec_meta = self.collection.metadata or {}
        # Vectors from another model live in a different space: searching them
        # with this model's query vectors would return nonsense, so refuse.
        built_with = vec_meta.get("model")
        if built_with and built_with != EMBED_MODEL:
            sys.exit(f"the index was built with {built_with}, config says {EMBED_MODEL}; run: uv run index.py")
        if not vec_meta.get("build") or vec_meta.get("build") != meta.get("build"):
            print("warning: the vector and keyword indexes come from different builds "
                  "(index.py was interrupted?); results may miss chunks. Run: uv run index.py", file=sys.stderr)
        self.model = TextEmbedding(EMBED_MODEL)
        self.bm25 = bm25s.BM25.load(str(BM25_DIR))
        self.bm25_ids = meta["ids"]
        self.content_mask = np.array([k == "content" for k in meta["kinds"]], dtype=float)

    def vector_ranking(self, query, n, include_history):
        vec = next(iter(self.model.embed([QUERY_PREFIX + query]))).tolist()
        where = None if include_history else {"kind": "content"}
        res = self.collection.query(query_embeddings=[vec], n_results=n, where=where, include=[])
        return res["ids"][0]

    def bm25_ranking(self, query, n, include_history):
        tokens = bm25_tokenize([query])
        if not tokens.vocab:  # nothing but stopwords
            return []
        mask = None if include_history else self.content_mask
        n = min(n, len(self.bm25_ids))  # bm25s refuses a k larger than the corpus
        docs, scores = self.bm25.retrieve(tokens, k=n, show_progress=False, weight_mask=mask)
        return [self.bm25_ids[d] for d, s in zip(docs[0], scores[0]) if s > 0]

    def search(self, query, k=8, mode="hybrid", history=None, pool=50):
        """Top k chunks for a query, best first.

        history: True/False to include History chunks or not, None to decide
        from the question. pool: how deep each ranking goes before merging.
        """
        include_history = wants_history(query) if history is None else history
        rankings = {}
        if mode in ("hybrid", "vector"):
            rankings["vector"] = self.vector_ranking(query, pool, include_history)
        if mode in ("hybrid", "bm25"):
            rankings["bm25"] = self.bm25_ranking(query, pool, include_history)

        scores, ranks = {}, {}
        for name, ids in rankings.items():
            for rank, cid in enumerate(ids, 1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
                ranks.setdefault(cid, {})[name] = rank
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        if not best:
            return []

        got = self.collection.get(ids=best, include=["documents", "metadatas"])
        by_id = {cid: (doc, meta) for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])}
        results = []
        for cid in best:
            if cid not in by_id:  # in the keyword index but not (yet) in the vector store
                continue
            doc, meta = by_id[cid]
            results.append({
                "id": cid,
                "title": meta["title"],
                "sections": meta["sections"].split(" | "),
                "kind": meta["kind"],
                "url": meta["url"],
                "score": scores[cid],
                "ranks": ranks[cid],
                "text": doc,
            })
        return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="+")
    ap.add_argument("-k", type=int, default=8, help="number of results (default 8)")
    ap.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--history", dest="history", action="store_true", default=None,
                       help="always include History chunks")
    group.add_argument("--no-history", dest="history", action="store_false",
                       help="never include History chunks")
    ap.add_argument("--full", action="store_true", help="print whole chunks, not the first lines")
    args = ap.parse_args()

    query = " ".join(args.query)
    searcher = Searcher()
    include = wants_history(query) if args.history is None else args.history
    print(f"query: {query!r}  mode: {args.mode}  history chunks: {'included' if include else 'left out'}\n")
    for i, r in enumerate(searcher.search(query, k=args.k, mode=args.mode, history=args.history), 1):
        ranks = ", ".join(f"{name} #{rank}" for name, rank in r["ranks"].items())
        print(f"{i}. {r['id']}  [{r['kind']}]  score {r['score']:.4f}  ({ranks})")
        print(f"   {' | '.join(r['sections'])}")
        print(f"   {r['url']}")
        lines = r["text"].splitlines()[1:]  # the first line repeats the section path
        shown = lines if args.full else lines[:4]
        for line in shown:
            print(f"   > {line[:200]}")
        if len(shown) < len(lines):
            print(f"   > … ({len(lines) - len(shown)} more lines)")
        print()


if __name__ == "__main__":
    main()
