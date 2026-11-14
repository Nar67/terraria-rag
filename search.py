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
import logging
import re
import sys

import bm25s
import chromadb
import numpy as np
import Stemmer
from chromadb.config import Settings
from fastembed import TextEmbedding

from config import (BM25_DIR, BM25_META, CHROMA_DIR, COLLECTION, EMBED_MODEL, QUERY_PREFIX,
                    add_logging_args, setup_logging)

log = logging.getLogger("rag.search")

RRF_K = 60  # the usual constant; larger values flatten the gap between ranks

# Search tuning, chosen with eval.py on eval/questions.jsonl (2026-11-14):
# hybrid k=8 went from 93% to 100% of questions with all evidence retrieved
# and MRR 0.69 -> 0.83. Trivia down-weighting made no difference, so it is off.
TUNING = {"pool": 100, "vector_weight": 2.0, "title_boost": 0.5, "trivia_weight": 1.0}

# Questions about versions or changes are the ones History chunks can answer.
HISTORY_WORDS = re.compile(
    r"\b(\d+\.\d+(\.\d+)*|history|patch(es)?|updates?|versions?|changed?|changes|"
    r"added|introduced|removed|nerf(ed)?|buff(ed)?|previously|used to|when was|when were)\b",
    re.I,
)


def normalize(text):
    """Lower case, letters and digits only: "Night's Edge" -> "night s edge"."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def titles_in_query(query, titles):
    """The titles that appear as whole words in the query, longest only.

    For "how do I summon the Eye of Cthulhu" both "Eye of Cthulhu" and the
    disambiguation page "Eye" appear; only the longer one counts."""
    q = f" {normalize(query)} "
    found = [t for t in titles if normalize(t) and f" {normalize(t)} " in q]
    return {t for t in found
            if not any(o != t and f" {normalize(t)} " in f" {normalize(o)} " for o in found)}


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
            log.warning("the vector and keyword indexes come from different builds "
                        "(index.py was interrupted?); results may miss chunks. Run: uv run index.py")
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

    def search(self, query, k=8, mode="hybrid", history=None, pool=None,
               vector_weight=None, title_boost=None, trivia_weight=None):
        """Top k chunks for a query, best first.

        history: True/False to include History chunks or not, None to decide
        from the question. The tuning knobs default to the TUNING values:
        pool           how deep each ranking goes before merging
        vector_weight  weight of the vector ranking in the merge (BM25 has 1)
        title_boost    extra score, in "ranked first" units, for chunks of a
                       page whose title is in the question
        trivia_weight  score factor for chunks that are only Trivia
        """
        pool = TUNING["pool"] if pool is None else pool
        vector_weight = TUNING["vector_weight"] if vector_weight is None else vector_weight
        title_boost = TUNING["title_boost"] if title_boost is None else title_boost
        trivia_weight = TUNING["trivia_weight"] if trivia_weight is None else trivia_weight
        include_history = wants_history(query) if history is None else history
        rankings = {}
        if mode in ("hybrid", "vector"):
            rankings["vector"] = self.vector_ranking(query, pool, include_history)
        if mode in ("hybrid", "bm25"):
            rankings["bm25"] = self.bm25_ranking(query, pool, include_history)
        weights = {"vector": vector_weight, "bm25": 1.0}

        scores, ranks = {}, {}
        for name, ids in rankings.items():
            for rank, cid in enumerate(ids, 1):
                scores[cid] = scores.get(cid, 0.0) + weights[name] / (RRF_K + rank)
                ranks.setdefault(cid, {})[name] = rank
        if not scores:
            return []

        # Metadata for every candidate: the adjustments need titles and sections.
        got = self.collection.get(ids=list(scores), include=["documents", "metadatas"])
        by_id = {cid: (doc, meta) for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])}
        scores = {cid: s for cid, s in scores.items() if cid in by_id}  # skip ids missing from the vector store
        if title_boost:
            boosted = titles_in_query(query, {by_id[cid][1]["title"] for cid in scores})
            for cid in scores:
                if by_id[cid][1]["title"] in boosted:
                    scores[cid] += title_boost / (RRF_K + 1)
        if trivia_weight != 1.0:
            for cid in scores:
                sections = by_id[cid][1]["sections"].split(" | ")
                if all(sec.rsplit(" > ", 1)[-1] == "Trivia" for sec in sections):
                    scores[cid] *= trivia_weight

        results = []
        for cid in sorted(scores, key=scores.get, reverse=True)[:k]:
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
    add_logging_args(ap)
    args = ap.parse_args()
    setup_logging(args)

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
