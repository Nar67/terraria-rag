"""Settings shared by the pipeline scripts.

The embedding model is named here only: chunk.py sizes chunks with its
tokenizer and index.py/search.py embed with it, so they must agree. The index
records the model it was built with, and search.py refuses an index built
with a different one.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

TEXT_DIR = HERE / "text"
CHUNKS = HERE / "chunks.jsonl"
INDEX_DIR = HERE / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_DIR = INDEX_DIR / "bm25"
BM25_META = BM25_DIR / "chunk_ids.json"
COLLECTION = "chunks"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_MAX_TOKENS = 512  # the model's limit, [CLS] and [SEP] included
# bge-*-en-v1.5 is trained with this prefix on queries only (not on passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
