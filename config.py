"""Settings shared by the pipeline scripts.

The embedding model is named here only: chunk.py sizes chunks with its
tokenizer and index.py/search.py embed with it, so they must agree. The index
records the model it was built with, and search.py refuses an index built
with a different one.
"""
import logging
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

# Logging: every script logs status to stderr through loggers named "rag.*";
# results (search hits, answers, --show output) are printed to stdout, so they
# can be piped or saved without the status lines.
#   default  info and above from our scripts, warnings from libraries
#   -v       also our debug messages, and info from libraries
#   -q       only warnings and errors
NOISY_LIBRARIES = ("httpx", "httpcore", "urllib3", "huggingface_hub", "chromadb", "fastembed", "openai", "bm25s")


class _Formatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"{record.levelname.lower()}: {message}"
        return message


def add_logging_args(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", help="show debug messages too")
    group.add_argument("-q", "--quiet", action="store_true", help="show only warnings and errors")


def setup_logging(args):
    verbose, quiet = getattr(args, "verbose", False), getattr(args, "quiet", False)
    handler = logging.StreamHandler()  # stderr
    handler.setFormatter(_Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO if verbose else logging.WARNING)
    logging.getLogger("rag").setLevel(logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO)
    if not verbose:
        for name in NOISY_LIBRARIES:
            logging.getLogger(name).setLevel(logging.ERROR)
        try:  # "Fetching 5 files" bars from model downloads
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except ImportError:
            pass
