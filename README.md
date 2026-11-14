# terraria-rag

Answer Terraria questions with RAG over the [Terraria Wiki](https://terraria.wiki.gg), using a local model.

- `download.py` downloads the wiki into `text/` as Markdown (`uv run download.py --limit 20` for a quick test).
- `extract.py` converts a wiki page's HTML to Markdown (`uv run extract.py samples/*.html -o /tmp/out`).
- `chunk.py` splits the pages into chunks of at most 512 bge tokens, written to `chunks.jsonl` (`uv run chunk.py`).
- `index.py` embeds the chunks with bge-small into ChromaDB and builds a BM25 index, in `index/` (`uv run index.py`).
- `search.py` searches them: vector, BM25 or hybrid (`uv run search.py "how do I summon the Eye of Cthulhu"`).

The code is licensed under [GPL-3.0](LICENSE). Wiki content, including `samples/`, is [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) by the Terraria Wiki contributors. It is used for retrieval only, not for training.
