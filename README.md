# terraria-rag

Answer Terraria questions with RAG over the [Terraria Wiki](https://terraria.wiki.gg), using a local model.

- `download.py` downloads the wiki into `text/` as Markdown (`python3 download.py --limit 20` for a quick test).
- `extract.py` converts a wiki page's HTML to Markdown (`python3 extract.py samples/*.html -o /tmp/out`).

The code is licensed under [GPL-3.0](LICENSE). Wiki content, including `samples/`, is [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) by the Terraria Wiki contributors. It is used for retrieval only, not for training.
