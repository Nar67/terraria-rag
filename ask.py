#!/usr/bin/env python3
"""Answer Terraria questions from the wiki with the local model.

Usage:
    uv run ask.py "how do I summon the Eye of Cthulhu"
    uv run ask.py                                  # ask several questions in a row
    uv run ask.py "Zenith recipe" --show-sources   # also print the retrieved chunks
    uv run ask.py "Zenith recipe" --dry-run        # print the prompt, call nothing
    uv run ask.py "best pre-hardmode armor" --think

How it works: search.py finds the top chunks (hybrid search), they go into the
prompt numbered [1], [2], ..., and the model answers from them only, citing
the numbers. The cited wiki pages are listed under the answer.

The model is any OpenAI-compatible chat server:
- RAG_LLM_URL    server address (default http://localhost:8000/v1)
- RAG_LLM_MODEL  model name (default: the first model the server lists)
- RAG_LLM_KEY    API key, if the server needs one

--think / --no-think switch reasoning on or off on servers whose chat
template supports it (Qwen3 and similar, via "chat_template_kwargs"). Without
either flag nothing extra is sent and the server's default applies.
"""
import argparse
import os
import re
import sys

from openai import OpenAI

from search import Searcher, wants_history

LLM_URL = os.environ.get("RAG_LLM_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL")  # None: ask the server


def pick_model(client):
    if LLM_MODEL:
        return LLM_MODEL
    models = client.models.list().data
    if not models:
        sys.exit(f"the server at {LLM_URL} lists no models; set RAG_LLM_MODEL")
    return models[0].id

SYSTEM_PROMPT = """You answer questions about the game Terraria using excerpts from the Terraria Wiki.

Rules:
- Use only the numbered sources given with the question. Do not add facts from memory.
- If the sources do not contain the answer, say that the wiki excerpts don't cover it, and stop. Do not guess.
- Cite the sources you used with their numbers in square brackets, like [2] or [1][4], right after the fact they support.
- Sources marked "history" are changelogs of past game versions. Use them only for questions about versions or changes, and never present an old value as the current one.
- Values often differ by difficulty (Classic / Expert / Master) or platform; say so when it matters.
- Be direct and short. Use a list for recipes, steps and drops."""

DIM, RESET = "\033[2m", "\033[0m"


def format_sources(results):
    parts = []
    for n, r in enumerate(results, 1):
        tag = " (history)" if r["kind"] == "history" else ""
        # The chunk text starts with its section path, which names the page.
        parts.append(f"[{n}]{tag} {r['url']}\n{r['text']}")
    return "\n\n".join(parts)


def build_messages(question, results):
    user = f"Sources:\n\n{format_sources(results)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def cited_numbers(answer, count):
    nums = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return sorted(n for n in nums if 1 <= n <= count)


class ThinkFilter:
    """Split streamed text into reasoning (inside <think>...</think>) and answer.

    Streamed pieces can end in the middle of a tag ("...</th" + "ink>..."),
    so up to a tag's length of text is held back until it can be decided.
    """
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self, on_reasoning, on_answer):
        self.on_reasoning, self.on_answer = on_reasoning, on_answer
        self.buf, self.inside = "", False

    def feed(self, text):
        self.buf += text
        while True:
            tag = self.CLOSE if self.inside else self.OPEN
            emit = self.on_reasoning if self.inside else self.on_answer
            i = self.buf.find(tag)
            if i >= 0:
                emit(self.buf[:i])
                self.buf, self.inside = self.buf[i + len(tag):], not self.inside
                continue
            # Keep a tail that could be the start of the tag; emit the rest.
            keep = next((n for n in range(min(len(tag) - 1, len(self.buf)), 0, -1)
                         if tag.startswith(self.buf[-n:])), 0)
            emit(self.buf[:len(self.buf) - keep])
            self.buf = self.buf[len(self.buf) - keep:]
            return

    def flush(self):
        (self.on_reasoning if self.inside else self.on_answer)(self.buf)
        self.buf = ""


def stream_answer(client, model, messages, think, temperature):
    """Print the answer as it arrives and return its text.

    think: True/False to switch reasoning on/off through the chat template,
    None to send nothing and leave it to the server.
    """
    extra = {} if think is None else {"chat_template_kwargs": {"enable_thinking": think}}
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
        extra_body=extra or None,
    )
    answer = []

    def show_reasoning(text):
        if text:
            print(f"{DIM}{text}{RESET}", end="", flush=True)

    def show_answer(text):
        if text:
            print(text, end="", flush=True)
            answer.append(text)

    # Some servers send reasoning in a separate field, others inline between
    # <think> tags; either way it is shown dimmed and kept out of the answer.
    tags = ThinkFilter(show_reasoning, show_answer)
    for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta
        show_reasoning(getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None))
        tags.feed(delta.content or "")
    tags.flush()
    print()
    return "".join(answer).strip()


def ask(question, searcher, client, model, args):
    results = searcher.search(question, k=args.k, mode=args.mode, history=args.history)
    if args.show_sources or args.dry_run:
        include = wants_history(question) if args.history is None else args.history
        print(f"{DIM}retrieved {len(results)} chunks (history {'included' if include else 'left out'}):{RESET}")
        for n, r in enumerate(results, 1):
            print(f"{DIM}  [{n}] {r['id']} [{r['kind']}]  {' | '.join(r['sections'])}{RESET}")
        print()
    if not results:
        print("No matching wiki excerpts found.")
        return
    messages = build_messages(question, results)
    if args.dry_run:
        for m in messages:
            print(f"----- {m['role']} -----\n{m['content']}\n")
        return

    answer = stream_answer(client, model, messages, args.think, args.temperature)
    cited = cited_numbers(answer, len(results))
    print(f"\n{DIM}Sources (Terraria Wiki, CC BY-NC-SA 4.0):{RESET}")
    shown = cited or range(1, len(results) + 1)
    seen = set()
    for n in shown:
        r = results[n - 1]
        label = " | ".join(r["sections"])
        if (r["url"], label) in seen:
            continue
        seen.add((r["url"], label))
        print(f"{DIM}  [{n}] {label} — {r['url']}{RESET}")
    if not cited:
        print(f"{DIM}  (the answer cited no source; all retrieved chunks are listed){RESET}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="*", help="leave out to ask questions one after another")
    ap.add_argument("-k", type=int, default=8, help="chunks given to the model (default 8)")
    ap.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--history", dest="history", action="store_true", default=None,
                       help="always include History chunks")
    group.add_argument("--no-history", dest="history", action="store_false",
                       help="never include History chunks")
    think = ap.add_mutually_exclusive_group()
    think.add_argument("--think", dest="think", action="store_true", default=None,
                       help="ask the model to reason before answering (slower)")
    think.add_argument("--no-think", dest="think", action="store_false",
                       help="ask the model not to reason")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--show-sources", action="store_true", help="list the retrieved chunks before the answer")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt and stop; no model needed")
    args = ap.parse_args()

    searcher = Searcher()
    client = OpenAI(base_url=LLM_URL, api_key=os.environ.get("RAG_LLM_KEY", "none"))

    model = None

    def run(question):
        nonlocal model
        try:
            if model is None and not args.dry_run:
                model = pick_model(client)
            ask(question, searcher, client, model, args)
        except Exception as e:  # e.g. the model server is not running
            name = type(e).__name__
            if "Connection" in name:
                sys.exit(f"cannot reach the model at {LLM_URL}; is the server running?")
            raise

    if args.question:
        run(" ".join(args.question))
        return
    print("Ask about Terraria (empty line or Ctrl-D to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        run(question)


if __name__ == "__main__":
    main()
