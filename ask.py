#!/usr/bin/env python3
"""Answer Terraria questions from the wiki with the local model.

Usage:
    uv run ask.py "how do I summon the Eye of Cthulhu"
    uv run ask.py                                  # ask several questions in a row
    uv run ask.py "Zenith recipe" --show-sources   # also print the retrieved chunks
    uv run ask.py "Zenith recipe" --dry-run        # print the prompt, call nothing
    uv run ask.py "best pre-hardmode armor" --think
    uv run ask.py "which hardmode bosses are optional" --agent   # model searches several times

How it works: search.py finds the top chunks (hybrid search), they go into the
prompt numbered [1], [2], ..., and the model answers from them only, citing
the numbers. The cited wiki pages are listed under the answer.

The model is any OpenAI-compatible chat server:
- RAG_LLM_URL    server address (default http://localhost:11434/v1, Ollama)
- RAG_LLM_MODEL  model name; set it empty to use the first model the server
                 lists. Default: qwen3.5-9b-rag-think when the model reasons
                 (the default, or --think; models/Modelfile.think) and
                 qwen3.5-9b-rag with --no-think (models/Modelfile)
- RAG_LLM_KEY    API key, if the server needs one
- RAG_LLM_TIMEOUT  seconds to wait for the server (default 300)

--think / --no-think switch reasoning on or off (sent both as the Qwen3
"chat_template_kwargs" flag and, for --no-think, as reasoning_effort "none",
which is what Ollama honours). Without either flag nothing extra is sent and
the server's default applies (Qwen3.5 reasons). Reasoning longer than
REASONING_BUDGET tokens is stopped and the model made to answer (Ollama only).
"""
import argparse
import json
import logging
import os
import re
import sys

import openai
from openai import OpenAI

from config import add_logging_args, setup_logging
from search import Searcher, wants_history

log = logging.getLogger("rag.ask")

LLM_URL = os.environ.get("RAG_LLM_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL")  # unset: by reasoning mode, below; "": ask the server
# Reasoning needs sampling: at temperature 0 Qwen3.5's reasoning looped until
# the cap on up to a third of the questions (DEVLOG §25). Without reasoning,
# temperature 0 keeps answers deterministic.
ANSWER_MODEL = "qwen3.5-9b-rag"        # models/Modelfile: temperature 0
THINK_MODEL = "qwen3.5-9b-rag-think"   # models/Modelfile.think: Qwen's thinking-mode sampling
LLM_TIMEOUT = float(os.environ.get("RAG_LLM_TIMEOUT", "300"))  # seconds; local models can be slow


def explain(error):
    """A one-line message for errors from the model server."""
    if isinstance(error, openai.APITimeoutError):
        return f"the model at {LLM_URL} did not answer within {LLM_TIMEOUT:.0f} s (RAG_LLM_TIMEOUT)"
    if isinstance(error, openai.APIConnectionError):
        return f"cannot reach the model at {LLM_URL}; is the server running?"
    if isinstance(error, openai.NotFoundError):
        return ("the server doesn't have the model; check RAG_LLM_MODEL, or create it from models/ "
                "(see the Modelfile headers)") if LLM_MODEL != "" else \
            f"not found at {LLM_URL}; is it an OpenAI-compatible /v1 address?"
    if isinstance(error, openai.AuthenticationError):
        return "the server rejected the API key; set RAG_LLM_KEY"
    if isinstance(error, openai.BadRequestError):  # e.g. the prompt is longer than the model's context
        return f"the server rejected the request: {error.message} (try a smaller -k)"
    if isinstance(error, openai.APIStatusError):
        return f"the model server returned an error {error.status_code}: {error.message}"
    return None


def model_for(think):
    """The model to use: RAG_LLM_MODEL if set, else by reasoning mode
    (think=None means the server default, which reasons for Qwen3.5)."""
    if LLM_MODEL is not None:
        return LLM_MODEL
    return ANSWER_MODEL if think is False else THINK_MODEL


def pick_model(client, think=None):
    if model_for(think):
        return model_for(think)
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
- State the requirements the sources give: whether something is Hardmode-only, needs a boss to be defeated first, or depends on the biome, layer, time of day or an event.
- Be direct and short. Use a list for recipes, steps and drops."""

DIM, RESET = "\033[2m", "\033[0m"


def format_sources(results, start=1):
    """The excerpts as the model sees them, numbered from start."""
    parts = []
    for n, r in enumerate(results, start):
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


def thinking_options(think):
    """Request fields that switch the model's reasoning on or off.

    Servers disagree on the switch, so both are sent: llama.cpp/vLLM-style
    servers read the Qwen3 chat-template flag, Ollama only honours
    reasoning_effort ("none" turns thinking off; its own think=false and the
    template flag were ignored through /v1 in testing). None sends nothing."""
    if think is None:
        return {}
    options = {"chat_template_kwargs": {"enable_thinking": think}}
    if not think:
        options["reasoning_effort"] = "none"
    return options


# Output caps. Without them a model stuck in a loop generates until the context
# is full: thinking + forced JSON once ran 19-29k tokens (5-7 min of full GPU
# load) and froze the whole PC (DEVLOG §20). Normal answers are 150-300 tokens;
# with reasoning first they were 2,200-2,600.
MAX_ANSWER_TOKENS = 1024
MAX_THINKING_TOKENS = 4096  # reasoning counts toward the cap


def max_answer_tokens(think):
    """The output cap: tight without reasoning, roomier when the model may reason
    (think=True, or None = server default, which reasons for Qwen3.5)."""
    return MAX_ANSWER_TOKENS if think is False else MAX_THINKING_TOKENS


# Reasoning budget (DEVLOG §26). Ollama has none, and a reasoning model can
# loop until the output cap, leaving no answer (8 of 92 even with the -think
# sampling model). Qwen's own recipe: once the reasoning reaches the budget,
# stop it, append this early-stop sentence and close the think block, and let
# the model write the answer from the reasoning it has. Counted in streamed
# pieces, which are about one token each.
REASONING_BUDGET = 3072
EARLY_STOP = ("\n\nConsidering the limited time by the user, I have to give the solution "
              "based on the thinking directly now.\n</think>\n\n")


def chatml(messages):
    """The messages in Qwen's chat format, as Ollama renders them for Qwen3.5
    (checked: identical output through /api/chat and this text via raw generate)."""
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


def finish_after_reasoning(model, messages, reasoning, on_answer, temperature=None, seed=None):
    """Let the model answer now, after its reasoning was stopped at the budget.

    Uses Ollama's raw /api/generate to continue the assistant turn after the
    early-stop sentence (the /v1 API can't continue a turn). Returns True if
    the answer itself was cut off."""
    import httpx  # installed with openai

    prompt = chatml(messages) + "<|im_start|>assistant\n<think>\n" + reasoning.strip() + EARLY_STOP
    options = {"num_predict": MAX_ANSWER_TOKENS}
    if temperature is not None:
        options["temperature"] = temperature
    if seed is not None:
        options["seed"] = seed
    base = LLM_URL.rstrip("/").removesuffix("/v1")
    body = {"model": model, "prompt": prompt, "raw": True, "stream": True, "options": options}
    try:
        with httpx.stream("POST", f"{base}/api/generate", json=body, timeout=LLM_TIMEOUT) as reply:
            reply.raise_for_status()
            for line in reply.iter_lines():
                if not line:
                    continue
                part = json.loads(line)
                on_answer(part.get("response", ""))
                if part.get("done"):
                    return part.get("done_reason") == "length"
    except httpx.HTTPError as e:  # not Ollama, or it failed: no answer, but say why
        log.warning(f"could not continue after the reasoning budget ({e}); only Ollama is supported")
    return True


def generate(client, model, messages, think, temperature=None, seed=None, max_tokens=None,
             on_reasoning=lambda text: None, on_answer=lambda text: None):
    """One answer, streamed through the callbacks; returns
    {"answer", "cut_off", "budget_forced"}.

    think: True/False to switch reasoning on/off through the chat template,
    None to send nothing and leave it to the server. Reasoning longer than
    REASONING_BUDGET is stopped and the model is made to answer (see above).
    """
    sampling = {} if temperature is None else {"temperature": temperature}
    if seed is not None:  # repeatable runs with a sampling model
        sampling["seed"] = seed
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True,
        max_tokens=max_tokens or max_answer_tokens(think),
        extra_body=thinking_options(think) or None, **sampling)
    reasoning, answer = [], []
    cut_off = over_budget = False

    def got_reasoning(text):
        if text:
            reasoning.append(text)
            on_reasoning(text)

    def got_answer(text):
        if text:
            answer.append(text)
            on_answer(text)

    # Some servers send reasoning in a separate field, others inline between
    # <think> tags; either way it is kept out of the answer.
    tags = ThinkFilter(got_reasoning, got_answer)
    budget = None if think is False else REASONING_BUDGET
    for event in stream:
        if not event.choices:
            continue
        choice = event.choices[0]
        cut_off = cut_off or choice.finish_reason == "length"
        got_reasoning(getattr(choice.delta, "reasoning_content", None) or getattr(choice.delta, "reasoning", None))
        tags.feed(choice.delta.content or "")
        if budget and not answer and len(reasoning) >= budget:
            over_budget = True
            stream.close()  # stops the generation on the server
            break
    tags.flush()
    if over_budget:
        cut_off = finish_after_reasoning(model, messages, "".join(reasoning), got_answer, temperature, seed)
    return {"answer": "".join(answer).strip(), "cut_off": cut_off, "budget_forced": over_budget}


def stream_answer(client, model, messages, think, temperature):
    """Print the reasoning (dimmed) and the answer as they arrive; return the answer."""
    def show_reasoning(text):
        print(f"{DIM}{text}{RESET}", end="", flush=True)

    def show_answer(text):
        print(text, end="", flush=True)

    result = generate(client, model, messages, think, temperature,
                      on_reasoning=show_reasoning, on_answer=show_answer)
    if result["budget_forced"]:
        print(f"\n{DIM}(reasoning stopped at {REASONING_BUDGET} tokens and the answer forced){RESET}", end="")
    if result["cut_off"]:
        print(f"\n{DIM}(answer cut off at the token limit){RESET}", end="")
    print()
    return result["answer"]


def ask(question, searcher, client, model, args, planner=None, book=None):
    """Answer one question. planner: the model that chooses searches in --agent
    mode. book: the FactBook (facts.py), None with --no-facts."""
    if args.agent:
        from agent import research

        def show_search(query, found):
            titles = ", ".join(dict.fromkeys(r["title"] for r in found)) or "nothing new"
            print(f"{DIM}search: {query}  ->  {len(found)} chunks: {titles}{RESET}")

        results, _ = research(question, searcher, client, planner, k=args.k, history=args.history,
                              on_search=show_search, facts=book)
        print()
    else:
        results = searcher.search(question, k=args.k, mode=args.mode, history=args.history)
        if book is not None:  # resolved stats/drops/recipes first, then the chunks
            results = book.for_question(question) + results
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
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature (default: the model's own setting)")
    ap.add_argument("--show-sources", action="store_true", help="list the retrieved chunks before the answer")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt and stop; no model needed")
    ap.add_argument("--agent", action="store_true",
                    help="let the model search the local index several times before answering (agent.py)")
    ap.add_argument("--no-facts", dest="facts", action="store_false",
                    help="don't add fact sources (NPC stats, drops, recipes from facts.py)")
    add_logging_args(ap)
    args = ap.parse_args()
    setup_logging(args)

    searcher = Searcher()
    from facts import open_factbook
    book = open_factbook(args.facts)
    client = OpenAI(base_url=LLM_URL, api_key=os.environ.get("RAG_LLM_KEY", "none"),
                    timeout=LLM_TIMEOUT, max_retries=1)

    model = planner = None

    def run(question):
        """Answer one question; return an error message instead of raising."""
        nonlocal model, planner
        try:
            if model is None and not args.dry_run:
                model = pick_model(client, args.think)
            if args.agent and planner is None:  # choosing searches needs no reasoning: fast, repeatable
                planner = pick_model(client, think=False)
            ask(question, searcher, client, model, args, planner, book)
        except openai.OpenAIError as e:
            message = explain(e)
            if message is None:
                raise
            return message
        return None

    if args.question:
        error = run(" ".join(args.question))
        if error:
            log.error(error)
            sys.exit(1)
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
        try:
            error = run(question)
        except KeyboardInterrupt:  # Ctrl-C stops this answer, not the session
            print("\n(stopped)")
            continue
        if error:
            log.error(error)


if __name__ == "__main__":
    main()
