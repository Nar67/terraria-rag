"""Agent mode: the model searches the local wiki index itself before answering.

One-shot RAG searches once with the question; if the pages the answer needs
aren't in that top k (lists spread over several pages, "what next after X"),
the answer is incomplete. Here the model looks at what the first search found,
decides what is still missing and calls the search tool for it, a few times.
The answer is then written by the usual answering path (ask.generate) from
everything collected, so citations, reasoning model and budget stay the same.

Searches go to our own index (search.py), never to the live wiki.

Used by ask.py --agent and eval.py --agent.
"""
import json
import logging

from ask import format_sources, thinking_options

log = logging.getLogger("rag.agent")

AGENT_STEPS = 4    # model turns that may call the tool
SEARCH_K = 2       # chunks per agent search (the first search keeps its own k): the
                   # top 2 of a search by name are usually the page intro and main section
MAX_CHUNKS = 20    # all collected chunks (~9k tokens), so the answer prompt stays a sane size
STEP_TOKENS = 1024  # per planning turn; no reasoning, a tool call is short

AGENT_PROMPT = """You gather excerpts from the Terraria Wiki so that a later step can answer a question. You don't write the answer.

You get the question and the excerpts found so far. Decide what is still missing to answer it completely, for example: every item of a list the question asks for, each boss, NPC or item the excerpts only name, the next step of a progression, a requirement, a recipe ingredient.

To choose searches you may use what you know about Terraria: if you think a boss, item, NPC, biome or page belongs in the answer but isn't in the excerpts, search for it by name. The answer will only use what the searches find, so a wrong guess costs nothing but a search.

For each missing piece call search with a short, specific query about one thing; use wiki page names ("Queen Slime", "Duke Fishron", "Bosses"). You can call search several times at once. If a search finds nothing new, try a different, more specific query. Don't repeat a search.

When the excerpts cover the question, reply DONE without calling a tool."""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the local Terraria Wiki index; returns the best matching excerpts not seen yet.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "a short, specific search query"}},
            "required": ["query"],
        },
    },
}]


def research(question, searcher, client, model, k=8, history=None, steps=AGENT_STEPS,
             on_search=lambda query, found: None):
    """Collect excerpts for the question: one normal search, then the model's own.

    Returns (results, trace). results are numbered in order ([1] first), the
    same numbering the answer prompt will use. trace: [{"query", "found": [ids]}],
    the first entry being the question itself. on_search(query, new_results) is
    called after every search, for printing."""
    results = searcher.search(question, k=k, history=history)
    seen = {r["id"] for r in results}
    trace = [{"query": question, "found": [r["id"] for r in results]}]
    on_search(question, results)
    messages = [{"role": "system", "content": AGENT_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nExcerpts found so far:\n\n{format_sources(results)}"}]
    asked = {question.strip().lower()}

    for _ in range(steps):
        if len(results) >= MAX_CHUNKS:
            break
        reply = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, max_tokens=STEP_TOKENS,
            extra_body=thinking_options(False))
        message = reply.choices[0].message
        if not message.tool_calls:  # DONE, or it answered anyway: stop gathering
            break
        messages.append({"role": "assistant", "content": message.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                        for c in message.tool_calls]})
        for call in message.tool_calls:
            query = _query(call)
            if len(results) >= MAX_CHUNKS:  # say so: a silent "nothing new" misleads the model
                content = "Not searched: the excerpt limit is reached."
            elif query is None or query.lower() in asked:
                content = "Not searched: give a new, non-empty query." if query is None else "Already searched."
            else:
                asked.add(query.lower())
                room = MAX_CHUNKS - len(results)
                found = [r for r in searcher.search(query, k=SEARCH_K, history=history) if r["id"] not in seen][:room]
                content = format_sources(found, start=len(results) + 1) if found else \
                    "Nothing new: the best matches are already in the excerpts."
                results += found
                seen.update(r["id"] for r in found)
                trace.append({"query": query, "found": [r["id"] for r in found]})
                on_search(query, found)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
    return results, trace


def _query(call):
    """The query of a search call, or None if it's malformed."""
    if call.function.name != "search":
        return None
    try:
        query = json.loads(call.function.arguments or "{}").get("query", "")
    except (ValueError, AttributeError):
        return None
    return query.strip() or None
