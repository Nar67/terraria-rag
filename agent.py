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

SEARCH_TOOL = {
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
}
TOOLS = [SEARCH_TOOL]


def _fact_tool(name, description, arg):
    properties = {arg: {"type": "string", "description": "exact wiki page name, e.g. \"King Slime\""}}
    if name in ("drops", "stats"):
        properties["mode"] = {"type": "string", "enum": ["classic", "expert", "master"],
                              "description": "only this difficulty (leave out for all)"}
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": [arg]}}}


# Fact tools (facts.py): exact infobox data, already resolved per difficulty.
FACT_TOOLS = [
    _fact_tool("drops", "Everything an enemy or boss drops, with rates per difficulty, groups and Treasure Bag contents.", "name"),
    _fact_tool("stats", "Health, damage, defense etc. of an enemy or boss per difficulty, plus its type and environment.", "name"),
    _fact_tool("recipe", "How an item is crafted: ingredients with amounts and the crafting station.", "item"),
    _fact_tool("conditions", "What makes an enemy, boss, NPC or event spawn, appear, move in or start: "
               "biome, layer, time, Hardmode, bosses defeated, special rules.", "name"),
    _fact_tool("progression", "What defeating a boss unlocks, and which bosses come next (optional or on the main path).", "name"),
    _fact_tool("bosses", "All bosses of a stage in progression order, each marked optional or on the main path. "
               "name: \"pre-hardmode\", \"hardmode\", \"event\" or \"all\".", "name"),
]
FACT_PROMPT = """

You can also look up exact data from the wiki's infoboxes, more reliable than reading tables: drops(name, mode) for what an enemy or boss drops and how often per difficulty, stats(name, mode) for its health, damage and defense, recipe(item) for how an item is crafted, conditions(name) for what makes an enemy, boss, NPC or event spawn, appear, move in or start, progression(name) for what a boss unlocks and what comes next, bosses(stage) for all bosses of a stage marked optional or not. Use them for questions about drops, stats, crafting, spawning, progression or boss lists, and for the items, bosses or ingredients an answer depends on."""


def research(question, searcher, client, model, k=8, history=None, steps=AGENT_STEPS,
             on_search=lambda query, found: None, facts=None):
    """Collect excerpts for the question: one normal search, then the model's own.

    Returns (results, trace). results are numbered in order ([1] first), the
    same numbering the answer prompt will use. trace: [{"query", "found": [ids]}],
    the first entry being the question itself. on_search(query, new_results) is
    called after every search or lookup, for printing. facts: a FactBook
    (facts.py) to offer the drops / stats / recipe tools and to add the fact
    sources the question clearly asks for up front, as one-shot does; None
    for search only."""
    results = (facts.for_question(question) if facts else []) + searcher.search(question, k=k, history=history)
    seen = {r["id"] for r in results}
    trace = [{"query": question, "found": [r["id"] for r in results]}]
    on_search(question, results)
    tools = TOOLS + (FACT_TOOLS if facts else [])
    messages = [{"role": "system", "content": AGENT_PROMPT + (FACT_PROMPT if facts else "")},
                {"role": "user", "content": f"Question: {question}\n\nExcerpts found so far:\n\n{format_sources(results)}"}]
    asked = {question.strip().lower()}

    for _ in range(steps):
        if len(results) >= MAX_CHUNKS:
            break
        reply = client.chat.completions.create(
            model=model, messages=messages, tools=tools, max_tokens=STEP_TOKENS,
            extra_body=thinking_options(False))
        message = reply.choices[0].message
        if not message.tool_calls:  # DONE, or it answered anyway: stop gathering
            break
        messages.append({"role": "assistant", "content": message.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                        for c in message.tool_calls]})
        for call in message.tool_calls:
            label, found = _call(call, facts)
            if len(results) >= MAX_CHUNKS:  # say so: a silent "nothing new" misleads the model
                content = "Not done: the excerpt limit is reached."
            elif label is None:
                content = "Not done: unknown tool or missing argument."
            elif label.lower() in asked:
                content = "Already done."
            else:
                asked.add(label.lower())
                if found is None:  # a search
                    found = searcher.search(label, k=SEARCH_K, history=history)
                found = [r for r in found if r["id"] not in seen][:MAX_CHUNKS - len(results)]
                content = format_sources(found, start=len(results) + 1) if found else (
                    "Nothing new: the best matches are already in the excerpts." if call.function.name == "search"
                    else "No such data: check the exact page name, or search instead.")
                results += found
                seen.update(r["id"] for r in found)
                trace.append({"query": label, "found": [r["id"] for r in found]})
                on_search(label, found)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
    return results, trace


def _call(call, facts):
    """(label, results) for one tool call. A search returns (query, None): the
    caller runs it. A fact lookup returns ("drops(King Slime, expert)", [source]
    or []). (None, None) if the call is malformed or the tool unknown."""
    try:
        args = json.loads(call.function.arguments or "{}")
    except ValueError:
        return None, None
    if not isinstance(args, dict):
        return None, None
    name = call.function.name
    if name == "search":
        query = str(args.get("query", "")).strip()
        return (query or None), None
    if facts is None or name not in ("drops", "stats", "recipe", "conditions", "progression", "bosses"):
        return None, None
    target = str(args.get("item" if name == "recipe" else "name", "")).strip()
    mode = args.get("mode") if args.get("mode") in ("classic", "expert", "master") else None
    if not target:
        return None, None
    source = facts.lookup(name, target, mode)
    return f"{name}({target}{', ' + mode if mode else ''})", [source] if source else []
