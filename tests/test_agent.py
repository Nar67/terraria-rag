import json
from types import SimpleNamespace

import pytest

import agent


class FakeSearcher:
    """Each query returns chunks named after it, e.g. 'queen slime#0'."""
    def search(self, query, k=8, history=None):
        return [{"id": f"{query.lower()}#{i}", "title": query, "sections": [query], "kind": "content",
                 "url": "u", "text": f"{query} text {i}"} for i in range(k)]


def tool_call(query, n=0):
    return SimpleNamespace(id=f"call{n}", function=SimpleNamespace(name="search", arguments=json.dumps({"query": query})))


def fake_client(turns):
    """turns: per model turn, the list of queries it calls ([] = DONE)."""
    replies = iter(turns)

    def create(**kwargs):
        queries = next(replies, [])
        message = SimpleNamespace(content="" if queries else "DONE",
                                  tool_calls=[tool_call(q, n) for n, q in enumerate(queries)] or None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


S = agent.SEARCH_K


@pytest.mark.parametrize("turns, queries, chunks", [
    ([[]], ["Q"], 8),                                                    # first search covers it
    ([["Queen Slime"], []], ["Q", "Queen Slime"], 8 + S),                # one extra search
    ([["Queen Slime", "Duke Fishron"], []], ["Q", "Queen Slime", "Duke Fishron"], 8 + 2 * S),
    ([["Queen Slime"], ["queen slime"], []], ["Q", "Queen Slime"], 8 + S),  # a repeated search is refused
])
def test_research(turns, queries, chunks):
    results, trace = agent.research("Q", FakeSearcher(), fake_client(turns), "m", k=8)
    assert [t["query"] for t in trace] == queries
    assert len(results) == chunks
    assert len({r["id"] for r in results}) == len(results)  # no chunk twice


def test_research_stops_at_chunk_limit():
    """Searches past the limit are refused (and said so), never silently empty."""
    turn = [f"topic {i}" for i in range(20)]
    results, trace = agent.research("Q", FakeSearcher(), fake_client([turn, turn]), "m", k=8)
    assert len(results) == agent.MAX_CHUNKS
    assert all(t["found"] for t in trace)  # no search recorded as finding nothing


def test_research_step_limit():
    many = [[f"topic {i}"] for i in range(10)]
    results, trace = agent.research("Q", FakeSearcher(), fake_client(many), "m", k=1)
    assert len(trace) == 1 + agent.AGENT_STEPS


def test_fact_tools(monkeypatch):
    """drops/stats/recipe calls return one fact source each; unknown names are said so."""
    class Book:
        def for_question(self, q):
            return []

        def lookup(self, kind, name, mode=None):
            if name != "King Slime":
                return None
            return {"id": f"facts:{kind}:{name}", "title": name, "sections": [name], "kind": "facts",
                    "url": "u", "text": f"{name} {kind} {mode}"}

    def call(tool, **args):
        return SimpleNamespace(id=tool, function=SimpleNamespace(name=tool, arguments=json.dumps(args)))

    replies = iter([[call("drops", name="King Slime", mode="expert"), call("stats", name="Nobody")], []])

    def create(**kwargs):
        calls = next(replies)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=calls or None))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    results, trace = agent.research("Q", FakeSearcher(), client, "m", k=2, facts=Book())
    assert [r["id"] for r in results][-1] == "facts:drops:King Slime"
    assert [t["query"] for t in trace] == ["Q", "drops(King Slime, expert)", "stats(Nobody)"]
    assert trace[-1]["found"] == []
