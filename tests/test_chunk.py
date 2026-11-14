"""Chunking rules. Uses the real bge tokenizer (downloaded once, then cached)."""
import pytest

from chunk import chunk_id, chunk_page, ntok

LIMIT = 512


@pytest.fixture
def chunks(tmp_path):
    """Chunks of a made-up page: an intro, a section far over the limit that
    holds a long table, and a History section."""
    rows = "\n".join(f"Item {i} | Wood ×{i}, Gel ×{i % 7 + 1} | Work Bench" for i in range(120))
    notes = "\n".join(f"- Note number {i} about how this item behaves in Expert Mode." for i in range(60))
    page = tmp_path / "Test_Item.md"
    page.write_text(
        "# Test Item\n\nSource: https://terraria.wiki.gg/wiki/Test_Item\nLicense: CC BY-NC-SA 4.0\n\n"
        "The Test Item is a made-up item.\n\n"
        f"## Crafting\n\nResult | Ingredients | Crafting station\n{rows}\n\n"
        f"## Notes\n\n{notes}\n\n"
        "## History\n\n- Desktop 1.4.4: Introduced.\n",
        encoding="utf-8")
    return chunk_page(page, LIMIT, 100)


def test_no_chunk_over_the_limit(chunks):
    assert len(chunks) > 3  # the page really was split
    assert all(ntok(c["text"]) <= LIMIT for c in chunks)


def test_every_chunk_starts_with_its_context_path(chunks):
    for c in chunks:
        assert c["text"].startswith(c["sections"][0])
        assert c["sections"][0].startswith("Test Item")


def test_split_table_repeats_its_header(chunks):
    table_chunks = [c for c in chunks if "Item 5 |" in c["text"] or "Item 100 |" in c["text"]]
    assert len({c["id"] for c in table_chunks}) == 2  # rows 5 and 100 ended up in different pieces
    assert all("Result | Ingredients | Crafting station" in c["text"] for c in table_chunks)


def test_history_is_kept_apart_and_tagged(chunks):
    history = [c for c in chunks if c["kind"] == "history"]
    assert len(history) == 1 and "Desktop 1.4.4" in history[0]["text"]
    assert all("History" not in s for c in chunks if c["kind"] == "content" for s in c["sections"])


def test_chunk_ids_follow_content_not_position():
    used = set()
    first = chunk_id("Zenith", "same text", used)
    assert first == chunk_id("Zenith", "same text", set())   # same text, same id
    assert chunk_id("Zenith", "same text", used) != first      # a duplicate on one page stays unique
    assert chunk_id("Zenith", "other text", set()) != first
