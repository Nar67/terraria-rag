import pytest

from search import normalize, titles_in_query, wants_history


@pytest.mark.parametrize("question, history", [
    ("When was the Zenith added?", True),
    ("What changed for the Eye of Cthulhu in 1.4.4?", True),
    ("Was the Night's Edge nerfed?", True),
    ("How do I summon the Eye of Cthulhu?", False),
    ("What does the Moon Lord drop?", False),
])
def test_wants_history(question, history):
    assert wants_history(question) is history


@pytest.mark.parametrize("query, titles, matched", [
    # only the longest matching title counts
    ("How do I summon the Eye of Cthulhu?", {"Eye of Cthulhu", "Eye", "Bosses"}, {"Eye of Cthulhu"}),
    # apostrophes and case don't matter
    ("what is the recipe for night's edge", {"Night's Edge", "Edge"}, {"Night's Edge"}),
    # whole words only: "Zenith" is not inside "Zenithal"
    ("What is a Zenithal sword?", {"Zenith"}, set()),
])
def test_titles_in_query(query, titles, matched):
    assert titles_in_query(query, titles) == matched


def test_normalize():
    assert normalize("Guide : Class setups (Pre-Hardmode)!") == "guide class setups pre hardmode"
