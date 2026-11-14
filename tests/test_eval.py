import pytest

from eval import fact_found, hits_target, score_retrieval

QUESTION = {
    "pages": ["Guide:Class setups"],
    "targets": [{"page": "Guide:Class setups", "section": "Pre- Wall of Flesh"}, {"page": "Star Cannon"}],
    "evidence": ["Star Cannon, Phoenix Blaster|by far the best pre-Hardmode ranged weapon"],
}


def result(title, section, text=""):
    return {"title": title, "sections": [f"{title} > {section}"], "text": text}


@pytest.mark.parametrize("chunk, hit", [
    (result("Guide : Class setups", "Pre- Wall of Flesh"), True),     # right page and section
    (result("Guide : Class setups", "Pre- Lunatic Cultist"), False),  # right page, wrong section
    (result("Star Cannon", "Notes"), True),                           # target without a section: any section
    (result("Hardmode", "Tips"), False),
])
def test_hits_target(chunk, hit):
    assert hits_target(chunk, QUESTION) is hit


def test_score_retrieval_rank_and_evidence():
    results = [
        result("Hardmode", "Tips", "Nothing useful here."),
        result("Guide : Class setups", "Pre- Wall of Flesh", "Best\nStar Cannon, Phoenix Blaster, Quad-Barrel Shotgun"),
    ]
    score = score_retrieval(QUESTION, results)
    assert score["rank"] == 2 and score["page_hit"]
    assert score["evidence"] == 1.0  # the first alternative of the "a|b" evidence item matched


@pytest.mark.parametrize("fact, answer, found", [
    ("Pickaxe Axe|Drax", "You need a Drax or better.", True),
    ("200%", "It needs 200% pickaxe power.", True),
    ("Witch Doctor", "The Merchant moves in.", False),
])
def test_fact_found(fact, answer, found):
    assert fact_found(fact, answer) is found
