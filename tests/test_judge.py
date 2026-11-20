import pytest

from judge import quote_in, score

SOURCES = [{"text": "\n".join([
    "Guide : Crafting a Shellphone > Goblin Tech",
    "- Stopwatch: Buy from Traveling Merchant (see above for tips).",
    "- DPS Meter: Buy from Traveling Merchant (see above for tips).",
    "- Metal Detector: Create a long tunnel in an Ice biome.",
    "Once you are satisfied, head to the underworld. Kill a Voodoo Demon to get a Guide Voodoo Doll.",
    "- Defeat Skeletron in order to access the Dungeon enemies and drop the Tally Counter.",
])}]


@pytest.mark.parametrize("quote, found", [
    # real support, quoted exactly, condensed from a list, or lightly paraphrased
    ("Kill a Voodoo Demon to get a Guide Voodoo Doll.", True),
    ("Goblin Tech: Stopwatch, DPS Meter, Metal Detector", True),
    ("Stopwatch: Buy from Traveling Merchant (see above for tips).\nDPS Meter: Buy from Traveling Merchant (see above for tips).", True),
    ("Tally Counter: Drop from Dungeon enemies (post-Skeletron)", True),
    # invented or misplaced support must fail
    ("Kill a Voodoo Demon in the Jungle Temple", False),
    ("Crafting the True Excalibur requires the Excalibur and 24 Chlorophyte Bars.", False),
    ("Stopwatch costs 5 Gold from the Traveling Merchant.", False),  # a number not in the excerpts
    ("", False),
])
def test_quote_in(quote, found):
    assert quote_in(quote, SOURCES) is found


def claim(verdict, differences="none", quote="Kill a Voodoo Demon to get a Guide Voodoo Doll."):
    return {"claim": "...", "quote": quote, "differences": differences, "verdict": verdict}


@pytest.mark.parametrize("c, supported", [
    (claim("supported"), True),
    # "differences" is only the judge's reasoning; the verdict decides
    (claim("supported", "doesn't say 'permanent'"), True),
    (claim("contradicted", "Jungle Temple, not Underworld"), False),
    (claim("not in excerpts", quote=""), False),
    # a quote that isn't in the excerpts can't support anything
    (claim("supported", quote="Kill a Voodoo Demon in the Jungle Temple"), False),
])
def test_score_claims(c, supported):
    verdict = score({"claims": [c], "coverage": "", "answers_question": "yes", "refuses": False}, SOURCES)
    assert verdict["claims"][0]["supported"] is supported
    assert verdict["faithful"] is supported


@pytest.mark.parametrize("answers, refuses, claims, answerable, rating", [
    ("yes", False, [], True, "yes"),
    ("partially", False, [], True, "partially"),
    # a "yes" that gets a detail wrong is only partly right
    ("yes", False, [claim("contradicted", "Jungle Temple, not Underworld")], True, "partially"),
    # out of scope: a refusal is right, whichever way the judge flags it
    ("no", True, [], False, "yes"),
    ("yes", False, [], False, "yes"),
    ("no", False, [], False, "no"),
])
def test_score_rating(answers, refuses, claims, answerable, rating):
    verdict = {"claims": claims, "coverage": "", "answers_question": answers, "refuses": refuses}
    scored = score(verdict, SOURCES, answerable)
    assert scored["rating"] == rating
    assert scored["correct"] is (rating == "yes")


from judge import cohen_kappa  # noqa: E402


@pytest.mark.parametrize("a, b, kappa", [
    ([True, False, True, False], [True, False, True, False], 1.0),     # perfect
    ([True, True, False, False], [True, False, True, False], 0.0),     # chance level
    # 80% raw agreement, but the judge just says True for everything: kappa 0
    ([True, True, True, True, False], [True, True, True, True, True], 0.0),
    (["yes", "partially", "no", "yes"], ["yes", "partially", "yes", "yes"], 5 / 9),  # (0.75 - 0.4375) / 0.5625
])
def test_cohen_kappa(a, b, kappa):
    assert cohen_kappa(a, b) == pytest.approx(kappa)


def test_cohen_kappa_undefined():
    assert cohen_kappa([True, True], [True, True]) is None
