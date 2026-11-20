import pytest

from compare import bootstrap_ci, compare, mcnemar_midp


@pytest.mark.parametrize("b, c, p", [
    (0, 0, 1.0),          # no disagreements: no evidence either way
    (5, 5, 1.0),          # balanced: no difference
    (10, 0, 2 * 0.5 * 0.5 ** 10),  # 10 vs 0: mid-p = 2 * (P(X=0) / 2)
    (0, 10, 2 * 0.5 * 0.5 ** 10),  # symmetric
])
def test_mcnemar_midp(b, c, p):
    assert mcnemar_midp(b, c) == pytest.approx(p)


def test_mcnemar_noise_vs_real_difference():
    """Our two-seed noise (22 vs 18 flips) is not significant; 20 vs 4 is."""
    assert mcnemar_midp(18, 22) > 0.5
    assert mcnemar_midp(4, 20) < 0.01


def test_bootstrap_ci_is_repeatable_and_contains_the_mean():
    diffs = [0.1, -0.2, 0.0, 0.3, 0.1, -0.1, 0.2, 0.0]
    low, high = bootstrap_ci(diffs, samples=2000)
    assert (low, high) == bootstrap_ci(diffs, samples=2000)
    assert low <= sum(diffs) / len(diffs) <= high


def test_compare_pairs_shared_ids_only():
    a = {"q1": {"facts": 1.0}, "q2": {"facts": 0.0}, "only-a": {"facts": 1.0}}
    b = {"q1": {"facts": 1.0}, "q2": {"facts": 1.0}, "only-b": {"facts": 0.0}}
    report = "\n".join(compare(a, b, list_changes=True))
    assert report.startswith("2 shared questions")
    assert "better in B: q2" in report
