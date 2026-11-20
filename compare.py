#!/usr/bin/env python3
"""Compare two saved runs question by question, with significance tests.

Usage:
    uv run compare.py eval/runs/A.judged.json eval/runs/B.judged.json
    uv run compare.py A.json B.json --list      # also name the questions that changed

Works on eval.py --save files and on judge.py --save files (which add the
judge's verdicts). Questions are paired by id; only ids in both runs count,
so runs over different question sets (92 vs 117) can still be compared.

Why paired: both runs answer the same questions, so only the questions where
they disagree say anything about which is better. Comparing overall
percentages hides that: two runs of the same setup flipped ~40 of 92 answers
and still differed by only a few points (DEVLOG §24).

- pass/fail metrics (faithful, fully answered, all facts, ...): how many
  questions B got right that A didn't and vice versa, and the exact McNemar
  test (mid-p), the usual choice for small sets.
- graded metrics (fact share, evidence share, claims supported): mean
  difference with a paired bootstrap 95% interval.
p < 0.05 or an interval not containing 0 is marked with *. With ~100
questions only differences of roughly 10+ net questions tend to reach that.
"""
import argparse
import json
import random
import statistics
from math import comb
from pathlib import Path

# name, how to read it from a row (None if the run doesn't have it)
PASS_FAIL = [
    ("judge: faithful", lambda r: r["judge"]["faithful"] if _judged(r) else None),
    ("judge: answers fully", lambda r: r["judge"].get("rating", r["judge"].get("answers_question")) == "yes"
        if _judged(r) else None),
    ("eval: all facts", lambda r: r["facts"] == 1 if "facts" in r else None),
    ("eval: page found", lambda r: r.get("page_hit")),
    ("eval: all evidence", lambda r: r["evidence"] == 1 if "evidence" in r else None),
]
GRADED = [
    ("judge: claims supported", lambda r: r["judge"]["support"] if _judged(r) else None),
    ("eval: fact share", lambda r: r.get("facts")),
    ("eval: evidence share", lambda r: r.get("evidence")),
]
BOOTSTRAP_SAMPLES = 10_000


def _judged(row):
    return "judge" in row and "error" not in row["judge"]


def load_rows(path):
    return {r["id"]: r for r in json.loads(Path(path).read_text(encoding="utf-8"))["rows"]}


def mcnemar_midp(b, c):
    """Two-sided exact McNemar mid-p for b and c discordant pairs (B better: c).

    Under "no difference" each discordant question is a coin flip, so the
    smaller count follows Binomial(b + c, 1/2). mid-p counts half of the
    observed outcome's own probability; it is less conservative than the
    plain exact test and still keeps its error rate at small n."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    below = sum(comb(n, i) for i in range(k)) / 2 ** n
    at = comb(n, k) / 2 ** n
    return min(1.0, 2 * (below + at / 2))


def bootstrap_ci(diffs, samples=BOOTSTRAP_SAMPLES, seed=0):
    """Paired bootstrap 95% interval of the mean of per-question differences."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def compare(a, b, list_changes=False):
    """Lines of the comparison report for two runs given as {id: row}."""
    shared = sorted(a.keys() & b.keys())
    lines = [f"{len(shared)} shared questions"]
    lines.append(f"\n{'pass/fail':<24}{'n':>4}{'A':>7}{'B':>7}{'B-A':>6}   {'B better':>8} {'B worse':>7}   p (McNemar mid-p)")
    changes = {}
    for name, get in PASS_FAIL:
        pairs = [(i, get(a[i]), get(b[i])) for i in shared]
        pairs = [(i, x, y) for i, x, y in pairs if x is not None and y is not None]
        if not pairs:
            continue
        better = [i for i, x, y in pairs if y and not x]
        worse = [i for i, x, y in pairs if x and not y]
        pa = statistics.mean(bool(x) for _, x, _ in pairs)
        pb = statistics.mean(bool(y) for _, _, y in pairs)
        p = mcnemar_midp(len(worse), len(better))
        lines.append(f"{name:<24}{len(pairs):>4}{pa:>7.0%}{pb:>7.0%}{(pb - pa) * 100:>+6.0f}"
                     f"   {len(better):>8} {len(worse):>7}   {p:.3f}{' *' if p < 0.05 else ''}")
        changes[name] = (better, worse)
    lines.append(f"\n{'graded':<24}{'n':>4}{'A':>7}{'B':>7}{'B-A':>7}   95% interval of B-A")
    for name, get in GRADED:
        pairs = [(get(a[i]), get(b[i])) for i in shared]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            continue
        diffs = [y - x for x, y in pairs]
        low, high = bootstrap_ci(diffs)
        significant = low > 0 or high < 0
        lines.append(f"{name:<24}{len(pairs):>4}{statistics.mean(x for x, _ in pairs):>7.2f}"
                     f"{statistics.mean(y for _, y in pairs):>7.2f}{statistics.mean(diffs):>+7.3f}"
                     f"   [{low:+.3f}, {high:+.3f}]{' *' if significant else ''}")
    if list_changes:
        for name, (better, worse) in changes.items():
            if better or worse:
                lines.append(f"\n{name}:\n  better in B: {', '.join(better) or '-'}\n  worse in B:  {', '.join(worse) or '-'}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="run A (the reference)")
    ap.add_argument("b", type=Path, help="run B (the change)")
    ap.add_argument("--list", action="store_true", help="name the questions that got better or worse")
    args = ap.parse_args()
    print(f"A: {args.a}\nB: {args.b}")
    print("\n".join(compare(load_rows(args.a), load_rows(args.b), args.list)))


if __name__ == "__main__":
    main()
