#!/usr/bin/env python3
"""Grade the answers of a saved eval run with an LLM judge.

Usage:
    uv run judge.py eval/runs/prompt-requirements.json
    uv run judge.py RUN.json --only progression --show      # print each verdict
    uv run judge.py RUN.json --save eval/runs/RUN.judged.json
    uv run judge.py RUN.json --calibrate                    # agreement with hand labels

eval.py's fact check only tests whether expected words appear in the answer;
it can't see a wrong claim ("kill a Voodoo Demon in the Jungle Temple") or a
right answer worded differently. The judge reads the same numbered sources the
answering model saw and returns a structured verdict:
- claims: every factual claim in the answer, each supported by the sources or
  not -> faithful = all supported (catches made-up or misread facts)
- answers_question: yes / partially / no, with the question's expected facts as
  a reference -> correct = "yes" (for out-of-scope questions: a refusal)

The judge is the same kind of model it grades, so it can be wrong or lenient.
Trust it only as far as --calibrate shows it agrees with the hand labels in
eval/judge-labels.jsonl. Uses the RAG_LLM_* settings of ask.py; the default
model has temperature 0, so verdicts repeat.
"""
import argparse
import json
import logging
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from ask import (LLM_MODEL, LLM_TIMEOUT, LLM_URL, MAX_THINKING_TOKENS, explain, format_sources, pick_model,
                 thinking_options)
from config import HERE, add_logging_args, setup_logging
from search import open_collection

log = logging.getLogger("rag.judge")

LABELS = HERE / "eval" / "judge-labels.jsonl"
MAX_VERDICT_TOKENS = 2048  # a verdict with ~20 claims is ~1,000 tokens

JUDGE_PROMPT = """You check answers written by an assistant that answers questions about the game Terraria from numbered wiki excerpts.

You get the question, the excerpts the assistant was given, a list of facts a complete answer should cover, and the assistant's answer.

1. List every factual claim the answer makes about the game (items, places, conditions, numbers, steps). For each claim, copy the sentence from the excerpts that supports it, word for word, as "quote". The quote must back every detail of the claim, including places, numbers and conditions: if the excerpts say "kill a Voodoo Demon in the Underworld" and the claim says "in the Jungle Temple", the claim is not supported, even if the Jungle Temple is mentioned somewhere else. Pieces from different sentences about different things don't add up to support. If no single passage supports the whole claim, leave "quote" empty and mark it not supported. Ignore the answer's citation numbers, wording and formatting.
2. Decide whether the answer answers the question: "yes" if it covers what was asked (the expected facts are a guide, not a checklist of exact words), "partially" if it misses an important part, "no" if it doesn't answer or is mostly wrong.
3. Note whether the answer says the excerpts don't cover the question.

Judge only against the excerpts, not your own knowledge of the game."""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {"claim": {"type": "string"}, "quote": {"type": "string"}, "supported": {"type": "boolean"}},
            "required": ["claim", "quote", "supported"]}},
        "answers_question": {"type": "string", "enum": ["yes", "partially", "no"]},
        "refuses": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["claims", "answers_question", "refuses", "reason"],
}


STOPWORDS = set("a an the and or of to in on at for from by with is are be it its this that as into than then "
                "can will you your their his her they them there when while if not no but".split())
QUOTE_MATCH = 0.8   # share of a quote line's words that must be in the excerpts
QUOTE_WINDOW = 4    # consecutive excerpt lines a quote line may draw its words from


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def quote_in(quote, sources):
    """Whether the judge's quote really comes from the excerpts.

    Judges condense tables and lists and paraphrase lightly, so an exact match
    is too strict (it rejected real support in testing). Each quote line counts
    as found when at least QUOTE_MATCH of its meaningful words occur within
    QUOTE_WINDOW consecutive excerpt lines and every number in it occurs there
    too: an invented quote ("requires … 24 Chlorophyte Bars") still fails."""
    lines = [l for s in sources for l in s["text"].split("\n") if l.strip()]
    windows = [set(words(" ".join(lines[i:i + QUOTE_WINDOW]))) for i in range(len(lines))]
    parts = [p for p in re.split(r"\n|\.\.\.|…", quote) if words(p)]
    if not parts:
        return False
    for part in parts:
        content = [w for w in words(part) if w not in STOPWORDS]
        numbers = [w for w in content if w.isdigit()]
        if not any(all(n in win for n in numbers)
                   and sum(w in win for w in content) >= QUOTE_MATCH * len(content) for win in windows):
            return False
    return True


def load_sources(collection, retrieved):
    """The chunks the answering model saw, in the same order and numbering."""
    ids = [r["id"] for r in retrieved]
    got = collection.get(ids=ids, include=["documents", "metadatas"])
    by_id = dict(zip(got["ids"], zip(got["documents"], got["metadatas"])))
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} chunks of this run are not in the index any more (re-run eval.py)")
    return [{"kind": by_id[i][1]["kind"], "url": by_id[i][1]["url"], "text": by_id[i][0]} for i in ids]


def judge_one(client, model, question, row, sources, think):
    expected = question.get("answer") or []
    if question.get("answerable", True) is False:
        expected_text = "None: the question is outside what the wiki covers; a correct answer says so."
    else:
        expected_text = "\n".join(f"- {f.replace('|', ' or ')}" for f in expected) or "- (none given)"
    user = (f"Question: {question['question']}\n\n"
            f"Excerpts:\n\n{format_sources(sources)}\n\n"
            f"Facts a complete answer should cover:\n{expected_text}\n\n"
            f"Answer to check:\n{row['answer']}")
    messages = [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}]
    if think:
        # Never force JSON while the model reasons: that combination looped
        # for 19-29k tokens and froze the PC. Ask for JSON and parse it instead.
        messages[0]["content"] += "\n\nAfter reasoning, reply with only a JSON object: " + json.dumps(VERDICT_SCHEMA)
        request = {"max_tokens": MAX_THINKING_TOKENS}
    else:
        request = {"max_tokens": MAX_VERDICT_TOKENS,
                   "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA}}}
    reply = client.chat.completions.create(model=model, messages=messages,
                                           extra_body=thinking_options(think) or None, **request)
    text = reply.choices[0].message.content or ""
    if reply.choices[0].finish_reason == "length":
        return {"error": f"verdict cut off at {request['max_tokens']} tokens"}
    try:
        verdict = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except ValueError:
        return {"error": "no valid JSON verdict"}
    # A claim counts as supported only if the judge says so AND its quote really is
    # in the excerpts: pointing at the evidence keeps the judge honest, and a
    # made-up or empty quote is caught here, not trusted.
    for c in verdict["claims"]:
        c["quote_found"] = quote_in(c["quote"], sources)
        c["judge_said"] = c["supported"]
        c["supported"] = c["supported"] and c["quote_found"]
    claims = verdict["claims"]
    verdict["faithful"] = all(c["supported"] for c in claims)
    verdict["support"] = sum(c["supported"] for c in claims) / len(claims) if claims else 1.0
    if question.get("answerable", True) is False:
        verdict["correct"] = verdict["refuses"]
    else:
        verdict["correct"] = verdict["answers_question"] == "yes"
    return verdict


def summarise(rows):
    groups = defaultdict(list)
    errors = [r["id"] for r in rows if "error" in r["judge"]]
    rows = [r for r in rows if "error" not in r["judge"]]
    for r in rows:
        groups[r["category"]].append(r["judge"])
        groups["ALL"].append(r["judge"])
    lines = [f"{'category':<17}{'n':>3}  {'faithful':>8}  {'claims ok':>9}  {'correct':>7}  {'partial':>7}"]
    for name in sorted(k for k in groups if k != "ALL") + ["ALL"]:
        g = groups[name]
        lines.append(f"{name:<17}{len(g):>3}  {statistics.mean(v['faithful'] for v in g):>8.0%}"
                     f"  {statistics.mean(v['support'] for v in g):>9.0%}"
                     f"  {statistics.mean(v['correct'] for v in g):>7.0%}"
                     f"  {statistics.mean(v['answers_question'] == 'partially' for v in g):>7.0%}")
    if errors:
        lines.append(f"not judged ({len(errors)}): {', '.join(errors)}")
    return "\n".join(lines)


def calibrate(rows, run_name):
    """Agreement between the judge and the hand labels for this run."""
    labels = [json.loads(line) for line in LABELS.open(encoding="utf-8")] if LABELS.exists() else []
    labels = {l["id"]: l for l in labels if l["run"] == run_name}
    pairs = [(labels[r["id"]], r["judge"]) for r in rows if r["id"] in labels and "error" not in r["judge"]]
    if not pairs:
        sys.exit(f"no hand labels for run {run_name!r} in {LABELS}")
    print(f"\ncalibration on {len(pairs)} hand-labelled answers:")
    for key, judged in (("faithful", lambda v: v["faithful"]), ("correct", lambda v: v["answers_question"])):
        agree = [l[key] == judged(v) for l, v in pairs]
        print(f"  {key:<9} judge agrees with the label on {sum(agree)}/{len(agree)} ({statistics.mean(agree):.0%})")
        for (l, v), ok in zip(pairs, agree):
            if not ok:
                print(f"    {l['id']}: label {l[key]!r}, judge {judged(v)!r}  ({l.get('note', '')})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="a results file saved by eval.py --answers --save")
    ap.add_argument("--only", help="one category or question id")
    ap.add_argument("--show", action="store_true", help="print each verdict")
    ap.add_argument("--save", type=Path, help="write the run with verdicts added")
    ap.add_argument("--calibrate", action="store_true", help="compare with hand labels in eval/judge-labels.jsonl")
    think = ap.add_mutually_exclusive_group()
    think.add_argument("--think", dest="think", action="store_true", default=None, help="let the judge reason first")
    think.add_argument("--no-think", dest="think", action="store_false", help="no reasoning (default)")
    add_logging_args(ap)
    args = ap.parse_args()
    setup_logging(args)
    think = False if args.think is None else args.think

    run = json.loads(args.run.read_text(encoding="utf-8"))
    questions = {json.loads(l)["id"]: json.loads(l) for l in (HERE / "eval" / "questions.jsonl").open(encoding="utf-8")}
    rows = [r for r in run["rows"] if "answer" in r and r["id"] in questions]
    if args.only:
        rows = [r for r in rows if args.only in (r["id"], r["category"])]
    if args.calibrate and LABELS.exists():
        labelled = {json.loads(l)["id"] for l in LABELS.open(encoding="utf-8") if json.loads(l)["run"] == args.run.stem}
        rows = [r for r in rows if r["id"] in labelled]
    if not rows:
        sys.exit("no answered questions to judge (was the run saved with eval.py --answers?)")

    client = OpenAI(base_url=LLM_URL, api_key=os.environ.get("RAG_LLM_KEY", "none"), timeout=LLM_TIMEOUT, max_retries=1)
    try:
        model = pick_model(client) if LLM_MODEL == "" else LLM_MODEL
        collection = open_collection()
        log.info(f"judging {len(rows)} answers from {args.run.name} with {model}")
        for n, row in enumerate(rows, 1):
            row["judge"] = judge_one(client, model, questions[row["id"]], row,
                                     load_sources(collection, row["retrieved"]), think)
            v = row["judge"]
            if "error" in v:
                print(f"ERR {row['id']:<26} {v['error']}")
                continue
            bad = [c["claim"] + ("" if c["quote_found"] else "   [quote not found in the excerpts]" if c["judge_said"] else "")
                   for c in v["claims"] if not c["supported"]]
            mark = "ok " if v["faithful"] and v["correct"] else "BAD"
            print(f"{mark} {row['id']:<26} faithful {str(v['faithful']):<5}  answers {v['answers_question']:<9}"
                  f"  claims {len(v['claims']) - len(bad)}/{len(v['claims'])}")
            for claim in bad:
                print(f"    unsupported: {claim}")
            if args.show:
                print(f"    reason: {v['reason']}")
            log.debug(f"{n}/{len(rows)}")
    except Exception as e:
        message = explain(e)
        if message is None:
            raise
        sys.exit(message)

    print()
    print(summarise(rows))
    if args.calibrate:
        calibrate(rows, args.run.stem)
    if args.save:
        run["rows"] = rows
        run["judge"] = {"model": model, "think": think}
        args.save.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info(f"saved to {args.save}")


if __name__ == "__main__":
    main()
