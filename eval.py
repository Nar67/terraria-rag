#!/usr/bin/env python3
"""Score retrieval (and optionally answers) on the questions in eval/questions.jsonl.

Usage:
    uv run eval.py                      # retrieval only, hybrid search, k=8
    uv run eval.py --mode bm25 -k 5     # compare search modes and depths
    uv run eval.py --misses             # also list what was retrieved for failures
    uv run eval.py --only misspelled    # one category, or a question id
    uv run eval.py --save runs/base.json
    uv run eval.py --answers            # also ask the model (see ask.py for RAG_LLM_*)

Retrieval, per answerable question:
- page hit: a retrieved chunk comes from one of the expected pages (or,
  when the question has "targets", from the expected page *and section*)
- rank: position of the first such chunk (summarised as MRR, mean 1/rank)
- evidence: share of the question's evidence strings found in the retrieved
  chunks' text (case-insensitive). 1.0 means everything the answer needs
  reached the prompt.
- history: whether History chunks were switched on, against whether the
  question is about versions.

Answers (--answers): each expected fact ("a|b" means a or b) must appear in
the model's answer; out-of-scope questions pass when the answer says the wiki
doesn't cover it. A plain text match is a rough judge: read the answers of
the questions it fails before trusting the number.

Question format (one JSON object per line):
  {"id", "category", "question", "pages": [...], "evidence": [...],
   "answer": [...], "history": true?, "answerable": false?,
   "targets": [{"page": ..., "section": ...}]?}
"evidence" and "answer" items may hold alternatives: "a|b".
"""
import argparse
import json
import logging
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from config import HERE, add_logging_args, setup_logging
from search import TUNING, Searcher, wants_history

log = logging.getLogger("rag.eval")

QUESTIONS = HERE / "eval" / "questions.jsonl"
# Phrases that count as "the wiki doesn't cover this" for out-of-scope questions.
REFUSAL = re.compile(r"(don't|do not|doesn't|does not)\s+(cover|contain|mention|include)|not covered|no information", re.I)


def norm_title(title):
    """'Guide : Class setups' and 'Guide:Class setups' compare equal."""
    return re.sub(r"\s*:\s*", ":", title).strip().lower()


def load_questions(path, only):
    questions = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if only:
        questions = [q for q in questions if only in (q["id"], q["category"])]
    if not questions:
        sys.exit(f"no questions match {only!r}" if only else f"no questions in {path}")
    return questions


def targets(q):
    """Where a good chunk may come from: [{"page": ..., "section": ...?}].

    "targets" in the question wins; otherwise every page in "pages", any
    section. A section matches when it is part of the chunk's section path,
    so "Pre- Wall of Flesh" matches "Guide : Class setups > Pre- Wall of Flesh"."""
    return q.get("targets") or [{"page": p} for p in q["pages"]]


def hits_target(result, q):
    title = norm_title(result["title"])
    for t in targets(q):
        if norm_title(t["page"]) != title:
            continue
        if "section" not in t or any(t["section"].lower() in s.lower() for s in result["sections"]):
            return True
    return False


def score_retrieval(q, results):
    rank = next((i for i, r in enumerate(results, 1) if hits_target(r, q)), None)
    text = "\n".join(r["text"] for r in results).lower()
    # An evidence item "a|b" is found when either alternative is.
    found = [e for e in q["evidence"] if any(alt.strip().lower() in text for alt in e.split("|"))]
    return {
        "page_hit": rank is not None,
        "rank": rank,
        "evidence": len(found) / len(q["evidence"]),
        "missing_evidence": [e for e in q["evidence"] if e not in found],
    }


def fact_found(fact, answer):
    return any(alt.strip().lower() in answer.lower() for alt in fact.split("|"))


def ask_model(question, results, client, model, think=None, temperature=None):
    """The model's answer, non-streaming, with any <think> block removed."""
    from ask import build_messages, max_answer_tokens, thinking_options  # only needed with --answers

    extra = thinking_options(think)
    sampling = {} if temperature is None else {"temperature": temperature}  # None: the model's own setting
    reply = client.chat.completions.create(
        model=model, messages=build_messages(question, results), stream=False,
        max_tokens=max_answer_tokens(think), extra_body=extra or None, **sampling)
    if reply.choices[0].finish_reason == "length":
        log.warning(f"answer cut off at {max_answer_tokens(think)} tokens: {question[:60]}")
    text = reply.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def score_answer(q, answer):
    if q.get("answerable", True) is False:
        return {"facts": 1.0 if REFUSAL.search(answer) else 0.0, "missing_facts": []}
    missing = [f for f in q["answer"] if not fact_found(f, answer)]
    return {"facts": 1 - len(missing) / len(q["answer"]), "missing_facts": missing,
            "cites": bool(re.search(r"\[\d+\]", answer))}


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else float("nan")


def summarise(rows, answers):
    """Per-category and overall averages."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
        groups["ALL"].append(row)
    lines = []
    header = f"{'category':<16}{'n':>3}  {'hit@k':>6}  {'MRR':>5}  {'evidence':>8}  {'all ev.':>7}  {'hist ok':>7}"
    if answers:
        header += f"  {'facts':>5}  {'all f.':>6}"
    lines.append(header)
    order = sorted(k for k in groups if k != "ALL") + ["ALL"]
    for name in order:
        g = groups[name]
        ret = [r for r in g if r["answerable"]]
        line = f"{name:<16}{len(g):>3}"
        if ret:
            line += (f"  {mean(r['page_hit'] for r in ret):>6.0%}"
                     f"  {mean(1 / r['rank'] if r['rank'] else 0 for r in ret):>5.2f}"
                     f"  {mean(r['evidence'] for r in ret):>8.0%}"
                     f"  {mean(r['evidence'] == 1 for r in ret):>7.0%}"
                     f"  {mean(r['history_ok'] for r in ret):>7.0%}")
        else:
            line += f"  {'-':>6}  {'-':>5}  {'-':>8}  {'-':>7}  {'-':>7}"
        if answers:
            scored = [r for r in g if "facts" in r]
            if scored:
                line += f"  {mean(r['facts'] for r in scored):>5.0%}  {mean(r['facts'] == 1 for r in scored):>6.0%}"
        lines.append(line)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", type=int, default=8, help="chunks retrieved per question (default 8, as ask.py)")
    ap.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    ap.add_argument("--pool", type=int, help="depth of each ranking before merging (default: search.TUNING)")
    ap.add_argument("--vector-weight", type=float, help="weight of the vector ranking; BM25 has 1")
    ap.add_argument("--title-boost", type=float, help="bonus for chunks whose page title is in the question")
    ap.add_argument("--trivia-weight", type=float, help="score factor for Trivia-only chunks")
    ap.add_argument("--title-max-share", type=float, help="no title boost for titles in more than this share of chunks")
    ap.add_argument("--spell-cutoff", type=float, help="similarity needed to correct an unknown word (0 = off)")
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--only", help="run one category or one question id")
    ap.add_argument("--misses", action="store_true", help="list the retrieved chunks of imperfect questions")
    ap.add_argument("--answers", action="store_true", help="also ask the model and check the answers")
    ap.add_argument("--temperature", default="model",
                    help='with --answers: sampling temperature; "model" (default) sends none, so the '
                         "model's own setting (e.g. from an Ollama Modelfile) applies")
    think = ap.add_mutually_exclusive_group()
    think.add_argument("--think", dest="think", action="store_true", default=None, help="with --answers: model reasons first")
    think.add_argument("--no-think", dest="think", action="store_false", help="with --answers: no reasoning (faster)")
    ap.add_argument("--save", type=Path, help="write all results to this JSON file")
    add_logging_args(ap)
    args = ap.parse_args()
    setup_logging(args)

    questions = load_questions(args.questions, args.only)
    searcher = Searcher()
    client = model = None
    if args.answers:
        import openai  # noqa: F401  (fail early if missing)
        from ask import LLM_TIMEOUT, LLM_URL, explain, pick_model
        from openai import OpenAI
        import os
        client = OpenAI(base_url=LLM_URL, api_key=os.environ.get("RAG_LLM_KEY", "none"),
                        timeout=LLM_TIMEOUT, max_retries=1)
        try:
            model = pick_model(client)
        except Exception as e:  # report server problems once, not per question
            sys.exit(explain(e) or str(e))
        log.info(f"answers from {model} at {LLM_URL}")

    knobs = {name: getattr(args, name) for name in ("pool", "vector_weight", "title_boost", "trivia_weight", "title_max_share", "spell_cutoff")}
    knobs = {name: value for name, value in knobs.items() if value is not None}
    settings = {**TUNING, **knobs}
    rows = []
    started = time.time()
    for q in questions:
        results = searcher.search(q["question"], k=args.k, mode=args.mode, **knobs)
        history_on = wants_history(q["question"])
        row = {"id": q["id"], "category": q["category"], "question": q["question"],
               "answerable": q.get("answerable", True), "history_on": history_on,
               "retrieved": [{"id": r["id"], "title": r["title"], "kind": r["kind"],
                              "sections": r["sections"]} for r in results]}
        if row["answerable"]:
            row.update(score_retrieval(q, results))
            row["history_ok"] = history_on == bool(q.get("history"))
        if client is not None:
            try:
                row["answer"] = ask_model(q["question"], results, client, model, args.think,
                                          None if args.temperature == "model" else float(args.temperature))
            except Exception as e:
                from ask import explain
                message = explain(e)
                if message is None:
                    raise
                sys.exit(message)
            row.update(score_answer(q, row["answer"]))
        rows.append(row)

        marks = []
        if row["answerable"]:
            marks.append(f"rank {row['rank'] or '-':>2}")
            marks.append(f"evidence {row['evidence']:>4.0%}")
            if not row["history_ok"]:
                marks.append("history " + ("on" if history_on else "off") + "!")
        if "facts" in row:
            marks.append(f"facts {row['facts']:>4.0%}")
        perfect = (not row["answerable"] or (row["page_hit"] and row["evidence"] == 1 and row["history_ok"])) \
            and row.get("facts", 1) == 1
        print(f"{'ok ' if perfect else 'BAD'} {q['id']:<26} {'  '.join(marks)}")
        if not perfect and row["answerable"] and row["missing_evidence"]:
            print(f"    missing evidence: {row['missing_evidence']}")
        if not perfect and row.get("missing_facts"):
            print(f"    answer lacks: {row['missing_facts']}")
        if not perfect and args.misses:
            for i, r in enumerate(row["retrieved"], 1):
                print(f"    {i}. {r['id']} [{r['kind']}]  {' | '.join(r['sections'])}")

    tuning = ", ".join(f"{name}={value}" for name, value in settings.items())
    print(f"\n{len(rows)} questions, mode {args.mode}, k={args.k}, {tuning}, {time.time() - started:.0f} s\n")
    print(summarise(rows, args.answers))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps({"mode": args.mode, "k": args.k, "tuning": settings, "rows": rows},
                                        ensure_ascii=False, indent=1), encoding="utf-8")
        log.info(f"saved results to {args.save}")


if __name__ == "__main__":
    main()
