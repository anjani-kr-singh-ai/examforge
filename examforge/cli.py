"""ExamForge AI command line.  python -m examforge.cli <command> --help"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT, Settings


def _engine(a, **kw):
    from .pipeline import Engine

    s = Settings()
    if a.provider:
        s.llm_provider = a.provider
    if getattr(a, "max_revisions", None) is not None:
        s.max_revisions = a.max_revisions
    return Engine.build(s, embedder_name=a.embedder, **kw)


def _req(a):
    from .schemas import GenerationRequest

    return GenerationRequest(subject=a.subject, chapter=a.chapter, topic=a.topic, n=getattr(a, "n", 1),
                             difficulty=getattr(a, "difficulty", "Medium"), question_type=getattr(a, "type", "single"),
                             mode=getattr(a, "mode", "conceptual"), attempt=a.attempt or None, allow_historical=a.allow_historical)


def _print_q(q, full=False):
    r = q.reliability
    print(f"\n[{q.id}] {q.status} | verdict={r.verdict} | {q.provenance} | {q.chapter} / {q.topic} | {q.difficulty} {q.question_type}"
          + (" | MOCK" if q.generation.mock else ""))
    print(q.mcq.question)
    for o in q.mcq.options:
        print(f"  ({o.label}){'*' if o.is_correct else ' '} {o.text}")
    print(f"  grounding={r.quote_grounding} evidence={r.evidence_status} blind-agree={r.answer_agreement} "
          f"difficulty est={r.estimated_difficulty} dup={r.duplicate_of} composite={r.composite_score}")
    for x in r.reasons[: (20 if full else 4)]:
        print(f"   - {x}")
    if full:
        print("Explanation:", q.mcq.explanation)
        for o in q.mcq.options:
            print(f"  ({o.label}) {o.explanation}")
        for c in q.citations:
            print(f"  SOURCE {c.title} pp.{c.page_start}-{c.page_end} {c.section} match={c.quote_match} :: {c.quote[:140]}")


def cmd_ingest(a):
    from .ingestion import ingest_all

    eng = _engine(a)
    s = eng.s
    docs = ingest_all(s.manifest_path, s.raw_dir, eng.index, s.chunk_size, s.chunk_overlap, s.data_dir / "processed")
    for d in docs:
        print(f"{d.document_id:32s} {str(d.extraction_status):12s} pages={d.page_count} chunks={d.chunk_count}")
    print("chunks in index:", eng.index.count())


def cmd_retrieve(a):
    eng = _engine(a)
    res = eng.retriever.retrieve(_req(a), method=a.method, k=a.k)
    print("query:", res.query, "\ninsufficient:", res.insufficient, res.reason)
    for w in res.warnings:
        print("WARNING:", w)
    for c in res.conflicts:
        print("CONFLICT:", c)
    for rc in res.chunks:
        print(f"\n- {rc.chunk.citation()}  dense={rc.dense_similarity} lex={rc.lexical_score}\n  {rc.chunk.text[:300]}")


def cmd_generate(a):
    eng = _engine(a)
    run = eng.generate(_req(a))
    print(f"run {run.run_id}: {run.status} {run.message}\nstats: {run.stats}")
    for e in (run.retrieval.warnings if run.retrieval else []):
        print("WARNING:", e)
    for q in run.questions:
        _print_q(q, full=a.full)


def cmd_list(a):
    from .review import QuestionBank

    for q in QuestionBank(Settings().bank_path).all():
        if not a.status or q.status == a.status:
            print(f"{q.id} {q.status:12s} {q.reliability.verdict:24s} {q.chapter[:20]:20s} {q.topic[:35]:35s} {q.mcq.question[:70]}")


def cmd_show(a):
    from .review import QuestionBank

    _print_q(QuestionBank(Settings().bank_path).get(a.id), full=True)


def cmd_review(a):
    from . import review

    bank = review.QuestionBank(Settings().bank_path)
    if a.action == "approve":
        q = review.approve(bank, a.id, a.reviewer, a.comment)
    elif a.action == "reject":
        q = review.reject(bank, a.id, a.reviewer, a.reason or "other", a.comment)
    else:
        q = review.add_comment(bank, a.id, a.reviewer, a.comment)
    print(q.id, "->", q.status)


def cmd_edit(a):
    eng = _engine(a)
    ch = {}
    if a.question:
        ch["question"] = a.question
    if a.correct:
        ch["correct_labels"] = a.correct.split(",")
    if a.difficulty:
        ch["difficulty"] = a.difficulty
    q = eng.edit(a.id, a.reviewer, comment=a.comment, **ch)
    _print_q(q)


def cmd_regenerate(a):
    run = _engine(a).regenerate(a.id, a.reviewer)
    print(run.status, run.message)
    for q in run.questions:
        _print_q(q)


def cmd_export(a):
    from .export import export
    from .review import QuestionBank

    qs = [q for q in QuestionBank(Settings().bank_path).all() if not a.status or q.status in a.status.split(",")]
    for k, p in export(qs, Path(a.out)).items():
        print(k, p)


def cmd_eval_retrieval(a):
    from .evaluation import evaluate_retrieval, load_retrieval_gold

    eng = _engine(a)
    res = evaluate_retrieval(eng.retriever, load_retrieval_gold(Path(a.gold)), k=a.k)
    for p in res["gold_problems"]:
        print("GOLD PROBLEM:", p)
    print(json.dumps(res["methods"], indent=1))
    out = ROOT / "reports" / "retrieval_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", out)


def cmd_benchmark(a):
    from .evaluation import benchmark_markdown, load_cases, run_benchmark

    s = Settings()
    if a.provider:
        s.llm_provider = a.provider
    if a.prompt_versions:
        s.prompt_versions = dict(kv.split("=", 1) for kv in a.prompt_versions.split(","))
    cases = load_cases(Path(a.cases))
    if a.only:
        cases = [c for c in cases if c["id"] in a.only.split(",")]
    rep = run_benchmark(s, cases, label=a.label, judge=not a.no_judge, embedder_name=a.embedder)
    print(benchmark_markdown(rep))


def cmd_compare(a):
    import pandas as pd

    from .evaluation import compare, load_report

    print(pd.DataFrame(compare([load_report(r) for r in a.runs])).to_string(index=False))


def cmd_regress(a):
    from .evaluation import load_report, regression_check

    v = regression_check(load_report(a.baseline), load_report(a.candidate))
    print("REGRESSION:\n" + "\n".join(v) if v else "no regression")
    sys.exit(1 if v else 0)


def cmd_coverage(a):
    from .analytics import coverage, feedback_analysis
    from .ingestion import load_manifest
    from .review import QuestionBank

    s = Settings()
    qs = QuestionBank(s.bank_path).all()
    cov = coverage(qs, load_manifest(s.manifest_path))
    print(cov["table"].to_string(index=False))
    print("\n" + cov["note"])
    for r in cov["recommendations"]:
        print("-", r)
    print("\nFeedback:", json.dumps(feedback_analysis(qs), indent=1, default=str))


def main(argv=None):
    p = argparse.ArgumentParser(prog="examforge")
    p.add_argument("--provider", help="override LLM_PROVIDER (anthropic|openai|mock)")
    p.add_argument("--embedder", default="onnx", choices=["onnx", "hash"])
    sub = p.add_subparsers(dest="cmd", required=True)

    def req_args(sp, gen=True):
        sp.add_argument("--subject", default="IDT")
        sp.add_argument("--chapter", required=True)
        sp.add_argument("--topic", required=True)
        sp.add_argument("--attempt", default="May 2027")
        sp.add_argument("--allow-historical", action="store_true")
        if gen:
            sp.add_argument("--n", type=int, default=3)
            sp.add_argument("--difficulty", default="Medium", choices=["Easy", "Medium", "Hard"])
            sp.add_argument("--type", default="single", choices=["single", "multiple"])
            sp.add_argument("--mode", default="conceptual", choices=["conceptual", "application", "case_based", "multi_concept", "amendment"])

    sub.add_parser("ingest").set_defaults(fn=cmd_ingest)
    sp = sub.add_parser("retrieve"); req_args(sp, gen=False)
    sp.add_argument("--method", default="hybrid", choices=["hybrid", "dense", "lexical"]); sp.add_argument("--k", type=int, default=6)
    sp.add_argument("--mode", default="conceptual"); sp.set_defaults(fn=cmd_retrieve)
    sp = sub.add_parser("generate"); req_args(sp); sp.add_argument("--full", action="store_true")
    sp.add_argument("--max-revisions", type=int); sp.set_defaults(fn=cmd_generate)
    sp = sub.add_parser("list"); sp.add_argument("--status"); sp.set_defaults(fn=cmd_list)
    sp = sub.add_parser("show"); sp.add_argument("id"); sp.set_defaults(fn=cmd_show)
    sp = sub.add_parser("review"); sp.add_argument("action", choices=["approve", "reject", "comment"]); sp.add_argument("id")
    sp.add_argument("--reviewer", required=True); sp.add_argument("--comment", default=""); sp.add_argument("--reason")
    sp.set_defaults(fn=cmd_review)
    sp = sub.add_parser("edit"); sp.add_argument("id"); sp.add_argument("--reviewer", required=True)
    sp.add_argument("--question"); sp.add_argument("--correct", help="e.g. A or A,C"); sp.add_argument("--difficulty")
    sp.add_argument("--comment", default=""); sp.set_defaults(fn=cmd_edit)
    sp = sub.add_parser("regenerate"); sp.add_argument("id"); sp.add_argument("--reviewer", default="cli"); sp.set_defaults(fn=cmd_regenerate)
    sp = sub.add_parser("export"); sp.add_argument("--status", help="e.g. APPROVED or APPROVED,NEEDS_REVIEW")
    sp.add_argument("--out", default=str(ROOT / "exports")); sp.set_defaults(fn=cmd_export)
    sp = sub.add_parser("eval-retrieval"); sp.add_argument("--gold", default=str(ROOT / "data/benchmark/retrieval_gold.yaml"))
    sp.add_argument("--k", type=int, default=6); sp.set_defaults(fn=cmd_eval_retrieval)
    sp = sub.add_parser("benchmark"); sp.add_argument("--label", default="")
    sp.add_argument("--cases", default=str(ROOT / "data/benchmark/generation_cases.yaml")); sp.add_argument("--only")
    sp.add_argument("--prompt-versions", help="e.g. question_setter=v1"); sp.add_argument("--no-judge", action="store_true")
    sp.set_defaults(fn=cmd_benchmark)
    sp = sub.add_parser("compare"); sp.add_argument("runs", nargs="+"); sp.set_defaults(fn=cmd_compare)
    sp = sub.add_parser("regress"); sp.add_argument("baseline"); sp.add_argument("candidate"); sp.set_defaults(fn=cmd_regress)
    sub.add_parser("coverage").set_defaults(fn=cmd_coverage)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
