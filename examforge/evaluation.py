"""Evaluation harness: retrieval benchmark, generation benchmark, prompt/model comparison, regression gate.

All numbers come from actual runs. Each saved report records provider, model, prompt versions, parameters
and timestamp, and whether it was produced by the MOCK provider (mock runs test plumbing, not quality).
"""
from __future__ import annotations

import json
import statistics
import uuid
from pathlib import Path

import yaml

from .config import ROOT, Settings, load_yaml
from .review import QuestionBank
from .schemas import GenerationRequest, now_iso

REPORT_DIR = ROOT / "reports" / "benchmarks"


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def _rate(flags) -> float | None:
    flags = [bool(f) for f in flags if f is not None]
    return round(sum(flags) / len(flags), 4) if flags else None


# --------------------------------------------------------------------------- retrieval
def load_retrieval_gold(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["queries"]


def _relevant(chunk, item: dict) -> bool:
    if chunk.document_id not in item["expected_docs"]:
        return False
    secs = item.get("expected_sections") or []
    phrases = [p.lower() for p in item.get("key_phrases", [])]
    return (not secs or any(chunk.section.startswith(s) for s in secs)) or any(p in chunk.text.lower() for p in phrases)


def validate_gold(retriever, gold: list[dict]) -> list[str]:
    """Gold-set sanity: every key phrase must exist verbatim in an expected document (no invented labels)."""
    problems = []
    for item in gold:
        if item.get("expect_insufficient"):
            continue
        texts = " ".join(c.text.lower() for c in retriever.index.chunks(item["expected_docs"]))
        if not texts:
            problems.append(f"{item['id']}: expected docs not indexed {item['expected_docs']}")
        for p in item.get("key_phrases", []):
            if p.lower() not in texts:
                problems.append(f"{item['id']}: key phrase not found in expected docs: {p!r}")
    return problems


def evaluate_retrieval(retriever, gold: list[dict], k: int = 6, methods=("dense", "lexical", "hybrid")) -> dict:
    """precision@k, hit@k, MRR, required-fact recall@k, metadata-filter violations, traceability."""
    by_doc = {d.document_id: d for d in retriever.docs}
    out = {"k": k, "n_queries": len(gold), "gold_problems": validate_gold(retriever, gold), "methods": {}, "per_query": []}
    for m in methods:
        rows = []
        for item in gold:
            req = GenerationRequest(**item["request"])
            res = retriever.retrieve(req, query=item.get("query"), method=m, k=k)
            rel = [_relevant(rc.chunk, item) for rc in res.chunks]
            text = " ".join(rc.chunk.text.lower() for rc in res.chunks)
            phrases = item.get("key_phrases", [])
            allowed = set(res.filters.get("documents", []))
            violations = sum(
                rc.chunk.document_id not in allowed
                or (by_doc[rc.chunk.document_id].chapter not in (None, req.chapter))
                or (req.mode == "amendment" and not rc.chunk.amendment_marker)
                for rc in res.chunks)
            traceable = all(rc.chunk.title and rc.chunk.source_url and rc.chunk.page_start >= 1 for rc in res.chunks)
            expect_insufficient = item.get("expect_insufficient", False)
            rows.append({
                "id": item["id"], "method": m,
                "precision": round(sum(rel) / k, 3) if not expect_insufficient else None,
                "hit": any(rel) if not expect_insufficient else None,
                "rr": next((1 / (i + 1) for i, r in enumerate(rel) if r), 0.0) if not expect_insufficient else None,
                "fact_recall": round(sum(p.lower() in text for p in phrases) / len(phrases), 3) if phrases else None,
                "filter_violations": violations, "traceable": traceable,
                "insufficient_ok": (res.insufficient == expect_insufficient),
                "top": [f"{rc.chunk.document_id} p{rc.chunk.page_start} {rc.chunk.section[:40]}" for rc in res.chunks[:3]],
            })
        out["methods"][m] = {
            "precision_at_k": _mean(r["precision"] for r in rows),
            "hit_rate_at_k": _rate(r["hit"] for r in rows),
            "mrr": _mean(r["rr"] for r in rows),
            "fact_recall_at_k": _mean(r["fact_recall"] for r in rows),
            "filter_violations": sum(r["filter_violations"] for r in rows),
            "traceable_rate": _rate(r["traceable"] for r in rows),
            "insufficient_handling_rate": _rate(r["insufficient_ok"] for r in rows),
        }
        out["per_query"] += rows
    return out


# --------------------------------------------------------------------------- generation benchmark
def load_cases(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]


def run_benchmark(settings: Settings, cases: list[dict], label: str = "", judge: bool = True, embedder_name: str = "onnx") -> dict:
    """Run every case through the full pipeline in an isolated question bank, then compute metrics."""
    from .pipeline import Engine

    run_id = f"{now_iso()[:10]}_{label or 'run'}_{uuid.uuid4().hex[:6]}"
    s = Settings(**{**settings.__dict__})
    eng = Engine.build(s, embedder_name=embedder_name, run_judge=judge)
    eng.bank = QuestionBank(s.runs_dir / f"bench_{run_id}" / "questions.json")  # isolated: no cross-run dedup/avoid
    case_rows, questions = [], []
    for case in cases:
        req = GenerationRequest(**case["request"])
        run = eng.generate(req)
        expect_insuff = case.get("expect_insufficient", False)
        case_rows.append({
            "id": case["id"], "status": run.status, "message": run.message[:300], "n_requested": req.n,
            "n_generated": len(run.questions), "expect_insufficient": expect_insuff,
            "insufficient_ok": (run.status == "insufficient_evidence") == expect_insuff, **run.stats,
        })
        questions += [(case["id"], q) for q in run.questions]
    report = {
        "run_id": run_id, "label": label, "timestamp": now_iso(),
        "provider": eng.llm.provider.name, "model": eng.llm.provider.model,
        "reviewer_model": eng.llm.reviewer.model if eng.llm.reviewer else eng.llm.provider.model,
        "mock": eng.llm.provider.is_mock, "prompt_versions": dict(eng.llm.prompt_versions),
        "params": {"effort": s.llm_effort, "top_k": s.top_k, "chunk_size": s.chunk_size, "max_revisions": s.max_revisions,
                   "max_retries": s.max_retries, "duplicate_threshold": s.duplicate_threshold},
        "cases": case_rows,
        "metrics": compute_metrics(case_rows, [q for _, q in questions], eng.llm.calls),
        "questions": [{"case": c, **summarise_question(q)} for c, q in questions],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{run_id}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def summarise_question(q) -> dict:
    r = q.reliability
    j = q.reviews.judge
    return {
        "id": q.id, "status": q.status, "verdict": r.verdict, "question": q.mcq.question,
        "options": {o.label: o.text for o in q.mcq.options}, "key": q.mcq.correct_labels,
        "deterministic_passed": r.deterministic_passed, "quote_grounding": r.quote_grounding,
        "evidence_status": r.evidence_status, "answer_agreement": r.answer_agreement,
        "requested_difficulty": r.requested_difficulty, "estimated_difficulty": r.estimated_difficulty,
        "duplicate_of": r.duplicate_of, "reasons": r.reasons[:8], "revisions": q.generation.revisions,
        "judge": j.model_dump() if j else None,
        "examiner_issues": [i.model_dump() for i in q.reviews.examiner.issues] if q.reviews.examiner else [],
    }


def compute_metrics(case_rows: list[dict], qs: list, calls: list) -> dict:
    first = []
    for q in qs:
        a = next((h.detail for h in q.history if h.action == "assessed"), None)
        first.append(a.get("deterministic_passed") if a else None)
    judge_keys = ["relevance", "factual_correctness", "clarity", "difficulty_fit", "distractor_quality", "answer_explanation_consistency"]
    judges = [q.reviews.judge for q in qs if q.reviews.judge]
    pricing = load_yaml("evaluation.yaml")["pricing"]
    tin, tout = sum(c.input_tokens for c in calls), sum(c.output_tokens for c in calls)
    # each call priced at its own model (setter and reviewers may differ); None if any model is unpriced
    prices = [pricing.get(c.model) for c in calls]
    cost = None if (not calls or None in prices) else round(sum(
        c.input_tokens / 1e6 * p["input"] + c.output_tokens / 1e6 * p["output"] for c, p in zip(calls, prices)), 4)
    gen_cases = [c for c in case_rows if not c["expect_insufficient"]]
    return {
        "n_cases": len(case_rows), "n_questions": len(qs),
        "yield_rate": _rate(c["n_generated"] >= c["n_requested"] for c in gen_cases),
        "valid_output_rate": _rate(q.reliability.deterministic_passed for q in qs),
        "first_pass_valid_rate": _rate(first),
        "quote_grounding_mean": _mean(q.reliability.quote_grounding for q in qs),
        "evidence_supported_rate": _rate(q.reliability.evidence_status == "supported" for q in qs if q.reliability.evidence_status != "not_checked"),
        "answer_agreement_rate": _rate(q.reliability.answer_agreement for q in qs),
        "difficulty_match_rate": _rate(q.reliability.difficulty_match for q in qs),
        "passed_for_review_rate": _rate(q.reliability.verdict == "passed_for_human_review" for q in qs),
        "rejected_rate": _rate(q.status == "REJECTED" for q in qs),
        "duplicate_rate": _rate(q.reliability.duplicate_of is not None for q in qs),
        "revision_rate": _rate(q.generation.revisions > 0 for q in qs),
        "insufficient_detection_rate": _rate(c["insufficient_ok"] for c in case_rows if c["expect_insufficient"]),
        "judge_means": {k: _mean(getattr(j, k) for j in judges) for k in judge_keys} if judges else None,
        "llm_calls": len(calls), "failed_calls": sum(not c.ok for c in calls),
        "retry_rate": round(sum(c.attempts - 1 for c in calls) / len(calls), 4) if calls else None,
        "latency_s_total": round(sum(c.latency_s for c in calls), 1),
        "latency_s_per_question": round(sum(c.latency_s for c in calls) / len(qs), 1) if qs else None,
        "input_tokens": tin, "output_tokens": tout,
        "est_cost_usd": cost,
        "status_counts": _counts(q.status for q in qs), "verdict_counts": _counts(q.reliability.verdict for q in qs),
    }


def _counts(xs) -> dict:
    out: dict = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


# --------------------------------------------------------------------------- comparison & regression
def load_report(path_or_id: str) -> dict:
    p = Path(path_or_id)
    if not p.exists():
        p = REPORT_DIR / f"{path_or_id}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def compare(reports: list[dict]) -> list[dict]:
    """One row per metric, one column per run (label/prompt versions), for side-by-side prompt comparison."""
    keys = [k for k, v in reports[0]["metrics"].items() if isinstance(v, (int, float)) or v is None]
    rows = []
    for k in keys:
        row = {"metric": k}
        for r in reports:
            row[f"{r['label'] or r['run_id']}{' (MOCK)' if r['mock'] else ''}"] = r["metrics"].get(k)
        rows.append(row)
    return rows


def regression_check(baseline: dict, candidate: dict, rules: dict | None = None) -> list[str]:
    """Return human-readable violations (empty list = pass)."""
    rules = rules or load_yaml("evaluation.yaml")["regression"]
    bm, cm = baseline["metrics"], candidate["metrics"]
    out = []
    for metric, rule in rules.items():
        b, c = bm.get(metric), cm.get(metric)
        if c is None:
            continue
        if "floor" in rule and c < rule["floor"]:
            out.append(f"{metric}={c} below floor {rule['floor']}")
        if b is not None and "max_drop" in rule and b - c > rule["max_drop"]:
            out.append(f"{metric} dropped {b} -> {c} (max drop {rule['max_drop']})")
        if b is not None and "max_rise" in rule and c - b > rule["max_rise"]:
            out.append(f"{metric} rose {b} -> {c} (max rise {rule['max_rise']})")
    return out


def benchmark_markdown(report: dict) -> str:
    m = report["metrics"]
    lines = [f"## Benchmark {report['run_id']}" + (" - MOCK PROVIDER (plumbing test, not quality)" if report["mock"] else ""),
             f"- model: `{report['model']}` (reviewers: `{report['reviewer_model']}`), prompts: `{report['prompt_versions']}`",
             f"- timestamp: {report['timestamp']}, params: `{report['params']}`", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in m.items() if not isinstance(v, dict)]
    if m.get("judge_means"):
        lines += ["", "LLM-judge means (1-5): " + ", ".join(f"{k}={v}" for k, v in m["judge_means"].items())]
    return "\n".join(lines)
