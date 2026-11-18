"""Syllabus coverage, distribution analytics and human-feedback analysis over the question bank.

No official ICAI weightage is used or implied: coverage reports the *actual* distribution of questions
against the configured topic hierarchy, and source availability per chapter.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from .config import load_yaml
from .schemas import QuestionRecord, SourceDocument


def questions_frame(qs: list[QuestionRecord]) -> pd.DataFrame:
    return pd.DataFrame([{
        "id": q.id, "subject": q.subject, "chapter": q.chapter, "topic": q.topic, "difficulty": q.difficulty,
        "estimated_difficulty": q.reliability.estimated_difficulty, "question_type": q.question_type, "mode": q.mode,
        "status": q.status, "verdict": q.reliability.verdict, "provenance": q.provenance,
        "evidence_status": q.reliability.evidence_status, "quote_grounding": q.reliability.quote_grounding,
        "answer_agreement": q.reliability.answer_agreement, "difficulty_match": q.reliability.difficulty_match,
        "duplicate_of": q.reliability.duplicate_of, "max_similarity": q.reliability.max_similarity,
        "composite": q.reliability.composite_score, "human_edited": q.human_edited, "revisions": q.generation.revisions,
        "model": q.generation.model, "mock": q.generation.mock,
        "setter_prompt": q.generation.prompt_versions.get("question_setter"), "created_at": q.created_at,
    } for q in qs])


def coverage(qs: list[QuestionRecord], docs: list[SourceDocument], subject: str = "IDT") -> dict:
    tree = load_yaml("subjects.yaml")["subjects"][subject]["chapters"]
    live = [q for q in qs if q.subject == subject and q.status != "REJECTED"]
    per_topic = Counter((q.chapter, q.topic) for q in live)
    sourced = {d.chapter for d in docs if d.subject == subject and (d.extraction_status or "").startswith("ok") and d.chapter}
    rows = []
    for ch, spec in tree.items():
        topics = list((spec or {}).get("topics", {}) or {})
        for t in topics or ["(no topics configured)"]:
            rows.append({"chapter": ch, "topic": t, "questions": per_topic.get((ch, t), 0), "chapter_has_sources": ch in sourced})
    df = pd.DataFrame(rows)
    gen_df = df[df.chapter_has_sources & (df.topic != "(no topics configured)")]
    nonzero = gen_df[gen_df.questions > 0].questions
    mean = float(nonzero.mean()) if len(nonzero) else 0.0
    missing = gen_df[gen_df.questions == 0][["chapter", "topic"]].to_dict("records")
    under = gen_df[(gen_df.questions > 0) & (gen_df.questions < 0.5 * mean)][["chapter", "topic", "questions"]].to_dict("records")
    over = gen_df[gen_df.questions > 2 * mean][["chapter", "topic", "questions"]].to_dict("records") if mean else []
    diff_imbalance = []
    for ch, grp in _by_chapter(live).items():
        c = Counter(q.difficulty for q in grp)
        top, n = c.most_common(1)[0]
        if len(grp) >= 5 and n / len(grp) > 0.6:
            diff_imbalance.append({"chapter": ch, "dominant": top, "share": round(n / len(grp), 2), "counts": dict(c)})
    recs = [f"Generate questions for '{m['topic']}' ({m['chapter']}) - none yet." for m in missing[:10]]
    recs += [f"Add questions for '{u['topic']}' ({u['chapter']}): {u['questions']} vs mean {mean:.1f}." for u in under]
    recs += [f"'{d['chapter']}' is {int(d['share']*100)}% {d['dominant']}: add other difficulty levels." for d in diff_imbalance]
    no_src = sorted({r["chapter"] for r in rows if not r["chapter_has_sources"]})
    if no_src:
        recs.append(f"{len(no_src)} chapters have no indexed ICAI source; ingest material before generating for them.")
    return {
        "table": df, "missing_topics": missing, "underrepresented": under, "overrepresented": over,
        "difficulty_imbalance": diff_imbalance, "chapters_without_sources": no_src, "recommendations": recs,
        "note": "Distribution of generated questions only - not ICAI examination weightage.",
    }


def _by_chapter(qs):
    out: dict[str, list] = {}
    for q in qs:
        out.setdefault(q.chapter, []).append(q)
    return out


def distributions(qs: list[QuestionRecord]) -> dict[str, dict]:
    df = questions_frame(qs)
    if df.empty:
        return {}
    return {c: df[c].value_counts(dropna=False).to_dict() for c in
            ("status", "verdict", "provenance", "difficulty", "question_type", "mode", "chapter", "evidence_status")}


def feedback_analysis(qs: list[QuestionRecord]) -> dict:
    """Recurring failure patterns from human feedback + pipeline history, used to drive prompt changes."""
    fb = [f for q in qs for f in q.feedback]
    decided = [q for q in qs if q.status in ("APPROVED", "REJECTED") and any(f.decision in ("approve", "reject") for f in q.feedback)]
    by_prompt: dict[str, list] = {}
    for q in decided:
        by_prompt.setdefault(f"{q.generation.model} / setter {q.generation.prompt_versions.get('question_setter', '?')}", []).append(q.status == "APPROVED")
    issue_cats = Counter(i.category for q in qs if q.reviews.examiner for i in q.reviews.examiner.issues)
    failed_checks = Counter(c.name for q in qs for c in q.reliability.checks if not c.passed)
    revision_reasons = Counter(
        r.split(":")[0] for q in qs for h in q.history if h.action == "assessed" for r in h.detail.get("reasons", []))
    return {
        "n_feedback": len(fb),
        "rejection_reasons": dict(Counter(f.rejection_reason for f in fb if f.decision == "reject")),
        "difficulty_correction_pairs": dict(Counter(
            f"{h.detail['changes']['difficulty']['before']}->{h.detail['changes']['difficulty']['after']}"
            for q in qs for h in q.history if h.action == "edit" and "difficulty" in h.detail.get("changes", {}))),
        "factual_issues": [f.factual_issue for f in fb if f.factual_issue],
        "human_approval_rate": round(sum(q.status == "APPROVED" for q in decided) / len(decided), 3) if decided else None,
        "approval_by_model_prompt": {k: round(sum(v) / len(v), 3) for k, v in by_prompt.items()},
        "examiner_issue_categories": dict(issue_cats.most_common()),
        "failed_checks": dict(failed_checks.most_common()),
        "pipeline_flag_reasons": dict(revision_reasons.most_common(15)),
    }
