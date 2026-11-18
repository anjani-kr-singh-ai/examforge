"""Export with provenance: JSON (full records), CSV (one flat row per question), Markdown paper + answer key."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schemas import QuestionRecord


def _sources(q: QuestionRecord) -> str:
    return "; ".join(f"{c.title} [{c.document_type}, {c.attempt}] pp.{c.page_start}-{c.page_end} {c.section} <{c.source_url}>"
                     for c in q.citations)


def flat_row(q: QuestionRecord) -> dict:
    row = {"id": q.id, "subject": q.subject, "chapter": q.chapter, "topic": q.topic, "difficulty": q.difficulty,
           "question_type": q.question_type, "mode": q.mode, "attempt": q.attempt, "question": q.mcq.question}
    for o in q.mcq.options:
        row[f"option_{o.label}"] = o.text
    row |= {
        "correct_answers": ",".join(q.mcq.correct_labels), "explanation": q.mcq.explanation,
        "option_explanations": json.dumps({o.label: o.explanation for o in q.mcq.options}, ensure_ascii=False),
        "sources": _sources(q), "page_refs": "; ".join(f"{c.document_id}:{c.page_start}-{c.page_end}" for c in q.citations),
        "evidence_status": q.reliability.evidence_status, "quote_grounding": q.reliability.quote_grounding,
        "reliability_verdict": q.reliability.verdict, "review_status": q.status, "provenance": q.provenance,
        "human_edited": q.human_edited, "model": q.generation.model, "mock": q.generation.mock,
        "prompt_versions": json.dumps(q.generation.prompt_versions),
    }
    return row


def export(qs: list[QuestionRecord], out_dir: Path, stem: str = "questions") -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / f"{stem}.json", "csv": out_dir / f"{stem}.csv",
             "paper": out_dir / f"{stem}_paper.md", "answer_key": out_dir / f"{stem}_answer_key.md"}
    paths["json"].write_text(json.dumps([q.model_dump() for q in qs], indent=1, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([flat_row(q) for q in qs]).to_csv(paths["csv"], index=False, encoding="utf-8-sig")
    paths["paper"].write_text(paper_markdown(qs), encoding="utf-8")
    paths["answer_key"].write_text(answer_key_markdown(qs), encoding="utf-8")
    return paths


def _banner(qs) -> str:
    approved = sum(q.status == "APPROVED" for q in qs)
    mock = any(q.generation.mock for q in qs)
    return (f"> {len(qs)} AI-generated question(s); {approved} human-approved. "
            + ("**MOCK provider output - not real model generations.** " if mock else "")
            + "Unapproved items are drafts.\n")


def paper_markdown(qs: list[QuestionRecord]) -> str:
    lines = ["# Practice Questions - CA Final Indirect Tax Laws", "", _banner(qs)]
    for i, q in enumerate(qs, 1):
        tag = "select all that apply" if q.question_type == "multiple" else "single correct answer"
        lines += [f"**Q{i}.** ({q.chapter} / {q.topic}; {q.difficulty}; {tag})", "", q.mcq.question, ""]
        if q.mcq.assumptions:
            lines += ["*Assumptions:* " + "; ".join(q.mcq.assumptions), ""]
        lines += [f"- ({o.label}) {o.text}" for o in q.mcq.options] + [""]
    return "\n".join(lines)


def answer_key_markdown(qs: list[QuestionRecord]) -> str:
    lines = ["# Answer Key & Explanations", "", _banner(qs)]
    for i, q in enumerate(qs, 1):
        lines += [f"## Q{i} - Answer: {', '.join(q.mcq.correct_labels)}  (`{q.id}`, status {q.status}, evidence {q.reliability.evidence_status})",
                  "", q.mcq.explanation, ""]
        lines += [f"- ({o.label}) {'Correct' if o.is_correct else 'Incorrect'}: {o.explanation}" for o in q.mcq.options]
        lines += ["", "**Sources:**"] + [f"- {c.title} ({c.document_type}, {c.attempt}), PDF pp. {c.page_start}-{c.page_end}, "
                                          f"{c.section} - \"{c.quote[:160]}\" [{c.source_url}]" for c in q.citations] + [""]
    return "\n".join(lines)
