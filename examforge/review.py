"""Question bank (JSON file) + human-in-the-loop review actions with an audit trail.

Rules enforced here:
  * only a human action can set APPROVED; the pipeline never does
  * approval is refused while error-level deterministic checks fail (fix first, then approve)
  * every human change is recorded in history (who, when, what changed) and the AI original is kept
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .schemas import MCQ, Feedback, HistoryEntry, Option, QuestionRecord


class QuestionBank:
    def __init__(self, path: Path):
        self.path = path
        self._cache: dict[str, QuestionRecord] | None = None

    def _load(self) -> dict[str, QuestionRecord]:
        if self._cache is None:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._cache = {r["id"]: QuestionRecord(**r) for r in raw}
            else:
                self._cache = {}
        return self._cache

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([r.model_dump() for r in self._load().values()], indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic: a crash mid-write never corrupts the bank

    def all(self) -> list[QuestionRecord]:
        return list(self._load().values())

    def get(self, qid: str) -> QuestionRecord:
        try:
            return self._load()[qid]
        except KeyError:
            raise KeyError(f"unknown question id {qid}") from None

    def save(self, *recs: QuestionRecord) -> None:
        for r in recs:
            self._load()[r.id] = r
        self._flush()

    def by_chapter(self, subject: str, chapter: str, topic: str | None = None, exclude_rejected: bool = True) -> list[QuestionRecord]:
        return [r for r in self.all() if r.subject == subject and r.chapter == chapter
                and (topic is None or r.topic == topic) and not (exclude_rejected and r.status == "REJECTED")]


# --------------------------------------------------------------------------- human actions
def _log(rec: QuestionRecord, reviewer: str, action: str, **detail) -> None:
    rec.history.append(HistoryEntry(actor=f"human:{reviewer}", action=action, detail=detail))


def approve(bank: QuestionBank, qid: str, reviewer: str, comment: str = "") -> QuestionRecord:
    rec = bank.get(qid)
    if not rec.reliability.deterministic_passed:
        raise ValueError("cannot approve: deterministic checks are failing - edit the question first")
    rec.status = "APPROVED"
    rec.feedback.append(Feedback(reviewer=reviewer, decision="approve", comment=comment))
    _log(rec, reviewer, "approve", comment=comment)
    bank.save(rec)
    return rec


def reject(bank: QuestionBank, qid: str, reviewer: str, reason: str, comment: str = "", factual_issue: str | None = None) -> QuestionRecord:
    rec = bank.get(qid)
    rec.status = "REJECTED"
    rec.feedback.append(Feedback(reviewer=reviewer, decision="reject", rejection_reason=reason, comment=comment, factual_issue=factual_issue))
    _log(rec, reviewer, "reject", reason=reason, comment=comment)
    bank.save(rec)
    return rec


def add_comment(bank: QuestionBank, qid: str, reviewer: str, comment: str, corrected_difficulty: str | None = None) -> QuestionRecord:
    rec = bank.get(qid)
    rec.feedback.append(Feedback(reviewer=reviewer, decision="comment", comment=comment, corrected_difficulty=corrected_difficulty))
    _log(rec, reviewer, "comment", comment=comment)
    bank.save(rec)
    return rec


def apply_edit(rec: QuestionRecord, reviewer: str, *, question: str | None = None, options: list[dict] | None = None,
               correct_labels: list[str] | None = None, explanation: str | None = None,
               difficulty: str | None = None, comment: str = "") -> QuestionRecord:
    """Edit in place (caller re-validates and saves). Keeps the AI original on first edit."""
    if rec.original_mcq is None:
        rec.original_mcq = rec.mcq.model_copy(deep=True)
    before = rec.mcq.model_dump()
    m = rec.mcq
    new_opts = [Option(label=o["label"], text=o["text"], is_correct=False, explanation=o.get("explanation", ""),
                       misconception_category=o.get("misconception_category")) for o in options] if options is not None else m.options
    keys = list(correct_labels) if correct_labels is not None else m.correct_labels
    for o in new_opts:
        o.is_correct = o.label in keys
    rec.mcq = MCQ(question=question if question is not None else m.question, options=new_opts, correct_labels=keys,
                  explanation=explanation if explanation is not None else m.explanation, evidence=m.evidence,
                  assumptions=m.assumptions, concepts_tested=m.concepts_tested)
    changes = {k: {"before": before[k], "after": v} for k, v in rec.mcq.model_dump().items() if before[k] != v}
    fb = Feedback(reviewer=reviewer, decision="edit", comment=comment)
    if difficulty and difficulty != rec.difficulty:
        changes["difficulty"] = {"before": rec.difficulty, "after": difficulty}
        fb.corrected_difficulty = difficulty  # type: ignore[assignment]
        rec.difficulty = difficulty  # type: ignore[assignment]
    rec.human_edited = True
    rec.status = "NEEDS_REVIEW" if rec.status == "APPROVED" else rec.status  # an edit voids a prior approval
    rec.feedback.append(fb)
    _log(rec, reviewer, "edit", changes=changes, comment=comment)
    return rec
