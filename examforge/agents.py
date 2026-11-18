"""AI Examiner: independent reviewer agents + a deterministic Review Coordinator.

Agents (each = separate versioned prompt + separate structured schema + separate LLM call):
  Question Setter     -> generation.py (writes; never reviews its own output)
  Critical Examiner   -> adversarial review of key, ambiguity, distractors, explanation
  Evidence Verifier   -> claim-by-claim support against the retrieved material
  Difficulty Reviewer -> solves the question BLIND (no key) and estimates difficulty
  Review Coordinator  -> plain Python rules below; no LLM decides acceptance

Why plain functions instead of LangGraph: the flow is a fixed sequence with one bounded revision loop.
A graph framework would add state plumbing without adding capability; the loop lives in pipeline.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .generation import base_vars
from .llm import LLMClient
from .schemas import (
    MCQ, CheckResult, DifficultyReview, EvidenceReview, ExaminerReview, GenerationRequest, JudgeScores,
)

LEVEL = {"Easy": 0, "Medium": 1, "Hard": 2}


def question_block(mcq: MCQ, with_key: bool = True) -> str:
    lines = [f"Question: {mcq.question}"]
    if mcq.assumptions:
        lines.append("Stated assumptions: " + "; ".join(mcq.assumptions))
    for o in mcq.options:
        lines.append(f"({o.label}) {o.text}")
    if with_key:
        lines.append(f"Answer key: {', '.join(mcq.correct_labels)}")
        lines.append(f"Explanation: {mcq.explanation}")
        for o in mcq.options:
            lines.append(f"Why ({o.label}) is {'correct' if o.is_correct else 'incorrect'}: {o.explanation}")
    return "\n".join(lines)


def evidence_block(mcq: MCQ) -> str:
    return "\n".join(f"{e.chunk_id}: \"{e.quote}\"" for e in mcq.evidence) or "(none cited)"


def _vars(req: GenerationRequest, mcq: MCQ, context_text: str, ctx) -> dict:
    return base_vars(req) | {
        "question_block": question_block(mcq), "question_blind": question_block(mcq, with_key=False),
        "evidence_block": evidence_block(mcq), "context": context_text, "_mcq": mcq, "_ctx": ctx,
    }


def critical_examiner(llm: LLMClient, req, mcq, context_text, ctx) -> ExaminerReview:
    return llm.run("critical_examiner", ExaminerReview, _vars(req, mcq, context_text, ctx))


def evidence_verifier(llm: LLMClient, req, mcq, context_text, ctx) -> EvidenceReview:
    return llm.run("evidence_verifier", EvidenceReview, _vars(req, mcq, context_text, ctx))


def difficulty_reviewer(llm: LLMClient, req, mcq, context_text, ctx) -> DifficultyReview:
    return llm.run("difficulty_reviewer", DifficultyReview, _vars(req, mcq, context_text, ctx))


def judge(llm: LLMClient, req, mcq, context_text, ctx) -> JudgeScores:
    return llm.run("judge", JudgeScores, _vars(req, mcq, context_text, ctx))


# --------------------------------------------------------------------------- coordinator
@dataclass
class Decision:
    verdict: str  # passed_for_human_review | needs_revision | rejected
    reasons: list[str] = field(default_factory=list)  # human-readable, shown in the reliability report
    revision_issues: list[str] = field(default_factory=list)  # fed to the reviser
    disagreements: list[str] = field(default_factory=list)
    answer_agreement: bool | None = None
    difficulty_match: bool | None = None


def coordinate(
    mcq: MCQ, req: GenerationRequest, checks: list[CheckResult], grounding: float,
    examiner: ExaminerReview | None, evidence: EvidenceReview | None, difficulty: DifficultyReview | None,
    duplicate_of: str | None,
) -> Decision:
    """Deterministic acceptance rules over deterministic checks + structured reviewer outputs.

    rejected            : duplicate of an existing question; examiner verdict 'reject'; or the keyed answer
                          is contradicted by the evidence verifier AND by the blind solver
    needs_revision      : any failed error-level check; examiner critical/major issue or verdict 'revise';
                          evidence not fully supported; blind solver disagrees with the key; examiner
                          disputes the key; requested vs estimated difficulty differs
    passed_for_human_review : none of the above (warnings stay visible). Never 'approved' - only humans approve.
    """
    d = Decision(verdict="passed_for_human_review")
    if duplicate_of:
        d.verdict = "rejected"
        d.reasons.append(f"duplicate of existing question {duplicate_of}")
        return d

    failed = [c for c in checks if not c.passed and c.severity == "error"]
    for c in failed:
        d.reasons.append(f"check failed: {c.name} ({c.detail})")
        d.revision_issues.append(f"Validation failure '{c.name}': {c.detail}")
    for c in checks:
        if not c.passed and c.severity == "warning":
            d.reasons.append(f"warning: {c.name} ({c.detail})")
            d.revision_issues.append(f"Item-writing warning '{c.name}': {c.detail}")

    key = set(mcq.correct_labels)
    disputed: list[str] = []
    if examiner:
        disputed = [v.label for v in examiner.option_verdicts
                    if (v.label in key and v.verdict != "correct") or (v.label not in key and v.verdict != "incorrect")]
        if disputed:
            d.reasons.append(f"examiner disputes key on options {disputed}")
            d.revision_issues.append(f"Examiner judged options {disputed} differently from the key (correct/arguable).")
        for i in examiner.issues:
            if i.severity in ("critical", "major"):
                d.reasons.append(f"examiner {i.severity}: {i.category} - {i.detail}")
                d.revision_issues.append(f"Examiner ({i.category}, {i.severity}): {i.detail}")
        if examiner.verdict == "revise" and not d.revision_issues:
            d.revision_issues.append(f"Examiner requested revision: {examiner.summary}")

    if evidence:
        if evidence.verdict != "supported":
            d.reasons.append(f"evidence verifier: {evidence.verdict}")
            for c in evidence.claims:
                if c.support != "supported":
                    d.revision_issues.append(f"Claim {c.support} by the material: '{c.claim}' - {c.comment}")
        if not evidence.answer_supported:
            d.revision_issues.append("The keyed answer is not supported by the reference material.")
        if evidence.outdated_or_conflicting:
            d.reasons.append(f"evidence verifier flags outdated/conflicting material: {evidence.outdated_note}")

    if difficulty:
        d.answer_agreement = set(difficulty.answer_labels) == key
        if not d.answer_agreement:
            d.reasons.append(f"blind solver answered {sorted(difficulty.answer_labels)}, key is {sorted(key)}")
            d.revision_issues.append(
                f"An independent solver (without the key) chose {sorted(difficulty.answer_labels)} instead of "
                f"{sorted(key)}: the question may be ambiguous or the key wrong.")
        d.difficulty_match = difficulty.estimated_difficulty == req.difficulty
        if not d.difficulty_match:
            d.reasons.append(f"difficulty: requested {req.difficulty}, estimated {difficulty.estimated_difficulty}")
            d.revision_issues.append(
                f"Requested difficulty is {req.difficulty} but reviewer estimated {difficulty.estimated_difficulty}: "
                f"{difficulty.rationale} Suggestion: {difficulty.suggestion}")

    # disagreements between independent reviewers are surfaced, not silently resolved
    if examiner and evidence and examiner.verdict == "accept" and evidence.verdict == "unsupported":
        d.disagreements.append("examiner accepted but evidence verifier found the question unsupported")
    if examiner and difficulty and examiner.verdict == "accept" and d.answer_agreement is False:
        d.disagreements.append("examiner accepted but blind solver disagreed with the key")
    d.reasons += [f"disagreement: {x}" for x in d.disagreements]

    contradicted = evidence is not None and evidence.verdict == "unsupported" and d.answer_agreement is False
    if (examiner and examiner.verdict == "reject") or contradicted:
        d.verdict = "rejected"
        d.reasons.append("rejected: " + ("examiner verdict reject" if examiner and examiner.verdict == "reject"
                                         else "answer contradicted by evidence verifier and blind solver"))
    elif failed or (examiner and examiner.verdict != "accept") or (evidence and evidence.verdict != "supported") \
            or d.answer_agreement is False or disputed or (examiner and any(i.severity in ("critical", "major") for i in examiner.issues)):
        d.verdict = "needs_revision"
    elif d.difficulty_match is False or grounding < 1.0:
        d.verdict = "needs_revision"  # soft: revisable; if revisions run out it still goes to a human with flags
    return d


def is_blocking(d: Decision) -> bool:
    """After revisions are exhausted, only these keep a question out of the human review queue as 'needs revision'."""
    soft = all(r.startswith(("difficulty:", "warning:")) for r in d.reasons)
    return d.verdict == "needs_revision" and not soft
