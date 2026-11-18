"""Offline MOCK provider - deterministic stand-in so the full workflow runs without API credentials.

It builds outputs from the retrieved chunks (questions quote real source sentences, so quote-grounding
passes) and its reviewers are lenient. Everything it writes is prefixed "[MOCK]".
Mock output demonstrates the plumbing only; it says nothing about real model quality, and benchmark
reports produced with it are labelled as mock runs.
"""
from __future__ import annotations

import re

from .llm import LLMResponse
from .schemas import (
    ClaimCheck, DifficultyReview, DistractorBatch, DistractorDraft, DistractorSet, DraftOption, EvidenceQuote,
    EvidenceReview, ExaminerReview, JudgeScores, MCQDraft, OptionVerdict, StemBatch, StemDraft,
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;])\s+|\n", text) if 12 <= len(s.split()) <= 45]


def _mutate(s: str, k: int) -> str:
    """Make the k-th distinct wrong version of a true sentence (number change / polarity flip / added condition)."""
    cands = []
    nums = re.findall(r"\d+", s)
    if nums:
        n = nums[0]
        cands.append(s.replace(n, str(int(n) * 2 if int(n) < 1000 else int(n) // 2), 1))
    for a, b in ((" shall not ", " shall "), (" shall ", " shall not "), (" is not ", " is "), (" is ", " is not "), (" may ", " may not ")):
        if a in s:
            cands.append(s.replace(a, b, 1))
            break
    for a, b in (("registered", "unregistered"), ("supplier", "recipient"), ("goods", "services"), ("credit", "refund")):
        if f" {a} " in s:
            cands.append(s.replace(f" {a} ", f" {b} ", 1))
            break
    cands.append(f"{s.rstrip('.')} except where the recipient is a composition taxpayer.")
    cands.append(f"{s.rstrip('.')} subject to prior approval of the proper officer.")
    uniq = list(dict.fromkeys(c for c in cands if c != s))
    return uniq[k % len(uniq)]


class MockProvider:
    name = "mock"
    model = "mock-deterministic-v1"
    is_mock = True

    def generate(self, system: str, user: str, schema, variables: dict) -> LLMResponse:
        handler = getattr(self, f"_{schema.__name__}", None)
        if handler is None:
            raise NotImplementedError(f"mock has no handler for {schema.__name__}")
        return LLMResponse(handler(variables), 0, 0, self.model)

    # --- generation
    def _StemBatch(self, v) -> StemBatch:
        ctx = v["_ctx"]
        n, multiple = int(v["n"]), v["question_type"] == "multiple"
        pool = [(cid, s) for cid, c in ctx.items() for s in _sentences(c.text)]
        qs = []
        for i in range(n):
            if len(pool) < (i + 1) * (3 if multiple else 1):
                break
            picks = pool[i * 3: i * 3 + 2] if multiple else [pool[i * 3 if len(pool) > i * 3 else i]]
            qs.append(StemDraft(
                question=f"[MOCK] With reference to {v['topic']}, which of the following statement(s) is/are correct as per the study material?"
                + (" (select all that apply)" if multiple else "") + f" [variant {i + 1}]",
                correct_options=[s for _, s in picks],
                correct_option_explanations=[f"[MOCK] The material states this in {cid}." for cid, _ in picks],
                explanation="[MOCK] The correct statement(s) are reproduced from the reference material: "
                + " ".join(f"({cid}) {s}" for cid, s in picks),
                evidence=[EvidenceQuote(chunk_id=cid, quote=s) for cid, s in picks],
                assumptions=[], concepts_tested=[v["topic"]],
            ))
        return StemBatch(insufficient_context=len(qs) < n, insufficient_reason="" if len(qs) == n else "[MOCK] not enough distinct sentences", questions=qs)

    def _DistractorBatch(self, v) -> DistractorBatch:
        cats = ["threshold_or_time_limit", "condition_misread", "exception_misapplied"]
        items = []
        for it in v["_items"]:
            base = it["correct_options"][0]
            ds = [DistractorDraft(text=_mutate(base, k),
                                  misconception_category=cats[k % 3],
                                  why_incorrect=f"[MOCK] Altered version of the source statement ({cats[k % 3]}).")
                  for k in range(it["distractors_needed"])]
            items.append(DistractorSet(question_index=it["question_index"], distractors=ds))
        return DistractorBatch(items=items)

    def _MCQDraft(self, v) -> MCQDraft:
        m = v["_mcq"]
        return MCQDraft(
            question=m.question, correct_labels=m.correct_labels, explanation=m.explanation, evidence=m.evidence,
            options=[DraftOption(label=o.label, text=o.text, explanation=o.explanation or "[MOCK] see source",
                                 misconception_category=o.misconception_category) for o in m.options],
            assumptions=m.assumptions, concepts_tested=m.concepts_tested, change_summary="[MOCK] no changes",
        )

    # --- reviewers (lenient by design; they do not judge anything)
    def _ExaminerReview(self, v) -> ExaminerReview:
        m = v["_mcq"]
        return ExaminerReview(
            option_verdicts=[OptionVerdict(label=o.label, verdict="correct" if o.is_correct else "incorrect", plausibility=3,
                                           comment="[MOCK]") for o in m.options],
            issues=[], clarity=3, distractor_quality=3, verdict="accept", summary="[MOCK] no real review performed")

    def _EvidenceReview(self, v) -> EvidenceReview:
        m = v["_mcq"]
        return EvidenceReview(
            claims=[ClaimCheck(claim=e.quote[:80], support="supported", chunk_ids=[e.chunk_id], comment="[MOCK]") for e in m.evidence],
            answer_supported=True, explanation_supported=True, distractors_incorrect_per_source=True,
            outdated_or_conflicting=False, verdict="supported")

    def _DifficultyReview(self, v) -> DifficultyReview:
        m = v["_mcq"]
        return DifficultyReview(answer_labels=list(m.correct_labels), confidence=3, reasoning_steps=1, provisions_involved=[],
                                estimated_difficulty=v["difficulty"], rationale="[MOCK] echoes requested difficulty")

    def _JudgeScores(self, v) -> JudgeScores:
        return JudgeScores(relevance=3, factual_correctness=3, clarity=3, difficulty_fit=3, distractor_quality=3,
                           answer_explanation_consistency=3, rationale="[MOCK] fixed scores")
