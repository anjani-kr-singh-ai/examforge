"""Deterministic checks (no LLM): structure, answer key, explanations, item-writing cues, evidence
grounding, source-version compatibility, and multi-level duplicate detection.

Everything here is reproducible and cheap, so it runs first and gates the (expensive) LLM reviewers.
"""
from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher

import numpy as np

from .schemas import LABELS, MCQ, CheckResult, Chunk, Citation, GenerationRequest, QuestionRecord

QUOTE_MATCH_THRESHOLD = 0.8
LABEL_REF_RE = re.compile(r"\b(all|none|both)\s+of\s+the\s+above\b|\b(both|either)\s*\(?[a-d]\)?\s*(and|or)\s*\(?[a-d]\)?\b|\boption\s*\(?[a-d]\)?\b", re.I)
KEY_CLAIM_RE = re.compile(r"\b(?:option|answer)\s*\(?([A-D])\)?\s*(?:is|are)\s*(?:the\s+)?correct", re.I)
ABSOLUTE_RE = re.compile(r"\b(always|never|only|all|none|under no circumstances|in every case)\b", re.I)


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w₹%\s]", " ", t.lower())).strip()


# --------------------------------------------------------------------------- evidence grounding
def quote_match(quote: str, source: str) -> float:
    """1.0 if the (normalised) quote occurs in the source; else the longest-common-substring coverage.
    Ellipses split the quote into segments that are matched independently."""
    src = _norm(source)
    segs = [_norm(s) for s in re.split(r"\.{3}|…", quote) if _norm(s)]
    if not segs or not src:
        return 0.0
    scores = []
    for seg in segs:
        if seg in src:
            scores.append(1.0)
            continue
        m = SequenceMatcher(None, seg, src, autojunk=False).find_longest_match(0, len(seg), 0, len(src))
        scores.append(m.size / len(seg))
    return round(sum(scores) / len(scores), 3)


def build_citations(mcq: MCQ, context: dict[str, Chunk]) -> list[Citation]:
    """Resolve evidence quotes to citations. Page/section/document come from chunk metadata, never from
    the model, so a citation can be wrong only by pointing at the wrong chunk - which the quote check catches."""
    cits = []
    for ev in mcq.evidence:
        c = context.get(ev.chunk_id)
        if c is None:
            continue
        score = quote_match(ev.quote, c.text)
        cits.append(Citation(
            chunk_id=c.chunk_id, document_id=c.document_id, title=c.title, document_type=c.document_type,
            attempt=c.attempt, page_start=c.page_start, page_end=c.page_end, section=c.section,
            source_url=c.source_url, quote=ev.quote, quote_found=score >= QUOTE_MATCH_THRESHOLD, quote_match=score,
        ))
    return cits


# --------------------------------------------------------------------------- deterministic checks
def deterministic_checks(mcq: MCQ, req: GenerationRequest, context: dict[str, Chunk], allowed_docs: set[str] | None = None,
                         historical_docs: set[str] = frozenset()) -> tuple[list[CheckResult], list[Citation], float]:
    """Return (checks, citations, quote_grounding). `context` maps short ids (C1..) to chunks.
    allowed_docs/historical_docs come from the same document-level applicability rules as retrieval."""
    checks: list[CheckResult] = []

    def add(name: str, ok: bool, detail: str = "", severity: str = "error"):
        checks.append(CheckResult(name=name, passed=ok, severity=severity if not ok else "info", detail=detail))

    labels = [o.label for o in mcq.options]
    add("schema_required_fields", bool(req.subject and req.chapter and req.topic and req.difficulty),
        "subject/chapter/topic/difficulty must be set")
    add("question_text_present", len(mcq.question.strip()) >= 15, f"{len(mcq.question.strip())} chars")
    add("exactly_four_options", len(mcq.options) == 4, f"{len(mcq.options)} options")
    add("option_labels_valid", labels == list(LABELS[: len(labels)]) and len(set(labels)) == len(labels),
        f"labels {labels}")
    add("option_text_present", all(o.text.strip() for o in mcq.options), "every option needs text")

    valid_refs = set(mcq.correct_labels) <= set(labels) and len(mcq.correct_labels) == len(set(mcq.correct_labels))
    add("answer_refs_valid", bool(mcq.correct_labels) and valid_refs,
        f"correct_labels {mcq.correct_labels} vs options {labels}")
    flagged = sorted(o.label for o in mcq.options if o.is_correct)
    add("answer_flags_consistent", flagged == sorted(mcq.correct_labels),
        f"is_correct flags {flagged} vs key {sorted(mcq.correct_labels)}")
    n = len(mcq.correct_labels)
    if req.question_type == "single":
        add("question_type_matches_answer_count", n == 1, f"single-correct needs 1 answer, got {n}")
    else:
        add("question_type_matches_answer_count", 2 <= n <= 3, f"multiple-correct needs 2-3 answers, got {n}")

    # exact match after normalisation only: good Hard distractors legitimately differ by one word
    # ("shall" vs "shall not", 90 vs 180 days), so fuzzy similarity here would punish them
    norms = [re.sub(r"\b(a|an|the)\b\s*", "", _norm(o.text)) for o in mcq.options]
    dup_opts = [(a, b) for i, a in enumerate(labels) for j, b in enumerate(labels) if i < j and norms[i] == norms[j]]
    add("options_distinct", not dup_opts, f"near-identical options {dup_opts}")

    add("explanation_present", len(mcq.explanation.strip()) >= 40, f"{len(mcq.explanation.strip())} chars")
    missing = [o.label for o in mcq.options if len(o.explanation.strip()) < 10]
    add("option_explanations_complete", not missing, f"missing per-option explanation for {missing}")
    claimed = {m.upper() for m in KEY_CLAIM_RE.findall(mcq.explanation)}
    add("explanation_consistent_with_key", claimed <= set(mcq.correct_labels),
        f"explanation calls {sorted(claimed - set(mcq.correct_labels))} correct but key is {mcq.correct_labels}")
    contradict = [o.label for o in mcq.options if o.is_correct and re.match(r"\s*(incorrect|wrong|not correct)", o.explanation, re.I)] + \
                 [o.label for o in mcq.options if not o.is_correct and re.match(r"\s*(correct|right)\b", o.explanation, re.I)]
    add("option_explanations_match_key", not contradict, f"explanation verdict contradicts key for {contradict}")

    # item-writing cues (warnings: a human may legitimately keep them)
    refs = [o.label for o in mcq.options if LABEL_REF_RE.search(o.text)]
    add("no_label_dependent_options", not refs, f"options {refs} reference other options/'all of the above'")
    if req.question_type == "single" and n == 1 and len(mcq.options) == 4:
        correct_len = len(next((o.text for o in mcq.options if o.is_correct), ""))
        others = [len(o.text) for o in mcq.options if not o.is_correct]
        mean_other = sum(others) / len(others) if others else 0
        cue = mean_other and correct_len > 1.6 * mean_other and correct_len - mean_other > 40
        add("no_length_cue", not cue, f"correct option {correct_len} chars vs distractor mean {mean_other:.0f}", "warning")
    abs_wrong = [o.label for o in mcq.options if not o.is_correct and ABSOLUTE_RE.search(o.text)]
    abs_right = any(ABSOLUTE_RE.search(o.text) for o in mcq.options if o.is_correct)
    add("no_absolute_word_cue", not (abs_wrong and not abs_right),
        f"absolute words only in distractors {abs_wrong}", "warning")

    # evidence grounding
    add("evidence_present", bool(mcq.evidence), "no evidence quotes")
    unknown = [e.chunk_id for e in mcq.evidence if e.chunk_id not in context]
    add("evidence_refs_valid", not unknown, f"evidence cites unknown chunk ids {unknown}")
    citations = build_citations(mcq, context)
    found = sum(c.quote_found for c in citations)
    grounding = round(found / len(mcq.evidence), 3) if mcq.evidence else 0.0
    add("evidence_quotes_grounded", grounding >= 0.5,
        f"{found}/{len(mcq.evidence)} quotes found verbatim (>= {QUOTE_MATCH_THRESHOLD} match) in cited chunks")
    if 0.5 <= grounding < 1.0:
        add("all_quotes_grounded", False, f"{len(mcq.evidence) - found} quote(s) not found in source", "warning")

    # source-version compatibility
    if allowed_docs is not None:
        off = sorted({c.document_id for c in citations if c.document_id not in allowed_docs})
        add("source_version_compatible", not off, f"cites documents not applicable to {req.attempt or 'the request'}: {off}")
        hist = sorted({c.document_id for c in citations if c.document_id in historical_docs})
        if hist:
            add("cites_historical_material", False, f"cites material not tagged for {req.attempt}: {hist}", "warning")
    return checks, citations, grounding


def passed(checks: list[CheckResult]) -> bool:
    return all(c.passed or c.severity != "error" for c in checks)


# --------------------------------------------------------------------------- duplicates
def question_signature(mcq: MCQ) -> str:
    # all options (sorted, label-free): statement-style items often share a generic stem
    return f"{mcq.question} || {' | '.join(sorted(o.text for o in mcq.options))}"


class DuplicateDetector:
    """Three levels: exact hash -> normalised text -> embedding cosine. Similar-but-not-duplicate questions
    are surfaced for a human instead of rejected: two items can test one concept in different ways."""

    def __init__(self, embedder, duplicate_threshold: float = 0.92, similar_threshold: float = 0.82):
        self.embedder, self.dup_t, self.sim_t = embedder, duplicate_threshold, similar_threshold

    def check(self, mcq: MCQ, others: list[QuestionRecord]) -> tuple[str | None, float, list[dict]]:
        sig = question_signature(mcq)
        h, n = hashlib.sha256(sig.encode()).hexdigest(), _norm(sig)
        for o in others:
            osig = question_signature(o.mcq)
            if hashlib.sha256(osig.encode()).hexdigest() == h:
                return o.id, 1.0, [{"id": o.id, "similarity": 1.0, "level": "exact"}]
            if _norm(osig) == n:
                return o.id, 1.0, [{"id": o.id, "similarity": 1.0, "level": "normalized"}]
        if not others:
            return None, 0.0, []
        vecs = self.embedder([sig] + [question_signature(o.mcq) for o in others])
        sims = vecs[1:] @ vecs[0]
        order = np.argsort(-sims)
        similar = [{"id": others[i].id, "similarity": round(float(sims[i]), 3), "level": "semantic"}
                   for i in order if sims[i] >= self.sim_t][:5]
        best = float(sims[order[0]])
        dup = others[order[0]].id if best >= self.dup_t else None
        return dup, round(best, 3), similar


def composite_score(r) -> float | None:
    """Review-priority index in [0,1] = mean of the available components:
    deterministic pass, quote grounding, evidence verdict (1/.5/0), blind-solver agreement, difficulty match.
    It is an unweighted heuristic for sorting the review queue - NOT a calibrated probability that the
    question is correct. The individual components stay visible next to it."""
    parts: list[float] = [1.0 if r.deterministic_passed else 0.0, r.quote_grounding]
    parts.append({"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}.get(r.evidence_status, np.nan))
    if r.answer_agreement is not None:
        parts.append(1.0 if r.answer_agreement else 0.0)
    if r.difficulty_match is not None:
        parts.append(1.0 if r.difficulty_match else 0.0)
    vals = [p for p in parts if not np.isnan(p)]
    return round(sum(vals) / len(vals), 3) if vals else None
