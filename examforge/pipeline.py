"""End-to-end orchestration.

request -> validate input -> amendment-aware retrieval -> (insufficient? stop with explicit status)
        -> Question Setter (stems) -> Distractor Generator -> assemble/shuffle
        -> per question: deterministic checks -> duplicate check -> [gate] -> Examiner + Evidence Verifier
           + blind Difficulty Reviewer -> Coordinator -> bounded revision loop -> reliability report -> bank
Every LLM failure is caught per stage and recorded; one bad question never aborts the batch.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from . import agents, generation, validation
from .config import Settings, load_yaml
from .ingestion import load_manifest
from .llm import LLMClient, LLMError, make_provider
from .retrieval import Retriever, VectorIndex, make_embedder, select_documents
from .review import QuestionBank, apply_edit
from .schemas import (
    Chunk, GenerationMeta, GenerationRequest, HistoryEntry, QuestionRecord, ReliabilityReport, RetrievalResult, Reviews,
)


RANK = {"passed_for_human_review": 0, "needs_revision": 1, "insufficient_evidence": 2, "rejected": 3}


@dataclass
class RunResult:
    run_id: str
    request: GenerationRequest
    retrieval: RetrievalResult | None
    questions: list[QuestionRecord] = field(default_factory=list)
    status: str = "ok"  # ok | insufficient_evidence | failed
    message: str = ""
    stats: dict = field(default_factory=dict)
    events: list[str] = field(default_factory=list)


class Engine:
    def __init__(self, settings: Settings, llm: LLMClient, index: VectorIndex, bank: QuestionBank, embedder,
                 run_reviewers: bool = True, run_judge: bool = False):
        self.s, self.llm, self.index, self.bank, self.embedder = settings, llm, index, bank, embedder
        self.run_reviewers, self.run_judge = run_reviewers, run_judge
        self.docs = load_manifest(settings.manifest_path) if settings.manifest_path.exists() else []
        self.retriever = Retriever(index, self.docs, settings.top_k, settings.min_relevance)
        self.dups = validation.DuplicateDetector(embedder, settings.duplicate_threshold, settings.similar_threshold)

    @classmethod
    def build(cls, settings: Settings | None = None, embedder_name: str = "onnx", **kw) -> "Engine":
        s = settings or Settings()
        reviewer = make_provider(s, reviewer=True) if (s.reviewer_provider or s.reviewer_model) else None
        versions = load_yaml("generation.yaml")["active_prompts"] | s.prompt_versions
        llm = LLMClient(make_provider(s), reviewer, s.max_retries, versions, s.runs_dir / "llm_calls.jsonl")
        emb = make_embedder(embedder_name)
        return cls(s, llm, VectorIndex(s.index_dir, emb), QuestionBank(s.bank_path), emb, **kw)

    def reload_sources(self) -> None:
        self.docs = load_manifest(self.s.manifest_path) if self.s.manifest_path.exists() else []
        self.retriever = Retriever(self.index, self.docs, self.s.top_k, self.s.min_relevance)

    # ------------------------------------------------------------------ generation
    def generate(self, req: GenerationRequest) -> RunResult:
        run = RunResult(run_id=uuid.uuid4().hex[:8], request=req, retrieval=None)
        mark = len(self.llm.calls)
        known = load_yaml("subjects.yaml")["subjects"]
        if req.subject not in known or req.chapter not in known[req.subject]["chapters"]:
            run.status, run.message = "failed", f"Unknown subject/chapter {req.subject}/{req.chapter}"
            return run

        ret = self.retriever.retrieve(req)
        run.retrieval = ret
        run.events.append(f"retrieved {len(ret.chunks)} chunks; warnings={len(ret.warnings)} conflicts={len(ret.conflicts)}")
        if ret.insufficient:
            run.status, run.message = "insufficient_evidence", ret.reason
            return run
        ctx_text, ctx = generation.format_context(ret.chunks)
        existing = self.bank.by_chapter(req.subject, req.chapter)
        # avoid-list includes rejected items so regeneration does not reproduce them
        avoid = [r.mcq.question for r in self.bank.by_chapter(req.subject, req.chapter, req.topic, exclude_rejected=False)]

        try:
            stems = generation.generate_stems(self.llm, req, ctx_text, ctx, avoid)
        except LLMError as e:
            run.status, run.message = "failed", f"question setter failed: {e}"
            return run
        if stems.insufficient_context:
            run.message = f"Setter reported insufficient context: {stems.insufficient_reason}"
            run.events.append(run.message)
        if not stems.questions:
            run.status = "insufficient_evidence"
            run.message = run.message or "Setter returned no questions."
            run.stats = self.llm.stats(mark)
            return run
        stem_list = stems.questions[: req.n]
        try:
            dbatch = generation.generate_distractors(self.llm, req, stem_list, ctx_text, ctx)
        except LLMError as e:
            run.status, run.message = "failed", f"distractor generator failed: {e}"
            run.stats = self.llm.stats(mark)
            return run

        accepted_in_batch: list[QuestionRecord] = []
        for i, stem in enumerate(stem_list):
            mcq, notes = generation.assemble(stem, dbatch, i)
            rec = self._new_record(req, run.run_id, mcq, ctx, ret)
            for n in notes:
                rec.history.append(HistoryEntry(actor="pipeline", action="note", detail={"note": n}))
            q_mark = len(self.llm.calls)
            self._review_loop(rec, req, ctx_text, ctx, existing + accepted_in_batch, ret)
            rec.generation = self._meta(run.run_id, q_mark, rec.generation.revisions)
            if rec.status != "REJECTED":
                accepted_in_batch.append(rec)
            run.questions.append(rec)
        # stem/distractor call cost is shared by the batch; attribute it to the run stats
        self.bank.save(*run.questions)
        run.stats = self.llm.stats(mark)
        return run

    def _new_record(self, req, run_id, mcq, ctx: dict[str, Chunk], ret: RetrievalResult) -> QuestionRecord:
        rec = QuestionRecord(
            id=f"q_{uuid.uuid4().hex[:10]}", subject=req.subject, chapter=req.chapter, topic=req.topic,
            difficulty=req.difficulty, question_type=req.question_type, mode=req.mode, attempt=req.attempt, mcq=mcq,
            context_map={k: c.chunk_id for k, c in ctx.items()},
            retrieval={"query": ret.query, "warnings": ret.warnings, "conflicts": ret.conflicts,
                       "chunks": [{"id": k, "chunk_id": c.chunk_id, "citation": c.citation()} for k, c in ctx.items()]},
        )
        rec.history.append(HistoryEntry(actor="pipeline", action="generated", detail={"run_id": run_id}))
        return rec

    def _meta(self, run_id: str, since: int, revisions: int) -> GenerationMeta:
        st = self.llm.stats(since)
        rv = self.llm.reviewer
        return GenerationMeta(
            run_id=run_id, provider=self.llm.provider.name, model=self.llm.provider.model,
            reviewer_model=rv.model if rv else self.llm.provider.model, mock=self.llm.provider.is_mock,
            prompt_versions=dict(self.llm.prompt_versions), llm_calls=st["llm_calls"], retries=st["retries"],
            revisions=revisions, latency_s=st["latency_s"], input_tokens=st["input_tokens"], output_tokens=st["output_tokens"],
        )

    def _doc_scope(self, req: GenerationRequest) -> tuple[set[str], set[str]]:
        """(allowed, historical) document ids - the same applicability rules retrieval used."""
        allowed, historical, _ = select_documents(self.docs, req)
        return set(allowed), set(historical)

    def _review_loop(self, rec: QuestionRecord, req, ctx_text, ctx, others: list[QuestionRecord], ret: RetrievalResult) -> None:
        revisions = 0
        best = None  # (rank, mcq, report, reviews, citations): a revision that makes things worse is discarded
        while True:
            decision, report, reviews = self._assess(rec, req, ctx_text, ctx, others, ret)
            rec.reliability, rec.reviews = report, reviews
            rank = (RANK[report.verdict], not report.deterministic_passed, len(decision.reasons))
            if best is None or rank < best[0]:
                best = (rank, rec.mcq, report, reviews, rec.citations)
            rec.history.append(HistoryEntry(actor="pipeline", action="assessed",
                                            detail={"verdict": report.verdict, "deterministic_passed": report.deterministic_passed,
                                                    "reasons": decision.reasons[:10], "revision": revisions}))
            if decision.verdict == "passed_for_human_review" or report.duplicate_of or revisions >= self.s.max_revisions:
                break
            if decision.verdict == "rejected" and not decision.revision_issues:
                break
            try:
                rec.mcq, summary = generation.revise(self.llm, req, rec.mcq, decision.revision_issues, ctx_text, ctx)
            except LLMError as e:
                rec.history.append(HistoryEntry(actor="pipeline", action="revision_failed", detail={"error": str(e)}))
                break
            revisions += 1
            rec.history.append(HistoryEntry(actor="pipeline", action="revised",
                                            detail={"issues": decision.revision_issues[:10], "summary": summary}))
        rec.generation.revisions = revisions
        if best and (rec.mcq is not best[1]):
            rec.history.append(HistoryEntry(actor="pipeline", action="kept_better_version",
                                            detail={"note": "latest revision assessed worse; restored earlier version"}))
            _, rec.mcq, rec.reliability, rec.reviews, rec.citations = best
        v = rec.reliability.verdict
        if v == "rejected":
            rec.status = "REJECTED"
        elif not rec.reliability.deterministic_passed:
            rec.status = "DRAFT"  # structurally unusable until a human fixes it
        elif not self.run_reviewers:
            rec.status = "VALIDATED"
        else:
            rec.status = "NEEDS_REVIEW"  # AI-reviewed; only a human can approve

    def _assess(self, rec, req, ctx_text, ctx, others, ret):
        mcq = rec.mcq
        checks, citations, grounding = validation.deterministic_checks(mcq, req, ctx, *self._doc_scope(req))
        rec.citations = citations
        det_ok = validation.passed(checks)
        dup, max_sim, similar = self.dups.check(mcq, [o for o in others if o.id != rec.id])
        reviews = Reviews()
        if det_ok and not dup and self.run_reviewers:  # gate: don't pay for LLM review of broken/duplicate items
            for name, fn in (("examiner", agents.critical_examiner), ("evidence", agents.evidence_verifier),
                             ("difficulty", agents.difficulty_reviewer)):
                try:
                    setattr(reviews, name, fn(self.llm, req, mcq, ctx_text, ctx))
                except LLMError as e:
                    checks.append(validation.CheckResult(name=f"{name}_review_completed", passed=False, kind="llm",
                                                         severity="warning", detail=str(e)))
            if self.run_judge:
                try:
                    reviews.judge = agents.judge(self.llm, req, mcq, ctx_text, ctx)
                except LLMError as e:
                    checks.append(validation.CheckResult(name="judge_completed", passed=False, kind="llm", severity="warning", detail=str(e)))
        decision = agents.coordinate(mcq, req, checks, grounding, reviews.examiner, reviews.evidence, reviews.difficulty, dup)
        llm_reviews_missing = self.run_reviewers and det_ok and not dup and not (reviews.examiner and reviews.evidence and reviews.difficulty)
        if decision.verdict == "passed_for_human_review" and llm_reviews_missing:
            decision.verdict = "needs_revision"
            decision.reasons.append("one or more AI reviews failed to run - cannot pass without them")
        verdict = decision.verdict
        if decision.verdict == "needs_revision" and not agents.is_blocking(decision) and det_ok:
            verdict = "passed_for_human_review"  # only soft flags (difficulty/warnings) remain
        report = ReliabilityReport(
            checks=checks, deterministic_passed=det_ok, quote_grounding=grounding,
            evidence_status=reviews.evidence.verdict if reviews.evidence else "not_checked",
            answer_agreement=decision.answer_agreement, requested_difficulty=req.difficulty,
            estimated_difficulty=reviews.difficulty.estimated_difficulty if reviews.difficulty else None,
            difficulty_match=decision.difficulty_match, duplicate_of=dup, max_similarity=max_sim, similar_to=similar,
            version_flags=list(ret.warnings) + [f"possible source conflict: {c['doc_a']} vs {c['doc_b']}" for c in ret.conflicts],
            disagreements=decision.disagreements, verdict=verdict, reasons=decision.reasons,
        )
        report.composite_score = validation.composite_score(report)
        return decision, report, reviews

    # ------------------------------------------------------------------ human-in-the-loop helpers
    def context_for(self, rec: QuestionRecord) -> dict[str, Chunk]:
        by_id = {c.chunk_id: c for c in self.index.chunks()} if rec.context_map else {}
        return {k: by_id[v] for k, v in rec.context_map.items() if v in by_id}

    def edit(self, qid: str, reviewer: str, **changes) -> QuestionRecord:
        """Human edit -> deterministic re-validation on the original context (no automatic re-approval)."""
        rec = apply_edit(self.bank.get(qid), reviewer, **changes)
        req = GenerationRequest(subject=rec.subject, chapter=rec.chapter, topic=rec.topic, n=1, difficulty=rec.difficulty,
                                question_type=rec.question_type, mode=rec.mode, attempt=rec.attempt)
        checks, citations, grounding = validation.deterministic_checks(rec.mcq, req, self.context_for(rec), *self._doc_scope(req))
        rec.citations = citations
        r = rec.reliability
        r.checks = checks + [c for c in r.checks if c.kind == "llm"]
        r.deterministic_passed, r.quote_grounding = validation.passed(checks), grounding
        r.reasons = [f"human edit by {reviewer}: AI reviews refer to the pre-edit version"] + [x for x in r.reasons if not x.startswith("human edit")]
        r.composite_score = validation.composite_score(r)
        if not r.deterministic_passed and rec.status != "REJECTED":
            rec.status = "DRAFT"
        elif rec.status == "DRAFT":
            rec.status = "NEEDS_REVIEW"
        self.bank.save(rec)
        return rec

    def regenerate(self, qid: str, reviewer: str = "system") -> RunResult:
        """Generate a replacement for one question with the same request; the old one is marked rejected."""
        old = self.bank.get(qid)
        req = GenerationRequest(subject=old.subject, chapter=old.chapter, topic=old.topic, n=1, difficulty=old.difficulty,
                                question_type=old.question_type, mode=old.mode, attempt=old.attempt)
        from .review import reject

        reject(self.bank, qid, reviewer, reason="regenerate", comment="replaced by regeneration")
        return self.generate(req)
