"""Pydantic contracts for every pipeline stage.

Two families of models live here:
  * LLM output models (StemBatch, DistractorBatch, MCQDraft, *Review, JudgeScores) - what a model must return.
    They are intentionally *not* stricter than the JSON schema a provider can enforce; business rules
    (exactly 4 options, answer count vs type, ...) are checked by validation.py so violations are
    reported as findings instead of being hidden inside a parse error.
  * Domain records (SourceDocument, Chunk, QuestionRecord, ...) - what the system stores.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Difficulty = Literal["Easy", "Medium", "Hard"]
QuestionType = Literal["single", "multiple"]
Mode = Literal["conceptual", "application", "case_based", "multi_concept", "amendment"]
Status = Literal["DRAFT", "VALIDATED", "NEEDS_REVIEW", "APPROVED", "REJECTED"]
Verdict = Literal["passed_for_human_review", "needs_revision", "rejected", "insufficient_evidence"]
LABELS = ("A", "B", "C", "D")
MISCONCEPTIONS = (
    "related_provision_confusion",
    "condition_misread",
    "exception_misapplied",
    "eligibility_vs_procedure",
    "threshold_or_time_limit",
    "rule_misapplied",
    "other",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- sources & chunks
class SourceDocument(BaseModel):
    document_id: str
    title: str
    filename: str
    source_url: str
    landing_page_url: str | None = None
    subject: str = "IDT"
    chapter: str | None = None  # None => multi-chapter paper (RTP/MTP)
    document_type: str  # Study Material | RTP | MTP | Statutory Update | Question Paper | Suggested Answers
    attempt: str | None = None  # exam attempt the document is prepared for, as printed (NOT a legal date)
    applicable_attempts: list[str] = []  # attempts this document may be used for (filtering key); [] => [attempt]
    publication_date: str | None = None
    law_cutoff: str | None = None  # date the law is stated "as amended up to", as printed (e.g. 30.04.2026)
    law_cutoff_statement: str | None = None  # the full cutoff statement, verbatim
    include_pages: list[list[int]] | None = None  # 1-based inclusive PDF page ranges to index; None = all
    valid_from: str | None = None  # legal applicability, only when the document states it
    valid_until: str | None = None
    notes: str | None = None

    def attempts(self) -> list[str]:
        return self.applicable_attempts or ([self.attempt] if self.attempt else [])
    # filled by ingestion
    sha256: str | None = None
    page_count: int | None = None
    extraction_status: str | None = None  # ok | needs_ocr | missing_file | error:<msg>
    chunk_count: int | None = None


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page_start: int
    page_end: int
    section: str = ""
    subject: str = "IDT"
    chapter: str = ""
    document_type: str = ""
    attempt: str = ""
    law_cutoff: str = ""
    title: str = ""
    source_url: str = ""
    amendment_marker: bool = False

    def citation(self) -> str:
        pages = f"p.{self.page_start}" if self.page_start == self.page_end else f"pp.{self.page_start}-{self.page_end}"
        sec = f", {self.section}" if self.section else ""
        return f"{self.title} ({self.document_type}, {self.attempt or 'attempt n/a'}), {pages}{sec}"


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float  # fused rank score (RRF) - ordering only, not a probability
    dense_similarity: float | None = None  # cosine similarity, used for the relevance floor
    lexical_score: float | None = None


class RetrievalResult(BaseModel):
    query: str
    filters: dict
    chunks: list[RetrievedChunk]
    warnings: list[str] = []
    conflicts: list[dict] = []
    insufficient: bool = False
    reason: str = ""


# --------------------------------------------------------------------------- request
class GenerationRequest(BaseModel):
    subject: str = "IDT"
    chapter: str
    topic: str
    n: int = Field(default=3, ge=1, le=10)
    difficulty: Difficulty = "Medium"
    question_type: QuestionType = "single"
    mode: Mode = "conceptual"
    attempt: str | None = None  # assessment period; None = any attempt (flagged)
    allow_historical: bool = False  # intentionally use material prepared for other attempts
    document_types: list[str] | None = None


# --------------------------------------------------------------------------- LLM outputs: generation
class EvidenceQuote(BaseModel):
    chunk_id: str = Field(description="id of the context chunk, e.g. 'C3'")
    quote: str = Field(description="verbatim sentence copied from that chunk that supports the claim")


class StemDraft(BaseModel):
    question: str
    correct_options: list[str] = Field(description="text of the correct option(s); 1 for single, 2-3 for multiple")
    correct_option_explanations: list[str] = Field(description="why each correct option is correct, same order")
    explanation: str = Field(description="detailed explanation of the answer, citing provisions")
    evidence: list[EvidenceQuote]
    assumptions: list[str] = Field(description="facts assumed by the scenario; empty if none")
    concepts_tested: list[str]


class StemBatch(BaseModel):
    insufficient_context: bool = Field(description="true if the context cannot support the requested questions")
    insufficient_reason: str = ""
    questions: list[StemDraft]


class DistractorDraft(BaseModel):
    text: str
    misconception_category: str = Field(description="one of: " + ", ".join(MISCONCEPTIONS))
    why_incorrect: str
    evidence_chunk_id: str | None = None


class DistractorSet(BaseModel):
    question_index: int
    distractors: list[DistractorDraft]


class DistractorBatch(BaseModel):
    items: list[DistractorSet]


class DraftOption(BaseModel):
    label: str
    text: str
    explanation: str = Field(description="why this option is correct or incorrect")
    misconception_category: str | None = None


class MCQDraft(BaseModel):
    """Full question as returned by the revision stage (explicit labels, so references can be wrong)."""

    question: str
    options: list[DraftOption]
    correct_labels: list[str]
    explanation: str
    evidence: list[EvidenceQuote]
    assumptions: list[str] = []
    concepts_tested: list[str] = []
    change_summary: str = ""


# --------------------------------------------------------------------------- assembled question
class Option(BaseModel):
    label: str
    text: str
    is_correct: bool
    explanation: str = ""
    misconception_category: str | None = None


class MCQ(BaseModel):
    question: str
    options: list[Option]
    correct_labels: list[str]
    explanation: str
    evidence: list[EvidenceQuote] = []
    assumptions: list[str] = []
    concepts_tested: list[str] = []


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    document_type: str
    attempt: str
    page_start: int
    page_end: int
    section: str
    source_url: str
    quote: str
    quote_found: bool  # deterministic: quote located in the cited chunk text
    quote_match: float  # 0..1 best fuzzy match ratio


# --------------------------------------------------------------------------- LLM outputs: reviewers
class ExaminerIssue(BaseModel):
    category: Literal[
        "ambiguity", "multiple_correct", "wrong_key", "weak_distractor", "too_easy",
        "unsupported_scenario", "incomplete_conditions", "explanation_inconsistent", "outdated", "other",
    ]
    severity: Literal["minor", "major", "critical"]
    detail: str


class OptionVerdict(BaseModel):
    label: str
    verdict: Literal["correct", "incorrect", "arguable"]
    plausibility: int = Field(description="1-5, how tempting this option is to a prepared but imperfect candidate")
    comment: str


class ExaminerReview(BaseModel):
    option_verdicts: list[OptionVerdict]
    issues: list[ExaminerIssue]
    clarity: int = Field(description="1-5")
    distractor_quality: int = Field(description="1-5")
    verdict: Literal["accept", "revise", "reject"]
    summary: str


class ClaimCheck(BaseModel):
    claim: str
    support: Literal["supported", "partially", "unsupported"]
    chunk_ids: list[str]
    comment: str


class EvidenceReview(BaseModel):
    claims: list[ClaimCheck]
    answer_supported: bool
    explanation_supported: bool
    distractors_incorrect_per_source: bool
    outdated_or_conflicting: bool
    outdated_note: str = ""
    verdict: Literal["supported", "partially_supported", "unsupported"]


class DifficultyReview(BaseModel):
    """Produced by a reviewer who does NOT see the answer key: it solves the question blind."""

    answer_labels: list[str]
    confidence: int = Field(description="1-5 confidence in own answer")
    reasoning_steps: int
    provisions_involved: list[str]
    estimated_difficulty: Difficulty
    rationale: str
    suggestion: str = ""


class JudgeScores(BaseModel):
    relevance: int = Field(description="1-5")
    factual_correctness: int = Field(description="1-5")
    clarity: int = Field(description="1-5")
    difficulty_fit: int = Field(description="1-5")
    distractor_quality: int = Field(description="1-5")
    answer_explanation_consistency: int = Field(description="1-5")
    rationale: str


# --------------------------------------------------------------------------- reliability & records
class CheckResult(BaseModel):
    name: str
    passed: bool
    severity: Literal["info", "warning", "error"] = "error"
    kind: Literal["deterministic", "llm"] = "deterministic"
    detail: str = ""


class ReliabilityReport(BaseModel):
    checks: list[CheckResult] = []
    deterministic_passed: bool = False
    quote_grounding: float = 0.0  # share of evidence quotes found verbatim/fuzzy in cited chunks
    evidence_status: Literal["supported", "partially_supported", "unsupported", "not_checked"] = "not_checked"
    answer_agreement: bool | None = None  # blind solver's answer == key
    requested_difficulty: Difficulty | None = None
    estimated_difficulty: Difficulty | None = None
    difficulty_match: bool | None = None
    duplicate_of: str | None = None
    max_similarity: float = 0.0
    similar_to: list[dict] = []
    version_flags: list[str] = []
    disagreements: list[str] = []
    verdict: Verdict = "needs_revision"
    reasons: list[str] = []
    composite_score: float | None = None  # see reliability.composite_score docstring; NOT a probability


class Reviews(BaseModel):
    examiner: ExaminerReview | None = None
    evidence: EvidenceReview | None = None
    difficulty: DifficultyReview | None = None
    judge: JudgeScores | None = None


class GenerationMeta(BaseModel):
    run_id: str = ""
    provider: str = ""
    model: str = ""
    reviewer_model: str = ""
    mock: bool = False
    prompt_versions: dict[str, str] = {}
    llm_calls: int = 0
    retries: int = 0
    revisions: int = 0
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class HistoryEntry(BaseModel):
    ts: str = Field(default_factory=now_iso)
    actor: str  # "pipeline" or "human:<name>"
    action: str
    detail: dict = {}


class Feedback(BaseModel):
    ts: str = Field(default_factory=now_iso)
    reviewer: str
    decision: Literal["approve", "reject", "edit", "regenerate", "comment"]
    comment: str = ""
    rejection_reason: str | None = None  # factual_error | ambiguous | weak_distractors | wrong_difficulty | off_topic | duplicate | other
    corrected_difficulty: Difficulty | None = None
    factual_issue: str | None = None


class QuestionRecord(BaseModel):
    id: str
    created_at: str = Field(default_factory=now_iso)
    subject: str
    chapter: str
    topic: str
    difficulty: Difficulty  # requested
    question_type: QuestionType
    mode: Mode
    attempt: str | None
    mcq: MCQ
    original_mcq: MCQ | None = None  # AI version, kept when a human edits
    citations: list[Citation] = []
    context_map: dict[str, str] = {}  # short id (C1..) -> chunk_id, so edits can be re-validated on the same context
    retrieval: dict = {}  # query, filters, warnings, conflicts
    reviews: Reviews = Reviews()
    reliability: ReliabilityReport = ReliabilityReport()
    status: Status = "DRAFT"
    human_edited: bool = False
    generation: GenerationMeta = GenerationMeta()
    history: list[HistoryEntry] = []
    feedback: list[Feedback] = []

    @property
    def provenance(self) -> str:
        """AI-generated vs AI-reviewed vs human-approved, for UI badges."""
        if self.status == "APPROVED":
            return "human-approved"
        if self.reviews.examiner or self.reviews.evidence:
            return "ai-reviewed"
        return "ai-generated"
