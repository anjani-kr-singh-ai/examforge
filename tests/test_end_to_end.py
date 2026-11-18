import pytest

from examforge import analytics, review
from examforge.export import export
from examforge.llm import TransientLLMError
from examforge.schemas import ExaminerIssue, ExaminerReview, GenerationRequest, OptionVerdict, StemBatch
from tests.conftest import ScriptedProvider, make_engine

REQ = GenerationRequest(chapter="Input Tax Credit", topic="Eligibility and conditions for taking ITC", n=2, attempt="May 2027")


def test_full_workflow_generate_review_export(engine, tmp_path):
    run = engine.generate(REQ)
    assert run.status == "ok" and run.retrieval.chunks and len(run.questions) == 2
    for q in run.questions:
        assert q.status in ("NEEDS_REVIEW", "REJECTED", "DRAFT") and q.status != "APPROVED"  # never auto-approved
        assert q.generation.mock and q.citations and q.context_map
        assert q.history[0].action == "generated"
    good = next(q for q in run.questions if q.status == "NEEDS_REVIEW")
    assert good.reliability.deterministic_passed and good.reviews.examiner and good.reviews.evidence and good.reviews.difficulty
    assert good.provenance == "ai-reviewed"

    approved = review.approve(engine.bank, good.id, "alice", "looks right")
    assert approved.status == "APPROVED" and approved.provenance == "human-approved"
    assert approved.history[-1].actor == "human:alice"

    paths = export(engine.bank.all(), tmp_path / "out")
    csv = paths["csv"].read_text(encoding="utf-8-sig")
    assert "sources" in csv and "review_status" in csv and "APPROVED" in csv and "evidence_status" in csv
    assert "MOCK provider output" in paths["paper"].read_text(encoding="utf-8")
    assert "Sources:" in paths["answer_key"].read_text(encoding="utf-8")

    cov = analytics.coverage(engine.bank.all(), engine.docs)
    assert "not ICAI examination weightage" in cov["note"]
    assert any(m["topic"] == "Blocked credits" for m in cov["missing_topics"])
    fb = analytics.feedback_analysis(engine.bank.all())
    assert fb["human_approval_rate"] == 1.0


def test_human_edit_keeps_original_revalidates_and_voids_approval(engine):
    q = next(q for q in engine.generate(REQ).questions if q.status == "NEEDS_REVIEW")
    review.approve(engine.bank, q.id, "alice")
    original_key = list(q.mcq.correct_labels)
    wrong = [l for l in "ABCD" if l not in original_key][:2]
    edited = engine.edit(q.id, "bob", correct_labels=wrong, comment="testing")  # 2 keys on a single-correct item
    assert edited.human_edited and edited.original_mcq.correct_labels == original_key and edited.mcq.correct_labels == wrong
    assert edited.status == "DRAFT" and not edited.reliability.deterministic_passed
    with pytest.raises(ValueError):
        review.approve(engine.bank, q.id, "alice")  # cannot approve while checks fail
    edit = next(h for h in edited.history if h.action == "edit")
    assert edit.actor == "human:bob" and "correct_labels" in edit.detail["changes"]


def test_reject_and_regenerate(engine):
    q = engine.generate(REQ.model_copy(update={"n": 1})).questions[0]
    review.reject(engine.bank, q.id, "carol", "ambiguous", "two readings")
    assert engine.bank.get(q.id).status == "REJECTED"
    assert analytics.feedback_analysis(engine.bank.all())["rejection_reasons"] == {"ambiguous": 1}
    q2 = engine.generate(REQ.model_copy(update={"n": 1})).questions[0]
    run = engine.regenerate(q2.id, "carol")
    assert engine.bank.get(q2.id).status == "REJECTED" and run.status in ("ok", "insufficient_evidence")


def _examiner_revise(v):
    m = v["_mcq"]
    return ExaminerReview(
        option_verdicts=[OptionVerdict(label=o.label, verdict="correct" if o.is_correct else "incorrect", plausibility=3, comment="") for o in m.options],
        issues=[ExaminerIssue(category="ambiguity", severity="major", detail="stem does not state the financial year")],
        clarity=2, distractor_quality=3, verdict="revise", summary="ambiguous")


def test_revision_loop_fixes_flagged_question(data_dir):
    eng = make_engine(data_dir, ScriptedProvider({"ExaminerReview": [_examiner_revise]}), max_revisions=1)
    q = eng.generate(REQ.model_copy(update={"n": 1})).questions[0]
    actions = [h.action for h in q.history]
    assert "revised" in actions and q.generation.revisions == 1
    revised = next(h for h in q.history if h.action == "revised")
    assert any("financial year" in i for i in revised.detail["issues"])  # examiner finding fed to the reviser
    assert q.reliability.verdict == "passed_for_human_review"


def test_reviewer_failure_never_passes_question(data_dir):
    fail = [TransientLLMError("503")] * 3
    eng = make_engine(data_dir, ScriptedProvider({"EvidenceReview": fail * 2}), max_revisions=0)
    q = eng.generate(REQ.model_copy(update={"n": 1})).questions[0]
    assert q.reliability.verdict != "passed_for_human_review"
    assert any(c.name == "evidence_review_completed" and not c.passed for c in q.reliability.checks)


def test_setter_reports_insufficient_context(data_dir):
    eng = make_engine(data_dir, ScriptedProvider({"StemBatch": [StemBatch(insufficient_context=True, insufficient_reason="no data", questions=[])]}))
    run = eng.generate(REQ)
    assert run.status == "insufficient_evidence" and "no data" in run.message and not run.questions


def test_no_generation_when_retrieval_insufficient(engine):
    run = engine.generate(REQ.model_copy(update={"attempt": "Nov 2030"}))
    assert run.status == "insufficient_evidence" and run.stats == {} and not engine.llm.calls


def test_in_batch_and_bank_duplicates_rejected(engine):
    engine.generate(REQ.model_copy(update={"n": 1}))
    again = engine.generate(REQ.model_copy(update={"n": 1}))  # mock is deterministic -> same question
    assert again.questions[0].status == "REJECTED" and again.questions[0].reliability.duplicate_of
