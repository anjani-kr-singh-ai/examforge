from examforge.agents import coordinate, is_blocking, question_block
from examforge.schemas import (
    CheckResult, ClaimCheck, DifficultyReview, EvidenceReview, ExaminerIssue, ExaminerReview, GenerationRequest, OptionVerdict,
)
from tests.test_validation import mcq

REQ = GenerationRequest(chapter="Input Tax Credit", topic="t", difficulty="Medium")
M = mcq()  # key = B
OK_CHECKS = [CheckResult(name="x", passed=True)]


def examiner(verdict="accept", issues=(), verdicts=None):
    verdicts = verdicts or {"A": "incorrect", "B": "correct", "C": "incorrect", "D": "incorrect"}
    return ExaminerReview(option_verdicts=[OptionVerdict(label=k, verdict=v, plausibility=3, comment="") for k, v in verdicts.items()],
                          issues=list(issues), clarity=4, distractor_quality=4, verdict=verdict, summary="")


def evidence(verdict="supported", answer=True):
    return EvidenceReview(claims=[ClaimCheck(claim="c", support="supported" if verdict == "supported" else "unsupported", chunk_ids=["C1"], comment="")],
                          answer_supported=answer, explanation_supported=answer, distractors_incorrect_per_source=True,
                          outdated_or_conflicting=False, verdict=verdict)


def diff(labels=("B",), est="Medium"):
    return DifficultyReview(answer_labels=list(labels), confidence=4, reasoning_steps=2, provisions_involved=["s16"],
                            estimated_difficulty=est, rationale="r", suggestion="s")


def test_clean_question_passes_to_human_review_never_approved():
    d = coordinate(M, REQ, OK_CHECKS, 1.0, examiner(), evidence(), diff(), None)
    assert d.verdict == "passed_for_human_review" and d.answer_agreement and d.difficulty_match


def test_ambiguous_question_needs_revision():
    e = examiner("revise", [ExaminerIssue(category="ambiguity", severity="major", detail="'credit' could mean two things")])
    d = coordinate(M, REQ, OK_CHECKS, 1.0, e, evidence(), diff(), None)
    assert d.verdict == "needs_revision" and is_blocking(d) and any("ambiguity" in i for i in d.revision_issues)


def test_unsupported_answer_rejected_when_independent_signals_agree():
    d = coordinate(M, REQ, OK_CHECKS, 1.0, examiner(), evidence("unsupported", answer=False), diff(("C",)), None)
    assert d.verdict == "rejected"
    assert any("blind solver" in r for r in d.reasons)


def test_difficulty_mismatch_is_soft():
    d = coordinate(M, REQ, OK_CHECKS, 1.0, examiner(), evidence(), diff(est="Easy"), None)
    assert d.verdict == "needs_revision" and not d.difficulty_match and not is_blocking(d)


def test_conflicting_reviewers_are_surfaced():
    d = coordinate(M, REQ, OK_CHECKS, 1.0, examiner("accept"), evidence("unsupported", answer=True), diff(), None)
    assert "examiner accepted but evidence verifier found the question unsupported" in d.disagreements
    d2 = coordinate(M, REQ, OK_CHECKS, 1.0, examiner("accept"), evidence(), diff(("A",)), None)
    assert any("blind solver disagreed" in x for x in d2.disagreements) and d2.verdict == "needs_revision"


def test_examiner_disputing_key_and_duplicates():
    e = examiner(verdicts={"A": "arguable", "B": "correct", "C": "incorrect", "D": "incorrect"})
    d = coordinate(M, REQ, OK_CHECKS, 1.0, e, evidence(), diff(), None)
    assert any("disputes key" in r for r in d.reasons) and d.verdict == "needs_revision"
    assert coordinate(M, REQ, OK_CHECKS, 1.0, None, None, None, "q_old").verdict == "rejected"


def test_failed_deterministic_check_blocks():
    d = coordinate(M, REQ, [CheckResult(name="exactly_four_options", passed=False, detail="3")], 1.0, None, None, None, None)
    assert d.verdict == "needs_revision" and is_blocking(d)


def test_blind_block_hides_key():
    blind = question_block(M, with_key=False)
    assert "Answer key" not in blind and "Explanation" not in blind and "Why (" not in blind
