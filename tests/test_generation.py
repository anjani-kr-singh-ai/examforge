import pytest
from pydantic import ValidationError

from examforge.generation import assemble, draft_to_mcq
from examforge.schemas import (
    DistractorBatch, DistractorDraft, DistractorSet, DraftOption, EvidenceQuote, GenerationRequest, MCQDraft, StemBatch, StemDraft,
)
from examforge.validation import deterministic_checks, passed
from tests.test_validation import CTX

STEM = StemDraft(question="By when must ITC for FY 2025-26 invoices be availed?", correct_options=["30th November 2026"],
                 correct_option_explanations=["Section 16(4) limit"], explanation="The limit is the thirtieth day of November after the FY end.",
                 evidence=[EvidenceQuote(chunk_id="C1", quote="thirtieth day of November following the end of financial year")],
                 assumptions=[], concepts_tested=["time limit"])


def batch(n):
    return DistractorBatch(items=[DistractorSet(question_index=0, distractors=[
        DistractorDraft(text=t, misconception_category="threshold_or_time_limit", why_incorrect="wrong date") for t in
        ["31st March 2027", "30th September 2026", "31st December 2026", "30th June 2027"][:n]])])


def test_assembly_shuffles_deterministically_and_maps_key():
    m1, notes = assemble(STEM, batch(3), 0)
    m2, _ = assemble(STEM, batch(3), 0)
    assert [o.text for o in m1.options] == [o.text for o in m2.options] and not notes
    assert [o.label for o in m1.options] == ["A", "B", "C", "D"]
    key = m1.correct_labels[0]
    assert next(o for o in m1.options if o.label == key).text == "30th November 2026"
    assert all(o.misconception_category for o in m1.options if not o.is_correct)


def test_wrong_distractor_count_is_reported_not_hidden():
    m, notes = assemble(STEM, batch(2), 0)
    assert notes and len(m.options) == 3
    checks, *_ = deterministic_checks(m, GenerationRequest(chapter="Input Tax Credit", topic="t"), CTX)
    assert not passed(checks)


def test_wrong_question_type_detected():
    two = STEM.model_copy(update={"correct_options": ["30th November 2026", "date of annual return if earlier"],
                                  "correct_option_explanations": ["a", "b"]})
    m, _ = assemble(two, batch(2), 0)
    checks, *_ = deterministic_checks(m, GenerationRequest(chapter="Input Tax Credit", topic="t", question_type="single"), CTX)
    assert not {c.name: c for c in checks}["question_type_matches_answer_count"].passed


def test_revision_draft_with_invalid_answer_reference_fails_validation():
    d = MCQDraft(question="Q? " * 6, correct_labels=["E"], explanation="x" * 50,
                 evidence=[EvidenceQuote(chunk_id="C1", quote="thirtieth day of November")],
                 options=[DraftOption(label=l, text=f"opt {l}", explanation="because of reasons") for l in "ABCD"])
    checks, *_ = deterministic_checks(draft_to_mcq(d), GenerationRequest(chapter="Input Tax Credit", topic="t"), CTX)
    assert not {c.name: c for c in checks}["answer_refs_valid"].passed


def test_missing_fields_in_llm_output_raise():
    with pytest.raises(ValidationError):
        StemBatch.model_validate({"questions": [{"question": "only a stem"}]})
    with pytest.raises(ValidationError):
        StemBatch.model_validate_json('{"insufficient_context": false, "questions": [')  # truncated JSON
