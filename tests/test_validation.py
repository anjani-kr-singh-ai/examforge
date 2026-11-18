from examforge.retrieval import HashEmbedder
from examforge.schemas import MCQ, Chunk, EvidenceQuote, GenerationRequest, Option, QuestionRecord
from examforge.validation import DuplicateDetector, deterministic_checks, passed, quote_match

CTX = {"C1": Chunk(chunk_id="d::c1", document_id="d", text="A registered person shall not be entitled to take input tax credit after the thirtieth day of November following the end of financial year.",
                   page_start=12, page_end=12, title="SM", attempt="May 2027", section="3 ELIGIBILITY")}
REQ = GenerationRequest(chapter="Input Tax Credit", topic="Time limit for availing ITC", attempt="May 2027")


def mcq(correct=("B",), n=4, explanation="The time limit is the thirtieth day of November following the end of the financial year, as stated in the material.",
        quote="shall not be entitled to take input tax credit after the thirtieth day of November", chunk="C1", texts=None):
    texts = texts or ["31st March of next year", "30th November following end of FY", "30th September of next year", "31st December of next year"]
    opts = [Option(label="ABCD"[i] if i < 4 else "E", text=texts[i] if i < len(texts) else f"x{i}", is_correct="ABCD"[min(i, 3)] in correct and i < 4,
                   explanation="Because of the stated limit in the material.") for i in range(n)]
    return MCQ(question="By when must ITC for an invoice of FY 2025-26 be availed?", options=opts, correct_labels=list(correct),
               explanation=explanation, evidence=[EvidenceQuote(chunk_id=chunk, quote=quote)])


def check(m, req=REQ, allowed=None):
    checks, cits, g = deterministic_checks(m, req, CTX, allowed)
    return {c.name: c for c in checks}, cits, g, passed(checks)


def test_valid_single_correct_passes_with_metadata_citation():
    by, cits, g, ok = check(mcq())
    assert ok and g == 1.0
    assert cits[0].page_start == 12 and cits[0].title == "SM" and cits[0].quote_found


def test_multiple_correct_answer_count():
    req = REQ.model_copy(update={"question_type": "multiple"})
    assert check(mcq(correct=("A", "B")), req)[3]
    by, *_ , ok = check(mcq(correct=("B",)), req)
    assert not ok and not by["question_type_matches_answer_count"].passed
    by, *_, ok = check(mcq(correct=("A", "B")))  # single-correct request with 2 keys
    assert not ok and not by["question_type_matches_answer_count"].passed


def test_option_count_and_invalid_answer_reference():
    by, *_, ok = check(mcq(n=3))
    assert not ok and not by["exactly_four_options"].passed
    bad = mcq()
    bad.correct_labels = ["E"]
    by, *_, ok = check(bad)
    assert not ok and not by["answer_refs_valid"].passed and not by["answer_flags_consistent"].passed


def test_missing_and_inconsistent_explanations():
    by, *_, ok = check(mcq(explanation="short"))
    assert not ok and not by["explanation_present"].passed
    by, *_, ok = check(mcq(explanation="Option (C) is correct because the material says so in the chapter text."))
    assert not ok and not by["explanation_consistent_with_key"].passed
    m = mcq()
    m.options[2].explanation = ""
    assert not check(m)[0]["option_explanations_complete"].passed


def test_invalid_source_reference_and_ungrounded_quote():
    by, cits, g, ok = check(mcq(chunk="C9"))
    assert not ok and not by["evidence_refs_valid"].passed and g == 0.0
    by, cits, g, ok = check(mcq(quote="ITC can be claimed at any time within six years"))
    assert not ok and not by["evidence_quotes_grounded"].passed and not cits[0].quote_found


def test_label_dependent_options_error_and_cues_warn():
    texts = ["All of the above", "30th November following end of FY", "None of the above", "31st December"]
    by, *_, ok = check(mcq(texts=texts))
    assert not ok and not by["no_label_dependent_options"].passed  # options are shuffled, so these break
    long = ["31 March", "30th November following the end of the financial year to which the invoice pertains, or annual return date if earlier", "30 Sep", "31 Dec"]
    by, *_ = check(mcq(texts=long))
    assert not by["no_length_cue"].passed and by["no_length_cue"].severity == "warning"


def test_identical_options_rejected_but_one_word_differences_allowed():
    by, *_, ok = check(mcq(texts=["30 days", "30 Days.", "60 days", "90 days"]))
    assert not by["options_distinct"].passed
    by, *_ = check(mcq(texts=["credit shall be allowed", "credit shall not be allowed", "credit is optional", "credit is deferred"]))
    assert by["options_distinct"].passed


def test_source_version_incompatibility():
    by, *_, ok = check(mcq(), allowed={"other_doc"})
    assert not ok and not by["source_version_compatible"].passed
    assert check(mcq(), allowed={"d"})[3]  # doc-level applicability, not the attempt display label
    checks, *_ = deterministic_checks(mcq(), REQ, CTX, {"d"}, {"d"})
    assert passed(checks) and any(c.name == "cites_historical_material" and c.severity == "warning" for c in checks)


def test_quote_match_handles_ellipsis_and_paraphrase():
    src = CTX["C1"].text
    assert quote_match("shall not be entitled ... thirtieth day of November", src) == 1.0
    assert quote_match("may claim credit until the end of the following year", src) < 0.8


def _rec(i, m):
    return QuestionRecord(id=i, subject="IDT", chapter="Input Tax Credit", topic="t", difficulty="Easy", question_type="single",
                          mode="conceptual", attempt="May 2027", mcq=m)


def test_duplicate_detection_levels():
    det = DuplicateDetector(HashEmbedder(), 0.92, 0.6)
    base = mcq()
    dup, sim, _ = det.check(base, [_rec("q1", base)])
    assert dup == "q1" and sim == 1.0
    reworded = base.model_copy(update={"question": base.question.upper() + "  "})  # normalised match
    assert det.check(reworded, [_rec("q1", base)])[0] == "q1"
    other = mcq(texts=["Motor vehicles are blocked", "Food is blocked", "Works contract is blocked", "Rent-a-cab is blocked"])
    other.question = "Which credit is blocked under section 17(5)?"
    dup, sim, similar = det.check(other, [_rec("q1", base)])
    assert dup is None and sim < 0.92
