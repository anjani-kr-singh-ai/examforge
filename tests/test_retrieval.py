import numpy as np

from examforge.retrieval import detect_conflicts, select_documents
from examforge.schemas import Chunk, GenerationRequest, RetrievedChunk

REQ = dict(chapter="Input Tax Credit", topic="Blocked credits", attempt="May 2027")


def test_relevant_query_returns_on_topic_section(engine):
    res = engine.retriever.retrieve(GenerationRequest(**REQ), query="motor vehicles seating capacity thirteen persons blocked credit")
    assert not res.insufficient
    assert "thirteen persons" in res.chunks[0].chunk.text


def test_period_filter_excludes_other_attempts(engine):
    res = engine.retriever.retrieve(GenerationRequest(**REQ))
    assert res.chunks and all(rc.chunk.document_id == "sm_itc" for rc in res.chunks)
    assert res.filters["documents"] == ["sm_itc"]


def test_historical_material_only_when_intended_and_flagged(engine):
    res = engine.retriever.retrieve(GenerationRequest(**REQ, allow_historical=True), k=20)
    assert {rc.chunk.document_id for rc in res.chunks} == {"sm_itc", "rtp_old"}
    assert any("Historical material" in w for w in res.warnings)
    assert any("different law cut-offs" in w for w in res.warnings)  # 30.04.2026 vs 31.10.2025 mixed


def test_no_period_selected_warns_about_mixing(engine):
    allowed, _, warnings = select_documents(engine.docs, GenerationRequest(chapter="Input Tax Credit", topic="x"))
    assert set(allowed) == {"sm_itc", "rtp_old"} and warnings


def test_empty_results_are_explicit(engine):
    r1 = engine.retriever.retrieve(GenerationRequest(chapter="Input Tax Credit", topic="x", attempt="Nov 2030"))
    assert r1.insufficient and "Available periods" in r1.reason and not r1.chunks
    r2 = engine.retriever.retrieve(GenerationRequest(chapter="Value of Supply", topic="Discounts", attempt="May 2027"))
    assert r2.insufficient
    engine.retriever.min_relevance = 0.999
    r3 = engine.retriever.retrieve(GenerationRequest(**REQ))
    assert r3.insufficient and "relevance floor" in r3.reason


def test_amendment_mode_without_documented_amendments_refuses(engine):
    res = engine.retriever.retrieve(GenerationRequest(**REQ, mode="amendment"))
    assert res.insufficient and "amendments" in res.reason


def test_source_references_on_every_chunk(engine):
    res = engine.retriever.retrieve(GenerationRequest(**REQ))
    for rc in res.chunks:
        c = rc.chunk
        assert c.title and c.source_url and c.page_start >= 1 and c.chunk_id.startswith(c.document_id)
        assert "p." in c.citation() or "pp." in c.citation()


class FakeIndex:
    def __init__(self, embs):
        self.embs = embs

    def embeddings(self, ids):
        return {i: self.embs[i] for i in ids}


def test_conflicting_numbers_across_documents_flagged():
    a = Chunk(chunk_id="a", document_id="d1", text="The recipient must pay within 180 days from the date of invoice.", page_start=1, page_end=1, title="Old", attempt="May 2025")
    b = Chunk(chunk_id="b", document_id="d2", text="The recipient must pay within 90 days from the date of invoice.", page_start=1, page_end=1, title="New", attempt="May 2027")
    same = np.array([1.0, 0.0])
    out = detect_conflicts([RetrievedChunk(chunk=a, score=1), RetrievedChunk(chunk=b, score=1)], FakeIndex({"a": same, "b": same}))
    assert len(out) == 1 and "180days" in out[0]["numbers_only_in_a"] and "90days" in out[0]["numbers_only_in_b"]
    # same document is never a "conflict"
    b2 = b.model_copy(update={"document_id": "d1"})
    assert detect_conflicts([RetrievedChunk(chunk=a, score=1), RetrievedChunk(chunk=b2, score=1)], FakeIndex({"a": same, "b": same})) == []
