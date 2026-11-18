import pymupdf

from examforge.ingestion import (
    Block, chunk_blocks, clean_blocks, detect_heading, extract_blocks, fix_glyphs, ingest_all, ingest_document, load_manifest,
)
from examforge.retrieval import HashEmbedder, VectorIndex
from examforge.schemas import SourceDocument
from tests.conftest import ITC_SECTIONS, make_pdf, write_manifest

DOC = SourceDocument(document_id="d1", title="T", filename="sm_itc.pdf", source_url="u", chapter="Input Tax Credit",
                     document_type="Study Material", attempt="May 2027", law_cutoff="30.04.2026")


def test_valid_pdf_chunks_keep_metadata_and_pages(data_dir):
    doc = DOC.model_copy()
    chunks = ingest_document(doc, data_dir / "raw", chunk_size=120, overlap=20)
    assert doc.extraction_status == "ok" and doc.page_count == 4 and doc.chunk_count == len(chunks) > 4
    assert all(c.document_id == "d1" and c.chapter == "Input Tax Credit" and c.attempt == "May 2027"
               and c.law_cutoff == "30.04.2026" for c in chunks)
    blocked = [c for c in chunks if "thirteen persons" in c.text]
    assert blocked and all(c.page_start == 3 and c.section.startswith("3 APPORTIONMENT") for c in blocked)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_headers_footers_and_page_numbers_removed(data_dir):
    chunks = ingest_document(DOC.model_copy(), data_dir / "raw", 350, 60)
    text = " ".join(c.text for c in chunks)
    assert "Institute of Chartered Accountants" not in text
    assert "INPUT TAX CREDIT  7." not in text and "INPUT TAX CREDIT 7." not in text


def test_scanned_pdf_reported_needs_ocr(tmp_path):
    pdf = pymupdf.open()
    for _ in range(3):
        page = pdf.new_page()
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), 0)
        pix.clear_with(200)
        page.insert_image(pymupdf.Rect(50, 50, 400, 400), pixmap=pix)
    pdf.save(tmp_path / "scan.pdf")
    doc = DOC.model_copy(update={"filename": "scan.pdf"})
    assert ingest_document(doc, tmp_path, 350, 60) == []
    assert doc.extraction_status.startswith("needs_ocr") and doc.chunk_count == 0


def test_empty_pdf_is_not_reported_as_success(tmp_path):
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.save(tmp_path / "empty.pdf")
    doc = DOC.model_copy(update={"filename": "empty.pdf"})
    assert ingest_document(doc, tmp_path, 350, 60) == []
    assert doc.extraction_status.startswith("error")


def test_missing_and_corrupt_files(tmp_path):
    missing = DOC.model_copy(update={"filename": "nope.pdf"})
    assert ingest_document(missing, tmp_path, 350, 60) == [] and missing.extraction_status == "missing_file"
    (tmp_path / "bad.pdf").write_bytes(b"<html>not a pdf</html>")
    bad = DOC.model_copy(update={"filename": "bad.pdf"})
    assert ingest_document(bad, tmp_path, 350, 60) == [] and bad.extraction_status.startswith("error")


def test_include_pages_filters(tmp_path):
    make_pdf(tmp_path / "a.pdf", ITC_SECTIONS)
    blocks, total, considered, _ = extract_blocks(tmp_path / "a.pdf", [[2, 3]])
    assert total == 4 and considered == 2 and {b.page for b in blocks} == {2, 3}


def test_glyph_repair_and_headings():
    assert fix_glyphs("value of `1,49,000 and (`)") == "value of ₹1,49,000 and (₹)"
    assert fix_glyphs(" item � dash") == "• item – dash"
    assert detect_heading("3. ELIGIBILITY AND CONDITIONS FOR TAKING", "Study Material") == "3 ELIGIBILITY AND CONDITIONS FOR TAKING"
    assert detect_heading("4 Truck purchased for delivery 80,000", "Study Material") is None  # numbered table row
    assert detect_heading("5. (a) Mr. X is a supplier", "RTP") == "Question 5"


def test_wrapped_heading_merged_and_subheading_labels():
    blocks = [Block(1, "3. ELIGIBILITY AND CONDITIONS FOR TAKING"), Block(1, "INPUT TAX CREDIT [SECTION 16]"),
              Block(1, "STATUTORY PROVISIONS"), Block(1, "Every registered person shall be entitled to take credit.")]
    chunks = chunk_blocks(clean_blocks(blocks, 1), DOC, 350, 60)
    assert chunks[0].section.startswith("3 ELIGIBILITY AND CONDITIONS FOR TAKING INPUT TAX CREDIT [SECTION 16]")


def test_duplicate_document_and_idempotent_reindex(data_dir):
    docs = [{"document_id": "a", "title": "A", "filename": "sm_itc.pdf", "source_url": "u", "document_type": "Study Material", "chapter": "Input Tax Credit"},
            {"document_id": "b", "title": "B", "filename": "sm_itc.pdf", "source_url": "u", "document_type": "Study Material", "chapter": "Input Tax Credit"}]
    write_manifest(data_dir, docs)
    idx = VectorIndex(data_dir / "idx", HashEmbedder())
    out = ingest_all(data_dir / "metadata" / "sources.json", data_dir / "raw", idx, 350, 60)
    assert out[1].extraction_status == "duplicate_of:a"
    n = idx.count()
    ingest_all(data_dir / "metadata" / "sources.json", data_dir / "raw", idx, 350, 60)
    assert idx.count() == n  # re-indexing replaces, never duplicates
    assert load_manifest(data_dir / "metadata" / "sources.json")[0].sha256
