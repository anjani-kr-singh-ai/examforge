"""PDF -> cleaned, section-aware, metadata-rich chunks.

Pipeline per document (driven by data/metadata/sources.json):
  1. locate file, hash it (duplicate-document detection)
  2. extract text blocks per page with PyMuPDF (blocks ~ paragraphs, keeps reading order)
  3. detect scanned pages (no text but images) -> status needs_ocr instead of pretending success
  4. clean: drop repeated headers/footers and bare page numbers, de-hyphenate, normalise whitespace
  5. detect section headings (numbered headings, ALL-CAPS headings, "Question N" in RTP/MTP)
  6. chunk inside sections with token budget + overlap; every chunk keeps doc id, pages, section,
     chapter, attempt, law cutoff and an amendment marker
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz

from .schemas import Chunk, SourceDocument

SCANNED_PAGE_CHARS = 40  # a page with fewer extracted chars but with images is treated as scanned
SCANNED_DOC_RATIO = 0.5
AMENDMENT_RE = re.compile(
    r"w\.?\s?e\.?\s?f\.?|with effect from|(inserted|substituted|omitted|amended)\s+(vide|by|w)"
    r"|Finance\s+(\(No\.\s*\d\)\s+)?Act,?\s+20\d\d|amendment",
    re.I,
)
HEADING_NUM_RE = re.compile(r"^(\d{1,2}(\.\d{1,2}){0,3})\.?\s+([A-Z][A-Za-z0-9 ,&()\[\]'’/:.\-–]{3,120})$")
QUESTION_RE = re.compile(r"^(Question|Q\.?)\s*(No\.?\s*)?(?P<n>\d{1,3})\b", re.I)
PAPER_Q_RE = re.compile(r"^(?P<n>\d{1,2})\.\s+(\(a\)|[A-Z])")  # "5. (a) ..." / "1. In respect of ..."
PAPER_TYPES = {"RTP", "MTP", "Question Paper", "Suggested Answers"}
PAGE_NUM_RE = re.compile(r"^(page\s*)?\d{1,3}(\.\d{1,3})?(\s*of\s*\d+)?$", re.I)


class IngestionError(Exception):
    pass


@dataclass
class Block:
    page: int  # 1-based PDF page number
    text: str
    edge: bool = False  # block lies in the top/bottom page margin (header/footer zone)


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text.split()) * 1.33))  # word-count estimate; use a real tokenizer if chunk budgets must be exact


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- extraction
def extract_blocks(path: Path, include_pages: list[list[int]] | None = None) -> tuple[list[Block], int, int, int]:
    """Return (blocks, total_pages, pages_considered, scanned_pages). include_pages: 1-based inclusive ranges."""
    if not path.exists():
        raise IngestionError(f"file not found: {path}")
    with path.open("rb") as f:
        if b"%PDF" not in f.read(1024):  # PyMuPDF "repairs" almost anything; an HTML error page is not a PDF
            raise IngestionError(f"{path.name} is not a PDF (no %PDF header) - e.g. a saved HTML error page")
    try:
        doc = fitz.open(path, filetype="pdf")  # without filetype PyMuPDF happily opens HTML/text as a "document"
    except Exception as e:  # corrupt / not a PDF
        raise IngestionError(f"cannot open PDF {path.name}: {e}") from e
    blocks: list[Block] = []
    scanned = considered = 0
    with doc:
        for i, page in enumerate(doc, start=1):
            if include_pages and not any(a <= i <= b for a, b in include_pages):
                continue
            considered += 1
            h = page.rect.height or 1
            page_blocks = [b for b in page.get_text("blocks", sort=True) if b[6] == 0]
            chars = sum(len(b[4].strip()) for b in page_blocks)
            if chars < SCANNED_PAGE_CHARS and page.get_images():
                scanned += 1
            blocks.extend(Block(i, b[4], b[1] < 0.14 * h or b[1] > 0.84 * h) for b in page_blocks if b[4].strip())
        return blocks, doc.page_count, considered, scanned


# --------------------------------------------------------------------------- cleaning
def _norm_key(text: str) -> str:
    key = re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.strip().lower()))
    key = re.sub(r"(\s*#(\.#)*)+$", "", key)  # running headers carry varying page numbers...
    return re.sub(r"^(#(\.#)*\s*)+", "", key).strip()  # ...at either end


PRIVATE_USE_RE = re.compile(r"[-]")


def fix_glyphs(t: str) -> str:
    """ICAI PDFs use symbol fonts: the rupee sign extracts as a backtick, bullets as private-use code points,
    dashes as U+FFFD. Map them back so amounts and list structure survive."""
    t = re.sub(r"`\s?(?=[\d(])", "₹", t).replace("(`)", "(₹)")
    t = PRIVATE_USE_RE.sub("•", t)
    return t.replace("�", "–")


def clean_blocks(blocks: list[Block], page_count: int) -> list[Block]:
    """Drop repeated headers/footers (margin zone only) and page numbers; fix glyphs, hyphenation, whitespace."""
    pages_with = Counter()
    for key in {(b.page, _norm_key(b.text)) for b in blocks if b.edge}:
        pages_with[key[1]] += 1
    min_repeat = max(3, int(page_count * 0.2))
    out: list[Block] = []
    for b in blocks:
        t = b.text.strip()
        key = _norm_key(t)
        if b.edge and key and page_count >= 3 and pages_with[key] >= min_repeat and len(t) < 200:
            continue  # running header/footer (e.g. "INPUT TAX CREDIT 1.11", copyright line)
        if PAGE_NUM_RE.match(t):
            continue
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)  # de-hyphenate line breaks
        t = t.replace("­", "")
        t = re.sub(r"[ \t ]+", " ", t)
        t = fix_glyphs(re.sub(r"\s*\n\s*", " ", t).strip())  # a block is one paragraph: join its visual lines
        if t:
            out.append(Block(b.page, t))
    return out


def detect_heading(text: str, document_type: str) -> str | None:
    if document_type in PAPER_TYPES:  # question papers: question numbers are the citation unit
        m = QUESTION_RE.match(text) or PAPER_Q_RE.match(text)
        if m:
            return f"Question {m.group('n')}"
    if len(text) > 140:
        return None
    letters = [c for c in text if c.isalpha()]
    caps = bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.85
    m = HEADING_NUM_RE.match(text)
    if m and caps:  # ICAI main headings are numbered AND upper-case; numbered table rows are not
        return f"{m.group(1)} {m.group(3).strip()}"
    if caps and 2 <= len(text.split()) <= 14 and "=" not in text and not re.search(r"\d[\d,]{3,}", text):
        return text.strip()
    return None


# --------------------------------------------------------------------------- chunking
def _split_long(text: str, max_tokens: int) -> list[str]:
    sents = re.split(r"(?<=[.;:])\s+(?=[A-Z(])", text)
    parts, cur = [], ""
    for s in sents:
        if cur and estimate_tokens(cur + " " + s) > max_tokens:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    # a single "sentence" longer than the budget (tables) is hard-split by words
    final: list[str] = []
    for p in parts:
        words = p.split()
        step = max(1, int(max_tokens / 1.33))
        final.extend(" ".join(words[i : i + step]) for i in range(0, len(words), step))
    return final


def chunk_blocks(
    blocks: list[Block], doc: SourceDocument, chunk_size: int = 350, overlap: int = 60
) -> list[Chunk]:
    """Section-aware chunking: a new primary section starts a new chunk; overlap only within a section.

    Two heading levels: numbered headings ("3. ELIGIBILITY ...") and "Question N" are primary and break
    chunks; unnumbered ALL-CAPS headings ("STATUTORY PROVISIONS") are sub-headings that only label the
    chunk ("3 ELIGIBILITY ... > STATUTORY PROVISIONS"), so chunks don't fragment at every caption.
    """
    units: list[list] = []  # [primary, sub, Block]
    primary = sub = ""
    prev_primary = False
    for b in blocks:
        h = detect_heading(b.text, doc.document_type)
        numbered = bool(h) and (h.startswith("Question ") or bool(HEADING_NUM_RE.match(b.text)))
        if numbered:
            primary, sub = h, ""
        elif h and prev_primary and not primary.endswith("]"):  # primary heading wrapped onto a 2nd line
            old, primary = primary, f"{primary} {h}"
            for u in units:
                if u[0] == old:
                    u[0] = primary
            numbered = False  # at most one continuation line
        elif h:
            sub = h
        prev_primary = numbered
        for piece in _split_long(b.text, chunk_size) if estimate_tokens(b.text) > chunk_size else [b.text]:
            units.append([primary, sub, Block(b.page, piece)])

    chunks: list[Chunk] = []
    cur: list[Block] = []
    cur_section = units[0][0] if units else ""
    cur_label = cur_section

    def flush():
        if not cur:
            return
        text = "\n".join(u.text for u in cur)
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.document_id}::c{idx:04d}",
                document_id=doc.document_id,
                text=text,
                page_start=cur[0].page,
                page_end=cur[-1].page,
                section=cur_label,
                subject=doc.subject,
                chapter=doc.chapter or "",
                document_type=doc.document_type,
                attempt=doc.attempt or "",
                law_cutoff=doc.law_cutoff or "",
                title=doc.title,
                source_url=doc.source_url,
                amendment_marker=bool(AMENDMENT_RE.search(text)),
            )
        )

    for sec, subh, unit in units:
        size = sum(estimate_tokens(u.text) for u in cur)
        if not cur:
            cur_section, cur_label = sec, (f"{sec} > {subh}" if subh else sec)
        if cur and (sec != cur_section or size + estimate_tokens(unit.text) > chunk_size):
            same_section = sec == cur_section
            flush()
            # carry trailing units as overlap, only when continuing the same section
            carry: list[Block] = []
            if same_section:
                for u in reversed(cur):
                    if sum(estimate_tokens(c.text) for c in carry) + estimate_tokens(u.text) > overlap:
                        break
                    carry.insert(0, u)
            cur = carry
            cur_section, cur_label = sec, (f"{sec} > {subh}" if subh else sec)
        cur.append(unit)
    flush()
    return chunks


# --------------------------------------------------------------------------- manifest-level ingestion
def load_manifest(path: Path) -> list[SourceDocument]:
    return [SourceDocument(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


def save_manifest(path: Path, docs: list[SourceDocument]) -> None:
    path.write_text(json.dumps([d.model_dump() for d in docs], indent=2, ensure_ascii=False), encoding="utf-8")


def ingest_document(doc: SourceDocument, raw_dir: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    """Mutates doc's ingestion fields; returns chunks ([] when the document cannot be indexed)."""
    path = raw_dir / doc.filename
    if not path.exists():
        doc.extraction_status = "missing_file"
        doc.chunk_count = 0
        return []
    doc.sha256 = sha256_file(path)
    try:
        blocks, total, pages, scanned = extract_blocks(path, doc.include_pages)
    except IngestionError as e:
        doc.extraction_status = f"error: {e}"
        doc.chunk_count = 0
        return []
    doc.page_count = total
    if pages and scanned / pages >= SCANNED_DOC_RATIO:
        doc.extraction_status = f"needs_ocr ({scanned}/{pages} pages have no text layer)"
        doc.chunk_count = 0
        return []
    cleaned = clean_blocks(blocks, pages)
    if not cleaned:
        doc.extraction_status = "error: no text extracted"
        doc.chunk_count = 0
        return []
    chunks = chunk_blocks(cleaned, doc, chunk_size, overlap)
    doc.extraction_status = "ok" if not scanned else f"ok ({scanned} scanned pages skipped)"
    doc.chunk_count = len(chunks)
    return chunks


def ingest_all(manifest: Path, raw_dir: Path, index, chunk_size: int, overlap: int, processed_dir: Path | None = None) -> list[SourceDocument]:
    """Ingest every manifest document into `index` (retrieval.VectorIndex). Re-runnable (idempotent)."""
    docs = load_manifest(manifest)
    seen: dict[str, str] = {}
    for doc in docs:
        chunks = ingest_document(doc, raw_dir, chunk_size, overlap)
        if doc.sha256 and doc.sha256 in seen:
            doc.extraction_status = f"duplicate_of:{seen[doc.sha256]}"
            doc.chunk_count = 0
            index.delete_document(doc.document_id)
            continue
        if doc.sha256:
            seen[doc.sha256] = doc.document_id
        index.delete_document(doc.document_id)  # re-indexing replaces, never duplicates
        if chunks:
            index.add(chunks)
            if processed_dir:
                processed_dir.mkdir(parents=True, exist_ok=True)
                with (processed_dir / f"{doc.document_id}.jsonl").open("w", encoding="utf-8") as f:
                    f.writelines(c.model_dump_json() + "\n" for c in chunks)
    save_manifest(manifest, docs)
    return docs
