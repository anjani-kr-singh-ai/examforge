# Dataset: ICAI reference material for CA Final Paper 5 (IDT)

## Selection

Goal: depth on a few GST chapters, plus exam-style material and an explicit amendment trail, all from
official ICAI sources (resource.cdn.icai.org / icai.org / boslive.icai.org). Every URL, title, attempt and cut-off
below was read from the ICAI landing page or the PDF itself; nothing is inferred. Complete documents are **not**
redistributed in this repo - `scripts/fetch_sources.py` downloads them from the recorded URLs and verifies SHA-256.

| document_id | Title (as printed) | Type | Applicable attempts | Law cut-off | PDF pages | Chunks |
|---|---|---|---|---|---|---|
| icai_idt_sm_ch1_supply | Chapter-1: Supply under GST | Study Material | May 2027, Nov 2027 | 30.04.2026 | 154 | 174 |
| icai_idt_sm_ch3_pos | Chapter 3: Place of Supply | Study Material | May 2027, Nov 2027 | 30.04.2026 | 136 | 160 |
| icai_idt_sm_ch5_tos | Chapter 5: Time of Supply | Study Material | May 2027, Nov 2027 | 30.04.2026 | 62 | 76 |
| icai_idt_sm_ch7_itc | Chapter 7: Input Tax Credit | Study Material | May 2027, Nov 2027 | 30.04.2026 | 280 | 312 |
| icai_idt_su_nov2026 | Statutory Update: Paper 5 - IDT (Nov 2026 examination) | Statutory Update | Nov 2026 | 30.04.2026 | 70 | 65 |
| icai_idt_rtp_nov2026 | Paper 5: IDT - Revision Test Papers - Nov 2026 | RTP | Nov 2026 | 30.04.2026 | 31 | 47 |
| icai_idt_rtp_may2026 | Paper 5: IDT - Revision Test Papers - May 2026 | RTP | May 2026 | 31.10.2025 | 30 | 46 |
| icai_final_gp2_model_tp_2025 | Model Test Papers (May & Nov 2025) Final Group II | MTP | May 2025, Nov 2025 | 31.10.2024 | 439 (only pp. 111-216, 346-439 indexed) | 424 |
| icai_idt_sa_may2026 | Suggested Answer - Final - May 2026 - Paper 5 IDT | Suggested Answers | May 2026 | not stated | 34 | 52 |

Total: 1,356 chunks. Source URLs, landing pages, verbatim cut-off statements and notes are in
`data/metadata/sources.json`.

Study-material edition and cut-off provenance: Module 1 initial pages
(https://resource.cdn.icai.org/94275bos-aps6312-p5-ip-m1.pdf) - "Relevant for May 2027 and November 2027
examinations", "Edition: September, 2026", law effective up to 30.04.2026. Not indexed (front matter).

### What could not be obtained, and substitutes

- **Live MTPs (2026 series)** are behind the BoS MTP portal login. Substitute: ICAI's published *Model Test
  Papers* compilation (Feb 2025). It also contains Paper 4 (DT), so only the Paper 5 page ranges are indexed
  (`include_pages`). Its law cut-off (31.10.2024) is the oldest in the set; it is only used when the user selects
  the May/Nov 2025 attempt or opts into other-period material.
- **May 2027 Statutory Update** is not yet published (ICAI promises notifications 01.05.2026-31.10.2026).
- **Value of Supply (Ch. 6)** deliberately not ingested: it stays in the topic tree so the system's
  "no source → refuse" path is demonstrable.
- Module 2 publication date (ITC chapter) was not verified, so it is `null`.

## Applicability model (amendment awareness)

Two different "dates" are kept apart:

- `attempt` / `applicable_attempts`: which examination attempt the document is prepared for (filtering key).
- `law_cutoff` (+ verbatim `law_cutoff_statement`): the date up to which the law in the document is amended.

`publication_date` is a third, separate field. Retrieval selects documents by **stated applicability**, never by
recency. For example the Nov 2026 RTP and the May 2027 study material share the 30.04.2026 cut-off, yet the study
material is not used for a Nov 2026 request unless the educator opts in - the system does not infer equivalence.

ICAI notes that most Finance Act 2026 amendments were *not* effective by 30.04.2026 and that their
applicability for May/Nov 2027 will be announced later; the study material describes some of them (e.g. ITC
chapter p. 90 footnote). Chunks containing amendment language are tagged `amendment_marker`; the setter prompt
forbids testing pending changes as current law; amendment mode only uses amendment-marked chunks.

## Extraction and cleaning (`examforge/ingestion.py`)

- PyMuPDF `get_text("blocks", sort=True)` → paragraph-like blocks with page numbers and positions.
- File must start with `%PDF` (PyMuPDF otherwise "opens" HTML error pages). Pages with < 40 characters and an
  image are counted as scanned; ≥ 50 % scanned → `needs_ocr`, not indexed.
- Running headers/footers: blocks in the top 14 % / bottom 16 % of the page whose normalised text (digits and
  leading/trailing page numbers removed) repeats on ≥ 20 % of pages - e.g. "INPUT TAX CREDIT 1.11",
  "© The Institute of Chartered Accountants of India", "33 NOVEMBER 2026 EXAMINATION".
- Glyph repair: ICAI fonts map **₹ to a backtick** (`` `1,49,000``), bullets to private-use code points (U+F076),
  dashes to U+FFFD. These are mapped back (693 ₹ restored in the ITC chapter alone).
- De-hyphenation and whitespace normalisation; legal text, numbers and section references are left untouched.

## Chunking

- **Section-aware**: numbered upper-case headings ("3. ELIGIBILITY AND CONDITIONS FOR TAKING / INPUT TAX CREDIT
  [SECTION 16]", wrapped lines merged) are primary sections and always start a new chunk; unnumbered upper-case
  captions ("STATUTORY PROVISIONS") label chunks as `primary > sub` without fragmenting them. In RTP/MTP/answers,
  question numbers are the section unit ("Question 5"), so citations can name a question.
- Size **350 tokens, 60 overlap** (word-count estimate). Overlap never crosses a primary section.
- Why not 700-1000 tokens: the embedding model truncates at 256 word-pieces, so long chunks would be embedded by
  their first ~200 words only (see DESIGN_DECISIONS.md).
- Every chunk keeps: chunk id, document id, title, URL, PDF page start/end, section, chapter, document type,
  attempt, law cut-off, amendment marker.

## Embeddings and retrieval

all-MiniLM-L6-v2 (384-d) via ChromaDB's ONNX runtime; embedding input = `chapter | section` header + text.
Chroma (cosine) stores vectors, text and metadata. Retrieval = document-level period filter → dense search
(Chroma `where`) + TF-IDF (1-2 grams, section-reference-aware tokenizer) over the same allowed set → reciprocal
rank fusion → top-k (6). See the measured comparison in the AI Quality Report.

## Limitations

Heuristic headings (a few merged labels); tables flattened to text; no OCR; page numbers are PDF page indices
(ICAI printed page labels such as "7.23" differ); the 2025 MTP law is older than the study material.
