"""Shared fixtures: synthetic ICAI-like PDFs, an isolated data dir, a hash-embedder engine, scripted LLMs.
No test needs network, API keys or the real ICAI files."""
from __future__ import annotations

import json
import os
from pathlib import Path

# Tests must never pick up a developer's .env (real providers cost money). Set before examforge imports:
# load_dotenv does not override variables that already exist, and "" means "use the default".
os.environ.update(LLM_PROVIDER="mock", LLM_MODEL="", REVIEWER_PROVIDER="", REVIEWER_MODEL="", PROMPT_VERSIONS="")

import pymupdf
import pytest

from examforge.config import Settings
from examforge.ingestion import ingest_all
from examforge.llm import LLMResponse
from examforge.mock_llm import MockProvider
from examforge.pipeline import Engine

ITC_SECTIONS = [
    ("1. INTRODUCTION", "Input tax credit is the backbone of the GST law and it ensures that tax is levied only on the value addition at each stage of the supply chain without any cascading effect."),
    ("2. ELIGIBILITY AND CONDITIONS FOR TAKING INPUT TAX CREDIT [SECTION 16]",
     "No registered person shall be entitled to the credit of any input tax unless he is in possession of a tax invoice or debit note issued by a registered supplier. "
     "The recipient shall pay to the supplier the value of supply along with tax within a period of one hundred and eighty days from the date of issue of invoice by the supplier. "
     "A registered person shall not be entitled to take input tax credit after the thirtieth day of November following the end of the financial year or furnishing of the relevant annual return, whichever is earlier."),
    ("3. APPORTIONMENT OF CREDIT & BLOCKED CREDITS [SECTION 17]",
     "Input tax credit shall not be available in respect of motor vehicles for transportation of persons having approved seating capacity of not more than thirteen persons including the driver. "
     "Where the goods or services are used partly for effecting taxable supplies and partly for exempt supplies, the amount of credit shall be restricted to so much of the input tax as is attributable to taxable supplies."),
    ("4. CREDIT IN SPECIAL CIRCUMSTANCES [SECTION 18]",
     "A person who has applied for registration within thirty days from the date on which he becomes liable to registration shall be entitled to take credit of input tax in respect of inputs held in stock on the day immediately preceding the date from which he becomes liable to pay tax."),
]


def make_pdf(path: Path, sections, header="INPUT TAX CREDIT", reps=3) -> Path:
    doc = pymupdf.open()
    for i, (head, body) in enumerate(sections, start=1):
        page = doc.new_page()
        page.insert_text((50, 30), f"{header}  7.{i}")
        page.insert_text((50, 100), head)
        page.insert_textbox(pymupdf.Rect(50, 120, 550, 700), (body + " ") * reps, fontsize=10)
        page.insert_text((50, 815), "The Institute of Chartered Accountants of India")
    doc.save(path)
    return path


def write_manifest(data: Path, docs: list[dict]) -> None:
    (data / "metadata").mkdir(parents=True, exist_ok=True)
    (data / "metadata" / "sources.json").write_text(json.dumps(docs), encoding="utf-8")


BASE_DOCS = [
    {"document_id": "sm_itc", "title": "Test SM Ch 7 ITC", "filename": "sm_itc.pdf", "source_url": "https://example.org/sm_itc.pdf",
     "chapter": "Input Tax Credit", "document_type": "Study Material", "attempt": "May 2027 and November 2027",
     "applicable_attempts": ["May 2027", "Nov 2027"], "law_cutoff": "30.04.2026"},
    {"document_id": "rtp_old", "title": "Test RTP May 2026", "filename": "rtp_old.pdf", "source_url": "https://example.org/rtp.pdf",
     "chapter": None, "document_type": "RTP", "attempt": "May 2026", "applicable_attempts": ["May 2026"], "law_cutoff": "31.10.2025"},
]


@pytest.fixture
def data_dir(tmp_path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_pdf(raw / "sm_itc.pdf", ITC_SECTIONS)
    make_pdf(raw / "rtp_old.pdf", [("Question 1", "Input tax credit on motor vehicles with approved seating capacity of not more than thirteen persons is blocked unless used for further supply of such vehicles or transportation of passengers.")] * 5, header="REVISION TEST PAPER")
    write_manifest(tmp_path, BASE_DOCS)
    return tmp_path


def make_engine(data_dir: Path, provider=None, **settings) -> Engine:
    s = Settings(llm_provider="mock", data_dir=data_dir, min_relevance=0.0, max_retries=3, **settings)
    eng = Engine.build(s, embedder_name="hash")
    eng.llm.sleep = lambda _: None
    if provider is not None:
        eng.llm.provider = provider
    ingest_all(s.manifest_path, s.raw_dir, eng.index, s.chunk_size, s.chunk_overlap)
    eng.reload_sources()
    return eng


@pytest.fixture
def engine(data_dir) -> Engine:
    return make_engine(data_dir)


class ScriptedProvider(MockProvider):
    """Mock that returns/raises queued items per schema name, falling back to the MockProvider behaviour.
    Queue items may be: a pydantic object, an Exception instance, or a callable(variables) -> object."""

    name = "scripted"
    model = "scripted-test"

    def __init__(self, script: dict[str, list] | None = None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.seen: list[tuple[str, str]] = []  # (schema, user prompt)

    def generate(self, system, user, schema, variables):
        self.seen.append((schema.__name__, user))
        q = self.script.get(schema.__name__)
        if q:
            item = q.pop(0)
            if isinstance(item, Exception):
                raise item
            return LLMResponse(item(variables) if callable(item) else item, 10, 5, self.model)
        return super().generate(system, user, schema, variables)
