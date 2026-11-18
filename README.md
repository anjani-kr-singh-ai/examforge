# ExamForge AI

**An evidence-grounded, self-evaluating MCQ generation engine for CA Final Indirect Tax Laws (IDT).**

ExamForge AI generates CA Final MCQs from official ICAI material. It treats every model output as an unverified
draft: questions are grounded in retrieved ICAI text with verbatim, machine-checked evidence quotes; validated
by deterministic checks; challenged by independent AI reviewers (including a *blind* solver that never sees the
answer key); and only ever approved by a human.

| | |
|---|---|
| Subject scope | IDT (mandatory scope). Chapters indexed: Supply under GST, Time of Supply, Place of Supply, Input Tax Credit + RTPs, MTP compilation, Statutory Update, Suggested Answers |
| Reference period | Default **May 2027** attempt = ICAI Study Material Sept 2026 edition, GST law as on **30.04.2026** |
| Interfaces | CLI (`python -m examforge.cli`) and Streamlit UI (`streamlit run app.py`) |
| LLM | Pluggable: AWS Bedrock open models (measured runs: **GLM 5** setter + **Qwen3 235B** reviewers, SSO auth), Claude via the Anthropic SDK, any OpenAI-compatible endpoint, or an offline **mock** |
| Tests | 59 offline tests (`pytest`) - no API key, network or ICAI files needed |

Docs: [Architecture](docs/ARCHITECTURE.md) · [Dataset](docs/DATASET.md) · [AI Quality Report](docs/AI_QUALITY_REPORT.md) ·
[Design Decisions](docs/DESIGN_DECISIONS.md) · **Demo video:** https://youtu.be/UDbLNO6wAzs

---

## 1. Problem

Educators need many fresh, correct CA Final MCQs. A plain "PDF → LLM → MCQ" generator fails in predictable ways:
hallucinated provisions, law from the wrong amendment cycle, two defensible answers, giveaway distractors, wrong
keys, near-duplicates, and difficulty labels that don't match the question. ExamForge is built around catching
those failures and making every claim traceable to an ICAI page.

## 2. Architecture (one screen)

```
request ─► input validation ─► amendment-aware retrieval (doc-level period filter → dense + TF-IDF → RRF)
            │                      └─ relevance floor / no applicable docs ─► INSUFFICIENT_EVIDENCE (no generation)
            ▼
      Question Setter (stems + correct answers + verbatim evidence)      ── prompts/question_setter/v2.md
            ▼
      Distractor Generator (misconception-tagged wrong options)          ── prompts/distractor_generator/v1.md
            ▼
      assemble + deterministic shuffle (removes answer-position bias)
            ▼
      Deterministic checks (structure, key, type, explanations, cues, quote grounding, version scope)
            ▼   gate: broken or duplicate items never reach paid reviewers
      Duplicate detector (exact → normalised → embedding)
            ▼
      Critical Examiner │ Evidence Verifier │ Blind Difficulty Reviewer   (separate prompts, schemas, calls)
            ▼
      Review Coordinator (plain-Python rules) ──needs revision──► Reviser (bounded, keeps best version)
            ▼
      Reliability report ─► question bank (DRAFT / NEEDS_REVIEW / REJECTED) ─► HUMAN approve / edit / reject
            ▼
      analytics · coverage · feedback patterns · benchmark · regression gate · export (JSON/CSV/paper/key)
```

Details and data contracts: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 3. Key features

- **Evidence grounding that is checked, not trusted.** The model must quote the chunk it relies on; each quote is
  fuzzy-matched against the chunk text. Page, section, title and URL in citations come from chunk metadata, never
  from the model, so a citation cannot point at an invented page.
- **Amendment-aware retrieval.** Documents carry the exam attempts they apply to and the printed law cut-off
  (e.g. RTP May 2026 = law up to 31.10.2025; Study Material for May 2027 = law as on 30.04.2026). Retrieval only
  uses documents applicable to the requested attempt; other material must be opted into and is flagged. Mixed
  cut-offs and numeric conflicts between documents are reported. Recency is never treated as authority.
- **Explicit refusal.** No applicable source, weak retrieval, or a setter that reports thin context →
  `insufficient_evidence`, not a guess.
- **Independent AI review.** Adversarial examiner, claim-by-claim evidence verifier, and a blind solver whose
  answer is compared with the key deterministically. Disagreements are surfaced, not averaged away.
- **Deterministic acceptance rules** in code; AI can only *pass a question to a human*. Approval is human-only and
  is refused while structural checks fail; edits void prior approval and keep the AI original + audit trail.
- **Evaluation you can rerun.** Retrieval benchmark (precision@k, hit@k, MRR, required-fact recall, filter
  violations), generation benchmark on fixed cases, LLM-as-judge, prompt v1 vs v2 comparison, regression gate.

## 4. Technology

Python 3.11+ · Pydantic v2 (all stage contracts) · PyMuPDF (extraction) · ChromaDB (persistent vector store with
metadata filters) · all-MiniLM-L6-v2 via Chroma's ONNX runtime (no torch) · scikit-learn TF-IDF (lexical channel) ·
boto3 Bedrock Converse / Anthropic SDK (native structured outputs) / OpenAI SDK · Streamlit · pandas · pytest. No LangChain/LangGraph - the flow is a fixed
pipeline with one bounded loop; see [design decisions](docs/DESIGN_DECISIONS.md).

## 5. Setup

```bash
git clone https://github.com/anjani-kr-singh-ai/examforge.git && cd examforge
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then pick a provider (below), or LLM_PROVIDER=mock
aws sso login --profile <profile>                     # only for LLM_PROVIDER=bedrock
python scripts/fetch_sources.py                       # downloads the ICAI PDFs from official URLs into data/raw/
python -m examforge.cli ingest                        # extract → clean → chunk → embed → index (~4 min CPU)
```

The first ingest downloads the 80 MB MiniLM ONNX model to `~/.cache/chroma`.

## 6. Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `mock` if unset | `bedrock`, `anthropic`, `openai` (any OpenAI-compatible endpoint via `LLM_BASE_URL`), `mock` |
| `LLM_MODEL` | `claude-opus-5-5` | generation model, e.g. `zai.glm-5` on Bedrock |
| `AWS_PROFILE`, `AWS_DEFAULT_REGION` | - | Bedrock auth via the standard AWS chain (SSO profile); no key in `.env` |
| `LLM_EFFORT` | `high` | Claude effort level (ignored by other providers) |
| `REVIEWER_PROVIDER` / `REVIEWER_MODEL` | same as generator | use a *different* model for reviewers to reduce correlated errors |
| `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_RELEVANCE` | 6, 350, 60, 0.25 | retrieval |
| `MAX_RETRIES`, `MAX_REVISIONS` | 3, 1 | reliability bounds |
| `DUPLICATE_THRESHOLD`, `SIMILAR_THRESHOLD` | 0.92, 0.82 | semantic duplicate / "similar, show a human" |
| `PROMPT_VERSIONS` | from `config/generation.yaml` | e.g. `question_setter=v1` |

Example used for the measured runs (Bedrock, ap-south-1):

```
LLM_PROVIDER=bedrock
LLM_MODEL=zai.glm-5
AWS_PROFILE=<your-sso-profile>
AWS_DEFAULT_REGION=ap-south-1
LLM_MAX_TOKENS=8000
REVIEWER_PROVIDER=bedrock
REVIEWER_MODEL=qwen.qwen3-235b-a22b-2507-v1:0
```

Keys are read by the SDKs from the environment and are never logged. **Mock mode** is labelled `[MOCK]` in every
output, badge and export - it demonstrates the workflow, not model quality.

## 7. Using it

```bash
# retrieval only (see exactly what context the model would get)
python -m examforge.cli retrieve --chapter "Input Tax Credit" --topic "Blocked credits"
# generate 3 hard case-based questions for May 2027
python -m examforge.cli generate --chapter "Input Tax Credit" --topic "Blocked credits" --n 3 --difficulty Hard --mode case_based --full
python -m examforge.cli list --status NEEDS_REVIEW
python -m examforge.cli review approve <id> --reviewer you --comment "verified against SM p.38"
python -m examforge.cli edit <id> --reviewer you --correct B --difficulty Hard
python -m examforge.cli export --status APPROVED          # JSON, CSV, question paper, answer key
python -m examforge.cli coverage                          # syllabus coverage + feedback patterns
streamlit run app.py --server.address 127.0.0.1          # Generate · Review · Documents · Analytics · Benchmark (local only: it spends your LLM budget)
```

## 8. Evaluation

```bash
python -m examforge.cli eval-retrieval                                    # no LLM cost
python -m examforge.cli benchmark --label v2                              # fixed 11-case suite, judge on
python -m examforge.cli benchmark --label v1 --prompt-versions question_setter=v1
python -m examforge.cli compare <run_v1> <run_v2>
python -m examforge.cli regress <baseline_run> <candidate_run>            # exit 1 on regression
pytest -q                                                                 # 59 offline tests
```

Measured results and failure analysis: [docs/AI_QUALITY_REPORT.md](docs/AI_QUALITY_REPORT.md).

## 9. Deterministic vs LLM vs human

| Deterministic (code) | LLM-assisted | Human (required) |
|---|---|---|
| schema, 4 options, labels, key refs, key/flag consistency, answer count vs type, explanation presence & label consistency, label-dependent options, length/absolute-word cues, evidence quote grounding, cited-document applicability, exact/normalised duplicates, acceptance rules | ambiguity, multiple-valid answers, distractor plausibility, claim support, outdatedness, blind solving, difficulty estimate, semantic duplicates (embeddings), judge scores | legal correctness sign-off, applicability of pending amendments, final difficulty, approval |

## 10. Known limitations (short)

- Real benchmark is small: 17 questions per prompt version on one model pair (GLM 5 + Qwen3, about $0.45 per run).
  No educator has reviewed the real questions yet, and the LLM judge proved lenient (see the quality report).
- Heading detection is heuristic; a few section labels merge a sub-heading (e.g. "7 HOW ITC IS UTILISED STATUTORY PROVISIONS").
- No OCR (scanned PDFs are detected and reported as `needs_ocr`, not indexed). Tables are kept as text.
- Live 2026 ICAI MTPs need a student login, so the 2025 Model Test Paper compilation (older law) is used and tagged.
- Using one model for setter and reviewers gives correlated errors; configure `REVIEWER_MODEL` to diversify.
- Direct Tax (optional bonus) is not implemented; adding it is a `subjects.yaml` block + manifest entries.

Full list: [AI Quality Report §8](docs/AI_QUALITY_REPORT.md#8-known-limitations).

## 11. Repository layout

```
examforge/            pipeline package: schemas, llm, prompts, ingestion, retrieval, generation, agents,
                      validation, pipeline, review, evaluation, analytics, export, mock_llm, cli
prompts/<stage>/vN.md versioned prompts (7 stages)
config/               subjects.yaml (topic tree), generation.yaml (rubric, modes, active prompts), evaluation.yaml
data/metadata/        sources.json (ICAI manifest)   data/benchmark/  gold sets + fixed cases
reports/              retrieval_eval.json, benchmarks/*.json (actual runs)
samples/              real GLM 5 output exported per prompt version: paper, answer key, CSV, JSON (unreviewed drafts)
tests/                offline test suite        scripts/fetch_sources.py   app.py (Streamlit)
```
