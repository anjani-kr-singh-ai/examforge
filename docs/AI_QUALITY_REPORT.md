# AI Quality Report

Status at time of writing (2026-11-18): the pipeline, evaluation harness and 59-test suite run end-to-end on the
real ICAI corpus, and the generation benchmark has been run with **real models on AWS Bedrock** (GLM 5 setter,
Qwen3 235B reviewers and judge; §5.1). Every number below comes from an actual run (raw JSON in
`reports/benchmarks/`); mock-provider numbers are labelled as plumbing tests and say nothing about question quality.
No question has been human-approved yet: legal correctness of the samples has **not** been signed off by an educator.

## 1. Methodology

Quality is assessed at four levels, each with its own evidence:

| Level | What is measured | How |
|---|---|---|
| Retrieval | does the model get the right ICAI context? | gold set of 13 queries with labels taken from the documents (`data/benchmark/retrieval_gold.yaml`); precision@6, hit@6, MRR, required-fact recall@6, metadata-filter violations, traceability, refusal correctness |
| Per-question reliability (every generation) | structure, grounding, key correctness, ambiguity, difficulty, duplicates | deterministic checks + 3 independent LLM reviewers + coded coordinator → reliability report |
| Batch quality (benchmark) | the assignment's quality criteria across a fixed suite | 11 fixed cases (`data/benchmark/generation_cases.yaml`) across 4 chapters, 3 difficulties, both question types, 5 modes, plus 2 must-refuse cases; LLM-as-judge (separate prompt, anchored 1-5 rubric) |
| Human | final correctness and usefulness | approve / edit / reject with reasons, corrected difficulty, factual issue; approval rate per model/prompt version |

Mapping to the assignment's criteria:

| Criterion | Automated signal | Human role |
|---|---|---|
| Relevance to subject/chapter/topic | retrieval relevance; judge `relevance` | confirm |
| Factual correctness vs material | quote grounding (deterministic); evidence verifier claim checks; blind-solver agreement; judge `factual_correctness` | **decides** |
| Clarity / no ambiguity | examiner issues (`ambiguity`, `multiple_correct`); blind-solver disagreement; judge `clarity` | decides |
| Difficulty appropriateness | blind reviewer estimate vs requested; judge `difficulty_fit` | may correct (recorded) |
| Distractor quality | per-option plausibility (examiner); cue checks (length, absolute words); judge | decides |
| Answer-explanation consistency | deterministic label-claim and per-option verdict checks; judge | confirm |
| Uniqueness / diversity | exact / normalised / embedding duplicates; avoid-list; coverage analytics | - |

## 2. Validation checks

Deterministic (code, every question): required metadata; question text; exactly 4 options; labels A-D; answer
refs valid; `is_correct` flags = key; answer count matches type (1, or 2-3); options distinct; explanation
present; per-option explanations; explanation does not call a non-key option correct; per-option explanation
verdict matches key; no label-dependent options ("all of the above", "both (a) and (b)"); length cue (warning);
absolute-word cue (warning); evidence present; evidence chunk ids valid; quotes found in cited chunks; cited
documents applicable to the requested period (historical = warning). Plus exact/normalised duplicate detection.

LLM-assisted: examiner (per-option verdict + plausibility, issue list with severity), evidence verifier
(claim-by-claim support, outdated/conflicting flag), blind difficulty reviewer (answer without key, steps,
provisions, level), semantic duplicates (embeddings), judge (evaluation only).

Human: legal sign-off, pending-amendment applicability, final difficulty, approval.

## 3. Test results (actual)

`pytest -q` → **59 passed** (≈ 20 s, offline: synthetic PDFs, hash embedder, scripted LLM; the test suite forces
the mock provider so a developer `.env` can never make it call a paid model).
Coverage of the brief's required cases: valid/empty/scanned/missing/corrupt PDF, metadata and page preservation,
header removal, page-range filtering, duplicate documents and idempotent re-indexing; relevant query, period
filters, historical opt-in, empty results, relevance floor, amendment refusal, source references, numeric
conflicts; valid output, missing fields, truncated JSON, wrong type, invalid answer refs, wrong option count;
single/multiple correct, duplicates, missing explanation, invalid source; ambiguous question, unsupported answer,
difficulty mismatch, conflicting reviewers, key disputes; retries/backoff, structured correction, bounded retries,
fatal errors, every prompt version renders; benchmark execution, metrics, comparison, regression; full
generate → review → approve → edit → reject → regenerate → export → coverage flow; revision loop; reviewer
failure; setter-reported insufficiency; no generation when retrieval is insufficient. All 5 Streamlit pages were
rendered with Streamlit's AppTest harness without exceptions, including a Generate click.

## 4. Retrieval results (actual, real ICAI corpus, k = 6)

| method | precision@6 | hit@6 | MRR | required-fact recall@6 | filter violations | traceable | refusal handling |
|---|---|---|---|---|---|---|---|
| dense (MiniLM) | 0.79 | 1.00 | 0.92 | 0.73 | 0 | 100 % | 2/2 |
| lexical (TF-IDF) | 0.91 | 1.00 | 1.00 | 0.91 | 0 | 100 % | 2/2 |
| **hybrid (RRF, default)** | 0.88 | 1.00 | 0.94 | 0.91 | 0 | 100 % | 2/2 |

Findings:
- Dense-only missed the required fact in 3 of 11 queries ("thirtieth day of November" for the time-limit query;
  the depreciation proviso, where it drifted to chapter section 7; the motor-vehicle seating-capacity sentence).
  Exact legal wording matters, which is why the lexical channel exists.
- Lexical slightly beats hybrid on this set. The gold queries reuse the material's own terms, which favours
  TF-IDF; hybrid is kept for paraphrased topics, but this is a judgement, not something this set proves. Hybrid's
  one miss had all six chunks on-topic (§17(5)) but not the specific sentence - a candidate for a larger k or
  per-sub-topic queries.
- The period-filter query (Nov 2026 ITC) returned only RTP/Statutory-Update chunks: the May/Nov 2027 study material
  was correctly excluded (0 filter violations across 39 query-method pairs).
- Both refusal cases (no source for Value of Supply; unknown period) returned `insufficient` with a reason.
- Gold-set validation: every key phrase was verified to exist verbatim in its expected document.

Raw output: `reports/retrieval_eval.json`.

## 5. Generation benchmark

### 5.1 Real LLM: GLM 5 + Qwen3 on AWS Bedrock, prompt v1 vs v2 (actual runs)

Setup: setter / distractor generator / reviser = `zai.glm-5`; examiner, evidence verifier, blind solver and judge =
`qwen.qwen3-235b-a22b-2507-v1:0` (different model family on purpose); Bedrock ap-south-1; top-k 6; max 1 revision.
Same 11 fixed cases, only `question_setter` differs. Runs: `2026-11-18_glm5-v1_cea428`, `2026-11-18_glm5-v2_2063f6`.

```bash
python -m examforge.cli benchmark --label glm5-v2
python -m examforge.cli benchmark --label glm5-v1 --prompt-versions question_setter=v1
python -m examforge.cli compare 2026-11-18_glm5-v1_cea428 2026-11-18_glm5-v2_2063f6
python -m examforge.cli regress 2026-11-18_glm5-v1_cea428 2026-11-18_glm5-v2_2063f6   # -> no regression
```

| metric | setter v1 | setter v2 |
|---|---|---|
| questions generated (9 generating cases) | 17 | 17 |
| yield (cases that got all requested questions) | 1.00 | 1.00 |
| valid output (all deterministic checks pass, final) | 1.00 | 1.00 |
| first-pass valid (before any revision) | 0.94 | **1.00** |
| mean quote grounding | 0.97 | 0.98 |
| evidence verifier: supported | 1.00 | 1.00 |
| blind solver agrees with key | **0.88** | 0.82 |
| difficulty estimate = requested | 0.76 | **0.94** |
| passed for human review (no open blocking issue) | 0.59 (10) | **0.71 (12)** |
| revised at least once | 0.76 | 0.76 |
| rejected / duplicates | 0 / 0 | 0 / 0 |
| must-refuse cases refused (no source; Nov 2030) | 2/2 | 2/2 |
| LLM calls / failed / retry rate | 147 / 0 / 0.014 | 147 / 0 / 0.007 |
| latency per question (sequential) | 55.5 s | 54.7 s |
| tokens in / out | 523k / 107k | 541k / 111k |
| **estimated cost, whole run** | **$0.44** | **$0.46** |
| judge means (1-5): factual / consistency / others | 4.88 / 5.00 / 5.00 | 4.76 / 4.76 / 5.00 |

Cost is priced per call at each model's Bedrock on-demand rate (AWS Pricing API, ap-south-1): about
**$0.027 per question** including all reviews and revisions. The earlier single `generate` run (2 questions) cost $0.04.

Reading the numbers honestly:
- With 17 questions, one question moves a rate by 0.06. v2's gains (difficulty match +3 questions, passed-for-review
  +2, first-pass validity +1) and v1's (blind agreement +1, judge factual) are **within noise**; the regression gate
  passes, and v2 stays the default mainly for the difficulty-match gain. A real decision needs a larger case set.
- Every question still ends as `NEEDS_REVIEW`; none is auto-approved. "Passed for human review" means no blocking
  reviewer issue remained after one revision, not that it is correct.
- **The judge is too lenient to be useful as a quality metric**: clarity, relevance and distractor quality were 5.0 on
  every question in both runs, including questions the examiner flagged as ambiguous and one with a distractor that
  contradicts the stem (§6). Treat judge means as a smoke signal only; the examiner, blind solver and evidence verifier
  are the informative reviewers.
- Reviewer disagreements found in v2 (5 items needing revision): 2 ambiguity / multiple-correct disputes in multi-answer
  items, 1 "prescribed condition" wording dispute, 1 explanation inconsistency in amendment mode (rule 43(1)(f)), and
  1 **false alarm** where the Qwen examiner, blind solver and judge contradicted a key that is stated verbatim in the ICAI study material (§6).
- Diversity inside a batch is weak: the two `itc_eligibility_easy_single` questions test the same section 16
  conditions with overlapping distractors. They are not duplicates by the detector's thresholds, so this is only
  visible to a human or in coverage analysis.

Human approval rate: not measured yet - no educator has reviewed these questions. Exports for that review are in
`samples/glm5-v2/` and `samples/glm5-v1/`.

### 5.2 Mock provider - plumbing test (actual run `2026-11-18_mock-v2_a82690`)

The mock builds questions from source sentences and has lenient reviewers by design, so its "quality" metrics
are meaningless. What this run *does* verify on the real corpus: 11 cases → 17 questions, 80 LLM-stage calls,
0 failed calls, both must-refuse cases refused (`insufficient_detection_rate` 1.0), amendment mode found
amendment-marked ITC chunks, reports/metrics/judge plumbing works. The deterministic validator still rejected
2 of 17 mock questions (`valid_output_rate` 0.88) because the mock produced two identical distractors - the
check working as intended.

## 6. Sample output

All 34 real questions (both runs) with options, keys, explanations, sources and review status:
`samples/glm5-v2/` and `samples/glm5-v1/` (`questions_paper.md`, `questions_answer_key.md`, `questions.csv`,
`questions.json`). They are unreviewed AI drafts.

### 6.1 Real, passed for human review (v2, `q_b762caeb61`, ITC / Blocked credits, Medium, application)

```
Alpha Ltd., a manufacturing company, avails works contract services for renovation of its factory building. The
renovation expenses are treated as revenue expenditure in the books of account and are not capitalized. Which of the
following statements is correct regarding the eligibility of input tax credit on such works contract services?
  (A)  ITC is blocked because renovation of a factory building is treated as construction of an immovable property
       other than plant and machinery.
  (B)  ITC is allowed only if Alpha Ltd. is in the business of providing works contract services to others.
  (C)  ITC is blocked since works contract services for renovation of factory building fall under Section 17(5)(c).
  (D)* ITC is allowed since renovation expenses not capitalized do not tantamount to construction under GST law.
Source: Chapter 7: Input Tax Credit (Study Material, May 2027 and November 2027), pp. 77-78, section "4 APPORTIONMENT OF CREDIT & BLOCKED CREDITS [SECTION 17]"
  quote (found verbatim, match 1.0): "Thus, if re-construction, renovation, additions or alterations or repairs are
  not capitalized, it would not tantamount to construction under GST law."
grounding 1.0 · evidence supported · blind solver answered D (confidence 5) · difficulty est. Medium · composite 1.0
```

### 6.2 Real, flagged - and the reviewers were wrong (v2, `q_53a4cf501b`, Place of Supply, Hard, case-based)

An unregistered recipient in Jaipur buys event-organisation services from a Mumbai supplier for a conference held in
Dubai. Key: (C) Jaipur, location of the recipient. Evidence (verbatim, Study Material Ch. 3 pp. 132-133): *"As per
section 12(7)(a)(ii), when service by way of organization of an event is provided to an unregistered person, the place
of supply is the location where the event is actually held and if the event is held outside India, the place of
supply is the location of recipient."*

The examiner flagged a "wrong key", the blind solver answered (A) Dubai via section 13, and the judge scored factual
correctness 1/5. Supplier and recipient are both in India, so section 12 applies and the ICAI text supports the key.
These three ran on the same model (Qwen3), so this is a correlated reviewer error (the evidence verifier did rate the key as supported). The item was correctly
**not** passed silently: it went to a human with the dispute visible, which is the intended outcome. The same item has a
weak distractor - (D) "Bengaluru (location where the event is actually held within India)" contradicts the stem - and
no reviewer flagged it. Lessons: reviewers are evidence, not verdicts; use more than one reviewer model; a human must
check every disputed key against the cited page.

### 6.3 Mock provider (plumbing test, real corpus)

```
[q_32c578395d] DRAFT | verdict=needs_revision | Input Tax Credit / Blocked credits | MOCK
  (A)  only a works contractor can avail ITC ... subject to prior approval of the proper officer.
  (B)* only a works contractor can avail ITC on works contract services received by him.
  (C)  ... except where the recipient is a composition taxpayer.
  (D)  ... except where the recipient is a composition taxpayer.
  grounding=1.0 evidence=not_checked   - check failed: options_distinct (near-identical options [('C','D')])
```

The duplicate distractor was caught before any reviewer call (gate), and the item stayed out of the review
queue as DRAFT. Citations resolve to *Chapter 7: Input Tax Credit (Study Material, May 2027 and November 2027),
pp. 75-78, 4 APPORTIONMENT OF CREDIT & BLOCKED CREDITS [SECTION 17]* with the source URL.

## 7. Issues discovered during development and testing, and fixes

| # | Issue (how found) | Fix |
|---|---|---|
| 1 | Running header "INPUT TAX CREDIT 1.11" survived cleaning: varying page numbers split the repeat count (inspecting real chunks) | strip leading/trailing page numbers from the repeat key; only remove repeated blocks in page margins |
| 2 | ₹ extracted as a backtick, bullets as U+F076, dashes as U+FFFD (real text) | glyph repair; 693 ₹ restored in the ITC chapter |
| 3 | Numbered table rows ("4 Truck purchased ... 80,000") became section headings; wrapped headings split | numbered headings must be upper-case; wrapped heading lines merged; sub-headings only label chunks |
| 4 | RTP footer "33 NOVEMBER 2026 EXAMINATION" became every section name (real RTP) | margin zones measured on the real layout (bottom > 84 %) |
| 5 | ICAI Model Test Paper compilation includes Paper 4 (DT) (source research) | `include_pages` → only Paper 5 pages indexed; verified no DT leak |
| 6 | Dense retrieval missed exact legal facts (retrieval eval) | hybrid dense + TF-IDF with RRF |
| 7 | Option-distinctness at fuzzy > 0.92 would penalise good one-word Hard distractors (smoke test) | equality after normalisation only |
| 8 | Duplicate signature (stem + answer) would collide for generic statement-style stems (smoke test) | signature includes all options |
| 9 | Citation version check compared the attempt *label* ("May 2027 and November 2027") with "May 2027" - would have blocked every study-material citation (end-to-end test) | check uses the same document-level applicability function as retrieval |
| 10 | Examiner marking a keyed option "arguable" did not block when its verdict was "accept" (unit test) | key disputes always require revision |
| 11 | PyMuPDF "opened" an HTML file named .pdf (unit test) | `%PDF` header check; non-PDFs reported as errors |
| 12 | Study material describes Finance Act 2026 changes not yet effective (source research, ITC p. 90) | setter prompt rule: never test pending changes as current law; amendment markers on chunks |
| 13 | A revision can make a question worse | revision loop keeps the best-assessed version |
| 14 | Benchmark cost priced every token at the setter's rate, overstating cost when reviewers use a cheaper model (real Bedrock run) | each call priced at its own model |
| 15 | Test suite read the developer's `.env`: tests pinned the setter to mock but reviewer calls went to the real Bedrock model (noticed when the suite ran slowly) | `tests/conftest.py` forces mock and clears reviewer settings before import; suite back to ~20 s, no network |
| 16 | Streamlit provider picker lacked `bedrock` and reused the `.env` model with other providers; missing Anthropic key crashed only at generation time (user report) | picker includes bedrock; `.env` model/reviewers only apply to the `.env` provider; key checked at start-up with a clear message |
| 17 | Open models wrap JSON in code fences / `<think>` blocks (Bedrock provider) | outermost JSON object extracted before Pydantic validation; schema failures still go to structured correction |
| 18 | Same-model examiner, blind solver and judge contradicted a key that ICAI states verbatim (real run, §6.2) | documented; recommend a second reviewer model; human check of every disputed key |

## 8. Known limitations

- Real benchmark is small (17 questions per prompt version, one model pair). No educator has approved or rejected the
  real questions yet, so there is no human approval rate. Claude was not benchmarked (no Anthropic key).
- Reviewers share blind spots when one model plays every reviewer role (seen in §6.2, where Qwen3 examiner, solver and
  judge agreed on a wrong reading). The setter/reviewer split helps; a second reviewer model would help more.
- The LLM judge is lenient (5/5 on most dimensions for every question); it is not a reliable quality score.
- AI checks are evidence for a human, not proof: "supported" means supported by the retrieved chunks, which can
  themselves be incomplete; a legally correct answer can be flagged if its basis was not retrieved.
- The composite score is a sorting heuristic, not a calibrated probability.
- Heuristic heading detection (a few merged labels); tables flattened; no OCR; PDF page numbers (not ICAI
  printed labels such as "7.23").
- Gold sets are small (13 retrieval queries, 11 generation cases): good for regression detection, too small to
  tune parameters or claim statistical significance.
- The 2025 MTP law (31.10.2024) is older than the 2027 study material; it is only used for its own attempts or by
  explicit opt-in, with warnings.
- Difficulty is calibrated by an LLM against a written rubric; there is no student-response data, and none is
  fabricated. The schema and analytics could take empirical item statistics later.
- Feedback improves the system through analysis and versioned prompt changes, not model training.

## 9. Human review requirements

Every question remains a draft until an educator approves it in the Review page or CLI. The reviewer should check:
the key against the cited pages (links and quotes are shown), any `disagreement` or `version_flags`, the blind
solver's answer, and the difficulty estimate. Approval is refused while deterministic checks fail; any edit voids a
previous approval, re-runs deterministic validation on the original context, and keeps the AI original and an
audit trail.
