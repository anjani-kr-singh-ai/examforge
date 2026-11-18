# Design decisions

Each entry: decision → why → alternative considered → trade-off.

## 1. No LangChain / LangGraph

The workflow is a fixed sequence with one bounded revision loop. Plain functions + Pydantic contracts keep every
step testable as a pure function (`agents.coordinate` has unit tests for each rule). LangGraph would add state
plumbing and a dependency without adding a capability we use (no dynamic routing, no persistence of graph state,
no human interrupt inside a run - human review happens after the run, on stored records). If the flow grows
branches (e.g. per-mode sub-pipelines with different reviewers), LangGraph becomes worth it.

## 2. Vector store: ChromaDB

Persistent, embedded (no server), native metadata `where` filters, and it ships an ONNX MiniLM runtime so no
torch install is needed. At ~1.4k chunks, FAISS would be equally fast but needs separate metadata storage.
Postgres/pgvector or a hosted DB is unnecessary infrastructure for this scope.

## 3. Embedding model + chunk size 350

all-MiniLM-L6-v2 is small, local, free and good enough for topical recall, **but it truncates at 256
word-pieces**. With the suggested 700-1000-token chunks, everything after the first ~200 words would be invisible
to dense retrieval. So: 350-token section-aware chunks, a contextual header (`chapter | section`) in the embedding
input, and a lexical channel that sees the full chunk. Trade-off: more chunks; each context window holds 6 chunks
(~2.1k tokens), which is enough for single-topic MCQs. A long-context embedding API would allow larger chunks.

## 4. Hybrid retrieval (dense + TF-IDF, RRF)

Legal text depends on exact tokens ("section 16(4)", "thirtieth day of November", "180 days"). Measured on the
gold set, dense-only missed the required fact in 3 of 11 queries (time limit, depreciation, motor-vehicle seating
capacity), lexical-only in 1, hybrid in 1 (fact recall 0.73 / 0.91 / 0.91). Hybrid's miss: all six chunks were
on-topic §17(5) chunks (precision 1.0) but not the one containing the seating-capacity sentence.
Lexical-only scored slightly higher precision on this set because the gold queries reuse the material's own
wording; hybrid is kept as the default for robustness to paraphrased user topics. RRF needs no score calibration
between channels. The small gold set (13 queries) is not enough to tune fusion weights, so none are tuned.

## 5. Document-level applicability instead of "latest wins"

Tax material is versioned by attempt and law cut-off. The filter is "documents stated to apply to this attempt",
decided in Python from the manifest (one ICAI edition serves several attempts; list membership is awkward in
Chroma metadata filters). Other-period material requires an explicit opt-in and is flagged. The deterministic
version check on citations uses the same function, so retrieval and validation cannot disagree (a bug where the
check compared display labels was caught by the end-to-end tests).

## 6. Two-stage generation (stems, then distractors) + deterministic shuffle

- The distractor stage sees the correct answer, so it can avoid accidentally-correct options and target named
  misconceptions (taxonomy in `config/generation.yaml`).
- LLMs place correct answers non-uniformly; assembling options in code and shuffling with a seed derived from the
  stem removes answer-position bias and makes runs reproducible.
- Options must not reference other options (shuffling would break "both (a) and (b)"); statement-style items put
  (i)/(ii)/(iii) in the stem instead.
Cost: one extra LLM call per batch (not per question).

## 7. Grounding via verbatim quotes checked in code

Asking for citations is not enough - models invent plausible page numbers. Here the model only names a context
chunk and copies a sentence; code checks the sentence exists in that chunk (normalised substring, else
longest-common-substring coverage ≥ 0.8, ellipsis-aware) and takes page/section/URL from metadata. Cheap,
deterministic, and it catches paraphrased or invented support.

## 8. Independent reviewers, blind solver, coded acceptance

- Separate prompts, schemas and calls for examiner, evidence verifier and difficulty reviewer; none sees another's
  output, and none reviews its own work.
- The difficulty reviewer never sees the key; its answer vs the key is the strongest cheap signal for wrong keys
  and ambiguity.
- Acceptance rules are Python (`agents.coordinate`), not an LLM "final judge": deterministic, auditable, testable.
- Correlated errors: if one model plays every role, reviewers share its blind spots. `REVIEWER_MODEL` /
  `REVIEWER_PROVIDER` route reviewer stages to a different model. Recommended production setting: setter
  `claude-opus-5-5`, reviewers a different model family or at least a different Claude model.

## 9. Model choice

Default `claude-opus-5-5` with adaptive thinking at effort `high`: legal MCQ writing and adversarial review are
reasoning-heavy, and a wrong key is expensive (human time, student harm). Native structured outputs
(`messages.parse` with the Pydantic model) remove most malformed-JSON failures; the remaining ones (truncation,
schema violations) go through structured correction. The server-side refusal fallback is enabled. For cost, a
cheaper reviewer model or lower effort per stage are the first levers to measure - the benchmark reports tokens,
latency and estimated cost per run so that choice can be made on data.

### 9a. What the measured runs actually used: open models on AWS Bedrock

The build environment had no Anthropic key but did have an AWS SSO account, so the measured runs use Bedrock
(ap-south-1): **GLM 5** (`zai.glm-5`) as setter / distractor generator / reviser and **Qwen3 235B A22B 2507** as
examiner, evidence verifier, blind solver and judge. This also implements §8's recommendation: reviewers come
from a different model family than the setter. Prices from the AWS Pricing API (USD per 1M tokens, on-demand):
GLM 5 1.20 in / 3.84 out, Qwen3 0.26 / 1.04.
Open models on Bedrock have no server-enforced JSON schema here, so the provider asks for JSON matching the
Pydantic schema, strips code fences / `<think>` blocks, and relies on the same validation → structured-correction
retry loop as every other provider. Auth uses the standard AWS credential chain (SSO profile); nothing is stored
in `.env`.

## 10. Composite score is a review-priority index, not a probability

`composite_score` = unweighted mean of available components (deterministic pass, quote grounding, evidence verdict,
blind-solver agreement, difficulty match). It only sorts the review queue (weakest first). All components stay
visible; the UI states it is not a probability of correctness.

## 11. Deterministic vs LLM checks

Anything decidable from the text is deterministic (structure, key consistency, cues, grounding, applicability,
exact duplicates) - it is free, reproducible and runs first as a gate, so broken or duplicate items never consume
reviewer calls. Judgement calls (ambiguity, legal support, plausibility, difficulty) use LLMs and are treated as
evidence for a human, not verdicts on truth.

## 12. Option-similarity rule

Near-identical options were first flagged at fuzzy similarity > 0.92. That would penalise exactly the Hard
distractors we want ("shall" vs "shall not", 90 vs 180 days). The check is now equality after normalisation;
semantic overlap is left to the examiner.

## 13. Cost and latency shape

Per batch of n questions: 2 calls (setter, distractors) + 3n reviewer calls (+n judge in benchmarks) + up to
`MAX_REVISIONS` × (1 + 3) per flagged question. Deterministic gating avoids reviewer calls for broken items.
Reviewer calls are independent and could run concurrently (not implemented: sequential is simpler to log and the
UI shows progress per run).

## 14. Storage

A JSON file with atomic replace is enough for an educator-scale bank and keeps records human-readable and
diffable. A database would be the next step for concurrent reviewers.
