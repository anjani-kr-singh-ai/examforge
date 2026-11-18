"""ExamForge AI - Streamlit UI (secondary to the pipeline; every action calls the same Engine as the CLI).
Run: streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from examforge import analytics, evaluation, review
from examforge.config import ROOT, Settings, load_yaml
from examforge.export import export
from examforge.ingestion import ingest_all
from examforge.pipeline import Engine
from examforge.schemas import GenerationRequest

st.set_page_config(page_title="ExamForge AI", layout="wide")
BADGE = {"ai-generated": ":material/smart_toy: AI-generated", "ai-reviewed": ":material/rate_review: AI-reviewed (not approved)",
         "human-approved": ":material/verified: Human-approved"}
OPT = {True: ":material/check_circle:", False: ":material/radio_button_unchecked:"}


PROVIDERS = ["bedrock", "anthropic", "openai", "mock"]


def settings_for(provider: str) -> Settings:
    s = Settings()
    if provider != s.llm_provider:  # .env model/reviewer belong to the .env provider; use provider defaults
        s.llm_provider, s.reviewer_provider, s.reviewer_model = provider, "", ""
        if provider == "anthropic":
            s.llm_model = "claude-opus-5-5"
    return s


@st.cache_resource
def get_engine(provider: str) -> Engine:
    return Engine.build(settings_for(provider))


def subjects():
    return load_yaml("subjects.yaml")["subjects"]


# ------------------------------------------------------------------ sidebar
st.sidebar.title("ExamForge AI")
st.sidebar.caption("Evidence-grounded, self-evaluating CA Final MCQ engine")
default_provider = Settings().llm_provider
provider = st.sidebar.selectbox("LLM provider", PROVIDERS,
                                index=PROVIDERS.index(default_provider) if default_provider in PROVIDERS else PROVIDERS.index("mock"))
try:
    eng = get_engine(provider)
except Exception as e:  # missing SDK credentials etc.
    st.error(f"Could not initialise provider '{provider}': {e}")
    st.stop()
if eng.llm.provider.is_mock:
    st.sidebar.warning("MOCK mode: outputs are deterministic stand-ins built from source text, NOT real LLM generations.")
else:
    st.sidebar.success(f"Model: {eng.llm.provider.model}")
page = st.sidebar.radio("Page", ["Generate", "Review", "Documents", "Analytics", "Benchmark"])
reviewer = st.sidebar.text_input("Reviewer name", value="educator")


def show_question(q, expanded=False):
    r = q.reliability
    head = f"{BADGE[q.provenance]} · `{q.id}` · **{q.status}** · verdict `{r.verdict}` · {q.difficulty} · {q.question_type}"
    if q.generation.mock:
        head += " · MOCK"
    with st.expander(f"{q.mcq.question[:110]}", expanded=expanded):
        st.markdown(head)
        st.markdown(f"**{q.mcq.question}**")
        if q.mcq.assumptions:
            st.caption("Assumptions: " + "; ".join(q.mcq.assumptions))
        for o in q.mcq.options:
            st.markdown(f"{OPT[o.is_correct]} **({o.label})** {o.text}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Quote grounding", f"{r.quote_grounding:.0%}")
        c2.metric("Evidence verifier", r.evidence_status)
        c3.metric("Blind solver agrees", {True: "yes", False: "NO", None: "n/a"}[r.answer_agreement])
        c4.metric("Difficulty est.", f"{r.estimated_difficulty or 'n/a'} (asked {r.requested_difficulty})")
        for x in r.reasons:
            st.markdown(f":material/warning: {x}")


# ------------------------------------------------------------------ page 1: generate
if page == "Generate":
    st.header("Generate MCQs")
    subs = subjects()
    c1, c2, c3 = st.columns(3)
    subject = c1.selectbox("Subject", list(subs))
    chapter = c2.selectbox("Chapter", list(subs[subject]["chapters"]))
    topics = list((subs[subject]["chapters"][chapter] or {}).get("topics", {}) or {})
    topic = c3.selectbox("Topic", topics + ["(custom)"]) if topics else "(custom)"
    if topic == "(custom)":
        topic = c3.text_input("Custom topic", "")
    attempts = sorted({a for d in eng.docs for a in d.attempts()})
    c4, c5, c6, c7 = st.columns(4)
    attempt = c4.selectbox("Assessment period (exam attempt)", attempts + ["(any - not recommended)"],
                           index=attempts.index("May 2027") if "May 2027" in attempts else 0)
    difficulty = c5.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    qtype = c6.selectbox("Question type", ["single", "multiple"])
    mode = c7.selectbox("Mode", ["conceptual", "application", "case_based", "multi_concept", "amendment"])
    c8, c9 = st.columns(2)
    n = c8.number_input("Number of questions", 1, 10, 2)
    allow_hist = c9.checkbox("Intentionally include material for other attempts (flagged)")
    if not topic:
        st.stop()
    req = GenerationRequest(subject=subject, chapter=chapter, topic=topic, n=int(n), difficulty=difficulty,
                            question_type=qtype, mode=mode, attempt=None if attempt.startswith("(any") else attempt,
                            allow_historical=allow_hist)
    b1, b2 = st.columns(2)
    if b1.button("Preview retrieval context"):
        res = eng.retriever.retrieve(req)
        st.session_state["preview"] = res
    if b2.button("Generate", type="primary"):
        with st.spinner("Retrieving → setting → distractors → validating → AI examiner…"):
            st.session_state["run"] = eng.generate(req)
    res = st.session_state.get("preview")
    if res:
        st.subheader("Retrieved context")
        st.caption(f"Query: {res.query}")
        if res.insufficient:
            st.error(f"Insufficient evidence: {res.reason}")
        for w in res.warnings:
            st.warning(w)
        for c in res.conflicts:
            st.warning(f"Possible source conflict: {c}")
        for rc in res.chunks:
            with st.expander(f"{rc.chunk.citation()} · dense={rc.dense_similarity} lexical={rc.lexical_score}"):
                st.write(rc.chunk.text)
                st.caption(rc.chunk.source_url)
    run = st.session_state.get("run")
    if run:
        st.subheader(f"Run {run.run_id}: {run.status}")
        if run.message:
            (st.error if run.status != "ok" else st.info)(run.message)
        st.caption(f"LLM calls {run.stats.get('llm_calls')}, retries {run.stats.get('retries')}, "
                   f"latency {run.stats.get('latency_s')}s, tokens in/out {run.stats.get('input_tokens')}/{run.stats.get('output_tokens')}")
        for q in run.questions:
            show_question(q)
        st.info("Questions are saved as drafts. Open the Review page to approve, edit or reject them.")

# ------------------------------------------------------------------ page 2: review
elif page == "Review":
    st.header("Question review (human-in-the-loop)")
    qs = eng.bank.all()
    if not qs:
        st.info("No questions yet - generate some first.")
        st.stop()
    f1, f2 = st.columns(2)
    status = f1.multiselect("Status", ["DRAFT", "VALIDATED", "NEEDS_REVIEW", "APPROVED", "REJECTED"], default=["NEEDS_REVIEW", "DRAFT", "VALIDATED"])
    chap = f2.multiselect("Chapter", sorted({q.chapter for q in qs}))
    qs = [q for q in qs if q.status in status and (not chap or q.chapter in chap)]
    qs.sort(key=lambda q: (q.reliability.composite_score or 0))  # weakest first
    if not qs:
        st.info("Nothing matches the filter.")
        st.stop()
    qid = st.selectbox("Question (weakest composite first)", [q.id for q in qs],
                       format_func=lambda i: f"{i} · {next(q for q in qs if q.id == i).mcq.question[:90]}")
    q = eng.bank.get(qid)
    r = q.reliability
    st.markdown(f"### {BADGE[q.provenance]} · {q.status}" + (" · MOCK output" if q.generation.mock else ""))
    st.caption(f"{q.subject} / {q.chapter} / {q.topic} · requested {q.difficulty} · {q.question_type} · {q.mode} · "
               f"period {q.attempt} · model {q.generation.model} · prompts {q.generation.prompt_versions}")
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**{q.mcq.question}**")
        for o in q.mcq.options:
            st.markdown(f"{OPT[o.is_correct]} **({o.label})** {o.text}  \n<small>{o.explanation}"
                        f"{' · misconception: ' + o.misconception_category if o.misconception_category else ''}</small>", unsafe_allow_html=True)
        st.markdown("**Explanation**")
        st.write(q.mcq.explanation)
        st.markdown("**Sources (page refs from document metadata; quote match is checked automatically)**")
        for c in q.citations:
            st.markdown(f"- {':material/check:' if c.quote_found else ':material/close:'} [{c.title}]({c.source_url}) · {c.document_type} · {c.attempt} · "
                        f"PDF pp. {c.page_start}-{c.page_end} · {c.section}  \n  > {c.quote}  \n  match {c.quote_match:.2f}")
    with right:
        st.markdown(f"**Reliability verdict:** `{r.verdict}`")
        st.caption(f"Composite review-priority index {r.composite_score} - an unweighted heuristic for sorting, NOT a probability of correctness.")
        st.dataframe(pd.DataFrame([c.model_dump() for c in r.checks])[["name", "kind", "passed", "severity", "detail"]],
                     hide_index=True, width="stretch")
        for x in r.version_flags:
            st.warning(x)
        for x in r.disagreements:
            st.error(f"Reviewer disagreement: {x}")
        if r.similar_to:
            st.info(f"Similar questions: {r.similar_to}")
        for name in ("examiner", "evidence", "difficulty", "judge"):
            rv = getattr(q.reviews, name)
            if rv:
                with st.expander(f"AI review: {name}"):
                    st.json(rv.model_dump())
    st.divider()
    a1, a2, a3 = st.columns(3)
    with a1.form("approve"):
        cm = st.text_input("Approval comment")
        if st.form_submit_button("Approve", type="primary"):
            try:
                review.approve(eng.bank, qid, reviewer, cm)
                st.success("Approved")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    with a2.form("reject"):
        reason = st.selectbox("Reason", ["factual_error", "ambiguous", "weak_distractors", "wrong_difficulty", "off_topic", "duplicate", "outdated", "other"])
        issue = st.text_input("Factual issue (optional)")
        cm = st.text_input("Comment")
        if st.form_submit_button("Reject"):
            review.reject(eng.bank, qid, reviewer, reason, cm, issue or None)
            st.rerun()
    with a3.form("comment"):
        cm = st.text_area("Reviewer feedback")
        cd = st.selectbox("Corrected difficulty (optional)", ["", "Easy", "Medium", "Hard"])
        if st.form_submit_button("Save feedback"):
            review.add_comment(eng.bank, qid, reviewer, cm, cd or None)
            st.rerun()
    with st.expander("Edit question (re-runs deterministic validation; a human edit voids prior approval)"):
        with st.form("edit"):
            qt = st.text_area("Question", q.mcq.question)
            opts = []
            for o in q.mcq.options:
                c1, c2 = st.columns([2, 2])
                opts.append({"label": o.label, "text": c1.text_input(f"Option {o.label}", o.text),
                             "explanation": c2.text_input(f"Why {o.label}", o.explanation),
                             "misconception_category": o.misconception_category})
            keys = st.multiselect("Correct option(s)", [o.label for o in q.mcq.options], default=q.mcq.correct_labels)
            ex = st.text_area("Explanation", q.mcq.explanation)
            df = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=["Easy", "Medium", "Hard"].index(q.difficulty))
            cm = st.text_input("Edit note")
            if st.form_submit_button("Save edit"):
                eng.edit(qid, reviewer, question=qt, options=opts, correct_labels=keys, explanation=ex, difficulty=df, comment=cm)
                st.rerun()
    if st.button("Regenerate this question (old one is marked rejected)"):
        with st.spinner("Regenerating…"):
            run = eng.regenerate(qid, reviewer)
        st.write(run.status, run.message)
    with st.expander("Audit trail"):
        st.dataframe(pd.DataFrame([h.model_dump() for h in q.history]), hide_index=True, width="stretch")
        if q.original_mcq:
            st.markdown("**Original AI version**")
            st.json(q.original_mcq.model_dump())

# ------------------------------------------------------------------ page 3: documents
elif page == "Documents":
    st.header("Reference documents (ICAI)")
    st.caption("Complete ICAI documents are not redistributed. Download them from the source URLs into data/raw/ (see DATASET.md).")
    docs = pd.DataFrame([d.model_dump() for d in eng.docs])
    if docs.empty:
        st.info("No manifest at data/metadata/sources.json")
        st.stop()
    st.dataframe(docs[["document_id", "document_type", "chapter", "attempt", "applicable_attempts", "law_cutoff", "extraction_status",
                       "page_count", "chunk_count", "source_url"]], hide_index=True, width="stretch")
    st.subheader("Version map: which law cut-off each assessment period uses")
    vm = docs.explode("applicable_attempts").groupby("applicable_attempts")["law_cutoff"].apply(lambda s: sorted(set(map(str, s)))).reset_index()
    st.dataframe(vm, hide_index=True)
    st.caption("Selection is by stated applicability, never by recency. Mixed cut-offs inside one context are flagged at retrieval time.")
    st.metric("Chunks in vector index", eng.index.count())
    doc_id = st.selectbox("Inspect chunks of", docs.document_id.tolist())
    chunks = eng.index.chunks([doc_id])
    st.write(f"{len(chunks)} chunks · {sum(c.amendment_marker for c in chunks)} carry amendment markers")
    for c in chunks[:15]:
        with st.expander(f"{c.chunk_id} · pp.{c.page_start}-{c.page_end} · {c.section[:80]}" + (" · AMENDMENT" if c.amendment_marker else "")):
            st.write(c.text)
    if st.button("Re-ingest all documents"):
        with st.spinner("Extracting, cleaning, chunking, embedding…"):
            ingest_all(eng.s.manifest_path, eng.s.raw_dir, eng.index, eng.s.chunk_size, eng.s.chunk_overlap, eng.s.data_dir / "processed")
            eng.reload_sources()
        st.success("Done")

# ------------------------------------------------------------------ page 4: analytics
elif page == "Analytics":
    st.header("Quality analytics (measured from the question bank)")
    qs = eng.bank.all()
    if not qs:
        st.info("No questions yet.")
        st.stop()
    df = analytics.questions_frame(qs)
    m = st.columns(6)
    m[0].metric("Generated", len(df))
    m[1].metric("Passed deterministic", int(df.verdict.ne("rejected").sum() - (df.status == "DRAFT").sum()))
    m[2].metric("Needs review", int((df.status == "NEEDS_REVIEW").sum()))
    m[3].metric("Human-approved", int((df.status == "APPROVED").sum()))
    m[4].metric("Rejected", int((df.status == "REJECTED").sum()))
    m[5].metric("Mock-generated", int(df.mock.sum()))
    c1, c2, c3 = st.columns(3)
    c1.bar_chart(df.difficulty.value_counts())
    c2.bar_chart(df.chapter.value_counts())
    c3.bar_chart(df.evidence_status.value_counts())
    st.markdown(f"Mean quote grounding **{df.quote_grounding.mean():.2f}** · blind-solver agreement **{df.answer_agreement.dropna().mean():.2f}** "
                f"· difficulty match **{df.difficulty_match.dropna().mean():.2f}**" if df.answer_agreement.notna().any() else "")
    st.subheader("Syllabus coverage")
    cov = analytics.coverage(qs, eng.docs)
    st.caption(cov["note"])
    st.dataframe(cov["table"], hide_index=True, width="stretch")
    for rcm in cov["recommendations"]:
        st.markdown(f":material/arrow_forward: {rcm}")
    st.subheader("Duplicates & near-duplicates")
    dups = df[df.duplicate_of.notna() | (df.max_similarity >= eng.s.similar_threshold)][["id", "duplicate_of", "max_similarity", "topic", "status"]]
    st.dataframe(dups, hide_index=True)
    st.subheader("Feedback & failure patterns")
    st.json(analytics.feedback_analysis(qs))
    st.subheader("Prompt / model comparison (saved benchmark runs)")
    reps = sorted(evaluation.REPORT_DIR.glob("*.json")) if evaluation.REPORT_DIR.exists() else []
    if reps:
        st.dataframe(pd.DataFrame(evaluation.compare([json.loads(p.read_text(encoding="utf-8")) for p in reps[-4:]])), hide_index=True)
    if st.button("Export bank (JSON, CSV, paper, answer key)"):
        st.write({k: str(v) for k, v in export(qs, ROOT / "exports").items()})

# ------------------------------------------------------------------ page 5: benchmark
elif page == "Benchmark":
    st.header("Benchmark & testing")
    st.markdown("All numbers below are produced by actually running the suites. Mock-provider runs are labelled and test plumbing only.")
    if st.button("Run retrieval evaluation (no LLM cost)"):
        gold = evaluation.load_retrieval_gold(ROOT / "data/benchmark/retrieval_gold.yaml")
        with st.spinner("Evaluating dense vs lexical vs hybrid…"):
            st.session_state["reval"] = evaluation.evaluate_retrieval(eng.retriever, gold)
    if "reval" in st.session_state:
        rv = st.session_state["reval"]
        st.dataframe(pd.DataFrame(rv["methods"]).T)
        for p in rv["gold_problems"]:
            st.warning(p)
        st.dataframe(pd.DataFrame(rv["per_query"]), hide_index=True)
    st.divider()
    cases = evaluation.load_cases(ROOT / "data/benchmark/generation_cases.yaml")
    sel = st.multiselect("Generation cases", [c["id"] for c in cases], default=[c["id"] for c in cases][:3])
    label = st.text_input("Run label", "ui")
    setter = st.selectbox("question_setter prompt version", ["v2", "v1"])
    judge = st.checkbox("Run LLM-as-judge", True)
    if not eng.llm.provider.is_mock:
        st.warning("This calls the real LLM for every case (setter, distractors, 3 reviewers, judge, revisions) and costs money.")
    if st.button("Run generation benchmark"):
        s = settings_for(provider)
        s.prompt_versions = {"question_setter": setter}
        with st.spinner("Running benchmark…"):
            rep = evaluation.run_benchmark(s, [c for c in cases if c["id"] in sel], label=label, judge=judge)
        st.markdown(evaluation.benchmark_markdown(rep))
    reps = sorted(evaluation.REPORT_DIR.glob("*.json")) if evaluation.REPORT_DIR.exists() else []
    if len(reps) >= 2:
        st.subheader("Compare / regression")
        names = [p.stem for p in reps]
        b = st.selectbox("Baseline", names, index=0)
        c = st.selectbox("Candidate", names, index=len(names) - 1)
        rb, rc = evaluation.load_report(b), evaluation.load_report(c)
        st.dataframe(pd.DataFrame(evaluation.compare([rb, rc])), hide_index=True)
        v = evaluation.regression_check(rb, rc)
        (st.error("Regression: " + "; ".join(v)) if v else st.success("No regression against baseline"))
