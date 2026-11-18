import json

from examforge import evaluation
from examforge.config import Settings
from examforge.evaluation import compare, evaluate_retrieval, regression_check, run_benchmark


def rep(label, **metrics):
    base = {"valid_output_rate": 1.0, "quote_grounding_mean": 0.95, "duplicate_rate": 0.0, "insufficient_detection_rate": 1.0}
    return {"run_id": label, "label": label, "mock": False, "metrics": base | metrics}


def test_regression_detects_drops_rises_and_floors():
    assert regression_check(rep("a"), rep("b")) == []
    v = regression_check(rep("a"), rep("b", valid_output_rate=0.7, duplicate_rate=0.2))
    assert any("valid_output_rate dropped" in x for x in v) and any("below floor" in x for x in v)
    assert any("duplicate_rate rose" in x for x in v)


def test_prompt_comparison_table():
    rows = compare([rep("v1", valid_output_rate=0.8), rep("v2")])
    row = next(r for r in rows if r["metric"] == "valid_output_rate")
    assert row["v1"] == 0.8 and row["v2"] == 1.0


def test_retrieval_evaluation_metrics(engine):
    gold = [
        {"id": "blocked", "request": {"chapter": "Input Tax Credit", "topic": "Blocked credits", "attempt": "May 2027"},
         "query": "motor vehicles seating capacity thirteen persons", "expected_docs": ["sm_itc"],
         "expected_sections": ["3 APPORTIONMENT"], "key_phrases": ["thirteen persons"]},
        {"id": "bad_label", "request": {"chapter": "Input Tax Credit", "topic": "x", "attempt": "May 2027"},
         "expected_docs": ["sm_itc"], "key_phrases": ["a phrase that is not in the corpus"]},
        {"id": "none", "request": {"chapter": "Input Tax Credit", "topic": "x", "attempt": "Nov 2030"}, "expected_docs": [], "expect_insufficient": True},
    ]
    out = evaluate_retrieval(engine.retriever, gold, k=3)
    assert any("bad_label" in p for p in out["gold_problems"])  # invented labels are caught
    h = out["methods"]["hybrid"]
    assert h["hit_rate_at_k"] == 1.0 and h["filter_violations"] == 0 and h["traceable_rate"] == 1.0
    assert h["insufficient_handling_rate"] == 1.0
    assert 0 < h["precision_at_k"] <= 1


def test_benchmark_runs_end_to_end_and_records_provenance(data_dir, engine, monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "REPORT_DIR", tmp_path / "reports")
    s = Settings(llm_provider="mock", data_dir=data_dir, min_relevance=0.0)
    cases = [
        {"id": "ok", "request": {"chapter": "Input Tax Credit", "topic": "Eligibility and conditions for taking ITC", "n": 1, "attempt": "May 2027"}},
        {"id": "refuse", "expect_insufficient": True, "request": {"chapter": "Input Tax Credit", "topic": "x", "n": 1, "attempt": "Nov 2030"}},
    ]
    r = run_benchmark(s, cases, label="t", judge=True, embedder_name="hash")
    m = r["metrics"]
    assert r["mock"] is True and r["prompt_versions"]["question_setter"] == "v2"
    assert m["n_questions"] == 1 and m["insufficient_detection_rate"] == 1.0 and m["valid_output_rate"] is not None
    assert m["judge_means"]["clarity"] == 3
    saved = json.loads((tmp_path / "reports" / f"{r['run_id']}.json").read_text(encoding="utf-8"))
    assert saved["timestamp"] and saved["params"]["top_k"]
