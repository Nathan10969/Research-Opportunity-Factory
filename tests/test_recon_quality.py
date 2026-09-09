"""Fast, contract-level coverage for Task A recon quality gates.

These tests intentionally avoid the expensive corpus/execution fixture.  They
exercise the frozen report projector and its relevance validator directly.
"""

from datetime import datetime, timezone
import json

import pytest


def _parts():
    from idea_factory.recon import _ReportInput, _hash

    queries = ["q-concept", "q-mechanism", "q-failure", "q-evaluation"]
    lanes = dict(zip(("CONCEPT", "MECHANISM", "FAILURE", "EVALUATION"), ([q] for q in queries)))
    evidence = {
        "evidence_id": "e0",
        "source": "ARXIV",
        "source_id": "2501.1",
        "title": "Collider phenomenology of hadronic resonances",
        "year": 2025,
        "doi": None,
        "url": "https://arxiv.org/abs/2501.00001v1",
        "authors": ["A. Physicist"],
        "abstract_text": "Particle physics cross sections in proton collisions.",
        "query_ids": queries,
        "source_aliases": ["ARXIV:2501.1", "https://export.arxiv.org/abs/2501.00001v1"],
    }
    job = {
        "job_id": "job0", "opportunity_id": "opp0", "query_pack_sha256": "a" * 64,
        "normalized_evidence_manifest_sha256": "b" * 64, "protocol_hash": "c" * 64,
        "cache_key": "d" * 64, "searched_query_ids": queries,
        "lane_query_ids": lanes, "evidence": [evidence],
        "result_schema_sha256": _hash(_ReportInput.model_json_schema()),
    }
    receipts = [{"query_id": q, "status": "SUCCESS", "completed_at": "2026-09-09T10:00:00+00:00"} for q in queries]
    return job, receipts, evidence


def _item(job, *, decision="NO_DIRECT_COVERAGE_FOUND", reviews=None, priors=None, schema_hash=None):
    from idea_factory.recon import _ReportInput, NO_DIRECT_COVERAGE_REASON

    queries = job["searched_query_ids"]
    if reviews is None:
        reviews = [{"query_id": q, "status": "RELEVANT", "relevant_evidence_ids": ["e0"], "reason": "Relevant."} for q in queries]
    return _ReportInput.model_validate({
        "schema_version": "idea_factory.recon_report_result.v2",
        "result_schema_sha256": schema_hash or job["result_schema_sha256"],
        "job_id": job["job_id"], "opportunity_id": job["opportunity_id"],
        "query_pack_sha256": job["query_pack_sha256"],
        "normalized_evidence_manifest_sha256": job["normalized_evidence_manifest_sha256"],
        "protocol_hash": job["protocol_hash"], "cache_key": job["cache_key"],
        "searched_query_ids": queries, "searched_at": datetime.now(timezone.utc),
        "relevance_reviews": reviews, "nearest_priors": priors or [],
        "decision": decision, "decision_reason": NO_DIRECT_COVERAGE_REASON if decision == "NO_DIRECT_COVERAGE_FOUND" else "e0 covers the residual.",
    })


def _reviews(job, statuses=None, ids=None):
    statuses = statuses or {q: "RELEVANT" for q in job["searched_query_ids"]}
    ids = ids or {q: (["e0"] if statuses[q] == "RELEVANT" else []) for q in job["searched_query_ids"]}
    return [{"query_id": q, "status": statuses[q], "relevant_evidence_ids": ids[q], "reason": "Operator assessment."} for q in job["searched_query_ids"]]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign"])
def test_query_reviews_must_cover_exactly_once(mutation):
    from idea_factory.recon import _validate_relevance_reviews

    job, receipts, _ = _parts()
    reviews = _reviews(job)
    if mutation == "missing":
        reviews = reviews[:-1]
    elif mutation == "duplicate":
        reviews[-1] = dict(reviews[0])
    else:
        reviews[-1] = dict(reviews[-1], query_id="q-foreign")
    with pytest.raises(ValueError, match="exactly cover"):
        _validate_relevance_reviews(_item(job, reviews=reviews), job, receipts)


def test_foreign_and_cross_query_evidence_are_rejected():
    from idea_factory.recon import _validate_relevance_reviews

    job, receipts, evidence = _parts()
    with pytest.raises(ValueError, match="belong to its query"):
        _validate_relevance_reviews(_item(job, reviews=_reviews(job, ids={q: (["foreign"] if q == "q-concept" else ["e0"]) for q in job["searched_query_ids"]})), job, receipts)
    job["evidence"] = [dict(evidence, query_ids=["q-mechanism"])]
    with pytest.raises(ValueError, match="belong to its query"):
        _validate_relevance_reviews(_item(job), job, receipts)


def test_relevant_requires_ids_and_empty_cannot_be_irrelevant():
    from idea_factory.recon import _validate_relevance_reviews

    job, receipts, _ = _parts()
    no_ids = _reviews(job, ids={q: [] if q == "q-concept" else ["e0"] for q in job["searched_query_ids"]})
    with pytest.raises(ValueError, match="RELEVANT review requires evidence"):
        _validate_relevance_reviews(_item(job, reviews=no_ids), job, receipts)
    empty = _reviews(job, statuses={q: ("IRRELEVANT" if q == "q-concept" else "RELEVANT") for q in job["searched_query_ids"]})
    empty_receipts = [dict(r, status="EMPTY") if r["query_id"] == "q-concept" else r for r in receipts]
    with pytest.raises(ValueError, match="IRRELEVANT review requires a SUCCESS"):
        _validate_relevance_reviews(_item(job, reviews=empty), job, empty_receipts)


def test_physics_success_evidence_can_be_explicitly_irrelevant():
    from idea_factory.recon import _validate_relevance_reviews

    job, receipts, evidence = _parts()
    assert "particle physics" in evidence["abstract_text"].lower()
    statuses = {q: ("IRRELEVANT" if q == "q-concept" else "RELEVANT") for q in job["searched_query_ids"]}
    reviews = _reviews(job, statuses=statuses)
    _validate_relevance_reviews(_item(job, decision="COVERED", reviews=reviews), job, receipts)


def test_error_receipt_cannot_be_relevant():
    from idea_factory.recon import _validate_relevance_reviews

    job, receipts, _ = _parts()
    receipts[0] = dict(receipts[0], status="ERROR")
    with pytest.raises(ValueError, match="SUCCESS"):
        _validate_relevance_reviews(_item(job), job, receipts)


@pytest.mark.parametrize("decision", ["NO_DIRECT_COVERAGE_FOUND", "NEAR_PRIOR_WITH_RESIDUAL"])
def test_all_empty_or_missing_lane_cannot_be_route_ready(decision):
    from idea_factory.recon import _report_projection

    job, receipts, _ = _parts()
    raw = _item(job, decision=decision, reviews=_reviews(job, statuses={q: "EMPTY" for q in job["searched_query_ids"]}, ids={q: [] for q in job["searched_query_ids"]})).model_dump_json()
    empty = [dict(r, status="EMPTY") for r in receipts]
    result = _report_projection([job], {job["job_id"]: raw}, empty)
    assert result[1][0]["error_code"] == "RETRIEVAL_QUALITY_INCOMPLETE"
    partial = list(receipts); partial[0] = dict(partial[0], status="ERROR")
    result = _report_projection([job], {job["job_id"]: raw}, partial)
    assert result[1]


def test_covered_cannot_bind_irrelevant_prior():
    from idea_factory.recon import _validate_relevance_reviews

    job, receipts, _ = _parts()
    statuses = {q: ("IRRELEVANT" if q == "q-concept" else "RELEVANT") for q in job["searched_query_ids"]}
    irrelevant = dict(job["evidence"][0], evidence_id="e1", query_ids=["q-concept"])
    job["evidence"] = [job["evidence"][0], irrelevant]
    prior = [{"paper": irrelevant["title"], "exact_overlap": "x", "residual_difference": "y", "evidence_url_or_id": "e1"}]
    reviews = _reviews(job, statuses=statuses, ids={q: ([] if q == "q-concept" else ["e0"]) for q in job["searched_query_ids"]})
    with pytest.raises(ValueError, match="supported by a RELEVANT"):
        _validate_relevance_reviews(_item(job, decision="COVERED", reviews=reviews, priors=prior), job, receipts)


@pytest.mark.parametrize("reference", ["e0", "ARXIV:2501.1", "https://arxiv.org/abs/2501.00001v1"])
def test_canonical_alias_and_url_prior_references_are_accepted(reference):
    from idea_factory.recon import _validate_relevance_reviews

    job, receipts, _ = _parts()
    prior = [{"paper": job["evidence"][0]["title"], "exact_overlap": "x", "residual_difference": "y", "evidence_url_or_id": reference}]
    _validate_relevance_reviews(_item(job, decision="COVERED", priors=prior), job, receipts)


def test_stale_result_schema_hash_is_rejected_by_projection():
    from idea_factory.recon import _report_projection

    job, receipts, _ = _parts()
    raw = _item(job, schema_hash="0" * 64).model_dump_json()
    result = _report_projection([job], {job["job_id"]: raw}, receipts)
    assert result[1][0]["error_code"] == "STALE_REPORT_BINDING"


def test_report_reason_secret_is_rejected_without_raw_persistence():
    from idea_factory.recon import _report_projection

    job, receipts, _ = _parts()
    # Deliberately synthetic and assembled at runtime: it only exercises the
    # detector's shape, and the literal is never committed to the repository.
    secret = "sk-" + ("x" * 8)
    payload = _item(job).model_dump(mode="json")
    payload["relevance_reviews"][0]["reason"] = "Evidence note " + secret
    raw = json.dumps(payload)
    result = _report_projection([job], {job["job_id"]: raw}, receipts)
    assert result[0] == [] and result[3] == [] and result[5] == []
    assert result[1][0]["error_code"] == "REPORT_SECRET_DETECTED"
    assert secret not in json.dumps(result[1])
    assert secret not in json.dumps(result[2])


def test_query_generation_keeps_short_domain_terms_and_arxiv_boolean_caps():
    from idea_factory.models import Opportunity, OpportunityOperator
    from idea_factory.recon import _expected_execution_job_templates, generate_recon_queries

    opportunity = Opportunity(schema_version="idea_factory.opportunity.v1", opportunity_id="opp-terms", operator=OpportunityOperator.ASSUMPTION_BREAK, assumption_x="KV RL AL T5 3D 中文", observation_y="retrieval loss", condition_z="long traces", failure_f="cache failure", missing_capability_w="RAG control", alternative_explanation_a="drift", decisive_experiment="measure recall", supporting_card_ids=["c"], nearest_internal_neighbors=["n"], scope_compatibility="same")
    queries = generate_recon_queries(opportunity)
    tokens = {token.casefold() for q in queries for token in q.query.split()}
    assert all(term.casefold() in tokens for term in ("KV", "RL", "AL", "T5", "3D", "中文"))
    pack = [dict(q.model_dump(), schema_version="idea_factory.recon_query_job.v1", opportunity=opportunity.model_dump(mode="json"), opportunity_sha256="a" * 64) for q in queries]
    templates = _expected_execution_job_templates(pack, {"query_pack_sha256": "a" * 64})
    for template in templates:
        if template["source"] != "ARXIV":
            continue
        query = template["args"][1]
        variant = next(q.variant for q in queries if q.query == template["query"])
        assert query.count("ti:") == query.count("abs:")
        assert query.count("ti:") <= (4 if variant == "CURRENT_TERMS" else 6)
        if query.count("ti:") > 1:
            assert (") AND (" in query) == (variant == "CURRENT_TERMS")
            assert (") OR (" in query) == (variant == "GENERIC_SHAPE")
        assert "ti:" in query and "abs:" in query


def test_report_job_projection_replay_inputs_are_explicit_and_tamper_sensitive():
    from idea_factory.recon import _report_job_projection

    class Bundle:
        run = None
        manifest = {"opportunity_ids": ["opp0"], "query_pack_sha256": "a" * 64}
        queries = ({"query_id": "q0", "opportunity_id": "opp0", "lane": "CONCEPT", "variant": "CURRENT_TERMS", "query": "KV cache", "opportunity": {"opportunity_id": "opp0", "x": 1}, "opportunity_sha256": "e" * 64},)

    class Normalized:
        manifest = {"evidence": [], "raw_bindings": [], "normalized_evidence_manifest_sha256": "b" * 64}

    job = _report_job_projection(Bundle(), Normalized())[0]
    assert job["opportunity"] == Bundle.queries[0]["opportunity"]
    assert job["opportunity_sha256"] == "e" * 64
    assert job["query_context"][0]["query"] == "KV cache"
    mutated = dict(Bundle.queries[0], query="tampered")
    Bundle.queries = (mutated,)
    assert _report_job_projection(Bundle(), Normalized())[0]["query_context"][0]["query"] == "tampered"


@pytest.mark.parametrize("field", ["query", "opportunity", "opportunity_sha256"])
def test_report_job_projection_changes_when_frozen_input_is_tampered(field):
    from idea_factory.recon import _report_job_projection

    class Bundle:
        manifest = {"opportunity_ids": ["opp0"], "query_pack_sha256": "a" * 64}
        queries = ({"query_id": "q0", "opportunity_id": "opp0", "lane": "CONCEPT", "variant": "CURRENT_TERMS", "query": "KV cache", "opportunity": {"opportunity_id": "opp0", "x": 1}, "opportunity_sha256": "e" * 64},)

    class Normalized:
        manifest = {"evidence": [], "raw_bindings": [], "normalized_evidence_manifest_sha256": "b" * 64}

    baseline = _report_job_projection(Bundle(), Normalized())[0]
    row = dict(Bundle.queries[0])
    row[field] = "tampered" if field != "opportunity" else {"opportunity_id": "opp0", "x": 999}
    Bundle.queries = (row,)
    changed = _report_job_projection(Bundle(), Normalized())[0]
    assert changed != baseline


def test_one_lane_irrelevant_with_other_successful_relevant_lanes_is_not_route_ready():
    from idea_factory.recon import _report_projection

    job, receipts, _ = _parts()
    statuses = {q: ("IRRELEVANT" if q == "q-concept" else "RELEVANT") for q in job["searched_query_ids"]}
    reviews = _reviews(job, statuses=statuses)
    raw = _item(job, reviews=reviews).model_dump_json()
    result = _report_projection([job], {job["job_id"]: raw}, receipts)
    assert result[1][0]["error_code"] == "RETRIEVAL_QUALITY_INCOMPLETE"
    assert result[3] == [] and result[4] == [] and result[5] == []


@pytest.mark.parametrize("field", ["result_schema", "query_context", "opportunity"])
def test_validate_report_jobs_rejects_tampered_projected_job(tmp_path, monkeypatch, field):
    import idea_factory.recon as recon
    from idea_factory.artifacts import write_jsonl

    class Bundle:
        manifest = {"opportunity_ids": ["opp0"], "query_pack_sha256": "a" * 64}
        queries = ({"query_id": "q0", "opportunity_id": "opp0", "lane": "CONCEPT", "variant": "CURRENT_TERMS", "query": "KV cache", "opportunity": {"opportunity_id": "opp0", "x": 1}, "opportunity_sha256": "e" * 64},)

    class Normalized:
        manifest = {"evidence": [], "raw_bindings": [], "normalized_evidence_manifest_sha256": "b" * 64}

    monkeypatch.setattr(recon, "validate_query_pack", lambda *_: Bundle())
    monkeypatch.setattr(recon, "validate_normalized_evidence_bundle", lambda *_: Normalized())
    run = tmp_path / "run"; (run / "recon").mkdir(parents=True)
    Bundle.run = run
    job = recon._report_job_projection(Bundle(), Normalized())[0]
    mutated = dict(job)
    if field == "result_schema":
        mutated[field] = {"tampered": True}
    elif field == "query_context":
        mutated[field] = [{"query_id": "q0", "lane": "CONCEPT", "variant": "CURRENT_TERMS", "query": "tampered", "provider_status": {}, "evidence_ids": []}]
    else:
        mutated[field] = {"opportunity_id": "opp0", "x": 999}
    write_jsonl(run / "recon" / "report_jobs.jsonl", [mutated])
    with pytest.raises(ValueError, match="report job replay"):
        recon.validate_report_jobs(run, None)
