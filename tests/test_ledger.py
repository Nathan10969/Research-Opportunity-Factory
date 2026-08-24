import pytest
import hashlib
import json
from pathlib import Path


def _shape(**updates: object) -> dict[str, object]:
    body = {
        "old_assumption": "fixed cache state", "failure_mechanism": "stale evidence",
        "missing_capability": "versioned invalidation", "target_scope": "KV cache serving",
        "mechanism_family": "KV_SEMANTIC_LEASE", "residual_difference": "",
        "source_opportunity_id": "opp-1",
    }
    body.update(updates)
    return body


def _legacy(**updates: object) -> dict[str, object]:
    body = {
        "legacy_id": "D06-LEASEKV-01", "display_name": "LeaseKV", "one_line": "versioned invalidation",
        "mechanism_family": "KV_SEMANTIC_LEASE", "target": "KV cache serving",
        "status": "SYSTEMS_CANDIDATE", "source": "legacy", "evidence": "legacy",
        "source_table": "LIVE", "parse_confidence": "HIGH", "source_line": 1, "raw_row": "raw",
    }
    body.update(updates)
    return body


def _reviewed_shape(**updates: object) -> dict[str, object]:
    body = {
        "schema_version": "idea_factory.reviewed_legacy_shape_assignment.v1", "reviewed": True,
        "old_assumption": "UNKNOWN", "failure_mechanism": "stale evidence",
        "missing_capability": "other capability", "target_scope": "KV cache serving",
    }
    body.update(updates)
    return body


def test_fingerprint_is_unicode_normalized_and_unambiguous() -> None:
    from idea_factory.ledger import shape_fingerprint

    first = shape_fingerprint(_shape(target_scope="café  serving"))
    second = shape_fingerprint(_shape(target_scope="cafe\u0301 serving"))
    assert first == second
    assert first != shape_fingerprint(_shape(missing_capability="other capability"))


def test_renamed_same_mechanism_and_target_is_internal_duplicate() -> None:
    from idea_factory.ledger import classify_shape

    decision = classify_shape(_shape(missing_capability="new renamed capability"), [_legacy()])
    assert decision.status == "INTERNAL_DUP"
    assert decision.blocked
    assert decision.nearest_legacy_ids == ("D06-LEASEKV-01",)


def test_killed_exact_match_is_externally_covered() -> None:
    from idea_factory.ledger import classify_shape

    decision = classify_shape(_shape(), [_legacy(status="EXTERNALLY_COVERED", source_table="KILLED_BLOCKLIST")])
    assert decision.status == "EXTERNALLY_COVERED"
    assert decision.blocked


def test_adjacent_requires_explicit_residual_but_can_remain_eligible() -> None:
    from idea_factory.ledger import classify_shape

    adjacent = _legacy(
        mechanism_family="unrelated explicit family", target="KV cache serving", one_line="prose only",
        reviewed_shape_assignment=_reviewed_shape(),
    )
    blocked = classify_shape(_shape(mechanism_family="another family", missing_capability="new capability"), [adjacent])
    assert blocked.status == "ADJACENT_NEEDS_RESIDUAL"
    assert blocked.blocked
    clear = classify_shape(_shape(mechanism_family="another family", missing_capability="new capability", residual_difference="handles mutable provenance"), [adjacent])
    assert clear.status == "INTERNAL_ADJACENT"
    assert not clear.blocked


def test_unreviewed_legacy_prose_cannot_fabricate_shape_or_adjacent_decision() -> None:
    from idea_factory.ledger import classify_shape, legacy_shape, shape_fingerprint

    prose = _legacy(mechanism_family="stale evidence", target="KV cache serving", one_line="other capability")
    projected = legacy_shape(prose)
    assert projected == {
        "old_assumption": None, "failure_mechanism": None,
        "missing_capability": None, "target_scope": None,
        "fingerprint_available": False, "shape_fingerprint": None,
    }
    with pytest.raises(ValueError, match="unavailable"):
        shape_fingerprint(projected)
    decision = classify_shape(
        _shape(mechanism_family="another family", missing_capability="new capability"), [prose]
    )
    assert decision.status == "CLEAR"


def test_low_confidence_legacy_never_hard_blocks() -> None:
    from idea_factory.ledger import classify_shape

    decision = classify_shape(_shape(), [_legacy(parse_confidence="LOW")])
    assert decision.status == "NEEDS_HUMAN_REVIEW"
    assert decision.blocked


def test_mechanism_family_is_required_not_title_inferred() -> None:
    from idea_factory.ledger import ShapeAssignment

    with pytest.raises(ValueError, match="mechanism_family"):
        ShapeAssignment.model_validate(_shape(mechanism_family=" "))


def test_empty_quality_handoff_requires_validated_bundles_and_is_idempotent(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.corpus import CorpusRouterConfig
    from idea_factory.legacy_ledger import import_legacy_ledger
    from idea_factory.ledger import (
        emit_shape_assignment_jobs, ingest_shape_assignments, publish_internal_dedup,
        validate_dedup_bundle,
    )
    from idea_factory.opportunities import emit_mining_jobs, ingest_opportunity_results
    from idea_factory.quality import publish_quality
    from test_opportunities import _build_run, _empty_result

    run, prompt = _build_run(tmp_path)
    jobs = read_jsonl(emit_mining_jobs(run, prompt))
    ingest_opportunity_results(run / "opportunities" / "mining_jobs.jsonl", [_empty_result(job) for job in jobs])
    publish_quality(run)
    papers = tmp_path / "papers"; legacy = papers / "_ideas" / "idea_ledger.md"
    config = CorpusRouterConfig(tmp_path / "notes", (papers / "list.txt",), legacy, 1, 1, ("KV_CACHE",), ())
    import_legacy_ledger(run, config)
    shape_jobs = emit_shape_assignment_jobs(run, config)
    assert read_jsonl(shape_jobs) == []
    ingest_shape_assignments(shape_jobs, [])
    publish_internal_dedup(run, config)
    first = {name: (run / "ledger" / name).read_bytes() for name in ("ready_after_internal_dedup.jsonl", "blocked.jsonl", "adjacent.jsonl", "dedup_outcomes.jsonl")}
    validate_dedup_bundle(run, config)
    publish_internal_dedup(run, config)
    assert first == {name: (run / "ledger" / name).read_bytes() for name in first}


def test_shape_result_must_bind_one_current_job_and_dedup_replays(tmp_path: Path) -> None:
    import json
    from idea_factory.artifacts import read_jsonl
    from idea_factory.corpus import CorpusRouterConfig
    from idea_factory.legacy_ledger import import_legacy_ledger
    from idea_factory.ledger import emit_shape_assignment_jobs, ingest_shape_assignments, publish_internal_dedup, validate_dedup_bundle
    from idea_factory.opportunities import emit_mining_jobs, ingest_opportunity_results
    from idea_factory.quality import publish_quality
    from test_opportunities import _build_run, _empty_result

    run, prompt = _build_run(tmp_path)
    mining = read_jsonl(emit_mining_jobs(run, prompt)); first = mining[0]
    results = [_empty_result(job) for job in mining]
    wrapper = json.loads(results[0])
    wrapper["opportunities"] = [{
        "schema_version": "idea_factory.opportunity.v1", "opportunity_id": "opp-current", "operator": first["operator"],
        "assumption_x": "The cache remains resident during long serving traces.", "observation_y": "Cards report retrieval degrades after eviction.",
        "condition_z": "under long serving traces", "failure_f": "cache eviction causes retrieval loss", "missing_capability_w": "eviction-aware retrieval control",
        "alternative_explanation_a": "cache eviction", "decisive_experiment": "Compare recall after forced cache eviction against cache eviction.",
        "supporting_card_ids": [first["card_ids"][0]], "nearest_internal_neighbors": [first["neighbor_ids"][0]],
        "scope_compatibility": "Both cards cover KV cache serving over long traces.", "inference_flags": [],
    }]
    results[0] = json.dumps(wrapper)
    ingest_opportunity_results(run / "opportunities" / "mining_jobs.jsonl", results); publish_quality(run)
    papers = tmp_path / "papers"; legacy = papers / "_ideas" / "idea_ledger.md"
    legacy.write_text("| id | one-line | mechanism family | target | status / venue | source |\n|---|---|---|---|---|---|\n| L | other | OTHER | elsewhere | old | source |\n", encoding="utf-8")
    config = CorpusRouterConfig(tmp_path / "notes", (papers / "list.txt",), legacy, 1, 1, ("KV_CACHE",), ())
    import_legacy_ledger(run, config)
    jobs = read_jsonl(emit_shape_assignment_jobs(run, config)); assert len(jobs) == 1
    result = {
        "schema_version": "idea_factory.shape_assignment_result.v1", **{key: jobs[0][key] for key in ("job_id", "opportunity_id", "quality_record_sha256", "quality_bundle_sha256", "legacy_index_sha256")},
        "old_assumption": "cache remains resident", "failure_mechanism": "cache eviction causes retrieval loss", "missing_capability": "eviction-aware retrieval control",
        "target_scope": "KV cache serving", "mechanism_family": "EVICTION_CONTROL", "residual_difference": "",
    }
    stale = dict(result); stale["opportunity_id"] = "stale"
    with pytest.raises(ValueError, match="stale"):
        ingest_shape_assignments(run / "ledger" / "shape_assignment_jobs.jsonl", [json.dumps(stale)])
    duplicate_key_raw = json.dumps(result).replace('"mechanism_family": "EVICTION_CONTROL"', '"mechanism_family": "EVICTION_CONTROL", "mechanism_family": "OTHER"')
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        ingest_shape_assignments(run / "ledger" / "shape_assignment_jobs.jsonl", [duplicate_key_raw])
    wrong_version = dict(result); wrong_version["schema_version"] = "idea_factory.shape_assignment_result.v999"
    with pytest.raises(ValueError, match="invalid shape assignment result"):
        ingest_shape_assignments(run / "ledger" / "shape_assignment_jobs.jsonl", [json.dumps(wrong_version)])
    ingest_shape_assignments(run / "ledger" / "shape_assignment_jobs.jsonl", [json.dumps(result)])
    assert read_jsonl(run / "ledger" / "shape_assignment_results.jsonl")[0]["fingerprint_available"] is True
    outputs = publish_internal_dedup(run, config)
    assert read_jsonl(outputs["ready"])[0]["status"] == "CLEAR"
    validate_dedup_bundle(run, config)


def _task11_reviewed_run(
    tmp_path: Path,
    decisions: tuple[str, str, str] = ("PASS_TO_HUMAN", "KILL", "NARROW"),
    *,
    near_prior: bool = False,
):
    from idea_factory.artifacts import read_jsonl
    from idea_factory.review import emit_review_jobs, ingest_review_results
    from idea_factory.routes import emit_route_jobs, ingest_route_results
    from test_routes_review import _ready_run, _review_result, _route_result

    decision = "NEAR_PRIOR_WITH_RESIDUAL" if near_prior else "NO_DIRECT_COVERAGE_FOUND"
    run, config = _ready_run(tmp_path, decision=decision)
    route_jobs_path = emit_route_jobs(run, config, allow_test_ready=True)
    route_job = read_jsonl(route_jobs_path)[0]
    ingest_route_results(route_jobs_path, [_route_result(route_job)], config, allow_test_ready=True)
    review_jobs_path = emit_review_jobs(run, config, allow_test_ready=True)
    review_jobs = read_jsonl(review_jobs_path)
    ingest_review_results(
        review_jobs_path,
        [_review_result(job, verdict) for job, verdict in zip(review_jobs, decisions)],
        config,
        allow_test_ready=True,
    )
    return run, config


def _human_score(job: dict[str, object], *, reasons: list[str] | None = None, identity: str = "reviewer-7") -> str:
    return json.dumps(
        {
            "schema_version": "idea_factory.human_score.v1",
            "job_id": job["job_id"],
            "human_job_sha256": job["human_job_sha256"],
            "specific_novelty": 4,
            "importance": 4,
            "paper_potential": 4,
            "feasibility": 5,
            "excitement": 4,
            "evidence_clarity": 3,
            "reason_codes": reasons or ["WANT_TO_TEST_NOW"],
            "human_identity": identity,
            "attested_at": "2026-08-04T12:30:00+08:00",
        },
        sort_keys=True,
    )


def _human_skip(job: dict[str, object], *, identity: str = "reviewer-7") -> str:
    return json.dumps(
        {
            "schema_version": "idea_factory.human_skip.v1",
            "job_id": job["job_id"],
            "human_job_sha256": job["human_job_sha256"],
            "reason_codes": ["NOT_MY_RESEARCH_PRIORITY"],
            "human_identity": identity,
            "attested_at": "2026-08-05T12:30:00+08:00",
        },
        sort_keys=True,
    )


def _task11_nonroute_run(tmp_path: Path, *, covered: bool):
    from idea_factory.artifacts import read_jsonl
    from idea_factory.recon import emit_recon_report_jobs, ingest_recon_reports
    from idea_factory.review import emit_review_jobs, ingest_review_results
    from idea_factory.routes import emit_route_jobs, ingest_route_results
    from test_recon import _build_dedup_run, _emit_normalized_chain, _report_result
    from test_routes_review import _zero_recon_run

    if covered:
        run, config = _build_dedup_run(tmp_path)
        _emit_normalized_chain(run, config, with_evidence=True)
        report_jobs_path = emit_recon_report_jobs(run, config)
        report_job = read_jsonl(report_jobs_path)[0]
        ingest_recon_reports(
            report_jobs_path, [_report_result(report_job, "COVERED", evidence=report_job["evidence"][0])], config,
        )
    else:
        run, config = _zero_recon_run(tmp_path)
    route_jobs_path = emit_route_jobs(run, config, allow_test_ready=True)
    assert read_jsonl(route_jobs_path) == []
    ingest_route_results(route_jobs_path, [], config, allow_test_ready=True)
    review_jobs_path = emit_review_jobs(run, config, allow_test_ready=True)
    assert read_jsonl(review_jobs_path) == []
    ingest_review_results(review_jobs_path, [], config, allow_test_ready=True)
    return run, config


def _finish_without_human_survivors(run: Path, config) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import emit_human_scoring_jobs, ingest_human_scores
    from idea_factory.render import render_idea_packs
    from idea_factory.review import emit_review_jobs, ingest_review_results
    from idea_factory.routes import emit_route_jobs, ingest_route_results

    route_jobs_path = emit_route_jobs(run, config, allow_test_ready=True)
    route_jobs = read_jsonl(route_jobs_path)
    if route_jobs:
        ingest_route_results(route_jobs_path, ["{}" for _ in route_jobs], config, allow_test_ready=True)
    else:
        ingest_route_results(route_jobs_path, [], config, allow_test_ready=True)
    review_jobs_path = emit_review_jobs(run, config, allow_test_ready=True)
    review_jobs = read_jsonl(review_jobs_path)
    ingest_review_results(
        review_jobs_path, ["{}" for _ in review_jobs], config, allow_test_ready=True,
    )
    human_jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    assert read_jsonl(human_jobs_path) == []
    ingest_human_scores(human_jobs_path, [], config, allow_test_ready=True)
    render_idea_packs(run, config, allow_test_ready=True)


def test_human_jobs_only_cover_terminal_pass_and_bind_exact_task10_provenance(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import emit_human_scoring_jobs
    from idea_factory.review import validate_review_bundle
    from test_routes_review import _canonical_hash

    run, config = _task11_reviewed_run(tmp_path)
    path = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    jobs = read_jsonl(path)
    review = validate_review_bundle(run, config, allow_test_ready=True)
    assert len(jobs) == 1
    job = jobs[0]
    source_job = next(row for row in review["jobs"] if row["route_id"] == job["route_id"])
    source_result = next(row for row in review["results"] if row["job_id"] == source_job["job_id"])
    assert job["opportunity_sha256"] == source_job["opportunity_sha256"]
    assert job["route_sha256"] == source_job["route_record_sha256"]
    assert job["recon_provenance_sha256"] == _canonical_hash(source_job["recon_provenance"])
    assert job["nearest_priors_sha256"] == _canonical_hash(source_job["route"]["nearest_prior_bindings"])
    assert job["review_record_sha256"] == _canonical_hash(source_result)
    assert job["authenticity_sha256"] == _canonical_hash(job["authenticity_binding"])
    assert job["result_schema_sha256"] == _canonical_hash(job["result_schema"])
    assert job["skip_result_schema_sha256"] == _canonical_hash(job["skip_result_schema"])
    assert job["skip_result_schema_version"] == "idea_factory.human_skip.v1"
    assert job["result_schema"]["additionalProperties"] is False
    assert job["result_schema"]["properties"]["reason_codes"]["items"]["enum"] == job["approved_reason_codes"]
    assert job["approved_reason_codes"] == [
        "WANT_TO_TEST_NOW",
        "INTERESTING_BUT_TOO_EXPENSIVE",
        "NOVEL_BUT_UNIMPORTANT",
        "IMPORTANT_BUT_ALREADY_COVERED",
        "MECHANISM_NOT_CONVINCING",
        "TOO_INCREMENTAL",
    ]
    assert job["approved_skip_reason_codes"] == ["NOT_MY_RESEARCH_PRIORITY"]
    assert job["accepted_job_immutable"] is True and job["rejected_job_retryable"] is True
    eligibility = read_jsonl(run / "ledger" / "human_scoring_eligibility.jsonl")
    assert {row["route_id"]: row["status"] for row in eligibility} == {
        review["jobs"][0]["route_id"]: "PENDING_HUMAN",
        review["jobs"][1]["route_id"]: "KILLED_REVIEW",
        review["jobs"][2]["route_id"]: "PENDING_REVIEW",
    }


def test_human_scores_are_strict_retryable_immutable_and_secret_free(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import emit_human_scoring_jobs, ingest_human_scores, validate_human_bundle

    run, config = _task11_reviewed_run(tmp_path, ("PASS_TO_HUMAN", "KILL", "KILL"))
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    job = read_jsonl(jobs_path)[0]
    with pytest.raises(ValueError, match="exactly cover"):
        ingest_human_scores(jobs_path, [], config, allow_test_ready=True)
    invalid = json.loads(_human_score(job)); invalid["specific_novelty"] = True
    ingest_human_scores(jobs_path, [json.dumps(invalid)], config, allow_test_ready=True)
    bundle = validate_human_bundle(run, config, allow_test_ready=True)
    assert bundle["results"] == () and bundle["outcomes"][0]["status"] == "REJECTED"
    obsolete = json.loads(_human_score(job)); obsolete["reason_codes"] = ["LOW_FEASIBILITY"]
    ingest_human_scores(jobs_path, [json.dumps(obsolete)], config, allow_test_ready=True)
    bundle = validate_human_bundle(run, config, allow_test_ready=True)
    assert bundle["results"] == ()
    assert bundle["outcomes"][0]["error_code"] == "HUMAN_SCORE_SCHEMA_INVALID"
    secret = _human_score(job, identity="password is HUMAN_SCORE_SENTINEL")
    ingest_human_scores(jobs_path, [secret], config, allow_test_ready=True)
    bundle = validate_human_bundle(run, config, allow_test_ready=True)
    assert bundle["outcomes"][0]["error_code"] == "HUMAN_SCORE_SECRET_DETECTED"
    assert "HUMAN_SCORE_SENTINEL" not in json.dumps(bundle)
    valid = _human_score(job)
    ingest_human_scores(jobs_path, [valid], config, allow_test_ready=True)
    bundle = validate_human_bundle(run, config, allow_test_ready=True)
    assert bundle["results"][0]["specific_novelty"] == 4
    ledger = run / "ledger"
    names = ("human_scores.jsonl", "human_scoring_outcomes.jsonl", "human_bundle_manifest.json")
    accepted_bytes = {name: (ledger / name).read_bytes() for name in names}
    ingest_human_scores(jobs_path, [valid], config, allow_test_ready=True)
    assert accepted_bytes == {name: (ledger / name).read_bytes() for name in names}
    changed = json.loads(valid); changed["specific_novelty"] = 5
    with pytest.raises(ValueError, match="immutable|replay"):
        ingest_human_scores(jobs_path, [json.dumps(changed)], config, allow_test_ready=True)
    assert accepted_bytes == {name: (ledger / name).read_bytes() for name in names}


def test_unselected_human_jobs_are_skipped_without_fabricated_scores(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import (
        build_expected_ledger_events, emit_human_scoring_jobs,
        ingest_human_scores, validate_human_bundle,
    )
    from idea_factory.render import render_idea_packs, validate_idea_pack_bundle

    run, config = _task11_reviewed_run(tmp_path, ("PASS_TO_HUMAN", "PASS_TO_HUMAN", "KILL"))
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    jobs = read_jsonl(jobs_path)
    assert len(jobs) == 2
    assert jobs[0]["skip_result_schema_version"] == "idea_factory.human_skip.v1"
    assert jobs[0]["skip_result_schema"]["properties"]["reason_codes"]["items"]["const"] == "NOT_MY_RESEARCH_PRIORITY"

    ingest_human_scores(
        jobs_path,
        [_human_score(jobs[0]), _human_skip(jobs[1])],
        config,
        allow_test_ready=True,
    )
    bundle = validate_human_bundle(run, config, allow_test_ready=True)
    assert len(bundle["results"]) == 1
    assert [row["status"] for row in bundle["outcomes"]] == ["ACCEPTED", "SKIPPED"]
    assert bundle["manifest"]["accepted_count"] == 1
    assert bundle["manifest"]["skipped_count"] == 1
    assert bundle["manifest"]["rejected_count"] == 0
    assert all(row["job_id"] != jobs[1]["job_id"] for row in bundle["results"])

    render_idea_packs(run, config, allow_test_ready=True)
    manifest = validate_idea_pack_bundle(run, config, allow_test_ready=True)
    assert manifest["pack_count"] == 1
    events = build_expected_ledger_events(run, config, allow_test_ready=True)
    skipped = next(
        row for row in events
        if row["stage"] == "HUMAN_SCORING" and row["status"] == "SKIPPED"
    )
    assert skipped["reason_codes"] == ["NOT_MY_RESEARCH_PRIORITY"]
    assert skipped["human_scores"] is None


def test_human_skip_rejects_nonselection_reason_spoofing(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import emit_human_scoring_jobs, ingest_human_scores, validate_human_bundle

    run, config = _task11_reviewed_run(tmp_path, ("PASS_TO_HUMAN", "KILL", "KILL"))
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    job = read_jsonl(jobs_path)[0]
    invalid = json.loads(_human_skip(job))
    invalid["reason_codes"] = ["WANT_TO_TEST_NOW"]
    ingest_human_scores(jobs_path, [json.dumps(invalid)], config, allow_test_ready=True)
    bundle = validate_human_bundle(run, config, allow_test_ready=True)
    assert bundle["results"] == ()
    assert bundle["outcomes"][0]["status"] == "REJECTED"
    assert bundle["outcomes"][0]["error_code"] == "HUMAN_SCORE_SCHEMA_INVALID"


def test_scored_human_result_cannot_use_the_skip_only_reason(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import emit_human_scoring_jobs, ingest_human_scores, validate_human_bundle

    run, config = _task11_reviewed_run(tmp_path, ("PASS_TO_HUMAN", "KILL", "KILL"))
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    job = read_jsonl(jobs_path)[0]
    invalid = json.loads(_human_score(job))
    invalid["reason_codes"] = ["NOT_MY_RESEARCH_PRIORITY"]
    ingest_human_scores(jobs_path, [json.dumps(invalid)], config, allow_test_ready=True)
    bundle = validate_human_bundle(run, config, allow_test_ready=True)
    assert bundle["results"] == ()
    assert bundle["outcomes"][0]["status"] == "REJECTED"
    assert bundle["outcomes"][0]["error_code"] == "HUMAN_SCORE_SCHEMA_INVALID"


def test_want_to_test_now_is_mutually_exclusive_in_schema_and_runtime(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import emit_human_scoring_jobs, ingest_human_scores, validate_human_bundle

    run, config = _task11_reviewed_run(tmp_path, ("PASS_TO_HUMAN", "KILL", "KILL"))
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    job = read_jsonl(jobs_path)[0]
    schema = job["result_schema"]
    assert schema["allOf"] == [{
        "if": {
            "properties": {
                "reason_codes": {"contains": {"const": "WANT_TO_TEST_NOW"}},
            },
            "required": ["reason_codes"],
        },
        "then": {"properties": {"reason_codes": {"maxItems": 1}}},
    }]
    conflict = json.loads(_human_score(job))
    conflict["reason_codes"] = ["WANT_TO_TEST_NOW", "TOO_INCREMENTAL"]
    ingest_human_scores(jobs_path, [json.dumps(conflict)], config, allow_test_ready=True)
    bundle = validate_human_bundle(run, config, allow_test_ready=True)
    assert bundle["results"] == ()
    assert bundle["outcomes"][0]["status"] == "REJECTED"
    assert bundle["outcomes"][0]["error_code"] == "HUMAN_REASON_CONFLICT"


def test_ready_idea_pack_renders_per_id_json_and_markdown_with_provenance(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import emit_human_scoring_jobs, ingest_human_scores
    from idea_factory.render import render_idea_packs, validate_idea_pack_bundle

    run, config = _task11_reviewed_run(
        tmp_path, ("PASS_TO_HUMAN", "KILL", "KILL"), near_prior=True
    )
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    job = read_jsonl(jobs_path)[0]
    ingest_human_scores(jobs_path, [_human_score(job)], config, allow_test_ready=True)
    pack_dir = render_idea_packs(run, config, allow_test_ready=True)
    manifest = validate_idea_pack_bundle(run, config, allow_test_ready=True)
    assert pack_dir == run / "idea_packs"
    assert manifest["pack_count"] == 1 and manifest["ready_count"] == 1
    idea_id = manifest["idea_ids"][0]
    json_path, md_path = pack_dir / f"{idea_id}.json", pack_dir / f"{idea_id}.md"
    pack = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")
    required = {
        "opportunity", "core_hypothesis", "why_now", "evidence_flags", "supporting_observations",
        "nearest_priors", "nearest_prior_exact_difference", "proposed_mechanism", "source_of_gain",
        "cheapest_decisive_test", "strongest_baseline", "kill_condition", "main_uncertainty",
        "expected_reviewer_2_objection", "response_to_objection",
        "evidence_that_would_make_reviewer_correct", "human_scores", "human_reason_codes",
        "authenticity_boundary", "provenance",
    }
    assert required <= set(pack) and pack["status"] == "READY_FOR_CHEAP_TEST"
    assert pack["nearest_priors"][0]["residual_difference"]
    assert pack["idea_pack_sha256"] in markdown and idea_id in markdown
    assert manifest["files"][idea_id]["json_sha256"] == hashlib.sha256(json_path.read_bytes()).hexdigest()
    assert manifest["files"][idea_id]["markdown_sha256"] == hashlib.sha256(md_path.read_bytes()).hexdigest()


def test_empty_inference_flags_are_an_explicit_readiness_failure(tmp_path: Path) -> None:
    from copy import deepcopy
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import emit_human_scoring_jobs, ingest_human_scores, validate_human_bundle
    from idea_factory.render import _idea_pack

    run, config = _task11_reviewed_run(
        tmp_path, ("PASS_TO_HUMAN", "KILL", "KILL"), near_prior=True
    )
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    job = read_jsonl(jobs_path)[0]
    ingest_human_scores(jobs_path, [_human_score(job)], config, allow_test_ready=True)
    score = validate_human_bundle(run, config, allow_test_ready=True)["results"][0]
    without_flags = deepcopy(job)
    without_flags["opportunity"]["inference_flags"] = []
    pack = _idea_pack(without_flags, score)
    assert pack["status"] == "HOLD"
    assert pack["readiness_gates"]["evidence_or_inference_flags"] is False


def test_zero_survivor_has_audit_report_but_no_fake_pack(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import emit_human_scoring_jobs, ingest_human_scores
    from idea_factory.render import render_idea_packs, validate_idea_pack_bundle

    run, config = _task11_reviewed_run(tmp_path, ("KILL", "KILL", "KILL"))
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    assert read_jsonl(jobs_path) == []
    ingest_human_scores(jobs_path, [], config, allow_test_ready=True)
    pack_dir = render_idea_packs(run, config, allow_test_ready=True)
    manifest = validate_idea_pack_bundle(run, config, allow_test_ready=True)
    report = json.loads((pack_dir / "zero_survivor_report.json").read_text(encoding="utf-8"))
    assert manifest["pack_count"] == 0 and report["classification"] == "ZERO_SURVIVOR"
    assert report["operational_completion_is_idea_yield"] is False
    assert not [path for path in pack_dir.glob("*.json") if path.name not in {"manifest.json", "zero_survivor_report.json"}]


def test_append_ledger_updates_covers_opportunity_and_routes_idempotently(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl, write_jsonl
    from idea_factory.ledger import (
        append_ledger_updates, build_expected_ledger_events,
        emit_human_scoring_jobs, ingest_human_scores,
        validate_ledger_source_locator, validate_ledger_updates,
    )
    from idea_factory.render import render_idea_packs

    run, config = _task11_reviewed_run(
        tmp_path, ("PASS_TO_HUMAN", "KILL", "KILL"), near_prior=True
    )
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    job = read_jsonl(jobs_path)[0]
    ingest_human_scores(jobs_path, [_human_score(job)], config, allow_test_ready=True)
    render_idea_packs(run, config, allow_test_ready=True)
    repository = tmp_path / "repository"; repository.mkdir()
    legacy_before = config.legacy_ledger.read_bytes()
    paths = append_ledger_updates(run, config, repository_root=repository, allow_test_ready=True)
    assert paths == {
        "run_updates": run / "ledger" / "updates.jsonl",
        "persistent_ledger": repository / "data" / "idea_ledger.jsonl",
    }
    updates = read_jsonl(paths["run_updates"])
    persistent = read_jsonl(paths["persistent_ledger"])
    assert updates == persistent
    assert updates == build_expected_ledger_events(run, config, allow_test_ready=True)
    assert len(updates) == 9
    snapshot = validate_ledger_updates(
        run, config, repository_root=repository, allow_test_ready=True,
    )
    assert snapshot["run_updates"] == tuple(updates)
    assert snapshot["persistent_updates"] == tuple(persistent)
    assert {row["ledger_transaction_id"] for row in updates} == {snapshot["ledger_transaction_id"]}
    required_event_fields = {
        "run_id", "source_run_id", "ledger_transaction_id", "source_locator",
        "opportunity_fingerprint", "mechanism_fingerprint", "mechanism_fingerprint_status",
        "supporting_card_ids", "supporting_paper_ids", "human_scores",
        "authenticity", "authenticity_boundary", "attestation_scope",
        "cheap_test_done", "subsequent_result",
    }
    assert all(required_event_fields <= set(row) for row in updates)
    assert all(row["run_id"] == row["source_run_id"] for row in updates)
    assert all(row["cheap_test_done"] is False for row in updates)
    assert all(row["subsequent_result"] is None for row in updates)
    assert all(
        row["supporting_card_ids"] and row["supporting_paper_ids"]
        and len(row["supporting_card_ids"]) == len(row["supporting_paper_ids"])
        for row in updates
    )
    for event in updates:
        source = validate_ledger_source_locator(event)
        assert event["source_record_sha256"] == event["source_locator"]["source_record_sha256"]
        assert event["source_record_sha256"] == hashlib.sha256(
            json.dumps(source, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    human_events = [row for row in updates if row["stage"] == "HUMAN_SCORING"]
    pack_events = [row for row in updates if row["stage"] == "IDEA_PACK"]
    assert all(set(row["human_scores"]) == {
        "specific_novelty", "importance", "paper_potential", "feasibility", "excitement", "evidence_clarity",
    } for row in human_events + pack_events)
    assert all(row["human_scores"] is None for row in updates if row["stage"] not in {"HUMAN_SCORING", "IDEA_PACK"})
    review_routes = [row for row in updates if row["stage"] == "REVIEW"]
    assert all(row["mechanism_fingerprint_status"] == "AVAILABLE" and row["mechanism_fingerprint"] for row in review_routes)
    assert sum(row["entity_type"] == "OPPORTUNITY" for row in updates) >= 1
    recon_events = [row for row in updates if row["stage"] == "RECON"]
    assert len(recon_events) == 1
    assert recon_events[0]["status"] == "NEAR_PRIOR_WITH_RESIDUAL"
    assert "NEAR_PRIOR_WITH_RESIDUAL" in recon_events[0]["reason_codes"]
    route_outcomes = [row for row in updates if row["stage"] == "ROUTES"]
    assert len(route_outcomes) == 3
    assert {row["status"] for row in route_outcomes} == {"ACCEPTED"}
    assert {row["mechanism_fingerprint_status"] for row in route_outcomes} == {"AVAILABLE"}
    assert len({row["mechanism_fingerprint"] for row in route_outcomes}) == 3
    assert all(row["mechanism_fingerprint"] for row in route_outcomes)
    assert all(row["source_locator"]["artifact_relpath"] == "routes/results.jsonl" for row in route_outcomes)
    assert all(row["source_locator"]["source_subrecord_id"] is None for row in route_outcomes)
    review_events = [row for row in updates if row["stage"] == "REVIEW"]
    assert {row["status"] for row in review_events} == {"PASS_TO_HUMAN", "KILL"}
    assert all(row["source_locator"]["artifact_relpath"] == "review/results.jsonl" for row in review_events)
    assert human_events[0]["source_locator"]["artifact_relpath"] == "ledger/human_scores.jsonl"
    assert pack_events[0]["source_locator"]["artifact_relpath"].startswith("idea_packs/")
    assert pack_events[0]["status"] == "READY_FOR_CHEAP_TEST"
    assert pack_events[0]["cheap_test_done"] is False
    assert pack_events[0]["subsequent_result"] is None
    assert len({row["ledger_update_id"] for row in updates}) == len(updates)
    assert all(len(row["source_record_sha256"]) == 64 for row in updates)
    assert all(row["reason_codes"] and "stage" in row and "nearest_priors" in row for row in updates)
    before = {key: path.read_bytes() for key, path in paths.items()}
    append_ledger_updates(run, config, repository_root=repository, allow_test_ready=True)
    assert before == {key: path.read_bytes() for key, path in paths.items()}
    assert config.legacy_ledger.read_bytes() == legacy_before
    corrupted = persistent; corrupted[0]["status"] = "CORRUPTED"
    write_jsonl(paths["persistent_ledger"], corrupted)
    corrupted_bytes = paths["persistent_ledger"].read_bytes()
    run_bytes = paths["run_updates"].read_bytes()
    with pytest.raises(ValueError, match="existing|ledger|hash|conflict"):
        append_ledger_updates(run, config, repository_root=repository, allow_test_ready=True)
    assert paths["persistent_ledger"].read_bytes() == corrupted_bytes
    assert paths["run_updates"].read_bytes() == run_bytes


def test_locked_ledger_reader_never_observes_split_brain_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    import idea_factory.stage_io as stage_io
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import (
        append_ledger_updates, emit_human_scoring_jobs, ingest_human_scores,
        validate_ledger_updates,
    )
    from idea_factory.render import render_idea_packs

    run, config = _task11_reviewed_run(
        tmp_path, ("PASS_TO_HUMAN", "KILL", "KILL"), near_prior=True,
    )
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    job = read_jsonl(jobs_path)[0]
    ingest_human_scores(jobs_path, [_human_score(job)], config, allow_test_ready=True)
    render_idea_packs(run, config, allow_test_ready=True)
    repository = tmp_path / "repository"; repository.mkdir()
    first_replace_done = threading.Event()
    allow_second_replace = threading.Event()
    real_replace = stage_io.os.replace

    def pause_between_ledgers(source, destination):
        result = real_replace(source, destination)
        if Path(destination) == run / "ledger" / "updates.jsonl":
            first_replace_done.set()
            assert allow_second_replace.wait(10)
        return result

    monkeypatch.setattr(stage_io.os, "replace", pause_between_ledgers)
    failure: list[BaseException] = []

    def publish() -> None:
        try:
            append_ledger_updates(run, config, repository_root=repository, allow_test_ready=True)
        except BaseException as exc:
            failure.append(exc)

    writer = threading.Thread(target=publish)
    writer.start()
    assert first_replace_done.wait(30)
    with pytest.raises(RuntimeError, match="lock|busy"):
        validate_ledger_updates(
            run, config, repository_root=repository, allow_test_ready=True,
        )
    allow_second_replace.set()
    writer.join(30)
    assert not writer.is_alive() and failure == []
    snapshot = validate_ledger_updates(
        run, config, repository_root=repository, allow_test_ready=True,
    )
    transaction_id = snapshot["ledger_transaction_id"]
    assert {row["ledger_transaction_id"] for row in snapshot["run_updates"]} == {transaction_id}
    persistent_current = [
        row for row in snapshot["persistent_updates"] if row["ledger_transaction_id"] == transaction_id
    ]
    assert persistent_current == list(snapshot["run_updates"])
    assert "raw jsonl reads are not snapshot-safe" in (validate_ledger_updates.__doc__ or "").lower()


def test_ledger_validator_replays_pipeline_and_rejects_jointly_rehashed_tampering(
    tmp_path: Path,
) -> None:
    from copy import deepcopy
    from idea_factory.artifacts import read_jsonl, write_jsonl
    from idea_factory.ledger import (
        append_ledger_updates, emit_human_scoring_jobs, ingest_human_scores,
        validate_ledger_updates,
    )
    from idea_factory.render import render_idea_packs

    run, config = _task11_reviewed_run(
        tmp_path, ("PASS_TO_HUMAN", "KILL", "KILL"), near_prior=True,
    )
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    job = read_jsonl(jobs_path)[0]
    ingest_human_scores(jobs_path, [_human_score(job)], config, allow_test_ready=True)
    render_idea_packs(run, config, allow_test_ready=True)
    repository = tmp_path / "repository"; repository.mkdir()
    paths = append_ledger_updates(
        run, config, repository_root=repository, allow_test_ready=True,
    )
    original = read_jsonl(paths["run_updates"])

    def rehash(row: dict[str, object]) -> None:
        body = {key: value for key, value in row.items() if key != "update_sha256"}
        row["update_sha256"] = hashlib.sha256(json.dumps(
            body, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    mutations = (
        ("HUMAN_SCORING", lambda row: row["human_scores"].__setitem__("importance", 0)),
        ("ROUTES", lambda row: row.__setitem__("mechanism_fingerprint", "0" * 64)),
        ("ROUTES", lambda row: row.__setitem__("supporting_card_ids", ["card-tampered"])),
        ("ROUTES", lambda row: row.__setitem__("authenticity", "TAMPERED_BUT_NONEMPTY")),
        ("IDEA_PACK", lambda row: row.__setitem__("cheap_test_done", True)),
        ("IDEA_PACK", lambda row: row.__setitem__("subsequent_result", {"result": "FABRICATED"})),
    )
    for stage, mutate in mutations:
        tampered = deepcopy(original)
        target = next(row for row in tampered if row["stage"] == stage)
        mutate(target)
        rehash(target)
        write_jsonl(paths["run_updates"], tampered)
        write_jsonl(paths["persistent_ledger"], tampered)
        with pytest.raises(ValueError, match="expected|replay|derived"):
            validate_ledger_updates(
                run, config, repository_root=repository, allow_test_ready=True,
            )

    write_jsonl(paths["run_updates"], original)
    write_jsonl(paths["persistent_ledger"], original)
    snapshot = validate_ledger_updates(
        run, config, repository_root=repository, allow_test_ready=True,
    )
    assert list(snapshot["run_updates"]) == original


def test_ledger_preserves_rejected_route_review_and_human_stage_events(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import append_ledger_updates, emit_human_scoring_jobs, ingest_human_scores
    from idea_factory.render import render_idea_packs
    from idea_factory.review import emit_review_jobs, ingest_review_results
    from idea_factory.routes import emit_route_jobs, ingest_route_results
    from test_routes_review import _ready_run, _route_result

    # Route rejection is a terminal route-generation event, not PENDING/ROUTES.
    route_run, route_config = _ready_run(tmp_path / "route")
    route_jobs_path = emit_route_jobs(route_run, route_config, allow_test_ready=True)
    ingest_route_results(route_jobs_path, ["{}"], route_config, allow_test_ready=True)
    review_jobs_path = emit_review_jobs(route_run, route_config, allow_test_ready=True)
    ingest_review_results(review_jobs_path, [], route_config, allow_test_ready=True)
    human_jobs_path = emit_human_scoring_jobs(route_run, route_config, allow_test_ready=True)
    ingest_human_scores(human_jobs_path, [], route_config, allow_test_ready=True)
    render_idea_packs(route_run, route_config, allow_test_ready=True)
    route_repository = tmp_path / "route-repository"; route_repository.mkdir()
    route_updates = read_jsonl(append_ledger_updates(
        route_run, route_config, repository_root=route_repository, allow_test_ready=True,
    )["run_updates"])
    route_event = next(row for row in route_updates if row["stage"] == "ROUTES")
    assert route_event["status"] == "REJECTED"
    assert route_event["reason_codes"][0] == "ROUTE_SCHEMA_INVALID"
    assert route_event["provenance"]["source_artifact"] == "routes/outcomes.jsonl"

    # Rejected reviewer output remains a REVIEW event tied to each route.
    review_run, review_config = _ready_run(tmp_path / "review")
    route_jobs_path = emit_route_jobs(review_run, review_config, allow_test_ready=True)
    route_job = read_jsonl(route_jobs_path)[0]
    ingest_route_results(
        route_jobs_path, [_route_result(route_job)], review_config, allow_test_ready=True,
    )
    review_jobs_path = emit_review_jobs(review_run, review_config, allow_test_ready=True)
    review_jobs = read_jsonl(review_jobs_path)
    ingest_review_results(
        review_jobs_path, ["{}" for _ in review_jobs], review_config, allow_test_ready=True,
    )
    human_jobs_path = emit_human_scoring_jobs(review_run, review_config, allow_test_ready=True)
    ingest_human_scores(human_jobs_path, [], review_config, allow_test_ready=True)
    render_idea_packs(review_run, review_config, allow_test_ready=True)
    review_repository = tmp_path / "review-repository"; review_repository.mkdir()
    review_updates = read_jsonl(append_ledger_updates(
        review_run, review_config, repository_root=review_repository, allow_test_ready=True,
    )["run_updates"])
    review_events = [row for row in review_updates if row["stage"] == "REVIEW"]
    assert len(review_events) == 3
    assert {row["status"] for row in review_events} == {"REJECTED"}
    source_review_errors = {
        row["error_code"] for row in read_jsonl(review_run / "review" / "outcomes.jsonl")
    }
    assert {row["reason_codes"][0] for row in review_events} == source_review_errors

    # Human schema rejection is retained even though no Idea Pack is rendered.
    human_run, human_config = _task11_reviewed_run(
        tmp_path / "human", ("PASS_TO_HUMAN", "KILL", "KILL"), near_prior=True,
    )
    human_jobs_path = emit_human_scoring_jobs(human_run, human_config, allow_test_ready=True)
    human_job = read_jsonl(human_jobs_path)[0]
    rejected_score = json.loads(_human_score(human_job))
    rejected_score["reason_codes"] = ["LOW_FEASIBILITY"]
    ingest_human_scores(
        human_jobs_path, [json.dumps(rejected_score)], human_config, allow_test_ready=True,
    )
    render_idea_packs(human_run, human_config, allow_test_ready=True)
    human_repository = tmp_path / "human-repository"; human_repository.mkdir()
    human_updates = read_jsonl(append_ledger_updates(
        human_run, human_config, repository_root=human_repository, allow_test_ready=True,
    )["run_updates"])
    human_event = next(row for row in human_updates if row["stage"] == "HUMAN_SCORING")
    assert human_event["status"] == "REJECTED"
    assert human_event["reason_codes"][0] == "HUMAN_SCORE_SCHEMA_INVALID"


def test_ledger_preserves_quality_dedup_and_recon_negative_source_rows(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.corpus import CorpusRouterConfig
    from idea_factory.legacy_ledger import import_legacy_ledger
    from idea_factory.ledger import (
        append_ledger_updates, emit_shape_assignment_jobs, ingest_shape_assignments,
        publish_internal_dedup,
    )
    from idea_factory.opportunities import emit_mining_jobs, ingest_opportunity_results
    from idea_factory.quality import publish_quality
    from idea_factory.recon import (
        emit_recon_report_jobs, ingest_recon_reports, normalize_recon_raw,
    )
    from test_opportunities import _build_run, _empty_result
    from test_recon import _build_dedup_run, _emit_empty_execution_chain, _emit_normalized_chain

    # A schema-valid candidate that fails the deterministic quality gate keeps
    # the gate's original reason codes even though it never reaches dedup.
    quality_root = tmp_path / "quality"
    quality_run, prompt = _build_run(quality_root)
    mining_jobs = read_jsonl(emit_mining_jobs(quality_run, prompt))
    mining_results = [_empty_result(job) for job in mining_jobs]
    first = mining_jobs[0]
    wrapper = json.loads(mining_results[0])
    wrapper["opportunities"] = [{
        "schema_version": "idea_factory.opportunity.v1",
        "opportunity_id": "opp-quality-rejected", "operator": first["operator"],
        "assumption_x": "fixed", "observation_y": "retrieval degrades after eviction",
        "condition_z": "under long serving traces", "failure_f": "eviction causes retrieval loss",
        "missing_capability_w": "eviction aware retrieval control",
        "alternative_explanation_a": "cache eviction",
        "decisive_experiment": "Compare recall after forced cache eviction against cache eviction.",
        "supporting_card_ids": [first["card_ids"][0]],
        "nearest_internal_neighbors": [first["neighbor_ids"][0]],
        "scope_compatibility": "Both cards cover KV cache serving over long traces.",
        "inference_flags": ["failure mechanism inferred"],
    }]
    mining_results[0] = json.dumps(wrapper)
    ingest_opportunity_results(
        quality_run / "opportunities" / "mining_jobs.jsonl", mining_results,
    )
    quality_paths = publish_quality(quality_run)
    quality_rejected = read_jsonl(quality_paths["rejected"])
    assert quality_rejected[0]["reason_codes"] == ["VAGUE_ASSUMPTION"]
    papers = quality_root / "papers"
    legacy = papers / "_ideas" / "idea_ledger.md"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
        "| id | one-line | mechanism family | target | status / venue | source |\n"
        "|---|---|---|---|---|---|\n"
        "| L | other | OTHER | elsewhere | old | source |\n",
        encoding="utf-8",
    )
    quality_config = CorpusRouterConfig(
        quality_root / "notes", (papers / "list.txt",), legacy, 1, 1, ("KV_CACHE",), (),
    )
    import_legacy_ledger(quality_run, quality_config)
    shape_jobs_path = emit_shape_assignment_jobs(quality_run, quality_config)
    assert read_jsonl(shape_jobs_path) == []
    ingest_shape_assignments(shape_jobs_path, [])
    publish_internal_dedup(quality_run, quality_config)
    _emit_empty_execution_chain(quality_run, quality_config)
    normalize_recon_raw(quality_run, quality_config)
    report_jobs_path = emit_recon_report_jobs(quality_run, quality_config)
    ingest_recon_reports(report_jobs_path, [], quality_config)
    _finish_without_human_survivors(quality_run, quality_config)
    quality_repository = tmp_path / "quality-repository"; quality_repository.mkdir()
    quality_updates = read_jsonl(append_ledger_updates(
        quality_run, quality_config, repository_root=quality_repository, allow_test_ready=True,
    )["run_updates"])
    quality_event = next(row for row in quality_updates if row["stage"] == "QUALITY")
    assert quality_event["status"] == "QUALITY_REJECTED"
    assert quality_event["reason_codes"] == ["VAGUE_ASSUMPTION"]
    assert quality_event["source_record_sha256"] == quality_event["provenance"]["source_row_sha256"]

    # A blocked internal duplicate keeps its exact dedup status and reason.
    dedup_run, dedup_config = _build_dedup_run(
        tmp_path / "dedup",
        legacy_row="| L | other | EVICTION_CONTROL | KV cache serving | old | source |",
    )
    _emit_empty_execution_chain(dedup_run, dedup_config)
    normalize_recon_raw(dedup_run, dedup_config)
    report_jobs_path = emit_recon_report_jobs(dedup_run, dedup_config)
    ingest_recon_reports(report_jobs_path, [], dedup_config)
    _finish_without_human_survivors(dedup_run, dedup_config)
    dedup_repository = tmp_path / "dedup-repository"; dedup_repository.mkdir()
    dedup_updates = read_jsonl(append_ledger_updates(
        dedup_run, dedup_config, repository_root=dedup_repository, allow_test_ready=True,
    )["run_updates"])
    dedup_event = next(row for row in dedup_updates if row["stage"] == "INTERNAL_DEDUP")
    assert dedup_event["status"] == "INTERNAL_DUP"
    assert dedup_event["reason_codes"] == [
        "INTERNAL_DUP", "same explicit mechanism family and target overlap",
    ]
    assert dedup_event["provenance"]["source_artifact"] == "ledger/dedup_outcomes.jsonl"

    # Malformed recon output remains a RECON rejection with its bounded error.
    recon_run, recon_config = _build_dedup_run(tmp_path / "recon")
    _emit_normalized_chain(recon_run, recon_config, with_evidence=False)
    report_jobs_path = emit_recon_report_jobs(recon_run, recon_config)
    report_job = read_jsonl(report_jobs_path)[0]
    malformed = json.dumps({"job_id": report_job["job_id"]})
    ingest_recon_reports(report_jobs_path, [malformed], recon_config)
    _finish_without_human_survivors(recon_run, recon_config)
    recon_repository = tmp_path / "recon-repository"; recon_repository.mkdir()
    recon_updates = read_jsonl(append_ledger_updates(
        recon_run, recon_config, repository_root=recon_repository, allow_test_ready=True,
    )["run_updates"])
    recon_event = next(row for row in recon_updates if row["stage"] == "RECON")
    assert recon_event["status"] == "REJECTED"
    assert recon_event["reason_codes"][0] == "REPORT_SCHEMA_INVALID"
    assert recon_event["provenance"]["source_artifact"] == "recon/rejected.jsonl"
    assert all(
        not (row["opportunity_id"] == recon_event["opportunity_id"] and row["status"] == "PENDING")
        for row in recon_updates
    )


def test_append_cross_directory_publish_failure_rolls_back_both_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import idea_factory.stage_io as stage_io
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import append_ledger_updates, emit_human_scoring_jobs, ingest_human_scores
    from idea_factory.render import render_idea_packs

    run, config = _task11_reviewed_run(tmp_path, ("PASS_TO_HUMAN", "KILL", "KILL"), near_prior=True)
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True); job = read_jsonl(jobs_path)[0]
    ingest_human_scores(jobs_path, [_human_score(job)], config, allow_test_ready=True)
    render_idea_packs(run, config, allow_test_ready=True)
    repository = tmp_path / "repository"; repository.mkdir()
    paths = append_ledger_updates(run, config, repository_root=repository, allow_test_ready=True)
    before = {key: path.read_bytes() for key, path in paths.items()}
    real_replace = stage_io.os.replace; calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected cross-ledger replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(stage_io.os, "replace", fail_second)
    with pytest.raises(OSError, match="injected cross-ledger"):
        append_ledger_updates(run, config, repository_root=repository, allow_test_ready=True)
    assert before == {key: path.read_bytes() for key, path in paths.items()}


@pytest.mark.parametrize(
    ("covered", "expected"),
    ((False, "ZERO_INPUT"), (True, "ZERO_SURVIVOR")),
)
def test_no_route_report_distinguishes_zero_input_from_upstream_survivor_loss(
    tmp_path: Path, covered: bool, expected: str,
) -> None:
    from idea_factory.ledger import emit_human_scoring_jobs, ingest_human_scores
    from idea_factory.render import render_idea_packs, validate_idea_pack_bundle

    run, config = _task11_nonroute_run(tmp_path, covered=covered)
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    ingest_human_scores(jobs_path, [], config, allow_test_ready=True)
    directory = render_idea_packs(run, config, allow_test_ready=True)
    manifest = validate_idea_pack_bundle(run, config, allow_test_ready=True)
    report = json.loads((directory / "zero_survivor_report.json").read_text(encoding="utf-8"))
    assert manifest["classification"] == report["classification"] == expected
    assert report["source_classification"] == expected
    assert report["opportunity_input_count"] == (1 if covered else 0)
    assert report["operational_completion_is_idea_yield"] is False
    if covered:
        from idea_factory.artifacts import read_jsonl
        from idea_factory.ledger import append_ledger_updates

        repository = tmp_path / "repository"; repository.mkdir()
        paths = append_ledger_updates(run, config, repository_root=repository, allow_test_ready=True)
        updates = read_jsonl(paths["run_updates"])
        covered_events = [row for row in updates if row["stage"] == "RECON"]
        assert len(covered_events) == 1
        assert covered_events[0]["entity_type"] == "OPPORTUNITY"
        assert covered_events[0]["status"] == "COVERED"
        assert "COVERED" in covered_events[0]["reason_codes"]


def test_want_to_test_now_without_nearest_prior_difference_remains_hold(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import emit_human_scoring_jobs, ingest_human_scores
    from idea_factory.render import render_idea_packs, validate_idea_pack_bundle

    run, config = _task11_reviewed_run(tmp_path, ("PASS_TO_HUMAN", "KILL", "KILL"))
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True); job = read_jsonl(jobs_path)[0]
    ingest_human_scores(jobs_path, [_human_score(job)], config, allow_test_ready=True)
    render_idea_packs(run, config, allow_test_ready=True)
    manifest = validate_idea_pack_bundle(run, config, allow_test_ready=True)
    pack = json.loads((run / "idea_packs" / f"{manifest['idea_ids'][0]}.json").read_text(encoding="utf-8"))
    assert pack["status"] == "HOLD"
    assert pack["readiness_gates"]["want_to_test_now"] is True
    assert pack["readiness_gates"]["nearest_prior_exact_difference"] is False


@pytest.mark.parametrize("mode", ("PENDING", "UNRESOLVED"))
def test_zero_pack_report_preserves_pending_and_unresolved_sources(tmp_path: Path, mode: str) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import emit_human_scoring_jobs, ingest_human_scores
    from idea_factory.render import render_idea_packs

    decisions = ("NARROW", "KILL", "KILL") if mode == "PENDING" else ("PASS_TO_HUMAN", "KILL", "KILL")
    run, config = _task11_reviewed_run(tmp_path, decisions)
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True)
    jobs = read_jsonl(jobs_path)
    if mode == "PENDING":
        ingest_human_scores(jobs_path, [], config, allow_test_ready=True)
    else:
        invalid = json.loads(_human_score(jobs[0])); invalid["importance"] = False
        ingest_human_scores(jobs_path, [json.dumps(invalid)], config, allow_test_ready=True)
    directory = render_idea_packs(run, config, allow_test_ready=True)
    report = json.loads((directory / "zero_survivor_report.json").read_text(encoding="utf-8"))
    assert report["classification"] == mode
    assert report["pack_count"] == 0


def test_task11_mutators_preserve_unknown_lock_and_reject_persistent_junction(tmp_path: Path) -> None:
    import os
    from idea_factory.artifacts import read_jsonl
    from idea_factory.ledger import append_ledger_updates, emit_human_scoring_jobs, ingest_human_scores
    from idea_factory.render import render_idea_packs

    run, config = _task11_reviewed_run(tmp_path, ("PASS_TO_HUMAN", "KILL", "KILL"), near_prior=True)
    lock = run / "ledger" / ".stage.lock"; lock.write_text("unknown-owner", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lock|busy"):
        emit_human_scoring_jobs(run, config, allow_test_ready=True)
    assert lock.read_text(encoding="utf-8") == "unknown-owner"
    lock.unlink()
    jobs_path = emit_human_scoring_jobs(run, config, allow_test_ready=True); job = read_jsonl(jobs_path)[0]
    ingest_human_scores(jobs_path, [_human_score(job)], config, allow_test_ready=True)
    render_idea_packs(run, config, allow_test_ready=True)
    repository = tmp_path / "repository"; repository.mkdir()
    outside = tmp_path / "outside"; outside.mkdir(); sentinel = outside / "sentinel"; sentinel.write_text("keep", encoding="utf-8")
    try:
        os.symlink(outside, repository / "data", target_is_directory=True)
    except OSError:
        import subprocess

        completed = subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(repository / "data"), str(outside)], capture_output=True,
        )
        if completed.returncode:
            pytest.skip("directory links unavailable")
    with pytest.raises(ValueError, match="data|escape|exact"):
        append_ledger_updates(run, config, repository_root=repository, allow_test_ready=True)
    assert sentinel.read_text(encoding="utf-8") == "keep"
