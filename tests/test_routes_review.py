from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from idea_factory.artifacts import read_jsonl, write_jsonl
from test_recon import _build_dedup_run, _emit_empty_execution_chain, _emit_normalized_chain, _report_result


ROOT = Path(__file__).parents[1]
ROUTE_PROMPT = ROOT / "prompts" / "route_generator.md"
REVIEW_PROMPT = ROOT / "prompts" / "reviewer.md"

MECHANISM_SPECS = [
    {
        "schema_version": "idea_factory.mechanism_spec.v2",
        "intervention_site": "observation boundary intervention",
        "operation": "mask observations under the bound condition",
        "target_state": "observed state representation",
        "learning_signal": "counterfactual response delta",
        "state_representation": "versioned state with evidence links",
    },
    {
        "schema_version": "idea_factory.mechanism_spec.v2",
        "intervention_site": "state transition intervention",
        "operation": "gate transition on evidence",
        "target_state": "latent state representation",
        "learning_signal": "held out regret signal",
        "state_representation": "evidence vector with provenance",
    },
    {
        "schema_version": "idea_factory.mechanism_spec.v2",
        "intervention_site": "output boundary intervention",
        "operation": "transform dependency output",
        "target_state": "dependency state representation",
        "learning_signal": "recovery delta signal",
        "state_representation": "dependency graph with evidence",
    },
]
GAIN_HYPOTHESES = (
    "masking the observation boundary isolates a counterfactual response delta",
    "gating the state transition isolates held out regret",
    "transforming output dependencies isolates recovery delta",
)


def _render_mechanism(spec: dict[str, str]) -> str:
    return "mechanism_spec:" + ":".join(
        spec[key]
        for key in ("intervention_site", "operation", "target_state", "learning_signal", "state_representation")
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _ready_run(tmp_path: Path, *, decision: str = "NO_DIRECT_COVERAGE_FOUND"):
    from idea_factory.recon import emit_recon_report_jobs, ingest_recon_reports

    run, config = _build_dedup_run(tmp_path)
    # Recon route-readiness requires evidenced relevance in every lane.  The
    # shared fixture supplies one canonical paper across all query receipts;
    # the report still makes the bounded decision explicitly.
    with_evidence = True
    _emit_normalized_chain(run, config, with_evidence=with_evidence)
    jobs_path = emit_recon_report_jobs(run, config)
    job = read_jsonl(jobs_path)[0]
    evidence = job["evidence"][0]
    ingest_recon_reports(jobs_path, [_report_result(job, decision, evidence=evidence, include_prior=decision != "NO_DIRECT_COVERAGE_FOUND")], config)
    return run, config


def _zero_recon_run(tmp_path: Path):
    from idea_factory.recon import emit_recon_report_jobs, ingest_recon_reports, normalize_recon_raw

    run, config = _build_dedup_run(tmp_path, with_ready=False)
    _emit_empty_execution_chain(run, config)
    normalize_recon_raw(run, config)
    jobs_path = emit_recon_report_jobs(run, config)
    ingest_recon_reports(jobs_path, [], config)
    return run, config


def _route_result(job: dict[str, object], *, duplicate_spec: bool = False) -> str:
    opportunity = job["opportunity"]
    residual = job["residual"]
    specs = [dict(spec) for spec in MECHANISM_SPECS]
    if duplicate_spec:
        specs[2] = dict(specs[0])
    routes = []
    for index, spec in enumerate(specs):
        route_id = job["route_ids"][index]
        gain_hypothesis = GAIN_HYPOTHESES[index]
        routes.append(
            {
                "schema_version": "idea_factory.route_proposal.v2",
                "route_id": route_id,
                "route_index": index + 1,
                "opportunity_id": job["opportunity_id"],
                "opportunity_sha256": job["opportunity_sha256"],
                "frozen_opportunity": job["opportunity"],
                "scope_constraints": job["scope_constraints"],
                "resource_constraints": job["resource_constraints"],
                "nearest_prior_bindings": job["nearest_prior_bindings"],
                "residual": job["residual"],
                "old_assumption_changed": opportunity["assumption_x"],
                "new_assumption": f"{opportunity['condition_z']}, test {gain_hypothesis}.",
                "new_mechanism": _render_mechanism(spec),
                "mechanism_spec": spec,
                "why_it_addresses_failure": f"It directly targets {opportunity['failure_f']}.",
                "why_nearest_priors_cannot": f"It preserves the bound residual: {residual['summary']}",
                "source_of_gain": f"{gain_hypothesis}; gain must come from {opportunity['missing_capability_w']}, not {opportunity['alternative_explanation_a']}.",
                "causal_gain_hypothesis": gain_hypothesis,
                "required_resources": list(job["allowed_resources"]),
                "cheapest_decisive_test": {
                    "setup": f"Run the supplied decisive experiment {opportunity['decisive_experiment']} {opportunity['condition_z']} for {gain_hypothesis}.",
                    "discriminates_against": opportunity["alternative_explanation_a"],
                    "expected_runtime_or_cost": job["resource_budget_description"],
                },
                "kill_condition": f"Kill {gain_hypothesis} if the test does not reduce {opportunity['failure_f']} against the bound baseline.",
            }
        )
    return json.dumps(
        {
            "schema_version": "idea_factory.route_generator_result.v2",
            "job_id": job["job_id"],
            "opportunity_id": job["opportunity_id"],
            "route_job_sha256": job["route_job_sha256"],
            "prompt_sha256": job["prompt_sha256"],
            "recon_ready_record_sha256": job["recon_ready_record_sha256"],
            "routes": routes,
        }
    )


def _accepted_routes(tmp_path: Path):
    from idea_factory.routes import emit_route_jobs, ingest_route_results

    run, config = _ready_run(tmp_path)
    jobs_path = emit_route_jobs(run, config, allow_test_ready=True)
    job = read_jsonl(jobs_path)[0]
    ingest_route_results(jobs_path, [_route_result(job)], config, allow_test_ready=True)
    return run, config


def _review_result(job: dict[str, object], decision: str, *, secret: str | None = None) -> str:
    component_ids = list(job["mechanism_component_ids"])
    scope_constraint_id = "CONDITION_Z" if decision == "NARROW" else None
    payload = {
        "schema_version": "idea_factory.reviewer_result.v1",
        "job_id": job["job_id"],
        "route_id": job["route_id"],
        "opportunity_id": job["opportunity_id"],
        "round": job["round"],
        "review_job_sha256": job["review_job_sha256"],
        "prompt_sha256": job["prompt_sha256"],
        "decision": decision,
        "strongest_baseline": {"evidence_id": job["allowed_baseline_evidence_ids"][0], "comparison": "MATCHED_BUDGET"},
        "a_plus_b_objection": {"component_ids": component_ids[:2], "verdict": "PLAUSIBLE_COMBINATION"},
        "source_of_gain_verdict": {"source_of_gain_sha256": job["source_of_gain_sha256"], "verdict": "ISOLATED"},
        "falsifiability_verdict": {
            "cheapest_test_sha256": job["cheapest_test_sha256"],
            "kill_condition_sha256": job["kill_condition_sha256"],
            "verdict": "FALSIFIABLE",
        },
        "cost_risk": {
            "required_resource_ids": job["required_resource_ids"],
            "resource_constraints_sha256": job["resource_constraints_sha256"],
            "verdict": "WITHIN_BOUND",
        },
        "residual_claim": {
            "residual_sha256": job["residual_sha256"],
            "opportunity_sha256": job["opportunity_sha256"],
            "disposition": decision,
            "scope_constraint_id": scope_constraint_id,
        },
    }
    if secret is not None:
        payload["secret"] = secret
    return json.dumps(payload)


def test_public_contract_and_prompts_freeze_truth_boundaries() -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results, validate_review_bundle
    from idea_factory.routes import emit_route_jobs, ingest_route_results, validate_route_bundle

    assert all(callable(item) for item in (emit_route_jobs, ingest_route_results, validate_route_bundle))
    assert all(callable(item) for item in (emit_review_jobs, ingest_review_results, validate_review_bundle))
    route_prompt = ROUTE_PROMPT.read_text(encoding="utf-8")
    review_prompt = REVIEW_PROMPT.read_text(encoding="utf-8")
    assert "between 1 and 3" in route_prompt and "absolute novelty" in route_prompt
    assert "MechanismSpec" in route_prompt and "required_resources" in route_prompt
    assert "result_schema_sha256" in route_prompt and "MechanismSpec" in route_prompt
    assert "causal_gain_hypothesis" in route_prompt and "NOT_AUTHENTICATED" in route_prompt
    assert "previously attempted" in route_prompt and "access_token" in route_prompt
    assert "do not rename" in route_prompt.lower() and "do not rename" in review_prompt.lower()
    assert "cannot judge novelty" in review_prompt.lower() and "KILL" in review_prompt and "NARROW" in review_prompt
    assert "structured" in review_prompt.lower() and "evidence_id" in review_prompt
    assert "result_schema_sha256" in review_prompt and "operator_attested_execution_ready" in review_prompt
    assert "independently verified" in review_prompt and "client_secret" in review_prompt
    assert "accepted job_id" in review_prompt and "exact idempotent replay" in review_prompt


def test_route_jobs_consume_only_replayed_ready_and_bind_exact_inputs(tmp_path: Path) -> None:
    from idea_factory.routes import emit_route_jobs, validate_route_jobs

    run, config = _ready_run(tmp_path)
    with pytest.raises(ValueError, match="TEST_ONLY"):
        emit_route_jobs(run, config)
    path = emit_route_jobs(run, config, allow_test_ready=True)
    jobs = validate_route_jobs(run, config, allow_test_ready=True)
    assert path == run / "routes" / "jobs.jsonl" and len(jobs) == 1
    job = jobs[0]
    assert job["decision"] == "NO_DIRECT_COVERAGE_FOUND"
    assert job["opportunity_sha256"] == _canonical_hash(job["opportunity"])
    assert job["evidence_flags"] == job["opportunity"]["inference_flags"]
    assert set(job["scope_constraints"]) == {"condition_z", "scope_compatibility", "decisive_experiment"}
    assert set(job["mechanism_facets"]) == {
        "intervention_site", "operation", "target_state", "learning_signal", "state_representation"
    }
    assert job["allowed_resources"] == ["BOUND_INPUT_ARTIFACTS", "SINGLE_SEED_SMOKE_COMPUTE"]
    assert job["resource_constraints"] == {
        "data_access": "PUBLIC_BOUND_ARTIFACTS_ONLY",
        "filesystem_access": "RUN_LOCAL_READ_WRITE_ONLY",
        "compute_budget": "SINGLE_SEED_SMOKE_ONLY",
    }
    assert job["nearest_prior_bindings"] == []
    assert set(job["recon_provenance"]) >= {"report_job_id", "query_pack_sha256", "normalized_evidence_manifest_sha256", "protocol_hash"}
    assert job["readiness_scope"] == "TEST_ONLY" and job["live_ready"] is False
    assert job["prompt_source"] == str(ROUTE_PROMPT.resolve())
    assert job["prompt_sha256"] == hashlib.sha256(ROUTE_PROMPT.read_bytes()).hexdigest()
    assert len(job["route_ids"]) == 3
    ready = run / "recon" / "ready_for_routes.jsonl"
    rows = read_jsonl(ready); rows[0]["decision"] = "COVERED"; write_jsonl(ready, rows)
    with pytest.raises(ValueError, match="recon|replay"):
        validate_route_jobs(run, config, allow_test_ready=True)


def test_route_job_uses_generic_mechanism_vocabulary_without_compatibility_claim(tmp_path: Path) -> None:
    from idea_factory.routes import emit_route_jobs

    run, config = _ready_run(tmp_path)
    job = read_jsonl(emit_route_jobs(run, config, allow_test_ready=True))[0]
    assert "mechanism_catalog" not in job
    vocabulary_text = json.dumps(job["mechanism_facets"]).casefold()
    assert "compatibility" not in vocabulary_text
    assert job["mechanism_grounding_claim"] == "STRUCTURED_AND_OPPORTUNITY_BOUND_ONLY"


def test_route_ingestion_accepts_exactly_three_bound_distinct_routes_and_replays(tmp_path: Path) -> None:
    from idea_factory.routes import emit_route_jobs, ingest_route_results, validate_route_bundle

    run, config = _ready_run(tmp_path)
    jobs_path = emit_route_jobs(run, config, allow_test_ready=True); job = read_jsonl(jobs_path)[0]
    paths = ingest_route_results(jobs_path, [_route_result(job)], config, allow_test_ready=True)
    bundle = validate_route_bundle(run, config, allow_test_ready=True)
    assert paths["results"] == run / "routes" / "results.jsonl"
    assert len(bundle["results"]) == 3 and len(bundle["outcomes"]) == 1
    assert [row["route_id"] for row in bundle["results"]] == job["route_ids"]
    assert all(row["new_mechanism"] for row in bundle["results"])
    assert all(row["mechanism_fingerprint"] == _canonical_hash(row["mechanism_spec"]) for row in bundle["results"])
    assert all(row["mechanism_spec_sha256"] == row["mechanism_fingerprint"] for row in bundle["results"])
    assert bundle["outcomes"][0]["status"] == "ACCEPTED"
    assert bundle["manifest"]["accepted_route_count"] == 3
    first = {name: (run / "routes" / name).read_bytes() for name in ("results.jsonl", "outcomes.jsonl", "bundle_manifest.json")}
    ingest_route_results(jobs_path, [_route_result(job)], config, allow_test_ready=True)
    assert first == {name: (run / "routes" / name).read_bytes() for name in first}


def test_mechanism_vocabulary_rejects_id_only_distinctness(tmp_path: Path) -> None:
    from idea_factory.routes import emit_route_jobs, ingest_route_results, validate_route_bundle

    run, config = _ready_run(tmp_path)
    jobs_path = emit_route_jobs(run, config, allow_test_ready=True); job = read_jsonl(jobs_path)[0]
    raw = json.loads(_route_result(job))
    for route in raw["routes"][1:]:
        for key in ("new_assumption", "source_of_gain", "cheapest_decisive_test", "kill_condition"):
            route[key] = raw["routes"][0][key]
    ingest_route_results(jobs_path, [json.dumps(raw)], config, allow_test_ready=True)
    outcome = validate_route_bundle(run, config, allow_test_ready=True)["outcomes"][0]
    assert outcome["status"] == "REJECTED"
    assert outcome["error_code"] == "ROUTE_DISTINCTNESS_INVALID"


def test_route_rejects_identical_normalized_source_of_gain_despite_distinct_hypotheses(tmp_path: Path) -> None:
    from idea_factory.routes import emit_route_jobs, ingest_route_results, validate_route_bundle

    run, config = _ready_run(tmp_path)
    jobs_path = emit_route_jobs(run, config, allow_test_ready=True); job = read_jsonl(jobs_path)[0]
    raw = json.loads(_route_result(job))
    opportunity = job["opportunity"]
    hypotheses = "; ".join(route["causal_gain_hypothesis"] for route in raw["routes"])
    same_source = (
        f"{hypotheses}; gain must come from {opportunity['missing_capability_w']}, "
        f"not {opportunity['alternative_explanation_a']}."
    )
    for route in raw["routes"]:
        route["source_of_gain"] = same_source
    ingest_route_results(jobs_path, [json.dumps(raw)], config, allow_test_ready=True)
    outcome = validate_route_bundle(run, config, allow_test_ready=True)["outcomes"][0]
    assert outcome["status"] == "REJECTED"
    assert outcome["error_code"] == "ROUTE_DISTINCTNESS_INVALID"


@pytest.mark.parametrize(
    "mutation",
    ["duplicate_spec", "blank_facet", "changed_question", "changed_frozen", "changed_scope", "wrong_count", "novelty_claim", "unauthorized_resource", "schema_secret"],
)
def test_route_ingestion_conservatively_rejects_non_distinct_or_reframed_routes(tmp_path: Path, mutation: str) -> None:
    from idea_factory.routes import emit_route_jobs, ingest_route_results, validate_route_bundle

    sentinel = "SECRET_ROUTE_SENTINEL"
    run, config = _ready_run(tmp_path)
    jobs_path = emit_route_jobs(run, config, allow_test_ready=True); job = read_jsonl(jobs_path)[0]
    raw = json.loads(_route_result(job, duplicate_spec=mutation == "duplicate_spec"))
    if mutation == "blank_facet":
        raw["routes"][0]["mechanism_spec"]["intervention_site"] = " "
    if mutation == "changed_question":
        raw["routes"][0]["old_assumption_changed"] = "a different research question"
    if mutation == "changed_frozen":
        raw["routes"][0]["frozen_opportunity"]["failure_f"] = "a different failure"
    if mutation == "changed_scope":
        raw["routes"][0]["scope_constraints"]["condition_z"] = "another scope"
    if mutation == "wrong_count":
        raw["routes"] = []
    if mutation == "novelty_claim":
        raw["routes"][0]["source_of_gain"] += " This route is novel."
    if mutation == "unauthorized_resource":
        raw["routes"][0]["required_resources"] = ["PRIVATE_DATA", "ROOT_ACCESS", "UNLIMITED_COMPUTE"]
    if mutation == "schema_secret":
        raw["routes"][0]["source_of_gain"] += " api_key=sk-SECRETROUTECREDENTIAL"
    ingest_route_results(jobs_path, [json.dumps(raw)], config, allow_test_ready=True)
    bundle = validate_route_bundle(run, config, allow_test_ready=True)
    assert bundle["results"] == () and bundle["outcomes"][0]["status"] == "REJECTED"
    if mutation == "schema_secret":
        assert "SECRETROUTECREDENTIAL" not in json.dumps(bundle)
        assert bundle["outcomes"][0]["error_code"] == "ROUTE_SECRET_DETECTED"


def test_near_prior_route_requires_exact_complete_prior_and_residual_binding(tmp_path: Path) -> None:
    from idea_factory.routes import emit_route_jobs, ingest_route_results, validate_route_bundle

    run, config = _ready_run(tmp_path, decision="NEAR_PRIOR_WITH_RESIDUAL")
    jobs_path = emit_route_jobs(run, config, allow_test_ready=True); job = read_jsonl(jobs_path)[0]
    assert job["nearest_prior_bindings"]
    prior = job["nearest_prior_bindings"][0]
    assert prior["evidence_url_or_id"] in prior["evidence_ids_and_aliases"]
    assert prior["residual_difference"] in job["residual"]["nearest_prior_residuals"]
    raw = json.loads(_route_result(job))
    raw["routes"][0]["nearest_prior_bindings"] = []
    ingest_route_results(jobs_path, [json.dumps(raw)], config, allow_test_ready=True)
    bundle = validate_route_bundle(run, config, allow_test_ready=True)
    assert bundle["results"] == ()
    assert bundle["outcomes"][0]["error_code"] == "ROUTE_BINDING_INVALID"


def test_route_global_priority_firewall_rejects_recursive_claim_variants(tmp_path: Path) -> None:
    from idea_factory.routes import emit_route_jobs, ingest_route_results, validate_route_bundle

    claims = (
        "world-first solution", "first-ever method", "never attempted before", "no prior work exists",
        "no one has studied this", "nobody has done this", "the highest priority direction",
    )
    run, config = _ready_run(tmp_path)
    jobs_path = emit_route_jobs(run, config, allow_test_ready=True); job = read_jsonl(jobs_path)[0]
    for claim in claims:
        raw = json.loads(_route_result(job)); raw["routes"][0]["new_assumption"] += f" {claim}."
        ingest_route_results(jobs_path, [json.dumps(raw)], config, allow_test_ready=True)
        bundle = validate_route_bundle(run, config, allow_test_ready=True)
        assert bundle["results"] == ()
        assert bundle["outcomes"][0]["error_code"] == "ROUTE_GLOBAL_PRIORITY_CLAIM_FORBIDDEN"
    assert "raw_result_json" not in bundle["outcomes"][0]


def test_route_emit_rejects_tampered_existing_bundle_without_mutation(tmp_path: Path) -> None:
    from idea_factory.routes import emit_route_jobs, ingest_route_results, validate_route_bundle

    run, config = _ready_run(tmp_path)
    jobs_path = emit_route_jobs(run, config, allow_test_ready=True); job = read_jsonl(jobs_path)[0]
    ingest_route_results(jobs_path, [_route_result(job)], config, allow_test_ready=True)
    rows = read_jsonl(run / "routes" / "results.jsonl"); rows[0]["new_mechanism"] = "tampered"; write_jsonl(run / "routes" / "results.jsonl", rows)
    with pytest.raises(ValueError, match="replay|manifest|hash"):
        validate_route_bundle(run, config, allow_test_ready=True)
    routes = run / "routes"
    names = ("jobs.jsonl", "results.jsonl", "outcomes.jsonl", "bundle_manifest.json")
    before = {name: (routes / name).read_bytes() for name in names}
    with pytest.raises(ValueError, match="replay|manifest|hash|bundle"):
        emit_route_jobs(run, config, allow_test_ready=True)
    assert before == {name: (routes / name).read_bytes() for name in names}


def test_route_mutation_lock_fails_fast_and_preserves_unknown_lock(tmp_path: Path) -> None:
    from idea_factory.routes import emit_route_jobs

    run, config = _ready_run(tmp_path)
    routes = run / "routes"; routes.mkdir()
    lock = routes / ".stage.lock"; lock.write_text("unknown-owner", encoding="utf-8")
    with pytest.raises((FileExistsError, RuntimeError, ValueError), match="lock|mutation|busy"):
        emit_route_jobs(run, config, allow_test_ready=True)
    assert lock.read_text(encoding="utf-8") == "unknown-owner"


def test_stage_lock_serializes_threads_and_processes(tmp_path: Path) -> None:
    from idea_factory.stage_io import stage_mutation_lock

    stage = tmp_path / "stage"; stage.mkdir()
    thread_errors: list[str] = []
    with stage_mutation_lock(stage):
        contender = threading.Thread(
            target=lambda: (
                thread_errors.append("acquired")
                if _try_stage_lock(stage)
                else thread_errors.append("busy")
            )
        )
        contender.start(); contender.join(timeout=5)
        assert thread_errors == ["busy"]

    ready = tmp_path / "ready"
    code = """
import sys
from pathlib import Path
from idea_factory.stage_io import stage_mutation_lock
with stage_mutation_lock(Path(sys.argv[1])):
    Path(sys.argv[2]).write_text('ready', encoding='utf-8')
    sys.stdin.readline()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(stage), str(ready)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.time() + 10
        while not ready.exists() and process.poll() is None and time.time() < deadline:
            time.sleep(0.02)
        assert ready.exists(), process.stderr.read()
        with pytest.raises(RuntimeError, match="lock|busy"):
            with stage_mutation_lock(stage):
                pass
    finally:
        if process.stdin:
            process.stdin.write("\n"); process.stdin.flush()
        process.communicate(timeout=10)


def _try_stage_lock(stage: Path) -> bool:
    from idea_factory.stage_io import stage_mutation_lock

    try:
        with stage_mutation_lock(stage):
            return True
    except RuntimeError:
        return False


def test_route_retry_publish_failure_rolls_back_old_bundle_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import idea_factory.artifacts as artifacts_module
    from idea_factory.routes import emit_route_jobs, ingest_route_results

    run, config = _ready_run(tmp_path)
    jobs_path = emit_route_jobs(run, config, allow_test_ready=True); job = read_jsonl(jobs_path)[0]
    rejected = json.loads(_route_result(job)); rejected["routes"][0]["required_resources"] = ["ROOT_ACCESS"]
    ingest_route_results(jobs_path, [json.dumps(rejected)], config, allow_test_ready=True)
    routes = run / "routes"
    names = ("jobs.jsonl", "results.jsonl", "outcomes.jsonl", "bundle_manifest.json")
    before = {name: (routes / name).read_bytes() for name in names}
    real_replace = artifacts_module.os.replace
    calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(artifacts_module.os, "replace", fail_second)
    with pytest.raises(OSError, match="injected"):
        ingest_route_results(jobs_path, [_route_result(job)], config, allow_test_ready=True)
    assert before == {name: (routes / name).read_bytes() for name in names}


def test_route_ingest_rejects_existing_bundle_tamper_before_mutation(tmp_path: Path) -> None:
    from idea_factory.routes import emit_route_jobs, ingest_route_results

    run, config = _ready_run(tmp_path)
    jobs_path = emit_route_jobs(run, config, allow_test_ready=True); job = read_jsonl(jobs_path)[0]
    raw = _route_result(job)
    ingest_route_results(jobs_path, [raw], config, allow_test_ready=True)
    outcomes_path = run / "routes" / "outcomes.jsonl"
    outcomes = read_jsonl(outcomes_path); outcomes[0]["status"] = "REJECTED"; write_jsonl(outcomes_path, outcomes)
    routes = run / "routes"; names = ("jobs.jsonl", "results.jsonl", "outcomes.jsonl", "bundle_manifest.json")
    before = {name: (routes / name).read_bytes() for name in names}
    with pytest.raises(ValueError, match="manifest|tamper|replay|bundle|outcome"):
        ingest_route_results(jobs_path, [raw], config, allow_test_ready=True)
    assert before == {name: (routes / name).read_bytes() for name in names}


def test_accepted_route_is_immutable_but_rejected_route_can_retry(tmp_path: Path) -> None:
    from idea_factory.routes import emit_route_jobs, ingest_route_results, validate_route_bundle

    run, config = _ready_run(tmp_path)
    jobs_path = emit_route_jobs(run, config, allow_test_ready=True); job = read_jsonl(jobs_path)[0]
    accepted_raw = _route_result(job)
    ingest_route_results(jobs_path, [accepted_raw], config, allow_test_ready=True)
    routes = run / "routes"
    names = ("jobs.jsonl", "results.jsonl", "outcomes.jsonl", "bundle_manifest.json")
    accepted_bytes = {name: (routes / name).read_bytes() for name in names}
    changed = json.loads(accepted_raw)
    changed["routes"][0]["source_of_gain"] += " The causal boundary is narrowed."
    with pytest.raises(ValueError, match="accepted|immutable|replay"):
        ingest_route_results(jobs_path, [json.dumps(changed)], config, allow_test_ready=True)
    assert accepted_bytes == {name: (routes / name).read_bytes() for name in names}
    ingest_route_results(jobs_path, [accepted_raw], config, allow_test_ready=True)
    assert accepted_bytes == {name: (routes / name).read_bytes() for name in names}

    other = tmp_path / "retry"; other.mkdir()
    run, config = _ready_run(other)
    jobs_path = emit_route_jobs(run, config, allow_test_ready=True); job = read_jsonl(jobs_path)[0]
    rejected = json.loads(_route_result(job)); rejected["routes"][0]["required_resources"] = ["ROOT_ACCESS"]
    ingest_route_results(jobs_path, [json.dumps(rejected)], config, allow_test_ready=True)
    assert validate_route_bundle(run, config, allow_test_ready=True)["outcomes"][0]["status"] == "REJECTED"
    ingest_route_results(jobs_path, [_route_result(job)], config, allow_test_ready=True)
    assert validate_route_bundle(run, config, allow_test_ready=True)["outcomes"][0]["status"] == "ACCEPTED"


def test_route_directory_link_is_rejected_without_touching_outside(tmp_path: Path) -> None:
    from idea_factory.routes import emit_route_jobs

    run, config = _ready_run(tmp_path)
    outside = tmp_path / "outside"; outside.mkdir(); sentinel = outside / "sentinel"; sentinel.write_text("keep", encoding="utf-8")
    try:
        os.symlink(outside, run / "routes", target_is_directory=True)
    except OSError:
        import subprocess

        completed = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(run / "routes"), str(outside)], capture_output=True)
        if completed.returncode:
            pytest.skip("directory links unavailable")
    with pytest.raises(ValueError, match="routes|escape|anchored"):
        emit_route_jobs(run, config, allow_test_ready=True)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_review_round_one_outputs_all_verdicts_and_round_two_only_for_narrow(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results, validate_review_bundle

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True)
    jobs = read_jsonl(jobs_path)
    assert len(jobs) == 3 and all(job["round"] == 1 for job in jobs)
    decisions = ("KILL", "NARROW", "PASS_TO_HUMAN")
    round_one_raw = [_review_result(job, decision) for job, decision in zip(jobs, decisions)]
    ingest_review_results(jobs_path, round_one_raw, config, allow_test_ready=True)
    first = validate_review_bundle(run, config, allow_test_ready=True)
    assert [row["decision"] for row in first["results"]] == list(decisions)
    assert first["manifest"]["resolution_status"] == "HAS_SURVIVORS"
    pending_path = emit_review_jobs(run, config, round_number=2, allow_test_ready=True)
    pending_jobs = read_jsonl(pending_path)
    assert len(pending_jobs) == 1
    second = pending_jobs[0]
    assert second["round"] == 2 and second["route_id"] == jobs[1]["route_id"]
    assert second["opportunity_id"] == jobs[1]["opportunity_id"]
    assert second["route"] == jobs[1]["route"]
    assert second["parent_round_one_result_sha256"] == _canonical_hash(first["results"][1])
    assert second["controlled_catalog"]["allowed_decisions"] == second["allowed_decisions"] == ["KILL", "PASS_TO_HUMAN"]
    ingest_review_results(pending_path, [_review_result(second, "PASS_TO_HUMAN")], config, allow_test_ready=True)
    final = validate_review_bundle(run, config, allow_test_ready=True)
    assert final["manifest"]["round_two_count"] == 1
    assert final["manifest"]["pass_to_human_count"] == 2
    assert all(row["authenticity"] == "NOT_AUTHENTICATED" for row in final["results"])
    assert final["manifest"]["authenticity"] == "NOT_AUTHENTICATED"
    assert final["manifest"]["independently_verified"] is False
    assert final["manifest"]["operator_attested_execution_ready"] is False
    assert final["manifest"]["result_schema_sha256s"] == [jobs[0]["result_schema_sha256"]]


def test_review_mutation_lock_fails_fast_and_preserves_unknown_lock(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs

    run, config = _accepted_routes(tmp_path)
    review = run / "review"; review.mkdir()
    lock = review / ".stage.lock"; lock.write_text("unknown-owner", encoding="utf-8")
    with pytest.raises((FileExistsError, RuntimeError, ValueError), match="lock|mutation|busy"):
        emit_review_jobs(run, config, allow_test_ready=True)
    assert lock.read_text(encoding="utf-8") == "unknown-owner"


def test_round_two_emit_uses_pending_handoff_without_mutating_main_bundle(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results, validate_review_bundle

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    round_one = [_review_result(job, "NARROW" if index == 0 else "KILL") for index, job in enumerate(jobs)]
    ingest_review_results(jobs_path, round_one, config, allow_test_ready=True)
    review = run / "review"
    names = ("jobs.jsonl", "results.jsonl", "outcomes.jsonl", "bundle_manifest.json")
    before = {name: (review / name).read_bytes() for name in names}
    pending_path = emit_review_jobs(run, config, round_number=2, allow_test_ready=True)
    assert pending_path == review / "round2_jobs.jsonl"
    pending = read_jsonl(pending_path)
    assert len(pending) == 1 and pending[0]["round"] == 2
    assert before == {name: (review / name).read_bytes() for name in names}
    validate_review_bundle(run, config, allow_test_ready=True)
    ingest_review_results(pending_path, [_review_result(pending[0], "PASS_TO_HUMAN")], config, allow_test_ready=True)
    assert not pending_path.exists()
    final = validate_review_bundle(run, config, allow_test_ready=True)
    assert len(final["jobs"]) == 4 and final["results"][-1]["round"] == 2


def test_review_retry_publish_failure_restores_old_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import idea_factory.artifacts as artifacts_module
    from idea_factory.review import emit_review_jobs, ingest_review_results

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    kills = [_review_result(job, "KILL") for job in jobs]
    rejected = json.loads(kills[0]); rejected["strongest_baseline"]["evidence_id"] = "invented"
    ingest_review_results(jobs_path, [json.dumps(rejected), *kills[1:]], config, allow_test_ready=True)
    review = run / "review"; names = ("jobs.jsonl", "results.jsonl", "outcomes.jsonl", "bundle_manifest.json")
    before = {name: (review / name).read_bytes() for name in names}
    real_replace = artifacts_module.os.replace; calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected review replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(artifacts_module.os, "replace", fail_second)
    with pytest.raises(OSError, match="injected"):
        ingest_review_results(jobs_path, kills, config, allow_test_ready=True)
    assert before == {name: (review / name).read_bytes() for name in names}


def test_round_two_merge_publish_failure_restores_main_and_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import idea_factory.artifacts as artifacts_module
    from idea_factory.review import emit_review_jobs, ingest_review_results

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    round_one = [_review_result(job, "NARROW" if index == 0 else "KILL") for index, job in enumerate(jobs)]
    ingest_review_results(jobs_path, round_one, config, allow_test_ready=True)
    pending_path = emit_review_jobs(run, config, round_number=2, allow_test_ready=True)
    pending = read_jsonl(pending_path)
    review = run / "review"
    names = ("jobs.jsonl", "results.jsonl", "outcomes.jsonl", "bundle_manifest.json", "round2_jobs.jsonl")
    before = {name: (review / name).read_bytes() for name in names}
    real_replace = artifacts_module.os.replace; calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected pending merge failure")
        return real_replace(source, destination)

    monkeypatch.setattr(artifacts_module.os, "replace", fail_second)
    with pytest.raises(OSError, match="injected pending"):
        ingest_review_results(pending_path, [_review_result(pending[0], "PASS_TO_HUMAN")], config, allow_test_ready=True)
    assert before == {name: (review / name).read_bytes() for name in names}


def test_review_forbids_second_round_without_narrow_and_rejects_round_two_narrow(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results, validate_review_bundle

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    raw = [_review_result(job, "KILL") for job in jobs]
    ingest_review_results(jobs_path, raw, config, allow_test_ready=True)
    with pytest.raises(ValueError, match="NARROW"):
        emit_review_jobs(run, config, round_number=2, allow_test_ready=True)

    other = tmp_path / "narrow"; other.mkdir()
    run, config = _accepted_routes(other)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    round_one = [_review_result(job, "NARROW" if index == 0 else "KILL") for index, job in enumerate(jobs)]
    ingest_review_results(jobs_path, round_one, config, allow_test_ready=True)
    pending_path = emit_review_jobs(run, config, round_number=2, allow_test_ready=True); pending = read_jsonl(pending_path)
    ingest_review_results(pending_path, [_review_result(pending[0], "NARROW")], config, allow_test_ready=True)
    bundle = validate_review_bundle(run, config, allow_test_ready=True)
    assert bundle["outcomes"][-1]["status"] == "REJECTED"


def test_round_two_rejects_stale_current_round_one_narrow_sha(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    round_one = [_review_result(job, "NARROW" if index == 0 else "KILL") for index, job in enumerate(jobs)]
    ingest_review_results(jobs_path, round_one, config, allow_test_ready=True)
    pending_path = emit_review_jobs(run, config, round_number=2, allow_test_ready=True); pending = read_jsonl(pending_path)
    current = read_jsonl(run / "review" / "results.jsonl")
    current[0]["residual_claim"]["scope_constraint_id"] = "DECISIVE_EXPERIMENT"
    write_jsonl(run / "review" / "results.jsonl", current)
    with pytest.raises(ValueError, match="stale|round-one|SHA|replay|manifest|bundle"):
        ingest_review_results(pending_path, [_review_result(pending[0], "PASS_TO_HUMAN")], config, allow_test_ready=True)


def test_round_two_rejects_incoming_kill_parent_atomically(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    round_one = [_review_result(job, "NARROW" if index == 0 else "KILL") for index, job in enumerate(jobs)]
    ingest_review_results(jobs_path, round_one, config, allow_test_ready=True)
    main_path = jobs_path
    pending_path = emit_review_jobs(run, config, round_number=2, allow_test_ready=True)
    review = run / "review"
    names = ("jobs.jsonl", "results.jsonl", "outcomes.jsonl", "bundle_manifest.json", "round2_jobs.jsonl")
    before = {name: (review / name).read_bytes() for name in names}
    incoming = [_review_result(jobs[0], "KILL"), *round_one[1:]]
    with pytest.raises(ValueError, match="accepted|immutable|replay|round-one|parent"):
        ingest_review_results(main_path, incoming, config, allow_test_ready=True)
    assert before == {name: (review / name).read_bytes() for name in before}
    assert pending_path.exists()


def test_accepted_kill_job_cannot_be_rewritten_to_pass_and_artifacts_are_atomic(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    kill_batch = [_review_result(job, "KILL") for job in jobs]
    ingest_review_results(jobs_path, kill_batch, config, allow_test_ready=True)
    review = run / "review"
    names = ("jobs.jsonl", "results.jsonl", "outcomes.jsonl", "bundle_manifest.json")
    before = {name: (review / name).read_bytes() for name in names}
    rewritten = [_review_result(jobs[0], "PASS_TO_HUMAN"), *kill_batch[1:]]
    with pytest.raises(ValueError, match="accepted|immutable|terminal|replay"):
        ingest_review_results(jobs_path, rewritten, config, allow_test_ready=True)
    assert before == {name: (review / name).read_bytes() for name in names}


def test_accepted_narrow_job_allows_only_exact_idempotent_replay(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    batch = [_review_result(job, "NARROW" if index == 0 else "KILL") for index, job in enumerate(jobs)]
    ingest_review_results(jobs_path, batch, config, allow_test_ready=True)
    review = run / "review"
    names = ("jobs.jsonl", "results.jsonl", "outcomes.jsonl", "bundle_manifest.json")
    before = {name: (review / name).read_bytes() for name in names}
    changed = json.loads(batch[0]); changed["source_of_gain_verdict"]["verdict"] = "CONFOUNDED"
    with pytest.raises(ValueError, match="accepted|immutable|terminal|replay"):
        ingest_review_results(jobs_path, [json.dumps(changed), *batch[1:]], config, allow_test_ready=True)
    assert before == {name: (review / name).read_bytes() for name in names}
    ingest_review_results(jobs_path, batch, config, allow_test_ready=True)
    assert before == {name: (review / name).read_bytes() for name in names}


def test_rejected_review_job_can_retry_without_overwriting_accepted_jobs(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results, validate_review_bundle

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    accepted_kills = [_review_result(job, "KILL") for job in jobs]
    rejected = json.loads(accepted_kills[0]); rejected["strongest_baseline"]["evidence_id"] = "invented"
    ingest_review_results(jobs_path, [json.dumps(rejected), *accepted_kills[1:]], config, allow_test_ready=True)
    first = validate_review_bundle(run, config, allow_test_ready=True)
    assert first["outcomes"][0]["status"] == "REJECTED"
    ingest_review_results(jobs_path, accepted_kills, config, allow_test_ready=True)
    final = validate_review_bundle(run, config, allow_test_ready=True)
    assert all(outcome["status"] == "ACCEPTED" for outcome in final["outcomes"])
    assert final["manifest"]["zero_survivor"] is True


def test_round_two_only_batch_binds_current_round_one_narrow(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results, validate_review_bundle

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    round_one = [_review_result(job, "NARROW" if index == 0 else "KILL") for index, job in enumerate(jobs)]
    ingest_review_results(jobs_path, round_one, config, allow_test_ready=True)
    pending_path = emit_review_jobs(run, config, round_number=2, allow_test_ready=True); pending = read_jsonl(pending_path)
    ingest_review_results(pending_path, [_review_result(pending[0], "PASS_TO_HUMAN")], config, allow_test_ready=True)
    bundle = validate_review_bundle(run, config, allow_test_ready=True)
    assert len(bundle["results"]) == len(jobs) + len(pending)
    assert bundle["results"][-1]["round"] == 2
    assert bundle["results"][-1]["decision"] == "PASS_TO_HUMAN"


def test_review_full_batch_canonicalizes_by_route_and_round(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results, validate_review_bundle

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    round_one = [_review_result(job, "NARROW" if index == 0 else "KILL") for index, job in enumerate(jobs)]
    ingest_review_results(jobs_path, round_one, config, allow_test_ready=True)
    pending_path = emit_review_jobs(run, config, round_number=2, allow_test_ready=True); pending = read_jsonl(pending_path)
    ingest_review_results(pending_path, [_review_result(pending[0], "PASS_TO_HUMAN")], config, allow_test_ready=True)
    combined = read_jsonl(run / "review" / "jobs.jsonl")
    full_batch = round_one + [_review_result(pending[0], "PASS_TO_HUMAN")]
    ingest_review_results(run / "review" / "jobs.jsonl", list(reversed(full_batch)), config, allow_test_ready=True)
    bundle = validate_review_bundle(run, config, allow_test_ready=True)
    assert [(row["route_id"], row["round"]) for row in bundle["outcomes"]] == [
        (job["route_id"], job["round"]) for job in combined
    ]


def test_jobs_embed_replay_bound_schema_and_unsigned_authenticity(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs
    from idea_factory.routes import emit_route_jobs, ingest_route_results, validate_route_bundle

    run, config = _ready_run(tmp_path)
    route_jobs_path = emit_route_jobs(run, config, allow_test_ready=True); route_job = read_jsonl(route_jobs_path)[0]
    assert route_job["result_schema"]["$defs"]
    assert route_job["result_schema_sha256"] == _canonical_hash(route_job["result_schema"])
    assert route_job["attestation_scope"] == "TEST_ONLY"
    assert route_job["authenticity"] == "NOT_AUTHENTICATED"
    assert route_job["operator_attested_execution_ready"] is False
    ingest_route_results(route_jobs_path, [_route_result(route_job)], config, allow_test_ready=True)
    route_bundle = validate_route_bundle(run, config, allow_test_ready=True)
    assert all(row["authenticity"] == "NOT_AUTHENTICATED" for row in route_bundle["results"])
    assert route_bundle["manifest"]["authenticity"] == "NOT_AUTHENTICATED"
    assert route_bundle["manifest"]["result_schema_sha256s"] == [route_job["result_schema_sha256"]]
    review_jobs_path = emit_review_jobs(run, config, allow_test_ready=True); review_job = read_jsonl(review_jobs_path)[0]
    assert review_job["result_schema"]["$defs"]
    assert review_job["result_schema_sha256"] == _canonical_hash(review_job["result_schema"])
    assert review_job["controlled_catalog"]["mechanism_component_ids"] == review_job["mechanism_component_ids"]
    assert review_job["attestation_scope"] == "TEST_ONLY"
    assert review_job["authenticity"] == "NOT_AUTHENTICATED"
    assert review_job["accepted_job_immutable"] is True


def test_priority_and_secret_firewalls_are_centralized_and_conservative(tmp_path: Path) -> None:
    from idea_factory.routes import emit_route_jobs, ingest_route_results, validate_route_bundle

    run, config = _ready_run(tmp_path)
    jobs_path = emit_route_jobs(run, config, allow_test_ready=True); job = read_jsonl(jobs_path)[0]
    for claim in ("first in the world", "first system ever", "never attempted", "no prior work", "world-first", "globally novel"):
        raw = json.loads(_route_result(job)); raw["routes"][0]["why_it_addresses_failure"] += f" {claim}."
        ingest_route_results(jobs_path, [json.dumps(raw)], config, allow_test_ready=True)
        assert validate_route_bundle(run, config, allow_test_ready=True)["outcomes"][0]["error_code"] == "ROUTE_GLOBAL_PRIORITY_CLAIM_FORBIDDEN"
    raw = json.loads(_route_result(job)); raw["metadata"] = {"nested": "first in the world"}
    ingest_route_results(jobs_path, [json.dumps(raw)], config, allow_test_ready=True)
    assert validate_route_bundle(run, config, allow_test_ready=True)["outcomes"][0]["error_code"] == "ROUTE_GLOBAL_PRIORITY_CLAIM_FORBIDDEN"
    for secret in (
        {"access_token": "opaque"},
        {"nested": "github_pat_" + "1234567890abcdefghijklmnopqrstuvwxyz"},
        {"nested": "xoxb-" + "123456789012-123456789012-abcdefghijklmnopqrstuvwx"},
        {"nested": "AIzaSy" + "DUMMYTOKEN1234567890123456789012"},
        {"nested": "-----BEGIN PRIVATE KEY-----"},
    ):
        raw = json.loads(_route_result(job)); raw["metadata"] = secret
        ingest_route_results(jobs_path, [json.dumps(raw)], config, allow_test_ready=True)
        outcome = validate_route_bundle(run, config, allow_test_ready=True)["outcomes"][0]
        assert outcome["error_code"] == "ROUTE_SECRET_DETECTED"
        assert "raw_result_json" not in outcome


def test_safety_normalizes_unicode_detects_dsn_and_avoids_priority_false_positives() -> None:
    from idea_factory.safety import contains_global_priority_claim, contains_secret

    assert contains_global_priority_claim("This is first in the world")
    assert contains_global_priority_claim("This is world-\u200bfirst")
    assert contains_global_priority_claim("This is glo\u200bbally novel")
    assert contains_global_priority_claim("This is no\u200bvel")
    assert contains_global_priority_claim("This is ｆｉｒｓｔ in the world")
    assert contains_global_priority_claim("This has never been attempted")
    assert not contains_global_priority_claim("Measure the first stage before ablation")
    assert not contains_global_priority_claim("A new control policy is evaluated")
    assert not contains_global_priority_claim("The previously attempted baseline is retained")
    assert contains_secret({"nested": "postgresql://alice:synthetic-pass@db.example/test"})
    assert contains_secret({"nested": "password is synthetic-passphrase"})
    assert contains_secret({"nested": "access secret is synthetic-passphrase"})
    assert contains_secret({"nested": "https://user:synthetic-pass@example.test/path"})
    assert contains_secret({"client\u200b_secret": "synthetic"})


@pytest.mark.parametrize(
    "claim",
    (
        "the world's first recovery system",
        "the world-first recovery system",
        "the world first recovery system",
        "the first in the world",
        "a first-of-its-kind controller",
        "the first system ever reported",
        "this has never previously been attempted",
        "this has never ever been done",
        "this has never been studied",
        "this was never built",
        "this was never reported",
        "this has not previously been attempted",
        "these have not ever been built",
        "there is no existing work",
        "there is no prior work",
        "there is no existing prior work",
        "there is no prior existing work",
        "there is no published prior work",
        "there is no previous published existing work",
        "no one has reported this",
        "nobody has built this",
        "an unprecedented result",
        "this mechanism is no\u200bvel",
        "an absolute novelty claim",
        "the world’s-\u200bfirst system",
    ),
)
def test_priority_matcher_rejects_canonical_absolute_claim_families(claim: str) -> None:
    from idea_factory.safety import contains_global_priority_claim

    assert contains_global_priority_claim(claim)


@pytest.mark.parametrize(
    "description",
    (
        "measure the first stage before ablation",
        "mask the first token in each sequence",
        "A new control policy is evaluated",
        "the first pass computes a checksum",
        "the first round uses the existing work queue",
        "the previously attempted baseline is retained",
        "no existing work queue is configured",
        "no prior work item remains in the queue",
        "no published work directory was mounted",
    ),
)
def test_priority_matcher_allows_procedural_first_and_new_descriptions(description: str) -> None:
    from idea_factory.safety import contains_global_priority_claim

    assert not contains_global_priority_claim(description)


def test_review_pending_narrow_is_not_zero_survivor(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results, validate_review_bundle

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    ingest_review_results(jobs_path, [_review_result(job, "NARROW") for job in jobs], config, allow_test_ready=True)
    manifest = validate_review_bundle(run, config, allow_test_ready=True)["manifest"]
    assert manifest["zero_survivor"] is False
    assert manifest["resolution_status"] == "PENDING"


@pytest.mark.parametrize(
    "mutation",
    ["baseline", "quantum_component", "source_gain", "falsifiability", "cost", "residual", "unscoped_narrow"],
)
def test_reviewer_structured_fields_must_bind_route_anchors(tmp_path: Path, mutation: str) -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results, validate_review_bundle

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    decision = "NARROW" if mutation == "unscoped_narrow" else "KILL"
    raw = json.loads(_review_result(jobs[0], decision))
    if mutation == "baseline": raw["strongest_baseline"]["evidence_id"] = "invented-evidence"
    if mutation == "quantum_component": raw["a_plus_b_objection"]["component_ids"][0] = "QUANTUM_COUPLER"
    if mutation == "source_gain": raw["source_of_gain_verdict"]["source_of_gain_sha256"] = "0" * 64
    if mutation == "falsifiability": raw["falsifiability_verdict"]["cheapest_test_sha256"] = "0" * 64
    if mutation == "cost": raw["cost_risk"]["required_resource_ids"] = ["PRIVATE_DATA", "ROOT_ACCESS"]
    if mutation == "residual": raw["residual_claim"]["residual_sha256"] = "0" * 64
    if mutation == "unscoped_narrow": raw["residual_claim"]["scope_constraint_id"] = None
    batch = [json.dumps(raw), _review_result(jobs[1], "KILL"), _review_result(jobs[2], "KILL")]
    ingest_review_results(jobs_path, batch, config, allow_test_ready=True)
    bundle = validate_review_bundle(run, config, allow_test_ready=True)
    assert bundle["outcomes"][0]["status"] == "REJECTED"
    assert bundle["outcomes"][0]["error_code"] == "REVIEW_ANCHOR_INVALID"


def test_reviewer_rejects_novelty_claim_without_persisting_raw_secret_and_supports_zero_survivor(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results, validate_review_bundle

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    sentinel = "SECRET_REVIEW_SENTINEL"
    bad = json.loads(_review_result(jobs[0], "PASS_TO_HUMAN"))
    bad["model_comment"] = {"nested": "This is a world-first direction."}
    secret = json.loads(_review_result(jobs[1], "KILL")); secret["metadata"] = {"authorization": "Bearer SECRET_REVIEW_CREDENTIAL"}
    raw = [json.dumps(bad), json.dumps(secret), _review_result(jobs[2], "KILL")]
    ingest_review_results(jobs_path, raw, config, allow_test_ready=True)
    bundle = validate_review_bundle(run, config, allow_test_ready=True)
    assert bundle["outcomes"][0]["status"] == "REJECTED"
    assert bundle["outcomes"][1]["status"] == "REJECTED"
    assert "SECRET_REVIEW_CREDENTIAL" not in json.dumps(bundle) and "raw_result_json" not in bundle["outcomes"][1]
    assert bundle["outcomes"][0]["error_code"] == "REVIEW_GLOBAL_PRIORITY_CLAIM_FORBIDDEN"
    assert bundle["outcomes"][1]["error_code"] == "REVIEW_SECRET_DETECTED"
    assert bundle["manifest"]["pass_to_human_count"] == 0
    assert bundle["manifest"]["zero_survivor"] is False
    assert bundle["manifest"]["resolution_status"] == "UNRESOLVED"

    zero = tmp_path / "zero"; zero.mkdir()
    run, config = _accepted_routes(zero)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    ingest_review_results(jobs_path, [_review_result(job, "KILL") for job in jobs], config, allow_test_ready=True)
    bundle = validate_review_bundle(run, config, allow_test_ready=True)
    assert bundle["manifest"]["pass_to_human_count"] == 0
    assert bundle["manifest"]["zero_survivor"] is True
    assert bundle["manifest"]["resolution_status"] == "ZERO_SURVIVOR"


def test_review_bundle_replays_and_rejects_directory_escape(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results, validate_review_bundle

    run, config = _accepted_routes(tmp_path)
    jobs_path = emit_review_jobs(run, config, allow_test_ready=True); jobs = read_jsonl(jobs_path)
    ingest_review_results(jobs_path, [_review_result(job, "KILL") for job in jobs], config, allow_test_ready=True)
    rows = read_jsonl(run / "review" / "results.jsonl"); rows[0]["cost_risk"] = "tampered"; write_jsonl(run / "review" / "results.jsonl", rows)
    with pytest.raises(ValueError, match="replay|manifest|hash"):
        validate_review_bundle(run, config, allow_test_ready=True)

    outside_case = tmp_path / "escape"; outside_case.mkdir()
    run, config = _accepted_routes(outside_case)
    outside = tmp_path / "outside-review"; outside.mkdir(); sentinel = outside / "sentinel"; sentinel.write_text("keep", encoding="utf-8")
    try:
        os.symlink(outside, run / "review", target_is_directory=True)
    except OSError:
        import subprocess

        completed = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(run / "review"), str(outside)], capture_output=True)
        if completed.returncode:
            pytest.skip("directory links unavailable")
    with pytest.raises(ValueError, match="review|escape|anchored"):
        emit_review_jobs(run, config, allow_test_ready=True)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_zero_input_route_and_review_bundles_remain_replayable_not_zero_survivor(tmp_path: Path) -> None:
    from idea_factory.review import emit_review_jobs, ingest_review_results, validate_review_bundle
    from idea_factory.routes import emit_route_jobs, ingest_route_results, validate_route_bundle

    run, config = _zero_recon_run(tmp_path)
    route_jobs = emit_route_jobs(run, config)
    assert read_jsonl(route_jobs) == []
    ingest_route_results(route_jobs, [], config)
    route_bundle = validate_route_bundle(run, config)
    assert route_bundle["results"] == ()
    assert route_bundle["manifest"]["attestation_scopes"] == []
    assert route_bundle["manifest"]["authenticity"] == "NOT_AUTHENTICATED"
    assert route_bundle["manifest"]["operator_attested_execution_ready"] is False
    review_jobs = emit_review_jobs(run, config)
    assert read_jsonl(review_jobs) == []
    ingest_review_results(review_jobs, [], config)
    manifest = validate_review_bundle(run, config)["manifest"]
    assert manifest["resolution_status"] == "ZERO_INPUT"
    assert manifest["zero_survivor"] is False
    assert manifest["attestation_scopes"] == []
    assert manifest["authenticity"] == "NOT_AUTHENTICATED"
    assert manifest["operator_attested_execution_ready"] is False
