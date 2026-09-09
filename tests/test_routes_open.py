from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from idea_factory.routes import _project_route_result


def _job() -> dict:
    opportunity = {
        "assumption_x": "state remains fixed during serving",
        "observation_y": "retrieval quality falls",
        "condition_z": "under long traces",
        "failure_f": "eviction causes retrieval loss",
        "missing_capability_w": "version-aware invalidation",
        "alternative_explanation_a": "measurement noise",
        "decisive_experiment": "the supplied paired trace test",
        "scope_compatibility": "same serving scope",
        "inference_flags": [],
    }
    return {
        "job_id": "job-1", "opportunity_id": "opp-1", "route_job_sha256": "1" * 64,
        "prompt_sha256": "2" * 64, "recon_ready_record_sha256": "3" * 64,
        "opportunity": opportunity, "opportunity_sha256": _hash(opportunity),
        "decision": "NO_DIRECT_COVERAGE_FOUND", "residual": {
            "decision": "NO_DIRECT_COVERAGE_FOUND", "decision_reason": "bounded search",
            "summary": "version-aware invalidation", "nearest_prior_residuals": [],
        },
        "scope_constraints": {"condition_z": opportunity["condition_z"], "scope_compatibility": opportunity["scope_compatibility"], "decisive_experiment": opportunity["decisive_experiment"]},
        "resource_constraints": {"data_access": "PUBLIC_BOUND_ARTIFACTS_ONLY", "filesystem_access": "RUN_LOCAL_READ_WRITE_ONLY", "compute_budget": "SINGLE_SEED_SMOKE_ONLY"},
        "nearest_prior_bindings": [], "allowed_resources": ["BOUND_INPUT_ARTIFACTS", "SINGLE_SEED_SMOKE_COMPUTE"],
        "resource_budget_description": "one bounded single-seed smoke run over public bound artifacts in the run-local workspace",
        "route_ids": [f"route-{i}" for i in range(1, 5)],
        "mechanism_fingerprint_axes": ["intervention_site", "operation", "target_state", "learning_signal", "state_representation"],
        "readiness_scope": "TEST_ONLY", "live_ready": False, "attestation_scope": "TEST_ONLY",
        "authenticity": "NOT_AUTHENTICATED", "authenticity_boundary": "test", "operator_attested_execution_ready": False,
    }


def _hash(value: object) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _payload(job: dict, n: int = 3) -> dict:
    o = job["opportunity"]
    routes = []
    for i in range(n):
        spec = {"schema_version": "idea_factory.mechanism_spec.v2", "intervention_site": f"site {i}", "operation": f"operation {i}", "target_state": f"target state {i}", "learning_signal": f"signal {i}", "state_representation": f"representation {i}"}
        gain = f"gain mechanism {i}"
        routes.append({"schema_version": "idea_factory.route_proposal.v2", "route_id": job["route_ids"][i], "route_index": i + 1, "opportunity_id": job["opportunity_id"], "opportunity_sha256": job["opportunity_sha256"], "frozen_opportunity": o, "scope_constraints": job["scope_constraints"], "resource_constraints": job["resource_constraints"], "nearest_prior_bindings": [], "residual": job["residual"], "old_assumption_changed": o["assumption_x"], "new_assumption": f"{o['condition_z']} uses {gain}", "new_mechanism": f"free mechanism explanation {i}", "mechanism_spec": spec, "why_it_addresses_failure": f"addresses {o['failure_f']}", "why_nearest_priors_cannot": f"residual {job['residual']['summary']}", "source_of_gain": f"{gain} from {o['missing_capability_w']} rather than {o['alternative_explanation_a']}", "causal_gain_hypothesis": gain, "required_resources": ["BOUND_INPUT_ARTIFACTS"], "cheapest_decisive_test": {"setup": f"{o['decisive_experiment']} {o['condition_z']}", "discriminates_against": o["alternative_explanation_a"], "expected_runtime_or_cost": job["resource_budget_description"]}, "kill_condition": f"kill if {o['failure_f']} persists"})
    return {"schema_version": "idea_factory.route_generator_result.v2", "job_id": job["job_id"], "opportunity_id": job["opportunity_id"], "route_job_sha256": job["route_job_sha256"], "prompt_sha256": job["prompt_sha256"], "recon_ready_record_sha256": job["recon_ready_record_sha256"], "routes": routes}


@pytest.mark.parametrize("n", [1, 2, 3])
def test_open_route_count_and_actual_ids(n: int) -> None:
    job = _job(); records, outcome = _project_route_result(job, json.dumps(_payload(job, n)))
    assert len(records) == n and outcome["route_ids"] == job["route_ids"][:n]


@pytest.mark.parametrize("n", [0, 4])
def test_route_count_outside_range_rejected(n: int) -> None:
    job = _job()
    payload = _payload(job, n)
    with pytest.raises(ValidationError, match="too_(short|long)"): _project_route_result(job, json.dumps(payload))


def test_noncanonical_open_facets_and_mechanism_are_accepted() -> None:
    job = _job(); payload = _payload(job, 1); route = payload["routes"][0]
    route["new_mechanism"] = "A free explanatory intervention not in any catalog"
    route["mechanism_spec"]["operation"] = "a domain-specific operator with two phases"
    assert _project_route_result(job, json.dumps(payload))[1]["status"] == "ACCEPTED"


@pytest.mark.parametrize("facet", ["intervention_site", "operation", "target_state", "learning_signal", "state_representation"])
def test_blank_facet_rejected(facet: str) -> None:
    job = _job(); payload = _payload(job, 1); payload["routes"][0]["mechanism_spec"][facet] = "  "
    with pytest.raises(ValidationError): _project_route_result(job, json.dumps(payload))


def test_normalized_spec_clone_rejected() -> None:
    job = _job(); payload = _payload(job, 2)
    for facet, value in list(payload["routes"][0]["mechanism_spec"].items()):
        if facet != "schema_version":
            payload["routes"][1]["mechanism_spec"][facet] = "  " + value.upper().replace(" ", "   ") + "  "
    with pytest.raises(ValueError, match="clone|distinct"): _project_route_result(job, json.dumps(payload))


@pytest.mark.parametrize("field", ["new_mechanism", "causal_gain_hypothesis", "source_of_gain"])
def test_normalized_content_clones_rejected(field: str) -> None:
    job = _job(); payload = _payload(job, 2)
    first = payload["routes"][0]; second = payload["routes"][1]
    if field == "new_mechanism":
        second["new_mechanism"] = "  FREE   MECHANISM   EXPLANATION 0 "
    elif field == "causal_gain_hypothesis":
        second["causal_gain_hypothesis"] = first["causal_gain_hypothesis"].upper()
        second["source_of_gain"] = f"{second['causal_gain_hypothesis']} from {job['opportunity']['missing_capability_w']} rather than {job['opportunity']['alternative_explanation_a']}"
    else:
        second["source_of_gain"] = first["source_of_gain"].upper()
        second["causal_gain_hypothesis"] = first["causal_gain_hypothesis"]
    with pytest.raises(ValueError, match="distinct|clone"): _project_route_result(job, json.dumps(payload))


def test_global_claim_in_free_facet_and_secret_rejected() -> None:
    job = _job(); payload = _payload(job, 1); payload["routes"][0]["mechanism_spec"]["operation"] = "world-first mechanism"
    with pytest.raises(ValueError, match="global-priority"): _project_route_result(job, json.dumps(payload))
    payload = _payload(job, 1); payload["routes"][0]["mechanism_spec"]["operation"] = "api_key=" + "sk-" + "X" * 30
    with pytest.raises(ValueError, match="secret"): _project_route_result(job, json.dumps(payload))


@pytest.mark.parametrize("mutation", ["scope_constraints", "nearest_prior_bindings", "residual", "budget", "alternative", "id", "old_schema"])
def test_stale_or_tampered_bindings_rejected(mutation: str) -> None:
    job = _job(); payload = _payload(job, 1); route = payload["routes"][0]
    if mutation == "scope_constraints": route["scope_constraints"] = dict(route["scope_constraints"], condition_z="other scope")
    elif mutation == "nearest_prior_bindings": route["nearest_prior_bindings"] = [{"paper": "x", "exact_overlap": "x", "residual_difference": "x", "evidence_url_or_id": "x", "evidence_ids_and_aliases": ["x"]}]
    elif mutation == "residual": route["residual"] = dict(route["residual"], summary="tampered residual")
    elif mutation == "budget": route["cheapest_decisive_test"] = dict(route["cheapest_decisive_test"], expected_runtime_or_cost="unlimited")
    elif mutation == "alternative": route["cheapest_decisive_test"] = dict(route["cheapest_decisive_test"], discriminates_against="another explanation")
    elif mutation == "id": route["route_id"] = "route-forged"
    else: payload["schema_version"] = "idea_factory.route_generator_result.v1"
    with pytest.raises(Exception): _project_route_result(job, json.dumps(payload))


def test_verified_frozen_quote_allowed_but_substring_and_forged_quote_are_not() -> None:
    job = _job(); job["opportunity"]["failure_f"] = "world-first method"; job["opportunity_sha256"] = _hash(job["opportunity"])
    payload = _payload(job, 1); payload["routes"][0]["new_mechanism"] = "world-first method"
    assert _project_route_result(job, json.dumps(payload))[1]["status"] == "ACCEPTED"
    job = _job(); job["opportunity"]["observation_y"] = "ever attempted"; job["opportunity_sha256"] = _hash(job["opportunity"]); payload = _payload(job, 1); payload["routes"][0]["new_mechanism"] = "never attempted"
    with pytest.raises(ValueError, match="global-priority"): _project_route_result(job, json.dumps(payload))
    payload = _payload(_job(), 1); payload["routes"][0]["frozen_opportunity"] = {"failure_f": "world-first method"}; payload["routes"][0]["new_mechanism"] = "world-first method"
    with pytest.raises(ValueError, match="global-priority"): _project_route_result(_job(), json.dumps(payload))
