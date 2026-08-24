"""Task-10 route generation from replay-validated Task-9 survivors only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator

from .artifacts import read_jsonl, sha256_file, stable_id
from .corpus import CorpusRouterConfig
from .models import FrozenStrictModel, NonEmptyStr
from .opportunities import _anchored_run, _owned_dir, _owned_file
from .recon import validate_execution_context, validate_query_pack, validate_recon_bundle, validate_report_jobs
from .safety import contains_global_priority_claim, contains_secret, normalize_free_text
from .stage_io import encode_json, encode_jsonl, publish_transaction, stage_mutation_lock


ROUTE_POLICY_VERSION = "idea_factory.route_policy.v1"
ROUTE_PROMPT_VERSION = "idea_factory.route_generator_prompt.v1"
ROUTE_JOB_SCHEMA_VERSION = "idea_factory.route_job.v1"
ROUTE_RESULT_SCHEMA_VERSION = "idea_factory.route_generator_result.v1"
ROUTE_PROPOSAL_SCHEMA_VERSION = "idea_factory.route_proposal.v1"
ROUTE_RECORD_SCHEMA_VERSION = "idea_factory.route_result.v1"
ROUTE_OUTCOME_SCHEMA_VERSION = "idea_factory.route_ingestion_outcome.v1"
ROUTE_MANIFEST_SCHEMA_VERSION = "idea_factory.route_bundle_manifest.v1"
_OUTPUTS = ("results.jsonl", "outcomes.jsonl", "bundle_manifest.json")
_ALLOWED_DECISIONS = {"NEAR_PRIOR_WITH_RESIDUAL", "NO_DIRECT_COVERAGE_FOUND"}
_DEFAULT_PROMPT = Path(__file__).resolve().parents[2] / "prompts" / "route_generator.md"

INTERVENTION_SITES = ("OBSERVATION_BOUNDARY", "STATE_TRANSITION", "OUTPUT_REPAIR")
OPERATIONS = ("MASK", "GATE", "TRANSFORM")
TARGET_STATES = ("OBSERVED_STATE", "LATENT_STATE", "DEPENDENCY_STATE")
LEARNING_SIGNALS = ("COUNTERFACTUAL_DELTA", "HELD_OUT_REGRET", "RECOVERY_DELTA")
STATE_REPRESENTATIONS = ("VERSIONED_STATE", "EVIDENCE_VECTOR", "DEPENDENCY_GRAPH")
ALLOWED_RESOURCES = ("BOUND_INPUT_ARTIFACTS", "SINGLE_SEED_SMOKE_COMPUTE")
RESOURCE_CONSTRAINTS = {
    "data_access": "PUBLIC_BOUND_ARTIFACTS_ONLY",
    "filesystem_access": "RUN_LOCAL_READ_WRITE_ONLY",
    "compute_budget": "SINGLE_SEED_SMOKE_ONLY",
}
RESOURCE_BUDGET_DESCRIPTION = "one bounded single-seed smoke run over public bound artifacts in the run-local workspace"


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _strict_load(raw: str | bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if type(text) is not str:
            raise TypeError("route result must be JSON text or bytes")
        value = json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("route result must be strict native JSON") from exc
    if type(value) is not dict:
        raise ValueError("route result must be one JSON object")
    return value


def _prompt_binding(prompt_path: Path | None) -> dict[str, str]:
    candidate = Path(os.path.abspath(prompt_path or _DEFAULT_PROMPT))
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError("route prompt must be a readable UTF-8 file") from exc
    if not candidate.is_file() or resolved != candidate:
        raise ValueError("route prompt source must have exact resolved identity")
    try:
        text = candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("route prompt must be UTF-8") from exc
    if not text.strip():
        raise ValueError("route prompt must be nonempty")
    return {
        "prompt_id": "route_generator",
        "prompt_version": ROUTE_PROMPT_VERSION,
        "prompt_source": str(candidate),
        "prompt_sha256": sha256_file(candidate),
        "prompt_text": text,
    }


def _routes_dir(run: Path, *, create: bool) -> Path:
    return _owned_dir(run, "routes", create=create)


def _truth_scope(run: Path, *, has_ready: bool, allow_test_ready: bool) -> tuple[str, bool, dict[str, Any]]:
    if not has_ready:
        return "NO_READY_RECORDS", False, {
            "attestation_scope": "NO_ATTESTATION",
            "authenticity": "NOT_AUTHENTICATED",
            "authenticity_boundary": "No ready record or execution attestation is claimed.",
            "operator_attested_execution_ready": False,
        }
    context = validate_execution_context(run)
    attestation_scope = str(context["operator_trust_scope"])
    if has_ready and attestation_scope == "TEST_ONLY" and not allow_test_ready:
        raise ValueError("TEST_ONLY reconnaissance cannot enter live-ready routes")
    if attestation_scope not in {"TEST_ONLY", "LIVE_OPERATOR_ATTESTED"}:
        raise ValueError("reconnaissance truth scope is invalid")
    truth = {
        "attestation_scope": attestation_scope,
        "authenticity": context["authenticity"],
        "authenticity_boundary": context["claim_boundary"],
        "operator_attested_execution_ready": has_ready and attestation_scope == "LIVE_OPERATOR_ATTESTED",
    }
    return (attestation_scope if has_ready else "NO_READY_RECORDS"), bool(truth["operator_attested_execution_ready"]), truth


def _opportunities_by_id(run: Path, config: CorpusRouterConfig) -> dict[str, dict[str, Any]]:
    pack = validate_query_pack(run, config)
    grouped: dict[str, dict[str, Any]] = {}
    for query in pack.queries:
        opportunity_id = str(query["opportunity_id"])
        opportunity = query.get("opportunity")
        if type(opportunity) is not dict:
            raise ValueError("recon query lacks exact opportunity binding")
        previous = grouped.setdefault(opportunity_id, opportunity)
        if previous != opportunity:
            raise ValueError("recon queries disagree on opportunity binding")
    return grouped


def _nearest_prior_bindings(priors: Sequence[Mapping[str, Any]], report_job: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence = report_job.get("evidence")
    if type(evidence) is not list:
        raise ValueError("recon report job evidence projection is invalid")
    bindings: list[dict[str, Any]] = []
    for prior in priors:
        reference = str(prior["evidence_url_or_id"])
        matched = next(
            (
                row
                for row in evidence
                if reference in {
                    str(row.get("evidence_id", "")),
                    str(row.get("source_id", "")),
                    str(row.get("url", "")),
                    str(row.get("doi", "")),
                    *[str(alias) for alias in row.get("source_aliases", [])],
                }
            ),
            None,
        )
        if matched is None:
            raise ValueError("nearest prior lacks a replayed evidence binding")
        aliases = sorted(
            {
                str(value)
                for value in (
                    matched.get("evidence_id"), matched.get("source_id"), matched.get("url"), matched.get("doi"),
                    *matched.get("source_aliases", []),
                )
                if value
            }
        )
        bindings.append(
            {
                "paper": prior["paper"],
                "exact_overlap": prior["exact_overlap"],
                "residual_difference": prior["residual_difference"],
                "evidence_url_or_id": reference,
                "evidence_ids_and_aliases": aliases,
            }
        )
    return bindings


def _route_job_projection(
    run: Path,
    config: CorpusRouterConfig,
    prompt: Mapping[str, str],
    *,
    allow_test_ready: bool,
) -> tuple[list[dict[str, Any]], str, bool]:
    recon = validate_recon_bundle(run, config)
    ready = list(recon["ready"])
    scope, live_ready, truth = _truth_scope(run, has_ready=bool(ready), allow_test_ready=allow_test_ready)
    opportunities = _opportunities_by_id(run, config)
    report_jobs = {row["opportunity_id"]: row for row in validate_report_jobs(run, config)}
    jobs: list[dict[str, Any]] = []
    for record in sorted(ready, key=lambda row: str(row.get("opportunity_id", ""))):
        decision = record.get("decision")
        opportunity_id = str(record.get("opportunity_id", ""))
        if decision not in _ALLOWED_DECISIONS:
            raise ValueError("only residual or bounded no-coverage READY records may enter route generation")
        opportunity = opportunities.get(opportunity_id)
        report_job = report_jobs.get(opportunity_id)
        if opportunity is None or report_job is None:
            raise ValueError("READY record lacks replayed opportunity or recon provenance")
        opportunity_sha = _hash(opportunity)
        ready_sha = _hash(record)
        priors = record.get("nearest_priors")
        if type(priors) is not list:
            raise ValueError("READY record nearest priors are invalid")
        if decision == "NEAR_PRIOR_WITH_RESIDUAL" and not priors:
            raise ValueError("NEAR_PRIOR_WITH_RESIDUAL requires at least one exact prior")
        prior_bindings = _nearest_prior_bindings(priors, report_job)
        residual_parts = [str(row["residual_difference"]) for row in priors]
        residual_summary = " | ".join(residual_parts) if residual_parts else f"{opportunity['missing_capability_w']} under {opportunity['condition_z']}"
        residual = {
            "decision": decision,
            "decision_reason": record["decision_reason"],
            "summary": residual_summary,
            "nearest_prior_residuals": residual_parts,
        }
        scope_constraints = {
            "condition_z": opportunity["condition_z"],
            "scope_compatibility": opportunity["scope_compatibility"],
            "decisive_experiment": opportunity["decisive_experiment"],
        }
        result_schema = _RouteGeneratorResult.model_json_schema()
        recon_provenance = {
            "report_job_id": record["job_id"],
            "report_record_id": record["record_id"],
            "report_raw_result_sha256": record["raw_result_sha256"],
            "query_pack_sha256": report_job["query_pack_sha256"],
            "normalized_evidence_manifest_sha256": report_job["normalized_evidence_manifest_sha256"],
            "protocol_hash": report_job["protocol_hash"],
            "searched_query_ids": record["searched_query_ids"],
        }
        job_id = stable_id("route_job", opportunity_id, ready_sha, prompt["prompt_sha256"], ROUTE_POLICY_VERSION)
        route_ids = [stable_id("route", job_id, str(index)) for index in range(1, 4)]
        body = {
            "schema_version": ROUTE_JOB_SCHEMA_VERSION,
            "result_schema_version": ROUTE_RESULT_SCHEMA_VERSION,
            "result_schema": result_schema,
            "result_schema_sha256": _hash(result_schema),
            "policy_version": ROUTE_POLICY_VERSION,
            "job_id": job_id,
            "opportunity_id": opportunity_id,
            "decision": decision,
            "opportunity": opportunity,
            "opportunity_sha256": opportunity_sha,
            "evidence_flags": opportunity.get("inference_flags", []),
            "nearest_priors": priors,
            "nearest_prior_bindings": prior_bindings,
            "residual": residual,
            "scope_constraints": scope_constraints,
            "allowed_resources": list(ALLOWED_RESOURCES),
            "resource_constraints": RESOURCE_CONSTRAINTS,
            "resource_budget_description": RESOURCE_BUDGET_DESCRIPTION,
            "mechanism_vocabulary": {
                "intervention_site": list(INTERVENTION_SITES),
                "operation": list(OPERATIONS),
                "target_state": list(TARGET_STATES),
                "learning_signal": list(LEARNING_SIGNALS),
                "state_representation": list(STATE_REPRESENTATIONS),
            },
            "mechanism_grounding_claim": "STRUCTURED_AND_OPPORTUNITY_BOUND_ONLY",
            "recon_provenance": recon_provenance,
            "recon_ready_record_sha256": ready_sha,
            "readiness_scope": scope,
            "live_ready": live_ready,
            **truth,
            "route_ids": route_ids,
            "required_route_count": 3,
            "mechanism_fingerprint_axes": ["intervention_site", "operation", "target_state", "learning_signal", "state_representation"],
            "pairwise_minimum_distinct_axes": 2,
            **prompt,
        }
        body["route_job_sha256"] = _hash(body)
        jobs.append(body)
    return jobs, scope, live_ready


def emit_route_jobs(
    run_dir: Path,
    config: CorpusRouterConfig,
    *,
    prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> Path:
    run = _anchored_run(Path(run_dir))
    routes = _routes_dir(run, create=True)
    jobs_path = routes / "jobs.jsonl"
    with stage_mutation_lock(routes):
        prompt = _prompt_binding(prompt_path)
        jobs, _scope, _live = _route_job_projection(run, config, prompt, allow_test_ready=allow_test_ready)
        existing = [jobs_path.exists(), *[(routes / name).exists() for name in _OUTPUTS]]
        if any(existing):
            if not existing[0] or (any(existing[1:]) and not all(existing[1:])):
                raise ValueError("existing route stage artifacts are incomplete")
            if all(existing[1:]):
                validate_route_bundle(run, config, prompt_path=prompt_path, allow_test_ready=allow_test_ready)
            else:
                validate_route_jobs(run, config, prompt_path=prompt_path, allow_test_ready=allow_test_ready)
            if read_jsonl(_owned_file(routes, "jobs.jsonl")) != jobs:
                raise ValueError("existing route jobs are immutable under changed replay inputs")
            return jobs_path
        publish_transaction({jobs_path: encode_jsonl(jobs)})
        return jobs_path


def validate_route_jobs(
    run_dir: Path,
    config: CorpusRouterConfig,
    *,
    prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> tuple[dict[str, Any], ...]:
    run = _anchored_run(Path(run_dir))
    routes = _routes_dir(run, create=False)
    actual = read_jsonl(_owned_file(routes, "jobs.jsonl"))
    expected, _scope, _live = _route_job_projection(
        run, config, _prompt_binding(prompt_path), allow_test_ready=allow_test_ready
    )
    if actual != expected:
        raise ValueError("route jobs replay mismatch")
    return tuple(actual)


class MechanismSpec(FrozenStrictModel):
    schema_version: Literal["idea_factory.mechanism_spec.v1"]
    intervention_site: Literal["OBSERVATION_BOUNDARY", "STATE_TRANSITION", "OUTPUT_REPAIR"]
    operation: Literal["MASK", "GATE", "TRANSFORM"]
    target_state: Literal["OBSERVED_STATE", "LATENT_STATE", "DEPENDENCY_STATE"]
    learning_signal: Literal["COUNTERFACTUAL_DELTA", "HELD_OUT_REGRET", "RECOVERY_DELTA"]
    state_representation: Literal["VERSIONED_STATE", "EVIDENCE_VECTOR", "DEPENDENCY_GRAPH"]


def render_mechanism_spec(spec: MechanismSpec | Mapping[str, Any]) -> str:
    data = spec.model_dump(mode="json") if isinstance(spec, MechanismSpec) else dict(spec)
    return "mechanism_spec:" + ":".join(
        str(data[key])
        for key in ("intervention_site", "operation", "target_state", "learning_signal", "state_representation")
    )


class _ScopeConstraints(FrozenStrictModel):
    condition_z: NonEmptyStr
    scope_compatibility: NonEmptyStr
    decisive_experiment: NonEmptyStr


class _ResourceConstraints(FrozenStrictModel):
    data_access: Literal["PUBLIC_BOUND_ARTIFACTS_ONLY"]
    filesystem_access: Literal["RUN_LOCAL_READ_WRITE_ONLY"]
    compute_budget: Literal["SINGLE_SEED_SMOKE_ONLY"]


class _NearestPriorBinding(FrozenStrictModel):
    paper: NonEmptyStr
    exact_overlap: NonEmptyStr
    residual_difference: NonEmptyStr
    evidence_url_or_id: NonEmptyStr
    evidence_ids_and_aliases: list[NonEmptyStr] = Field(min_length=1)


class _ResidualBinding(FrozenStrictModel):
    decision: Literal["NEAR_PRIOR_WITH_RESIDUAL", "NO_DIRECT_COVERAGE_FOUND"]
    decision_reason: NonEmptyStr
    summary: NonEmptyStr
    nearest_prior_residuals: list[NonEmptyStr]


class _DecisiveTest(FrozenStrictModel):
    setup: NonEmptyStr
    discriminates_against: NonEmptyStr
    expected_runtime_or_cost: NonEmptyStr


class _RouteProposal(FrozenStrictModel):
    schema_version: Literal["idea_factory.route_proposal.v1"]
    route_id: NonEmptyStr
    route_index: int = Field(strict=True, ge=1, le=3)
    opportunity_id: NonEmptyStr
    opportunity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    frozen_opportunity: dict[str, Any]
    scope_constraints: _ScopeConstraints
    resource_constraints: _ResourceConstraints
    nearest_prior_bindings: list[_NearestPriorBinding]
    residual: _ResidualBinding
    old_assumption_changed: NonEmptyStr
    new_assumption: NonEmptyStr
    new_mechanism: NonEmptyStr
    mechanism_spec: MechanismSpec
    why_it_addresses_failure: NonEmptyStr
    why_nearest_priors_cannot: NonEmptyStr
    source_of_gain: NonEmptyStr
    causal_gain_hypothesis: NonEmptyStr
    required_resources: list[Literal["BOUND_INPUT_ARTIFACTS", "SINGLE_SEED_SMOKE_COMPUTE"]] = Field(min_length=1)
    cheapest_decisive_test: _DecisiveTest
    kill_condition: NonEmptyStr


class _RouteGeneratorResult(FrozenStrictModel):
    schema_version: Literal["idea_factory.route_generator_result.v1"]
    job_id: NonEmptyStr
    opportunity_id: NonEmptyStr
    route_job_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    recon_ready_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    routes: list[_RouteProposal] = Field(min_length=3, max_length=3)


def _contains(text: str, binding: str) -> bool:
    return binding.casefold() in text.casefold()


def _validate_route_semantics(item: _RouteGeneratorResult, job: Mapping[str, Any]) -> None:
    for key in ("job_id", "opportunity_id", "route_job_sha256", "prompt_sha256", "recon_ready_record_sha256"):
        if getattr(item, key) != job[key]:
            raise ValueError("route result binding mismatch")
    if len(item.routes) != 3:
        raise ValueError("route result must contain exactly three routes")
    opportunity = job["opportunity"]
    residual = job["residual"]["summary"]
    fingerprints: list[tuple[str, ...]] = []
    gain_fingerprints: list[str] = []
    causal_hypotheses: list[str] = []
    for index, route in enumerate(item.routes):
        if route.route_id != job["route_ids"][index] or route.route_index != index + 1:
            raise ValueError("route identity or canonical order mismatch")
        if route.opportunity_id != job["opportunity_id"] or route.opportunity_sha256 != job["opportunity_sha256"]:
            raise ValueError("route opportunity binding mismatch")
        if route.frozen_opportunity != opportunity or _hash(route.frozen_opportunity) != job["opportunity_sha256"]:
            raise ValueError("route frozen opportunity binding mismatch")
        if route.scope_constraints.model_dump(mode="json") != job["scope_constraints"]:
            raise ValueError("route scope constraints binding mismatch")
        if route.resource_constraints.model_dump(mode="json") != job["resource_constraints"]:
            raise ValueError("route resource constraints binding mismatch")
        if [prior.model_dump(mode="json") for prior in route.nearest_prior_bindings] != job["nearest_prior_bindings"]:
            raise ValueError("route nearest-prior evidence binding mismatch")
        if route.residual.model_dump(mode="json") != job["residual"]:
            raise ValueError("route frozen residual binding mismatch")
        if job["decision"] == "NEAR_PRIOR_WITH_RESIDUAL" and not route.nearest_prior_bindings:
            raise ValueError("near-prior route requires at least one exact prior")
        if route.old_assumption_changed != opportunity["assumption_x"]:
            raise ValueError("route changes the bound research question")
        if route.new_mechanism != render_mechanism_spec(route.mechanism_spec):
            raise ValueError("new mechanism is not the canonical MechanismSpec rendering")
        vocabulary = job["mechanism_vocabulary"]
        spec_data = route.mechanism_spec.model_dump(mode="json")
        if any(spec_data[axis] not in vocabulary[axis] for axis in job["mechanism_fingerprint_axes"]):
            raise ValueError("route mechanism uses an uncontrolled vocabulary value")
        if not _contains(route.source_of_gain, route.causal_gain_hypothesis):
            raise ValueError("route source-of-gain fingerprint is not bound to its causal gain hypothesis")
        required = (
            (route.new_assumption, opportunity["condition_z"]),
            (route.why_it_addresses_failure, opportunity["failure_f"]),
            (route.why_nearest_priors_cannot, residual),
            (route.source_of_gain, opportunity["missing_capability_w"]),
            (route.source_of_gain, opportunity["alternative_explanation_a"]),
            (route.cheapest_decisive_test.setup, opportunity["decisive_experiment"]),
            (route.cheapest_decisive_test.setup, opportunity["condition_z"]),
            (route.kill_condition, opportunity["failure_f"]),
        )
        if any(not _contains(text, binding) for text, binding in required):
            raise ValueError("route field is not bound to the exact opportunity")
        route_texts = (
            route.new_assumption,
            route.new_mechanism,
            route.why_it_addresses_failure,
            route.why_nearest_priors_cannot,
            route.source_of_gain,
            route.cheapest_decisive_test.setup,
            route.cheapest_decisive_test.discriminates_against,
            route.kill_condition,
        )
        if route.cheapest_decisive_test.discriminates_against != opportunity["alternative_explanation_a"]:
            raise ValueError("route decisive test changes the alternative explanation")
        if route.cheapest_decisive_test.expected_runtime_or_cost != job["resource_budget_description"]:
            raise ValueError("route decisive test changes the authorized resource budget")
        if len(set(route.required_resources)) != len(route.required_resources) or not set(route.required_resources).issubset(set(job["allowed_resources"])):
            raise ValueError("route requests a resource outside the job allowlist")
        fingerprint = tuple(
            str(getattr(route.mechanism_spec, axis))
            for axis in job["mechanism_fingerprint_axes"]
        )
        fingerprints.append(fingerprint)
        causal_hypotheses.append(normalize_free_text(route.causal_gain_hypothesis))
        gain_fingerprints.append(_hash(normalize_free_text(route.source_of_gain)))
        if contains_global_priority_claim(route_texts):
            raise ValueError("route generator makes a forbidden global-priority claim")
    for left_index, left in enumerate(fingerprints):
        for right in fingerprints[left_index + 1 :]:
            if sum(a != b for a, b in zip(left, right)) < int(job["pairwise_minimum_distinct_axes"]):
                raise ValueError("route mechanism fingerprints are not substantively distinct")
    if len(set(causal_hypotheses)) != len(causal_hypotheses):
        raise ValueError("route causal gain hypotheses are not substantively distinct")
    if len(set(gain_fingerprints)) != len(gain_fingerprints):
        raise ValueError("route source-of-gain fingerprints are not substantively distinct")


def _model_authored_route_text(payload: Mapping[str, Any]) -> dict[str, Any]:
    copied_top = {
        "schema_version", "job_id", "opportunity_id", "route_job_sha256", "prompt_sha256",
        "recon_ready_record_sha256", "routes",
    }
    projection = {key: value for key, value in payload.items() if key not in copied_top}
    copied_route = {
        "schema_version", "route_id", "route_index", "opportunity_id", "opportunity_sha256",
        "frozen_opportunity", "scope_constraints", "resource_constraints", "nearest_prior_bindings",
        "residual", "old_assumption_changed", "mechanism_spec", "required_resources",
        "cheapest_decisive_test",
    }
    authored_routes: list[dict[str, Any]] = []
    routes = payload.get("routes")
    if type(routes) is list:
        for route in routes:
            if type(route) is not dict:
                authored_routes.append({"unexpected_route": route})
                continue
            authored = {key: value for key, value in route.items() if key not in copied_route}
            test = route.get("cheapest_decisive_test")
            if type(test) is dict:
                authored["cheapest_decisive_test"] = {
                    key: value
                    for key, value in test.items()
                    if key not in {"discriminates_against", "expected_runtime_or_cost"}
                }
            authored_routes.append(authored)
    projection["routes"] = authored_routes
    return projection


def _project_route_result(job: Mapping[str, Any], raw: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    payload = _strict_load(raw)
    if contains_secret(payload):
        raise ValueError("route payload contains a credential-shaped secret")
    if contains_global_priority_claim(_model_authored_route_text(payload)):
        raise ValueError("route payload contains a forbidden global-priority claim")
    item = _RouteGeneratorResult.model_validate(payload, strict=True)
    _validate_route_semantics(item, job)
    records: list[dict[str, Any]] = []
    for route in item.routes:
        data = route.model_dump(mode="json")
        data.pop("schema_version")
        mechanism_sha = _hash(data["mechanism_spec"])
        records.append(
            {
                "schema_version": ROUTE_RECORD_SCHEMA_VERSION,
                "record_id": route.route_id,
                "job_id": job["job_id"],
                "raw_result_sha256": raw_sha,
                "route_job_sha256": job["route_job_sha256"],
                "recon_ready_record_sha256": job["recon_ready_record_sha256"],
                "readiness_scope": job["readiness_scope"],
                "live_ready": job["live_ready"],
                "attestation_scope": job["attestation_scope"],
                "authenticity": job["authenticity"],
                "authenticity_boundary": job["authenticity_boundary"],
                "operator_attested_execution_ready": job["operator_attested_execution_ready"],
                "mechanism_fingerprint": mechanism_sha,
                "mechanism_spec_sha256": mechanism_sha,
                **data,
            }
        )
    outcome = {
        "schema_version": ROUTE_OUTCOME_SCHEMA_VERSION,
        "job_id": job["job_id"],
        "opportunity_id": job["opportunity_id"],
        "status": "ACCEPTED",
        "raw_result_json": raw,
        "raw_result_sha256": raw_sha,
        "route_ids": list(job["route_ids"]),
        "accepted_projection_sha256": _hash(records),
    }
    return records, outcome


def _rejection(job: Mapping[str, Any], raw_sha: str, exc: Exception) -> dict[str, Any]:
    text = str(exc)
    if "credential-shaped secret" in text:
        code, reason = "ROUTE_SECRET_DETECTED", "route result contains credential-shaped content"
    elif "global-priority" in text:
        code, reason = "ROUTE_GLOBAL_PRIORITY_CLAIM_FORBIDDEN", "route generator cannot make global-priority claims"
    elif "distinct" in text or "fingerprint" in text:
        code, reason = "ROUTE_DISTINCTNESS_INVALID", "route mechanisms are not conservatively distinct"
    elif "exactly three" in text:
        code, reason = "ROUTE_COUNT_INVALID", "route result must contain exactly three routes"
    elif "binding" in text or "research question" in text or "alternative" in text:
        code, reason = "ROUTE_BINDING_INVALID", "route result changes or loses an exact opportunity binding"
    else:
        code, reason = "ROUTE_SCHEMA_INVALID", "route result schema is invalid"
    return {
        "schema_version": ROUTE_OUTCOME_SCHEMA_VERSION,
        "job_id": job["job_id"],
        "opportunity_id": job["opportunity_id"],
        "status": "REJECTED",
        "raw_result_sha256": raw_sha,
        "error_code": code,
        "error": reason,
        "route_ids": [],
        "accepted_projection_sha256": _hash([]),
    }


def _manifest(
    routes: Path,
    jobs: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    artifact_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    accepted_jobs = sum(row.get("status") == "ACCEPTED" for row in outcomes)
    rejected_jobs = len(outcomes) - accepted_jobs
    context = validate_execution_context(routes.parent) if jobs else None
    body = {
        "schema_version": ROUTE_MANIFEST_SCHEMA_VERSION,
        "policy_version": ROUTE_POLICY_VERSION,
        "prompt_version": jobs[0]["prompt_version"] if jobs else ROUTE_PROMPT_VERSION,
        "prompt_source": jobs[0]["prompt_source"] if jobs else str(_DEFAULT_PROMPT),
        "prompt_sha256": jobs[0]["prompt_sha256"] if jobs else sha256_file(_DEFAULT_PROMPT),
        "jobs_sha256": artifact_hashes["jobs"] if artifact_hashes else sha256_file(routes / "jobs.jsonl"),
        "results_sha256": artifact_hashes["results"] if artifact_hashes else sha256_file(routes / "results.jsonl"),
        "outcomes_sha256": artifact_hashes["outcomes"] if artifact_hashes else sha256_file(routes / "outcomes.jsonl"),
        "job_count": len(jobs),
        "accepted_job_count": accepted_jobs,
        "rejected_job_count": rejected_jobs,
        "accepted_route_count": len(results),
        "readiness_scopes": sorted({str(job["readiness_scope"]) for job in jobs}),
        "live_ready": bool(jobs) and all(bool(job["live_ready"]) for job in jobs),
        "attestation_scopes": sorted({str(job["attestation_scope"]) for job in jobs}),
        "authenticity": context["authenticity"] if context else "NOT_AUTHENTICATED",
        "authenticity_boundary": context["claim_boundary"] if context else "No ready record or execution attestation is claimed.",
        "operator_attested_execution_ready": bool(jobs)
        and all(bool(job["operator_attested_execution_ready"]) for job in jobs),
        "independently_verified": False,
        "result_schema_sha256s": sorted(
            {str(job["result_schema_sha256"]) for job in jobs}
            or {_hash(_RouteGeneratorResult.model_json_schema())}
        ),
    }
    return body | {"bundle_manifest_sha256": _hash(body)}


def _raw_text(source: str | bytes | Path) -> str:
    raw = source.read_bytes() if isinstance(source, Path) else source.encode("utf-8") if type(source) is str else source
    if not isinstance(raw, bytes):
        raise TypeError("route result must be text, bytes, or a path")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("route result must be UTF-8") from exc


def _validate_accepted_route_replays(
    current: Mapping[str, Any] | None,
    accepted: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> None:
    if current is None:
        return
    incoming_outcomes = {str(row["job_id"]): row for row in outcomes}
    incoming_records: dict[str, list[Mapping[str, Any]]] = {}
    for record in accepted:
        incoming_records.setdefault(str(record["job_id"]), []).append(record)
    current_records: dict[str, list[Mapping[str, Any]]] = {}
    for record in current["results"]:
        current_records.setdefault(str(record["job_id"]), []).append(record)
    for old in current["outcomes"]:
        if old.get("status") != "ACCEPTED":
            continue
        job_id = str(old["job_id"])
        new = incoming_outcomes.get(job_id)
        if (
            new is None
            or new.get("status") != "ACCEPTED"
            or new.get("raw_result_sha256") != old.get("raw_result_sha256")
            or new.get("accepted_projection_sha256") != old.get("accepted_projection_sha256")
            or incoming_records.get(job_id) != current_records.get(job_id)
        ):
            raise ValueError("accepted route job is immutable; only exact idempotent replay is allowed")


def ingest_route_results(
    jobs_path: Path,
    results: Sequence[str | bytes | Path],
    config: CorpusRouterConfig,
    *,
    prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> dict[str, Path]:
    run = _anchored_run(Path(jobs_path).parent.parent)
    routes = _routes_dir(run, create=False)
    with stage_mutation_lock(routes):
        if Path(os.path.abspath(jobs_path)) != _owned_file(routes, "jobs.jsonl"):
            raise ValueError("route jobs must use the active run path")
        jobs = validate_route_jobs(run, config, prompt_path=prompt_path, allow_test_ready=allow_test_ready)
        output_exists = [(routes / name).exists() for name in _OUTPUTS]
        if any(output_exists) and not all(output_exists):
            raise ValueError("existing route bundle is incomplete")
        current = (
            validate_route_bundle(run, config, prompt_path=prompt_path, allow_test_ready=allow_test_ready)
            if all(output_exists)
            else None
        )
        if len(results) != len(jobs):
            raise ValueError("route results must exactly cover route jobs in canonical order")
        accepted: list[dict[str, Any]] = []
        outcomes: list[dict[str, Any]] = []
        for job, source in zip(jobs, results):
            raw = _raw_text(source)
            raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            try:
                records, outcome = _project_route_result(job, raw)
            except (ValidationError, ValueError, TypeError) as exc:
                records, outcome = [], _rejection(job, raw_sha, exc)
            accepted.extend(records)
            outcomes.append(outcome)
        _validate_accepted_route_replays(current, accepted, outcomes)
        paths = {
            "results": routes / "results.jsonl",
            "outcomes": routes / "outcomes.jsonl",
            "manifest": routes / "bundle_manifest.json",
        }
        results_bytes = encode_jsonl(accepted)
        outcomes_bytes = encode_jsonl(outcomes)
        hashes = {
            "jobs": sha256_file(routes / "jobs.jsonl"),
            "results": hashlib.sha256(results_bytes).hexdigest(),
            "outcomes": hashlib.sha256(outcomes_bytes).hexdigest(),
        }
        manifest_bytes = encode_json(_manifest(routes, jobs, accepted, outcomes, hashes))
        publish_transaction(
            {paths["results"]: results_bytes, paths["outcomes"]: outcomes_bytes, paths["manifest"]: manifest_bytes}
        )
        return paths


_REJECTION_PAIRS = {
    ("ROUTE_DISTINCTNESS_INVALID", "route mechanisms are not conservatively distinct"),
    ("ROUTE_COUNT_INVALID", "route result must contain exactly three routes"),
    ("ROUTE_BINDING_INVALID", "route result changes or loses an exact opportunity binding"),
    ("ROUTE_SCHEMA_INVALID", "route result schema is invalid"),
    ("ROUTE_GLOBAL_PRIORITY_CLAIM_FORBIDDEN", "route generator cannot make global-priority claims"),
    ("ROUTE_SECRET_DETECTED", "route result contains credential-shaped content"),
}


def validate_route_bundle(
    run_dir: Path,
    config: CorpusRouterConfig,
    *,
    prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> dict[str, Any]:
    run = _anchored_run(Path(run_dir))
    routes = _routes_dir(run, create=False)
    jobs = validate_route_jobs(run, config, prompt_path=prompt_path, allow_test_ready=allow_test_ready)
    actual_results = read_jsonl(_owned_file(routes, "results.jsonl"))
    actual_outcomes = read_jsonl(_owned_file(routes, "outcomes.jsonl"))
    try:
        actual_manifest = _strict_load(_owned_file(routes, "bundle_manifest.json").read_bytes())
    except ValueError as exc:
        raise ValueError("route bundle manifest is invalid") from exc
    if len(actual_outcomes) != len(jobs):
        raise ValueError("route outcomes must exactly cover jobs")
    expected_results: list[dict[str, Any]] = []
    expected_outcomes: list[dict[str, Any]] = []
    for job, outcome in zip(jobs, actual_outcomes):
        if outcome.get("status") == "ACCEPTED":
            raw = outcome.get("raw_result_json")
            if type(raw) is not str or hashlib.sha256(raw.encode("utf-8")).hexdigest() != outcome.get("raw_result_sha256"):
                raise ValueError("route accepted outcome raw replay binding mismatch")
            records, replayed = _project_route_result(job, raw)
            if outcome != replayed:
                raise ValueError("route accepted outcome replay mismatch")
            expected_results.extend(records); expected_outcomes.append(replayed)
            continue
        keys = {"schema_version", "job_id", "opportunity_id", "status", "raw_result_sha256", "error_code", "error", "route_ids", "accepted_projection_sha256"}
        if set(outcome) != keys or outcome.get("schema_version") != ROUTE_OUTCOME_SCHEMA_VERSION or outcome.get("status") != "REJECTED":
            raise ValueError("route rejected outcome schema mismatch")
        if (outcome.get("error_code"), outcome.get("error")) not in _REJECTION_PAIRS:
            raise ValueError("route rejected outcome reason is not bounded")
        if not re.fullmatch(r"[0-9a-f]{64}", str(outcome.get("raw_result_sha256", ""))) or outcome.get("route_ids") != [] or outcome.get("accepted_projection_sha256") != _hash([]):
            raise ValueError("route rejected outcome binding mismatch")
        if outcome.get("job_id") != job["job_id"] or outcome.get("opportunity_id") != job["opportunity_id"]:
            raise ValueError("route rejected outcome job binding mismatch")
        expected_outcomes.append(outcome)
    if actual_results != expected_results or actual_outcomes != expected_outcomes:
        raise ValueError("route accepted results replay mismatch")
    expected_manifest = _manifest(routes, jobs, expected_results, expected_outcomes)
    if actual_manifest != expected_manifest:
        raise ValueError("route bundle manifest replay mismatch")
    return {
        "results": tuple(actual_results),
        "outcomes": tuple(actual_outcomes),
        "manifest": actual_manifest,
    }
