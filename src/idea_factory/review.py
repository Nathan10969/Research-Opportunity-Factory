"""Task-10 bounded one- or two-round review of accepted route proposals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import Field, ValidationError

from .artifacts import read_jsonl, sha256_file, stable_id
from .corpus import CorpusRouterConfig
from .models import FrozenStrictModel, NonEmptyStr
from .opportunities import _anchored_run, _owned_dir, _owned_file
from .routes import validate_route_bundle, validate_route_jobs
from .safety import contains_global_priority_claim, contains_secret
from .stage_io import encode_json, encode_jsonl, publish_transaction, stage_mutation_lock


REVIEW_POLICY_VERSION = "idea_factory.review_policy.v1"
REVIEW_PROMPT_VERSION = "idea_factory.reviewer_prompt.v1"
REVIEW_JOB_SCHEMA_VERSION = "idea_factory.review_job.v1"
REVIEW_RESULT_SCHEMA_VERSION = "idea_factory.reviewer_result.v1"
REVIEW_RECORD_SCHEMA_VERSION = "idea_factory.review_result.v1"
REVIEW_OUTCOME_SCHEMA_VERSION = "idea_factory.review_ingestion_outcome.v1"
REVIEW_MANIFEST_SCHEMA_VERSION = "idea_factory.review_bundle_manifest.v1"
_DEFAULT_PROMPT = Path(__file__).resolve().parents[2] / "prompts" / "reviewer.md"
_OUTPUTS = ("results.jsonl", "outcomes.jsonl", "bundle_manifest.json")
_ROUND_ONE_DECISIONS = {"KILL", "NARROW", "PASS_TO_HUMAN"}
_ROUND_TWO_DECISIONS = {"KILL", "PASS_TO_HUMAN"}

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
            raise TypeError("review result must be JSON text or bytes")
        value = json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("review result must be strict native JSON") from exc
    if type(value) is not dict:
        raise ValueError("review result must be one JSON object")
    return value


def _review_dir(run: Path, *, create: bool) -> Path:
    return _owned_dir(run, "review", create=create)


def _prompt_binding(prompt_path: Path | None) -> dict[str, str]:
    candidate = Path(os.path.abspath(prompt_path or _DEFAULT_PROMPT))
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError("reviewer prompt must be a readable UTF-8 file") from exc
    if not candidate.is_file() or resolved != candidate:
        raise ValueError("reviewer prompt source must have exact resolved identity")
    try:
        text = candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("reviewer prompt must be UTF-8") from exc
    if not text.strip():
        raise ValueError("reviewer prompt must be nonempty")
    return {
        "prompt_id": "reviewer",
        "prompt_version": REVIEW_PROMPT_VERSION,
        "prompt_source": str(candidate),
        "prompt_sha256": sha256_file(candidate),
        "prompt_text": text,
    }


def _round_one_jobs(
    run: Path,
    config: CorpusRouterConfig,
    prompt: Mapping[str, str],
    *,
    route_prompt_path: Path | None,
    allow_test_ready: bool,
) -> list[dict[str, Any]]:
    route_bundle = validate_route_bundle(
        run, config, prompt_path=route_prompt_path, allow_test_ready=allow_test_ready
    )
    route_jobs = {
        row["job_id"]: row
        for row in validate_route_jobs(
            run, config, prompt_path=route_prompt_path, allow_test_ready=allow_test_ready
        )
    }
    route_manifest = route_bundle["manifest"]
    jobs: list[dict[str, Any]] = []
    for route in route_bundle["results"]:
        parent = route_jobs.get(route["job_id"])
        if parent is None or route["opportunity_id"] != parent["opportunity_id"]:
            raise ValueError("route result lacks its exact opportunity job")
        route_sha = _hash(route)
        mechanism_spec = route["mechanism_spec"]
        mechanism_component_ids = [
            mechanism_spec[key]
            for key in ("intervention_site", "operation", "target_state", "learning_signal", "state_representation")
        ]
        allowed_evidence = set(parent["opportunity"].get("supporting_card_ids", []))
        for prior in parent["nearest_prior_bindings"]:
            allowed_evidence.update(prior["evidence_ids_and_aliases"])
        if not allowed_evidence:
            raise ValueError("review route has no evidence-bound baseline ID")
        result_schema = _ReviewerResult.model_json_schema()
        job_id = stable_id(
            "review_job",
            route["route_id"],
            "1",
            route_sha,
            prompt["prompt_sha256"],
            REVIEW_POLICY_VERSION,
        )
        body = {
            "schema_version": REVIEW_JOB_SCHEMA_VERSION,
            "result_schema_version": REVIEW_RESULT_SCHEMA_VERSION,
            "result_schema": result_schema,
            "result_schema_sha256": _hash(result_schema),
            "policy_version": REVIEW_POLICY_VERSION,
            "job_id": job_id,
            "round": 1,
            "route_id": route["route_id"],
            "opportunity_id": route["opportunity_id"],
            "opportunity_sha256": parent["opportunity_sha256"],
            "opportunity": parent["opportunity"],
            "route": route,
            "route_record_sha256": route_sha,
            "route_bundle_manifest_sha256": route_manifest["bundle_manifest_sha256"],
            "residual": parent["residual"],
            "recon_provenance": parent["recon_provenance"],
            "allowed_baseline_evidence_ids": sorted(allowed_evidence),
            "mechanism_component_ids": mechanism_component_ids,
            "source_of_gain_sha256": _hash(route["source_of_gain"]),
            "cheapest_test_sha256": _hash(route["cheapest_decisive_test"]),
            "kill_condition_sha256": _hash(route["kill_condition"]),
            "required_resource_ids": route["required_resources"],
            "resource_constraints_sha256": _hash(route["resource_constraints"]),
            "residual_sha256": _hash(parent["residual"]),
            "scope_constraint_ids": ["CONDITION_Z", "SCOPE_COMPATIBILITY", "DECISIVE_EXPERIMENT"],
            "readiness_scope": route["readiness_scope"],
            "live_ready": route["live_ready"],
            "attestation_scope": route["attestation_scope"],
            "authenticity": route["authenticity"],
            "authenticity_boundary": route["authenticity_boundary"],
            "operator_attested_execution_ready": route["operator_attested_execution_ready"],
            "allowed_decisions": ["KILL", "NARROW", "PASS_TO_HUMAN"],
            "accepted_job_immutable": True,
            "rejected_job_retryable": True,
            "controlled_catalog": {
                "allowed_decisions": ["KILL", "NARROW", "PASS_TO_HUMAN"],
                "baseline_evidence_ids": sorted(allowed_evidence),
                "mechanism_component_ids": mechanism_component_ids,
                "scope_constraint_ids": ["CONDITION_Z", "SCOPE_COMPATIBILITY", "DECISIVE_EXPERIMENT"],
                "source_of_gain_verdicts": ["ISOLATED", "CONFOUNDED", "UNRESOLVED"],
                "falsifiability_verdicts": ["FALSIFIABLE", "UNDER_SPECIFIED"],
                "cost_verdicts": ["WITHIN_BOUND", "EXCEEDS_BOUND", "UNRESOLVED"],
            },
            "novelty_judgment_forbidden": True,
            "mechanism_change_forbidden": True,
            **prompt,
        }
        body["review_job_sha256"] = _hash(body)
        jobs.append(body)
    return jobs


def _round_two_job(
    parent_job: Mapping[str, Any],
    prior_review: Mapping[str, Any],
    prior_raw: str,
    prompt: Mapping[str, str],
) -> dict[str, Any]:
    if prior_review.get("decision") != "NARROW" or prior_review.get("round") != 1:
        raise ValueError("only a validated round-one NARROW can enter round two")
    prior_sha = _hash(prior_review)
    job_id = stable_id(
        "review_job",
        parent_job["route_id"],
        "2",
        prior_sha,
        prompt["prompt_sha256"],
        REVIEW_POLICY_VERSION,
    )
    body = {
        **{key: value for key, value in parent_job.items() if key not in {"job_id", "round", "allowed_decisions", "review_job_sha256"}},
        "job_id": job_id,
        "round": 2,
        "allowed_decisions": ["KILL", "PASS_TO_HUMAN"],
        "parent_job_id": parent_job["job_id"],
        "parent_review": prior_review,
        "parent_round_one_result_sha256": prior_sha,
        "parent_raw_result_json": prior_raw,
        "parent_raw_result_sha256": hashlib.sha256(prior_raw.encode("utf-8")).hexdigest(),
    }
    body["controlled_catalog"] = {
        **body["controlled_catalog"],
        "allowed_decisions": ["KILL", "PASS_TO_HUMAN"],
    }
    body["review_job_sha256"] = _hash(body)
    return body


def _validate_embedded_round_two(
    extra: Mapping[str, Any],
    parent: Mapping[str, Any],
    prompt: Mapping[str, str],
) -> dict[str, Any]:
    raw = extra.get("parent_raw_result_json")
    if type(raw) is not str or hashlib.sha256(raw.encode("utf-8")).hexdigest() != extra.get("parent_raw_result_sha256"):
        raise ValueError("round-two parent raw replay binding mismatch")
    records, outcome = _project_review_result(parent, raw)
    if outcome["status"] != "ACCEPTED" or len(records) != 1 or records[0]["decision"] != "NARROW":
        raise ValueError("round two requires an accepted round-one NARROW")
    expected = _round_two_job(parent, records[0], raw, prompt)
    if dict(extra) != expected:
        raise ValueError("round-two review job replay mismatch")
    return expected


def _validate_current_round_one_parent(review: Path, extra: Mapping[str, Any]) -> None:
    try:
        current_results = read_jsonl(_owned_file(review, "results.jsonl"))
        current_outcomes = read_jsonl(_owned_file(review, "outcomes.jsonl"))
    except ValueError as exc:
        raise ValueError("round-two review job lacks the current round-one artifacts") from exc
    parent_job_id = extra["parent_job_id"]
    current = next((row for row in current_results if row.get("job_id") == parent_job_id and row.get("round") == 1), None)
    outcome = next((row for row in current_outcomes if row.get("job_id") == parent_job_id and row.get("round") == 1), None)
    if current is None or outcome is None or current.get("decision") != "NARROW" or outcome.get("status") != "ACCEPTED":
        raise ValueError("round-two parent is stale or no longer a current round-one NARROW")
    if _hash(current) != extra.get("parent_round_one_result_sha256") or current != extra.get("parent_review"):
        raise ValueError("round-two parent round-one result SHA is stale")
    raw = outcome.get("raw_result_json")
    if type(raw) is not str or hashlib.sha256(raw.encode("utf-8")).hexdigest() != extra.get("parent_raw_result_sha256"):
        raise ValueError("round-two parent accepted raw SHA is stale")


def emit_review_jobs(
    run_dir: Path,
    config: CorpusRouterConfig,
    *,
    round_number: Literal[1, 2] = 1,
    prompt_path: Path | None = None,
    route_prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> Path:
    run = _anchored_run(Path(run_dir))
    review = _review_dir(run, create=True)
    main_path = review / "jobs.jsonl"
    pending_path = review / "round2_jobs.jsonl"
    with stage_mutation_lock(review):
        prompt = _prompt_binding(prompt_path)
        base = _round_one_jobs(
            run,
            config,
            prompt,
            route_prompt_path=route_prompt_path,
            allow_test_ready=allow_test_ready,
        )
        if round_number == 1:
            exists = [main_path.exists(), *[(review / name).exists() for name in _OUTPUTS]]
            if any(exists):
                if not exists[0] or (any(exists[1:]) and not all(exists[1:])):
                    raise ValueError("existing review stage artifacts are incomplete")
                if all(exists[1:]):
                    validate_review_bundle(
                        run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
                        allow_test_ready=allow_test_ready,
                    )
                else:
                    validate_review_jobs(
                        run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
                        allow_test_ready=allow_test_ready,
                    )
                if read_jsonl(_owned_file(review, "jobs.jsonl")) != base:
                    raise ValueError("existing review jobs are immutable under changed replay inputs")
                return main_path
            publish_transaction({main_path: encode_jsonl(base)})
            return main_path
        if round_number != 2:
            raise ValueError("review round_number must be 1 or 2")
        prior = validate_review_bundle(
            run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
            allow_test_ready=allow_test_ready,
        )
        if any(job.get("round") == 2 for job in prior["jobs"]):
            raise ValueError("review round two has already been ingested")
        outcome_by_job = {row["job_id"]: row for row in prior["outcomes"]}
        result_by_job = {row["job_id"]: row for row in prior["results"] if row["round"] == 1}
        second: list[dict[str, Any]] = []
        for parent in base:
            result = result_by_job.get(parent["job_id"])
            if result is None or result.get("decision") != "NARROW":
                continue
            outcome = outcome_by_job.get(parent["job_id"])
            raw = outcome.get("raw_result_json") if outcome else None
            if type(raw) is not str:
                raise ValueError("NARROW outcome lacks replayable accepted raw")
            second.append(_round_two_job(parent, result, raw, prompt))
        if not second:
            raise ValueError("round two is allowed only when round one returned NARROW")
        if pending_path.exists():
            if read_jsonl(_owned_file(review, "round2_jobs.jsonl")) != second:
                raise ValueError("existing round-two pending handoff replay mismatch")
            return pending_path
        publish_transaction({pending_path: encode_jsonl(second)})
        return pending_path


def validate_review_jobs(
    run_dir: Path,
    config: CorpusRouterConfig,
    *,
    prompt_path: Path | None = None,
    route_prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> tuple[dict[str, Any], ...]:
    run = _anchored_run(Path(run_dir))
    review = _review_dir(run, create=False)
    actual = read_jsonl(_owned_file(review, "jobs.jsonl"))
    prompt = _prompt_binding(prompt_path)
    base = _round_one_jobs(
        run,
        config,
        prompt,
        route_prompt_path=route_prompt_path,
        allow_test_ready=allow_test_ready,
    )
    if actual[: len(base)] != base:
        raise ValueError("round-one review jobs replay mismatch")
    extras = actual[len(base) :]
    if len(extras) > len(base):
        raise ValueError("review jobs contain too many round-two records")
    parent_by_id = {job["job_id"]: job for job in base}
    seen_routes: set[str] = set()
    for extra in extras:
        parent = parent_by_id.get(str(extra.get("parent_job_id", "")))
        if parent is None or extra.get("route_id") in seen_routes:
            raise ValueError("round-two review parent binding is invalid")
        _validate_embedded_round_two(extra, parent, prompt)
        _validate_current_round_one_parent(review, extra)
        seen_routes.add(str(extra["route_id"]))
    if actual != [*base, *extras]:
        raise ValueError("review jobs canonical order mismatch")
    expected_order = [job["route_id"] for job in base if job["route_id"] in seen_routes]
    if [job["route_id"] for job in extras] != expected_order:
        raise ValueError("round-two review jobs are not in canonical route order")
    return tuple(actual)


def _validate_pending_round_two(
    review: Path,
    base: Sequence[Mapping[str, Any]],
    prompt: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    actual = read_jsonl(_owned_file(review, "round2_jobs.jsonl"))
    parent_by_id = {str(job["job_id"]): job for job in base if int(job["round"]) == 1}
    seen: set[str] = set()
    for extra in actual:
        parent = parent_by_id.get(str(extra.get("parent_job_id", "")))
        route_id = str(extra.get("route_id", ""))
        if parent is None or route_id in seen:
            raise ValueError("round-two pending parent binding is invalid")
        _validate_embedded_round_two(extra, parent, prompt)
        _validate_current_round_one_parent(review, extra)
        seen.add(route_id)
    expected_order = [str(job["route_id"]) for job in base if str(job["route_id"]) in seen]
    if [str(job["route_id"]) for job in actual] != expected_order:
        raise ValueError("round-two pending jobs are not in canonical route order")
    return tuple(actual)


class _StrongestBaseline(FrozenStrictModel):
    evidence_id: NonEmptyStr
    comparison: Literal["MATCHED_BUDGET"]


class _APlusBObjection(FrozenStrictModel):
    component_ids: list[NonEmptyStr] = Field(min_length=2, max_length=2)
    verdict: Literal["PLAUSIBLE_COMBINATION", "NOT_A_SIMPLE_COMBINATION", "UNRESOLVED"]


class _SourceOfGainVerdict(FrozenStrictModel):
    source_of_gain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    verdict: Literal["ISOLATED", "CONFOUNDED", "UNRESOLVED"]


class _FalsifiabilityVerdict(FrozenStrictModel):
    cheapest_test_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    kill_condition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    verdict: Literal["FALSIFIABLE", "UNDER_SPECIFIED"]


class _CostRisk(FrozenStrictModel):
    required_resource_ids: list[NonEmptyStr] = Field(min_length=1)
    resource_constraints_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    verdict: Literal["WITHIN_BOUND", "EXCEEDS_BOUND", "UNRESOLVED"]


class _ResidualClaim(FrozenStrictModel):
    residual_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    opportunity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    disposition: Literal["KILL", "NARROW", "PASS_TO_HUMAN"]
    scope_constraint_id: Literal["CONDITION_Z", "SCOPE_COMPATIBILITY", "DECISIVE_EXPERIMENT"] | None


class _ReviewerResult(FrozenStrictModel):
    schema_version: Literal["idea_factory.reviewer_result.v1"]
    job_id: NonEmptyStr
    route_id: NonEmptyStr
    opportunity_id: NonEmptyStr
    round: int = Field(strict=True, ge=1, le=2)
    review_job_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", strict=True)
    decision: Literal["KILL", "NARROW", "PASS_TO_HUMAN"]
    strongest_baseline: _StrongestBaseline
    a_plus_b_objection: _APlusBObjection
    source_of_gain_verdict: _SourceOfGainVerdict
    falsifiability_verdict: _FalsifiabilityVerdict
    cost_risk: _CostRisk
    residual_claim: _ResidualClaim


def _validate_review_semantics(item: _ReviewerResult, job: Mapping[str, Any]) -> None:
    for key in ("job_id", "route_id", "opportunity_id", "round", "review_job_sha256", "prompt_sha256"):
        if getattr(item, key) != job[key]:
            raise ValueError("review result binding mismatch")
    allowed = _ROUND_ONE_DECISIONS if item.round == 1 else _ROUND_TWO_DECISIONS
    if item.decision not in allowed or item.decision not in set(job["allowed_decisions"]):
        raise ValueError("review decision is forbidden for this round")
    if item.strongest_baseline.evidence_id not in set(job["allowed_baseline_evidence_ids"]):
        raise ValueError("review baseline evidence anchor is invalid")
    components = item.a_plus_b_objection.component_ids
    if len(set(components)) != 2 or not set(components).issubset(set(job["mechanism_component_ids"])):
        raise ValueError("review A+B mechanism component anchors are invalid")
    if item.source_of_gain_verdict.source_of_gain_sha256 != job["source_of_gain_sha256"]:
        raise ValueError("review source-of-gain hash anchor is invalid")
    if item.falsifiability_verdict.cheapest_test_sha256 != job["cheapest_test_sha256"] or item.falsifiability_verdict.kill_condition_sha256 != job["kill_condition_sha256"]:
        raise ValueError("review falsifiability hash anchors are invalid")
    if item.cost_risk.required_resource_ids != job["required_resource_ids"] or item.cost_risk.resource_constraints_sha256 != job["resource_constraints_sha256"]:
        raise ValueError("review cost/resource anchors are invalid")
    claim = item.residual_claim
    if claim.residual_sha256 != job["residual_sha256"] or claim.opportunity_sha256 != job["opportunity_sha256"] or claim.disposition != item.decision:
        raise ValueError("review residual/opportunity anchors are invalid")
    if item.decision == "NARROW":
        if claim.scope_constraint_id not in set(job["scope_constraint_ids"]):
            raise ValueError("NARROW requires a bound scoped narrowing")
    elif claim.scope_constraint_id is not None:
        raise ValueError("terminal review decisions cannot add a scoped narrowing")


def _project_review_result(job: Mapping[str, Any], raw: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    payload = _strict_load(raw)
    if contains_secret(payload):
        raise ValueError("review payload contains a credential-shaped secret")
    if contains_global_priority_claim(payload):
        raise ValueError("review payload contains a forbidden global-priority claim")
    item = _ReviewerResult.model_validate(payload, strict=True)
    _validate_review_semantics(item, job)
    data = item.model_dump(mode="json")
    data.pop("schema_version")
    record_id = stable_id("review_result", job["job_id"], raw_sha)
    record = {
        "schema_version": REVIEW_RECORD_SCHEMA_VERSION,
        "record_id": record_id,
        "raw_result_sha256": raw_sha,
        "route_record_sha256": job["route_record_sha256"],
        "route_bundle_manifest_sha256": job["route_bundle_manifest_sha256"],
        "readiness_scope": job["readiness_scope"],
        "live_ready": job["live_ready"],
        "attestation_scope": job["attestation_scope"],
        "authenticity": job["authenticity"],
        "authenticity_boundary": job["authenticity_boundary"],
        "operator_attested_execution_ready": job["operator_attested_execution_ready"],
        **data,
    }
    outcome = {
        "schema_version": REVIEW_OUTCOME_SCHEMA_VERSION,
        "job_id": job["job_id"],
        "route_id": job["route_id"],
        "opportunity_id": job["opportunity_id"],
        "round": job["round"],
        "status": "ACCEPTED",
        "raw_result_json": raw,
        "raw_result_sha256": raw_sha,
        "decision": item.decision,
        "review_record_id": record_id,
        "accepted_projection_sha256": _hash(record),
    }
    return [record], outcome


def _rejection(job: Mapping[str, Any], raw_sha: str, exc: Exception) -> dict[str, Any]:
    text = str(exc)
    if "credential-shaped secret" in text:
        code, reason = "REVIEW_SECRET_DETECTED", "review result contains credential-shaped content"
    elif "global-priority" in text:
        code, reason = "REVIEW_GLOBAL_PRIORITY_CLAIM_FORBIDDEN", "reviewer cannot make global-priority claims"
    elif "anchor" in text or "scoped narrowing" in text:
        code, reason = "REVIEW_ANCHOR_INVALID", "review result loses a structured route anchor"
    elif "round" in text or "decision is forbidden" in text:
        code, reason = "REVIEW_ROUND_INVALID", "review decision is forbidden for this round"
    elif "binding" in text:
        code, reason = "REVIEW_BINDING_INVALID", "review result changes an exact route or opportunity binding"
    else:
        code, reason = "REVIEW_SCHEMA_INVALID", "review result schema is invalid"
    return {
        "schema_version": REVIEW_OUTCOME_SCHEMA_VERSION,
        "job_id": job["job_id"],
        "route_id": job["route_id"],
        "opportunity_id": job["opportunity_id"],
        "round": job["round"],
        "status": "REJECTED",
        "raw_result_sha256": raw_sha,
        "error_code": code,
        "error": reason,
        "decision": None,
        "review_record_id": None,
        "accepted_projection_sha256": _hash([]),
    }


def _raw_text(source: str | bytes | Path) -> str:
    raw = source.read_bytes() if isinstance(source, Path) else source.encode("utf-8") if type(source) is str else source
    if not isinstance(raw, bytes):
        raise TypeError("review result must be text, bytes, or a path")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("review result must be UTF-8") from exc


def _validate_incoming_round_two_parents(
    review: Path,
    projected: Sequence[tuple[Mapping[str, Any], str, Sequence[Mapping[str, Any]], Mapping[str, Any]]],
) -> None:
    canonical: dict[tuple[str, int], tuple[Mapping[str, Any], str, Sequence[Mapping[str, Any]], Mapping[str, Any]]] = {}
    for entry in projected:
        job = entry[0]
        key = (str(job["route_id"]), int(job["round"]))
        if key in canonical:
            raise ValueError("incoming review batch repeats a route and round")
        canonical[key] = entry
    for job, _raw, _records, _outcome in projected:
        if int(job["round"]) != 2:
            continue
        parent_entry = canonical.get((str(job["route_id"]), 1))
        if parent_entry is None:
            _validate_current_round_one_parent(review, job)
            continue
        _parent_job, parent_raw, parent_records, parent_outcome = parent_entry
        if (
            parent_outcome.get("status") != "ACCEPTED"
            or len(parent_records) != 1
            or parent_records[0].get("decision") != "NARROW"
        ):
            raise ValueError("incoming round-two parent must be an accepted round-one NARROW")
        parent_record = parent_records[0]
        if (
            _hash(parent_record) != job.get("parent_round_one_result_sha256")
            or parent_record != job.get("parent_review")
            or hashlib.sha256(parent_raw.encode("utf-8")).hexdigest() != job.get("parent_raw_result_sha256")
            or parent_outcome.get("raw_result_json") != job.get("parent_raw_result_json")
        ):
            raise ValueError("incoming round-two parent round-one result or raw SHA is stale")


def _canonical_job_sources(
    jobs: Sequence[Mapping[str, Any]],
    results: Sequence[str | bytes | Path],
) -> list[tuple[Mapping[str, Any], str]]:
    expected = {(str(job["route_id"]), int(job["round"])): job for job in jobs}
    by_key: dict[tuple[str, int], str] = {}
    try:
        for source in results:
            raw = _raw_text(source)
            payload = _strict_load(raw)
            key = (str(payload.get("route_id", "")), int(payload.get("round", 0)))
            if key not in expected or key in by_key:
                raise ValueError("incoming review batch has an unknown or duplicate route and round")
            by_key[key] = raw
    except (TypeError, ValueError):
        return [(job, _raw_text(source)) for job, source in zip(jobs, results)]
    if set(by_key) != set(expected):
        raise ValueError("incoming review batch does not cover the expected route and round keys")
    return [(job, by_key[(str(job["route_id"]), int(job["round"]))]) for job in jobs]


def _validate_accepted_job_replays(
    review: Path,
    projected: Sequence[tuple[Mapping[str, Any], str, Sequence[Mapping[str, Any]], Mapping[str, Any]]],
) -> None:
    paths = [review / name for name in _OUTPUTS]
    if not any(path.exists() for path in paths):
        return
    if not all(path.is_file() for path in paths):
        raise ValueError("existing accepted review artifacts are incomplete")
    current_results = read_jsonl(_owned_file(review, "results.jsonl"))
    current_outcomes = read_jsonl(_owned_file(review, "outcomes.jsonl"))
    current_result_by_job = {str(row["job_id"]): row for row in current_results}
    incoming_by_job = {str(entry[0]["job_id"]): entry for entry in projected}
    for current_outcome in current_outcomes:
        if current_outcome.get("status") != "ACCEPTED":
            continue
        job_id = str(current_outcome["job_id"])
        current_result = current_result_by_job.get(job_id)
        incoming = incoming_by_job.get(job_id)
        if current_result is None or incoming is None:
            raise ValueError("accepted review job is immutable and requires exact replay")
        _job, incoming_raw, incoming_records, incoming_outcome = incoming
        if (
            incoming_outcome.get("status") != "ACCEPTED"
            or len(incoming_records) != 1
            or incoming_records[0] != current_result
            or incoming_outcome.get("raw_result_sha256") != current_outcome.get("raw_result_sha256")
            or incoming_outcome.get("accepted_projection_sha256")
            != current_outcome.get("accepted_projection_sha256")
            or hashlib.sha256(incoming_raw.encode("utf-8")).hexdigest()
            != current_outcome.get("raw_result_sha256")
        ):
            raise ValueError("accepted review job is immutable; only exact idempotent replay is allowed")


def _manifest(
    review: Path,
    jobs: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    artifact_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    latest: dict[str, Mapping[str, Any]] = {}
    for result in results:
        prior = latest.get(str(result["route_id"]))
        if prior is None or int(result["round"]) > int(prior["round"]):
            latest[str(result["route_id"])] = result
    pass_count = sum(row["decision"] == "PASS_TO_HUMAN" for row in latest.values())
    kill_count = sum(row["decision"] == "KILL" for row in latest.values())
    narrow_count = sum(row["decision"] == "NARROW" for row in latest.values())
    accepted_count = sum(row.get("status") == "ACCEPTED" for row in outcomes)
    expected_routes = {str(job["route_id"]) for job in jobs}
    rejected_count = len(outcomes) - accepted_count
    if not expected_routes:
        resolution_status = "ZERO_INPUT"
    elif rejected_count or set(latest) != expected_routes:
        resolution_status = "UNRESOLVED"
    elif pass_count:
        resolution_status = "HAS_SURVIVORS"
    elif narrow_count:
        resolution_status = "PENDING"
    elif kill_count == len(expected_routes):
        resolution_status = "ZERO_SURVIVOR"
    else:
        resolution_status = "UNRESOLVED"
    body = {
        "schema_version": REVIEW_MANIFEST_SCHEMA_VERSION,
        "policy_version": REVIEW_POLICY_VERSION,
        "prompt_version": jobs[0]["prompt_version"] if jobs else REVIEW_PROMPT_VERSION,
        "prompt_source": jobs[0]["prompt_source"] if jobs else str(_DEFAULT_PROMPT),
        "prompt_sha256": jobs[0]["prompt_sha256"] if jobs else sha256_file(_DEFAULT_PROMPT),
        "jobs_sha256": artifact_hashes["jobs"] if artifact_hashes else sha256_file(review / "jobs.jsonl"),
        "results_sha256": artifact_hashes["results"] if artifact_hashes else sha256_file(review / "results.jsonl"),
        "outcomes_sha256": artifact_hashes["outcomes"] if artifact_hashes else sha256_file(review / "outcomes.jsonl"),
        "job_count": len(jobs),
        "round_two_count": sum(job["round"] == 2 for job in jobs),
        "accepted_result_count": len(results),
        "accepted_job_count": accepted_count,
        "rejected_job_count": rejected_count,
        "pass_to_human_count": pass_count,
        "kill_count": kill_count,
        "narrow_count": narrow_count,
        "pending_route_count": narrow_count,
        "resolution_status": resolution_status,
        "zero_survivor": resolution_status == "ZERO_SURVIVOR",
        "readiness_scopes": sorted({str(job["readiness_scope"]) for job in jobs}),
        "live_ready": bool(jobs) and all(bool(job["live_ready"]) for job in jobs),
        "attestation_scopes": sorted({str(job["attestation_scope"]) for job in jobs}),
        "authenticity": jobs[0]["authenticity"] if jobs else "NOT_AUTHENTICATED",
        "authenticity_boundaries": sorted({str(job["authenticity_boundary"]) for job in jobs}),
        "operator_attested_execution_ready": bool(jobs)
        and all(bool(job["operator_attested_execution_ready"]) for job in jobs),
        "independently_verified": False,
        "result_schema_sha256s": sorted(
            {str(job["result_schema_sha256"]) for job in jobs}
            or {_hash(_ReviewerResult.model_json_schema())}
        ),
    }
    return body | {"bundle_manifest_sha256": _hash(body)}


def ingest_review_results(
    jobs_path: Path,
    results: Sequence[str | bytes | Path],
    config: CorpusRouterConfig,
    *,
    prompt_path: Path | None = None,
    route_prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> dict[str, Path]:
    run = _anchored_run(Path(jobs_path).parent.parent)
    review = _review_dir(run, create=False)
    main_path = review / "jobs.jsonl"
    pending_path = review / "round2_jobs.jsonl"
    with stage_mutation_lock(review):
        supplied = Path(os.path.abspath(jobs_path))
        output_exists = [(review / name).exists() for name in _OUTPUTS]
        if any(output_exists) and not all(output_exists):
            raise ValueError("existing review bundle is incomplete")
        current = (
            validate_review_bundle(
                run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
                allow_test_ready=allow_test_ready,
            )
            if all(output_exists)
            else None
        )
        pending_merge = supplied == pending_path and pending_path.exists()
        projected: list[tuple[Mapping[str, Any], str, Sequence[Mapping[str, Any]], Mapping[str, Any]]] = []
        if pending_merge:
            if current is None:
                raise ValueError("round-two pending ingest requires a validated current bundle")
            base = list(current["jobs"])
            prompt = _prompt_binding(prompt_path)
            round_two_jobs = list(_validate_pending_round_two(review, base, prompt))
            if len(results) != len(round_two_jobs):
                raise ValueError("round-two results must exactly cover pending jobs")
            current_result_by_job = {str(row["job_id"]): row for row in current["results"]}
            for job, outcome in zip(base, current["outcomes"]):
                raw = outcome.get("raw_result_json") if outcome.get("status") == "ACCEPTED" else ""
                records = [current_result_by_job[str(job["job_id"])] ] if outcome.get("status") == "ACCEPTED" else []
                projected.append((job, raw, records, outcome))
            job_sources = _canonical_job_sources(round_two_jobs, results)
            jobs = [*base, *round_two_jobs]
        else:
            if supplied != _owned_file(review, "jobs.jsonl"):
                raise ValueError("review jobs must use the active main or pending run path")
            jobs = list(validate_review_jobs(
                run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
                allow_test_ready=allow_test_ready,
            ))
            round_two_jobs = [job for job in jobs if int(job["round"]) == 2]
            if len(results) == len(jobs):
                job_sources = _canonical_job_sources(jobs, results)
            elif round_two_jobs and len(results) == len(round_two_jobs) and current is not None:
                current_outcomes = {str(row["job_id"]): row for row in current["outcomes"]}
                job_sources = []
                for job in jobs:
                    if int(job["round"]) == 2:
                        continue
                    old = current_outcomes.get(str(job["job_id"]))
                    raw = old.get("raw_result_json") if old else None
                    if old is None or old.get("status") != "ACCEPTED" or type(raw) is not str:
                        raise ValueError("round-two-only batch lacks a replayable current round-one result")
                    job_sources.append((job, raw))
                job_sources.extend(_canonical_job_sources(round_two_jobs, results))
            else:
                raise ValueError("review results must cover every job or the canonical round-two jobs")
        for job, source in job_sources:
            raw = _raw_text(source)
            raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            try:
                records, outcome = _project_review_result(job, raw)
            except (ValidationError, ValueError, TypeError) as exc:
                records, outcome = [], _rejection(job, raw_sha, exc)
            projected.append((job, raw, records, outcome))
        _validate_incoming_round_two_parents(review, projected)
        _validate_accepted_job_replays(review, projected)
        accepted: list[dict[str, Any]] = []
        outcomes: list[dict[str, Any]] = []
        for _job, _raw, records, outcome in projected:
            accepted.extend(records); outcomes.append(outcome)
        paths = {
            "results": review / "results.jsonl",
            "outcomes": review / "outcomes.jsonl",
            "manifest": review / "bundle_manifest.json",
        }
        jobs_bytes = encode_jsonl(jobs)
        results_bytes = encode_jsonl(accepted)
        outcomes_bytes = encode_jsonl(outcomes)
        hashes = {
            "jobs": hashlib.sha256(jobs_bytes).hexdigest(),
            "results": hashlib.sha256(results_bytes).hexdigest(),
            "outcomes": hashlib.sha256(outcomes_bytes).hexdigest(),
        }
        manifest_bytes = encode_json(_manifest(review, jobs, accepted, outcomes, hashes))
        transaction: dict[Path, bytes | None] = {
            main_path: jobs_bytes,
            paths["results"]: results_bytes,
            paths["outcomes"]: outcomes_bytes,
            paths["manifest"]: manifest_bytes,
        }
        if pending_merge:
            transaction[pending_path] = None
        publish_transaction(transaction)
        return paths


_REJECTION_PAIRS = {
    ("REVIEW_ROUND_INVALID", "review decision is forbidden for this round"),
    ("REVIEW_BINDING_INVALID", "review result changes an exact route or opportunity binding"),
    ("REVIEW_SCHEMA_INVALID", "review result schema is invalid"),
    ("REVIEW_ANCHOR_INVALID", "review result loses a structured route anchor"),
    ("REVIEW_GLOBAL_PRIORITY_CLAIM_FORBIDDEN", "reviewer cannot make global-priority claims"),
    ("REVIEW_SECRET_DETECTED", "review result contains credential-shaped content"),
}


def validate_review_bundle(
    run_dir: Path,
    config: CorpusRouterConfig,
    *,
    prompt_path: Path | None = None,
    route_prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> dict[str, Any]:
    run = _anchored_run(Path(run_dir))
    review = _review_dir(run, create=False)
    jobs = validate_review_jobs(
        run,
        config,
        prompt_path=prompt_path,
        route_prompt_path=route_prompt_path,
        allow_test_ready=allow_test_ready,
    )
    actual_results = read_jsonl(_owned_file(review, "results.jsonl"))
    actual_outcomes = read_jsonl(_owned_file(review, "outcomes.jsonl"))
    actual_manifest = _strict_load(_owned_file(review, "bundle_manifest.json").read_bytes())
    if len(actual_outcomes) != len(jobs):
        raise ValueError("review outcomes must exactly cover jobs")
    expected_results: list[dict[str, Any]] = []
    expected_outcomes: list[dict[str, Any]] = []
    for job, outcome in zip(jobs, actual_outcomes):
        if outcome.get("status") == "ACCEPTED":
            raw = outcome.get("raw_result_json")
            if type(raw) is not str or hashlib.sha256(raw.encode("utf-8")).hexdigest() != outcome.get("raw_result_sha256"):
                raise ValueError("accepted review outcome raw replay binding mismatch")
            records, replayed = _project_review_result(job, raw)
            if outcome != replayed:
                raise ValueError("accepted review outcome replay mismatch")
            expected_results.extend(records); expected_outcomes.append(replayed)
            continue
        keys = {"schema_version", "job_id", "route_id", "opportunity_id", "round", "status", "raw_result_sha256", "error_code", "error", "decision", "review_record_id", "accepted_projection_sha256"}
        if set(outcome) != keys or outcome.get("schema_version") != REVIEW_OUTCOME_SCHEMA_VERSION or outcome.get("status") != "REJECTED":
            raise ValueError("rejected review outcome schema mismatch")
        if (outcome.get("error_code"), outcome.get("error")) not in _REJECTION_PAIRS:
            raise ValueError("rejected review outcome reason is not bounded")
        if not re.fullmatch(r"[0-9a-f]{64}", str(outcome.get("raw_result_sha256", ""))) or outcome.get("decision") is not None or outcome.get("review_record_id") is not None or outcome.get("accepted_projection_sha256") != _hash([]):
            raise ValueError("rejected review outcome binding mismatch")
        if any(outcome.get(key) != job[key] for key in ("job_id", "route_id", "opportunity_id", "round")):
            raise ValueError("rejected review outcome job binding mismatch")
        expected_outcomes.append(outcome)
    if actual_results != expected_results or actual_outcomes != expected_outcomes:
        raise ValueError("review accepted results replay mismatch")
    expected_manifest = _manifest(review, jobs, expected_results, expected_outcomes)
    if actual_manifest != expected_manifest:
        raise ValueError("review bundle manifest replay mismatch")
    return {
        "jobs": tuple(jobs),
        "results": tuple(actual_results),
        "outcomes": tuple(actual_outcomes),
        "manifest": actual_manifest,
    }
