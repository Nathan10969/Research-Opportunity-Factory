"""Deterministic internal-prior shape fingerprints and conservative dedup decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from contextlib import ExitStack
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import ValidationError

from .artifacts import read_jsonl, stable_id, write_jsonl_bundle
from .corpus import CorpusRouterConfig
from .legacy_ledger import validate_legacy_import
from .models import FrozenStrictModel, NonEmptyStr
from .opportunities import _anchored_run, _owned_dir, _owned_file, _safe_unlink
from .safety import contains_secret
from .stage_io import (
    encode_json, encode_jsonl, publish_cross_directory_transaction, publish_transaction,
    stage_mutation_lock,
)


SHAPE_SCHEMA_VERSION = "idea_factory.shape.v1"
FINGERPRINT_POLICY_VERSION = "unicode-nfkc-casefold-space-punctuation.v1"
DEDUP_POLICY_VERSION = "idea_factory.internal_dedup.v1"
TARGET_OVERLAP_THRESHOLD = 0.60
_FIELDS = ("old_assumption", "failure_mechanism", "missing_capability", "target_scope")
_PUNCTUATION = re.compile(r"[‐‑‒–—―]+")
_NONWORD = re.compile(r"[^\w\s/-]+", re.UNICODE)


def normalize_shape_text(value: str) -> str:
    """Small, explicit normalization: no synonym, embedding, or semantic expansion."""

    if not isinstance(value, str):
        raise TypeError("shape text must be a string")
    value = unicodedata.normalize("NFKC", value).casefold()
    value = _PUNCTUATION.sub("-", value)
    value = _NONWORD.sub(" ", value)
    return " ".join(value.split())


def _shape_values(value: Mapping[str, Any]) -> tuple[str, str, str, str]:
    raw_values = tuple(value.get(field) for field in _FIELDS)
    if any(item is None for item in raw_values):
        raise ValueError("shape fingerprint is unavailable when a required field is unavailable")
    if any(type(item) is not str for item in raw_values):
        raise TypeError("shape fingerprint fields must be strings")
    return tuple(normalize_shape_text(item) for item in raw_values)  # type: ignore[arg-type,return-value]


def shape_fingerprint(shape: Mapping[str, Any]) -> str:
    """Hash a versioned JSON/unit-separator shape tuple without ambiguous concatenation."""

    encoded = json.dumps(
        [SHAPE_SCHEMA_VERSION, FINGERPRINT_POLICY_VERSION, "\x1f".join(_shape_values(shape))],
        ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def mechanism_target_key(mechanism_family: str, target: str) -> str:
    return "\x1f".join((normalize_shape_text(mechanism_family), normalize_shape_text(target)))


class ShapeAssignment(FrozenStrictModel):
    """Externally supplied assignment; mechanism family cannot be inferred."""

    schema_version: str = "idea_factory.shape_assignment.v1"
    source_opportunity_id: NonEmptyStr
    old_assumption: str
    failure_mechanism: NonEmptyStr
    missing_capability: NonEmptyStr
    target_scope: NonEmptyStr
    mechanism_family: NonEmptyStr
    residual_difference: str = ""

    def fingerprint(self) -> str:
        return shape_fingerprint(self.model_dump(mode="json"))


@dataclass(frozen=True)
class DedupDecision:
    status: str
    blocked: bool
    reason: str
    nearest_legacy_ids: tuple[str, ...]
    nearest_evidence: tuple[str, ...]
    fingerprint: str
    mechanism_target_key: str


def _tokens(value: str) -> set[str]:
    return {token for token in normalize_shape_text(value).replace("/", " ").replace("-", " ").split() if token}


def _target_overlap(left: str, right: str) -> bool:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return False
    return normalize_shape_text(left) == normalize_shape_text(right) or len(a & b) / len(a | b) >= TARGET_OVERLAP_THRESHOLD


class ReviewedLegacyShapeAssignment(FrozenStrictModel):
    """Separately reviewed four-field evidence; never derived from ledger prose."""

    schema_version: Literal["idea_factory.reviewed_legacy_shape_assignment.v1"]
    reviewed: Literal[True]
    old_assumption: str
    failure_mechanism: NonEmptyStr
    missing_capability: NonEmptyStr
    target_scope: NonEmptyStr


def legacy_shape(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a fingerprintable shape only when explicit reviewed evidence exists."""

    raw = record.get("reviewed_shape_assignment")
    if raw is None:
        return {
            "old_assumption": None, "failure_mechanism": None,
            "missing_capability": None, "target_scope": None,
            "fingerprint_available": False, "shape_fingerprint": None,
        }
    reviewed = ReviewedLegacyShapeAssignment.model_validate(raw)
    shape = {
        "old_assumption": reviewed.old_assumption,
        "failure_mechanism": reviewed.failure_mechanism,
        "missing_capability": reviewed.missing_capability,
        "target_scope": reviewed.target_scope,
        "fingerprint_available": True,
    }
    return shape | {"shape_fingerprint": shape_fingerprint(shape)}


def _potential_match(candidate: ShapeAssignment, record: Mapping[str, Any]) -> bool:
    legacy = legacy_shape(record)
    return (
        (
            legacy["fingerprint_available"]
            and shape_fingerprint(candidate.model_dump(mode="json")) == legacy["shape_fingerprint"]
        )
        or (
            normalize_shape_text(candidate.mechanism_family) == normalize_shape_text(str(record.get("mechanism_family", "")))
            and _target_overlap(candidate.target_scope, str(record.get("target", "")))
        )
        or (
            legacy["fingerprint_available"]
            and normalize_shape_text(candidate.failure_mechanism) == normalize_shape_text(legacy["failure_mechanism"])
            and _target_overlap(candidate.target_scope, legacy["target_scope"])
            and normalize_shape_text(candidate.missing_capability) != normalize_shape_text(legacy["missing_capability"])
        )
    )


def classify_shape(shape: Mapping[str, Any] | ShapeAssignment, legacy_records: Iterable[Mapping[str, Any]]) -> DedupDecision:
    candidate = shape if isinstance(shape, ShapeAssignment) else ShapeAssignment.model_validate(shape)
    records = [dict(row) for row in legacy_records]
    fingerprint = candidate.fingerprint()
    nearest_ids: list[str] = []
    evidence: list[str] = []
    low_confidence = [row for row in records if row.get("parse_confidence") != "HIGH" and _potential_match(candidate, row)]
    if low_confidence:
        return DedupDecision("NEEDS_HUMAN_REVIEW", True, "ambiguous or low-confidence legacy match", tuple(str(row["legacy_id"]) for row in low_confidence), tuple(str(row.get("evidence", "")) for row in low_confidence), fingerprint, mechanism_target_key(candidate.mechanism_family, candidate.target_scope))
    exact: list[dict[str, Any]] = []
    renamed: list[dict[str, Any]] = []
    adjacent: list[dict[str, Any]] = []
    for row in records:
        if row.get("parse_confidence") != "HIGH":
            continue
        projected = legacy_shape(row)
        if projected["fingerprint_available"] and projected["shape_fingerprint"] == fingerprint:
            exact.append(row)
        elif normalize_shape_text(candidate.mechanism_family) == normalize_shape_text(str(row.get("mechanism_family", ""))) and _target_overlap(candidate.target_scope, str(row.get("target", ""))):
            renamed.append(row)
        elif (
            projected["fingerprint_available"]
            and normalize_shape_text(candidate.failure_mechanism) == normalize_shape_text(projected["failure_mechanism"])
            and _target_overlap(candidate.target_scope, projected["target_scope"])
            and normalize_shape_text(candidate.missing_capability) != normalize_shape_text(projected["missing_capability"])
        ):
            adjacent.append(row)
    matches = exact or renamed
    if matches:
        nearest_ids = [str(row["legacy_id"]) for row in matches]
        evidence = [str(row.get("evidence", "")) for row in matches]
        covered = any(row.get("status") == "EXTERNALLY_COVERED" or row.get("source_table") == "KILLED_BLOCKLIST" for row in matches)
        return DedupDecision("EXTERNALLY_COVERED" if covered else "INTERNAL_DUP", True, "exact shape fingerprint" if exact else "same explicit mechanism family and target overlap", tuple(nearest_ids), tuple(evidence), fingerprint, mechanism_target_key(candidate.mechanism_family, candidate.target_scope))
    if adjacent:
        nearest_ids = [str(row["legacy_id"]) for row in adjacent]
        evidence = [str(row.get("evidence", "")) for row in adjacent]
        if not candidate.residual_difference.strip():
            return DedupDecision("ADJACENT_NEEDS_RESIDUAL", True, "same failure mechanism and overlapping target requires explicit residual", tuple(nearest_ids), tuple(evidence), fingerprint, mechanism_target_key(candidate.mechanism_family, candidate.target_scope))
        return DedupDecision("INTERNAL_ADJACENT", False, "same failure mechanism and overlapping target with explicit residual", tuple(nearest_ids), tuple(evidence), fingerprint, mechanism_target_key(candidate.mechanism_family, candidate.target_scope))
    return DedupDecision("CLEAR", False, "no deterministic internal match", (), (), fingerprint, mechanism_target_key(candidate.mechanism_family, candidate.target_scope))


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


_JOB_NAMES = ("shape_assignment_jobs.jsonl",)
_RESULT_NAMES = ("shape_assignment_results.jsonl",)
_DEDUP_NAMES = ("ready_after_internal_dedup.jsonl", "blocked.jsonl", "adjacent.jsonl", "dedup_outcomes.jsonl")


def _quality_hash(ready: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_hash(list(ready))


def _assert_task4_legacy_binding(run: Path, config: CorpusRouterConfig) -> None:
    """Bind the import to the exact legacy path carried by Task-4 selection."""

    corpus = _owned_dir(run, "corpus", create=False)
    selection = read_jsonl(_owned_file(corpus, "selection_manifest.jsonl"))
    expected = str(config.legacy_ledger)
    paths = {row.get("legacy_ledger_path") for row in selection}
    if not paths or paths != {expected}:
        raise ValueError("configured legacy ledger does not match Task-4 provenance binding")


def _shape_jobs(run: Path, config: CorpusRouterConfig) -> list[dict[str, Any]]:
    from .quality import validate_quality_bundle

    quality = validate_quality_bundle(run)
    _assert_task4_legacy_binding(run, config)
    legacy = validate_legacy_import(run, config)
    legacy_index_hash = _canonical_hash(list(legacy.records))
    quality_bundle_hash = _quality_hash(quality.ready)
    rows: list[dict[str, Any]] = []
    for ready in quality.ready:
        opportunity = ready["opportunity"]
        if not isinstance(opportunity, dict):
            raise ValueError("quality-ready opportunity is not an object")
        opportunity_id = str(opportunity["opportunity_id"])
        fields = {
            "assumption_x": str(opportunity["assumption_x"]), "failure_f": str(opportunity["failure_f"]),
            "missing_capability_w": str(opportunity["missing_capability_w"]), "condition_z": str(opportunity["condition_z"]),
            "scope_compatibility": str(opportunity["scope_compatibility"]),
        }
        job_id = stable_id("shape_assignment", opportunity_id, str(ready["candidate_sha256"]), legacy_index_hash)
        rows.append({
            "schema_version": "idea_factory.shape_assignment_job.v1", "job_id": job_id,
            "result_schema_version": "idea_factory.shape_assignment_result.v1",
            "opportunity_id": opportunity_id, "quality_record_sha256": str(ready["candidate_sha256"]),
            "quality_bundle_sha256": quality_bundle_hash, "legacy_index_sha256": legacy_index_hash,
            **fields,
        })
    return sorted(rows, key=lambda row: (row["opportunity_id"], row["job_id"]))


def emit_shape_assignment_jobs(run_dir: Path, config: CorpusRouterConfig) -> Path:
    """Emit deterministic no-LLM assignment requests for quality-ready opportunities."""

    run = _anchored_run(Path(run_dir))
    ledger = _owned_dir(run, "ledger", create=False)
    _safe_unlink(ledger, [*_JOB_NAMES, *_RESULT_NAMES, *_DEDUP_NAMES])
    try:
        jobs = _shape_jobs(run, config)
        target = ledger / _JOB_NAMES[0]
        write_jsonl_bundle({target: jobs})
        return target
    except BaseException:
        _safe_unlink(ledger, [*_JOB_NAMES, *_RESULT_NAMES, *_DEDUP_NAMES])
        raise


class _ShapeAssignmentResult(FrozenStrictModel):
    schema_version: Literal["idea_factory.shape_assignment_result.v1"]
    job_id: NonEmptyStr
    opportunity_id: NonEmptyStr
    quality_record_sha256: NonEmptyStr
    quality_bundle_sha256: NonEmptyStr
    legacy_index_sha256: NonEmptyStr
    old_assumption: str
    failure_mechanism: NonEmptyStr
    missing_capability: NonEmptyStr
    target_scope: NonEmptyStr
    mechanism_family: NonEmptyStr
    residual_difference: str = ""


def _strict_json(text: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value} is forbidden")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    data = json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )
    if type(data) is not dict:
        raise ValueError("shape assignment result must be one JSON object")
    return data


def _read_raw(value: str | bytes | Path) -> str:
    raw = value.read_bytes() if isinstance(value, Path) else value.encode("utf-8") if type(value) is str else value
    if type(raw) is not bytes:
        raise TypeError("shape assignment result must be JSON text, bytes, or path")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("shape assignment result must be UTF-8") from exc


def _project_assignments(jobs: Sequence[Mapping[str, Any]], raw_by_job: Mapping[str, str]) -> list[dict[str, Any]]:
    if set(raw_by_job) != {str(job["job_id"]) for job in jobs}:
        raise ValueError("shape assignment results must exactly cover one result per job")
    projected: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job["job_id"])
        raw_text = raw_by_job[job_id]
        try:
            result = _ShapeAssignmentResult.model_validate_json(json.dumps(_strict_json(raw_text), ensure_ascii=False, sort_keys=True, separators=(",", ":")), strict=True)
        except (ValidationError, ValueError) as exc:
            raise ValueError(f"invalid shape assignment result for {job_id}") from exc
        expected = {key: job[key] for key in ("job_id", "opportunity_id", "quality_record_sha256", "quality_bundle_sha256", "legacy_index_sha256")}
        if (
            result.schema_version != job.get("result_schema_version")
            or any(getattr(result, key) != value for key, value in expected.items())
        ):
            raise ValueError(f"stale shape assignment result binding for {job_id}")
        assignment = ShapeAssignment.model_validate({
            "source_opportunity_id": result.opportunity_id, "old_assumption": result.old_assumption,
            "failure_mechanism": result.failure_mechanism, "missing_capability": result.missing_capability,
            "target_scope": result.target_scope, "mechanism_family": result.mechanism_family,
            "residual_difference": result.residual_difference,
        })
        projected.append({
            "schema_version": "idea_factory.shape_assignment_projection.v1", "record_id": assignment.source_opportunity_id,
            "job_id": job_id, "raw_result_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "raw_result_json": raw_text, "job_sha256": _canonical_hash(job),
            "assignment": assignment.model_dump(mode="json"),
            "fingerprint_available": True,
            "shape_fingerprint": assignment.fingerprint(),
            "mechanism_target_key": mechanism_target_key(assignment.mechanism_family, assignment.target_scope),
        })
    return sorted(projected, key=lambda row: (row["record_id"], row["job_id"]))


def ingest_shape_assignments(jobs_path: Path, results: Sequence[str | bytes | Path]) -> Path:
    run = _anchored_run(Path(jobs_path).parent.parent)
    ledger = _owned_dir(run, "ledger", create=False)
    jobs_file = _owned_file(ledger, _JOB_NAMES[0])
    if Path(os.path.abspath(jobs_path)) != jobs_file:
        raise ValueError("shape jobs must use the exact active run path")
    jobs = read_jsonl(jobs_file)
    raw_by_job: dict[str, str] = {}
    for item in results:
        text = _read_raw(item)
        job_id = _strict_json(text).get("job_id")
        if type(job_id) is not str or job_id in raw_by_job:
            raise ValueError("shape assignment results contain unknown or duplicate job_id")
        raw_by_job[job_id] = text
    projections = _project_assignments(jobs, raw_by_job)
    target = ledger / _RESULT_NAMES[0]
    _safe_unlink(ledger, [*_RESULT_NAMES, *_DEDUP_NAMES])
    try:
        write_jsonl_bundle({target: projections})
        return target
    except BaseException:
        _safe_unlink(ledger, [*_RESULT_NAMES, *_DEDUP_NAMES])
        raise


@dataclass(frozen=True)
class ShapeAssignmentBundle:
    run: Path
    jobs: tuple[dict[str, Any], ...]
    assignments: tuple[dict[str, Any], ...]
    legacy_records: tuple[dict[str, Any], ...]


def validate_shape_assignment_bundle(run_dir: Path, config: CorpusRouterConfig) -> ShapeAssignmentBundle:
    run = _anchored_run(Path(run_dir))
    ledger = _owned_dir(run, "ledger", create=False)
    jobs = read_jsonl(_owned_file(ledger, _JOB_NAMES[0]))
    expected_jobs = _shape_jobs(run, config)
    if jobs != expected_jobs:
        raise ValueError("shape assignment jobs replay mismatch")
    assignments = read_jsonl(_owned_file(ledger, _RESULT_NAMES[0]))
    raw_by_job: dict[str, str] = {}
    for row in assignments:
        raw = row.get("raw_result_json")
        if type(raw) is not str or hashlib.sha256(raw.encode("utf-8")).hexdigest() != row.get("raw_result_sha256"):
            raise ValueError("shape assignment raw replay binding mismatch")
        job_id = row.get("job_id")
        if type(job_id) is not str or job_id in raw_by_job:
            raise ValueError("duplicate shape assignment projection")
        raw_by_job[job_id] = raw
    if assignments != _project_assignments(jobs, raw_by_job):
        raise ValueError("shape assignment projection replay mismatch")
    legacy = validate_legacy_import(run, config)
    return ShapeAssignmentBundle(run, tuple(jobs), tuple(assignments), tuple(legacy.records))


def _dedup_projections(bundle: ShapeAssignmentBundle) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ready: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    adjacent: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for row in bundle.assignments:
        assignment = ShapeAssignment.model_validate(row["assignment"])
        decision = classify_shape(assignment, bundle.legacy_records)
        outcome = {
            "schema_version": "idea_factory.internal_dedup_decision.v1", "record_id": assignment.source_opportunity_id,
            "assignment_sha256": _canonical_hash(row), "shape_fingerprint": decision.fingerprint,
            "mechanism_target_key": decision.mechanism_target_key, "dedup_policy_version": DEDUP_POLICY_VERSION,
            "target_overlap_threshold": TARGET_OVERLAP_THRESHOLD, "status": decision.status, "blocked": decision.blocked,
            "reason": decision.reason, "nearest_legacy_ids": list(decision.nearest_legacy_ids),
            "nearest_evidence": list(decision.nearest_evidence), "assignment": assignment.model_dump(mode="json"),
        }
        outcomes.append(outcome)
        if decision.status in {"INTERNAL_ADJACENT", "ADJACENT_NEEDS_RESIDUAL"}:
            adjacent.append(outcome)
        (blocked if decision.blocked else ready).append(outcome)
    for group in (ready, blocked, adjacent, outcomes):
        group.sort(key=lambda row: row["record_id"])
    return ready, blocked, adjacent, outcomes


def publish_internal_dedup(run_dir: Path, config: CorpusRouterConfig) -> dict[str, Path]:
    bundle = validate_shape_assignment_bundle(run_dir, config)
    ledger = _owned_dir(bundle.run, "ledger", create=False)
    targets = {name: ledger / name for name in _DEDUP_NAMES}
    _safe_unlink(ledger, _DEDUP_NAMES)
    try:
        ready, blocked, adjacent, outcomes = _dedup_projections(bundle)
        write_jsonl_bundle({targets[_DEDUP_NAMES[0]]: ready, targets[_DEDUP_NAMES[1]]: blocked, targets[_DEDUP_NAMES[2]]: adjacent, targets[_DEDUP_NAMES[3]]: outcomes})
        return {"ready": targets[_DEDUP_NAMES[0]], "blocked": targets[_DEDUP_NAMES[1]], "adjacent": targets[_DEDUP_NAMES[2]], "outcomes": targets[_DEDUP_NAMES[3]]}
    except BaseException:
        _safe_unlink(ledger, _DEDUP_NAMES)
        raise


def validate_dedup_bundle(run_dir: Path, config: CorpusRouterConfig) -> dict[str, tuple[dict[str, Any], ...]]:
    bundle = validate_shape_assignment_bundle(run_dir, config)
    ledger = _owned_dir(bundle.run, "ledger", create=False)
    actual = tuple(read_jsonl(_owned_file(ledger, name)) for name in _DEDUP_NAMES)
    expected = _dedup_projections(bundle)
    if actual != tuple(expected):
        raise ValueError("internal dedup bundle replay mismatch")
    return {"ready": tuple(actual[0]), "blocked": tuple(actual[1]), "adjacent": tuple(actual[2]), "outcomes": tuple(actual[3])}


HUMAN_REASON_CODES = (
    "WANT_TO_TEST_NOW",
    "INTERESTING_BUT_TOO_EXPENSIVE",
    "NOVEL_BUT_UNIMPORTANT",
    "IMPORTANT_BUT_ALREADY_COVERED",
    "MECHANISM_NOT_CONVINCING",
    "TOO_INCREMENTAL",
    "NOT_MY_RESEARCH_PRIORITY",
)
HUMAN_SCORE_REASON_CODES = HUMAN_REASON_CODES[:-1]
HUMAN_SKIP_REASON_CODES = ("NOT_MY_RESEARCH_PRIORITY",)
_HUMAN_JOB_NAME = "human_scoring_jobs.jsonl"
_HUMAN_ELIGIBILITY_NAME = "human_scoring_eligibility.jsonl"
_HUMAN_REPORT_NAME = "human_scoring_report.json"
_HUMAN_OUTPUT_NAMES = ("human_scores.jsonl", "human_scoring_outcomes.jsonl", "human_bundle_manifest.json")
_HUMAN_JOB_SCHEMA_VERSION = "idea_factory.human_scoring_job.v2"
_HUMAN_OUTCOME_SCHEMA_VERSION = "idea_factory.human_score_outcome.v2"
_HUMAN_MANIFEST_SCHEMA_VERSION = "idea_factory.human_bundle_manifest.v2"
_SCORE_FIELDS = (
    "specific_novelty", "importance", "paper_potential", "feasibility", "excitement", "evidence_clarity",
)


def _human_result_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "schema_version": {"const": "idea_factory.human_score.v1", "type": "string"},
        "job_id": {"minLength": 1, "type": "string"},
        "human_job_sha256": {"pattern": "^[0-9a-f]{64}$", "type": "string"},
        **{field: {"maximum": 5, "minimum": 0, "type": "integer"} for field in _SCORE_FIELDS},
        "reason_codes": {
            "items": {"enum": list(HUMAN_SCORE_REASON_CODES), "type": "string"},
            "minItems": 1, "type": "array", "uniqueItems": True,
        },
        "human_identity": {"minLength": 1, "type": "string"},
        "attested_at": {"format": "date-time", "type": "string"},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "allOf": [{
            "if": {
                "properties": {
                    "reason_codes": {"contains": {"const": "WANT_TO_TEST_NOW"}},
                },
                "required": ["reason_codes"],
            },
            "then": {"properties": {"reason_codes": {"maxItems": 1}}},
        }],
        "properties": properties,
        "required": list(properties),
        "type": "object",
    }


def _human_skip_result_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "schema_version": {"const": "idea_factory.human_skip.v1", "type": "string"},
        "job_id": {"minLength": 1, "type": "string"},
        "human_job_sha256": {"pattern": "^[0-9a-f]{64}$", "type": "string"},
        "reason_codes": {
            "items": {"const": HUMAN_SKIP_REASON_CODES[0], "type": "string"},
            "maxItems": 1,
            "minItems": 1,
            "type": "array",
            "uniqueItems": True,
        },
        "human_identity": {"minLength": 1, "type": "string"},
        "attested_at": {"format": "date-time", "type": "string"},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
        "type": "object",
    }


def _terminal_review_projection(review_bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = {str(row["job_id"]): row for row in review_bundle["results"]}
    outcomes = {str(row["job_id"]): row for row in review_bundle["outcomes"]}
    by_route: dict[str, list[Mapping[str, Any]]] = {}
    route_order: list[str] = []
    for job in review_bundle["jobs"]:
        route_id = str(job["route_id"])
        if route_id not in by_route:
            route_order.append(route_id)
            by_route[route_id] = []
        by_route[route_id].append(job)
    projected: list[dict[str, Any]] = []
    for route_id in route_order:
        terminal = max(by_route[route_id], key=lambda row: int(row["round"]))
        outcome = outcomes[str(terminal["job_id"])]
        result = results.get(str(terminal["job_id"]))
        decision = result.get("decision") if result is not None else None
        if outcome.get("status") != "ACCEPTED" or result is None:
            status = "UNRESOLVED"
        elif decision == "PASS_TO_HUMAN":
            status = "PENDING_HUMAN"
        elif decision == "KILL":
            status = "KILLED_REVIEW"
        else:
            status = "PENDING_REVIEW"
        projected.append({
            "schema_version": "idea_factory.human_scoring_eligibility.v1",
            "route_id": route_id,
            "opportunity_id": terminal["opportunity_id"],
            "terminal_review_job_id": terminal["job_id"],
            "terminal_round": terminal["round"],
            "review_decision": decision,
            "review_outcome_status": outcome["status"],
            "status": status,
            "stage": "HUMAN_SCORING",
            "review_job_sha256": terminal["review_job_sha256"],
            "review_record_sha256": _canonical_hash(result) if result is not None else None,
        })
    return projected


def _human_report(
    eligibility: Sequence[Mapping[str, Any]], opportunity_input_count: int,
) -> dict[str, Any]:
    statuses = [str(row["status"]) for row in eligibility]
    if not statuses:
        classification = "ZERO_INPUT" if opportunity_input_count == 0 else "ZERO_SURVIVOR"
    elif "PENDING_HUMAN" in statuses:
        classification = "PENDING"
    elif "UNRESOLVED" in statuses:
        classification = "UNRESOLVED"
    elif "PENDING_REVIEW" in statuses:
        classification = "PENDING"
    else:
        classification = "ZERO_SURVIVOR"
    return {
        "schema_version": "idea_factory.human_scoring_report.v1",
        "classification": classification,
        "source_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "opportunity_input_count": opportunity_input_count,
        "route_count": len(statuses),
        "human_job_count": statuses.count("PENDING_HUMAN"),
        "operational_completion_is_idea_yield": False,
    }


def _human_job_projection(review_bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligibility = _terminal_review_projection(review_bundle)
    source_jobs = {str(row["job_id"]): row for row in review_bundle["jobs"]}
    source_results = {str(row["job_id"]): row for row in review_bundle["results"]}
    jobs: list[dict[str, Any]] = []
    for row in eligibility:
        if row["status"] != "PENDING_HUMAN":
            continue
        source_job = source_jobs[str(row["terminal_review_job_id"])]
        source_result = source_results[str(row["terminal_review_job_id"])]
        authenticity = {
            "authenticity": source_job["authenticity"],
            "authenticity_boundary": source_job["authenticity_boundary"],
            "attestation_scope": source_job["attestation_scope"],
            "operator_attested_execution_ready": source_job["operator_attested_execution_ready"],
        }
        result_schema = _human_result_schema()
        skip_result_schema = _human_skip_result_schema()
        body: dict[str, Any] = {
            "schema_version": _HUMAN_JOB_SCHEMA_VERSION,
            "job_id": stable_id(
                "human_score", source_job["route_id"], _canonical_hash(source_result),
                review_bundle["manifest"]["bundle_manifest_sha256"],
            ),
            "result_schema_version": "idea_factory.human_score.v1",
            "result_schema": result_schema,
            "result_schema_sha256": _canonical_hash(result_schema),
            "skip_result_schema_version": "idea_factory.human_skip.v1",
            "skip_result_schema": skip_result_schema,
            "skip_result_schema_sha256": _canonical_hash(skip_result_schema),
            "route_id": source_job["route_id"],
            "opportunity_id": source_job["opportunity_id"],
            "opportunity": source_job["opportunity"],
            "route": source_job["route"],
            "recon_provenance": source_job["recon_provenance"],
            "nearest_priors": source_job["route"]["nearest_prior_bindings"],
            "review": source_result,
            "opportunity_sha256": source_job["opportunity_sha256"],
            "route_sha256": source_job["route_record_sha256"],
            "recon_provenance_sha256": _canonical_hash(source_job["recon_provenance"]),
            "nearest_priors_sha256": _canonical_hash(source_job["route"]["nearest_prior_bindings"]),
            "review_record_sha256": _canonical_hash(source_result),
            "review_bundle_manifest_sha256": review_bundle["manifest"]["bundle_manifest_sha256"],
            "authenticity_binding": authenticity,
            "authenticity_sha256": _canonical_hash(authenticity),
            "approved_reason_codes": list(HUMAN_SCORE_REASON_CODES),
            "approved_skip_reason_codes": list(HUMAN_SKIP_REASON_CODES),
            "score_fields": list(_SCORE_FIELDS),
            "score_range": [0, 5],
            "boolean_scores_forbidden": True,
            "automatic_scoring_forbidden": True,
            "accepted_job_immutable": True,
            "rejected_job_retryable": True,
        }
        body["human_job_sha256"] = _canonical_hash(body)
        jobs.append(body)
    return jobs, eligibility


def emit_human_scoring_jobs(
    run_dir: Path,
    config: CorpusRouterConfig,
    *,
    prompt_path: Path | None = None,
    route_prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> Path:
    from .review import validate_review_bundle

    run = _anchored_run(Path(run_dir))
    ledger = _owned_dir(run, "ledger", create=False)
    jobs_path = ledger / _HUMAN_JOB_NAME
    eligibility_path = ledger / _HUMAN_ELIGIBILITY_NAME
    report_path = ledger / _HUMAN_REPORT_NAME
    with stage_mutation_lock(ledger):
        review_bundle = validate_review_bundle(
            run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
            allow_test_ready=allow_test_ready,
        )
        opportunity_input_count = len(validate_dedup_bundle(run, config)["outcomes"])
        jobs, eligibility = _human_job_projection(review_bundle)
        expected = {
            jobs_path: encode_jsonl(jobs),
            eligibility_path: encode_jsonl(eligibility),
            report_path: encode_json(_human_report(eligibility, opportunity_input_count)),
        }
        exists = [path.exists() for path in expected]
        if any(exists):
            if not all(exists):
                raise ValueError("existing human scoring handoff is incomplete")
            if any(path.read_bytes() != payload for path, payload in expected.items()):
                raise ValueError("existing human scoring handoff replay mismatch")
            return jobs_path
        publish_transaction(expected)
        return jobs_path


def _validate_human_jobs(
    run: Path,
    config: CorpusRouterConfig,
    *,
    prompt_path: Path | None,
    route_prompt_path: Path | None,
    allow_test_ready: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from .review import validate_review_bundle

    ledger = _owned_dir(run, "ledger", create=False)
    review_bundle = validate_review_bundle(
        run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
        allow_test_ready=allow_test_ready,
    )
    jobs, eligibility = _human_job_projection(review_bundle)
    opportunity_input_count = len(validate_dedup_bundle(run, config)["outcomes"])
    if read_jsonl(_owned_file(ledger, _HUMAN_JOB_NAME)) != jobs:
        raise ValueError("human scoring jobs replay mismatch")
    if read_jsonl(_owned_file(ledger, _HUMAN_ELIGIBILITY_NAME)) != eligibility:
        raise ValueError("human scoring eligibility replay mismatch")
    report = json.loads(_owned_file(ledger, _HUMAN_REPORT_NAME).read_text(encoding="utf-8"))
    if report != _human_report(eligibility, opportunity_input_count):
        raise ValueError("human scoring report replay mismatch")
    return jobs, eligibility, report


def _human_rejection(job: Mapping[str, Any], raw_sha: str, code: str) -> dict[str, Any]:
    errors = {
        "HUMAN_SCORE_SCHEMA_INVALID": "human score schema is invalid",
        "HUMAN_SCORE_BINDING_INVALID": "human score binding is stale",
        "HUMAN_SCORE_SECRET_DETECTED": "human score contains credential-shaped content",
        "HUMAN_REASON_CONFLICT": "WANT_TO_TEST_NOW cannot be combined with another reason code",
    }
    return {
        "schema_version": _HUMAN_OUTCOME_SCHEMA_VERSION,
        "job_id": job["job_id"],
        "route_id": job["route_id"],
        "opportunity_id": job["opportunity_id"],
        "status": "REJECTED",
        "raw_result_sha256": raw_sha,
        "error_code": code,
        "error": errors[code],
        "human_score_id": None,
        "accepted_projection_sha256": _canonical_hash([]),
    }


def _project_human_score(job: Mapping[str, Any], raw: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if contains_secret(raw):
        return [], _human_rejection(job, raw_sha, "HUMAN_SCORE_SECRET_DETECTED")
    try:
        payload = _strict_json(raw)
    except ValueError:
        return [], _human_rejection(job, raw_sha, "HUMAN_SCORE_SCHEMA_INVALID")
    if payload.get("schema_version") == job.get("skip_result_schema_version"):
        expected_skip_keys = {
            "schema_version", "job_id", "human_job_sha256", "reason_codes",
            "human_identity", "attested_at",
        }
        if (
            set(payload) != expected_skip_keys
            or payload.get("job_id") != job["job_id"]
            or payload.get("human_job_sha256") != job["human_job_sha256"]
            or payload.get("reason_codes") != list(HUMAN_SKIP_REASON_CODES)
        ):
            return [], _human_rejection(job, raw_sha, "HUMAN_SCORE_SCHEMA_INVALID")
        identity, attested_at = payload.get("human_identity"), payload.get("attested_at")
        if type(identity) is not str or not identity.strip() or type(attested_at) is not str:
            return [], _human_rejection(job, raw_sha, "HUMAN_SCORE_SCHEMA_INVALID")
        try:
            timestamp = datetime.fromisoformat(attested_at.replace("Z", "+00:00"))
        except ValueError:
            return [], _human_rejection(job, raw_sha, "HUMAN_SCORE_SCHEMA_INVALID")
        if timestamp.tzinfo is None:
            return [], _human_rejection(job, raw_sha, "HUMAN_SCORE_SCHEMA_INVALID")
        return [], {
            "schema_version": _HUMAN_OUTCOME_SCHEMA_VERSION,
            "job_id": job["job_id"],
            "route_id": job["route_id"],
            "opportunity_id": job["opportunity_id"],
            "status": "SKIPPED",
            "raw_result_sha256": raw_sha,
            "raw_result_json": raw,
            "error_code": None,
            "error": None,
            "human_score_id": None,
            "accepted_projection_sha256": _canonical_hash([]),
            "reason_codes": list(HUMAN_SKIP_REASON_CODES),
            "human_identity": identity,
            "attested_at": attested_at,
        }
    expected_keys = {
        "schema_version", "job_id", "human_job_sha256", *_SCORE_FIELDS,
        "reason_codes", "human_identity", "attested_at",
    }
    if set(payload) != expected_keys:
        return [], _human_rejection(job, raw_sha, "HUMAN_SCORE_SCHEMA_INVALID")
    if (
        payload.get("schema_version") != job["result_schema_version"]
        or payload.get("job_id") != job["job_id"]
        or payload.get("human_job_sha256") != job["human_job_sha256"]
    ):
        return [], _human_rejection(job, raw_sha, "HUMAN_SCORE_BINDING_INVALID")
    if any(type(payload.get(field)) is not int or not 0 <= payload[field] <= 5 for field in _SCORE_FIELDS):
        return [], _human_rejection(job, raw_sha, "HUMAN_SCORE_SCHEMA_INVALID")
    reasons = payload.get("reason_codes")
    if (
        type(reasons) is not list or not reasons or any(type(code) is not str for code in reasons)
        or len(set(reasons)) != len(reasons) or not set(reasons) <= set(HUMAN_SCORE_REASON_CODES)
    ):
        return [], _human_rejection(job, raw_sha, "HUMAN_SCORE_SCHEMA_INVALID")
    if "WANT_TO_TEST_NOW" in reasons and len(reasons) != 1:
        return [], _human_rejection(job, raw_sha, "HUMAN_REASON_CONFLICT")
    identity, attested_at = payload.get("human_identity"), payload.get("attested_at")
    if type(identity) is not str or not identity.strip() or type(attested_at) is not str:
        return [], _human_rejection(job, raw_sha, "HUMAN_SCORE_SCHEMA_INVALID")
    try:
        timestamp = datetime.fromisoformat(attested_at.replace("Z", "+00:00"))
    except ValueError:
        return [], _human_rejection(job, raw_sha, "HUMAN_SCORE_SCHEMA_INVALID")
    if timestamp.tzinfo is None:
        return [], _human_rejection(job, raw_sha, "HUMAN_SCORE_SCHEMA_INVALID")
    record = {
        "schema_version": "idea_factory.human_score_record.v1",
        "human_score_id": stable_id("human_score_record", job["job_id"], raw_sha),
        "job_id": job["job_id"],
        "route_id": job["route_id"],
        "opportunity_id": job["opportunity_id"],
        **{field: payload[field] for field in _SCORE_FIELDS},
        "reason_codes": reasons,
        "human_identity": identity,
        "attested_at": attested_at,
        "human_job_sha256": job["human_job_sha256"],
        "raw_result_sha256": raw_sha,
    }
    outcome = {
        "schema_version": _HUMAN_OUTCOME_SCHEMA_VERSION,
        "job_id": job["job_id"],
        "route_id": job["route_id"],
        "opportunity_id": job["opportunity_id"],
        "status": "ACCEPTED",
        "raw_result_sha256": raw_sha,
        "raw_result_json": raw,
        "error_code": None,
        "error": None,
        "human_score_id": record["human_score_id"],
        "accepted_projection_sha256": _canonical_hash(record),
    }
    return [record], outcome


def _human_manifest(
    jobs: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    body = {
        "schema_version": _HUMAN_MANIFEST_SCHEMA_VERSION,
        "jobs_sha256": hashes["jobs"],
        "results_sha256": hashes["results"],
        "outcomes_sha256": hashes["outcomes"],
        "job_count": len(jobs),
        "accepted_count": sum(row["status"] == "ACCEPTED" for row in outcomes),
        "skipped_count": sum(row["status"] == "SKIPPED" for row in outcomes),
        "rejected_count": sum(row["status"] == "REJECTED" for row in outcomes),
        "automatic_scoring_used": False,
    }
    return body | {"bundle_manifest_sha256": _canonical_hash(body)}


def _validate_accepted_human_replays(
    current: Mapping[str, Any] | None,
    results: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> None:
    if current is None:
        return
    incoming_results = {str(row["job_id"]): row for row in results}
    incoming_outcomes = {str(row["job_id"]): row for row in outcomes}
    current_results = {str(row["job_id"]): row for row in current["results"]}
    for old in current["outcomes"]:
        if old["status"] != "ACCEPTED":
            continue
        job_id = str(old["job_id"])
        new = incoming_outcomes.get(job_id)
        if (
            new is None or new.get("status") != "ACCEPTED"
            or new.get("raw_result_sha256") != old.get("raw_result_sha256")
            or new.get("accepted_projection_sha256") != old.get("accepted_projection_sha256")
            or incoming_results.get(job_id) != current_results.get(job_id)
        ):
            raise ValueError("accepted human score is immutable; only exact replay is allowed")


def ingest_human_scores(
    jobs_path: Path,
    results: Sequence[str | bytes | Path],
    config: CorpusRouterConfig,
    *,
    prompt_path: Path | None = None,
    route_prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> dict[str, Path]:
    run = _anchored_run(Path(jobs_path).parent.parent)
    ledger = _owned_dir(run, "ledger", create=False)
    with stage_mutation_lock(ledger):
        if Path(os.path.abspath(jobs_path)) != _owned_file(ledger, _HUMAN_JOB_NAME):
            raise ValueError("human jobs must use the exact active run path")
        jobs, _eligibility, _report = _validate_human_jobs(
            run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
            allow_test_ready=allow_test_ready,
        )
        output_paths = [ledger / name for name in _HUMAN_OUTPUT_NAMES]
        exists = [path.exists() for path in output_paths]
        if any(exists) and not all(exists):
            raise ValueError("existing human score bundle is incomplete")
        current = validate_human_bundle(
            run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
            allow_test_ready=allow_test_ready,
        ) if all(exists) else None
        if len(results) != len(jobs):
            raise ValueError("human score results must exactly cover canonical jobs")
        accepted: list[dict[str, Any]] = []
        outcomes: list[dict[str, Any]] = []
        for job, source in zip(jobs, results):
            raw = _read_raw(source)
            records, outcome = _project_human_score(job, raw)
            accepted.extend(records); outcomes.append(outcome)
        _validate_accepted_human_replays(current, accepted, outcomes)
        results_bytes = encode_jsonl(accepted)
        outcomes_bytes = encode_jsonl(outcomes)
        hashes = {
            "jobs": hashlib.sha256((ledger / _HUMAN_JOB_NAME).read_bytes()).hexdigest(),
            "results": hashlib.sha256(results_bytes).hexdigest(),
            "outcomes": hashlib.sha256(outcomes_bytes).hexdigest(),
        }
        manifest_bytes = encode_json(_human_manifest(jobs, accepted, outcomes, hashes))
        paths = {
            "results": ledger / _HUMAN_OUTPUT_NAMES[0],
            "outcomes": ledger / _HUMAN_OUTPUT_NAMES[1],
            "manifest": ledger / _HUMAN_OUTPUT_NAMES[2],
        }
        publish_transaction({paths["results"]: results_bytes, paths["outcomes"]: outcomes_bytes, paths["manifest"]: manifest_bytes})
        return paths


def validate_human_bundle(
    run_dir: Path,
    config: CorpusRouterConfig,
    *,
    prompt_path: Path | None = None,
    route_prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> dict[str, Any]:
    run = _anchored_run(Path(run_dir))
    ledger = _owned_dir(run, "ledger", create=False)
    jobs, eligibility, report = _validate_human_jobs(
        run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
        allow_test_ready=allow_test_ready,
    )
    results = read_jsonl(_owned_file(ledger, _HUMAN_OUTPUT_NAMES[0]))
    outcomes = read_jsonl(_owned_file(ledger, _HUMAN_OUTPUT_NAMES[1]))
    manifest = json.loads(_owned_file(ledger, _HUMAN_OUTPUT_NAMES[2]).read_text(encoding="utf-8"))
    if len(outcomes) != len(jobs):
        raise ValueError("human outcomes must exactly cover jobs")
    expected_results: list[dict[str, Any]] = []
    expected_outcomes: list[dict[str, Any]] = []
    for job, outcome in zip(jobs, outcomes):
        if outcome.get("status") in {"ACCEPTED", "SKIPPED"}:
            raw = outcome.get("raw_result_json")
            if type(raw) is not str:
                raise ValueError("accepted or skipped human outcome lacks replayable raw")
            projected, replayed = _project_human_score(job, raw)
            if replayed != outcome:
                raise ValueError("accepted or skipped human outcome replay mismatch")
            expected_results.extend(projected); expected_outcomes.append(replayed)
        else:
            allowed = {
                "HUMAN_SCORE_SCHEMA_INVALID": "human score schema is invalid",
                "HUMAN_SCORE_BINDING_INVALID": "human score binding is stale",
                "HUMAN_SCORE_SECRET_DETECTED": "human score contains credential-shaped content",
                "HUMAN_REASON_CONFLICT": "WANT_TO_TEST_NOW cannot be combined with another reason code",
            }
            if (
                set(outcome) != {
                    "schema_version", "job_id", "route_id", "opportunity_id", "status",
                    "raw_result_sha256", "error_code", "error", "human_score_id", "accepted_projection_sha256",
                }
                or outcome.get("schema_version") != _HUMAN_OUTCOME_SCHEMA_VERSION
                or outcome.get("status") != "REJECTED"
                or allowed.get(str(outcome.get("error_code"))) != outcome.get("error")
                or any(outcome.get(key) != job[key] for key in ("job_id", "route_id", "opportunity_id"))
            ):
                raise ValueError("rejected human outcome schema mismatch")
            expected_outcomes.append(outcome)
    if results != expected_results or outcomes != expected_outcomes:
        raise ValueError("human score projection replay mismatch")
    expected_manifest = _human_manifest(
        jobs, expected_results, expected_outcomes,
        {
            "jobs": hashlib.sha256((ledger / _HUMAN_JOB_NAME).read_bytes()).hexdigest(),
            "results": hashlib.sha256((ledger / _HUMAN_OUTPUT_NAMES[0]).read_bytes()).hexdigest(),
            "outcomes": hashlib.sha256((ledger / _HUMAN_OUTPUT_NAMES[1]).read_bytes()).hexdigest(),
        },
    )
    if manifest != expected_manifest:
        raise ValueError("human bundle manifest replay mismatch")
    return {
        "jobs": tuple(jobs), "eligibility": tuple(eligibility), "results": tuple(results),
        "outcomes": tuple(outcomes), "manifest": manifest, "report": report,
    }


def _repository_data_dir(repository_root: Path) -> Path:
    root = Path(os.path.abspath(repository_root))
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("repository root is missing") from exc
    if not root.is_dir() or resolved != root:
        raise ValueError("repository root must have exact resolved identity")
    data = root / "data"
    if not data.exists():
        data.mkdir()
    if not data.is_dir() or data.resolve(strict=True) != data:
        raise ValueError("repository data path escapes exact location")
    for child in data.iterdir():
        if child.name == ".stage.lock":
            continue
        if child.resolve(strict=True).parent != data:
            raise ValueError("repository data artifact escapes exact location")
    return data


def _pack_records(run: Path, manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    directory = _owned_dir(run, "idea_packs", create=False)
    records: dict[str, dict[str, Any]] = {}
    for idea_id in manifest["idea_ids"]:
        payload = json.loads(_owned_file(directory, f"{idea_id}.json").read_text(encoding="utf-8"))
        if payload.get("idea_id") != idea_id:
            raise ValueError("idea pack identity mismatch")
        records[str(idea_id)] = payload
    return records


def _run_identity(run: Path) -> str:
    state_path = run / "state.json"
    if state_path.exists():
        if not state_path.is_file() or state_path.resolve(strict=True) != state_path:
            raise ValueError("run state path is not anchored")
        state = _strict_json(state_path.read_text(encoding="utf-8"))
        run_id = state.get("run_id")
        if type(run_id) is not str or not run_id.strip():
            raise ValueError("run state lacks a valid run_id")
        return run_id
    return stable_id("source_run", str(run))


def _source_record_id(record: Mapping[str, Any]) -> str:
    for key in ("record_id", "job_id", "idea_id"):
        value = record.get(key)
        if type(value) is str and value.strip():
            return value
    raise ValueError("ledger source record lacks a stable record identifier")


def _source_locator(
    run: Path,
    artifact_relpath: str,
    source: Mapping[str, Any],
    source_record_index: int | None,
    source_subrecord_id: str | None,
) -> dict[str, Any]:
    source_sha = _canonical_hash(source)
    return {
        "resolved_run_root": str(run),
        "artifact_relpath": artifact_relpath,
        "artifact_kind": "JSON" if source_record_index is None else "JSONL",
        "source_record_id": _source_record_id(source),
        "source_record_index": source_record_index,
        "source_subrecord_id": source_subrecord_id,
        "source_record_sha256": source_sha,
    }


def validate_ledger_source_locator(event: Mapping[str, Any]) -> dict[str, Any]:
    """Recover and hash-check the exact source row named by one v2 ledger event."""

    locator = event.get("source_locator")
    if type(locator) is not dict or set(locator) != {
        "resolved_run_root", "artifact_relpath", "artifact_kind", "source_record_id",
        "source_record_index", "source_subrecord_id", "source_record_sha256",
    }:
        raise ValueError("ledger source locator schema is invalid")
    root = Path(str(locator["resolved_run_root"]))
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("ledger source run root is missing or unanchored")
    relpath = locator["artifact_relpath"]
    if type(relpath) is not str or not relpath or Path(relpath).is_absolute() or ".." in Path(relpath).parts:
        raise ValueError("ledger source artifact relpath is invalid")
    artifact = root / Path(relpath)
    if not artifact.is_file() or artifact.resolve(strict=True) != artifact or not artifact.is_relative_to(root):
        raise ValueError("ledger source artifact is missing or escapes the run root")
    kind = locator["artifact_kind"]
    index = locator["source_record_index"]
    if kind == "JSONL":
        if type(index) is not int or index < 0:
            raise ValueError("JSONL source locator requires a nonnegative record index")
        records = read_jsonl(artifact)
        if index >= len(records):
            raise ValueError("ledger source record index is out of range")
        record = records[index]
    elif kind == "JSON":
        if index is not None:
            raise ValueError("JSON source locator record index must be null")
        record = _strict_json(artifact.read_text(encoding="utf-8"))
    else:
        raise ValueError("ledger source artifact kind is invalid")
    source_sha = _canonical_hash(record)
    subrecord_id = locator["source_subrecord_id"]
    if subrecord_id is not None and (
        type(subrecord_id) is not str
        or subrecord_id not in record.get("route_ids", [])
    ):
        raise ValueError("ledger source subrecord is not bound by its source row")
    if (
        _source_record_id(record) != locator["source_record_id"]
        or source_sha != locator["source_record_sha256"]
        or source_sha != event.get("source_record_sha256")
    ):
        raise ValueError("ledger source locator record identity or hash mismatch")
    return record


def _ledger_update_body(
    *,
    run_id: str,
    source_run_id: str,
    ledger_transaction_id: str,
    update_id: str,
    entity_type: str,
    entity_id: str,
    opportunity_id: str,
    route_id: str | None,
    status: str,
    stage: str,
    reason_codes: Sequence[str],
    nearest_priors: Sequence[Mapping[str, Any]],
    source_record_sha256: str,
    source_locator: Mapping[str, Any],
    opportunity_fingerprint: str,
    mechanism_fingerprint: str | None,
    mechanism_fingerprint_status: str,
    supporting_card_ids: Sequence[str],
    supporting_paper_ids: Sequence[str],
    human_scores: Mapping[str, int] | None,
    authenticity: str,
    authenticity_boundary: str,
    attestation_scope: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": "idea_factory.ledger_update.v2",
        "run_id": run_id,
        "source_run_id": source_run_id,
        "ledger_transaction_id": ledger_transaction_id,
        "ledger_update_id": update_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "opportunity_id": opportunity_id,
        "route_id": route_id,
        "status": status,
        "stage": stage,
        "reason_codes": list(reason_codes),
        "nearest_priors": list(nearest_priors),
        "source_record_sha256": source_record_sha256,
        "source_locator": dict(source_locator),
        "opportunity_fingerprint": opportunity_fingerprint,
        "mechanism_fingerprint": mechanism_fingerprint,
        "mechanism_fingerprint_status": mechanism_fingerprint_status,
        "supporting_card_ids": list(supporting_card_ids),
        "supporting_paper_ids": list(supporting_paper_ids),
        "human_scores": dict(human_scores) if human_scores is not None else None,
        "authenticity": authenticity,
        "authenticity_boundary": authenticity_boundary,
        "attestation_scope": attestation_scope,
        "cheap_test_done": False,
        "subsequent_result": None,
        "provenance": dict(provenance),
    }
    return body | {"update_sha256": _canonical_hash(body)}


def build_expected_ledger_events(
    run_dir: Path,
    config: CorpusRouterConfig,
    *,
    prompt_path: Path | None = None,
    route_prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> list[dict[str, Any]]:
    """Rebuild the only valid current-run ledger projection from validated artifacts."""

    from .cards import validate_card_result_bundle
    from .quality import validate_quality_bundle
    from .recon import validate_execution_context, validate_recon_bundle, validate_report_jobs
    from .review import validate_review_bundle
    from .render import validate_idea_pack_bundle
    from .routes import validate_route_bundle, validate_route_jobs

    run = _anchored_run(Path(run_dir))
    quality = validate_quality_bundle(run)
    dedup = validate_dedup_bundle(run, config)
    recon = validate_recon_bundle(run, config)
    recon_jobs = validate_report_jobs(run, config)
    routes = validate_route_bundle(
        run, config, prompt_path=route_prompt_path, allow_test_ready=allow_test_ready,
    )
    route_jobs = validate_route_jobs(
        run, config, prompt_path=route_prompt_path, allow_test_ready=allow_test_ready,
    )
    review = validate_review_bundle(
        run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
        allow_test_ready=allow_test_ready,
    )
    human = validate_human_bundle(
        run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
        allow_test_ready=allow_test_ready,
    )
    pack_manifest = validate_idea_pack_bundle(
        run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
        allow_test_ready=allow_test_ready,
    )
    packs = _pack_records(run, pack_manifest)
    _card_paths, accepted_cards, _rejected_cards, _card_outcomes = validate_card_result_bundle(
        run / "cards" / "card_jobs.jsonl", run,
    )
    paper_by_card_id = {str(row["record_id"]): str(row["slug"]) for row in accepted_cards}
    opportunity_records = {
        str(row["record_id"]): row["opportunity"]
        for row in (*quality.ready, *quality.rejected)
    }
    run_id = _run_identity(run)
    transaction_basis = {
        "run_id": run_id,
        "quality_outcomes": _canonical_hash(quality.outcomes),
        "dedup_outcomes": _canonical_hash(dedup["outcomes"]),
        "recon_outcomes": _canonical_hash(recon["outcomes"]),
        "route_manifest": routes["manifest"]["bundle_manifest_sha256"],
        "review_manifest": review["manifest"]["bundle_manifest_sha256"],
        "human_manifest": human["manifest"]["bundle_manifest_sha256"],
        "idea_pack_manifest": pack_manifest["manifest_sha256"],
    }
    ledger_transaction_id = stable_id(
        "ledger_transaction", run_id, _canonical_hash(transaction_basis),
    )
    recon_context = validate_execution_context(run) if recon["outcomes"] else None
    updates: list[dict[str, Any]] = []

    def add_event(
        *, source_artifact: str, source: Mapping[str, Any], entity_type: str,
        entity_id: str, opportunity_id: str, route_id: str | None, status: str,
        stage: str, reason_codes: Sequence[str], nearest_priors: Sequence[Mapping[str, Any]] = (),
        source_record_index: int | None, mechanism_fingerprint: str | None,
        mechanism_fingerprint_status: str, human_scores: Mapping[str, int] | None,
        authenticity: str, authenticity_boundary: str, attestation_scope: str,
        source_subrecord_id: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        source_sha = _canonical_hash(source)
        opportunity = opportunity_records.get(opportunity_id)
        if type(opportunity) is not dict:
            raise ValueError("ledger event lacks its validated opportunity source")
        supporting_card_ids = [str(value) for value in opportunity["supporting_card_ids"]]
        try:
            supporting_paper_ids = list(dict.fromkeys(paper_by_card_id[card_id] for card_id in supporting_card_ids))
        except KeyError as exc:
            raise ValueError("ledger opportunity references an unvalidated supporting card") from exc
        locator = _source_locator(
            run, source_artifact, source, source_record_index, source_subrecord_id,
        )
        updates.append(_ledger_update_body(
            run_id=run_id, source_run_id=run_id,
            ledger_transaction_id=ledger_transaction_id,
            update_id=stable_id(
                "ledger_event", stage, source_artifact, entity_type, entity_id, source_sha,
            ),
            entity_type=entity_type, entity_id=entity_id, opportunity_id=opportunity_id,
            route_id=route_id, status=status, stage=stage, reason_codes=reason_codes,
            nearest_priors=nearest_priors, source_record_sha256=source_sha,
            source_locator=locator,
            opportunity_fingerprint=_canonical_hash(opportunity),
            mechanism_fingerprint=mechanism_fingerprint,
            mechanism_fingerprint_status=mechanism_fingerprint_status,
            supporting_card_ids=supporting_card_ids,
            supporting_paper_ids=supporting_paper_ids,
            human_scores=human_scores,
            authenticity=authenticity,
            authenticity_boundary=authenticity_boundary,
            attestation_scope=attestation_scope,
            provenance={
                "source_artifact": source_artifact,
                "source_row_sha256": source_sha,
                **dict(provenance or {}),
            },
        ))

    # Quality rejects are terminal evidence in their own right.  Do not wait for a
    # downstream stage to invent a status for an opportunity that never passed.
    for source_index, record in enumerate(quality.rejected):
        opportunity_id = str(record["record_id"])
        add_event(
            source_artifact="quality/rejected.jsonl", source=record,
            entity_type="OPPORTUNITY", entity_id=opportunity_id,
            opportunity_id=opportunity_id, route_id=None, status="QUALITY_REJECTED",
            stage="QUALITY", reason_codes=[str(code) for code in record["reason_codes"]],
            source_record_index=source_index, mechanism_fingerprint=None,
            mechanism_fingerprint_status="NOT_AVAILABLE_BEFORE_METHOD_GENERATION",
            human_scores=None, authenticity="NOT_APPLICABLE",
            authenticity_boundary="External execution authenticity does not apply before RECON.",
            attestation_scope="NOT_APPLICABLE",
            provenance={"quality_job_id": record["job_id"]},
        )

    # Preserve every non-clear dedup disposition, including a permitted adjacency.
    # The same opportunity may later have another stage event; event IDs, not entity
    # IDs, are the uniqueness boundary.
    for source_index, outcome in enumerate(dedup["outcomes"]):
        opportunity_id = str(outcome["record_id"])
        dedup_status = str(outcome["status"])
        if dedup_status != "CLEAR":
            nearest = [
                {"legacy_id": legacy_id, "evidence": evidence}
                for legacy_id, evidence in zip(
                    outcome.get("nearest_legacy_ids", []), outcome.get("nearest_evidence", []),
                )
            ]
            add_event(
                source_artifact="ledger/dedup_outcomes.jsonl", source=outcome,
                entity_type="OPPORTUNITY", entity_id=opportunity_id,
                opportunity_id=opportunity_id, route_id=None, status=dedup_status,
                stage="INTERNAL_DEDUP",
                reason_codes=[dedup_status, str(outcome["reason"])],
                nearest_priors=nearest,
                source_record_index=source_index, mechanism_fingerprint=None,
                mechanism_fingerprint_status="NOT_AVAILABLE_BEFORE_METHOD_GENERATION",
                human_scores=None, authenticity="NOT_APPLICABLE",
                authenticity_boundary="External execution authenticity does not apply before RECON.",
                attestation_scope="NOT_APPLICABLE",
                provenance={"blocked": outcome["blocked"], "assignment_sha256": outcome["assignment_sha256"]},
            )

    recon_job_by_id = {str(row["job_id"]): row for row in recon_jobs}
    report_by_job = {str(row["job_id"]): (row, index) for index, row in enumerate(recon["reports"])}
    rejected_recon_by_job = {str(row["job_id"]): (row, index) for index, row in enumerate(recon["rejected"])}
    for outcome in recon["outcomes"]:
        if recon_context is None:
            raise ValueError("recon event lacks its validated execution context")
        job_id = str(outcome["job_id"])
        job = recon_job_by_id[job_id]
        opportunity_id = str(job["opportunity_id"])
        report_entry = report_by_job.get(job_id)
        rejected_entry = rejected_recon_by_job.get(job_id)
        if report_entry is not None:
            report, source_index = report_entry
            source = report
            status = str(report["decision"])
            reasons = [status, str(report["decision_reason"])]
            nearest = report["nearest_priors"]
            source_artifact = "recon/reports.jsonl"
        else:
            if rejected_entry is None:
                raise ValueError("recon rejected outcome lacks its rejected source record")
            rejected, source_index = rejected_entry
            source = rejected
            status = "REJECTED"
            reasons = [str(outcome["error_code"]), str(outcome["error"])]
            nearest = []
            source_artifact = "recon/rejected.jsonl"
        add_event(
            source_artifact=source_artifact, source=source,
            entity_type="OPPORTUNITY", entity_id=opportunity_id,
            opportunity_id=opportunity_id, route_id=None, status=status, stage="RECON",
            reason_codes=reasons, nearest_priors=nearest,
            source_record_index=source_index, mechanism_fingerprint=None,
            mechanism_fingerprint_status="NOT_AVAILABLE_BEFORE_METHOD_GENERATION",
            human_scores=None, authenticity=str(recon_context["authenticity"]),
            authenticity_boundary=str(recon_context["claim_boundary"]),
            attestation_scope=str(recon_context["operator_trust_scope"]),
            provenance={
                "recon_outcome_sha256": _canonical_hash(outcome),
                "recon_job_id": job_id,
                "recon_job_sha256": _canonical_hash(job),
            },
        )

    route_job_by_id = {str(row["job_id"]): row for row in route_jobs}
    route_results_by_job: dict[str, list[tuple[Mapping[str, Any], int]]] = {}
    for result_index, result in enumerate(routes["results"]):
        route_results_by_job.setdefault(str(result["job_id"]), []).append((result, result_index))
    for source_index, outcome in enumerate(routes["outcomes"]):
        job_id = str(outcome["job_id"])
        job = route_job_by_id[job_id]
        opportunity_id = str(job["opportunity_id"])
        if outcome["status"] == "REJECTED":
            add_event(
                source_artifact="routes/outcomes.jsonl", source=outcome,
                entity_type="ROUTE_GENERATION_JOB", entity_id=job_id,
                opportunity_id=opportunity_id, route_id=None, status="REJECTED",
                stage="ROUTES",
                reason_codes=[str(outcome["error_code"]), str(outcome["error"])],
                nearest_priors=job["nearest_prior_bindings"],
                source_record_index=source_index, mechanism_fingerprint=None,
                mechanism_fingerprint_status="NO_ACCEPTED_METHOD",
                human_scores=None, authenticity=str(job["authenticity"]),
                authenticity_boundary=str(job["authenticity_boundary"]),
                attestation_scope=str(job["attestation_scope"]),
                provenance={"route_job_sha256": _canonical_hash(job)},
            )
            continue
        route_entries = route_results_by_job.get(job_id, [])
        if [str(row["record_id"]) for row, _index in route_entries] != [str(value) for value in outcome["route_ids"]]:
            raise ValueError("accepted route outcome lacks its ordered route projections")
        for route_record, route_result_index in route_entries:
            route_id = str(route_record["record_id"])
            add_event(
                source_artifact="routes/results.jsonl", source=route_record,
                entity_type="ROUTE", entity_id=route_id,
                opportunity_id=opportunity_id, route_id=route_id, status="ACCEPTED",
                stage="ROUTES", reason_codes=["ACCEPTED"],
                nearest_priors=job["nearest_prior_bindings"],
                source_record_index=route_result_index,
                mechanism_fingerprint=str(route_record["mechanism_fingerprint"]),
                mechanism_fingerprint_status="AVAILABLE",
                human_scores=None, authenticity=str(route_record["authenticity"]),
                authenticity_boundary=str(route_record["authenticity_boundary"]),
                attestation_scope=str(route_record["attestation_scope"]),
                provenance={
                    "route_job_sha256": _canonical_hash(job),
                    "route_result_sha256": _canonical_hash(route_record),
                },
            )

    review_job_by_id = {str(row["job_id"]): row for row in review["jobs"]}
    review_result_by_job = {
        str(row["job_id"]): (row, index) for index, row in enumerate(review["results"])
    }
    for source_index, outcome in enumerate(review["outcomes"]):
        job_id = str(outcome["job_id"])
        job = review_job_by_id[job_id]
        route_id = str(job["route_id"])
        result_entry = review_result_by_job.get(job_id)
        if result_entry is not None:
            result, result_index = result_entry
            status = str(result["decision"])
            reasons = [status]
            source = result
            event_source_index = result_index
            source_artifact = "review/results.jsonl"
        else:
            result = None
            status = "REJECTED"
            reasons = [str(outcome["error_code"]), str(outcome["error"])]
            source = outcome
            event_source_index = source_index
            source_artifact = "review/outcomes.jsonl"
        add_event(
            source_artifact=source_artifact, source=source,
            entity_type="ROUTE", entity_id=route_id,
            opportunity_id=str(job["opportunity_id"]), route_id=route_id,
            status=status, stage="REVIEW", reason_codes=reasons,
            nearest_priors=job["route"]["nearest_prior_bindings"],
            source_record_index=event_source_index,
            mechanism_fingerprint=str(job["route"]["mechanism_fingerprint"]),
            mechanism_fingerprint_status="AVAILABLE", human_scores=None,
            authenticity=str(job["authenticity"]),
            authenticity_boundary=str(job["authenticity_boundary"]),
            attestation_scope=str(job["attestation_scope"]),
            provenance={
                "review_job_sha256": _canonical_hash(job),
                "review_result_sha256": _canonical_hash(result) if result is not None else None,
                "round": job["round"],
            },
        )

    human_job_by_id = {str(row["job_id"]): row for row in human["jobs"]}
    human_result_by_job = {
        str(row["job_id"]): (row, index) for index, row in enumerate(human["results"])
    }
    for source_index, outcome in enumerate(human["outcomes"]):
        job_id = str(outcome["job_id"])
        job = human_job_by_id[job_id]
        score_entry = human_result_by_job.get(job_id)
        if score_entry is not None:
            score, score_index = score_entry
            status = "ACCEPTED"
            reasons = [str(code) for code in score["reason_codes"]]
            source = score
            event_source_index = score_index
            source_artifact = "ledger/human_scores.jsonl"
        elif outcome["status"] == "SKIPPED":
            score = None
            status = "SKIPPED"
            reasons = [str(code) for code in outcome["reason_codes"]]
            source = outcome
            event_source_index = source_index
            source_artifact = "ledger/human_scoring_outcomes.jsonl"
        else:
            score = None
            status = "REJECTED"
            reasons = [str(outcome["error_code"]), str(outcome["error"])]
            source = outcome
            event_source_index = source_index
            source_artifact = "ledger/human_scoring_outcomes.jsonl"
        add_event(
            source_artifact=source_artifact, source=source,
            entity_type="ROUTE", entity_id=str(job["route_id"]),
            opportunity_id=str(job["opportunity_id"]), route_id=str(job["route_id"]),
            status=status, stage="HUMAN_SCORING", reason_codes=reasons,
            nearest_priors=job["nearest_priors"],
            source_record_index=event_source_index,
            mechanism_fingerprint=str(job["route"]["mechanism_fingerprint"]),
            mechanism_fingerprint_status="AVAILABLE",
            human_scores={field: int(score[field]) for field in _SCORE_FIELDS} if score is not None else None,
            authenticity=str(job["authenticity_binding"]["authenticity"]),
            authenticity_boundary=str(job["authenticity_binding"]["authenticity_boundary"]),
            attestation_scope=str(job["authenticity_binding"]["attestation_scope"]),
            provenance={
                "human_job_sha256": job["human_job_sha256"],
                "human_score_sha256": _canonical_hash(score) if score is not None else None,
            },
        )

    human_jobs_by_route = {str(row["route_id"]): row for row in human["jobs"]}
    scores_by_job = {str(row["job_id"]): row for row in human["results"]}
    for idea_id, pack in packs.items():
        matching_job: Mapping[str, Any] | None = None
        for route_id, job in human_jobs_by_route.items():
            score = scores_by_job.get(str(job["job_id"]))
            if score is None:
                continue
            expected_id = stable_id(
                "idea_pack", route_id, score["human_score_id"], job["review_record_sha256"],
                job["authenticity_sha256"],
            )
            if expected_id == idea_id:
                matching_job = job
                break
        if matching_job is None:
            raise ValueError("idea pack lacks an exact human job source")
        add_event(
            source_artifact=f"idea_packs/{idea_id}.json", source=pack,
            entity_type="IDEA_PACK", entity_id=idea_id,
            opportunity_id=str(matching_job["opportunity_id"]),
            route_id=str(matching_job["route_id"]), status=str(pack["status"]),
            stage="IDEA_PACK", reason_codes=[str(code) for code in pack["human_reason_codes"]],
            nearest_priors=pack["nearest_priors"],
            source_record_index=None,
            mechanism_fingerprint=str(matching_job["route"]["mechanism_fingerprint"]),
            mechanism_fingerprint_status="AVAILABLE",
            human_scores={field: int(pack["human_scores"][field]) for field in _SCORE_FIELDS},
            authenticity=str(matching_job["authenticity_binding"]["authenticity"]),
            authenticity_boundary=str(matching_job["authenticity_binding"]["authenticity_boundary"]),
            attestation_scope=str(matching_job["authenticity_binding"]["attestation_scope"]),
            provenance={
                "idea_pack_manifest_sha256": pack_manifest["manifest_sha256"],
                "human_bundle_manifest_sha256": human["manifest"]["bundle_manifest_sha256"],
            },
        )
    updates.sort(key=lambda row: str(row["ledger_update_id"]))
    return updates


def _validate_persistent_updates(records: Sequence[Mapping[str, Any]]) -> None:
    expected_keys = {
        "schema_version", "run_id", "source_run_id", "ledger_transaction_id",
        "ledger_update_id", "entity_type", "entity_id", "opportunity_id", "route_id",
        "status", "stage", "reason_codes", "nearest_priors", "source_record_sha256",
        "source_locator", "opportunity_fingerprint", "mechanism_fingerprint",
        "mechanism_fingerprint_status", "supporting_card_ids", "supporting_paper_ids",
        "human_scores", "authenticity", "authenticity_boundary", "attestation_scope",
        "cheap_test_done", "subsequent_result",
        "provenance", "update_sha256",
    }
    score_keys = set(_SCORE_FIELDS)
    seen: dict[str, Mapping[str, Any]] = {}
    for row in records:
        update_id = row.get("ledger_update_id")
        if type(update_id) is not str or update_id in seen:
            raise ValueError("existing persistent ledger has duplicate or invalid IDs")
        body = {key: value for key, value in row.items() if key != "update_sha256"}
        scores = row.get("human_scores")
        mechanism = row.get("mechanism_fingerprint")
        mechanism_status = row.get("mechanism_fingerprint_status")
        if (
            row.get("schema_version") != "idea_factory.ledger_update.v2"
            or set(row) != expected_keys
            or any(type(row.get(key)) is not str or not str(row[key]).strip() for key in (
                "run_id", "source_run_id", "ledger_transaction_id", "entity_type", "entity_id",
                "opportunity_id", "status", "stage", "source_record_sha256",
                "opportunity_fingerprint", "mechanism_fingerprint_status", "authenticity",
                "authenticity_boundary", "attestation_scope",
            ))
            or row.get("run_id") != row.get("source_run_id")
            or row.get("cheap_test_done") is not False
            or row.get("subsequent_result") is not None
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_record_sha256", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("opportunity_fingerprint", "")))
            or type(row.get("reason_codes")) is not list or not row["reason_codes"]
            or any(type(value) is not str or not value for value in row["reason_codes"])
            or type(row.get("supporting_card_ids")) is not list or not row["supporting_card_ids"]
            or type(row.get("supporting_paper_ids")) is not list or not row["supporting_paper_ids"]
            or any(type(value) is not str or not value for value in (*row["supporting_card_ids"], *row["supporting_paper_ids"]))
            or (
                scores is not None
                and (
                    type(scores) is not dict or set(scores) != score_keys
                    or any(type(value) is not int or not 0 <= value <= 5 for value in scores.values())
                )
            )
            or (
                mechanism_status == "AVAILABLE"
                and (type(mechanism) is not str or not re.fullmatch(r"[0-9a-f]{64}", mechanism))
            )
            or (mechanism_status != "AVAILABLE" and mechanism is not None)
            or row.get("update_sha256") != _canonical_hash(body)
        ):
            raise ValueError("existing persistent ledger update hash or schema is invalid")
        seen[update_id] = row


def _validate_ledger_snapshot_locked(
    run: Path,
    run_target: Path,
    persistent_target: Path,
    expected_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not run_target.is_file() or run_target.resolve(strict=True) != run_target:
        raise ValueError("run ledger updates path is missing or unsafe")
    if not persistent_target.is_file() or persistent_target.resolve(strict=True) != persistent_target:
        raise ValueError("persistent ledger path is missing or unsafe")
    run_updates = read_jsonl(run_target)
    persistent = read_jsonl(persistent_target)
    run_id = _run_identity(run)
    expected = [dict(row) for row in expected_events]
    if run_updates != expected:
        raise ValueError("run ledger does not equal expected artifact-derived replay")
    persistent_current = [
        row for row in persistent
        if row.get("run_id") == run_id
    ]
    if persistent_current != expected:
        raise ValueError("persistent current-run ledger does not equal expected artifact-derived replay")
    _validate_persistent_updates(run_updates)
    _validate_persistent_updates(persistent)
    if any(row["run_id"] != run_id for row in run_updates):
        raise ValueError("run ledger contains an event from a different source run")
    transaction_ids = {str(row["ledger_transaction_id"]) for row in expected}
    if len(transaction_ids) > 1:
        raise ValueError("expected ledger replay contains multiple transaction snapshots")
    transaction_id = next(iter(transaction_ids), stable_id("ledger_transaction", run_id, "EMPTY"))
    for event in persistent:
        validate_ledger_source_locator(event)
    return {
        "schema_version": "idea_factory.ledger_snapshot.v1",
        "run_id": run_id,
        "ledger_transaction_id": transaction_id,
        "run_updates": tuple(run_updates),
        "persistent_updates": tuple(persistent),
    }


def validate_ledger_updates(
    run_dir: Path,
    config: CorpusRouterConfig,
    *,
    repository_root: Path | None = None,
    prompt_path: Path | None = None,
    route_prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> dict[str, Any]:
    """Official snapshot-safe ledger reader.

    It acquires the same ordered run/persistent locks as the writer and verifies
    the shared transaction ID plus every recoverable source locator.  Raw JSONL reads are not snapshot-safe and are not an official ledger read API.
    """

    run = _anchored_run(Path(run_dir))
    ledger = _owned_dir(run, "ledger", create=False)
    root = repository_root or Path(__file__).resolve().parents[2]
    data = _repository_data_dir(Path(root))
    run_target = ledger / "updates.jsonl"
    persistent_target = data / "idea_ledger.jsonl"
    lock_dirs = sorted((ledger, data), key=lambda path: str(path).casefold())
    with ExitStack() as stack:
        for directory in lock_dirs:
            stack.enter_context(stage_mutation_lock(directory))
        expected = build_expected_ledger_events(
            run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
            allow_test_ready=allow_test_ready,
        )
        return _validate_ledger_snapshot_locked(
            run, run_target, persistent_target, expected,
        )


def append_ledger_updates(
    run_dir: Path,
    config: CorpusRouterConfig,
    *,
    repository_root: Path | None = None,
    prompt_path: Path | None = None,
    route_prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> dict[str, Path]:
    run = _anchored_run(Path(run_dir))
    ledger = _owned_dir(run, "ledger", create=False)
    root = repository_root or Path(__file__).resolve().parents[2]
    data = _repository_data_dir(Path(root))
    run_target = ledger / "updates.jsonl"
    persistent_target = data / "idea_ledger.jsonl"
    lock_dirs = sorted((ledger, data), key=lambda path: str(path).casefold())
    with ExitStack() as stack:
        for directory in lock_dirs:
            stack.enter_context(stage_mutation_lock(directory))
        updates = build_expected_ledger_events(
            run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
            allow_test_ready=allow_test_ready,
        )
        if run_target.exists():
            if not persistent_target.exists():
                raise ValueError("existing run ledger has no matching persistent snapshot")
            existing_snapshot = _validate_ledger_snapshot_locked(
                run, run_target, persistent_target, updates,
            )
            existing_run = list(existing_snapshot["run_updates"])
            if existing_run != updates:
                raise ValueError("existing run ledger updates replay conflict")
        persistent: list[dict[str, Any]] = []
        if persistent_target.exists():
            if not persistent_target.is_file() or persistent_target.resolve(strict=True) != persistent_target:
                raise ValueError("existing persistent ledger path is unsafe")
            persistent = read_jsonl(persistent_target)
            _validate_persistent_updates(persistent)
        existing_by_id = {str(row["ledger_update_id"]): row for row in persistent}
        for update in updates:
            update_id = str(update["ledger_update_id"])
            old = existing_by_id.get(update_id)
            if old is not None and old != update:
                raise ValueError("existing persistent ledger update ID conflicts with different content")
            if old is None:
                persistent.append(update); existing_by_id[update_id] = update
        persistent.sort(key=lambda row: str(row["ledger_update_id"]))
        publish_cross_directory_transaction({
            run_target: encode_jsonl(updates),
            persistent_target: encode_jsonl(persistent),
        })
    return {"run_updates": run_target, "persistent_ledger": persistent_target}
