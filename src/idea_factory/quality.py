"""Deterministic, replayable opportunity quality gate; deliberately no LLM scores."""

from __future__ import annotations

import re
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .artifacts import read_jsonl, write_jsonl_bundle
from .models import EvidenceStatus, Opportunity, PaperCard, QualityDecision


QUALITY_GATE_VERSION = "idea_factory.opportunity_quality.v1"
_FIELDS = (
    ("assumption_x", "EMPTY_ASSUMPTION_X"), ("observation_y", "EMPTY_OBSERVATION_Y"),
    ("condition_z", "EMPTY_CONDITION_Z"), ("failure_f", "EMPTY_FAILURE_F"),
    ("missing_capability_w", "EMPTY_MISSING_CAPABILITY_W"),
    ("alternative_explanation_a", "EMPTY_ALTERNATIVE_A"),
    ("decisive_experiment", "EMPTY_DECISIVE_EXPERIMENT"),
)
_RELATION = re.compile(r"\b(assum|because|caus|fail|under|when|if|unless|after|before|during|remain|produ|lead|degrad|evict)\w*\b", re.I)
_KEYWORD_STACK = re.compile(r"\b[a-z][a-z0-9_-]*\s*(?:\+|and)\s*[a-z][a-z0-9_-]*\b", re.I)
_TEST_WORDS = re.compile(r"\b(compare|versus|vs\.?|against|ablat|control)\b", re.I)
_MEASURE_WORDS = re.compile(r"\b(measure|metric|recall|accuracy|latency|loss|rate|error|throughput)\b", re.I)
_SCOPE_CONTRADICTION = re.compile(r"\b(incompatible|contradict(?:s|ory)?|cannot compare|mutually exclusive)\b", re.I)
QUALITY_GATE_POLICY = {
    "version": QUALITY_GATE_VERSION,
    "policy_kind": "deterministic-transparent-lexical-v1",
    "blank_field_codes": [code for _field, code in _FIELDS],
    "relation_pattern": _RELATION.pattern,
    "keyword_stack_pattern": _KEYWORD_STACK.pattern,
    "test_pattern": _TEST_WORDS.pattern,
    "measure_pattern": _MEASURE_WORDS.pattern,
    "scope_contradiction_pattern": _SCOPE_CONTRADICTION.pattern,
    "alternative_rule": "normalized exact substring in decisive_experiment",
}
QUALITY_GATE_POLICY_SHA256 = hashlib.sha256(
    json.dumps(QUALITY_GATE_POLICY, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[\w-]+", value.lower()))


def evaluate_opportunity(
    opportunity: Opportunity, cards: Mapping[str, PaperCard] | Iterable[PaperCard], *,
    allowed_neighbor_ids: set[str] | None = None,
) -> QualityDecision:
    """Return stable reason codes. ``cards`` must be the validated job universe."""

    card_map = dict(cards) if isinstance(cards, Mapping) else {card.card_id: card for card in cards}
    reasons: list[str] = []
    for field, code in _FIELDS:
        if not getattr(opportunity, field).strip():
            reasons.append(code)
    if not opportunity.supporting_card_ids:
        reasons.append("NO_SUPPORTING_CARD")
    for card_id in opportunity.supporting_card_ids:
        card = card_map.get(card_id)
        if card is None:
            reasons.append("UNKNOWN_SUPPORTING_CARD")
        elif card.failure_mechanism.status == EvidenceStatus.UNKNOWN or not card.failure_mechanism.text.strip():
            reasons.append("UNKNOWN_FAILURE_MECHANISM")
    if not opportunity.nearest_internal_neighbors:
        reasons.append("NO_INTERNAL_NEIGHBOR")
    else:
        permitted = allowed_neighbor_ids if allowed_neighbor_ids is not None else set(card_map)
        if any(neighbor not in permitted for neighbor in opportunity.nearest_internal_neighbors):
            reasons.append("UNKNOWN_INTERNAL_NEIGHBOR")
    if not opportunity.scope_compatibility.strip():
        reasons.append("BLANK_SCOPE_COMPATIBILITY")
    elif _SCOPE_CONTRADICTION.search(opportunity.scope_compatibility):
        reasons.append("SCOPE_CONTRADICTION")
    alternative = _normalized(opportunity.alternative_explanation_a)
    experiment = _normalized(opportunity.decisive_experiment)
    if alternative and alternative not in experiment:
        reasons.append("ALTERNATIVE_NOT_NAMED")
    relation_text = " ".join((opportunity.assumption_x, opportunity.observation_y, opportunity.condition_z, opportunity.failure_f))
    if not _RELATION.search(opportunity.assumption_x) or len(_normalized(opportunity.assumption_x).split()) < 5:
        reasons.append("VAGUE_ASSUMPTION")
    if _KEYWORD_STACK.search(relation_text) and not _RELATION.search(relation_text):
        reasons.append("KEYWORD_COMBINATION")
    if not experiment or not _TEST_WORDS.search(opportunity.decisive_experiment) or not _MEASURE_WORDS.search(opportunity.decisive_experiment):
        reasons.append("NO_DISCRIMINATING_TEST")
    return QualityDecision(opportunity_id=opportunity.opportunity_id, passed=not reasons, reason_codes=sorted(set(reasons)))


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _quality_paths(quality: Path) -> dict[str, Path]:
    return {
        "ready": quality / "ready_for_internal_dedup.jsonl",
        "rejected": quality / "rejected.jsonl",
        "outcomes": quality / "opportunity_job_outcomes.jsonl",
    }


def _quality_projections(result_bundle: object) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    jobs = {row["job_id"]: row for row in result_bundle.mining.jobs}
    result_outcomes = {row["job_id"]: row for row in result_bundle.outcomes}
    ready: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    by_job: dict[str, list[dict[str, object]]] = {job_id: [] for job_id in jobs}
    for candidate in result_bundle.candidates:
        job = jobs[candidate["job_id"]]
        opportunity = Opportunity.model_validate_json(
            json.dumps(candidate["opportunity"], ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")),
            strict=True,
        )
        decision = evaluate_opportunity(
            opportunity, result_bundle.mining.landscape.cards,
            allowed_neighbor_ids=set(job["neighbor_ids"]),
        )
        record = {
            "schema_version": "idea_factory.quality_decision.v2",
            "record_id": opportunity.opportunity_id, "job_id": job["job_id"],
            "cluster_id": job["cluster_id"], "raw_result_sha256": candidate["raw_result_sha256"],
            "candidate_sha256": _canonical_hash(candidate), "landscape_hashes": job["landscape_hashes"],
            "card_ids": job["card_ids"], "neighbor_ids": job["neighbor_ids"],
            "quality_gate_version": QUALITY_GATE_VERSION,
            "quality_gate_policy_sha256": QUALITY_GATE_POLICY_SHA256,
            "quality_gate_policy": QUALITY_GATE_POLICY,
            "opportunity": opportunity.model_dump(mode="json"),
            "passed": decision.passed, "reason_codes": decision.reason_codes,
        }
        by_job[job["job_id"]].append(record)
        (ready if decision.passed else rejected).append(record)
    ready.sort(key=lambda row: (str(row["job_id"]), str(row["record_id"])))
    rejected.sort(key=lambda row: (str(row["job_id"]), str(row["record_id"])))
    outcomes: list[dict[str, object]] = []
    for job_id in sorted(jobs):
        job_ready = [row for row in by_job[job_id] if row["passed"]]
        job_rejected = [row for row in by_job[job_id] if not row["passed"]]
        result_outcome = result_outcomes[job_id]
        if result_outcome["status"] == "VALID_EMPTY":
            status = "VALID_EMPTY"
        elif not by_job[job_id]:
            status = "NO_VALID_CANDIDATES"
        elif job_ready and job_rejected:
            status = "PARTIAL_READY"
        elif job_ready:
            status = "READY"
        else:
            status = "QUALITY_REJECTED"
        outcomes.append({
            "schema_version": "idea_factory.opportunity_quality_outcome.v2",
            "job_id": job_id, "cluster_id": jobs[job_id]["cluster_id"],
            "raw_result_sha256": result_outcome["raw_result_sha256"],
            "result_outcome_sha256": _canonical_hash(result_outcome),
            "quality_status": status,
            "ready_opportunity_ids": [row["record_id"] for row in job_ready],
            "ready_count": len(job_ready), "rejected_opportunity_ids": [row["record_id"] for row in job_rejected],
            "rejected_count": len(job_rejected),
            "ready_projection_sha256": _canonical_hash(job_ready),
            "rejected_projection_sha256": _canonical_hash(job_rejected),
            "decision_reason_codes": [
                {"opportunity_id": row["record_id"], "passed": row["passed"], "reason_codes": row["reason_codes"]}
                for row in by_job[job_id]
            ],
            "quality_gate_version": QUALITY_GATE_VERSION,
            "quality_gate_policy_sha256": QUALITY_GATE_POLICY_SHA256,
            "landscape_hashes": jobs[job_id]["landscape_hashes"], "card_ids": jobs[job_id]["card_ids"],
        })
    return ready, rejected, outcomes


@dataclass(frozen=True)
class QualityBundle:
    run: Path
    ready: tuple[dict[str, object], ...]
    rejected: tuple[dict[str, object], ...]
    outcomes: tuple[dict[str, object], ...]


def publish_quality(run_dir: Path) -> dict[str, Path]:
    """Consume only a replay-validated result bundle and publish a separate gate bundle."""
    from .opportunities import _anchored_run, _owned_dir, _safe_unlink, validate_opportunity_result_bundle

    run = _anchored_run(Path(run_dir))
    quality = _owned_dir(run, "quality", create=True)
    paths = _quality_paths(quality)
    _safe_unlink(quality, [path.name for path in paths.values()])
    try:
        result_bundle = validate_opportunity_result_bundle(run)
        ready, rejected, outcomes = _quality_projections(result_bundle)
        write_jsonl_bundle({paths["ready"]: ready, paths["rejected"]: rejected, paths["outcomes"]: outcomes})
        return paths
    except BaseException:
        _safe_unlink(quality, [path.name for path in paths.values()])
        raise


def validate_quality_bundle(run_dir: Path) -> QualityBundle:
    """Replay Task-7 decisions exactly for the future internal-dedup consumer."""
    from .opportunities import _anchored_run, _owned_dir, _owned_file, validate_opportunity_result_bundle

    run = _anchored_run(Path(run_dir))
    quality = _owned_dir(run, "quality", create=False)
    paths = _quality_paths(quality)
    ready = read_jsonl(_owned_file(quality, paths["ready"].name))
    rejected = read_jsonl(_owned_file(quality, paths["rejected"].name))
    outcomes = read_jsonl(_owned_file(quality, paths["outcomes"].name))
    expected = _quality_projections(validate_opportunity_result_bundle(run))
    if ready != expected[0] or rejected != expected[1] or outcomes != expected[2]:
        raise ValueError("quality bundle replay projection mismatch")
    return QualityBundle(run, tuple(ready), tuple(rejected), tuple(outcomes))
