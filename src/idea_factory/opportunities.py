"""Single-basis opportunity jobs and replayable strict-JSON result artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from .artifacts import read_jsonl, stable_id, write_jsonl_bundle
from .landscape import (
    ValidatedLandscapeBundle as LandscapeBundle,
    canonical_scope_key,
    normalize_concept,
    validate_landscape_bundle,
)
from .models import Opportunity, OpportunityOperator, PaperCard, StrictModel


OPPORTUNITY_PROMPT_VERSION = "idea_factory.opportunity_miner_prompt.v2"
OPERATOR_SCHEDULE_VERSION = "idea_factory.operator_schedule.v1"
MINING_JOB_SCHEMA_VERSION = "idea_factory.opportunity_mining_job.v2"
CLUSTER_AUDIT_SCHEMA_VERSION = "idea_factory.opportunity_cluster_audit.v1"
OPPORTUNITY_RESULT_SCHEMA_VERSION = "idea_factory.opportunity_result.v1"
OPPORTUNITY_OUTCOME_SCHEMA_VERSION = "idea_factory.opportunity_result_outcome.v2"
_JOB_NAMES = ("mining_jobs.jsonl", "cluster_audit.jsonl")
_RESULT_NAMES = ("opportunity_candidates.jsonl", "rejected_results.jsonl", "result_outcomes.jsonl")
_QUALITY_NAMES = ("ready_for_internal_dedup.jsonl", "rejected.jsonl", "opportunity_job_outcomes.jsonl")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _strict_json(raw: bytes | str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        value = json.loads(
            text,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
            object_pairs_hook=_strict_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("result must be strict native JSON") from exc
    if type(value) is not dict:
        raise ValueError("result wrapper must be a JSON object")
    return value


def _anchored_run(run_dir: Path) -> Path:
    run = Path(os.path.abspath(run_dir))
    try:
        resolved = run.resolve(strict=True)
    except OSError as exc:
        raise ValueError("expected run directory is missing") from exc
    if not run.is_dir() or run != resolved:
        raise ValueError("expected run directory must have exact resolved identity")
    return run


def _owned_dir(run: Path, name: str, *, create: bool) -> Path:
    directory = run / name
    if directory.exists():
        try:
            resolved = directory.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"{name} directory is unresolved") from exc
        if not directory.is_dir() or resolved != run / name:
            raise ValueError(f"{name} directory escapes and is not anchored to the expected run")
        for child in directory.iterdir():
            try:
                resolved_child = child.resolve(strict=True)
            except OSError as exc:
                raise ValueError(f"{name} contains an unresolved artifact") from exc
            if not resolved_child.is_relative_to(directory):
                raise ValueError(f"{name} artifact escapes the expected run")
        return directory
    if not create:
        raise ValueError(f"{name} directory is missing")
    if directory.parent.resolve(strict=True) != run:
        raise ValueError(f"{name} directory escapes the expected run")
    directory.mkdir()
    if directory.resolve(strict=True) != run / name:
        raise ValueError(f"{name} directory is not anchored to the expected run")
    return directory


def _owned_file(directory: Path, name: str) -> Path:
    path = directory / name
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"required artifact is missing: {name}") from exc
    if not path.is_file() or resolved != directory / name:
        raise ValueError(f"artifact is not anchored to the expected run: {name}")
    return path


def _safe_unlink(directory: Path, names: Sequence[str]) -> None:
    for name in names:
        (directory / name).unlink(missing_ok=True)


def _invalidate_result_and_quality(run: Path, opportunities: Path) -> None:
    quality = run / "quality"
    anchored_quality = _owned_dir(run, "quality", create=False) if quality.exists() else None
    _safe_unlink(opportunities, _RESULT_NAMES)
    if anchored_quality is not None:
        _safe_unlink(anchored_quality, _QUALITY_NAMES)


def scheduled_operators() -> tuple[OpportunityOperator, ...]:
    """V1 emits eight calls per eligible cluster, one per explicit operator."""
    return tuple(OpportunityOperator)


def _basis_id(basis: dict[str, Any]) -> str:
    return stable_id(
        "landscape_basis", basis["dimension"], basis["facet"],
        basis["normalized_text"], basis["scope_key"],
    )


def _entry_basis(dimension: str, entry: dict[str, Any]) -> dict[str, Any]:
    basis = {
        "dimension": dimension,
        "facet": entry["facet"],
        "normalized_text": entry["normalized_text"],
        "scope": entry["scope"],
        "scope_key": entry["scope_key"],
        "source_card_ids": sorted(set(entry["source_card_ids"])),
    }
    return {"basis_id": _basis_id(basis)} | basis


def _balanced_chunks(card_ids: list[str]) -> list[list[str]]:
    chunk_count = max(1, (len(card_ids) + 7) // 8)
    base, remainder = divmod(len(card_ids), chunk_count)
    chunks: list[list[str]] = []
    offset = 0
    for index in range(chunk_count):
        size = base + (1 if index >= chunk_count - remainder else 0)
        chunks.append(card_ids[offset:offset + size])
        offset += size
    if any(not 3 <= len(chunk) <= 8 for chunk in chunks):
        raise ValueError("eligible basis cannot be split into bounded 3-8 card chunks")
    return chunks


def _partition_bases(bases: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique = {_canonical_json(basis): basis for basis in bases}
    clusters: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for basis in sorted(unique.values(), key=lambda row: (row["dimension"], row["facet"], row["normalized_text"], row["scope_key"], row["basis_id"])):
        card_ids = basis["source_card_ids"]
        if len(card_ids) < 3:
            audit.append({
                "schema_version": CLUSTER_AUDIT_SCHEMA_VERSION, "basis_id": basis["basis_id"],
                "cluster_basis": basis, "status": "SKIPPED_TOO_SMALL", "source_count": len(card_ids),
                "card_ids": card_ids, "chunk_count": 0, "jobs_emitted": 0,
                "calls_per_cluster": 8, "estimated_external_calls": 0,
                "reason": "basis has fewer than three unique accepted cards",
            })
            continue
        chunks = _balanced_chunks(card_ids)
        cluster_ids: list[str] = []
        for chunk_index, chunk in enumerate(chunks):
            cluster_id = stable_id("opportunity_cluster", basis["basis_id"], str(chunk_index), ",".join(chunk))
            cluster_ids.append(cluster_id)
            clusters.append({
                "cluster_id": cluster_id, "chunk_index": chunk_index,
                "cluster_basis": basis, "card_ids": chunk,
            })
        audit.append({
            "schema_version": CLUSTER_AUDIT_SCHEMA_VERSION, "basis_id": basis["basis_id"],
            "cluster_basis": basis, "status": "ELIGIBLE", "source_count": len(card_ids),
            "card_ids": card_ids, "chunk_count": len(chunks), "cluster_ids": cluster_ids,
            "jobs_emitted": len(chunks) * len(scheduled_operators()), "calls_per_cluster": 8,
            "estimated_external_calls": len(chunks) * len(scheduled_operators()), "reason": "",
        })
    return clusters, audit


def cluster_cards(cards: Sequence[PaperCard | dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build non-transitive exact scientific bases directly from card fields."""
    bases: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for item in cards:
        card = item.model_dump(mode="json") if isinstance(item, PaperCard) else item
        scope = card["scope"]
        scope_key = canonical_scope_key(scope)
        values = [
            ("assumptions", "assumption", card["assumption"]["text"]),
            ("failures", "failure_observation", card["failure_observation"]["text"]),
        ]
        failure = card.get("failure_mechanism", {})
        if failure.get("status") != "UNKNOWN" and str(failure.get("text", "")).strip():
            values.append(("failures", "failure_mechanism", failure["text"]))
        for dimension, facet, raw in values:
            normalized = normalize_concept(raw)
            key = (dimension, facet, normalized, scope_key)
            basis = bases.setdefault(key, {
                "dimension": dimension, "facet": facet, "normalized_text": normalized,
                "scope": scope, "scope_key": scope_key, "source_card_ids": [],
            })
            basis["source_card_ids"].append(card["card_id"])
    normalized_bases = []
    for basis in bases.values():
        basis["source_card_ids"] = sorted(set(basis["source_card_ids"]))
        normalized_bases.append({"basis_id": _basis_id(basis)} | basis)
    return _partition_bases(normalized_bases)


def _landscape_bases(bundle: LandscapeBundle) -> list[dict[str, Any]]:
    bases = []
    for dimension in ("assumptions", "failures"):
        for entry in bundle.maps[dimension]["entries"]:
            if dimension == "failures" and entry["facet"] not in {"failure_observation", "failure_mechanism"}:
                continue
            bases.append(_entry_basis(dimension, entry))
    return bases


def _all_entry_summaries(bundle: LandscapeBundle) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dimension in ("assumptions", "failures", "mechanisms", "evaluations"):
        for entry in bundle.maps[dimension]["entries"]:
            summary = _entry_basis(dimension, entry)
            summary["neighbor_id"] = stable_id(
                "landscape_neighbor", dimension, entry["facet"],
                entry["normalized_text"], entry["scope_key"],
            )
            summary["raw_phrases"] = entry["raw_phrases"]
            summary["evidence_statuses"] = entry["evidence_statuses"]
            summaries.append(summary)
    return summaries


def _neighbors(bundle: LandscapeBundle, cluster: dict[str, Any]) -> list[dict[str, Any]]:
    basis = cluster["cluster_basis"]
    cluster_cards = set(cluster["card_ids"])
    neighbors: list[dict[str, Any]] = []
    for summary in _all_entry_summaries(bundle):
        if summary["basis_id"] == basis["basis_id"]:
            continue
        relations: list[str] = []
        if summary["normalized_text"] == basis["normalized_text"] and summary["scope_key"] != basis["scope_key"]:
            relations.append("SAME_LABEL_OTHER_SCOPE")
        overlap = cluster_cards.intersection(summary["source_card_ids"])
        if overlap:
            relations.append(
                "SAME_SCOPE_CARD_CONTEXT"
                if summary["scope_key"] == basis["scope_key"]
                else "SAME_CARD_CONTEXT_OTHER_SCOPE"
            )
        if summary["scope_key"] == basis["scope_key"] and summary["facet"] == basis["facet"] and set(summary["source_card_ids"]) - cluster_cards:
            relations.append("SAME_SCOPE_FACET_ADJACENT")
        if relations:
            neighbors.append(summary | {"relations": relations})
    return sorted(neighbors, key=lambda row: row["neighbor_id"])


def _job_records(bundle: LandscapeBundle, prompt_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    clusters, audit = _partition_bases(_landscape_bases(bundle))
    audit = [row | {
        "operator_schedule_version": OPERATOR_SCHEDULE_VERSION,
        "prompt_version": OPPORTUNITY_PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "prompt_text": prompt_text,
        "landscape_hashes": bundle.hashes,
    } for row in audit]
    records: list[dict[str, Any]] = []
    for cluster in clusters:
        neighbors = _neighbors(bundle, cluster)
        neighbor_ids = [row["neighbor_id"] for row in neighbors]
        for operator in scheduled_operators():
            job_id = stable_id(
                "opportunity_mining_job", MINING_JOB_SCHEMA_VERSION,
                OPERATOR_SCHEDULE_VERSION, cluster["cluster_id"], operator.value,
                prompt_hash, _canonical_hash(bundle.hashes),
            )
            records.append({
                "schema_version": MINING_JOB_SCHEMA_VERSION, "job_id": job_id,
                "cluster_id": cluster["cluster_id"], "cluster_basis": cluster["cluster_basis"],
                "operator": operator.value, "operator_schedule_version": OPERATOR_SCHEDULE_VERSION,
                "estimated_external_calls": 1, "calls_per_cluster": 8,
                "prompt_version": OPPORTUNITY_PROMPT_VERSION, "prompt_sha256": prompt_hash,
                "prompt_text": prompt_text, "landscape_hashes": bundle.hashes,
                "card_ids": cluster["card_ids"],
                "cards": [bundle.cards[card_id].model_dump(mode="json") for card_id in cluster["card_ids"]],
                "neighbor_ids": neighbor_ids, "neighbor_summaries": neighbors,
            })
    records.sort(key=lambda row: (row["cluster_id"], row["operator"], row["job_id"]))
    return records, audit


@dataclass(frozen=True)
class MiningBundle:
    run: Path
    landscape: LandscapeBundle
    jobs: tuple[dict[str, Any], ...]
    audit: tuple[dict[str, Any], ...]


def emit_mining_jobs(run_dir: Path, prompt_path: Path) -> Path:
    bundle = validate_landscape_bundle(run_dir)
    run = bundle.run
    if (run / "quality").exists():
        _owned_dir(run, "quality", create=False)
    opportunities = _owned_dir(run, "opportunities", create=True)
    _invalidate_result_and_quality(run, opportunities)
    targets = {"jobs": opportunities / _JOB_NAMES[0], "audit": opportunities / _JOB_NAMES[1]}
    _safe_unlink(opportunities, _JOB_NAMES)
    try:
        prompt = Path(prompt_path)
        if not prompt.is_file() or prompt.resolve(strict=True) != prompt.resolve():
            raise ValueError("opportunity prompt is missing or unresolved")
        prompt_text = prompt.read_text(encoding="utf-8")
        jobs, audit = _job_records(bundle, prompt_text)
        write_jsonl_bundle({targets["jobs"]: jobs, targets["audit"]: audit})
        return targets["jobs"]
    except BaseException:
        _safe_unlink(opportunities, _JOB_NAMES)
        raise


emit_opportunity_jobs = emit_mining_jobs


def validate_mining_job_bundle(run_dir: Path) -> MiningBundle:
    bundle = validate_landscape_bundle(run_dir)
    opportunities = _owned_dir(bundle.run, "opportunities", create=False)
    jobs = read_jsonl(_owned_file(opportunities, _JOB_NAMES[0]))
    audit = read_jsonl(_owned_file(opportunities, _JOB_NAMES[1]))
    prompt_texts = {row.get("prompt_text") for row in [*jobs, *audit]}
    if jobs or audit:
        if len(prompt_texts) != 1 or type(next(iter(prompt_texts))) is not str:
            raise ValueError("mining jobs and audit have inconsistent prompt snapshots")
        expected_jobs, expected_audit = _job_records(bundle, next(iter(prompt_texts)))
    else:
        expected_jobs = []
        expected_audit = []
    if jobs != expected_jobs or audit != expected_audit:
        raise ValueError("mining job or cluster audit replay mismatch")
    return MiningBundle(bundle.run, bundle, tuple(jobs), tuple(audit))


class _ResultWrapper(StrictModel):
    schema_version: str
    job_id: str
    cluster_id: str
    operator: OpportunityOperator
    prompt_sha256: str
    landscape_hashes: dict[str, str]
    card_ids: list[str]
    neighbor_ids: list[str]
    opportunities: list[dict[str, Any]]


def _raw_result(value: str | bytes | Path) -> tuple[bytes, str]:
    if isinstance(value, Path):
        raw = value.read_bytes()
    elif type(value) is str:
        raw = value.encode("utf-8")
    elif type(value) is bytes:
        raw = value
    else:
        raise TypeError("opportunity results must be external strict JSON text or paths")
    try:
        return raw, raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("opportunity result must be UTF-8 JSON") from exc


def _rejection(job: dict[str, Any], raw_hash: str, index: int | None, reason_codes: list[str], raw_value: object, error: str) -> dict[str, Any]:
    return {
        "schema_version": "idea_factory.rejected_opportunity_result.v2",
        "record_id": stable_id("rejected_opportunity", job["job_id"], str(index), raw_hash, ",".join(reason_codes)),
        "job_id": job["job_id"], "cluster_id": job["cluster_id"],
        "raw_result_sha256": raw_hash, "index": index, "reason_codes": reason_codes,
        "error": error, "raw_value": raw_value,
    }


def _project_results(mining: MiningBundle, raw_by_job: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = {row["job_id"]: row for row in mining.jobs}
    if set(raw_by_job) != set(jobs):
        raise ValueError("opportunity results must exactly cover one result per mining job")
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    global_ids: set[str] = set()
    for job_id in sorted(jobs):
        job = jobs[job_id]
        raw_text = raw_by_job[job_id]
        raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        accepted_for_job: list[dict[str, Any]] = []
        rejected_for_job: list[dict[str, Any]] = []
        wrapper_reconstructable = False
        try:
            data = _strict_json(raw_text)
            wrapper = _ResultWrapper.model_validate_json(_canonical_json(data), strict=True)
            expected = {
                "schema_version": OPPORTUNITY_RESULT_SCHEMA_VERSION, "job_id": job_id,
                "cluster_id": job["cluster_id"], "operator": job["operator"],
                "prompt_sha256": job["prompt_sha256"], "landscape_hashes": job["landscape_hashes"],
                "card_ids": job["card_ids"], "neighbor_ids": job["neighbor_ids"],
            }
            if any(getattr(wrapper, key) != value for key, value in expected.items()):
                raise ValueError("stale opportunity result binding")
            wrapper_reconstructable = True
        except (ValidationError, ValueError) as exc:
            data = _strict_json(raw_text)
            rejected_for_job.append(_rejection(job, raw_hash, None, ["INVALID_RESULT_WRAPPER"], data, str(exc)))
            wrapper = None
        if wrapper is not None:
            for index, raw_opportunity in enumerate(wrapper.opportunities):
                reason_codes: list[str] = []
                try:
                    if type(raw_opportunity) is not dict:
                        raise ValueError("opportunity must be a native JSON object")
                    opportunity = Opportunity.model_validate_json(_canonical_json(raw_opportunity), strict=True)
                    if opportunity.operator != wrapper.operator:
                        reason_codes.append("OPERATOR_BINDING_MISMATCH")
                    if not set(opportunity.supporting_card_ids) <= set(job["card_ids"]):
                        reason_codes.append("UNKNOWN_SUPPORTING_CARD_ID")
                    if not set(opportunity.nearest_internal_neighbors) <= set(job["neighbor_ids"]):
                        reason_codes.append("UNKNOWN_NEIGHBOR_ID")
                    if opportunity.opportunity_id in global_ids:
                        reason_codes.append("DUPLICATE_OPPORTUNITY_ID")
                    if reason_codes:
                        raise ValueError("; ".join(reason_codes))
                    global_ids.add(opportunity.opportunity_id)
                    record = {
                        "schema_version": "idea_factory.opportunity_candidate.v2",
                        "record_id": opportunity.opportunity_id, "job_id": job_id,
                        "cluster_id": job["cluster_id"], "index": index,
                        "raw_result_sha256": raw_hash, "job_sha256": _canonical_hash(job),
                        "landscape_hashes": job["landscape_hashes"], "card_ids": job["card_ids"],
                        "neighbor_ids": job["neighbor_ids"], "opportunity": opportunity.model_dump(mode="json"),
                    }
                    accepted_for_job.append(record)
                except (ValidationError, ValueError) as exc:
                    if not reason_codes:
                        reason_codes = ["INVALID_OPPORTUNITY_SCHEMA"]
                    rejected_for_job.append(_rejection(job, raw_hash, index, sorted(set(reason_codes)), raw_opportunity, str(exc)))
        candidates.extend(accepted_for_job)
        rejected.extend(rejected_for_job)
        accepted_ids = [row["record_id"] for row in accepted_for_job]
        rejected_ids = [row["record_id"] for row in rejected_for_job]
        if wrapper is None:
            status = "INVALID"
        elif not wrapper.opportunities:
            status = "VALID_EMPTY"
        elif accepted_for_job and rejected_for_job:
            status = "PARTIAL_VALID"
        elif accepted_for_job:
            status = "ACCEPTED"
        else:
            status = "INVALID"
        outcomes.append({
            "schema_version": OPPORTUNITY_OUTCOME_SCHEMA_VERSION, "job_id": job_id,
            "cluster_id": job["cluster_id"], "raw_result_sha256": raw_hash,
            "raw_result_json": raw_text, "status": status,
            "accepted_opportunity_ids": accepted_ids, "accepted_count": len(accepted_ids),
            "rejected_result_ids": rejected_ids, "rejected_count": len(rejected_ids),
            "accepted_projection_sha256": _canonical_hash(accepted_for_job),
            "rejected_projection_sha256": _canonical_hash(rejected_for_job),
            "raw_wrapper_reconstructable": wrapper_reconstructable,
        })
    return candidates, rejected, outcomes


@dataclass(frozen=True)
class OpportunityResultBundle:
    run: Path
    mining: MiningBundle
    candidates: tuple[dict[str, Any], ...]
    rejected: tuple[dict[str, Any], ...]
    outcomes: tuple[dict[str, Any], ...]


def _result_paths(opportunities: Path) -> dict[str, Path]:
    return {
        "candidates": opportunities / _RESULT_NAMES[0],
        "rejected": opportunities / _RESULT_NAMES[1],
        "outcomes": opportunities / _RESULT_NAMES[2],
    }


def ingest_opportunity_results(jobs_path: Path, results: Sequence[str | bytes | Path]) -> dict[str, Path]:
    run = _anchored_run(Path(jobs_path).parent.parent)
    opportunities = _owned_dir(run, "opportunities", create=False)
    if Path(os.path.abspath(jobs_path)) != opportunities / _JOB_NAMES[0]:
        raise ValueError("mining jobs must use the exact active run path")
    mining = validate_mining_job_bundle(run)
    _invalidate_result_and_quality(run, opportunities)
    paths = _result_paths(opportunities)
    try:
        raw_by_job: dict[str, str] = {}
        job_ids = {job["job_id"] for job in mining.jobs}
        for result in results:
            _raw, text = _raw_result(result)
            data = _strict_json(text)
            job_id = data.get("job_id")
            if type(job_id) is not str or job_id not in job_ids:
                raise ValueError("unexpected opportunity result job_id")
            if job_id in raw_by_job:
                raise ValueError("duplicate opportunity result job_id")
            raw_by_job[job_id] = text
        candidates, rejected, outcomes = _project_results(mining, raw_by_job)
        write_jsonl_bundle({paths["candidates"]: candidates, paths["rejected"]: rejected, paths["outcomes"]: outcomes})
        return paths
    except BaseException:
        _safe_unlink(opportunities, _RESULT_NAMES)
        raise


def validate_opportunity_result_bundle(run_dir: Path) -> OpportunityResultBundle:
    mining = validate_mining_job_bundle(run_dir)
    opportunities = _owned_dir(mining.run, "opportunities", create=False)
    paths = _result_paths(opportunities)
    candidates = read_jsonl(_owned_file(opportunities, paths["candidates"].name))
    rejected = read_jsonl(_owned_file(opportunities, paths["rejected"].name))
    outcomes = read_jsonl(_owned_file(opportunities, paths["outcomes"].name))
    job_ids = [row.get("job_id") for row in outcomes]
    if len(job_ids) != len(set(job_ids)) or set(job_ids) != {row["job_id"] for row in mining.jobs}:
        raise ValueError("result outcomes do not exactly cover mining jobs")
    raw_by_job: dict[str, str] = {}
    for outcome in outcomes:
        raw_text = outcome.get("raw_result_json")
        if type(raw_text) is not str or hashlib.sha256(raw_text.encode("utf-8")).hexdigest() != outcome.get("raw_result_sha256"):
            raise ValueError("result outcome raw replay binding mismatch")
        raw_by_job[outcome["job_id"]] = raw_text
    expected = _project_results(mining, raw_by_job)
    if candidates != expected[0] or rejected != expected[1] or outcomes != expected[2]:
        raise ValueError("opportunity result replay projection mismatch")
    return OpportunityResultBundle(mining.run, mining, tuple(candidates), tuple(rejected), tuple(outcomes))
