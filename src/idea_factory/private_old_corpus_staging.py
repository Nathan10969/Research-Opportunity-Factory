"""Strict private staging of remaining frozen old-corpus Router/Card inputs.

No run initialization, ingestion, network, model, Card, or source mutation occurs.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .corpus import CorpusRouterConfig, enumerate_candidates, _load_primary_source_manifest
from .private_router_aggregate import (
    _json, _lines, _physical_row, _pinned, _sha, _write_new,
    adapt_legacy_router_envelope,
)


POSITION_LINES = re.compile(r"^- Shard ([1-5]) [^:]+: `([0-9, -]+)`\.$", re.M)
SHA = re.compile(r"[0-9a-f]{64}\Z")
HARD_HOLDS = {
    "RETAINED_REVIEW_LINE_POINTER_DISCREPANCY_HOLD",
    "RETAINED_REVIEW_REFERENCE_BINDING_MISMATCH_HOLD",
}


def expand_census_positions(report: str, *, expected_count: int = 811) -> set[tuple[int, int]]:
    """Read only the ten frozen position lines, never result counts or prose."""
    matches = POSITION_LINES.findall(report)
    if len(matches) != 10 and expected_count == 811:
        raise ValueError("expected exactly ten frozen census position lines")
    if not matches:
        raise ValueError("no frozen census positions")
    positions: set[tuple[int, int]] = set()
    for shard, expression in matches:
        for part in expression.split(","):
            bounds = [int(number) for number in part.strip().split("-")]
            if len(bounds) not in (1, 2) or (len(bounds) == 2 and bounds[0] > bounds[1]):
                raise ValueError("invalid census position range")
            for position in range(bounds[0], bounds[-1] + 1):
                item = (int(shard), position)
                if item in positions:
                    raise ValueError("overlap in frozen census positions")
                positions.add(item)
    if len(positions) != expected_count:
        raise ValueError(f"frozen census count is {len(positions)}, expected {expected_count}")
    return positions


def load_frozen_parent(package: Path, summary: dict[str, Any],
                       positions: set[tuple[int, int]]
                       ) -> list[tuple[dict[str, Any], Path, str, int, str]]:
    """Rehash each frozen shard and return only physical rows at cited positions."""
    result: list[tuple[dict[str, Any], Path, str, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for shard_no, pin in enumerate(summary["shards"], 1):
        path = Path(package) / pin["file"]
        try:
            data = _pinned(path, pin["sha256"])
        except ValueError as exc:
            raise ValueError(f"frozen shard SHA-256 failure: {path}") from exc
        lines = _lines(data)
        if len(lines) != pin["rows"]:
            raise ValueError("frozen shard row count mismatch")
        for position, line in enumerate(lines, 1):
            key = (shard_no, position)
            if key in positions:
                row = _json(line)
                result.append((row, path, pin["sha256"], position, _sha(line)))
                seen.add(key)
    if seen != positions:
        raise ValueError("frozen census position outside shard range")
    return result


def select_remaining(rows: list[dict[str, Any]], formal: set[str], pilot: set[str]
                     ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Exclude by exact identity, with hard HOLD taking absolute precedence."""
    selected: list[dict[str, Any]] = []
    excluded: collections.Counter[str] = collections.Counter()
    seen: set[str] = set()
    for row in rows:
        slug = row.get("slug")
        if not isinstance(slug, str) or slug in seen:
            raise ValueError("duplicate or invalid frozen slug")
        seen.add(slug)
        binding = row.get("assignment_input_binding") or {}
        if binding.get("status") in HARD_HOLDS:
            excluded["FROZEN_HARD_HOLD"] += 1
        elif slug in pilot:
            excluded["PRIOR_PILOT100"] += 1
        elif slug in formal:
            excluded["FORMAL_CARD"] += 1
        else:
            selected.append(row)
    return selected, dict(excluded)


def select_accepted_subset(original: list[dict[str, Any]], jobs: list[dict[str, Any]],
                           outcomes: list[dict[str, Any]], cards: list[dict[str, Any]]
                           ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Exact one-to-one old run Card outcome filter, without repairing invalid rows."""
    indexed: list[dict[str, dict[str, Any]]] = []
    for rows in (original, jobs, outcomes, cards):
        mapping = {row["slug"]: row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError("duplicate slug in accepted subset inputs")
        indexed.append(mapping)
    old, by_job, by_outcome, by_card = indexed
    if set(old) != set(by_job) or set(old) != set(by_outcome):
        raise ValueError("prior run job/outcome set differs from original allowlist")
    ready: list[dict[str, Any]] = []
    holds: list[dict[str, str]] = []
    accepted: set[str] = set()
    for row in original:
        slug = row["slug"]
        job, outcome = by_job[slug], by_outcome[slug]
        if (job.get("job_id") != outcome.get("job_id") or
            job.get("note_sha256") != row.get("note_sha256") or
            outcome.get("note_sha256") != row.get("note_sha256")):
            raise ValueError(f"prior Card job identity mismatch: {slug}")
        for key in ("prompt_sha256",):
            if key in job and key in outcome and job[key] != outcome[key]:
                raise ValueError(f"prior Card job identity mismatch: {slug}")
        if outcome.get("status") == "ACCEPTED":
            ids = outcome.get("accepted_card_ids")
            card = by_card.get(slug)
            if (outcome.get("accepted_count") != 1 or type(ids) is not list or len(ids) != 1 or
                card is None or card.get("job_id") != job["job_id"] or card.get("record_id") != ids[0]):
                raise ValueError(f"accepted Card projection mismatch: {slug}")
            accepted.add(slug)
            ready.append(row)
        else:
            if slug in by_card:
                raise ValueError(f"nonaccepted Card has formal projection: {slug}")
            holds.append({"slug": slug, "status": "HOLD",
                          "reason": f"prior Card outcome {outcome.get('status')}"})
    if set(by_card) != accepted:
        raise ValueError("accepted Card projection set mismatch")
    return ready, holds


def _result_hashes(value: Any, context: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            label = f"{context}.{key}".lower()
            if isinstance(item, str) and SHA.fullmatch(item) and "result" in label:
                found.add(item)
            else:
                found.update(_result_hashes(item, label))
    elif isinstance(value, list):
        for item in value:
            found.update(_result_hashes(item, context))
    return found


def _corrected_hashes(value: Any, context: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            label = f"{context}.{key}".lower()
            if isinstance(item, str) and SHA.fullmatch(item) and "corrected" in label and "result" in label:
                found.add(item)
            else:
                found.update(_corrected_hashes(item, label))
    elif isinstance(value, list):
        for item in value:
            found.update(_corrected_hashes(item, context))
    return found


def _preferred_result_hashes(value: Any, context: str = "") -> set[str]:
    """Current/corrected result pins exclude explicitly original/superseded pins."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            label = f"{context}.{key}".lower()
            if isinstance(item, str) and SHA.fullmatch(item) and "result" in label:
                if "original" not in label and "superseded" not in label:
                    found.add(item)
            else:
                found.update(_preferred_result_hashes(item, label))
    elif isinstance(value, list):
        for item in value:
            found.update(_preferred_result_hashes(item, context))
    return found


def _item_statuses(value: Any, slug: str) -> list[str]:
    statuses: list[str] = []
    if isinstance(value, dict):
        if value.get("slug") == slug:
            statuses.extend(value[key] for key in ("status", "verdict", "disposition")
                            if isinstance(value.get(key), str))
        for item in value.values():
            statuses.extend(_item_statuses(item, slug))
    elif isinstance(value, list):
        for item in value:
            statuses.extend(_item_statuses(item, slug))
    return statuses


def qa_binds_result(qa: dict[str, Any], result_sha: str, slug: str,
                    *, result_count: int = 1) -> bool:
    if result_sha not in _result_hashes(qa):
        return False
    top = [qa.get(key) for key in ("verdict", "status", "disposition", "qa_verdict", "overall_verdict")]
    top_pass = any(isinstance(value, str) and value.startswith("PASS") for value in top)
    if any(isinstance(value, str) and (value.startswith("HOLD") or value.startswith("FAIL")) for value in top):
        return False
    statuses = _item_statuses(qa, slug)
    if any(not value.startswith("PASS") for value in statuses):
        return False
    if any(value.startswith("PASS") for value in statuses):
        return True
    if not top_pass:
        return False
    checks = json.dumps({key: qa.get(key) for key in
                         ("checks", "independent_checks", "binding_and_schema_checks", "mechanical_audit")},
                        ensure_ascii=False)
    return bool(re.search(rf"\b{result_count}/{result_count}\b", checks))


def choose_unique_qa_result(candidates: list[dict[str, Any]], reports: list[dict[str, Any]],
                            slug: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use explicit corrected pins, never QA/result filesystem timestamps."""
    covered: list[tuple[dict[str, Any], dict[str, Any], bool]] = []
    for report in reports:
        qa = report["qa"]
        pinned = _result_hashes(qa)
        corrected = _corrected_hashes(qa)
        preferred = _preferred_result_hashes(qa)
        for candidate in candidates:
            digest = candidate["file_sha256"]
            if digest not in pinned or (preferred and digest not in preferred):
                continue
            statuses = _item_statuses(qa, slug)
            if any(value.startswith(("HOLD", "FAIL")) for value in statuses):
                raise ValueError("explicit item QA HOLD/FAIL")
            if qa_binds_result(qa, digest, slug,
                               result_count=candidate.get("result_count", 1)):
                covered.append((candidate, report, digest in corrected or
                                (len(preferred) < len(pinned) and digest in preferred)))
    if not covered:
        raise ValueError("no exact private result physical row plus QA-PASS pin")
    if any(item[2] for item in covered):
        covered = [item for item in covered if item[2]]
    distinct = {(item[0]["file_sha256"], item[0]["row_sha256"]) for item in covered}
    if len(distinct) != 1:
        raise ValueError("ambiguous QA-covered private result revisions")
    # Same bytes may be reported by multiple QA files; choose stable path, not mtime.
    covered.sort(key=lambda item: (item[0]["path"], item[1]["path"]))
    return covered[0][0], covered[0][1]


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False,
                                 sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                    for row in rows)


def _pin(path: str, digest: str) -> bytes:
    return _pinned(Path(path), digest)


def validate_worker_card_links(frozen: dict[str, Any], review: dict[str, Any],
                               receipt: dict[str, Any]) -> None:
    """A review records creation; the receipt records terminal Card status."""
    bindings = review.get("bindings") if isinstance(review.get("bindings"), dict) else {}
    candidate_paths = [review.get(key) for key in
                       ("note_path", "canonical_note_path", "source_note_path")]
    candidate_paths.append(bindings.get("note_path"))
    candidate_hashes = [review.get(key) for key in
                        ("note_sha256", "canonical_note_sha256", "source_note_sha256")]
    candidate_hashes.append(bindings.get("note_sha256"))
    anchored_paths = [value for value in candidate_paths if isinstance(value, str)]
    anchored_hashes = [value for value in candidate_hashes if isinstance(value, str)]
    status = review.get("status", review.get("review_status"))
    reviewed_raw_path = review.get("reviewed_raw_path")
    if (not anchored_paths and not anchored_hashes or
        (review.get("item_id") is not None and review["item_id"] != frozen["slug"]) or
        any(value != frozen["canonical_note_path"] for value in anchored_paths) or
        any(value != frozen["canonical_note_sha256"] for value in anchored_hashes) or
        (status is not None and status not in {"CREATED", "ACCEPT", "PASS"}) or
        (reviewed_raw_path is not None and reviewed_raw_path != frozen["raw_path"])):
        raise ValueError("old Card review binding mismatch")
    if (receipt.get("slug") != frozen["slug"] or
        receipt.get("status") != "SCHEMA_VALID" or
        receipt.get("accepted_cards") != 1 or
        receipt.get("raw_result_sha256") != frozen["raw_sha256"]):
        raise ValueError("old Card receipt terminal status/binding mismatch")


def _candidates_from_result_files(private_root: Path) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for path in sorted(private_root.rglob("*.jsonl")):
        if "router-result" not in path.name.lower() and "router_result" not in path.name.lower():
            continue
        data = path.read_bytes()
        digest = _sha(data)
        try:
            lines = _lines(data)
        except ValueError:
            continue
        for number, line in enumerate(lines, 1):
            try:
                result = _json(line)
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if result.get("schema_version") != "idea_factory.corpus_router_result.v3":
                continue
            slug = result.get("slug")
            if isinstance(slug, str):
                candidates[slug].append({"path": str(path), "file_sha256": digest,
                                         "line": number, "row_sha256": _sha(line),
                                         "row": result, "mtime_ns": path.stat().st_mtime_ns})
    return candidates


def _qa_by_result_hash(quality: Path, private_root: Path) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    paths = sorted(quality.glob("*.json")) + sorted(private_root.rglob("*.json"))
    for path in paths:
        try:
            raw = path.read_bytes()
            qa = _json(raw)
        except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for digest in _result_hashes(qa):
            found[digest].append({"path": str(path), "sha256": _sha(raw),
                                  "qa": qa, "mtime_ns": path.stat().st_mtime_ns})
    return found


def _pinned_row(row: dict[str, Any], shard_path: Path, shard_sha: str,
                position: int, frozen_hash: str, source_records: dict[str, dict[str, Any]],
                results: dict[str, list[dict[str, Any]]],
                qa_by_hash: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    slug = row["slug"]
    if row.get("accepted_card_count") != 1 or row.get("current_card_status") not in {
        "SCHEMA_VALID", "REUSED_SCHEMA_VALID"
    }:
        raise ValueError("no reusable nonempty private Card")
    bind = row.get("assignment_input_binding") or {}
    if bind.get("status") != "EXACT_WORKER_QUEUE_BOUND":
        raise ValueError("old worker queue is not exact-bound")
    if row.get("source_status") != "PRIMARY_SOURCE_VERIFIED":
        raise ValueError("source identity is not primary-verified")
    source = source_records.get(row["canonical_note_path"])
    if source is None or any(source.get(key) != value for key, value in {
        "slug": slug, "note_sha256": row["canonical_note_sha256"],
        "primary_pdf_path": row["primary_pdf_path"],
        "primary_pdf_sha256": row["primary_pdf_sha256"],
        "source_record_ids": row["source_record_ids"],
        "source_status": row["source_status"], "source_version": row["source_version"],
    }.items()):
        raise ValueError("frozen source record mismatch")
    for path_key, sha_key in (("canonical_note_path", "canonical_note_sha256"),
                              ("primary_pdf_path", "primary_pdf_sha256"),
                              ("task_path", "task_sha256"), ("raw_path", "raw_sha256"),
                              ("receipt_path", "receipt_sha256"),
                              ("prompt_path", "prompt_sha256"),
                              ("primary_source_manifest_path", "primary_source_manifest_sha256"),
                              ("run_manifest_path", "run_manifest_sha256"),
                              ("router_jobs_path", "router_jobs_file_sha256")):
        _pin(row[path_key], row[sha_key])
    queue_data = _pin(bind["path"], bind["file_sha256"])
    _physical_row(queue_data, bind["line"], bind["row_sha256"])
    job_data = _pin(row["router_jobs_path"], row["router_jobs_file_sha256"])
    _, job = _physical_row(job_data, row["router_job_line"], row["router_job_line_sha256"])
    for key, value in (("job_id", row["job_id"]), ("slug", slug),
                       ("note_sha256", row["canonical_note_sha256"]),
                       ("prompt_sha256", row["prompt_sha256"])):
        if job.get(key) != value:
            raise ValueError(f"old Router job {key} mismatch")
    review_path = Path(row["task_path"]).parent / "review.json"
    if not review_path.is_file() or row["current_card_status"] != "SCHEMA_VALID":
        raise ValueError("reused/absent Card review needs separate exact QA")
    review_bytes = review_path.read_bytes()
    review = _json(review_bytes)
    receipt = _json(_pin(row["receipt_path"], row["receipt_sha256"]))
    validate_worker_card_links(row, review, receipt)
    matched_candidates: list[dict[str, Any]] = []
    relevant_reports: dict[str, dict[str, Any]] = {}
    for result in results.get(slug, []):
        body = result["row"]
        if any(body.get(key) != value for key, value in {
            "job_id": row["job_id"], "slug": slug,
            "note_sha256": row["canonical_note_sha256"],
            "prompt_sha256": row["prompt_sha256"],
        }.items()):
            continue
        _, legacy_hash = adapt_legacy_router_envelope(body)
        result = {**result, "result_count": len(_lines(Path(result["path"]).read_bytes())),
                  "legacy_router_canonical_sha256": legacy_hash}
        matched_candidates.append(result)
        for qa in qa_by_hash.get(result["file_sha256"], []):
            relevant_reports[qa["path"]] = qa
    selected, qa = choose_unique_qa_result(matched_candidates, list(relevant_reports.values()), slug)
    source_data = _pin(selected["path"], selected["file_sha256"])
    _physical_row(source_data, selected["line"], selected["row_sha256"])
    _pin(qa["path"], qa["sha256"])
    return {
        "schema_version": "engineering.old811_card_pilot100_strict_v3_input_allowlist.v1",
        "ordinal": 0, "slug": slug,
        "venue": slug.split("-")[1].upper() if slug.startswith("2026-") else "UNKNOWN",
        "status": "INPUT_PRECHECK_PASS_SCIENCE_PENDING", "hold_reason": None,
        "frozen_shard_path": str(shard_path), "frozen_shard_sha256": shard_sha,
        "frozen_position": position, "frozen_row_sha256": frozen_hash,
        "source_manifest_path": row["primary_source_manifest_path"],
        "source_manifest_sha256": row["primary_source_manifest_sha256"],
        "source_record_ids": row["source_record_ids"],
        "source_status": row["source_status"], "source_version": row["source_version"],
        "pdf_path": row["primary_pdf_path"], "pdf_sha256": row["primary_pdf_sha256"],
        "note_path": row["canonical_note_path"], "note_sha256": row["canonical_note_sha256"],
        "task_path": row["task_path"], "task_sha256": row["task_sha256"],
        "raw_path": row["raw_path"], "raw_sha256": row["raw_sha256"],
        "review_path": str(review_path), "review_sha256": _sha(review_bytes),
        "reuse_path": None, "reuse_sha256": None,
        "card_receipt_path": row["receipt_path"], "card_receipt_sha256": row["receipt_sha256"],
        "queue_path": bind["path"], "queue_sha256": bind["file_sha256"],
        "queue_line": bind["line"], "queue_row_sha256": bind["row_sha256"],
        "router_job_id": row["job_id"],
        "old_router_job_path": row["router_jobs_path"],
        "old_router_job_file_sha256": row["router_jobs_file_sha256"],
        "old_router_job_line": row["router_job_line"],
        "old_router_job_row_sha256": row["router_job_line_sha256"],
        "router_prompt_path": row["prompt_path"], "router_prompt_sha256": row["prompt_sha256"],
        "private_router_result_path": selected["path"],
        "private_router_result_file_sha256": selected["file_sha256"],
        "private_router_result_line": selected["line"],
        "private_router_result_row_sha256": selected["row_sha256"],
        "legacy_router_canonical_sha256": selected["legacy_router_canonical_sha256"],
        "private_qa_path": qa["path"], "private_qa_sha256": qa["sha256"],
        "private_qa_scope": "PASS",
        "current_card_status": row["current_card_status"],
        "source_admission_approved": False, "science_approved": False,
        "human_approved": False, "run_init_approved": False,
        "ingest_approved": False, "graph_approved": False,
    }


def write_staging(output_dir: Path, ready: list[dict[str, Any]], holds: list[dict[str, Any]],
                  *, notes_root: Path, legacy_ledger: Path, input_pins: dict[str, Any],
                  preflight: bool = False, parent_count: int = 811,
                  receipt_schema: str = "engineering.old811_remaining_strict_v3_staging_receipt.v1"
                  ) -> dict[str, Any]:
    """Publish a new private input packet; never create or change a run."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not output_dir.is_absolute() or output_dir.parent.resolve(strict=True) != output_dir.parent:
        raise ValueError("output must be a new absolute directory")
    slugs = [row["slug"] for row in ready]
    if len(slugs) != len(set(slugs)) or len({row["note_path"] for row in ready}) != len(ready):
        raise ValueError("duplicate ready identity or canonical note")
    if set(slugs) & {row["slug"] for row in holds}:
        raise ValueError("READY/HOLD overlap")
    if not ready:
        raise ValueError("no pinned ready candidate; no run input may be staged")
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + ".staging-", dir=output_dir.parent))
    allowlist = [{**row, "ordinal": index} for index, row in enumerate(ready, 1)]
    sources = [{"slug": row["slug"], "canonical_note_path": row["note_path"],
                "note_sha256": row["note_sha256"], "primary_pdf_path": row["pdf_path"],
                "primary_pdf_sha256": row["pdf_sha256"],
                "source_record_ids": row["source_record_ids"],
                "source_status": row["source_status"], "source_version": row["source_version"],
                "adjudication_path": None, "adjudication_sha256": None} for row in ready]
    files = {
        "allowlist.jsonl": _jsonl_bytes(allowlist),
        "hold_ledger.jsonl": _jsonl_bytes(holds),
        "candidate-list.txt": "".join(row["note_path"] + "\n" for row in ready).encode("utf-8"),
        "primary-source-manifest.json": (json.dumps(
            {"schema_version": "idea_factory.primary_source_manifest.v1", "records": sources},
            ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    }
    for name, data in files.items():
        _write_new(staging / name, data)
    config = {
        "notes_root": str(Path(notes_root).resolve(strict=True)),
        "candidate_lists": [str(output_dir / "candidate-list.txt")],
        "legacy_ledger": str(Path(legacy_ledger).resolve(strict=True)),
        "target_min": len(ready), "target_max": len(ready),
        "allowed_labels": ["GENERAL_RESEARCH"], "bridge_regression_slugs": [],
        "protocol_version": "v3",
        "source_primary_manifest": str(output_dir / "primary-source-manifest.json"),
    }
    config_bytes = (json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _write_new(staging / "corpus-router-v3-config.json", config_bytes)
    if preflight:
        _, _, loaded = _load_primary_source_manifest(staging / "primary-source-manifest.json")
        if len(loaded) != len(ready):
            raise ValueError("strict source manifest size mismatch")
        checked = CorpusRouterConfig(
            notes_root=Path(config["notes_root"]), candidate_lists=(staging / "candidate-list.txt",),
            legacy_ledger=Path(config["legacy_ledger"]), target_min=len(ready), target_max=len(ready),
            allowed_labels=("GENERAL_RESEARCH",), bridge_regression_slugs=(), protocol_version="v3",
            source_primary_manifest=staging / "primary-source-manifest.json")
        enumerated = enumerate_candidates(checked)
        if {item.slug for item in enumerated} != set(slugs) or len(enumerated) != len(ready):
            raise ValueError("frozen loader candidate set mismatch")
    distribution = dict(sorted(collections.Counter(row["venue"] for row in ready).items()))
    reasons = dict(sorted(collections.Counter(row["reason"] for row in holds).items()))
    receipt = {"schema_version": receipt_schema,
               "status": "PRIVATE_INPUT_PRECHECK_ONLY", "parent_count": parent_count,
               "ready_count": len(ready), "hold_count": len(holds),
               "ready_by_venue": distribution, "hold_by_reason": reasons,
               "input_pins": input_pins,
               "files": {name: {"sha256": _sha(data), "bytes": len(data)} for name, data in
                         {**files, "corpus-router-v3-config.json": config_bytes}.items()},
               "strict_loader_preflight": preflight,
               "source_admission_approved": False, "science_approved": False,
               "human_approved": False, "run_init_approved": False,
               "router_ingest_approved": False, "card_ingest_approved": False,
               "graph_approved": False, "paid_or_network_calls": False}
    _write_new(staging / "receipt.v1.json", (json.dumps(receipt, ensure_ascii=False,
                                                         sort_keys=True, indent=2) + "\n").encode("utf-8"))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    os.rename(staging, output_dir)
    if preflight:
        final_config = CorpusRouterConfig.from_json(output_dir / "corpus-router-v3-config.json")
        final_candidates = enumerate_candidates(final_config)
        if len(final_candidates) != len(ready):
            raise ValueError("final strict loader candidate count mismatch")
    return receipt


def build_remaining(runs_root: Path, package: Path, census_path: Path, census_sha256: str,
                    pilot_allowlist: Path, pilot_sha256: str, output_dir: Path,
                    *, expected_parent_count: int = 811) -> dict[str, Any]:
    """Construct the remaining exact-position subset; nonmatching rows stay HOLD."""
    runs_root, package = Path(runs_root), Path(package)
    report = _pinned(Path(census_path), census_sha256).decode("utf-8")
    positions = expand_census_positions(report, expected_count=expected_parent_count)
    summary_path = package / "summary.json"
    summary_bytes = summary_path.read_bytes()
    summary = _json(summary_bytes)
    if summary.get("schema_version") != "engineering.real_source1534_router_shards_summary.v1":
        raise ValueError("unexpected frozen shard summary schema")
    parent = load_frozen_parent(package, summary, positions)
    if len(parent) != expected_parent_count:
        raise ValueError("frozen parent count mismatch")
    pilot = [_json(line) for line in _lines(_pinned(Path(pilot_allowlist), pilot_sha256))]
    if expected_parent_count == 811 and len(pilot) != 100:
        raise ValueError("prior pilot allowlist is not the exact 100")
    pilot_slugs = {row["slug"] for row in pilot}
    if len(pilot_slugs) != len(pilot):
        raise ValueError("duplicate prior pilot slug")
    formal_files = sorted(path for path in runs_root.rglob("paper_cards.jsonl")
                          if path.parent.name == "results")
    formal_slugs: set[str] = set()
    formal_pins: list[dict[str, Any]] = []
    for path in formal_files:
        data = path.read_bytes()
        rows = [_json(line) for line in _lines(data)] if data else []
        for row in rows:
            if isinstance(row.get("slug"), str):
                formal_slugs.add(row["slug"])
        formal_pins.append({"path": str(path), "sha256": _sha(data), "rows": len(rows)})
    selected, excluded = select_remaining([item[0] for item in parent], formal_slugs, pilot_slugs)
    by_slug = {row["slug"]: item for item in parent for row in (item[0],)}
    if len(by_slug) != len(parent):
        raise ValueError("duplicate frozen parent slug")
    source_manifest_path = Path(parent[0][0]["primary_source_manifest_path"])
    source_manifest_sha = parent[0][0]["primary_source_manifest_sha256"]
    source_manifest = _json(_pinned(source_manifest_path, source_manifest_sha))
    source_records = {row["canonical_note_path"]: row for row in source_manifest["records"]}
    if len(source_records) != len(source_manifest["records"]):
        raise ValueError("duplicate original source manifest note")
    private_root = package / "private-results"
    quality = package.parent.parent / "quality"
    result_index = _candidates_from_result_files(private_root)
    qa_index = _qa_by_result_hash(quality, private_root)
    ready: list[dict[str, Any]] = []
    holds: list[dict[str, Any]] = []
    for row in selected:
        _, shard_path, shard_sha, position, row_sha = by_slug[row["slug"]]
        try:
            staged = _pinned_row(row, shard_path, shard_sha, position, row_sha,
                                 source_records, result_index, qa_index)
            ready.append(staged)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            holds.append({"slug": row["slug"], "status": "HOLD",
                          "reason": str(exc), "frozen_shard_path": str(shard_path),
                          "frozen_shard_sha256": shard_sha, "frozen_position": position,
                          "frozen_row_sha256": row_sha,
                          "current_card_status": row.get("current_card_status")})
    pdf_groups: dict[str, list[str]] = collections.defaultdict(list)
    for row in ready:
        pdf_groups[row["pdf_sha256"]].append(row["slug"])
    duplicate_pdf_slugs = {slug for members in pdf_groups.values() if len(members) > 1 for slug in members}
    if duplicate_pdf_slugs:
        retained = []
        for row in ready:
            if row["slug"] in duplicate_pdf_slugs:
                holds.append({"slug": row["slug"], "status": "HOLD",
                              "reason": "duplicate primary PDF SHA among remaining rows",
                              "frozen_shard_path": row["frozen_shard_path"],
                              "frozen_shard_sha256": row["frozen_shard_sha256"],
                              "frozen_position": row["frozen_position"],
                              "frozen_row_sha256": row["frozen_row_sha256"],
                              "current_card_status": row["current_card_status"]})
            else:
                retained.append(row)
        ready = retained
    if len(ready) + len(holds) + sum(excluded.values()) != expected_parent_count:
        raise ValueError("frozen parent conservation failed")
    if not ready:
        raise ValueError("no strictly pinned reusable rows remain; cannot create run input")
    old_job_path = Path(ready[0]["old_router_job_path"])
    old_job_data = _pinned(old_job_path, ready[0]["old_router_job_file_sha256"])
    _, old_job = _physical_row(old_job_data, ready[0]["old_router_job_line"],
                               ready[0]["old_router_job_row_sha256"])
    legacy_ledger = Path(old_job["legacy_ledger_path"])
    _pinned(legacy_ledger, _sha(legacy_ledger.read_bytes()))
    inputs = {"census": {"path": str(census_path), "sha256": census_sha256},
              "summary": {"path": str(summary_path), "sha256": _sha(summary_bytes)},
              "shards": [{"path": str(package / pin["file"]), "sha256": pin["sha256"]}
                         for pin in summary["shards"]],
              "pilot_allowlist": {"path": str(pilot_allowlist), "sha256": pilot_sha256},
              "source_manifest": {"path": str(source_manifest_path),
                                  "sha256": source_manifest_sha},
              "formal_results": formal_pins,
              "formal_unique_slug_count": len(formal_slugs),
              "excluded": excluded,
              "selected_after_exclusions": len(selected),
              "legacy_ledger": {"path": str(legacy_ledger),
                                "sha256": _sha(legacy_ledger.read_bytes())}}
    for pin in formal_pins:
        _pinned(Path(pin["path"]), pin["sha256"])
    _pinned(Path(census_path), census_sha256)
    _pinned(Path(pilot_allowlist), pilot_sha256)
    return write_staging(output_dir, ready, holds, notes_root=quality.parent / "workers",
                         legacy_ledger=legacy_ledger, input_pins=inputs, preflight=True)


def build_accepted_subset(original_allowlist: Path, original_sha256: str,
                          prior_run: Path, output_dir: Path) -> dict[str, Any]:
    """Restage exact ACCEPTED Card outcomes from the first 100, never its two INVALIDs."""
    original_allowlist, prior_run = Path(original_allowlist), Path(prior_run)
    original = [_json(line) for line in _lines(_pinned(original_allowlist, original_sha256))]
    paths = {"jobs": prior_run / "cards" / "card_jobs.jsonl",
             "outcomes": prior_run / "results" / "card_job_outcomes.jsonl",
             "cards": prior_run / "results" / "paper_cards.jsonl"}
    raw = {key: path.read_bytes() for key, path in paths.items()}
    jobs = [_json(line) for line in _lines(raw["jobs"])]
    outcomes = [_json(line) for line in _lines(raw["outcomes"])]
    cards = [_json(line) for line in _lines(raw["cards"])]
    ready, holds = select_accepted_subset(original, jobs, outcomes, cards)
    if len(original) != 100 or len(ready) != 98 or len(holds) != 2:
        raise ValueError("prior 100-run is not exact 98 ACCEPTED plus 2 nonaccepted")
    if {item["slug"] for item in holds} != {"2026-aaai-37125", "2026-aaai-37195"}:
        raise ValueError("unexpected nonaccepted prior pilot identities")
    for row in ready:
        if row.get("status") != "INPUT_PRECHECK_PASS_SCIENCE_PENDING":
            raise ValueError("prior allowlist row not in private precheck status")
        for path_key, digest_key in (
            ("frozen_shard_path", "frozen_shard_sha256"),
            ("source_manifest_path", "source_manifest_sha256"),
            ("pdf_path", "pdf_sha256"), ("note_path", "note_sha256"),
            ("task_path", "task_sha256"), ("raw_path", "raw_sha256"),
            ("review_path", "review_sha256"),
            ("card_receipt_path", "card_receipt_sha256"),
            ("queue_path", "queue_sha256"),
            ("old_router_job_path", "old_router_job_file_sha256"),
            ("router_prompt_path", "router_prompt_sha256"),
            ("private_router_result_path", "private_router_result_file_sha256"),
            ("private_qa_path", "private_qa_sha256"),
        ):
            _pin(row[path_key], row[digest_key])
        for path_key, line_key, digest_key in (
            ("frozen_shard_path", "frozen_position", "frozen_row_sha256"),
            ("queue_path", "queue_line", "queue_row_sha256"),
            ("old_router_job_path", "old_router_job_line", "old_router_job_row_sha256"),
            ("private_router_result_path", "private_router_result_line",
             "private_router_result_row_sha256"),
        ):
            _physical_row(Path(row[path_key]).read_bytes(), row[line_key], row[digest_key])
    config_path = original_allowlist.parent / "corpus-router-v3-config.json"
    config_bytes = config_path.read_bytes()
    original_config = _json(config_bytes)
    original_manifest = _json(_pinned(Path(original[0]["source_manifest_path"]),
                                      original[0]["source_manifest_sha256"]))
    by_slug = {item["slug"]: item for item in original_manifest["records"]}
    for row in ready:
        source = by_slug.get(row["slug"])
        if source is None or any(source.get(key) != value for key, value in {
            "canonical_note_path": row["note_path"], "note_sha256": row["note_sha256"],
            "primary_pdf_path": row["pdf_path"], "primary_pdf_sha256": row["pdf_sha256"],
            "source_record_ids": row["source_record_ids"],
            "source_status": row["source_status"], "source_version": row["source_version"],
        }.items()):
            raise ValueError(f"original source manifest identity mismatch: {row['slug']}")
    for key, path in paths.items():
        _pinned(path, _sha(raw[key]))
    _pinned(original_allowlist, original_sha256)
    pins = {"original_allowlist": {"path": str(original_allowlist), "sha256": original_sha256},
            "original_config": {"path": str(config_path), "sha256": _sha(config_bytes)},
            "prior_run": str(prior_run),
            "prior_run_files": {key: {"path": str(path), "sha256": _sha(raw[key]),
                                      "rows": len(_lines(raw[key]))}
                                for key, path in paths.items()},
            "accepted_card_count": len(cards),
            "nonaccepted_exact_slugs": sorted(row["slug"] for row in holds)}
    return write_staging(output_dir, ready, holds,
                         notes_root=Path(original_config["notes_root"]),
                         legacy_ledger=Path(original_config["legacy_ledger"]),
                         input_pins=pins, preflight=True, parent_count=100,
                         receipt_schema="engineering.old811_accepted98_strict_v3_staging_receipt.v1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("remaining", "accepted_subset"), default="remaining")
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--frozen-package", type=Path)
    parser.add_argument("--census", type=Path)
    parser.add_argument("--census-sha256")
    parser.add_argument("--pilot-allowlist", type=Path)
    parser.add_argument("--pilot-sha256")
    parser.add_argument("--prior-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == "remaining":
            if not all((args.runs_root, args.frozen_package, args.census,
                        args.census_sha256, args.pilot_allowlist, args.pilot_sha256)):
                parser.error("remaining mode needs runs-root, frozen-package, census, and pilot pins")
            result = build_remaining(args.runs_root, args.frozen_package, args.census,
                                     args.census_sha256, args.pilot_allowlist,
                                     args.pilot_sha256, args.output_dir)
        else:
            if not all((args.pilot_allowlist, args.pilot_sha256, args.prior_run)):
                parser.error("accepted_subset needs pilot allowlist pin and prior-run")
            result = build_accepted_subset(args.pilot_allowlist, args.pilot_sha256,
                                           args.prior_run, args.output_dir)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"strict old-corpus staging HOLD: {exc}\n")
    print(json.dumps({"ready_count": result["ready_count"], "hold_count": result["hold_count"],
                      "ready_by_venue": result["ready_by_venue"],
                      "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
