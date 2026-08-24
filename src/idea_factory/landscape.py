"""Deterministic, evidence-preserving concept landscapes for accepted paper cards."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from .cards import validate_card_result_bundle
from .artifacts import sha256_file
from .models import (
    ConceptAssignment,
    PaperCard,
    ReviewedConceptAssignmentV2,
    Scope,
)


LANDSCAPE_SCHEMA_VERSION = "idea_factory.landscape.v1"
ASSIGNMENT_ARTIFACT_SCHEMA_VERSION = "idea_factory.concept_assignments.v1"
ASSIGNMENT_ARTIFACT_SCHEMA_VERSION_V2 = "idea_factory.concept_assignments.v2"
NO_ASSIGNMENT_ARTIFACT_SCHEMA_VERSION = "idea_factory.concept_assignments.none.v1"
_OUTPUT_NAMES = {
    "assumptions": "assumptions.json",
    "failures": "failures.json",
    "mechanisms": "mechanisms.json",
    "evaluations": "evaluations.json",
    "edges": "card_concept_edges.csv",
    "assignments": "reviewed_assignments.json",
}
_EDGE_FIELDS = (
    "card_id", "source_paper", "source_path", "dimension", "facet", "raw_text",
    "normalized_text", "scope", "scope_key", "cluster_scope", "cluster_scope_key",
    "scope_reviewed", "evidence_status", "reviewed", "disputed", "merge_reason",
    "review_status",
)


@dataclass(frozen=True)
class _AssignmentDecision:
    assignment: ConceptAssignment
    cluster_scope: Scope
    scope_reviewed: bool


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _same_json_value(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            _same_json_value(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _same_json_value(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def normalize_concept(value: str) -> str:
    """Perform the only allowed automatic normalization: lowercase + whitespace."""

    if not isinstance(value, str):
        raise ValueError("concept text must be a string")
    normalized = re.sub(r"\s+", " ", value.strip()).lower()
    if not normalized:
        raise ValueError("concept text normalizes to blank")
    return normalized


def canonical_scope_key(scope: Scope | dict[str, Any]) -> str:
    """Return a delimiter-safe canonical structured scope identifier."""

    parsed = Scope.model_validate(scope)
    return _canonical_json(parsed.model_dump(mode="json"))


def _scope_payload(scope: Scope | dict[str, Any]) -> dict[str, str]:
    return Scope.model_validate(scope).model_dump(mode="json")


def _safe_output_paths(run_value: Path) -> tuple[Path, dict[str, Path]]:
    lexical_run = Path(os.path.abspath(run_value))
    if not lexical_run.is_dir() or str(lexical_run.resolve(strict=True)) != str(lexical_run):
        raise ValueError("landscape run must be a non-junction active run directory")
    landscape = lexical_run / "landscape"
    ancestor = landscape
    while not ancestor.exists() and ancestor.parent != ancestor:
        ancestor = ancestor.parent
    if not ancestor.resolve(strict=True).is_relative_to(lexical_run):
        raise ValueError("landscape directory escapes the same active run")
    if landscape.exists() and str(landscape.resolve(strict=True)) != str(landscape):
        raise ValueError("landscape directory escapes the same active run")
    landscape.mkdir(exist_ok=True)
    return landscape, {key: landscape / name for key, name in _OUTPUT_NAMES.items()}


def _invalidate(outputs: dict[str, Path]) -> None:
    for target in outputs.values():
        target.unlink(missing_ok=True)


def _stage_text(path: Path, data: str) -> Path:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        staged = Path(temporary)
        temporary = None
        return staged
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _csv_text(edges: list[dict[str, Any]]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_EDGE_FIELDS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for edge in edges:
        writer.writerow({key: edge[key] for key in _EDGE_FIELDS})
    return buffer.getvalue()


def _publish(
    outputs: dict[str, Path],
    maps: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    assignment_artifact: dict[str, Any],
) -> None:
    staged: dict[Path, Path] = {}
    try:
        for name in ("assumptions", "failures", "mechanisms", "evaluations"):
            staged[outputs[name]] = _stage_text(outputs[name], json.dumps(maps[name], ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n")
        staged[outputs["edges"]] = _stage_text(outputs["edges"], _csv_text(edges))
        staged[outputs["assignments"]] = _stage_text(
            outputs["assignments"],
            json.dumps(assignment_artifact, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n",
        )
        for target in outputs.values():
            os.replace(staged[target], target)
            staged.pop(target)
    except BaseException:
        _invalidate(outputs)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def _base_assignments(cards: Iterable[PaperCard]) -> tuple[list[ConceptAssignment], int]:
    assignments: list[ConceptAssignment] = []
    missing_unknown_mechanisms = 0
    for card in cards:
        scope_key = canonical_scope_key(card.scope)
        values = (
            ("assumptions", "assumption", card.assumption.text, card.assumption.status.value),
            ("failures", "failure_observation", card.failure_observation.text, card.failure_observation.status.value),
            ("mechanisms", "mechanism", card.mechanism, None),
            ("evaluations", "measurement", card.evaluation.measurement, None),
            ("evaluations", "regime", card.evaluation.regime, None),
        )
        if card.failure_mechanism.status.value == "UNKNOWN" and not card.failure_mechanism.text.strip():
            missing_unknown_mechanisms += 1
        else:
            values += (("failures", "failure_mechanism", card.failure_mechanism.text, card.failure_mechanism.status.value),)
        for dimension, facet, raw_text, _status in values:
            raw = raw_text.strip()
            assignments.append(ConceptAssignment(
                dimension=dimension, facet=facet, raw_text=raw,
                normalized_text=normalize_concept(raw), card_id=card.card_id,
                scope_key=scope_key,
            ))
    return assignments, missing_unknown_mechanisms


def _assignment_target(row: ConceptAssignment) -> tuple[str, str, str, str, str]:
    return (row.card_id, row.dimension, row.facet, row.raw_text, row.scope_key)


def _scope_from_key(scope_key: str) -> Scope:
    try:
        value = json.loads(scope_key, object_pairs_hook=_strict_json_object)
        scope = Scope.model_validate(value, strict=True)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        raise ValueError("assignment source scope key must be canonical Scope JSON") from exc
    if canonical_scope_key(scope) != scope_key:
        raise ValueError("assignment source scope key is not canonical")
    return scope


def _review_overrides(
    path: Path | None,
    base: list[ConceptAssignment],
) -> tuple[
    dict[tuple[str, str, str, str, str], _AssignmentDecision],
    str | None,
    dict[str, Any],
]:
    if path is None:
        artifact = {
            "schema_version": NO_ASSIGNMENT_ARTIFACT_SCHEMA_VERSION,
            "assignments": [],
            "artifact_sha256": _sha256([]),
        }
        return {}, None, artifact
    try:
        data = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            object_pairs_hook=_strict_json_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("reviewed assignment artifact must be strict UTF-8 JSON") from exc
    if (
        not isinstance(data, dict)
        or set(data) != {"schema_version", "assignments", "artifact_sha256"}
        or data["schema_version"] not in {
            ASSIGNMENT_ARTIFACT_SCHEMA_VERSION,
            ASSIGNMENT_ARTIFACT_SCHEMA_VERSION_V2,
            NO_ASSIGNMENT_ARTIFACT_SCHEMA_VERSION,
        }
        or not isinstance(data["assignments"], list)
    ):
        raise ValueError("invalid reviewed assignment artifact schema")
    canonical_records = sorted(data["assignments"], key=_canonical_json)
    if data["artifact_sha256"] != _sha256(canonical_records):
        raise ValueError("reviewed assignment artifact hash mismatch")
    if data["schema_version"] == NO_ASSIGNMENT_ARTIFACT_SCHEMA_VERSION:
        if canonical_records:
            raise ValueError("no-assignment artifact cannot contain assignments")
        artifact = {
            "schema_version": NO_ASSIGNMENT_ARTIFACT_SCHEMA_VERSION,
            "assignments": [],
            "artifact_sha256": _sha256([]),
        }
        return {}, None, artifact
    decisions: list[_AssignmentDecision] = []
    for raw_row in data["assignments"]:
        if type(raw_row) is not dict:
            raise ValueError("reviewed assignment rows must be strict JSON objects")
        row_json = _canonical_json(raw_row)
        try:
            if data["schema_version"] == ASSIGNMENT_ARTIFACT_SCHEMA_VERSION:
                row = ConceptAssignment.model_validate_json(row_json, strict=True)
                decision = _AssignmentDecision(row, _scope_from_key(row.scope_key), False)
                model_value = row.model_dump(mode="json")
            else:
                reviewed_v2 = ReviewedConceptAssignmentV2.model_validate_json(row_json, strict=True)
                row = ConceptAssignment(
                    dimension=reviewed_v2.dimension,
                    facet=reviewed_v2.facet,
                    raw_text=reviewed_v2.raw_text,
                    normalized_text=reviewed_v2.normalized_text,
                    card_id=reviewed_v2.card_id,
                    scope_key=reviewed_v2.source_scope_key,
                    merge_reason=reviewed_v2.merge_reason,
                    disputed=reviewed_v2.disputed,
                    reviewed=reviewed_v2.reviewed,
                    review_status=reviewed_v2.review_status,
                    reviewer_reason=reviewed_v2.reviewer_reason,
                )
                decision = _AssignmentDecision(
                    row,
                    Scope.model_validate(reviewed_v2.cluster_scope.model_dump(mode="json"), strict=True),
                    True,
                )
                model_value = reviewed_v2.model_dump(mode="json")
        except ValidationError as exc:
            raise ValueError("invalid reviewed assignment row under strict JSON validation") from exc
        if not _same_json_value(json.loads(row_json), model_value):
            raise ValueError("reviewed assignment validation normalizes or coerces raw JSON")
        decisions.append(decision)
    base_by_target = {_assignment_target(row): row for row in base}
    overrides: dict[tuple[str, str, str, str, str], _AssignmentDecision] = {}
    for decision in decisions:
        row = decision.assignment
        target = _assignment_target(row)
        if target in overrides:
            raise ValueError("duplicate reviewed assignment override")
        if target not in base_by_target:
            raise ValueError("stale or orphan reviewed assignment override")
        if row.review_status == "APPROVED":
            if not row.reviewed or row.disputed or not row.merge_reason.strip():
                raise ValueError("approved override requires reviewed non-disputed merge reason")
            if row.normalized_text != normalize_concept(row.normalized_text):
                raise ValueError("approved override normalized text must use mechanical normalization")
            if decision.scope_reviewed and not row.reviewer_reason.strip():
                raise ValueError("v2 approved override requires a canonical-scope reviewer reason")
        elif row.review_status == "DISPUTED":
            if not row.reviewed or not row.disputed:
                raise ValueError("disputed override must be reviewed and disputed")
            decision = _AssignmentDecision(row, _scope_from_key(row.scope_key), False)
        else:
            raise ValueError("reviewed assignment artifact may contain only approved or disputed overrides")
        overrides[target] = decision
    artifact = {
        "schema_version": data["schema_version"],
        "assignments": canonical_records,
        "artifact_sha256": data["artifact_sha256"],
    }
    return overrides, data["artifact_sha256"], artifact


def _edge(card: PaperCard, decision: _AssignmentDecision, scope: dict[str, str], evidence_status: str | None) -> dict[str, Any]:
    assignment = decision.assignment
    cluster_scope = _scope_payload(decision.cluster_scope)
    return {
        "card_id": card.card_id,
        "source_paper": card.paper.title,
        "source_path": card.paper.source_path,
        "dimension": assignment.dimension,
        "facet": assignment.facet,
        "raw_text": assignment.raw_text,
        "normalized_text": assignment.normalized_text,
        "scope": _canonical_json(scope),
        "scope_key": assignment.scope_key,
        "cluster_scope": _canonical_json(cluster_scope),
        "cluster_scope_key": canonical_scope_key(cluster_scope),
        "scope_reviewed": "true" if decision.scope_reviewed else "false",
        "evidence_status": evidence_status or "",
        "reviewed": "true" if assignment.reviewed else "false",
        "disputed": "true" if assignment.disputed else "false",
        "merge_reason": assignment.merge_reason,
        "review_status": assignment.review_status,
    }


def _final_assignment_decisions(
    base: list[ConceptAssignment],
    overrides: dict[tuple[str, str, str, str, str], _AssignmentDecision],
) -> list[_AssignmentDecision]:
    final: list[_AssignmentDecision] = []
    for assignment in base:
        reviewed = overrides.get(_assignment_target(assignment))
        if reviewed is None:
            final.append(_AssignmentDecision(assignment, _scope_from_key(assignment.scope_key), False))
        elif reviewed.assignment.review_status == "DISPUTED":
            final.append(_AssignmentDecision(assignment.model_copy(update={
                "reviewed": True, "disputed": True, "review_status": "DISPUTED",
                "merge_reason": reviewed.assignment.merge_reason,
                "reviewer_reason": reviewed.assignment.reviewer_reason,
            }), _scope_from_key(assignment.scope_key), False))
        else:
            final.append(reviewed)
    return final


def _edges_for_cards(
    cards_by_id: dict[str, PaperCard],
    decisions: list[_AssignmentDecision],
) -> list[dict[str, Any]]:
    status_by_assignment = {
        (card.card_id, "assumptions", "assumption"): card.assumption.status.value
        for card in cards_by_id.values()
    }
    status_by_assignment |= {
        (card.card_id, "failures", "failure_observation"): card.failure_observation.status.value
        for card in cards_by_id.values()
    }
    status_by_assignment |= {
        (card.card_id, "failures", "failure_mechanism"): card.failure_mechanism.status.value
        for card in cards_by_id.values() if card.failure_mechanism.text.strip()
    }
    edges = [
        _edge(
            cards_by_id[item.assignment.card_id], item,
            _scope_payload(cards_by_id[item.assignment.card_id].scope),
            status_by_assignment.get((item.assignment.card_id, item.assignment.dimension, item.assignment.facet)),
        )
        for item in decisions
    ]
    edges.sort(key=lambda edge: (
        edge["dimension"], edge["facet"], edge["normalized_text"],
        edge["cluster_scope_key"], edge["scope_key"], edge["card_id"], edge["raw_text"],
    ))
    return edges


def _maps(edges: list[dict[str, Any]], hashes: dict[str, str], counts: dict[str, int], assignment_hash: str | None, missing_unknown: int) -> dict[str, dict[str, Any]]:
    maps: dict[str, dict[str, Any]] = {}
    for dimension in ("assumptions", "failures", "mechanisms", "evaluations"):
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            if edge["dimension"] == dimension:
                grouped[(edge["facet"], edge["normalized_text"], edge["cluster_scope_key"])].append(edge)
        entries: list[dict[str, Any]] = []
        for (facet, normalized, scope_key), group in sorted(grouped.items()):
            first = group[0]
            entries.append({
                "facet": facet,
                "normalized_text": normalized,
                "scope": json.loads(first["cluster_scope"]),
                "scope_key": scope_key,
                "source_card_ids": sorted({edge["card_id"] for edge in group}),
                "raw_phrases": sorted({edge["raw_text"] for edge in group}),
                "source_papers": sorted({edge["source_paper"] for edge in group}),
                "source_paths": sorted({edge["source_path"] for edge in group}),
                "evidence_statuses": sorted({edge["evidence_status"] for edge in group if edge["evidence_status"]}),
                "reviewed_count": sum(edge["reviewed"] == "true" for edge in group),
                "disputed_count": sum(edge["disputed"] == "true" for edge in group),
                "has_disputed_assignments": any(edge["disputed"] == "true" for edge in group),
            })
        maps[dimension] = {
            "schema_version": LANDSCAPE_SCHEMA_VERSION,
            "dimension": dimension,
            "input_bundle_hashes": dict(sorted(hashes.items())),
            "input_bundle_counts": dict(sorted(counts.items())),
            "assignment_artifact_hash": assignment_hash,
            "assignment_policy": (
                "reviewed-canonical-scope-overrides-v2"
                if assignment_hash and any(edge["scope_reviewed"] == "true" for edge in edges)
                else "reviewed-overrides-v1"
                if assignment_hash
                else "mechanical-lowercase-whitespace-v1"
            ),
            "entries": entries,
        }
    maps["failures"]["missing_unknown_failure_mechanism_count"] = missing_unknown
    return maps


def build_landscape(run: Path, reviewed_assignments_path: Path | None = None) -> dict[str, Path]:
    """Build all landscape artifacts from one complete, revalidated Task-5 bundle."""

    run_path = Path(run)
    if run_path.name == "card_jobs.jsonl":
        run_path = run_path.parent.parent
    _, outputs = _safe_output_paths(run_path)
    _invalidate(outputs)
    try:
        jobs_path = Path(run_path) / "cards" / "card_jobs.jsonl"
        paths, accepted, _rejected, _outcomes = validate_card_result_bundle(
            jobs_path,
            expected_run_dir=run_path,
        )
        cards = [PaperCard.model_validate(row["card"]) for row in accepted]
        cards_by_id = {card.card_id: card for card in cards}
        base, missing_unknown = _base_assignments(cards)
        overrides, assignment_hash, assignment_artifact = _review_overrides(reviewed_assignments_path, base)
        final_assignments = _final_assignment_decisions(base, overrides)
        hashes = {
            "paper_cards.jsonl": _sha256(sorted(accepted, key=_canonical_json)),
            "rejected_paper_cards.jsonl": _sha256(sorted(_rejected, key=_canonical_json)),
            "card_job_outcomes.jsonl": _sha256(sorted(_outcomes, key=_canonical_json)),
        }
        counts = {
            "accepted_card_count": len(accepted),
            "rejected_card_record_count": len(_rejected),
            "job_outcome_count": len(_outcomes),
        }
        edges = _edges_for_cards(cards_by_id, final_assignments)
        maps = _maps(edges, hashes, counts, assignment_hash, missing_unknown)
        _publish(outputs, maps, edges, assignment_artifact)
        return {key: outputs[key] for key in _OUTPUT_NAMES}
    except BaseException:
        _invalidate(outputs)
        raise


@dataclass(frozen=True)
class ValidatedLandscapeBundle:
    """Fully replayed Task-6 bundle rooted at one exact active run."""

    run: Path
    cards: dict[str, PaperCard]
    hashes: dict[str, str]
    maps: dict[str, dict[str, Any]]
    edges: tuple[dict[str, str], ...]


def _validated_run(run_dir: Path) -> Path:
    run = Path(os.path.abspath(run_dir))
    try:
        resolved = run.resolve(strict=True)
    except OSError as exc:
        raise ValueError("expected run directory is missing") from exc
    if not run.is_dir() or run != resolved:
        raise ValueError("expected run directory must have exact resolved identity")
    return run


def _strict_json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
            object_pairs_hook=_strict_json_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid strict landscape JSON: {path.name}") from exc
    if type(value) is not dict:
        raise ValueError(f"landscape map must be a JSON object: {path.name}")
    return value


def _card_edge_targets(cards: dict[str, PaperCard]) -> dict[tuple[str, str, str, str, str], tuple[PaperCard, str]]:
    targets: dict[tuple[str, str, str, str, str], tuple[PaperCard, str]] = {}
    for card in cards.values():
        scope_key = canonical_scope_key(card.scope)
        values = [
            ("assumptions", "assumption", card.assumption.text.strip(), card.assumption.status.value),
            ("failures", "failure_observation", card.failure_observation.text.strip(), card.failure_observation.status.value),
            ("mechanisms", "mechanism", card.mechanism.strip(), ""),
            ("evaluations", "measurement", card.evaluation.measurement.strip(), ""),
            ("evaluations", "regime", card.evaluation.regime.strip(), ""),
        ]
        if card.failure_mechanism.text.strip():
            values.append(("failures", "failure_mechanism", card.failure_mechanism.text.strip(), card.failure_mechanism.status.value))
        for dimension, facet, raw_text, status in values:
            key = (card.card_id, dimension, facet, raw_text, scope_key)
            if key in targets:
                raise ValueError("accepted cards produce duplicate landscape edge targets")
            targets[key] = (card, status)
    return targets


def validate_landscape_bundle(run_dir: Path) -> ValidatedLandscapeBundle:
    """Validate all Task-5 bindings and the exact ordered Task-6 projection."""

    run = _validated_run(Path(run_dir))
    _paths, accepted, rejected, outcomes = validate_card_result_bundle(
        run / "cards" / "card_jobs.jsonl", expected_run_dir=run
    )
    cards = {row["record_id"]: PaperCard.model_validate(row["card"]) for row in accepted}
    if len(cards) != len(accepted):
        raise ValueError("landscape input has duplicate card IDs")
    landscape = run / "landscape"
    try:
        resolved_landscape = landscape.resolve(strict=True)
    except OSError as exc:
        raise ValueError("complete landscape directory is missing") from exc
    if not landscape.is_dir() or resolved_landscape != run / "landscape":
        raise ValueError("landscape directory is not anchored to the expected run")
    if {child.name for child in landscape.iterdir()} != set(_OUTPUT_NAMES.values()):
        raise ValueError("landscape bundle has unexpected or missing owned paths")
    paths = {key: landscape / name for key, name in _OUTPUT_NAMES.items()}
    for path in paths.values():
        if not path.is_file() or path.resolve(strict=True) != run / "landscape" / path.name:
            raise ValueError("landscape artifact is not anchored to the expected run")

    maps = {dimension: _strict_json_file(paths[dimension]) for dimension in ("assumptions", "failures", "mechanisms", "evaluations")}
    assignment_artifact = _strict_json_file(paths["assignments"])
    with paths["edges"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _EDGE_FIELDS:
            raise ValueError("landscape CSV has an invalid exact schema")
        edge_rows = [dict(row) for row in reader]
    if len({_canonical_json(row) for row in edge_rows}) != len(edge_rows):
        raise ValueError("duplicate landscape CSV edge row")
    expected_order = sorted(
        edge_rows,
        key=lambda edge: (
            edge["dimension"], edge["facet"], edge["normalized_text"],
            edge["cluster_scope_key"], edge["scope_key"], edge["card_id"], edge["raw_text"],
        ),
    )
    if edge_rows != expected_order:
        raise ValueError("landscape CSV ordered projection mismatch")

    base_assignments, _missing_unknown = _base_assignments(cards.values())
    overrides, replayed_assignment_hash, canonical_assignment_artifact = _review_overrides(
        paths["assignments"], base_assignments,
    )
    if not _same_json_value(assignment_artifact, canonical_assignment_artifact):
        raise ValueError("run-owned assignment artifact canonical replay mismatch")
    replayed_edges = _edges_for_cards(
        cards, _final_assignment_decisions(base_assignments, overrides),
    )
    if edge_rows != replayed_edges:
        raise ValueError("landscape edge assignment replay mismatch")

    targets = _card_edge_targets(cards)
    seen_targets: set[tuple[str, str, str, str, str]] = set()
    for edge in edge_rows:
        target = (edge["card_id"], edge["dimension"], edge["facet"], edge["raw_text"], edge["scope_key"])
        if target in seen_targets:
            raise ValueError("duplicate landscape edge target")
        seen_targets.add(target)
        expected = targets.get(target)
        if expected is None:
            raise ValueError("landscape edge does not project an accepted card field")
        card, evidence_status = expected
        scope_json = _canonical_json(_scope_payload(card.scope))
        cluster_scope = _scope_from_key(edge["cluster_scope_key"])
        if (
            edge["source_paper"] != card.paper.title
            or edge["source_path"] != card.paper.source_path
            or edge["scope"] != scope_json
            or edge["cluster_scope"] != _canonical_json(_scope_payload(cluster_scope))
            or edge["evidence_status"] != evidence_status
        ):
            raise ValueError("landscape edge card, paper, scope, or status projection mismatch")
        if edge["reviewed"] not in {"true", "false"} or edge["disputed"] not in {"true", "false"} or edge["scope_reviewed"] not in {"true", "false"}:
            raise ValueError("landscape edge has invalid review flags")
        if edge["review_status"] == "MECHANICAL":
            if edge["reviewed"] != "false" or edge["disputed"] != "false" or edge["scope_reviewed"] != "false" or edge["cluster_scope"] != edge["scope"] or edge["merge_reason"] or edge["normalized_text"] != normalize_concept(edge["raw_text"]):
                raise ValueError("invalid mechanical landscape edge")
        elif edge["review_status"] == "DISPUTED":
            if edge["reviewed"] != "true" or edge["disputed"] != "true" or edge["scope_reviewed"] != "false" or edge["cluster_scope"] != edge["scope"] or edge["normalized_text"] != normalize_concept(edge["raw_text"]):
                raise ValueError("invalid disputed landscape edge")
        elif edge["review_status"] == "APPROVED":
            if edge["reviewed"] != "true" or edge["disputed"] != "false" or not edge["merge_reason"].strip() or normalize_concept(edge["normalized_text"]) != edge["normalized_text"]:
                raise ValueError("invalid approved landscape edge")
            if edge["scope_reviewed"] == "false" and edge["cluster_scope"] != edge["scope"]:
                raise ValueError("legacy approved edge cannot change cluster scope")
        else:
            raise ValueError("landscape edge has unknown review status")
    if seen_targets != set(targets):
        raise ValueError("landscape edges do not exactly cover accepted card fields")

    bindings = {
        "paper_cards.jsonl": _sha256(sorted(accepted, key=_canonical_json)),
        "rejected_paper_cards.jsonl": _sha256(sorted(rejected, key=_canonical_json)),
        "card_job_outcomes.jsonl": _sha256(sorted(outcomes, key=_canonical_json)),
    }
    counts = {
        "accepted_card_count": len(accepted),
        "rejected_card_record_count": len(rejected),
        "job_outcome_count": len(outcomes),
    }
    assignment_hashes = {data.get("assignment_artifact_hash") for data in maps.values()}
    policies = {data.get("assignment_policy") for data in maps.values()}
    if len(assignment_hashes) != 1 or len(policies) != 1:
        raise ValueError("landscape maps have inconsistent assignment policy bindings")
    assignment_hash = next(iter(assignment_hashes))
    if assignment_hash is not None and (type(assignment_hash) is not str or not re.fullmatch(r"[0-9a-f]{64}", assignment_hash)):
        raise ValueError("landscape assignment hash is invalid")
    if assignment_hash != replayed_assignment_hash:
        raise ValueError("landscape assignment artifact binding mismatch")
    expected_maps = _maps(
        edge_rows, bindings, counts, assignment_hash,
        sum(card.failure_mechanism.status.value == "UNKNOWN" and not card.failure_mechanism.text.strip() for card in cards.values()),
    )
    for dimension in maps:
        if not _same_json_value(maps[dimension], expected_maps[dimension]):
            raise ValueError(f"landscape aggregate projection mismatch: {dimension}")
    # The run-owned assignment artifact is already bound by the hash embedded in
    # every map. Keep the historical five-file downstream hash contract stable
    # while requiring and replay-validating the sixth provenance file here.
    hashes = {
        path.name: sha256_file(path)
        for key, path in paths.items()
        if key != "assignments"
    }
    return ValidatedLandscapeBundle(run, cards, dict(sorted(hashes.items())), maps, tuple(edge_rows))
