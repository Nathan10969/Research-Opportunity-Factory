"""Strict validation for explicitly reviewed two-card opportunity entries."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .models import OpportunityOperator


_SCHEMA = "idea_factory.reviewed_local_entries.v1"
_DOC_FIELDS = {"schema_version", "landscape_hashes", "entries", "artifact_sha256"}
_ENTRY_FIELDS = {
    "entry_id", "entry_kind", "anchors", "shared_object", "interface_relation",
    "compatibility_reason", "scope_differences", "proposed_hypothesis",
    "alternative_explanation", "decisive_test", "novelty_boundary",
    "nearest_prior_card_ids", "operator", "review",
}
_ANCHOR_FIELDS = {"card_id", "dimension", "facet", "raw_text", "source_scope_key", "role"}
_REVIEW_FIELDS = {"status", "reviewer_id", "reviewer_role", "reason"}


def canonical_artifact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(document: object) -> dict[str, Any]:
    if isinstance(document, Path):
        text = document.read_text(encoding="utf-8")
    elif isinstance(document, str):
        text = document
    elif isinstance(document, Mapping):
        return dict(document)
    else:
        raise ValueError("local-entry document must be JSON text, path, or object")
    try:
        value = json.loads(text, object_pairs_hook=_strict_pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {x}")))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid strict local-entry JSON") from exc
    except ValueError as exc:
        if str(exc).startswith("duplicate JSON key"):
            raise
        raise ValueError("invalid strict local-entry JSON") from exc
    except TypeError as exc:
        raise ValueError("invalid strict local-entry JSON") from exc
    if type(value) is not dict:
        raise ValueError("local-entry document must be a JSON object")
    return value


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")
    return value


def _fields(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} has unexpected or missing fields")
    return value


def _hash_without_self(document: dict[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    try:
        encoded = canonical_artifact_json(payload).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("local-entry artifact is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def canonical_artifact_sha256(document: Mapping[str, Any]) -> str:
    """Hash a local-entry document excluding its self-referential hash field."""
    return _hash_without_self(dict(document))


def _canonical_path(value: object) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("source path must be a nonblank string")
    return os.path.normcase(os.path.normpath(value.replace("\\", "/"))).casefold()


def validate_local_entries(document: object, validated_landscape: Any) -> list[dict[str, Any]]:
    """Validate local reviewed entries against an already validated landscape.

    Returns all entries, including REJECTED review rows; callers must omit
    rejected rows when constructing jobs.
    """
    doc = _load(document)
    _fields(doc, _DOC_FIELDS, "document")
    if doc["schema_version"] != _SCHEMA:
        raise ValueError("unsupported local-entry schema")
    if type(doc["landscape_hashes"]) is not dict or doc["landscape_hashes"] != validated_landscape.hashes:
        raise ValueError("stale landscape hash binding")
    if type(doc["entries"]) is not list:
        raise ValueError("entries must be a list")
    if type(doc["artifact_sha256"]) is not str or doc["artifact_sha256"] != _hash_without_self(doc):
        raise ValueError("local-entry artifact hash mismatch")

    edges = list(validated_landscape.edges)
    edge_by_key = {(e["card_id"], e["dimension"], e["facet"], e["raw_text"], e["scope_key"]): e for e in edges}
    card_paths = {card_id: card.paper.source_path for card_id, card in validated_landscape.cards.items()}
    card_ids = set(validated_landscape.cards)
    seen_ids: set[str] = set()
    output: list[dict[str, Any]] = []
    for entry in doc["entries"]:
        row = _fields(entry, _ENTRY_FIELDS, "entry")
        entry_id = _string(row["entry_id"], "entry_id")
        if entry_id in seen_ids:
            raise ValueError("duplicate entry_id")
        seen_ids.add(entry_id)
        if type(row["entry_kind"]) is not str or row["entry_kind"] not in {"PAIR_RELATION", "CROSS_FACET"}:
            raise ValueError("invalid entry_kind")
        anchors = row["anchors"]
        if type(anchors) is not list or len(anchors) != 2:
            raise ValueError("entry must have exactly two anchors")
        checked: list[dict[str, Any]] = []
        for anchor in anchors:
            item = _fields(anchor, _ANCHOR_FIELDS, "anchor")
            for key in _ANCHOR_FIELDS:
                _string(item[key], f"anchor.{key}")
            key = (item["card_id"], item["dimension"], item["facet"], item["raw_text"], item["source_scope_key"])
            if key not in edge_by_key:
                raise ValueError("stale or mismatched anchor binding")
            checked.append(item)
        if checked[0]["card_id"] == checked[1]["card_id"]:
            raise ValueError("anchors must reference two distinct cards")
        if _canonical_path(card_paths[checked[0]["card_id"]]) == _canonical_path(card_paths[checked[1]["card_id"]]):
            raise ValueError("anchors must reference two distinct source papers")
        if row["entry_kind"] == "PAIR_RELATION":
            left, right = (edge_by_key[(a["card_id"], a["dimension"], a["facet"], a["raw_text"], a["source_scope_key"])] for a in checked)
            if not (left["facet"] == right["facet"] and left["normalized_text"] == right["normalized_text"] and left["cluster_scope_key"] == right["cluster_scope_key"]):
                raise ValueError("PAIR_RELATION anchors must share facet, normalized text, and cluster scope")
        elif checked[0]["facet"] == checked[1]["facet"]:
            raise ValueError("CROSS_FACET anchors must have different facets")
        for key in _ENTRY_FIELDS - {"entry_id", "entry_kind", "anchors", "nearest_prior_card_ids", "operator", "review"}:
            _string(row[key], f"entry.{key}")
        priors = row["nearest_prior_card_ids"]
        if type(priors) is not list or not priors or any(type(item) is not str or not item.strip() for item in priors) or len(priors) != len(set(priors)) or not set(priors) <= card_ids:
            raise ValueError("nearest_prior_card_ids must be unique existing card IDs")
        if type(row["operator"]) is not str or row["operator"] not in {operator.value for operator in OpportunityOperator}:
            raise ValueError("invalid operator")
        review = _fields(row["review"], _REVIEW_FIELDS, "review")
        if type(review["status"]) is not str or review["status"] not in {"APPROVED", "REJECTED"} or type(review["reviewer_role"]) is not str or review["reviewer_role"] not in {"AI_PM", "HUMAN"}:
            raise ValueError("invalid review status or role")
        for key in ("reviewer_id", "reason"):
            _string(review[key], f"review.{key}")
        output.append(row)
    return output
