import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from idea_factory.opportunities import scheduled_operators
from idea_factory.landscape import validate_landscape_bundle


def _fixture(tmp_path: Path):
    from test_opportunities import _build_run

    run, _prompt = _build_run(tmp_path, cards_per_note=2)
    return validate_landscape_bundle(run)


def _entry(bundle, kind="PAIR_RELATION"):
    edges = [e for e in bundle.edges if e["dimension"] == "assumptions"]
    anchors = []
    first = edges[0]
    second = next(e for e in edges[1:] if e["source_path"] != first["source_path"])
    for edge in (first, second):
        anchors.append({k: edge[k] for k in ("card_id", "dimension", "facet", "raw_text", "scope_key")})
    for anchor in anchors:
        anchor["source_scope_key"] = anchor.pop("scope_key")
        anchor["role"] = "evidence anchor"
    if kind == "CROSS_FACET":
        other = next(e for e in bundle.edges if e["card_id"] != edges[0]["card_id"] and e["facet"] != edges[0]["facet"] and e["source_path"] != edges[0]["source_path"])
        anchors[1] = {k: other[k] for k in ("card_id", "dimension", "facet", "raw_text", "scope_key")}
        anchors[1]["source_scope_key"] = anchors[1].pop("scope_key")
        anchors[1]["role"] = "cross-facet anchor"
    return {
        "entry_id": "entry-1", "entry_kind": kind, "anchors": anchors,
        "shared_object": "shared object", "interface_relation": "interface relation",
        "compatibility_reason": "compatible reason", "scope_differences": "scope differences",
        "proposed_hypothesis": "proposed hypothesis", "alternative_explanation": "alternative explanation",
        "decisive_test": "decisive test", "novelty_boundary": "novelty boundary",
        "nearest_prior_card_ids": [edges[0]["card_id"]], "operator": scheduled_operators()[0].value,
        "review": {"status": "APPROVED", "reviewer_id": "pm-1", "reviewer_role": "AI_PM", "reason": "review reason"},
    }


def _document(bundle, entries):
    doc = {"schema_version": "idea_factory.reviewed_local_entries.v1", "landscape_hashes": bundle.hashes, "entries": entries}
    doc["artifact_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return doc


def _rehash(doc):
    doc.pop("artifact_sha256", None)
    doc["artifact_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return doc


def test_validates_two_card_pair_relation_against_real_landscape(tmp_path):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    assert validate_local_entries(_document(bundle, [_entry(bundle)]), bundle)[0]["entry_id"] == "entry-1"


def test_validates_cross_facet_entry_and_rejected_entry(tmp_path):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    rejected = copy.deepcopy(_entry(bundle, "CROSS_FACET"))
    rejected["entry_id"] = "entry-2"
    rejected["review"] = {"status": "REJECTED", "reviewer_id": "pm-1", "reviewer_role": "AI_PM", "reason": "not approved"}
    assert [e["entry_id"] for e in validate_local_entries(_document(bundle, [_entry(bundle, "CROSS_FACET"), rejected]), bundle)] == ["entry-1", "entry-2"]


def test_empty_document_is_valid(tmp_path):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    assert validate_local_entries(_document(bundle, []), bundle) == []


@pytest.mark.parametrize("mutate", [
    lambda d: d["entries"][0]["anchors"][0].__setitem__("bad", {}),
    lambda d: d["entries"][0].__setitem__("shared_object", " "),
    lambda d: d["entries"][0]["anchors"][0].__setitem__("raw_text", "stale"),
    lambda d: d["entries"][0].__setitem__("nearest_prior_card_ids", ["missing"]),
])
def test_rejects_tampered_or_unknown_local_entry_fields(tmp_path, mutate):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    doc = _document(bundle, [_entry(bundle)])
    mutate(doc)
    with pytest.raises(ValueError):
        validate_local_entries(doc, bundle)


def test_rejects_duplicate_json_keys(tmp_path):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    text = json.dumps(_document(bundle, []))[:-1] + ', "entries": []}'
    with pytest.raises(ValueError, match="duplicate"):
        validate_local_entries(text, bundle)


def test_rejects_duplicate_entry_id_after_rehash(tmp_path):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    row = _entry(bundle)
    duplicate = copy.deepcopy(row)
    doc = _rehash(_document(bundle, [row, duplicate]))
    with pytest.raises(ValueError, match="duplicate entry_id"):
        validate_local_entries(doc, bundle)


def test_rejects_same_paper_cards_after_rehash(tmp_path):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    edges = [e for e in bundle.edges if e["dimension"] == "assumptions"]
    same = [e for e in edges if e["source_path"] == edges[0]["source_path"]][:2]
    row = _entry(bundle)
    row["anchors"] = [{"card_id": e["card_id"], "dimension": e["dimension"], "facet": e["facet"], "raw_text": e["raw_text"], "source_scope_key": e["scope_key"], "role": "anchor"} for e in same]
    with pytest.raises(ValueError, match="distinct source papers"):
        validate_local_entries(_rehash(_document(bundle, [row])), bundle)


@pytest.mark.parametrize("field,value,error", [
    ("dimension", "wrong", "stale or mismatched"),
    ("facet", "wrong", "stale or mismatched"),
    ("raw_text", "wrong", "stale or mismatched"),
    ("source_scope_key", "wrong", "stale or mismatched"),
])
def test_rejects_each_stale_anchor_binding_after_rehash(tmp_path, field, value, error):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    doc = _document(bundle, [_entry(bundle)])
    doc["entries"][0]["anchors"][0][field] = value
    with pytest.raises(ValueError, match=error):
        validate_local_entries(_rehash(doc), bundle)


@pytest.mark.parametrize("field,value", [("entry_id", " "), ("shared_object", " "), ("reviewer_id", " "), ("reason", " "), ("nearest_prior_card_ids", []), ("nearest_prior_card_ids", [" "]), ("nearest_prior_card_ids", ["x", "x"])])
def test_rejects_blank_explanations_and_review_fields_after_rehash(tmp_path, field, value):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    doc = _document(bundle, [_entry(bundle)])
    target = doc["entries"][0]["review"] if field in {"reviewer_id", "reason"} else doc["entries"][0]
    target[field] = value
    with pytest.raises(ValueError):
        validate_local_entries(_rehash(doc), bundle)


def test_source_path_comparison_is_case_and_separator_insensitive(tmp_path):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    edges = [e for e in bundle.edges if e["dimension"] == "assumptions"]
    first = edges[0]
    second = next(e for e in edges[1:] if e["source_path"] != first["source_path"])
    first_card = bundle.cards[first["card_id"]]
    second_card = bundle.cards[second["card_id"]]
    bundle.cards[second["card_id"]] = second_card.model_copy(update={"paper": second_card.paper.model_copy(update={"source_path": first_card.paper.source_path.replace("/", "\\").upper()})})
    with pytest.raises(ValueError, match="distinct source papers"):
        validate_local_entries(_document(bundle, [_entry(bundle)]), bundle)


@pytest.mark.parametrize("location,field,value", [
    ("doc", "extra", 1), ("entry", "extra", 1), ("anchor", "extra", 1), ("review", "extra", 1),
    ("entry", "nearest_prior_card_ids", ["missing"]),
])
def test_rejects_unknown_or_invalid_fields_after_rehash(tmp_path, location, field, value):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    doc = _document(bundle, [_entry(bundle)])
    target = doc if location == "doc" else doc["entries"][0] if location == "entry" else doc["entries"][0]["anchors"][0] if location == "anchor" else doc["entries"][0]["review"]
    target[field] = value
    with pytest.raises(ValueError):
        validate_local_entries(_rehash(doc), bundle)


@pytest.mark.parametrize("field,value", [("entry_kind", []), ("operator", {}), ("entry_kind", "bad"), ("operator", "bad")])
def test_rejects_invalid_scalar_enums_cleanly(tmp_path, field, value):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    doc = _document(bundle, [_entry(bundle)])
    doc["entries"][0][field] = value
    with pytest.raises(ValueError):
        validate_local_entries(_rehash(doc), bundle)


def test_rejects_pair_mismatch_dimensions_and_scope_fields(tmp_path):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    for field in ("dimension", "facet", "raw_text", "source_scope_key"):
        doc = _document(bundle, [_entry(bundle)])
        doc["entries"][0]["anchors"][1][field] = "different-value"
        with pytest.raises(ValueError):
            validate_local_entries(_rehash(doc), bundle)


@pytest.mark.parametrize("field", ["normalized_text", "cluster_scope_key"])
def test_rejects_pair_relation_semantic_mismatch(tmp_path, field):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    edges = [dict(edge) for edge in bundle.edges]
    target = next(edge for edge in edges if edge["card_id"] == _entry(bundle)["anchors"][1]["card_id"] and edge["dimension"] == "assumptions")
    target[field] = "different-semantic-value"
    altered = replace(bundle, edges=tuple(edges))
    with pytest.raises(ValueError, match="PAIR_RELATION"):
        validate_local_entries(_document(altered, [_entry(bundle)]), altered)


def test_rejects_pair_relation_with_valid_cross_facet_anchors(tmp_path):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    row = _entry(bundle, "CROSS_FACET")
    row["entry_kind"] = "PAIR_RELATION"
    with pytest.raises(ValueError, match="PAIR_RELATION"):
        validate_local_entries(_document(bundle, [row]), bundle)


def test_rejects_cross_facet_same_facet(tmp_path):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    doc = _document(bundle, [_entry(bundle)])
    doc["entries"][0]["entry_kind"] = "CROSS_FACET"
    with pytest.raises(ValueError):
        validate_local_entries(_rehash(doc), bundle)


def test_rejects_duplicate_card_anchor(tmp_path):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    doc = _document(bundle, [_entry(bundle)])
    doc["entries"][0]["anchors"][1] = copy.deepcopy(doc["entries"][0]["anchors"][0])
    with pytest.raises(ValueError, match="distinct cards"):
        validate_local_entries(_rehash(doc), bundle)


@pytest.mark.parametrize("field", ["status", "reviewer_role"])
@pytest.mark.parametrize("value", [[], {}, 0, "bad"])
def test_rejects_non_string_review_status_and_role(tmp_path, field, value):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    doc = _document(bundle, [_entry(bundle)])
    doc["entries"][0]["review"][field] = value
    with pytest.raises(ValueError):
        validate_local_entries(_rehash(doc), bundle)


def test_rejects_stale_landscape_hash_after_rehash(tmp_path):
    from idea_factory.local_entries import validate_local_entries
    bundle = _fixture(tmp_path)
    doc = _document(bundle, [])
    doc["landscape_hashes"] = copy.deepcopy(doc["landscape_hashes"])
    doc["landscape_hashes"]["tampered"] = "x"
    with pytest.raises(ValueError, match="landscape hash"):
        validate_local_entries(_rehash(doc), bundle)
