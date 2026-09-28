import copy
import hashlib
import json

import pytest

from idea_factory.acl_task4_evidence_adapter import (
    load_task4_page1_witnesses,
    normalize_task4_page1_witness,
)


ITEM_ID = "2026.findings-acl.1077"
PDF_SHA256 = "a" * 64
TEXT = "Exact captured page-1 text – café\r\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record():
    text_bytes = TEXT.encode("utf-8")
    return {
        "schema_version": "acl_metadata_page1_witness_input.v1",
        "item_id": ITEM_ID,
        "pdf": {"path": "untrusted-filename.pdf", "sha256": PDF_SHA256},
        "pdf_page1": {
            "text": TEXT,
            "text_sha256": _sha(text_bytes),
            "text_utf8_bytes": len(text_bytes),
            "pdftotext_version": "pdftotext version 24.08.0",
        },
        "witness_author_id": "explicit-witness-author",
    }


def test_maps_only_documented_nested_task4_fields_and_recomputes_exact_utf8_hash():
    normalized, hold_reason = normalize_task4_page1_witness(
        _record(), expected_item_id=ITEM_ID, expected_pdf_sha256=PDF_SHA256
    )

    assert hold_reason is None
    assert normalized == {
        "item_id": ITEM_ID,
        "pdf_sha256": PDF_SHA256,
        "page1_text_sha256": _sha(TEXT.encode("utf-8")),
        "page1_text": TEXT,
        "pdftotext_version": "pdftotext version 24.08.0",
        "witness_author_id": "explicit-witness-author",
    }


@pytest.mark.parametrize("mutation", ["wrong_id", "wrong_pdf", "wrong_text_hash", "conflicting_byte_count"])
def test_contradictory_identity_or_text_hash_is_rejected(mutation):
    record = _record()
    if mutation == "wrong_id":
        record["item_id"] = "2026.findings-acl.1105"
    elif mutation == "wrong_pdf":
        record["pdf"]["sha256"] = "b" * 64
    elif mutation == "wrong_text_hash":
        record["pdf_page1"]["text_sha256"] = "0" * 64
    else:
        record["pdf_page1"]["text_utf8_bytes"] += 1

    with pytest.raises(ValueError):
        normalize_task4_page1_witness(
            record, expected_item_id=ITEM_ID, expected_pdf_sha256=PDF_SHA256
        )


@pytest.mark.parametrize(
    "mutation, reason",
    [
        ("missing_text", "PAGE1_TEXT_MISSING"),
        ("missing_witness_author", "PAGE1_WITNESS_AUTHOR_MISSING"),
        ("missing_extractor_version", "PAGE1_EXTRACTOR_VERSION_MISSING"),
        ("missing_pdf_hash", "PAGE1_PDF_HASH_MISSING"),
        ("missing_text_hash", "PAGE1_TEXT_HASH_MISSING"),
        ("missing_item_id", "PAGE1_ITEM_ID_MISSING"),
        ("unsupported_schema", "PAGE1_SCHEMA_UNSUPPORTED"),
    ],
)
def test_missing_or_unsupported_per_id_evidence_returns_reasoned_hold(mutation, reason):
    record = _record()
    if mutation == "missing_text":
        del record["pdf_page1"]["text"]
    elif mutation == "missing_witness_author":
        del record["witness_author_id"]
    elif mutation == "missing_extractor_version":
        del record["pdf_page1"]["pdftotext_version"]
    elif mutation == "missing_pdf_hash":
        del record["pdf"]["sha256"]
    elif mutation == "missing_text_hash":
        del record["pdf_page1"]["text_sha256"]
    elif mutation == "missing_item_id":
        del record["item_id"]
    else:
        record["schema_version"] = "acl_metadata_page1_witness_input.v0"

    normalized, hold_reason = normalize_task4_page1_witness(
        record, expected_item_id=ITEM_ID, expected_pdf_sha256=PDF_SHA256
    )
    assert normalized is None
    assert hold_reason == reason


def test_jsonl_loader_rejects_duplicate_keys_and_duplicate_item_ids():
    duplicate_key = (
        b'{"schema_version":"acl_metadata_page1_witness_input.v1",'
        b'"item_id":"2026.findings-acl.1077","item_id":"2026.findings-acl.1105"}\n'
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_task4_page1_witnesses(duplicate_key)

    record = _record()
    repeated = json.dumps(record, separators=(",", ":")).encode("utf-8")
    with pytest.raises(ValueError, match="duplicate .*item_id"):
        load_task4_page1_witnesses(repeated + b"\n" + repeated + b"\n")
