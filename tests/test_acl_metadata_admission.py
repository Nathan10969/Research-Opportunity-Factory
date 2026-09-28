import hashlib
import json

import pytest

from idea_factory.acl_metadata_admission import validate_admission


def _canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fixture():
    item_id = "2026.findings-acl.1077"
    pdf_sha = "a" * 64
    page1_text = "ATAAT: Adaptive Threat-Aware Tuning\nAlice Smith and Bob Jones\nFindings of ACL 2026"
    page1 = {
        "item_id": item_id,
        "pdf_sha256": pdf_sha,
        "page1_text_sha256": hashlib.sha256(page1_text.encode("utf-8")).hexdigest(),
        "page1_text": page1_text,
        "pdftotext_version": "pdftotext version 24.02.0",
        "witness_author_id": "native-pdf-reviewer-a",
    }
    proof = {
        "item_id": item_id,
        "queue_path": "F:/acl/workers/acl-01/queue.jsonl",
        "queue_file_sha256": "b" * 64,
        "queue_line_number": 4,
        "queue_raw_line_sha256": "c" * 64,
        "manifest_path": "F:/acl/manifest.jsonl",
        "manifest_file_sha256": "d" * 64,
        "manifest_line_number": 88,
        "manifest_raw_line_sha256": "e" * 64,
        "official_html_path": "F:/acl/anthology.html",
        "official_html_sha256": "f" * 64,
        "pdf_path": f"F:/acl/papers/{item_id}__bib.pdf",
        "pdf_sha256": pdf_sha,
        "cache_receipt_path": f"F:/acl/cache/{item_id}.receipt.json",
        "cache_receipt_sha256": "1" * 64,
        "cache_text_path": f"F:/acl/cache/{item_id}.txt",
        "cache_text_sha256": "2" * 64,
        "current_expected_title": "bib",
        "current_source_title": "bib",
        "official_html_witness": {
            "outcome": "UNIQUE_WITNESS",
            "title": "ATAAT: Adaptive Threat-Aware Tuning",
            "authors": ["Alice Smith", "Bob Jones"],
            "publication_section_witness": {
                "text": "Findings of the Association for Computational Linguistics: ACL 2026",
                "year": 2026,
            },
        },
        "outcome": "UNIQUE_WITNESS",
        "overlay_is_approval": False,
        "source_status_mutation": False,
    }
    review = {
        "item_id": item_id,
        "proof_row_sha256": _canonical_sha256(proof),
        "pdf_sha256": pdf_sha,
        "page1_text_sha256": page1["page1_text_sha256"],
        "decision": "SUPPORTED",
        "reviewer_id": "independent-reviewer-b",
        "reviewed_at_utc": "2026-09-28T01:02:03Z",
        "reason": "Native first page independently agrees with the official metadata witness.",
        "native_pdf_identity_verified": True,
        "field_verdicts": {
            "title": "PASS",
            "ordered_authors": "PASS",
            "venue": "PASS",
            "year": "PASS",
        },
    }
    return proof, page1, review


def test_supported_admission_projects_only_reviewed_metadata_and_preserves_queue_identity():
    proof, page1, review = _fixture()

    result = validate_admission(proof, page1, review)

    assert result["queue_path"] == proof["queue_path"]
    assert result["queue_file_sha256"] == proof["queue_file_sha256"]
    assert result["queue_line_number"] == proof["queue_line_number"]
    assert result["queue_raw_line_sha256"] == proof["queue_raw_line_sha256"]
    assert result["item_id"] == proof["item_id"]
    assert result["pdf_sha256"] == proof["pdf_sha256"]
    assert result["proof_row_sha256"] == _canonical_sha256(proof)
    assert result["review_sha256"] == _canonical_sha256(review)
    assert set(result["fields"]) == {"title", "ordered_authors", "venue", "year"}
    assert result["fields"]["title"] == {
        "before": "bib",
        "proposed": "ATAAT: Adaptive Threat-Aware Tuning",
        "approved": "ATAAT: Adaptive Threat-Aware Tuning",
        "disposition": "APPROVED",
    }
    assert result["fields"]["ordered_authors"]["before"] is None
    assert result["fields"]["ordered_authors"]["proposed"] == ["Alice Smith", "Bob Jones"]
    assert result["fields"]["ordered_authors"]["approved"] == ["Alice Smith", "Bob Jones"]
    assert result["fields"]["venue"]["proposed"] == "Findings of the Association for Computational Linguistics: ACL 2026"
    assert result["fields"]["year"]["proposed"] == 2026
    assert "pages" not in result["fields"]
    assert result["status"] == "SUPPORTED"
    assert result["source_admission_approved"] is False
    assert result["human_approved"] is False
    assert result["graph_ingested"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "item_id",
        "proof_sha",
        "pdf_sha",
        "page1_text_sha",
        "empty_reviewer",
        "same_reviewer_as_author",
        "native_pdf_unverified",
        "title_disagreement",
        "ordered_author_disagreement",
        "missing_venue",
        "missing_year",
    ],
)
def test_supported_admission_fails_closed_on_identity_review_or_witness_disagreement(mutation):
    proof, page1, review = _fixture()
    if mutation == "item_id":
        review["item_id"] = "2026.findings-acl.1078"
    elif mutation == "proof_sha":
        review["proof_row_sha256"] = "0" * 64
    elif mutation == "pdf_sha":
        review["pdf_sha256"] = "0" * 64
    elif mutation == "page1_text_sha":
        review["page1_text_sha256"] = "0" * 64
    elif mutation == "empty_reviewer":
        review["reviewer_id"] = "  "
    elif mutation == "same_reviewer_as_author":
        review["reviewer_id"] = page1["witness_author_id"]
    elif mutation == "native_pdf_unverified":
        review["native_pdf_identity_verified"] = False
    elif mutation == "title_disagreement":
        review["field_verdicts"]["title"] = "FAIL"
    elif mutation == "ordered_author_disagreement":
        review["field_verdicts"]["ordered_authors"] = "FAIL"
    elif mutation == "missing_venue":
        proof["official_html_witness"]["publication_section_witness"]["text"] = ""
        review["proof_row_sha256"] = _canonical_sha256(proof)
    elif mutation == "missing_year":
        proof["official_html_witness"]["publication_section_witness"].pop("year")
        review["proof_row_sha256"] = _canonical_sha256(proof)

    with pytest.raises(ValueError):
        validate_admission(proof, page1, review)


def test_explicit_metadata_hold_has_no_approved_replacements():
    proof, page1, review = _fixture()
    review["decision"] = "METADATA_HOLD"
    review["reason"] = "The ordered author sequence is uncertain."
    review["field_verdicts"]["ordered_authors"] = "UNCLEAR"

    result = validate_admission(proof, page1, review)

    assert result["status"] == "METADATA_HOLD"
    assert result["hold_reason"] == review["reason"]
    assert all(field["approved"] is None for field in result["fields"].values())
    assert all(field["disposition"] == "HOLD" for field in result["fields"].values())
    assert result["source_admission_approved"] is False
    assert result["human_approved"] is False
    assert result["graph_ingested"] is False


def test_page1_and_review_reject_non_exact_schemas_and_invalid_decisions():
    proof, page1, review = _fixture()
    page1["page_number"] = 1
    with pytest.raises(ValueError):
        validate_admission(proof, page1, review)

    proof, page1, review = _fixture()
    review["decision"] = "APPROVE"
    with pytest.raises(ValueError):
        validate_admission(proof, page1, review)


def test_missing_or_empty_page1_text_rejects_supported_admission():
    proof, page1, review = _fixture()
    page1["page1_text"] = ""
    page1["page1_text_sha256"] = hashlib.sha256(b"").hexdigest()
    review["page1_text_sha256"] = page1["page1_text_sha256"]
    with pytest.raises(ValueError):
        validate_admission(proof, page1, review)


@pytest.mark.parametrize("field", ["queue_path", "queue_file_sha256", "queue_line_number", "queue_raw_line_sha256"])
def test_queue_identity_fields_must_be_valid_before_projection(field):
    proof, page1, review = _fixture()
    if field == "queue_path":
        proof[field] = "  "
    elif field in {"queue_file_sha256", "queue_raw_line_sha256"}:
        proof[field] = "not-a-sha256"
    else:
        proof[field] = 0
    review["proof_row_sha256"] = _canonical_sha256(proof)

    with pytest.raises(ValueError):
        validate_admission(proof, page1, review)


@pytest.mark.parametrize("pdf_path", [None, "  ", 42])
def test_pdf_path_must_be_present_and_nonempty_before_supported_projection(pdf_path):
    proof, page1, review = _fixture()
    proof["pdf_path"] = pdf_path
    review["proof_row_sha256"] = _canonical_sha256(proof)

    with pytest.raises(ValueError):
        validate_admission(proof, page1, review)


def test_pdf_path_must_exist_in_proof_before_supported_projection():
    proof, page1, review = _fixture()
    proof.pop("pdf_path")
    review["proof_row_sha256"] = _canonical_sha256(proof)

    with pytest.raises(ValueError):
        validate_admission(proof, page1, review)


def test_supported_admission_rejects_section_year_conflicting_with_exact_item_id():
    proof, page1, review = _fixture()
    proof["official_html_witness"]["publication_section_witness"]["year"] = 2025
    review["proof_row_sha256"] = _canonical_sha256(proof)

    with pytest.raises(ValueError):
        validate_admission(proof, page1, review)
