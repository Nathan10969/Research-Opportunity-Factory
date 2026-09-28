"""Validate independent review of proof-only ACL metadata witnesses."""

from __future__ import annotations

import hashlib
import json
import re


_PAGE1_KEYS = {
    "item_id", "pdf_sha256", "page1_text_sha256", "page1_text",
    "pdftotext_version", "witness_author_id",
}
_REVIEW_KEYS = {
    "item_id", "proof_row_sha256", "pdf_sha256", "page1_text_sha256",
    "decision", "reviewer_id", "reviewed_at_utc", "reason",
    "native_pdf_identity_verified", "field_verdicts",
}
_FIELD_NAMES = ("title", "ordered_authors", "venue", "year")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256(encoded)


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _require_nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def validate_admission(proof: dict, page1: dict, review: dict) -> dict:
    """Project a proof-only witness only after exact, independent review."""
    if not isinstance(proof, dict) or not isinstance(page1, dict) or not isinstance(review, dict):
        raise ValueError("proof, page1, and review must be JSON objects")
    if set(page1) != _PAGE1_KEYS:
        raise ValueError("page1 witness does not match the exact required schema")
    if set(review) != _REVIEW_KEYS:
        raise ValueError("review does not match the exact required schema")

    item_id = _require_nonempty(proof.get("item_id"), "proof item_id")
    if not re.fullmatch(r"20\d{2}\.findings-acl\.\d+", item_id):
        raise ValueError("proof item_id is not a canonical ACL Findings ID")
    if page1["item_id"] != item_id or review["item_id"] != item_id:
        raise ValueError("proof, page1, and review item IDs must match exactly")
    if proof.get("current_expected_title") != "bib" or proof.get("current_source_title") != "bib":
        raise ValueError("proof is outside the exact bib-title repair scope")
    if proof.get("outcome") != "UNIQUE_WITNESS" or proof.get("overlay_is_approval") is not False:
        raise ValueError("proof is not a unique proof-only witness")
    if proof.get("source_status_mutation") is not False:
        raise ValueError("proof must certify that source status was not mutated")

    witness = proof.get("official_html_witness")
    if not isinstance(witness, dict) or witness.get("outcome") != "UNIQUE_WITNESS":
        raise ValueError("proof has no unique official HTML metadata witness")
    title = _require_nonempty(witness.get("title"), "official witness title")
    authors = witness.get("authors")
    if not isinstance(authors, list) or not authors or any(not isinstance(author, str) or not author.strip() for author in authors):
        raise ValueError("official witness ordered authors must be a nonempty list of nonempty strings")
    section = witness.get("publication_section_witness")
    if not isinstance(section, dict):
        raise ValueError("official witness has no publication section witness")
    venue = _require_nonempty(section.get("text"), "official witness venue")
    year = section.get("year")
    if not isinstance(year, int) or isinstance(year, bool):
        raise ValueError("official witness year must be an integer")

    pdf_sha256 = _require_sha256(proof.get("pdf_sha256"), "proof PDF hash")
    _require_nonempty(proof.get("pdf_path"), "proof PDF path")
    _require_nonempty(proof.get("queue_path"), "proof queue path")
    _require_sha256(proof.get("queue_file_sha256"), "proof queue file hash")
    queue_line_number = proof.get("queue_line_number")
    if not isinstance(queue_line_number, int) or isinstance(queue_line_number, bool) or queue_line_number < 1:
        raise ValueError("proof queue line number must be a positive one-based integer")
    _require_sha256(proof.get("queue_raw_line_sha256"), "proof queue raw-line hash")
    if _require_sha256(page1["pdf_sha256"], "page1 PDF hash") != pdf_sha256:
        raise ValueError("proof and page1 PDF hashes differ")
    if _require_sha256(review["pdf_sha256"], "review PDF hash") != pdf_sha256:
        raise ValueError("review PDF hash differs from proof")
    proof_sha256 = _canonical_json_sha256(proof)
    if _require_sha256(review["proof_row_sha256"], "review proof-row hash") != proof_sha256:
        raise ValueError("review proof-row hash does not match canonical proof JSON")

    page1_text = _require_nonempty(page1["page1_text"], "page1 text")
    page1_text_sha256 = _sha256(page1_text.encode("utf-8"))
    if _require_sha256(page1["page1_text_sha256"], "page1 text hash") != page1_text_sha256:
        raise ValueError("page1 text hash does not match exact UTF-8 text bytes")
    if _require_sha256(review["page1_text_sha256"], "review page1 text hash") != page1_text_sha256:
        raise ValueError("review page1 text hash differs from the page1 witness")
    _require_nonempty(page1["pdftotext_version"], "pdftotext version")
    witness_author_id = _require_nonempty(page1["witness_author_id"], "page1 witness author ID")

    reviewer_id = _require_nonempty(review["reviewer_id"], "reviewer ID")
    if reviewer_id == witness_author_id:
        raise ValueError("reviewer must differ from the page1 witness author")
    _require_nonempty(review["reviewed_at_utc"], "reviewed_at_utc")
    reason = _require_nonempty(review["reason"], "review reason")
    verdicts = review["field_verdicts"]
    if not isinstance(verdicts, dict) or set(verdicts) != set(_FIELD_NAMES):
        raise ValueError("field_verdicts must contain exactly title, ordered_authors, venue, and year")
    if any(not isinstance(value, str) or not value.strip() for value in verdicts.values()):
        raise ValueError("every field verdict must be a nonempty string")

    decision = review["decision"]
    if decision not in {"SUPPORTED", "METADATA_HOLD"}:
        raise ValueError("unsupported metadata review decision")
    if decision == "SUPPORTED":
        if review["native_pdf_identity_verified"] is not True:
            raise ValueError("SUPPORTED requires verified native PDF identity")
        if any(verdicts[name] != "PASS" for name in _FIELD_NAMES):
            raise ValueError("SUPPORTED requires PASS for every metadata field")
        status = "SUPPORTED"
        hold_reason = None
    else:
        status = "METADATA_HOLD"
        hold_reason = reason

    fields = {
        "title": {
            "before": proof["current_expected_title"],
            "proposed": title,
            "approved": title if decision == "SUPPORTED" else None,
            "disposition": "APPROVED" if decision == "SUPPORTED" else "HOLD",
        },
        "ordered_authors": {
            "before": None,
            "proposed": authors,
            "approved": authors if decision == "SUPPORTED" else None,
            "disposition": "APPROVED" if decision == "SUPPORTED" else "HOLD",
        },
        "venue": {
            "before": None,
            "proposed": venue,
            "approved": venue if decision == "SUPPORTED" else None,
            "disposition": "APPROVED" if decision == "SUPPORTED" else "HOLD",
        },
        "year": {
            "before": None,
            "proposed": year,
            "approved": year if decision == "SUPPORTED" else None,
            "disposition": "APPROVED" if decision == "SUPPORTED" else "HOLD",
        },
    }
    result = {
        "queue_path": proof["queue_path"],
        "queue_file_sha256": proof["queue_file_sha256"],
        "queue_line_number": proof["queue_line_number"],
        "queue_raw_line_sha256": proof["queue_raw_line_sha256"],
        "item_id": item_id,
        "pdf_sha256": pdf_sha256,
        "proof_row_sha256": proof_sha256,
        "review_sha256": _canonical_json_sha256(review),
        "fields": fields,
        "status": status,
        "source_admission_approved": False,
        "human_approved": False,
        "graph_ingested": False,
    }
    if hold_reason is not None:
        result["hold_reason"] = hold_reason
    return result
