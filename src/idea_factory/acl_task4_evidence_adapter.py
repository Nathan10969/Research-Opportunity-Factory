"""Normalize only the inventoried Task 4 nested ACL page-1 witness fields.

Task 4 schema v1 mapping (all JSON paths are rooted at one JSONL record):
``item_id <- $.item_id``; ``pdf_sha256 <- $.pdf.sha256``;
``page1_text <- $.pdf_page1.text``; ``page1_text_sha256`` is recomputed from
the exact UTF-8 encoding of that text and checked against
``$.pdf_page1.text_sha256``; ``pdftotext_version <-
$.pdf_page1.pdftotext_version``; and ``witness_author_id <-
$.witness_author_id``. Missing values produce a reasoned HOLD. Contradictory
IDs, PDF pins, or declared text hashes are rejected. No value is inferred from
paths, receipts, reviewer decisions, neighboring rows, or defaults.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


TASK4_PAGE1_SCHEMA = "acl_metadata_page1_witness_input.v1"
ADAPTER_PROTOCOL_VERSION = "acl_task4_page1_normalization.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_task4_page1_witnesses(data: bytes) -> list[dict[str, Any]]:
    """Strictly decode unique JSONL records; never let JSON key shadowing win."""
    try:
        decoded = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Task 4 page1 JSONL must be strict UTF-8: {exc}") from exc
    if decoded.startswith("\ufeff") or any(char in decoded for char in "\v\f\x1c\x1d\x1e\x85\u2028\u2029"):
        raise ValueError("Task 4 page1 JSONL must use UTF-8 LF/CRLF line endings")
    if "\r" in decoded.replace("\r\n", ""):
        raise ValueError("Task 4 page1 JSONL contains a bare carriage return")
    records: list[dict[str, Any]] = []
    seen_item_ids: set[str] = set()
    for line_number, line in enumerate(decoded.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank Task 4 page1 JSONL row at line {line_number}")
        try:
            row = json.loads(line, object_pairs_hook=_object_without_duplicate_keys)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid Task 4 page1 JSONL row at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Task 4 page1 line {line_number} must be a JSON object")
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            # Without a stable key there is no safe way to choose which selected
            # item should receive a visible HOLD row; assigning one would guess.
            raise ValueError(
                f"Task 4 page1 line {line_number} has no explicit item_id; cannot safely assign evidence without guessing"
            )
        if item_id in seen_item_ids:
            raise ValueError(f"duplicate Task 4 page1 item_id: {item_id}")
        seen_item_ids.add(item_id)
        records.append(row)
    return records


def normalize_task4_page1_witness(
    record: dict[str, Any],
    *,
    expected_item_id: str,
    expected_pdf_sha256: str,
) -> tuple[dict[str, str] | None, str | None]:
    """Return an exact Task 2 page-1 witness or ``(None, HOLD reason)``.

    Contradictory identity/content pins raise ``ValueError`` so the caller can
    fail the entire run before writing either sidecar. Genuinely absent values
    remain item-visible HOLDs.
    """
    if not isinstance(record, dict):
        return None, "PAGE1_RECORD_INVALID"
    if record.get("schema_version") != TASK4_PAGE1_SCHEMA:
        return None, "PAGE1_SCHEMA_UNSUPPORTED"

    item_id = record.get("item_id")
    if not isinstance(item_id, str) or not item_id:
        return None, "PAGE1_ITEM_ID_MISSING"
    if item_id != expected_item_id:
        raise ValueError(f"Task 4 page1 item_id {item_id!r} does not match parent item {expected_item_id!r}")

    pdf = record.get("pdf")
    pdf_sha256 = pdf.get("sha256") if isinstance(pdf, dict) else None
    if pdf_sha256 is None:
        return None, "PAGE1_PDF_HASH_MISSING"
    if not isinstance(pdf_sha256, str) or not _SHA256_RE.fullmatch(pdf_sha256):
        raise ValueError(f"{item_id}: Task 4 page1 PDF hash is malformed")
    if pdf_sha256 != expected_pdf_sha256:
        raise ValueError(f"{item_id}: Task 4 page1 PDF hash does not match parent proof")

    page1 = record.get("pdf_page1")
    if not isinstance(page1, dict):
        return None, "PAGE1_TEXT_MISSING"
    text = page1.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "PAGE1_TEXT_MISSING"
    text_bytes = text.encode("utf-8")
    computed_text_sha256 = _sha256(text_bytes)
    declared_text_sha256 = page1.get("text_sha256")
    if declared_text_sha256 is None:
        return None, "PAGE1_TEXT_HASH_MISSING"
    if not isinstance(declared_text_sha256, str) or not _SHA256_RE.fullmatch(declared_text_sha256):
        raise ValueError(f"{item_id}: Task 4 page1 text hash is malformed")
    if declared_text_sha256 != computed_text_sha256:
        raise ValueError(f"{item_id}: Task 4 page1 text hash does not match exact UTF-8 text")
    if "text_utf8_bytes" in page1:
        byte_count = page1["text_utf8_bytes"]
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count != len(text_bytes):
            raise ValueError(f"{item_id}: conflicting Task 4 page1 UTF-8 byte count")

    extractor_version = page1.get("pdftotext_version")
    if not isinstance(extractor_version, str) or not extractor_version.strip():
        return None, "PAGE1_EXTRACTOR_VERSION_MISSING"
    witness_author_id = record.get("witness_author_id")
    if not isinstance(witness_author_id, str) or not witness_author_id.strip():
        return None, "PAGE1_WITNESS_AUTHOR_MISSING"

    return {
        "item_id": item_id,
        "pdf_sha256": pdf_sha256,
        "page1_text_sha256": computed_text_sha256,
        "page1_text": text,
        "pdftotext_version": extractor_version,
        "witness_author_id": witness_author_id,
    }, None
