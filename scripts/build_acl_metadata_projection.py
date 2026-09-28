#!/usr/bin/env python3
"""Build a fail-closed, reviewed ACL metadata projection in a new run directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from idea_factory import acl_task4_evidence_adapter
from idea_factory.acl_task4_evidence_adapter import (
    ADAPTER_PROTOCOL_VERSION,
    load_task4_page1_witnesses,
    normalize_task4_page1_witness,
)
from idea_factory.acl_metadata_admission import validate_admission


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ACL_ITEM_ID_RE = re.compile(r"2026\.findings-acl\.\d+\Z")
_FIELDS = ("title", "ordered_authors", "venue", "year")
_EXACT_REVIEW_KEYS = {
    "item_id", "proof_row_sha256", "pdf_sha256", "page1_text_sha256",
    "decision", "reviewer_id", "reviewed_at_utc", "reason",
    "native_pdf_identity_verified", "field_verdicts",
}
_RECEIPT_SCHEMA = "acl670_metadata_proof_only_run_receipt.v1"
_REPARSE_POINT = 0x0400
_PARENT_RECEIPT_RELATIVE_PATH = Path(
    "runs/parallel24-20260927-1340/engineering/ACL670_WITNESS_FINAL_RUN_RECEIPT.json"
)
_PARENT_RECEIPT_SHA256 = "2cf9982c0f39c7d82098baaf7ed41c759b83f45fd46b61fd752c899a0a669205"
_PARENT_PROOF_RELATIVE_PATH = Path(
    "runs/parallel24-20260927-1340/engineering/acl670-metadata-witness-proofonly-v5-20260927/acl_metadata_witness.jsonl"
)
_PARENT_PROOF_SHA256 = "375b75a511042b5e617a69649814d4c599a9b5d6a411d8f076c13ac48796cb8a"
_FROZEN_SUBSET_IDS = (
    "2026.findings-acl.1077", "2026.findings-acl.1105", "2026.findings-acl.1174",
    "2026.findings-acl.1266", "2026.findings-acl.1320", "2026.findings-acl.135",
    "2026.findings-acl.1371", "2026.findings-acl.1388", "2026.findings-acl.1412",
    "2026.findings-acl.1530",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha(value: object) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha(data)


def _digest_argument(value: str, label: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _inside(path: Path, root: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the explicit corpus root") from exc
    _reject_reparse_components(absolute)
    return absolute


def _reject_reparse_components(path: Path) -> None:
    """Reject symlink/junction/reparse components, including existing ancestors."""
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        attrs = getattr(info, "st_file_attributes", 0)
        if current.is_symlink() or attrs & _REPARSE_POINT:
            raise ValueError(f"symlink or reparse path component is forbidden: {current}")


def _canonical_path(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path.resolve(strict=True))))


def _read_pinned(path: Path, expected_sha: str, label: str, root: Path, snapshots: dict[Path, str]) -> bytes:
    _digest_argument(expected_sha, f"{label} hash")
    safe_path = _inside(path, root, label)
    try:
        data = safe_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {safe_path}: {exc}") from exc
    actual = _sha(data)
    if actual != expected_sha:
        raise ValueError(f"{label} hash mismatch: expected {expected_sha}, found {actual}")
    previous = snapshots.setdefault(safe_path, expected_sha)
    if previous != expected_sha:
        raise ValueError(f"conflicting input pins for {safe_path}")
    return data


def _json_object_lines(data: bytes, label: str) -> list[dict]:
    rows: list[dict] = []
    for number, raw in enumerate(data.splitlines(), 1):
        if not raw.strip():
            raise ValueError(f"blank JSONL row in {label} at line {number}")
        try:
            row = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid JSONL row in {label} at line {number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {number} must be a JSON object")
        rows.append(row)
    return rows


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _unique_by_item(rows: list[dict], label: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{label} row has no nonempty item_id")
        if item_id in result:
            raise ValueError(f"duplicate {label} item_id: {item_id}")
        result[item_id] = row
    return result


def _physical_line(data: bytes, line_number: object, label: str) -> bytes:
    if not isinstance(line_number, int) or isinstance(line_number, bool) or line_number < 1:
        raise ValueError(f"{label} line number must be a positive one-based integer")
    lines = data.splitlines(keepends=True)
    if line_number > len(lines):
        raise ValueError(f"{label} physical line {line_number} is absent")
    raw = lines[line_number - 1]
    if not raw.endswith(b"\r\n"):
        raise ValueError(f"{label} physical line {line_number} does not use the frozen CRLF convention")
    return raw


def _json_physical_line(data: bytes, line_number: int, label: str) -> dict:
    raw = _physical_line(data, line_number, label)
    try:
        value = json.loads(raw[:-2].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON at physical line {line_number}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} physical line {line_number} must be a JSON object")
    return value


def _verify_proof_source_rows(proof_rows: list[dict], root: Path, snapshots: dict[Path, str]) -> None:
    file_cache: dict[Path, bytes] = {}
    for proof in proof_rows:
        item_id = proof.get("item_id")
        if not isinstance(item_id, str) or not _ACL_ITEM_ID_RE.fullmatch(item_id):
            raise ValueError(f"proof row has a noncanonical 2026 ACL Findings item_id: {item_id!r}")
        if proof.get("current_expected_title") != "bib" or proof.get("current_source_title") != "bib":
            raise ValueError(f"{item_id}: proof row is outside the exact bib-title repair scope")
        if proof.get("outcome") != "UNIQUE_WITNESS" or proof.get("overlay_is_approval") is not False:
            raise ValueError(f"{item_id}: proof row is not a unique proof-only witness")
        if proof.get("source_status_mutation") is not False:
            raise ValueError(f"{item_id}: proof row does not certify immutable source status")
        witness = proof.get("official_html_witness")
        if not isinstance(witness, dict) or witness.get("outcome") != "UNIQUE_WITNESS":
            raise ValueError(f"{item_id}: proof row lacks a unique official HTML witness")
        if not isinstance(witness.get("title"), str) or not witness["title"].strip() or witness["title"].casefold() == "bib":
            raise ValueError(f"{item_id}: official HTML witness has no admissible title")
        authors = witness.get("authors")
        if not isinstance(authors, list) or not authors or any(not isinstance(author, str) or not author.strip() for author in authors):
            raise ValueError(f"{item_id}: official HTML witness has no ordered author list")
        section = witness.get("publication_section_witness")
        if not isinstance(section, dict) or not isinstance(section.get("text"), str) or not section["text"].strip():
            raise ValueError(f"{item_id}: official HTML witness has no publication section")
        if section.get("year") != 2026 or isinstance(section.get("year"), bool):
            raise ValueError(f"{item_id}: official HTML witness year is not exactly 2026")
        for path_key, hash_key, label in (
            ("queue_path", "queue_file_sha256", "queue"),
            ("manifest_path", "manifest_file_sha256", "manifest"),
            ("official_html_path", "official_html_sha256", "official HTML"),
            ("pdf_path", "pdf_sha256", "PDF"),
            ("cache_receipt_path", "cache_receipt_sha256", "cache receipt"),
            ("cache_text_path", "cache_text_sha256", "cache text"),
        ):
            raw_path = proof.get(path_key)
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"{item_id}: missing {label} path")
            path = _inside(Path(raw_path), root, f"{item_id} {label}")
            expected = proof.get(hash_key)
            if not isinstance(expected, str):
                raise ValueError(f"{item_id}: missing {label} hash")
            if path not in file_cache:
                file_cache[path] = _read_pinned(path, expected, f"{item_id} {label}", root, snapshots)
            elif _sha(file_cache[path]) != expected:
                raise ValueError(f"{item_id}: inconsistent {label} hash pins")

        queue_path = Path(proof["queue_path"])
        manifest_path = Path(proof["manifest_path"])
        queue_data = file_cache[_inside(queue_path, root, f"{item_id} queue")]
        manifest_data = file_cache[_inside(manifest_path, root, f"{item_id} manifest")]
        queue_raw = _physical_line(queue_data, proof.get("queue_line_number"), f"{item_id} queue")
        manifest_raw = _physical_line(manifest_data, proof.get("manifest_line_number"), f"{item_id} manifest")
        if _sha(queue_raw) != proof.get("queue_raw_line_sha256"):
            raise ValueError(f"{item_id}: queue physical row hash mismatch")
        if _sha(manifest_raw) != proof.get("manifest_raw_line_sha256"):
            raise ValueError(f"{item_id}: manifest physical row hash mismatch")
        queue_row = _json_physical_line(queue_data, proof["queue_line_number"], f"{item_id} queue")
        manifest_row = _json_physical_line(manifest_data, proof["manifest_line_number"], f"{item_id} manifest")
        if queue_row.get("item_id") != item_id:
            raise ValueError(f"{item_id}: queue physical row identifies a different item")
        manifest_id = manifest_row.get("anthology_id", manifest_row.get("item_id"))
        if manifest_id != item_id:
            raise ValueError(f"{item_id}: manifest physical row identifies a different item")
        if queue_row.get("expected_title") != "bib" or manifest_row.get("title") != "bib":
            raise ValueError(f"{item_id}: frozen source row is outside bib-title scope")


def _parse_selection(data: bytes) -> tuple[str, ...]:
    try:
        decoded = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"selection list must be strict UTF-8: {exc}") from exc
    if decoded.startswith("\ufeff") or any(char in decoded for char in "\v\f\x1c\x1d\x1e\x85\u2028\u2029"):
        raise ValueError("selection list must use plain UTF-8 with one ID per LF/CRLF line")
    if "\r" in decoded.replace("\r\n", ""):
        raise ValueError("selection list contains a bare carriage return")
    ids = decoded.splitlines()
    if len(ids) != len(_FROZEN_SUBSET_IDS) or any(not item_id for item_id in ids):
        raise ValueError("selection list must contain exactly ten nonblank IDs, one per line")
    for item_id in ids:
        if not _ACL_ITEM_ID_RE.fullmatch(item_id) or str(int(item_id.rsplit(".", 1)[1])) != item_id.rsplit(".", 1)[1]:
            raise ValueError(f"selection ID is not a canonical ACL Findings ID: {item_id!r}")
        if "TEST_ONLY" in item_id:
            raise ValueError("TEST_ONLY IDs are forbidden in the production selection list")
    if len(set(ids)) != len(ids):
        raise ValueError("selection list contains duplicate IDs")
    if set(ids) != set(_FROZEN_SUBSET_IDS):
        raise ValueError("selection list IDs do not equal the frozen Task 4 ten-ID set")
    return tuple(ids)


def _hold_fields(proof: dict) -> dict:
    witness = proof.get("official_html_witness") or {}
    section = witness.get("publication_section_witness") or {}
    proposed = {
        "title": witness.get("title"),
        "ordered_authors": witness.get("authors"),
        "venue": section.get("text"),
        "year": section.get("year"),
    }
    return {
        name: {
            "before": proof.get("current_expected_title") if name == "title" else None,
            "proposed": proposed[name],
            "approved": None,
            "disposition": "HOLD",
        }
        for name in _FIELDS
    }


def _hold_row(proof: dict, reason: str, *, page1: dict | None = None, review: dict | None = None) -> dict:
    row = {
        "queue_path": proof["queue_path"],
        "queue_file_sha256": proof["queue_file_sha256"],
        "queue_line_number": proof["queue_line_number"],
        "queue_raw_line_sha256": proof["queue_raw_line_sha256"],
        "item_id": proof["item_id"],
        "pdf_sha256": proof["pdf_sha256"],
        "proof_row_sha256": _canonical_sha(proof),
        "page1_witness_sha256": _canonical_sha(page1) if page1 is not None else None,
        "review_sha256": _canonical_sha(review) if review is not None else None,
        "fields": _hold_fields(proof),
        "status": "METADATA_HOLD",
        "hold_reason": reason,
        "source_admission_approved": False,
        "human_approved": False,
        "graph_ingested": False,
    }
    return row


def _diagnostic_row(
    proof: dict,
    page1: dict | None,
    review: dict | None,
    projection: dict,
    *,
    page1_raw_sha256: str | None = None,
) -> dict:
    return {
        "item_id": proof["item_id"],
        "proof_row_sha256": _canonical_sha(proof),
        "page1_witness_sha256": _canonical_sha(page1) if page1 is not None else None,
        "page1_witness_raw_sha256": page1_raw_sha256,
        "page1_witness": page1,
        "review_sha256": _canonical_sha(review) if review is not None else None,
        "review_reference": None if review is None else {
            "reviewer_id": review.get("reviewer_id"),
            "decision": review.get("decision"),
            "reviewed_at_utc": review.get("reviewed_at_utc"),
        },
        "fields": projection["fields"],
        "status": projection["status"],
        "hold_reason": projection.get("hold_reason"),
        "source_admission_approved": False,
        "human_approved": False,
        "graph_ingested": False,
    }


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )


def _capture_snapshot_recheck(snapshots: dict[Path, str]) -> None:
    for path, expected in snapshots.items():
        _reject_reparse_components(path)
        try:
            actual = _sha(path.read_bytes())
        except OSError as exc:
            raise ValueError(f"pinned input disappeared before output: {path}: {exc}") from exc
        if actual != expected:
            raise ValueError(f"pinned input drift before output: {path}")


def _resolve_pdftotext(snapshots: dict[Path, str]) -> tuple[Path, str, str]:
    found = shutil.which("pdftotext")
    if not found:
        raise ValueError("pdftotext executable is required to verify native page-1 evidence")
    candidate = Path(found)
    _reject_reparse_components(candidate)
    try:
        executable = candidate.resolve(strict=True)
        executable_bytes = executable.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot pin pdftotext executable: {exc}") from exc
    if not executable.is_file():
        raise ValueError("resolved pdftotext executable is not a regular file")
    _reject_reparse_components(executable)
    executable_sha256 = _sha(executable_bytes)
    snapshots[executable] = executable_sha256
    try:
        result = subprocess.run(
            [str(executable), "-v"], capture_output=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"cannot obtain pdftotext version: {exc}") from exc
    version_output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    version_lines = [line.strip() for line in version_output.splitlines() if line.strip()]
    if result.returncode != 0 or not version_lines or not version_lines[0].startswith("pdftotext version "):
        raise ValueError("resolved pdftotext executable did not report a usable version")
    return executable, executable_sha256, version_lines[0]


def _page1_verification_hold_reason(
    proof: dict,
    page1: dict,
    executable: Path,
    executable_version: str,
) -> str | None:
    if page1.get("pdf_sha256") != proof.get("pdf_sha256"):
        return "PAGE1_PDF_HASH_MISMATCH"
    if page1.get("pdftotext_version") != executable_version:
        return "PDFTOTEXT_VERSION_MISMATCH"
    page1_text = page1.get("page1_text")
    if not isinstance(page1_text, str):
        return "PAGE1_TEXT_INVALID"
    page1_bytes = page1_text.encode("utf-8")
    if page1.get("page1_text_sha256") != _sha(page1_bytes):
        return "PAGE1_TEXT_HASH_MISMATCH"
    try:
        result = subprocess.run(
            [str(executable), "-f", "1", "-l", "1", "-layout", proof["pdf_path"], "-"],
            capture_output=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "PAGE1_EXTRACTION_FAILED"
    if result.returncode != 0:
        return "PAGE1_EXTRACTION_FAILED"
    if result.stdout != page1_bytes or _sha(result.stdout) != page1.get("page1_text_sha256"):
        return "PAGE1_EXTRACTION_MISMATCH"
    return None


def build(args: argparse.Namespace) -> Path:
    root = Path(os.path.abspath(args.corpus_root))
    if not root.is_dir():
        raise ValueError("corpus root must already exist as a directory")
    _reject_reparse_components(root)
    root = root.resolve(strict=True)
    output_dir = Path(os.path.abspath(args.output_dir))
    _inside(output_dir, root, "output directory")
    parent = output_dir.parent
    _inside(parent, root, "output parent")
    if output_dir.exists():
        raise ValueError(f"output directory already exists; exclusive creation refused: {output_dir}")

    snapshots: dict[Path, str] = {}
    pdftotext_path, pdftotext_sha256, pdftotext_version = _resolve_pdftotext(snapshots)
    adapter_path = Path(acl_task4_evidence_adapter.__file__).resolve(strict=True)
    adapter_sha256 = _sha(adapter_path.read_bytes())
    snapshots[adapter_path] = adapter_sha256
    proof_sha = _digest_argument(args.proof_sha256, "proof hash")
    receipt_sha = _digest_argument(args.proof_receipt_sha256, "proof receipt hash")
    page1_sha = _digest_argument(args.page1_sha256, "page1 witnesses hash")
    reviews_sha = _digest_argument(args.reviews_sha256, "reviews hash")
    proof_path = _inside(Path(args.proof), root, "proof rows")
    proof_receipt_path = _inside(Path(args.proof_receipt), root, "proof receipt")
    page1_path = _inside(Path(args.page1_witnesses), root, "page1 witnesses")
    reviews_path = _inside(Path(args.reviews), root, "reviews")

    expected_receipt_path = _inside(root / _PARENT_RECEIPT_RELATIVE_PATH, root, "authoritative parent receipt")
    expected_proof_path = _inside(root / _PARENT_PROOF_RELATIVE_PATH, root, "authoritative parent proof")
    if os.path.normcase(os.path.normpath(str(proof_receipt_path))) != os.path.normcase(os.path.normpath(str(expected_receipt_path))):
        raise ValueError("proof receipt path is not the authoritative frozen 670-row parent receipt")
    if os.path.normcase(os.path.normpath(str(proof_path))) != os.path.normcase(os.path.normpath(str(expected_proof_path))):
        raise ValueError("proof path is not the authoritative frozen 670-row parent proof")
    if receipt_sha != _PARENT_RECEIPT_SHA256:
        raise ValueError("proof receipt hash differs from the authoritative frozen parent pin")
    if proof_sha != _PARENT_PROOF_SHA256:
        raise ValueError("proof hash differs from the authoritative frozen parent pin")

    selection_ids: tuple[str, ...] | None = None
    selection_path: Path | None = None
    selection_sha: str | None = None
    if args.mode == "pilot-subset":
        if not args.selection_list or not args.selection_sha256:
            raise ValueError("pilot-subset mode requires --selection-list and --selection-sha256")
        selection_sha = _digest_argument(args.selection_sha256, "selection-list hash")
        selection_path = _inside(Path(args.selection_list), root, "selection list")
        selection_data = _read_pinned(selection_path, selection_sha, "selection list", root, snapshots)
        selection_ids = _parse_selection(selection_data)
    elif args.mode != "full":
        raise ValueError("mode must be exactly full or pilot-subset")
    elif args.selection_list is not None or args.selection_sha256 is not None:
        raise ValueError("selection-list inputs are only valid in pilot-subset mode")

    proof_data = _read_pinned(proof_path, proof_sha, "proof rows", root, snapshots)
    receipt_data = _read_pinned(proof_receipt_path, receipt_sha, "proof receipt", root, snapshots)
    page1_data = _read_pinned(page1_path, page1_sha, "page1 witnesses", root, snapshots)
    reviews_data = _read_pinned(reviews_path, reviews_sha, "reviews", root, snapshots)
    try:
        proof_receipt = json.loads(receipt_data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid proof receipt JSON: {exc}") from exc
    if not isinstance(proof_receipt, dict) or proof_receipt.get("schema_version") != _RECEIPT_SCHEMA:
        raise ValueError("proof receipt does not have the frozen ACL proof-only schema")
    run_receipt = proof_receipt.get("final_proof_only_run")
    if not isinstance(run_receipt, dict):
        raise ValueError("proof receipt lacks final_proof_only_run scope")
    if _canonical_path(Path(run_receipt.get("rows_path", ""))) != _canonical_path(proof_path):
        raise ValueError("proof receipt does not identify the exact proof-row file")
    if run_receipt.get("rows_sha256") != proof_sha:
        raise ValueError("proof receipt does not pin the exact proof-row hash")
    proof_rows = _json_object_lines(proof_data, "proof rows")
    proof_by_id = _unique_by_item(proof_rows, "proof")
    if len(proof_rows) != 670 or len(proof_by_id) != 670:
        raise ValueError("proof row conservation failed: authoritative parent must contain exactly 670 unique rows")
    if run_receipt.get("row_count") != 670 or run_receipt.get("unique_item_ids") != 670:
        raise ValueError("proof receipt must identify exactly 670 unique parent rows")
    scope = proof_receipt.get("scope")
    if not isinstance(scope, str) or "670 frozen ACL Findings" not in scope:
        raise ValueError("proof receipt scope does not name the exact frozen 670-row ACL Findings count")
    if not ("expected_title" in scope and "source_record.title" in scope and "bib" in scope):
        raise ValueError("proof receipt does not identify the frozen 670-row bib-title scope")

    if selection_ids is not None:
        missing_parent_ids = set(selection_ids) - set(proof_by_id)
        if missing_parent_ids:
            raise ValueError(f"selected IDs are absent from the authoritative parent proof: {sorted(missing_parent_ids)}")

    if selection_ids is not None:
        selected_proofs = {item_id: proof_by_id[item_id] for item_id in selection_ids}
    else:
        selected_proofs = proof_by_id

    _verify_proof_source_rows(proof_rows, root, snapshots)
    raw_page1_rows = load_task4_page1_witnesses(page1_data)
    raw_page1_by_id = _unique_by_item(raw_page1_rows, "Task 4 page1 witness")
    raw_page1_lines = page1_data.splitlines(keepends=True)
    if len(raw_page1_lines) != len(raw_page1_rows):
        raise ValueError("Task 4 page1 JSONL row/byte-line count mismatch")
    raw_page1_sha_by_id = {
        row["item_id"]: _sha(raw_line)
        for row, raw_line in zip(raw_page1_rows, raw_page1_lines, strict=True)
    }
    review_by_id = _unique_by_item(_json_object_lines(reviews_data, "reviews"), "review")
    unknown_page1 = set(raw_page1_by_id) - set(proof_by_id)
    unknown_reviews = set(review_by_id) - set(proof_by_id)
    if unknown_page1:
        raise ValueError(f"page1 witnesses are broader than proof scope: {sorted(unknown_page1)}")
    if unknown_reviews:
        raise ValueError(f"reviews are broader than proof scope: {sorted(unknown_reviews)}")

    page1_by_id: dict[str, dict] = {}
    page1_hold_reason_by_id: dict[str, str] = {}
    for item_id in sorted(set(raw_page1_by_id).intersection(selected_proofs)):
        normalized, hold_reason = normalize_task4_page1_witness(
            raw_page1_by_id[item_id],
            expected_item_id=item_id,
            expected_pdf_sha256=selected_proofs[item_id]["pdf_sha256"],
        )
        if normalized is None:
            page1_hold_reason_by_id[item_id] = hold_reason or "PAGE1_WITNESS_INVALID"
        else:
            page1_by_id[item_id] = normalized

    for item_id, proof in selected_proofs.items():
        page1 = page1_by_id.get(item_id)
        review = review_by_id.get(item_id)
        if page1 is not None and page1.get("pdf_sha256") != proof.get("pdf_sha256"):
            raise ValueError(f"{item_id}: page1 PDF hash does not match the parent proof")
        if review is not None:
            if set(review) != _EXACT_REVIEW_KEYS:
                continue
            if review.get("pdf_sha256") != proof.get("pdf_sha256"):
                raise ValueError(f"{item_id}: review PDF hash does not match the parent proof")
            if review.get("proof_row_sha256") != _canonical_sha(proof):
                raise ValueError(f"{item_id}: review proof-row hash does not match the parent proof")
            if page1 is not None and review.get("page1_text_sha256") != page1.get("page1_text_sha256"):
                raise ValueError(f"{item_id}: review page1 text hash does not match the page1 witness")
    projection_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    page1_raw_row_sha256: dict[str, str | None] = {}
    for item_id in sorted(selected_proofs):
        proof = selected_proofs[item_id]
        page1 = page1_by_id.get(item_id)
        page1_raw_sha = raw_page1_sha_by_id.get(item_id)
        page1_raw_row_sha256[item_id] = page1_raw_sha
        review = review_by_id.get(item_id)
        page1_error = (
            _page1_verification_hold_reason(proof, page1, pdftotext_path, pdftotext_version)
            if page1 is not None
            else None
        )
        if page1 is None and item_id not in raw_page1_by_id:
            projected = _hold_row(proof, "PAGE1_WITNESS_MISSING", review=review)
        elif page1 is None:
            projected = _hold_row(
                proof,
                page1_hold_reason_by_id[item_id],
                review=review,
            )
        elif page1_error is not None:
            projected = _hold_row(proof, page1_error, page1=page1, review=review)
        elif review is None:
            projected = _hold_row(proof, "REVIEW_MISSING", page1=page1)
        elif set(review) != _EXACT_REVIEW_KEYS:
            projected = _hold_row(proof, "REVIEW_EXACT_RECORD_MISSING", page1=page1, review=review)
        else:
            # A stale or malformed reviewer record is an input error, never a projection.
            projected = validate_admission(proof, page1, review)
            projected["page1_witness_sha256"] = _canonical_sha(page1)
        projected["review_authentication"] = "EXTERNAL_QA_REQUIRED"
        projected["downstream_card_use_approved"] = False
        projection_rows.append(projected)
        diagnostic_rows.append(
            _diagnostic_row(proof, page1, review, projected, page1_raw_sha256=page1_raw_sha)
        )

    expected_output_count = 10 if args.mode == "pilot-subset" else 670
    if len(projection_rows) != expected_output_count or len(diagnostic_rows) != expected_output_count:
        raise ValueError("all-row conservation failed before output")
    expected_ids = sorted(selected_proofs)
    if [row["item_id"] for row in projection_rows] != expected_ids or [row["item_id"] for row in diagnostic_rows] != expected_ids:
        raise ValueError("output item IDs do not exactly conserve the selected parent set")
    _capture_snapshot_recheck(snapshots)
    if selection_path is not None and _sha(selection_path.read_bytes()) != selection_sha:
        raise ValueError("selection-list hash changed immediately before output")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _inside(output_dir.parent, root, "output parent")
    try:
        output_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(f"output directory already exists; exclusive creation refused: {output_dir}") from exc
    _reject_reparse_components(output_dir)

    witness_bytes = _jsonl_bytes(diagnostic_rows)
    projection_bytes = _jsonl_bytes(projection_rows)
    witness_path = output_dir / "acl_metadata_witness.v1.jsonl"
    projection_path = output_dir / "acl_metadata_projection.v1.jsonl"
    for path, data in ((witness_path, witness_bytes), (projection_path, projection_bytes)):
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    counts = {
        "SUPPORTED": sum(row["status"] == "SUPPORTED" for row in projection_rows),
        "METADATA_HOLD": sum(row["status"] == "METADATA_HOLD" for row in projection_rows),
    }
    script_hash = _sha(Path(__file__).read_bytes())
    receipt = {
        "schema_version": "acl_metadata_projection_run_receipt.v1",
        "run_mode": "PILOT_SUBSET" if args.mode == "pilot-subset" else "FULL",
        "parent_proof_receipt_path": str(proof_receipt_path),
        "parent_proof_receipt_sha256": receipt_sha,
        "parent_proof_path": str(proof_path),
        "parent_proof_sha256": proof_sha,
        "parent_row_count": 670,
        "selection_list_path": str(selection_path) if selection_path is not None else None,
        "selection_list_sha256": selection_sha,
        "selected_ids": list(sorted(selected_proofs)),
        "selected_row_count": len(selected_proofs),
        "output_row_count": len(projection_rows),
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol_sha256": script_hash,
        "toolchain": {
            "pdftotext_path": str(pdftotext_path),
            "pdftotext_sha256": pdftotext_sha256,
            "pdftotext_version": pdftotext_version,
        },
        "input_sha256": {
            "proof_rows": proof_sha,
            "proof_receipt": receipt_sha,
            "page1_witnesses": page1_sha,
            "reviews": reviews_sha,
        },
        "page1_adapter": {
            "protocol_version": ADAPTER_PROTOCOL_VERSION,
            "protocol_sha256": adapter_sha256,
            "raw_input_path": str(page1_path),
            "raw_input_sha256": page1_sha,
            "raw_input_row_sha256": page1_raw_row_sha256,
        },
        "pinned_files_sha256": {
            str(path): digest for path, digest in sorted(snapshots.items(), key=lambda pair: str(pair[0]))
        },
        "output_sha256": {
            "acl_metadata_witness.v1.jsonl": _sha(witness_bytes),
            "acl_metadata_projection.v1.jsonl": _sha(projection_bytes),
        },
        "counts": counts,
        "row_count": len(projection_rows),
        "unique_item_ids": len({row["item_id"] for row in projection_rows}),
        "source_admission_approved": False,
        "human_approved": False,
        "graph_ingested": False,
        "review_authentication": "EXTERNAL_QA_REQUIRED",
        "downstream_card_use_approved": False,
    }
    receipt_path = output_dir / "receipt.json"
    receipt_bytes = json.dumps(receipt, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    with receipt_path.open("xb") as handle:
        handle.write(receipt_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", required=True)
    parser.add_argument("--proof-sha256", required=True)
    parser.add_argument("--proof-receipt", required=True)
    parser.add_argument("--proof-receipt-sha256", required=True)
    parser.add_argument("--page1-witnesses", required=True)
    parser.add_argument("--page1-sha256", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--reviews-sha256", required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", required=True, choices=("full", "pilot-subset"))
    parser.add_argument("--selection-list")
    parser.add_argument("--selection-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        output = build(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
