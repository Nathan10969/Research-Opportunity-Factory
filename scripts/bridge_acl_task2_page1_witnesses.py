#!/usr/bin/env python3
"""Convert only the ten frozen flat ACL page-1 witnesses to Task 4 schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from idea_factory.acl_task4_evidence_adapter import (
    TASK4_PAGE1_SCHEMA,
    load_task4_page1_witnesses,
    normalize_task4_page1_witness,
)


_FLAT_IDS = frozenset({
    "2026.findings-acl.1077", "2026.findings-acl.1105", "2026.findings-acl.1174",
    "2026.findings-acl.1266", "2026.findings-acl.1320", "2026.findings-acl.135",
    "2026.findings-acl.1371", "2026.findings-acl.1388", "2026.findings-acl.1412",
    "2026.findings-acl.1530",
})
_FLAT_KEYS = frozenset({
    "item_id", "pdf_sha256", "page1_text_sha256", "page1_text",
    "pdftotext_version", "witness_author_id",
})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REPARSE_POINT = 0x0400
_OUTPUT_NAME = "page1-witnesses.task4-compatible.v1.jsonl"
_AUTHORITATIVE_INPUT_PATH = Path(
    r"F:\LLM_Evoke\runs\parallel24-20260927-1340\engineering\acl-page1-lexical-bridge-670-20260928-v1\page1-witnesses.adapter-safe.v1.jsonl"
)
_AUTHORITATIVE_INPUT_SHA256 = "53123ef352929e9e5601ae0379400a9d8e763dd1f7540f9487c49ad725e4e05a"
_AUTHORITATIVE_PROOF_PATH = Path(
    r"F:\LLM_Evoke\runs\parallel24-20260927-1340\engineering\acl670-metadata-witness-proofonly-v5-20260927\acl_metadata_witness.jsonl"
)
_AUTHORITATIVE_PROOF_SHA256 = "375b75a511042b5e617a69649814d4c599a9b5d6a411d8f076c13ac48796cb8a"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _no_constant(value: str) -> None:
    raise ValueError(f"non-JSON constant: {value}")


def _jsonl(data: bytes, label: str) -> list[tuple[dict, bytes]]:
    if not data or not data.endswith(b"\n"):
        raise ValueError(f"{label} JSONL must be newline-terminated")
    rows: list[tuple[dict, bytes]] = []
    seen: set[str] = set()
    for number, line in enumerate(data.split(b"\n")[:-1], 1):
        raw = line[:-1] if line.endswith(b"\r") else line
        if not raw or b"\r" in raw:
            raise ValueError(f"invalid {label} JSONL line {number}: blank or bare carriage return")
        try:
            row = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys,
                             parse_constant=_no_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid {label} JSONL line {number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} JSONL line {number} must be a JSON object")
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{label} JSONL line {number} lacks item_id")
        if item_id in seen:
            raise ValueError(f"duplicate {label} item_id: {item_id}")
        seen.add(item_id)
        rows.append((row, line + b"\n"))
    return rows


def _reject_reparse(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if current.is_symlink() or getattr(info, "st_file_attributes", 0) & _REPARSE_POINT:
            raise ValueError(f"symlink or reparse path component is forbidden: {current}")


def _inside(value: str | Path, root: Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    if ".." in path.parts:
        raise ValueError(f"{label} must not contain parent traversal")
    path = Path(os.path.abspath(path))
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside corpus root") from exc
    _reject_reparse(path)
    return path


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _read_pinned(path: Path, expected: str, label: str) -> bytes:
    _reject_reparse(path)
    data = path.read_bytes()
    if _sha(data) != expected:
        raise ValueError(f"{label} hash mismatch")
    return data


def _write_verified(path: Path, data: bytes) -> bytes:
    with path.open("xb") as handle:
        if handle.write(data) != len(data):
            raise OSError(f"short output write: {path.name}")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        actual = path.read_bytes()
    except OSError as exc:
        raise OSError(f"output readback failed: {path.name}: {exc}") from exc
    if len(actual) != len(data) or _sha(actual) != _sha(data) or actual != data:
        raise ValueError(f"output readback mismatch: {path.name}")
    return actual


def _flat_to_nested(flat: dict) -> dict:
    if set(flat) != _FLAT_KEYS:
        raise ValueError(f"flat row has wrong keys: {flat.get('item_id')}")
    item_id = flat["item_id"]
    _digest(flat["pdf_sha256"], f"{item_id} PDF hash")
    _digest(flat["page1_text_sha256"], f"{item_id} page1 text hash")
    text = flat["page1_text"]
    if not isinstance(text, str) or not text.strip() or _sha(text.encode("utf-8")) != flat["page1_text_sha256"]:
        raise ValueError(f"{item_id}: page1 text hash does not match exact UTF-8 text")
    for key in ("pdftotext_version", "witness_author_id"):
        if not isinstance(flat[key], str) or not flat[key].strip():
            raise ValueError(f"{item_id}: {key} is missing")
    return {
        "schema_version": TASK4_PAGE1_SCHEMA,
        "item_id": item_id,
        "pdf": {"sha256": flat["pdf_sha256"]},
        "pdf_page1": {"text": text, "text_sha256": flat["page1_text_sha256"],
                      "pdftotext_version": flat["pdftotext_version"]},
        "witness_author_id": flat["witness_author_id"],
    }


def bridge(args: argparse.Namespace) -> Path:
    if args.expected_count != 670:
        raise ValueError("expected-count must be exactly 670")
    input_sha = _digest(args.input_sha256, "input-sha256")
    proof_sha = _digest(args.proof_sha256, "proof-sha256")
    if input_sha != _AUTHORITATIVE_INPUT_SHA256:
        raise ValueError("input hash differs from authoritative frozen pin")
    if proof_sha != _AUTHORITATIVE_PROOF_SHA256:
        raise ValueError("proof hash differs from authoritative frozen pin")
    root = Path(args.corpus_root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("corpus root must be an existing absolute directory")
    root = Path(os.path.abspath(root))
    _reject_reparse(root)
    source = _inside(args.input, root, "input")
    proof_path = _inside(args.proof, root, "proof")
    if _normalized_path(source) != _normalized_path(_AUTHORITATIVE_INPUT_PATH):
        raise ValueError("authoritative input path mismatch")
    if _normalized_path(proof_path) != _normalized_path(_AUTHORITATIVE_PROOF_PATH):
        raise ValueError("authoritative proof path mismatch")
    output_dir = _inside(args.output_dir, root, "output directory")
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")

    source_rows = _jsonl(_read_pinned(source, input_sha, "input"), "input")
    proof_rows = _jsonl(_read_pinned(proof_path, proof_sha, "proof"), "proof")
    if len(source_rows) != 670 or len(proof_rows) != 670:
        raise ValueError("input and proof must each contain exactly 670 rows")
    proof_by_id = {}
    for row, _ in proof_rows:
        item_id = row["item_id"]
        proof_by_id[item_id] = _digest(row.get("pdf_sha256"), f"{item_id} proof PDF hash")
    source_ids = {row["item_id"] for row, _ in source_rows}
    if source_ids != set(proof_by_id):
        raise ValueError("input and proof ID sets differ")

    output_lines = []
    semantic_by_id = {}
    flat_ids: set[str] = set()
    nested_count = 0
    for row, raw_line in source_rows:
        item_id = row["item_id"]
        expected_pdf = proof_by_id[item_id]
        if set(row) == _FLAT_KEYS:
            flat_ids.add(item_id)
            if row["pdf_sha256"] != expected_pdf:
                raise ValueError(f"{item_id}: flat PDF hash does not match proof")
            nested = _flat_to_nested(row)
            normalized, hold = normalize_task4_page1_witness(
                nested, expected_item_id=item_id, expected_pdf_sha256=expected_pdf)
            if hold is not None or normalized != row:
                raise ValueError(f"{item_id}: flat semantic fields changed during conversion")
            ending = b"\r\n" if raw_line.endswith(b"\r\n") else b"\n"
            output_lines.append(json.dumps(nested, sort_keys=True, ensure_ascii=True,
                                           separators=(",", ":")).encode("utf-8") + ending)
            semantic_by_id[item_id] = row
        elif row.get("schema_version") == TASK4_PAGE1_SCHEMA:
            normalized, hold = normalize_task4_page1_witness(
                row, expected_item_id=item_id, expected_pdf_sha256=expected_pdf)
            if hold is not None:
                raise ValueError(f"{item_id}: nested row cannot normalize: {hold}")
            output_lines.append(raw_line)
            semantic_by_id[item_id] = normalized
            nested_count += 1
        else:
            raise ValueError(f"{item_id}: unsupported page1 row schema")
    if flat_ids != _FLAT_IDS:
        raise ValueError(f"flat ID set differs from exact ten pilot IDs: missing={sorted(_FLAT_IDS - flat_ids)}, extra={sorted(flat_ids - _FLAT_IDS)}")
    if nested_count != 660:
        raise ValueError(f"expected 660 unchanged nested rows, found {nested_count}")

    output_bytes = b"".join(output_lines)
    output_rows = load_task4_page1_witnesses(output_bytes)
    if len(output_rows) != 670 or [row["item_id"] for row in output_rows] != [row["item_id"] for row, _ in source_rows]:
        raise ValueError("output row count or exact ID order changed")
    for row in output_rows:
        item_id = row["item_id"]
        normalized, hold = normalize_task4_page1_witness(
            row, expected_item_id=item_id, expected_pdf_sha256=proof_by_id[item_id])
        if hold is not None or normalized != semantic_by_id[item_id]:
            raise ValueError(f"{item_id}: output semantic fields differ from source")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _inside(output_dir.parent, root, "output parent")
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        _inside(staging, root, "staging directory")
        actual_output = _write_verified(staging / _OUTPUT_NAME, output_bytes)
        receipt = {
            "schema_version": "acl_task2_page1_structural_bridge_receipt.v1",
            "source": {"path": str(source), "sha256": input_sha, "rows": len(source_rows)},
            "proof": {"path": str(proof_path), "sha256": proof_sha, "rows": len(proof_rows)},
            "output": {"path": str(output_dir / _OUTPUT_NAME), "sha256": _sha(actual_output),
                       "bytes": len(actual_output), "rows": len(output_rows)},
            "converted_flat_ids": sorted(flat_ids),
            "converted_flat_count": len(flat_ids),
            "unchanged_nested_count": nested_count,
            "all_pdf_hashes_bound_to_proof": True,
            "all_six_semantic_fields_preserved": True,
            "review_authentication": "EXTERNAL_QA_REQUIRED",
            "source_admission_approved": False,
            "downstream_card_use_approved": False,
            "human_approved": False,
            "graph_ingested": False,
            "overlay_is_approval": False,
        }
        receipt_bytes = (json.dumps(receipt, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        _write_verified(staging / "receipt.json", receipt_bytes)
        _read_pinned(source, input_sha, "input pre-publish")
        _read_pinned(proof_path, proof_sha, "proof pre-publish")
        os.rename(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "input-sha256", "proof", "proof-sha256", "corpus-root", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        output = bridge(args)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
