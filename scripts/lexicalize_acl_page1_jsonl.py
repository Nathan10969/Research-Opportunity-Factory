#!/usr/bin/env python3
"""Escape one literal U+2028 in pinned ACL page-1 JSONL without changing data."""

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

from idea_factory.acl_task4_evidence_adapter import load_task4_page1_witnesses


_LITERAL_U2028 = "\u2028".encode("utf-8")
_ESCAPED_U2028 = b"\\u2028"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REPARSE_POINT = 0x0400
_OUTPUT_NAME = "page1-witnesses.adapter-safe.v1.jsonl"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _no_constant(value: str) -> None:
    raise ValueError(f"non-JSON constant: {value}")


def _strict_input_rows(data: bytes) -> list[dict]:
    if not data or not data.endswith(b"\n"):
        raise ValueError("input JSONL rows must be newline-terminated")
    rows = []
    seen: set[str] = set()
    for number, line in enumerate(data.split(b"\n")[:-1], 1):
        raw = line[:-1] if line.endswith(b"\r") else line
        if not raw or b"\r" in raw:
            raise ValueError(f"invalid input JSONL line {number}: blank or bare carriage return")
        try:
            row = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys,
                             parse_constant=_no_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid input JSONL line {number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"input JSONL line {number} must be a JSON object")
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"input JSONL line {number} lacks item_id")
        if item_id in seen:
            raise ValueError(f"duplicate input item_id: {item_id}")
        seen.add(item_id)
        rows.append(row)
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


def lexicalize(args: argparse.Namespace) -> Path:
    if args.expected_count != 670:
        raise ValueError("expected-count must be exactly 670")
    if args.expected_literal_u2028 != 1:
        raise ValueError("expected literal U+2028 count must be exactly 1")
    if not isinstance(args.input_sha256, str) or not _SHA256.fullmatch(args.input_sha256):
        raise ValueError("input-sha256 must be a lowercase SHA-256 digest")
    root = Path(args.corpus_root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("corpus root must be an existing absolute directory")
    root = Path(os.path.abspath(root))
    _reject_reparse(root)
    source = _inside(args.input, root, "input")
    output_dir = _inside(args.output_dir, root, "output directory")
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")

    source_bytes = source.read_bytes()
    if _sha(source_bytes) != args.input_sha256:
        raise ValueError("input hash mismatch")
    source_rows = _strict_input_rows(source_bytes)
    if len(source_rows) != 670:
        raise ValueError(f"input row count must be 670, found {len(source_rows)}")
    replacement_count = source_bytes.count(_LITERAL_U2028)
    if replacement_count != 1:
        raise ValueError(f"literal U+2028 count must be 1, found {replacement_count}")

    output_bytes = source_bytes.replace(_LITERAL_U2028, _ESCAPED_U2028)
    output_rows = load_task4_page1_witnesses(output_bytes)
    if len(output_rows) != 670 or [row["item_id"] for row in output_rows] != [row["item_id"] for row in source_rows]:
        raise ValueError("adapter output row count or exact ID order changed")
    if output_rows != source_rows:
        raise ValueError("adapter output parsed objects differ from source")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _inside(output_dir.parent, root, "output parent")
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        _inside(staging, root, "staging directory")
        actual_output = _write_verified(staging / _OUTPUT_NAME, output_bytes)
        receipt = {
            "schema_version": "acl_page1_lexical_bridge_receipt.v1",
            "source": {"path": str(source), "sha256": args.input_sha256,
                       "bytes": len(source_bytes), "rows": len(source_rows)},
            "output": {"path": str(output_dir / _OUTPUT_NAME), "sha256": _sha(actual_output),
                       "bytes": len(actual_output), "rows": len(output_rows)},
            "literal_u2028_replacements": replacement_count,
            "byte_length_delta": len(actual_output) - len(source_bytes),
            "parsed_objects_equal": True,
            "review_authentication": "EXTERNAL_QA_REQUIRED",
            "source_admission_approved": False,
            "downstream_card_use_approved": False,
            "human_approved": False,
            "graph_ingested": False,
            "overlay_is_approval": False,
        }
        receipt_bytes = (json.dumps(receipt, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        _write_verified(staging / "receipt.json", receipt_bytes)
        if _sha(source.read_bytes()) != args.input_sha256:
            raise ValueError("input hash changed before publish")
        os.rename(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "input-sha256", "corpus-root", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-literal-u2028", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        output = lexicalize(args)
    except (OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
