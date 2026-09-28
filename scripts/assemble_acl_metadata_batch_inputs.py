#!/usr/bin/env python3
"""Assemble pinned ACL page-1/review batches without authenticating decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path


SCHEMA = "acl_metadata_batch_inputs.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REPARSE_POINT = 0x0400
_WITNESS_KEYS = {"batch_number", "path", "sha256"}
_REVIEW_KEYS = {"batch_number", "path", "sha256", "qa_report_path", "qa_report_sha256"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _no_constant(value: str) -> None:
    raise ValueError(f"non-JSON constant: {value}")


def _json_object(data: bytes, label: str) -> dict:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=_no_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _jsonl(data: bytes, label: str) -> list[dict]:
    if data and not data.endswith(b"\n"):
        raise ValueError(f"{label} JSONL rows must be newline-terminated")
    rows = []
    for number, raw in enumerate(data.split(b"\n")[:-1], 1):
        if raw.endswith(b"\r"):
            raw = raw[:-1]
        if not raw or b"\r" in raw:
            raise ValueError(f"{label} JSONL line {number} is blank or has a bare carriage return")
        rows.append(_json_object(raw, f"{label} JSONL line {number}"))
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


def _inside(value: str, root: Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute exact path")
    if ".." in path.parts:
        raise ValueError(f"{label} must not contain parent traversal")
    path = Path(os.path.abspath(path))
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside corpus root") from exc
    _reject_reparse(path)
    return path


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _read_pinned(path: Path, expected: str, label: str, pins: dict[Path, str]) -> bytes:
    _reject_reparse(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    actual = _sha(data)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, got {actual}")
    if path in pins and pins[path] != expected:
        raise ValueError(f"conflicting hash pins for {path}")
    pins[path] = expected
    return data


def _batch_entries(value: object, *, kind: str, root: Path, expected: int) -> dict[int, dict]:
    if not isinstance(value, list):
        raise ValueError(f"manifest {kind} must be a list")
    required = _WITNESS_KEYS if kind == "witnesses" else _REVIEW_KEYS
    entries: dict[int, dict] = {}
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError(f"manifest {kind} entry has incorrect keys")
        number = raw["batch_number"]
        if not isinstance(number, int) or isinstance(number, bool):
            raise ValueError(f"{kind} batch_number must be an integer")
        if number < 1 or number > expected:
            raise ValueError(f"unexpected {kind} batch number: {number}")
        if number in entries:
            raise ValueError(f"duplicate {kind} batch number: {number}")
        entry = dict(raw)
        for path_key, sha_key in (("path", "sha256"), ("qa_report_path", "qa_report_sha256")):
            if path_key not in entry:
                continue
            if not isinstance(entry[path_key], str):
                raise ValueError(f"{kind} {path_key} must be a path")
            entry[path_key] = _inside(entry[path_key], root, f"{kind} {path_key}")
            entry[sha_key] = _digest(entry[sha_key], f"{kind} {sha_key}")
        entries[number] = entry
    return entries


def _ids(rows: list[dict], label: str) -> set[str]:
    seen: set[str] = set()
    for row in rows:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{label} row lacks a nonempty item_id")
        if item_id in seen:
            raise ValueError(f"duplicate {label} item_id: {item_id}")
        seen.add(item_id)
    return seen


def _id_set_sha(ids: set[str]) -> str:
    return _sha("".join(f"{item_id}\n" for item_id in sorted(ids)).encode("utf-8"))


def assemble(args: argparse.Namespace) -> Path:
    if args.expected_batches < 1 or args.expected_total < 1:
        raise ValueError("expected batch and total counts must be positive")
    if args.expected_total != args.expected_batches * 10:
        raise ValueError("expected total must be exactly ten times expected batches")
    root = Path(args.corpus_root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("corpus root must be an existing absolute directory")
    root = Path(os.path.abspath(root))
    _reject_reparse(root)
    manifest_path = _inside(args.manifest, root, "manifest")
    proof_path = _inside(args.proof, root, "proof")
    output = _inside(args.output_dir, root, "output directory")
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    proof_sha = _digest(args.proof_sha256, "proof-sha256")

    pins: dict[Path, str] = {}
    manifest_data = manifest_path.read_bytes()
    manifest_sha = _sha(manifest_data)
    pins[manifest_path] = manifest_sha
    manifest = _json_object(manifest_data, "manifest")
    if set(manifest) != {"schema_version", "witnesses", "reviews"}:
        raise ValueError("manifest has incorrect keys")
    if manifest["schema_version"] != SCHEMA:
        raise ValueError(f"manifest schema_version must be {SCHEMA}")
    witnesses = _batch_entries(manifest["witnesses"], kind="witnesses", root=root,
                               expected=args.expected_batches)
    reviews = _batch_entries(manifest["reviews"], kind="reviews", root=root,
                             expected=args.expected_batches)
    expected_numbers = set(range(1, args.expected_batches + 1))
    if set(witnesses) != expected_numbers:
        raise ValueError(f"missing witness batches: {sorted(expected_numbers - set(witnesses))}")
    if not set(reviews) <= set(witnesses):
        raise ValueError("review batch with no witness batch")

    proof_data = _read_pinned(proof_path, proof_sha, "proof", pins)
    proof_rows = _jsonl(proof_data, "proof")
    proof_ids = _ids(proof_rows, "proof")
    if len(proof_rows) != args.expected_total:
        raise ValueError(f"proof count differs from expected total: {len(proof_rows)}")

    witness_bytes: list[bytes] = []
    review_bytes: list[bytes] = []
    witness_ids: set[str] = set()
    review_ids: set[str] = set()
    witness_ids_by_batch: dict[int, set[str]] = {}
    for number in sorted(witnesses):
        entry = witnesses[number]
        data = _read_pinned(entry["path"], entry["sha256"], f"witness batch {number}", pins)
        rows = _jsonl(data, f"witness batch {number}")
        batch_ids = _ids(rows, f"witness batch {number}")
        if len(rows) != 10:
            raise ValueError(f"witness batch {number} must contain exactly ten unique IDs")
        overlap = witness_ids & batch_ids
        if overlap:
            raise ValueError(f"duplicate witness IDs across batches: {sorted(overlap)}")
        witness_ids_by_batch[number] = batch_ids
        witness_ids.update(batch_ids)
        witness_bytes.append(data)
    if witness_ids != proof_ids:
        raise ValueError(f"witness IDs do not exactly cover proof IDs: missing={len(proof_ids - witness_ids)}, extra={len(witness_ids - proof_ids)}")

    for number in sorted(reviews):
        entry = reviews[number]
        data = _read_pinned(entry["path"], entry["sha256"], f"review batch {number}", pins)
        rows = _jsonl(data, f"review batch {number}")
        batch_ids = _ids(rows, f"review batch {number}")
        overlap = review_ids & batch_ids
        if overlap:
            raise ValueError(f"duplicate review IDs across batches: {sorted(overlap)}")
        if not batch_ids <= witness_ids_by_batch[number]:
            raise ValueError(f"review batch {number} contains ID outside its witness batch")
        review_ids.update(batch_ids)
        review_bytes.append(data)
        _read_pinned(entry["qa_report_path"], entry["qa_report_sha256"],
                     f"QA report batch {number}", pins)

    # Close the read-to-write gap as far as a local custody operation can:
    # re-read every source, including the unpinned manifest, before creating output.
    for path, expected in pins.items():
        _read_pinned(path, expected, "input pre-output recheck", pins)
    page1_output = b"".join(witness_bytes)
    review_output = b"".join(review_bytes)
    receipt = {
        "schema_version": "acl_metadata_batch_assembly_receipt.v1",
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
            "proof": {"path": str(proof_path), "sha256": proof_sha},
            "witnesses": [{"batch_number": n, "path": str(e["path"]), "sha256": e["sha256"]}
                          for n, e in sorted(witnesses.items())],
            "reviews": [{"batch_number": n, "path": str(e["path"]), "sha256": e["sha256"],
                         "qa_report_path": str(e["qa_report_path"]),
                         "qa_report_sha256": e["qa_report_sha256"]}
                        for n, e in sorted(reviews.items())],
        },
        "outputs": {
            "page1-witnesses.v1.jsonl": {"sha256": _sha(page1_output), "bytes": len(page1_output)},
            "reviews.v1.jsonl": {"sha256": _sha(review_output), "bytes": len(review_output)},
        },
        "counts": {"expected_batches": args.expected_batches, "expected_total": args.expected_total,
                   "proof_rows": len(proof_rows), "witness_batches": len(witnesses),
                   "witness_rows": len(witness_ids), "review_batches": len(reviews),
                   "review_rows": len(review_ids)},
        "proof_id_set_sha256": _id_set_sha(proof_ids),
        "witness_id_set_sha256": _id_set_sha(witness_ids),
        "proof_id_set_covered": witness_ids == proof_ids,
        "review_authentication": "EXTERNAL_QA_REQUIRED",
        "source_admission_approved": False,
        "downstream_card_use_approved": False,
        "human_approved": False,
        "graph_ingested": False,
        "overlay_is_approval": False,
    }
    output.mkdir(exist_ok=False)
    with (output / "page1-witnesses.v1.jsonl").open("xb") as handle:
        handle.write(page1_output)
    with (output / "reviews.v1.jsonl").open("xb") as handle:
        handle.write(review_output)
    receipt_data = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8")
    pending_receipt = output / "receipt.json.pending"
    with pending_receipt.open("xb") as handle:
        if handle.write(receipt_data) != len(receipt_data):
            raise OSError("short receipt write")
        handle.flush()
        os.fsync(handle.fileno())
    pending_receipt.rename(output / "receipt.json")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "proof", "proof-sha256", "corpus-root", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--expected-batches", type=int, required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        result = assemble(args)
    except (OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
