"""Read-only, hash-bound transport of private Card objects into v3 job wrappers.

This module does not ingest results, judge scientific claims, or call a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .cards import _canonical_hash, _job_rows, _validate_card


SHA = re.compile(r"[0-9a-f]{64}\Z")
RUN_FILES = {
    "jobs": "cards/card_jobs.jsonl",
    "selection": "corpus/selection_manifest.jsonl",
    "rejected": "corpus/rejected_manifest.jsonl",
    "router_jobs": "corpus/router_jobs.jsonl",
    "router_results": "corpus/router_results.jsonl",
    "selection_policy": "corpus/selection_policy.jsonl",
}
ROW_KEYS = {
    "job_id", "slug", "old_raw", "old_task", "old_review", "old_schema_validated",
    "note", "prompt", "pdf", "source_record_id", "source_version", "science_gate",
    "science_hold_reason",
}
PIN_KEYS = {"path", "sha256"}
REPARSE_POINT = 0x0400


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(path: Path) -> Path:
    if not path.is_absolute() or str(path.resolve(strict=True)) != str(path):
        raise ValueError(f"path alias or missing path: {path}")
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink() or (os.name == "nt" and ancestor.lstat().st_file_attributes & REPARSE_POINT):
            raise ValueError(f"symlink/reparse path: {path}")
    return path


def _read_pin(pin: object, *, expected: Path | None = None) -> bytes:
    if type(pin) is not dict or set(pin) != PIN_KEYS:
        raise ValueError("pin must have exact path and sha256 keys")
    path_raw, digest = pin["path"], pin["sha256"]
    if type(path_raw) is not str or type(digest) is not str or not SHA.fullmatch(digest):
        raise ValueError("invalid path or SHA-256 pin")
    path = _canonical(Path(path_raw))
    if expected is not None and path != expected:
        raise ValueError(f"pinned path is not expected path: {path}")
    if not path.is_file():
        raise ValueError(f"pinned input is not a file: {path}")
    data = path.read_bytes()
    if _sha(data) != digest:
        raise ValueError(f"pinned SHA-256 mismatch: {path}")
    return data


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _json(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_keys,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {value}")))
    if type(value) is not dict:
        raise ValueError("expected JSON object")
    return value


def _jsonl(data: bytes) -> list[dict[str, Any]]:
    if data and not data.endswith(b"\n"):
        raise ValueError("JSONL must end in newline")
    return [_json(line) for line in data.splitlines()]


def _row_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False,
                                 sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)


def _write_verified(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as handle:
        count = handle.write(data)
        if count != len(data):
            raise OSError(f"short output write: {path.name}")
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != data:
        raise OSError(f"output readback mismatch: {path.name}")


def _check_row(row: dict[str, Any], job: dict[str, Any], selected: dict[str, Any], data: dict[str, bytes]) -> tuple[dict[str, Any], str]:
    raw = _json(data["old_raw"])
    task = _json(data["old_task"])
    review = _json(data["old_review"])
    validated = _json(data["old_schema_validated"])
    if set(raw) != {"schema_version", "slug", "note_sha256", "prompt_sha256", "cards", "empty_reason"}:
        raise ValueError("old raw has extra or missing keys")
    if raw["schema_version"] != "idea_factory.bulk_card_response.v1" or raw["empty_reason"] != "":
        raise ValueError("old raw is not a nonempty bulk Card response")
    if type(raw["cards"]) is not list or not raw["cards"]:
        raise ValueError("old raw has no Card objects")
    if (type(task.get("shard_id")) is not str or not task["shard_id"] or
        raw["slug"] != task.get("slug") or raw["slug"] != task["shard_id"] + "-" + job["slug"]):
        raise ValueError("old worker slug binding mismatch")
    if raw["note_sha256"] != job["note_sha256"] or raw["prompt_sha256"] != job["prompt_sha256"]:
        raise ValueError("old raw note/prompt binding mismatch")
    expected = {
        "note_path": job["note_path"], "note_sha256": job["note_sha256"],
        "prompt_path": row["prompt"]["path"], "prompt_sha256": job["prompt_sha256"],
        "source_pdf_path": row["pdf"]["path"], "source_pdf_sha256": row["pdf"]["sha256"],
        "output_path": row["old_raw"]["path"],
    }
    if any(task.get(key) != value for key, value in expected.items()):
        raise ValueError("old task note/prompt/PDF/raw binding mismatch")
    if data["prompt"] != job["prompt_text"].encode("utf-8"):
        raise ValueError("old and new prompt bytes differ")
    if data["note"] != job["note_text"].encode("utf-8"):
        raise ValueError("old and new note bytes differ")
    primary = selected.get("primary_source_record")
    if type(row["source_record_id"]) is not str or type(row["source_version"]) is not str or not row["source_record_id"].endswith(row["source_version"]):
        raise ValueError("invalid source record ID/version binding")
    if primary is not None and (primary.get("primary_pdf_path") != row["pdf"]["path"] or
                                primary.get("primary_pdf_sha256") != row["pdf"]["sha256"] or
                                row["source_record_id"] not in primary.get("source_record_ids", []) or
                                primary.get("source_version") != row["source_version"]):
        raise ValueError("selected source identity/PDF mismatch")
    aliases = task.get("source_aliases")
    if primary is None and aliases is None:
        raise ValueError("source record identity has no independent old/new evidence")
    if aliases is not None and not any(
        f"arxiv:{alias.get('arxiv_id')}" == row["source_record_id"] for alias in aliases if type(alias) is dict
    ):
        raise ValueError("old task source record identity mismatch")
    if (review.get("raw_path") != row["old_raw"]["path"] or
        review.get("raw_sha256") != row["old_raw"]["sha256"] or
        review.get("note_path") != job["note_path"] or
        review.get("note_sha256") != job["note_sha256"] or
        review.get("source_pdf", {}).get("path") != row["pdf"]["path"] or
        review.get("source_pdf", {}).get("sha256") != row["pdf"]["sha256"] or
        review.get("terminal_disposition") != "SCHEMA_VALID" or
        review.get("validator_status") not in (None, "SCHEMA_VALID")):
        raise ValueError("old review binding/status mismatch")
    if (validated.get("schema_version") != "idea_factory.bulk_schema_validated_card.v1" or
        validated.get("slug") != raw["slug"] or
        validated.get("note_path") != job["note_path"] or
        validated.get("note_sha256") != job["note_sha256"] or
        validated.get("prompt_sha256") != job["prompt_sha256"] or
        validated.get("raw_result_sha256") != row["old_raw"]["sha256"]):
        raise ValueError("old schema_validated binding mismatch")
    if len(raw["cards"]) != 1 or validated.get("card") != raw["cards"][0]:
        raise ValueError("old schema_validated Card body mismatch")
    card_ids = []
    for card in raw["cards"]:
        _validate_card(card, job)
        card_ids.append(card["card_id"])
    if task.get("card_id") not in card_ids or validated.get("record_id") not in card_ids:
        raise ValueError("old Card ID binding mismatch")
    body_hash = _canonical_hash(raw["cards"][0])
    wrapper = {key: job[key] for key in ("job_id", "slug", "note_sha256", "prompt_sha256")}
    wrapper.update(schema_version="idea_factory.paper_card_result.v1", cards=raw["cards"])
    if _canonical_hash(wrapper["cards"][0]) != body_hash:
        raise ValueError("Card object changed during transport")
    return wrapper, body_hash


def transport(run: Path, allowlist: Path, allowlist_sha256: str, output_dir: Path) -> dict[str, Any]:
    """Preflight a pinned batch, then publish a private candidate and HOLD ledger."""
    run = _canonical(Path(run))
    allowlist = _canonical(Path(allowlist))
    if type(allowlist_sha256) is not str or not SHA.fullmatch(allowlist_sha256):
        raise ValueError("invalid allowlist SHA-256")
    manifest_bytes = allowlist.read_bytes()
    if _sha(manifest_bytes) != allowlist_sha256:
        raise ValueError("allowlist SHA-256 mismatch")
    manifest = _json(manifest_bytes)
    if set(manifest) != {"schema_version", "run_path", "run_files", "rows"} or manifest["schema_version"] != "private_card_transport_allowlist.v1" or manifest["run_path"] != str(run):
        raise ValueError("invalid or wrong-run allowlist")
    if type(manifest["run_files"]) is not dict or set(manifest["run_files"]) != set(RUN_FILES):
        raise ValueError("incomplete run file pins")
    pinned: list[dict[str, str]] = []
    for name, rel in RUN_FILES.items():
        pin = manifest["run_files"][name]
        _read_pin(pin, expected=run / rel)
        pinned.append(pin)
    _, jobs = _job_rows(run / RUN_FILES["jobs"])
    selected = _jsonl(_read_pin(manifest["run_files"]["selection"]))
    by_slug = {row["slug"]: row for row in selected}
    if type(manifest["rows"]) is not list or len(manifest["rows"]) != len(jobs):
        raise ValueError("allowlist does not exactly cover jobs")
    by_job = {job["job_id"]: job for job in jobs}
    seen_jobs: set[str] = set()
    input_pins: list[dict[str, str]] = [*pinned]
    ready: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    card_ids: set[str] = set()
    for row in manifest["rows"]:
        if type(row) is not dict or set(row) != ROW_KEYS or row["job_id"] not in by_job or row["job_id"] in seen_jobs:
            raise ValueError("invalid, extra, or duplicate allowlist job")
        seen_jobs.add(row["job_id"])
        job = by_job[row["job_id"]]
        if row["slug"] != job["slug"] or row["science_gate"] not in {"PASS", "HOLD", "UNREVIEWED"} or type(row["science_hold_reason"]) is not str or (row["science_gate"] == "HOLD") != bool(row["science_hold_reason"].strip()):
            raise ValueError("allowlist job or science gate mismatch")
        expected_paths = {"note": Path(job["note_path"])}
        data: dict[str, bytes] = {}
        for name in ("old_raw", "old_task", "old_review", "old_schema_validated", "note", "prompt", "pdf"):
            data[name] = _read_pin(row[name], expected=expected_paths.get(name))
            input_pins.append(row[name])
        status, reason, body_hash = "READY", "", None
        try:
            wrapper, body_hash = _check_row(row, job, by_slug[job["slug"]], data)
            for card in wrapper["cards"]:
                card_id = card["card_id"]
                if card_id in card_ids:
                    raise ValueError(f"duplicate Card ID: {card_id}")
                card_ids.add(card_id)
            if row["science_gate"] == "HOLD":
                status, reason = "HOLD", row["science_hold_reason"]
            else:
                ready.append(wrapper)
        except (ValueError, TypeError, KeyError) as exc:
            status, reason = "HOLD", str(exc)
        ledger.append({"job_id": job["job_id"], "slug": job["slug"], "status": status,
                       "reason": reason, "science_gate": row["science_gate"],
                       "card_body_sha256": body_hash,
                       "input_pins": {name: row[name] for name in data}})
    if seen_jobs != set(by_job):
        raise ValueError("allowlist job set mismatch")
    if _sha(allowlist.read_bytes()) != allowlist_sha256:
        raise ValueError("allowlist drift before output")
    for pin in input_pins:
        _read_pin(pin)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not output_dir.is_absolute() or output_dir.parent.resolve(strict=True) != output_dir.parent or output_dir.is_relative_to(run):
        raise ValueError("output must be an exact external private directory")
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + ".staging-", dir=output_dir.parent))
    candidate_bytes, ledger_bytes = _row_bytes(ready), _row_bytes(ledger)
    candidate = staging / "paper_card_results.candidate.v1.jsonl"
    holds = staging / "hold_ledger.v1.jsonl"
    _write_verified(candidate, candidate_bytes)
    _write_verified(holds, ledger_bytes)
    receipt = {
        "schema_version": "private_card_transport_receipt.v1", "run_path": str(run),
        "allowlist_path": str(allowlist), "allowlist_sha256": allowlist_sha256,
        "run_files": manifest["run_files"], "requested_count": len(jobs),
        "ready_count": len(ready), "hold_count": len(ledger) - len(ready),
        "candidate_sha256": _sha(candidate_bytes), "ledger_sha256": _sha(ledger_bytes),
        "scientific_entailment_audited": False, "human_approved": False,
        "graph_ingested": False, "paid_calls_made": False,
    }
    receipt_bytes = json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    temp = staging / ".receipt.v1.json.tmp"
    _write_verified(temp, receipt_bytes)
    os.rename(temp, staging / "receipt.v1.json")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    os.rename(staging, output_dir)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--allowlist", required=True, type=Path)
    parser.add_argument("--allowlist-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = transport(args.run, args.allowlist, args.allowlist_sha256, args.output_dir)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f"private transport HOLD: {exc}\n")
    print(json.dumps({"ready_count": result["ready_count"], "hold_count": result["hold_count"],
                      "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
