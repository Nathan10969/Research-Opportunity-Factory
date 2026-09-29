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
OLD_ROW_COMMON = {
    "origin_format", "job_id", "slug", "old_task", "old_raw", "old_review",
    "old_receipt", "note", "prompt", "pdf", "source_record_id",
    "source_version", "science_gate", "science_hold_reason", "staging_line",
    "staging_row_sha256",
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


def _check_old_corpus_row(
    row: dict[str, Any], job: dict[str, Any], selected: dict[str, Any],
    staging_row: dict[str, Any], data: dict[str, bytes],
) -> tuple[dict[str, Any], str]:
    """Validate one old-corpus Card against its frozen source row and new job."""
    task, raw = _json(data["old_task"]), _json(data["old_raw"])
    receipt = _json(data["old_receipt"])
    review = _json(data["old_review"])
    if (staging_row.get("status") != "INPUT_PRECHECK_PASS_SCIENCE_PENDING" or
        staging_row.get("science_approved") is not False or
        staging_row.get("slug") != job["slug"] or row["slug"] != job["slug"] or
        row["science_gate"] != "UNREVIEWED"):
        raise ValueError("old staging identity/science gate mismatch")
    source = selected.get("primary_source_record")
    if (type(source) is not dict or
        source.get("primary_pdf_path") != staging_row.get("pdf_path") or
        source.get("primary_pdf_sha256") != staging_row.get("pdf_sha256") or
        source.get("source_record_ids") != staging_row.get("source_record_ids") or
        source.get("source_version") != staging_row.get("source_version") or
        row["source_record_id"] not in staging_row.get("source_record_ids", []) or
        row["source_version"] != staging_row.get("source_version")):
        raise ValueError("old staging and new selected source/PDF disagree")
    for name, prefix in (
        ("old_task", "task"), ("old_raw", "raw"), ("old_review", "review"),
        ("old_receipt", "card_receipt"), ("note", "note"), ("pdf", "pdf"),
    ):
        if (row[name]["path"] != staging_row[f"{prefix}_path"] or
            row[name]["sha256"] != staging_row[f"{prefix}_sha256"]):
            raise ValueError(f"old staging {name} pin mismatch")
    if (task.get("schema_version") != "idea_factory.bulk_card_task.v1" or
        task.get("slug") != job["slug"] or task.get("note_path") != job["note_path"] or
        task.get("note_sha256") != job["note_sha256"] or
        task.get("prompt_path") != row["prompt"]["path"] or
        task.get("prompt_sha256") != job["prompt_sha256"] or
        task.get("output_path") != row["old_raw"]["path"]):
        raise ValueError("old task/new job binding mismatch")
    if (row["prompt"]["sha256"] != job["prompt_sha256"] or
        data["prompt"] != job["prompt_text"].encode("utf-8") or
        data["note"] != job["note_text"].encode("utf-8")):
        raise ValueError("old and new prompt/note bytes differ")
    if (set(raw) != {"schema_version", "slug", "note_sha256", "prompt_sha256", "cards", "empty_reason"} or
        raw["schema_version"] != "idea_factory.bulk_card_response.v1" or
        raw["slug"] != job["slug"] or raw["note_sha256"] != job["note_sha256"] or
        raw["prompt_sha256"] != job["prompt_sha256"] or raw["empty_reason"] != "" or
        type(raw["cards"]) is not list or len(raw["cards"]) != 1):
        raise ValueError("old raw Card envelope mismatch")
    if (receipt.get("status") != "SCHEMA_VALID" or receipt.get("slug") != job["slug"] or
        receipt.get("shard_id") != task.get("shard_id") or receipt.get("errors") != [] or
        receipt.get("accepted_cards") != 1 or
        receipt.get("raw_result_sha256") != row["old_raw"]["sha256"]):
        raise ValueError("old structural receipt mismatch")
    if row["origin_format"] == "old_worker_v1":
        validated = _json(data["old_schema_validated"])
        if (row["old_schema_validated"]["path"] != str(Path(row["old_task"]["path"]).with_name("schema_validated.json")) or
            validated.get("schema_version") != "idea_factory.bulk_schema_validated_card.v1" or
            validated.get("slug") != job["slug"] or
            validated.get("note_path") != job["note_path"] or
            validated.get("note_sha256") != job["note_sha256"] or
            validated.get("prompt_sha256") != job["prompt_sha256"] or
            validated.get("raw_result_sha256") != row["old_raw"]["sha256"] or
            validated.get("card") != raw["cards"][0] or
            validated.get("record_id") != raw["cards"][0].get("card_id")):
            raise ValueError("old schema_validated Card binding mismatch")
        if (review.get("item_id", job["slug"]) not in (None, job["slug"]) or
            review.get("slug", job["slug"]) not in (None, job["slug"]) or
            review.get("note_path", job["note_path"]) not in (None, job["note_path"]) or
            review.get("note_sha256", job["note_sha256"]) not in (None, job["note_sha256"]) or
            review.get("reviewed_raw_path", row["old_raw"]["path"]) not in (None, row["old_raw"]["path"])):
            raise ValueError("old review provenance mismatch")
    elif row["origin_format"] == "reused_old_corpus_v1":
        reuse = _json(data["old_reuse"])
        qa = _json(data["old_reuse_qa"])
        validated = reuse.get("card")
        if (row["old_reuse"]["path"] != staging_row.get("reuse_path") or
            row["old_reuse"]["sha256"] != staging_row.get("reuse_sha256") or
            type(validated) is not dict or
            reuse.get("item_id") != job["slug"] or
            reuse.get("reused_task_path") != row["old_task"]["path"] or
            reuse.get("reused_task_sha256") != row["old_task"]["sha256"] or
            reuse.get("derived_raw_path") != row["old_raw"]["path"] or
            reuse.get("derived_raw_sha256") != row["old_raw"]["sha256"] or
            reuse.get("review_path") != row["old_review"]["path"] or
            reuse.get("review_sha256") != row["old_review"]["sha256"] or
            reuse.get("qa_report_path") != row["old_reuse_qa"]["path"] or
            reuse.get("qa_report_sha256") != row["old_reuse_qa"]["sha256"] or
            reuse.get("adapter_status") != "SCHEMA_VALID" or
            validated.get("schema_version") != "idea_factory.bulk_schema_validated_card.v1" or
            validated.get("slug") != job["slug"] or
            validated.get("note_path") != job["note_path"] or
            validated.get("note_sha256") != job["note_sha256"] or
            validated.get("prompt_sha256") != job["prompt_sha256"] or
            validated.get("raw_result_sha256") != row["old_raw"]["sha256"] or
            validated.get("card") != raw["cards"][0] or
            validated.get("record_id") != raw["cards"][0].get("card_id")):
            raise ValueError("reused old Card/QA binding mismatch")
        qa_items = qa.get("delta_items")
        matches = [item for item in qa_items if type(item) is dict and item.get("slug") == job["slug"]] if type(qa_items) is list else []
        if (qa.get("errors") != [] or len(matches) != 1 or
            matches[0].get("review_status") != "PASS" or
            matches[0].get("independent_pending") is not False or
            matches[0].get("adapter_status") != "SCHEMA_VALID" or
            matches[0].get("derived_raw_sha256") != row["old_raw"]["sha256"] or
            matches[0].get("review_path") != row["old_review"]["path"] or
            matches[0].get("note_sha256") != job["note_sha256"] or
            matches[0].get("prompt_sha256") != job["prompt_sha256"] or
            matches[0].get("conflict") not in (None, False) or
            matches[0].get("errors", []) != []):
            raise ValueError("reused QA has no exact-item PASS without pending/conflict")
    else:
        raise ValueError("unsupported old-corpus origin format")
    card = raw["cards"][0]
    _validate_card(card, job)
    if task.get("card_id") != card["card_id"]:
        raise ValueError("old task Card ID mismatch")
    body_hash = _canonical_hash(card)
    wrapper = {key: job[key] for key in ("job_id", "slug", "note_sha256", "prompt_sha256")}
    wrapper.update(schema_version="idea_factory.paper_card_result.v1", cards=[card])
    if _canonical_hash(wrapper["cards"][0]) != body_hash:
        raise ValueError("old Card body changed during transport")
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
    version = manifest.get("schema_version")
    expected_manifest_keys = {"schema_version", "run_path", "run_files", "rows"}
    if version == "private_card_transport_allowlist.v2":
        expected_manifest_keys.add("source_staging")
    if set(manifest) != expected_manifest_keys or version not in {"private_card_transport_allowlist.v1", "private_card_transport_allowlist.v2"} or manifest["run_path"] != str(run):
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
    staging_lines: list[bytes] = []
    if version == "private_card_transport_allowlist.v2":
        staging_bytes = _read_pin(manifest["source_staging"])
        staging_lines = staging_bytes.splitlines(keepends=True)
        if len(staging_lines) != len(jobs) or any(not line.endswith(b"\n") for line in staging_lines):
            raise ValueError("old-corpus staging does not exactly cover jobs")
    if type(manifest["rows"]) is not list or len(manifest["rows"]) != len(jobs):
        raise ValueError("allowlist does not exactly cover jobs")
    by_job = {job["job_id"]: job for job in jobs}
    seen_jobs: set[str] = set()
    ready: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    card_ids: set[str] = set()
    for row in manifest["rows"]:
        if type(row) is not dict or row.get("job_id") not in by_job or row["job_id"] in seen_jobs:
            raise ValueError("invalid, extra, or duplicate allowlist job")
        if version == "private_card_transport_allowlist.v1":
            expected_row_keys = ROW_KEYS
        elif row.get("origin_format") == "old_worker_v1":
            expected_row_keys = OLD_ROW_COMMON | {"old_schema_validated"}
        elif row.get("origin_format") == "reused_old_corpus_v1":
            expected_row_keys = OLD_ROW_COMMON | {"old_reuse", "old_reuse_qa"}
        else:
            raise ValueError("unsupported old-corpus origin format")
        if set(row) != expected_row_keys:
            raise ValueError("allowlist row has extra or missing keys")
        seen_jobs.add(row["job_id"])
        job = by_job[row["job_id"]]
        if row["slug"] != job["slug"] or row["science_gate"] not in {"PASS", "HOLD", "UNREVIEWED"} or type(row["science_hold_reason"]) is not str or (row["science_gate"] == "HOLD") != bool(row["science_hold_reason"].strip()):
            raise ValueError("allowlist job or science gate mismatch")
        expected_paths = {"note": Path(job["note_path"])}
        pin_names = ("old_raw", "old_task", "old_review", "note", "prompt", "pdf")
        if version == "private_card_transport_allowlist.v1":
            pin_names += ("old_schema_validated",)
        elif row["origin_format"] == "old_worker_v1":
            pin_names += ("old_schema_validated", "old_receipt")
        else:
            pin_names += ("old_receipt", "old_reuse", "old_reuse_qa")
        data: dict[str, bytes] = {}
        status, reason, body_hash = "READY", "", None
        try:
            for name in pin_names:
                data[name] = _read_pin(row[name], expected=expected_paths.get(name))
            if version == "private_card_transport_allowlist.v2":
                line_number = row["staging_line"]
                if type(line_number) is not int or not 1 <= line_number <= len(staging_lines):
                    raise ValueError("old-corpus staging line is invalid")
                raw_line = staging_lines[line_number - 1]
                if _sha(raw_line) != row["staging_row_sha256"]:
                    raise ValueError("old-corpus staging row SHA mismatch")
                wrapper, body_hash = _check_old_corpus_row(
                    row, job, by_slug[job["slug"]], _json(raw_line), data,
                )
            else:
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
        except (ValueError, TypeError, KeyError, OSError, AttributeError) as exc:
            status, reason = "HOLD", str(exc)
        ledger.append({"job_id": job["job_id"], "slug": job["slug"], "status": status,
                       "reason": reason, "science_gate": row["science_gate"],
                       "card_body_sha256": body_hash,
                       "input_pins": {name: row[name] for name in pin_names}})
    if seen_jobs != set(by_job):
        raise ValueError("allowlist job set mismatch")
    if _sha(allowlist.read_bytes()) != allowlist_sha256:
        raise ValueError("allowlist drift before output")
    for pin in pinned:
        _read_pin(pin)
    if version == "private_card_transport_allowlist.v2":
        _read_pin(manifest["source_staging"])
    ready_by_job = {wrapper["job_id"]: wrapper for wrapper in ready}
    for entry in ledger:
        if entry["status"] != "READY":
            continue
        try:
            for name, pin in entry["input_pins"].items():
                _read_pin(pin, expected=Path(by_job[entry["job_id"]]["note_path"]) if name == "note" else None)
        except (ValueError, TypeError, OSError) as exc:
            entry["status"] = "HOLD"
            entry["reason"] = f"pin drift before publish: {exc}"
            ready_by_job.pop(entry["job_id"])
    ready = [wrapper for wrapper in ready if wrapper["job_id"] in ready_by_job]
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
        **({"source_staging": manifest["source_staging"]} if version == "private_card_transport_allowlist.v2" else {}),
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
