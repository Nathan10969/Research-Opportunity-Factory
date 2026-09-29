"""Build a pinned, science-unreviewed Card transport allowlist for old corpus jobs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .cards import _job_rows
from .private_card_transport import (
    RUN_FILES, SHA, _canonical, _check_old_corpus_row, _json, _read_pin,
    _row_bytes, _sha, _write_verified,
)


def _pin(path: object, digest: object) -> dict[str, str] | None:
    if type(path) is not str or type(digest) is not str or not SHA.fullmatch(digest):
        return None
    return {"path": path, "sha256": digest}


def _actual_pin(path: Path) -> dict[str, str] | None:
    try:
        path = _canonical(path)
        return {"path": str(path), "sha256": _sha(path.read_bytes())}
    except (OSError, ValueError):
        return None


def _row(source: dict[str, Any], job: dict[str, Any], line: int, line_sha: str) -> dict[str, Any]:
    reused = source.get("reuse_path") is not None
    task_pin = _pin(source.get("task_path"), source.get("task_sha256"))
    task = None
    if task_pin is not None:
        try:
            task = _json(_read_pin(task_pin))
        except (OSError, ValueError):
            pass
    prompt = _actual_pin(Path(task["prompt_path"])) if task and type(task.get("prompt_path")) is str else None
    row: dict[str, Any] = {
        "origin_format": "reused_old_corpus_v1" if reused else "old_worker_v1",
        "job_id": job["job_id"], "slug": job["slug"],
        "old_task": task_pin, "old_raw": _pin(source.get("raw_path"), source.get("raw_sha256")),
        "old_review": _pin(source.get("review_path"), source.get("review_sha256")),
        "old_receipt": _pin(source.get("card_receipt_path"), source.get("card_receipt_sha256")),
        "note": _pin(source.get("note_path"), source.get("note_sha256")),
        "prompt": prompt, "pdf": _pin(source.get("pdf_path"), source.get("pdf_sha256")),
        "source_record_id": source["source_record_ids"][0],
        "source_version": source.get("source_version"),
        "science_gate": "UNREVIEWED", "science_hold_reason": "",
        "staging_line": line, "staging_row_sha256": line_sha,
    }
    if reused:
        row["old_reuse"] = _pin(source.get("reuse_path"), source.get("reuse_sha256"))
        reuse = None
        if row["old_reuse"] is not None:
            try:
                reuse = _json(_read_pin(row["old_reuse"]))
            except (OSError, ValueError):
                pass
        row["old_reuse_qa"] = _pin(reuse.get("qa_report_path"), reuse.get("qa_report_sha256")) if reuse else None
    else:
        row["old_schema_validated"] = _actual_pin(Path(source["task_path"]).with_name("schema_validated.json"))
    return row


def build_old_corpus_allowlist(run: Path, staging: Path, staging_sha256: str, output_dir: Path) -> dict[str, Any]:
    run, staging = _canonical(Path(run)), _canonical(Path(staging))
    if type(staging_sha256) is not str or not SHA.fullmatch(staging_sha256):
        raise ValueError("invalid staging SHA-256")
    staged_bytes = staging.read_bytes()
    if _sha(staged_bytes) != staging_sha256:
        raise ValueError("staging SHA-256 mismatch")
    raw_lines = staged_bytes.splitlines(keepends=True)
    if not raw_lines or any(not line.endswith(b"\n") for line in raw_lines):
        raise ValueError("frozen staging must contain newline-terminated rows")
    sources = [_json(line) for line in raw_lines]
    if len({source.get("slug") for source in sources}) != len(sources) or [source.get("ordinal") for source in sources] != list(range(1, len(sources) + 1)):
        raise ValueError("frozen staging has duplicate/missing slug or ordinal")
    run_files = {name: _actual_pin(run / relative) for name, relative in RUN_FILES.items()}
    if any(pin is None for pin in run_files.values()):
        raise ValueError("new run is missing a required bundle file")
    _, jobs = _job_rows(run / RUN_FILES["jobs"])
    if len(jobs) != len(sources) or {job["slug"] for job in jobs} != {source["slug"] for source in sources}:
        raise ValueError("new Card job set does not exactly equal frozen staging")
    selected = {_row["slug"]: _row for _row in
                (_json(line) for line in (run / RUN_FILES["selection"]).read_bytes().splitlines())}
    jobs_by_slug = {job["slug"]: job for job in jobs}
    rows = []
    ledger = []
    for line_no, (source, raw_line) in enumerate(zip(sources, raw_lines, strict=True), 1):
        job = jobs_by_slug[source["slug"]]
        row = _row(source, job, line_no, _sha(raw_line))
        rows.append(row)
        try:
            data = {name: _read_pin(row[name]) for name in (
                "old_task", "old_raw", "old_review", "old_receipt", "note", "prompt", "pdf",
                *( ("old_reuse", "old_reuse_qa") if row["origin_format"] == "reused_old_corpus_v1" else ("old_schema_validated",) ),
            )}
            _check_old_corpus_row(row, job, selected[job["slug"]], source, data)
            status, reason = "READY_UNREVIEWED", ""
        except (ValueError, TypeError, KeyError, OSError, AttributeError) as exc:
            status, reason = "HOLD", str(exc)
        ledger.append({"slug": job["slug"], "job_id": job["job_id"],
                       "origin_format": row["origin_format"], "status": status, "reason": reason})
    if _sha(staging.read_bytes()) != staging_sha256:
        raise ValueError("staging drift before publish")
    for pin in run_files.values():
        _read_pin(pin)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not output_dir.is_absolute() or output_dir.parent.resolve(strict=True) != output_dir.parent or output_dir.is_relative_to(run):
        raise ValueError("output must be a new external private directory")
    manifest = {
        "schema_version": "private_card_transport_allowlist.v2", "run_path": str(run),
        "run_files": run_files, "source_staging": {"path": str(staging), "sha256": staging_sha256},
        "rows": rows,
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    ledger_bytes = _row_bytes(ledger)
    receipt = {
        "schema_version": "private_card_allowlist_build_receipt.v1", "input_count": len(sources),
        "ready_unreviewed_count": sum(x["status"] == "READY_UNREVIEWED" for x in ledger),
        "hold_count": sum(x["status"] == "HOLD" for x in ledger),
        "source_staging": manifest["source_staging"],
        "transport_allowlist_sha256": _sha(manifest_bytes), "build_ledger_sha256": _sha(ledger_bytes),
        "scientific_entailment_audited": False, "human_approved": False,
        "graph_ingested": False, "paid_calls_made": False,
    }
    receipt_bytes = json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".staging-", dir=output_dir.parent))
    _write_verified(temporary / "transport_allowlist.v2.json", manifest_bytes)
    _write_verified(temporary / "build_ledger.v1.jsonl", ledger_bytes)
    _write_verified(temporary / "receipt.v1.json", receipt_bytes)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    os.rename(temporary, output_dir)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--staging", required=True, type=Path)
    parser.add_argument("--staging-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        receipt = build_old_corpus_allowlist(args.run, args.staging, args.staging_sha256, args.output_dir)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f"old-corpus allowlist HOLD: {exc}\n")
    print(json.dumps({"ready_unreviewed_count": receipt["ready_unreviewed_count"],
                      "hold_count": receipt["hold_count"], "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
