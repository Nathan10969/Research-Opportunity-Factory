"""Aggregate pinned, privately QA-passed RouterResult rows without ingesting them.

Inputs are a frozen allowlist and a new run's router_jobs.jsonl. Strict v3
source rows are copied byte-for-byte; a pinned legacy row with only a verified
canonical-result hash envelope is stripped to strict v3 and its original
physical row remains in provenance. No model, source fetch, or run write occurs.
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

from .corpus import RouterResult, _canonical_result_hash


SHA = re.compile(r"[0-9a-f]{64}\Z")
ALLOWLIST_SCHEMA = "engineering.old811_card_pilot100_strict_v3_input_allowlist.v1"
RESULT_SCHEMA = "idea_factory.corpus_router_result.v3"
REPARSE_POINT = 0x0400


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact_file(path: Path) -> Path:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError(f"noncanonical or missing path: {path}")
    for part in (path, *path.parents):
        if part.is_symlink() or (os.name == "nt" and part.lstat().st_file_attributes & REPARSE_POINT):
            raise ValueError(f"symlink or reparse path: {path}")
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    return path


def _pinned(path: Path, digest: str) -> bytes:
    if not isinstance(digest, str) or not SHA.fullmatch(digest):
        raise ValueError(f"invalid SHA-256 pin: {path}")
    data = _exact_file(path).read_bytes()
    if _sha(data) != digest:
        raise ValueError(f"file SHA-256 mismatch: {path}")
    return data


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if type(value) is not dict:
        raise ValueError("expected JSON object")
    return value


def _lines(data: bytes) -> list[bytes]:
    if not data or not data.endswith(b"\n"):
        raise ValueError("JSONL must be nonempty and LF terminated")
    rows = data.split(b"\n")[:-1]
    if any(not row.strip() for row in rows):
        raise ValueError("blank JSONL row")
    return rows


def _physical_row(data: bytes, number: int, digest: str) -> tuple[bytes, dict[str, Any]]:
    rows = _lines(data)
    if type(number) is not int or number < 1 or number > len(rows):
        raise ValueError("invalid physical line number")
    row = rows[number - 1]
    if not isinstance(digest, str) or _sha(row) != digest:
        raise ValueError("physical row SHA-256 mismatch")
    return row, _json(row)


def _qa_pass(qa: dict[str, Any], slug: str) -> bool:
    gates = [qa.get(key) for key in ("verdict", "disposition", "status", "qa_verdict")]
    top_pass = any(isinstance(gate, str) and gate.startswith("PASS") for gate in gates)
    item_seen = False
    for key in ("per_item", "items", "item_dispositions", "per_result"):
        items = qa.get(key)
        if isinstance(items, dict):
            items = list(items.values())
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and item.get("slug") == slug:
                disposition = next((item.get(name) for name in ("status", "verdict", "disposition")
                                    if isinstance(item.get(name), str)), None)
                if disposition is not None and not disposition.startswith("PASS"):
                    return False
                if disposition is not None:
                    item_seen = True
    return top_pass or item_seen


def _write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as handle:
        if handle.write(data) != len(data):
            raise OSError(f"short write: {path}")
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != data:
        raise OSError(f"write readback mismatch: {path}")


def adapt_legacy_router_envelope(result: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Strip only a verified canonical RouterResult hash, never scientific fields."""
    required = set(RouterResult.model_fields)
    keys = set(result)
    if keys == required:
        RouterResult.model_validate(result)
        return result, None
    if keys != required | {"raw_result_sha256"}:
        raise ValueError("extra or missing RouterResult envelope keys")
    provenance = result["raw_result_sha256"]
    body = {key: value for key, value in result.items() if key != "raw_result_sha256"}
    parsed = RouterResult.model_validate(body)
    if type(provenance) is not str or provenance != _canonical_result_hash(parsed):
        raise ValueError("legacy RouterResult canonical hash mismatch")
    return body, provenance


def aggregate(allowlist: Path, allowlist_sha256: str, router_jobs: Path,
              router_jobs_sha256: str, output_dir: Path) -> dict[str, Any]:
    """Preflight the exact job set and every pin, then publish a private snapshot."""
    allowlist, router_jobs, output_dir = Path(allowlist), Path(router_jobs), Path(output_dir)
    allowed = [_json(line) for line in _lines(_pinned(allowlist, allowlist_sha256))]
    jobs = [_json(line) for line in _lines(_pinned(router_jobs, router_jobs_sha256))]
    run = router_jobs.parent.parent if router_jobs.parent.name == "corpus" else router_jobs.parent
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if (not output_dir.is_absolute() or output_dir.parent.resolve(strict=True) != output_dir.parent
            or output_dir.is_relative_to(run)):
        raise ValueError("output must be a new canonical private directory outside run")
    if len(allowed) != len(jobs):
        raise ValueError("allowlist and new job set size differ")
    indexed: dict[str, dict[str, Any]] = {}
    slugs: set[str] = set()
    for row in allowed:
        if (row.get("schema_version") != ALLOWLIST_SCHEMA or
            row.get("status") != "INPUT_PRECHECK_PASS_SCIENCE_PENDING" or
            row.get("hold_reason") is not None or
            row.get("private_qa_scope") not in {"PASS", "BATCH_PASS"}):
            raise ValueError("allowlist row is not private QA-pass input")
        jid, slug = row.get("router_job_id"), row.get("slug")
        if not isinstance(jid, str) or not isinstance(slug, str) or jid in indexed or slug in slugs:
            raise ValueError("duplicate or invalid allowlist identity")
        indexed[jid] = row
        slugs.add(slug)
    if len({job.get("job_id") for job in jobs}) != len(jobs) or set(indexed) != {job.get("job_id") for job in jobs}:
        raise ValueError("new job set does not exactly equal allowlist job set")
    candidate_rows: list[bytes] = []
    custody: list[dict[str, Any]] = []
    for job in jobs:
        row = indexed[job["job_id"]]
        binding = {"job_id": row["router_job_id"], "slug": row["slug"],
                   "note_sha256": row["note_sha256"], "prompt_sha256": row["router_prompt_sha256"],
                   "note_path": row["note_path"], "prompt_path": row["router_prompt_path"]}
        if any(job.get(key) != value for key, value in binding.items()):
            raise ValueError(f"new job identity or note/prompt binding mismatch: {row['slug']}")
        _pinned(Path(row["note_path"]), row["note_sha256"])
        _pinned(Path(row["router_prompt_path"]), row["router_prompt_sha256"])
        old_job_data = _pinned(Path(row["old_router_job_path"]), row["old_router_job_file_sha256"])
        _, old_job = _physical_row(old_job_data, row["old_router_job_line"],
                                   row["old_router_job_row_sha256"])
        if any(old_job.get(key) != value for key, value in binding.items()):
            raise ValueError(f"old job binding mismatch: {row['slug']}")
        qa = _json(_pinned(Path(row["private_qa_path"]), row["private_qa_sha256"]))
        if not _qa_pass(qa, row["slug"]):
            raise ValueError(f"old QA is not PASS for {row['slug']}")
        source_data = _pinned(Path(row["private_router_result_path"]),
                              row["private_router_result_file_sha256"])
        raw, result = _physical_row(source_data, row["private_router_result_line"],
                                    row["private_router_result_row_sha256"])
        if result.get("schema_version") != RESULT_SCHEMA:
            raise ValueError(f"not a v3 RouterResult: {row['slug']}")
        adapted, legacy_hash = adapt_legacy_router_envelope(result)
        if "legacy_router_canonical_sha256" in row and row["legacy_router_canonical_sha256"] != legacy_hash:
            raise ValueError(f"legacy RouterResult provenance pin mismatch: {row['slug']}")
        if any(adapted.get(key) != value for key, value in binding.items()
               if key in {"job_id", "slug", "note_sha256", "prompt_sha256"}):
            raise ValueError(f"old result binding mismatch: {row['slug']}")
        candidate_rows.append((raw if legacy_hash is None else
                               json.dumps(adapted, ensure_ascii=False, allow_nan=False,
                                          sort_keys=True, separators=(",", ":")).encode("utf-8")) + b"\n")
        custody.append({"job_id": job["job_id"], "slug": row["slug"],
                        "source_path": row["private_router_result_path"],
                        "source_file_sha256": row["private_router_result_file_sha256"],
                        "source_line": row["private_router_result_line"],
                        "source_row_sha256": row["private_router_result_row_sha256"],
                        "legacy_router_canonical_sha256": legacy_hash,
                        "qa_path": row["private_qa_path"], "qa_sha256": row["private_qa_sha256"],
                        "qa_scope": row["private_qa_scope"]})
    # Detect drift from concurrent modification before any output is published.
    _pinned(allowlist, allowlist_sha256)
    _pinned(router_jobs, router_jobs_sha256)
    for row in allowed:
        for path_key, hash_key in (("note_path", "note_sha256"),
                                   ("router_prompt_path", "router_prompt_sha256"),
                                   ("old_router_job_path", "old_router_job_file_sha256"),
                                   ("private_qa_path", "private_qa_sha256"),
                                   ("private_router_result_path", "private_router_result_file_sha256")):
            _pinned(Path(row[path_key]), row[hash_key])
    candidate = b"".join(candidate_rows)
    receipt = {"schema_version": "engineering.private_router_aggregate_receipt.v1",
               "allowlist_path": str(allowlist), "allowlist_sha256": allowlist_sha256,
               "router_jobs_path": str(router_jobs), "router_jobs_sha256": router_jobs_sha256,
               "candidate_count": len(candidate_rows), "candidate_sha256": _sha(candidate),
               "custody": custody, "ingested": False, "scientific_approval": False,
               "source_admission_approved": False, "human_approved": False,
               "graph_ingested": False, "paid_calls_made": False}
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + ".staging-", dir=output_dir.parent))
    _write_new(staging / "router_results.candidate.v3.jsonl", candidate)
    receipt_bytes = json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    _write_new(staging / "receipt.v1.json", receipt_bytes)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    os.rename(staging, output_dir)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--allowlist-sha256", required=True)
    parser.add_argument("--router-jobs", type=Path, required=True)
    parser.add_argument("--router-jobs-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = aggregate(args.allowlist, args.allowlist_sha256, args.router_jobs,
                            args.router_jobs_sha256, args.output_dir)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f"private RouterResult aggregation HOLD: {exc}\n")
    print(json.dumps({"candidate_count": receipt["candidate_count"],
                      "candidate_sha256": receipt["candidate_sha256"],
                      "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
