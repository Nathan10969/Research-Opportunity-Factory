"""Conservative, read-only import of the configured historical idea ledger."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import read_jsonl
from .corpus import CorpusRouterConfig
from .opportunities import _anchored_run, _owned_dir, _owned_file, _safe_unlink


PARSER_VERSION = "idea_factory.legacy_ledger.v2"
_NAMES = ("legacy_index.jsonl", "legacy_unparsed.md", "legacy_manifest.json")
_LIVE_HEADER = ("id", "one-line", "mechanism family", "target", "status / venue", "source")
_KILLED_HEADER = ("candidate", "mechanism + target", "covered by (verified)")
_LIVE_ID = re.compile(r"\(([^()]+)\)")
_SHORT_PREFIX = re.compile(r"^(\S+)")


@dataclass(frozen=True)
class LegacyParse:
    records: tuple[dict[str, Any], ...]
    unparsed: tuple[dict[str, Any], ...]
    errors: tuple[dict[str, Any], ...]
    source_sha256: str
    source_line_count: int


@dataclass(frozen=True)
class LegacyImportBundle:
    run: Path
    records: tuple[dict[str, Any], ...]
    unparsed: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def _clean(value: str) -> str:
    value = value.strip()
    # Markdown decoration carries no field semantics; raw_row keeps it intact.
    value = re.sub(r"(?<!\\)[`*]+", "", value)
    return " ".join(value.split())


def _cells(line: str) -> list[str] | None:
    text = line.strip()
    if not text.startswith("|") or not text.endswith("|"):
        return None
    # Escaped pipes are preserved inside their cell.  This is not a general
    # Markdown parser: recognized schemas reject any ambiguous column shape.
    cells = re.split(r"(?<!\\)\|", text[1:-1])
    return [cell.replace("\\|", "|").strip() for cell in cells]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _live_record(cells: list[str], line: int, raw: str) -> dict[str, Any]:
    if len(cells) != len(_LIVE_HEADER):
        raise ValueError(f"live ledger row has wrong column count at line {line}")
    fields = [_clean(cell) for cell in cells]
    if any(not field for field in fields):
        raise ValueError(f"live ledger row has blank required field at line {line}")
    match = _LIVE_ID.search(fields[0])
    legacy_id = _clean(match.group(1)) if match else fields[0]
    display_name = _clean(fields[0][: match.start()]) if match else fields[0]
    if not legacy_id or not display_name:
        raise ValueError(f"live ledger row has invalid identifier at line {line}")
    return {
        "schema_version": "idea_factory.legacy_record.v1",
        "record_id": legacy_id,
        "legacy_id": legacy_id,
        "display_name": display_name,
        "one_line": fields[1],
        "mechanism_family": fields[2],
        "target": fields[3],
        "status": fields[4],
        "source": fields[5],
        "evidence": fields[5],
        "source_table": "LIVE",
        "parse_confidence": "HIGH",
        "old_assumption": None,
        "failure_mechanism": None,
        "missing_capability": None,
        "target_scope": None,
        "fingerprint_available": False,
        "shape_fingerprint": None,
        "source_line": line,
        "raw_row": raw,
    }


def _killed_record(cells: list[str], line: int, raw: str) -> dict[str, Any]:
    if len(cells) != len(_KILLED_HEADER):
        raise ValueError(f"killed ledger row has wrong column count at line {line}")
    candidate, mechanism_target, covered_by = (_clean(cell) for cell in cells)
    if not candidate or not mechanism_target or not covered_by:
        raise ValueError(f"killed ledger row has blank required field at line {line}")
    match = _SHORT_PREFIX.match(candidate)
    if match is None:
        raise ValueError(f"killed ledger row has invalid identifier at line {line}")
    parts = mechanism_target.split(" / ")
    if len(parts) == 2 and all(parts):
        mechanism_family, target = parts
        confidence = "HIGH"
    else:
        # Avoid producing a fake mechanism/target split from natural prose.
        mechanism_family, target, confidence = "", "", "LOW"
    legacy_id = match.group(1)
    return {
        "schema_version": "idea_factory.legacy_record.v1",
        "record_id": legacy_id,
        "legacy_id": legacy_id,
        "display_name": candidate,
        "one_line": candidate,
        "mechanism_family": mechanism_family,
        "target": target,
        "status": "EXTERNALLY_COVERED",
        "source": covered_by,
        "evidence": covered_by,
        "source_table": "KILLED_BLOCKLIST",
        "parse_confidence": confidence,
        "old_assumption": None,
        "failure_mechanism": None,
        "missing_capability": None,
        "target_scope": None,
        "fingerprint_available": False,
        "shape_fingerprint": None,
        "source_line": line,
        "raw_row": raw,
        "raw_mechanism_target": mechanism_target,
        "_short_prefix": legacy_id,
    }


def _error(number: int, raw: str, message: str) -> dict[str, Any]:
    return {
        "line_number": number,
        "error": re.sub(r" at line \d+$", "", message),
        "line": raw,
    }


def _disambiguated_killed_id(prefix: str, candidate: str) -> str:
    remainder = candidate[len(prefix):].strip()
    slug = re.sub(r"[^a-z0-9]+", "-", remainder.casefold()).strip("-")[:32] or "candidate"
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{slug}-{digest}"


def parse_legacy_markdown(source: bytes, *, source_path: str) -> LegacyParse:
    """Parse only exact known tables; malformed rows become auditable residue."""

    del source_path  # retained in the public API; the manifest owns path binding.
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("legacy ledger must be UTF-8") from exc
    provisional: list[dict[str, Any]] = []
    parsed_lines: set[int] = set()
    errors: list[dict[str, Any]] = []
    active: str | None = None
    lines = text.splitlines()
    for number, raw in enumerate(lines, start=1):
        cells = _cells(raw)
        normalized_header = tuple(_clean(cell).lower() for cell in cells) if cells else ()
        if normalized_header == _LIVE_HEADER:
            active = "LIVE"
            parsed_lines.add(number)
            continue
        if normalized_header == _KILLED_HEADER:
            active = "KILLED"
            parsed_lines.add(number)
            continue
        if cells is not None and active is not None and _is_separator(cells):
            expected = len(_LIVE_HEADER) if active == "LIVE" else len(_KILLED_HEADER)
            if len(cells) == expected:
                parsed_lines.add(number)
            else:
                errors.append(_error(number, raw, "ledger table separator has wrong column count"))
            continue
        if cells is not None and active is not None:
            try:
                record = _live_record(cells, number, raw) if active == "LIVE" else _killed_record(cells, number, raw)
            except ValueError as exc:
                errors.append(_error(number, raw, str(exc)))
            else:
                provisional.append(record)
            continue
        if cells is None and raw.strip():
            active = None

    records: list[dict[str, Any]] = []
    live_by_id: dict[str, list[dict[str, Any]]] = {}
    killed_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for record in provisional:
        if record["source_table"] == "LIVE":
            live_by_id.setdefault(record["legacy_id"], []).append(record)
        else:
            killed_by_candidate.setdefault(record["display_name"], []).append(record)

    for rows in live_by_id.values():
        if len(rows) == 1:
            records.append(rows[0])
            parsed_lines.add(rows[0]["source_line"])
        else:
            for row in rows:
                errors.append(_error(row["source_line"], row["raw_row"], "duplicate live legacy ID"))

    unique_killed: list[dict[str, Any]] = []
    for rows in killed_by_candidate.values():
        if len(rows) == 1:
            unique_killed.append(rows[0])
        else:
            for row in rows:
                errors.append(_error(row["source_line"], row["raw_row"], "duplicate killed candidate"))

    killed_by_prefix: dict[str, list[dict[str, Any]]] = {}
    for record in unique_killed:
        killed_by_prefix.setdefault(record["_short_prefix"], []).append(record)
    reserved_ids = {record["legacy_id"] for record in records}
    for prefix, rows in killed_by_prefix.items():
        for row in rows:
            legacy_id = (
                prefix
                if len(rows) == 1 and prefix not in reserved_ids
                else _disambiguated_killed_id(prefix, row["display_name"])
            )
            row = {key: value for key, value in row.items() if key != "_short_prefix"}
            row["legacy_id"] = legacy_id
            row["record_id"] = legacy_id
            records.append(row)
            parsed_lines.add(row["source_line"])

    by_final_id: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_final_id.setdefault(record["legacy_id"], []).append(record)
    conflicting_ids = {
        legacy_id for legacy_id, rows in by_final_id.items() if len(rows) > 1
    }
    if conflicting_ids:
        retained: list[dict[str, Any]] = []
        for record in records:
            if record["legacy_id"] in conflicting_ids:
                parsed_lines.discard(record["source_line"])
                errors.append(
                    _error(
                        record["source_line"], record["raw_row"],
                        "final global legacy ID collision",
                    )
                )
            else:
                retained.append(record)
        records = retained
    if len({record["legacy_id"] for record in records}) != len(records):
        raise AssertionError("legacy ID allocator emitted a duplicate final ID")

    records.sort(key=lambda row: row["source_line"])
    errors.sort(key=lambda row: row["line_number"])
    unparsed = tuple(
        {"line_number": number, "line": raw}
        for number, raw in enumerate(lines, start=1)
        if number not in parsed_lines
    )
    return LegacyParse(
        tuple(records), unparsed, tuple(errors),
        hashlib.sha256(source).hexdigest(), len(lines),
    )


def _manifest(config: CorpusRouterConfig, parsed: LegacyParse, unparsed_text: str) -> dict[str, Any]:
    return {
        "schema_version": "idea_factory.legacy_import_manifest.v1",
        "parser_version": PARSER_VERSION,
        "legacy_source_path": str(config.legacy_ledger),
        "legacy_source_sha256": parsed.source_sha256,
        "legacy_source_line_count": parsed.source_line_count,
        "parsed_record_ids": [record["legacy_id"] for record in parsed.records],
        "parsed_count": len(parsed.records),
        "unparsed_count": len(parsed.unparsed),
        "parse_error_count": len(parsed.errors),
        "parse_errors": list(parsed.errors),
        "unparsed_sha256": hashlib.sha256(unparsed_text.encode("utf-8")).hexdigest(),
    }


def _unparsed_text(unparsed: tuple[dict[str, Any], ...]) -> str:
    return "".join(f"{row['line']}\n" for row in unparsed)


def _stage_text(path: Path, text: str) -> Path:
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)
    finally:
        handle.close()


def _write_bundle(ledger: Path, parsed: LegacyParse, manifest: dict[str, Any]) -> dict[str, Path]:
    paths = {"index": ledger / _NAMES[0], "unparsed": ledger / _NAMES[1], "manifest": ledger / _NAMES[2]}
    unparsed_text = _unparsed_text(parsed.unparsed)
    staged: dict[Path, Path] = {}
    try:
        # Every owned member is staged before any final replacement.  Consumers
        # still require the whole validated bundle because replacement itself is
        # not crash-atomic across files.
        index_text = "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
            for record in parsed.records
        )
        staged[paths["index"]] = _stage_text(paths["index"], index_text)
        staged[paths["unparsed"]] = _stage_text(paths["unparsed"], unparsed_text)
        staged[paths["manifest"]] = _stage_text(paths["manifest"], json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        for path in (paths["index"], paths["unparsed"], paths["manifest"]):
            os.replace(staged.pop(path), path)
        return paths
    except BaseException:
        _safe_unlink(ledger, _NAMES)
        raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


def import_legacy_ledger(run_dir: Path, config: CorpusRouterConfig) -> dict[str, Path]:
    """Create run-owned legacy artifacts; the configured source is never written."""

    run = _anchored_run(Path(run_dir))
    source = config.legacy_ledger
    if not source.is_file() or source.resolve(strict=True) != source:
        raise ValueError("legacy source must be the exact configured regular file")
    source_bytes = source.read_bytes()
    parsed = parse_legacy_markdown(source_bytes, source_path=str(source))
    ledger = _owned_dir(run, "ledger", create=True)
    _safe_unlink(ledger, _NAMES)
    manifest = _manifest(config, parsed, _unparsed_text(parsed.unparsed))
    try:
        return _write_bundle(ledger, parsed, manifest)
    except BaseException:
        _safe_unlink(ledger, _NAMES)
        raise


def validate_legacy_import(run_dir: Path, config: CorpusRouterConfig) -> LegacyImportBundle:
    run = _anchored_run(Path(run_dir))
    source = config.legacy_ledger
    if not source.is_file() or source.resolve(strict=True) != source:
        raise ValueError("legacy source must be the exact configured regular file")
    ledger = _owned_dir(run, "ledger", create=False)
    index = read_jsonl(_owned_file(ledger, _NAMES[0]))
    unparsed_path = _owned_file(ledger, _NAMES[1])
    manifest_path = _owned_file(ledger, _NAMES[2])
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("legacy manifest is invalid JSON") from exc
    parsed = parse_legacy_markdown(source.read_bytes(), source_path=str(source))
    expected_manifest = _manifest(config, parsed, _unparsed_text(parsed.unparsed))
    if index != list(parsed.records) or unparsed_path.read_text(encoding="utf-8") != _unparsed_text(parsed.unparsed) or manifest != expected_manifest:
        raise ValueError("legacy import replay mismatch")
    return LegacyImportBundle(run, tuple(index), parsed.unparsed, manifest)
