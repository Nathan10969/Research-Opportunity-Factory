"""Durable JSON artifacts for auditable Idea Factory runs."""

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_ID_PREFIX = re.compile(r"^[A-Za-z0-9_]+$")
_SEPARATOR = "\x1f"
_HASH_CHUNK_SIZE = 1024 * 1024


def _require_dict(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("artifact payload must be a dict")
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value} is forbidden")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one UTF-8 JSON object to *path*."""

    data = _require_dict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(
                data,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temp_name).replace(path)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Durably append one compact UTF-8 JSON object record.

    V1 assumes a single writer per JSONL file. ``O_APPEND`` plus one ``os.write``
    minimizes partial/interleaved records, but this is not a cross-process lock.
    """

    data = _require_dict(payload)
    record = (
        json.dumps(
            data,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o666)
    try:
        written = os.write(descriptor, record)
        if written != len(record):
            raise OSError(
                f"partial JSONL append at {path}: wrote {written} of {len(record)} bytes"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    data = [_require_dict(record) for record in records]
    encoded = "".join(
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for record in data
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        staged = Path(temp_name)
        temp_name = None
        return staged
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically replace a JSONL artifact with deterministic compact records."""

    staged = _stage_jsonl(path, records)
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def write_jsonl_bundle(
    records_by_path: Mapping[Path, list[dict[str, Any]]],
) -> None:
    """Stage every JSONL member before publishing, invalidating all on failure.

    Cross-file replacement cannot be crash-atomic. Consumers must require every
    expected bundle member and validate their record bindings before use.
    """

    targets = [Path(path) for path in records_by_path]
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("JSONL bundle paths must be nonempty and unique")
    staged: dict[Path, Path] = {}
    try:
        for target in targets:
            staged[target] = _stage_jsonl(target, records_by_path[target])
        for target in targets:
            os.replace(staged[target], target)
            staged.pop(target)
    except BaseException:
        for target in targets:
            target.unlink(missing_ok=True)
        raise
    finally:
        for temp_path in staged.values():
            temp_path.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read nonblank JSONL records, preserving line-aware failure context."""

    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"invalid UTF-8 at {path} line {line_number}"
                ) from exc
            try:
                record = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid JSONL record at {path} line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL record at {path} line {line_number} must be an object"
                )
            records.append(record)
    return records


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: str) -> str:
    """Produce a stable, unambiguous short identifier from nonblank text parts."""

    if not isinstance(prefix, str) or not _ID_PREFIX.fullmatch(prefix):
        raise ValueError("stable ID prefix must contain only letters, digits, or underscores")
    if not parts:
        raise ValueError("stable ID requires at least one part")
    if any(
        not isinstance(part, str) or not part.strip() or _SEPARATOR in part
        for part in parts
    ):
        raise ValueError("stable ID parts must be nonblank strings without unit separators")
    digest = hashlib.sha256(_SEPARATOR.join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"
