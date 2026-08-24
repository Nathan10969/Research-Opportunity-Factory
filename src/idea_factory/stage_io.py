"""Cross-process stage mutation lock and rollback-safe multi-file publication."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import uuid
from typing import Iterator


LOCK_NAME = ".stage.lock"


def encode_json(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(dict(payload), ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def encode_jsonl(records: list[dict[str, object]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")


@contextmanager
def stage_mutation_lock(stage: Path) -> Iterator[Path]:
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    lock = stage / LOCK_NAME
    token = f"pid={os.getpid()} token={uuid.uuid4().hex}\n".encode("ascii")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(lock, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"stage mutation lock is busy: {stage.name}") from exc
    try:
        try:
            if os.write(descriptor, token) != len(token):
                raise OSError("partial stage lock write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        yield lock
    finally:
        try:
            if lock.read_bytes() == token:
                lock.unlink()
        except FileNotFoundError:
            pass


def _stage_bytes(target: Path, payload: bytes, suffix: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=suffix, delete=False
        ) as handle:
            temp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        staged = Path(temp_name)
        temp_name = None
        return staged
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def _publish_transaction(payloads: Mapping[Path, bytes | None]) -> None:
    targets = [Path(path) for path in payloads]
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("transaction targets must be nonempty and unique")
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    try:
        for target in targets:
            if target.exists() and (not target.is_file() or target.is_symlink()):
                raise ValueError("transaction target must be an anchored regular file")
            backups[target] = _stage_bytes(target, target.read_bytes(), ".bak") if target.exists() else None
            payload = payloads[target]
            if payload is not None:
                staged[target] = _stage_bytes(target, payload, ".tmp")
        try:
            for target in targets:
                if payloads[target] is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(staged.pop(target), target)
        except BaseException as publish_error:
            rollback_error: BaseException | None = None
            for target in reversed(targets):
                try:
                    backup = backups[target]
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        os.replace(backup, target)
                        backups[target] = None
                except BaseException as exc:
                    rollback_error = rollback_error or exc
            if rollback_error is not None:
                raise RuntimeError("stage transaction publish and rollback both failed") from rollback_error
            raise publish_error
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
        for path in backups.values():
            if path is not None:
                path.unlink(missing_ok=True)


def publish_transaction(payloads: Mapping[Path, bytes | None]) -> None:
    targets = [Path(path) for path in payloads]
    parents = {target.parent.resolve(strict=True) for target in targets}
    if len(parents) != 1:
        raise ValueError("transaction targets must share one anchored stage directory")
    _publish_transaction(payloads)


def publish_cross_directory_transaction(payloads: Mapping[Path, bytes | None]) -> None:
    """Publish one rollback-safe transaction across exact, already anchored directories."""

    targets = [Path(path) for path in payloads]
    if any(target.parent.resolve(strict=True) != target.parent for target in targets):
        raise ValueError("cross-directory transaction parent is not anchored")
    _publish_transaction(payloads)
