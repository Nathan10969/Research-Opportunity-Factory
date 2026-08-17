from __future__ import annotations

import hashlib
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .contracts import ContractViolation, OpticalAsset, validate_member_path


@dataclass(frozen=True)
class ExtractionRecord:
    sample_id: str
    condition_id: str
    asset_id: str
    source_member_path: str
    output_path: str
    byte_size: int
    sha256: str


def _copy_and_hash(source, destination: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_size = 0
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temp_path.open("wb") as output:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                byte_size += len(chunk)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return byte_size, digest.hexdigest()


def extract_selected_members(
    tar_path: str | Path,
    assets: Iterable[OpticalAsset],
    output_dir: str | Path,
) -> list[ExtractionRecord]:
    selected = list(assets)
    if not selected:
        raise ContractViolation("no optical sources selected")
    by_member: dict[str, OpticalAsset] = {}
    for row in selected:
        validate_member_path(row.member_path, row.sample_id)
        if row.member_path in by_member:
            raise ContractViolation(f"duplicate requested member: {row.member_path}")
        by_member[row.member_path] = row

    output_root = Path(output_dir)
    records: list[ExtractionRecord] = []
    found: set[str] = set()
    with tarfile.open(Path(tar_path), "r:*") as archive:
        for member in archive:
            row = by_member.get(member.name)
            if row is None:
                continue
            if not member.isfile():
                raise ContractViolation(f"selected member is not a file: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ContractViolation(f"cannot read selected member: {member.name}")
            destination = output_root / row.sample_id / row.file_name
            with source:
                byte_size, sha256 = _copy_and_hash(source, destination)
            records.append(
                ExtractionRecord(
                    sample_id=row.sample_id,
                    condition_id=row.condition_id,
                    asset_id=row.asset_id,
                    source_member_path=row.member_path,
                    output_path=str(destination.resolve()),
                    byte_size=byte_size,
                    sha256=sha256,
                )
            )
            found.add(member.name)
            if len(found) == len(by_member):
                break

    missing = sorted(set(by_member) - found)
    if missing:
        raise ContractViolation(f"selected tar members missing: {missing}")
    return sorted(records, key=lambda record: record.sample_id)
