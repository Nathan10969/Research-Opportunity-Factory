from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ELIGIBLE_SAMPLES = frozenset(
    {"SS01", "SS02", "SS03", "SS04", "SS05", "SS06", "SS07", "SS08", "SS31"}
)
FORBIDDEN_SAMPLES = frozenset({"SS28", "SS32", "SS36"})
FORBIDDEN_SEGMENTS = ("/VISUALIZATION/MARKER/", "/VISUALIZATION/crop-7mm/")
REQUIRED_SEGMENT = "/DATA/"
REQUIRED_SUFFIX = "_10X_Nugget-Region-Etched.jpg"


class ContractViolation(ValueError):
    """Raised when an input falls outside the frozen PNNL data contract."""


@dataclass(frozen=True)
class OpticalAsset:
    asset_id: str
    record_id: str
    sample_id: str
    condition_id: str
    dataset_id: str
    file_name: str
    declared_digest: str
    member_path: str


def validate_member_path(member_path: str, sample_id: str) -> None:
    normalized = "/" + member_path.replace("\\", "/").lstrip("/")
    if sample_id in FORBIDDEN_SAMPLES or sample_id not in ELIGIBLE_SAMPLES:
        raise ContractViolation(f"sample is not eligible: {sample_id}")
    if any(segment.lower() in normalized.lower() for segment in FORBIDDEN_SEGMENTS):
        raise ContractViolation(f"forbidden visualization path: {member_path}")
    if REQUIRED_SEGMENT.lower() not in normalized.lower():
        raise ContractViolation(f"source is not under DATA: {member_path}")
    if not normalized.endswith(REQUIRED_SUFFIX):
        raise ContractViolation(f"source is not the frozen 10X etched image: {member_path}")


def load_assets(path: str | Path) -> list[OpticalAsset]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [OpticalAsset(**row) for row in csv.DictReader(handle)]
    for row in rows:
        validate_member_path(row.member_path, row.sample_id)
        if Path(row.member_path).name != row.file_name:
            raise ContractViolation(f"filename/member mismatch for {row.asset_id}")
    return rows


def select_optical_sources(rows: Iterable[OpticalAsset]) -> list[OpticalAsset]:
    selected = list(rows)
    by_sample: dict[str, OpticalAsset] = {}
    member_paths: set[str] = set()
    for row in selected:
        validate_member_path(row.member_path, row.sample_id)
        if row.sample_id in by_sample:
            raise ContractViolation(f"duplicate source for sample {row.sample_id}")
        if row.member_path in member_paths:
            raise ContractViolation(f"duplicate tar member {row.member_path}")
        by_sample[row.sample_id] = row
        member_paths.add(row.member_path)
    if set(by_sample) != set(ELIGIBLE_SAMPLES):
        missing = sorted(ELIGIBLE_SAMPLES - set(by_sample))
        extra = sorted(set(by_sample) - ELIGIBLE_SAMPLES)
        raise ContractViolation(f"whitelist must resolve 9 samples; missing={missing}, extra={extra}")
    return sorted(selected, key=lambda row: row.sample_id)

