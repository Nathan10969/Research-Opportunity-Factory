from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from pnnl_pilot.manifest import authorize_crop_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    gate_report = json.loads(args.gate_report.read_text(encoding="utf-8"))
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        authorized = [
            candidate
            for row in reader
            if (candidate := authorize_crop_row(row, gate_report))[
                "eligible_for_model"
            ]
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(authorized)

    summary = {
        "status": "passed",
        "schema_version": "pnnl-model-manifest-summary-v1",
        "rows": len(authorized),
        "unique_crop_ids": len({row["crop_id"] for row in authorized}),
        "samples": sorted({row["sample_id"] for row in authorized}),
        "rows_by_scale": dict(
            sorted(Counter(row["crop_size_mm"] for row in authorized).items())
        ),
        "all_rows_authorized": all(
            row["status"] == "eligible_region_level" for row in authorized
        ),
        "input_sha256": sha256(args.input),
        "output_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
