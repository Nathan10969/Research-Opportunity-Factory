from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from pnnl_pilot.background import DownsampledMaterialIndex
from pnnl_pilot.manifest import evaluate_crop_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--minimum-material-fraction", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Image.MAX_IMAGE_PIXELS = None
    bindings = list(csv.DictReader(args.bindings.open("r", encoding="utf-8-sig")))
    by_sample: dict[str, list[dict[str, str]]] = {}
    for row in bindings:
        by_sample.setdefault(row["sample_id"], []).append(row)

    candidates = []
    for image_path in sorted(args.images.glob("SS*/*.jpg")):
        sample_id = image_path.parent.name
        if sample_id not in by_sample:
            continue
        image = Image.open(image_path)
        original_size = image.size
        image.draft("L", (max(1, image.width // 8), max(1, image.height // 8)))
        preview = image.convert("L")
        index = DownsampledMaterialIndex(preview, original_size=original_size)
        clean_y_max = int(original_size[1] * 0.94)
        for binding in by_sample[sample_id]:
            for crop_size_mm in (0.5, 1.0, 2.0):
                candidates.append(
                    evaluate_crop_candidate(
                        binding,
                        crop_size_mm,
                        original_size,
                        clean_y_max,
                        index.fraction,
                        args.minimum_material_fraction,
                    )
                )
        image.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "crop_id", "hardness_point_id", "sample_id", "condition_id",
        "crop_size_mm", "hardness_hv", "center_u_px", "center_v_px",
        "left_px", "top_px", "right_px", "bottom_px", "material_fraction",
        "status", "eligible_for_model", "outer_test_fold",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            row = asdict(candidate)
            box = row.pop("crop_box")
            if box is None:
                row.update(left_px="", top_px="", right_px="", bottom_px="")
            else:
                row.update(
                    left_px=box[0], top_px=box[1], right_px=box[2], bottom_px=box[3]
                )
            writer.writerow(row)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    status_by_scale = Counter(
        f"{candidate.crop_size_mm:.1f}mm:{candidate.status}"
        for candidate in candidates
    )
    summary = {
        "status": "generated_pending_indent_qa",
        "schema_version": "pnnl-crop-manifest-summary-v1",
        "rows": len(candidates),
        "samples": len({candidate.sample_id for candidate in candidates}),
        "status_counts": dict(sorted(Counter(candidate.status for candidate in candidates).items())),
        "status_counts_by_scale": dict(sorted(status_by_scale.items())),
        "eligible_for_model": sum(candidate.eligible_for_model for candidate in candidates),
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": digest,
        "minimum_material_fraction": args.minimum_material_fraction,
    }
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
