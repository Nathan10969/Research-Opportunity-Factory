from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

from pnnl_pilot.images import apply_center_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="facebook/dinov2-base")
    parser.add_argument("--crop-scale-mm", type=float, default=1.0)
    parser.add_argument("--center-mask-fraction", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    Image.MAX_IMAGE_PIXELS = None
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if float(row["crop_size_mm"]) == args.crop_scale_mm
            and row["status"] == "eligible_region_level"
        ]
    if not rows:
        raise RuntimeError("no authorized rows match the requested crop scale")
    if len({row["crop_id"] for row in rows}) != len(rows):
        raise RuntimeError("crop_id values are not unique")

    image_by_sample = {}
    for image_path in sorted(args.images.glob("SS*/*.jpg")):
        image_by_sample[image_path.parent.name] = image_path
    missing = sorted({row["sample_id"] for row in rows} - set(image_by_sample))
    if missing:
        raise RuntimeError(f"missing source images for samples: {missing}")

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to(device)
    hidden_size = int(model.config.hidden_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cls_path = args.output_dir / "features_cls.npy"
    mean_path = args.output_dir / "features_mean_patch.npy"
    cls_features = open_memmap(
        cls_path, mode="w+", dtype=np.float32, shape=(len(rows), hidden_size)
    )
    mean_features = open_memmap(
        mean_path, mode="w+", dtype=np.float32, shape=(len(rows), hidden_size)
    )

    indexed_by_sample: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        indexed_by_sample[row["sample_id"]].append((index, row))

    with torch.inference_mode():
        for sample_id in sorted(indexed_by_sample):
            with Image.open(image_by_sample[sample_id]) as source:
                source = source.convert("RGB")
                indexed_rows = indexed_by_sample[sample_id]
                for start in range(0, len(indexed_rows), args.batch_size):
                    batch = indexed_rows[start : start + args.batch_size]
                    images = []
                    indices = []
                    for index, row in batch:
                        box = tuple(
                            int(row[name])
                            for name in ("left_px", "top_px", "right_px", "bottom_px")
                        )
                        crop = source.crop(box)
                        crop = apply_center_mask(
                            crop, fraction=args.center_mask_fraction
                        )
                        images.append(crop)
                        indices.append(index)
                    inputs = processor(images=images, return_tensors="pt")
                    inputs = {name: value.to(device) for name, value in inputs.items()}
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=device.type == "cuda",
                    ):
                        hidden = model(**inputs).last_hidden_state
                    cls = hidden[:, 0].float().cpu().numpy()
                    mean = hidden[:, 1:].mean(dim=1).float().cpu().numpy()
                    cls_features[indices] = cls
                    mean_features[indices] = mean
    cls_features.flush()
    mean_features.flush()

    metadata_path = args.output_dir / "feature_rows.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "feature_row", "crop_id", "hardness_point_id", "sample_id",
            "condition_id", "crop_size_mm", "hardness_hv", "center_u_px",
            "center_v_px", "outer_test_fold",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"feature_row": index, **{name: row[name] for name in fields[1:]}})

    report = {
        "status": "passed",
        "schema_version": "pnnl-dino-feature-report-v1",
        "model": args.model,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "crop_scale_mm": args.crop_scale_mm,
        "center_mask_fraction": args.center_mask_fraction,
        "rows": len(rows),
        "hidden_size": hidden_size,
        "pooling_outputs": ["cls", "mean_patch"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "manifest_sha256": sha256(args.manifest),
        "features_cls_sha256": sha256(cls_path),
        "features_mean_patch_sha256": sha256(mean_path),
        "feature_rows_sha256": sha256(metadata_path),
    }
    (args.output_dir / "feature_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
