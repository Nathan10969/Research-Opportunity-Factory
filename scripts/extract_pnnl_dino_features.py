from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image
import torch

from pnnl_pilot.dino_features import (
    build_official_dinov3_transform,
    load_official_dinov3,
    pool_official_dinov3_outputs,
)
from pnnl_pilot.images import apply_center_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("huggingface", "official-dinov3"), default="huggingface"
    )
    parser.add_argument("--model", default="facebook/dinov2-base")
    parser.add_argument("--official-repo", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--entrypoint", default="dinov3_vitb16")
    parser.add_argument("--crop-scale-mm", type=float, default=1.0)
    parser.add_argument("--center-mask-fraction", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-rows",
        type=int,
        help="diagnostic-only row limit; omitted for a complete extraction",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


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
    complete_row_count = len(rows)
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        rows = rows[: args.max_rows]

    image_by_sample = {}
    for image_path in sorted(args.images.glob("SS*/*.jpg")):
        image_by_sample[image_path.parent.name] = image_path
    missing = sorted({row["sample_id"] for row in rows} - set(image_by_sample))
    if missing:
        raise RuntimeError(f"missing source images for samples: {missing}")

    processor = None
    transform = None
    model_commit = None
    weights_sha256 = None
    if args.backend == "official-dinov3":
        if args.official_repo is None or args.weights is None:
            raise ValueError(
                "--official-repo and --weights are required for official-dinov3"
            )
        model, hidden_size = load_official_dinov3(
            args.official_repo, args.entrypoint, args.weights
        )
        transform = build_official_dinov3_transform()
        model_commit = git_commit(args.official_repo)
        weights_sha256 = sha256(args.weights)
        model_name = args.entrypoint
    else:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model)
        hidden_size = int(model.config.hidden_size)
        model_commit = getattr(model.config, "_commit_hash", None)
        model_name = args.model
    model = model.eval().to(device)
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
                    if args.backend == "official-dinov3":
                        inputs = torch.stack([transform(image) for image in images]).to(
                            device
                        )
                    else:
                        inputs = processor(images=images, return_tensors="pt")
                        inputs = {
                            name: value.to(device) for name, value in inputs.items()
                        }
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=device.type == "cuda",
                    ):
                        if args.backend == "official-dinov3":
                            outputs = model.forward_features(inputs)
                            cls_tensor, mean_tensor = pool_official_dinov3_outputs(
                                outputs
                            )
                        else:
                            hidden = model(**inputs).last_hidden_state
                            cls_tensor = hidden[:, 0]
                            mean_tensor = hidden[:, 1:].mean(dim=1)
                    cls = cls_tensor.float().cpu().numpy()
                    mean = mean_tensor.float().cpu().numpy()
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
        "status": "diagnostic" if args.max_rows is not None else "passed",
        "schema_version": "pnnl-dino-feature-report-v2",
        "backend": args.backend,
        "model": model_name,
        "model_commit": model_commit,
        "weights_sha256": weights_sha256,
        "crop_scale_mm": args.crop_scale_mm,
        "center_mask_fraction": args.center_mask_fraction,
        "rows": len(rows),
        "complete_row_count": complete_row_count,
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
