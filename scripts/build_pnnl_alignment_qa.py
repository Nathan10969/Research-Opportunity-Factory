from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pnnl_pilot.spatial import project_mm_to_pixel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-width", type=int, default=1800)
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
    rows_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.bindings.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_sample[row["sample_id"]].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    overlay_paths: list[Path] = []
    for image_path in sorted(args.images.glob("SS*/*.jpg")):
        sample_id = image_path.parent.name
        bindings = rows_by_sample.get(sample_id, [])
        if not bindings:
            continue
        with Image.open(image_path) as original:
            scale = min(1.0, args.max_width / original.width)
            overlay = original.convert("RGB")
            if scale < 1.0:
                overlay.thumbnail(
                    (args.max_width, max(1, round(original.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            sx = overlay.width / original.width
            sy = overlay.height / original.height
            draw = ImageDraw.Draw(overlay)
            clean_y = int(original.height * 0.94)
            draw.line(
                [(0, round(clean_y * sy)), (overlay.width, round(clean_y * sy))],
                fill=(255, 215, 0),
                width=3,
            )
            in_bounds = 0
            clean_bounds = 0
            for row in bindings:
                u, v = project_mm_to_pixel(
                    float(row["x_rel_stir_mm"]),
                    float(row["y_rel_stir_mm"]),
                    float(row["image_center_u_px"]),
                    float(row["image_center_v_px"]),
                    float(row["rotation_deg"]),
                    float(row["scale_px_per_mm"]),
                )
                if 0 <= u < original.width and 0 <= v < original.height:
                    in_bounds += 1
                    if v < clean_y:
                        clean_bounds += 1
                    x, y = round(u * sx), round(v * sy)
                    radius = 3
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius),
                        outline=(255, 40, 40),
                        width=2,
                    )
            label = (
                f"{sample_id} points={len(bindings)} "
                f"in_image={in_bounds} metadata_safe={clean_bounds}"
            )
            draw.rectangle((0, 0, min(900, overlay.width), 34), fill=(0, 0, 0))
            draw.text((10, 8), label, fill=(255, 255, 255), font=ImageFont.load_default())
            output_path = args.output_dir / f"{sample_id}_alignment_overlay.jpg"
            overlay.save(output_path, quality=90)
        overlay_paths.append(output_path)
        records.append(
            {
                "sample_id": sample_id,
                "binding_points": len(bindings),
                "projected_in_image": in_bounds,
                "projected_metadata_safe": clean_bounds,
                "overlay_path": str(output_path.resolve()),
                "overlay_sha256": sha256(output_path),
                "review_status": "pending_manual_review",
                "review_note": "",
            }
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    thumbs = []
    for path in overlay_paths:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((900, 450), Image.Resampling.LANCZOS)
            thumbs.append(thumb.copy())
    width = 900 * 3
    height = max(image.height for image in thumbs) * 3
    sheet = Image.new("RGB", (width, height), "white")
    cell_height = height // 3
    for index, image in enumerate(thumbs):
        x = (index % 3) * 900
        y = (index // 3) * cell_height
        sheet.paste(image, (x, y))
    sheet.save(args.output_dir / "alignment_contact_sheet.jpg", quality=90)


if __name__ == "__main__":
    main()
