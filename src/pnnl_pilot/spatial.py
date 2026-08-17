from __future__ import annotations

import math


class CropGeometryError(ValueError):
    """Raised when a physical crop is not fully inside the accepted pixel domain."""


def project_mm_to_pixel(
    x_rel_mm: float,
    y_rel_mm: float,
    center_u_px: float,
    center_v_px: float,
    rotation_deg: float,
    scale_px_per_mm: float,
) -> tuple[int, int]:
    angle = math.radians(rotation_deg)
    rotated_x = math.cos(angle) * x_rel_mm - math.sin(angle) * y_rel_mm
    rotated_y = math.sin(angle) * x_rel_mm + math.cos(angle) * y_rel_mm
    u = center_u_px + scale_px_per_mm * rotated_x
    v = center_v_px - scale_px_per_mm * rotated_y
    return round(u), round(v)


def crop_box_for_mm(
    center_u_px: int,
    center_v_px: int,
    size_mm: float,
    scale_px_per_mm: float,
    image_width: int,
    image_height: int,
    clean_y_max: int,
) -> tuple[int, int, int, int]:
    if size_mm <= 0 or scale_px_per_mm <= 0:
        raise CropGeometryError("crop size and scale must be positive")
    half = round(size_mm * scale_px_per_mm / 2.0)
    box = (
        center_u_px - half,
        center_v_px - half,
        center_u_px + half,
        center_v_px + half,
    )
    if box[0] < 0 or box[1] < 0 or box[2] > image_width or box[3] > image_height:
        raise CropGeometryError(f"crop exceeds image bounds: {box}")
    if box[3] > clean_y_max:
        raise CropGeometryError(f"crop intersects frozen metadata band: {box}")
    return box

