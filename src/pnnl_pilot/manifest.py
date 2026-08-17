from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Mapping

from .spatial import CropGeometryError, crop_box_for_mm, project_mm_to_pixel


class GateNotPassed(ValueError):
    """Raised when model-input authorization is attempted before Gate A."""


def authorize_crop_row(
    row: Mapping[str, object], gate_report: Mapping[str, object]
) -> dict[str, object]:
    if gate_report.get("status") != "passed":
        raise GateNotPassed("PNNL Gate A must pass before authorizing model input")
    authorized = dict(row)
    if row.get("status") == "material_safe_pending_indent_qa":
        authorized["status"] = "eligible_region_level"
        authorized["eligible_for_model"] = True
    else:
        authorized["eligible_for_model"] = False
    return authorized


@dataclass(frozen=True)
class CropCandidate:
    crop_id: str
    hardness_point_id: str
    sample_id: str
    condition_id: str
    crop_size_mm: float
    hardness_hv: float
    center_u_px: int
    center_v_px: int
    crop_box: tuple[int, int, int, int] | None
    material_fraction: float | None
    status: str
    eligible_for_model: bool
    outer_test_fold: str


def _crop_id(point_id: str, crop_size_mm: float) -> str:
    token = f"{point_id}|{crop_size_mm:.3f}|pnnl-gate-a-v1"
    return "CROP-" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:20]


def evaluate_crop_candidate(
    binding: Mapping[str, str],
    crop_size_mm: float,
    image_size: tuple[int, int],
    clean_y_max: int,
    material_fraction: Callable[[tuple[int, int, int, int]], float],
    minimum_material_fraction: float = 0.99,
) -> CropCandidate:
    point_id = binding["hardness_point_id"]
    sample_id = binding["sample_id"]
    center_u, center_v = project_mm_to_pixel(
        float(binding["x_rel_stir_mm"]),
        float(binding["y_rel_stir_mm"]),
        float(binding["image_center_u_px"]),
        float(binding["image_center_v_px"]),
        float(binding["rotation_deg"]),
        float(binding["scale_px_per_mm"]),
    )
    common = dict(
        crop_id=_crop_id(point_id, crop_size_mm),
        hardness_point_id=point_id,
        sample_id=sample_id,
        condition_id=binding["condition_id"],
        crop_size_mm=crop_size_mm,
        hardness_hv=float(binding["hardness_hv"]),
        center_u_px=center_u,
        center_v_px=center_v,
        outer_test_fold=f"test_{sample_id}",
    )
    try:
        box = crop_box_for_mm(
            center_u,
            center_v,
            crop_size_mm,
            float(binding["scale_px_per_mm"]),
            image_size[0],
            image_size[1],
            clean_y_max,
        )
    except CropGeometryError:
        return CropCandidate(
            **common,
            crop_box=None,
            material_fraction=None,
            status="geometry_invalid",
            eligible_for_model=False,
        )
    fraction = material_fraction(box)
    status = (
        "material_safe_pending_indent_qa"
        if fraction >= minimum_material_fraction
        else "background_contaminated"
    )
    return CropCandidate(
        **common,
        crop_box=box,
        material_fraction=fraction,
        status=status,
        eligible_for_model=False,
    )
