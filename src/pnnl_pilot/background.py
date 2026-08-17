from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image


def build_material_mask(
    image: Image.Image, intensity_threshold: int = 20
) -> np.ndarray:
    if not 0 <= intensity_threshold <= 255:
        raise ValueError("intensity_threshold must be in [0, 255]")
    grayscale = np.asarray(image.convert("L"))
    return grayscale > intensity_threshold


def material_fraction_for_box(
    material_mask: np.ndarray, box: Sequence[int]
) -> float:
    if material_mask.ndim != 2:
        raise ValueError("material_mask must be two-dimensional")
    if len(box) != 4:
        raise ValueError("box must contain left, top, right, bottom")
    left, top, right, bottom = (int(value) for value in box)
    height, width = material_mask.shape
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError("box exceeds material-mask bounds")
    if right <= left or bottom <= top:
        raise ValueError("box must have positive area")
    return float(material_mask[top:bottom, left:right].mean())


class DownsampledMaterialIndex:
    """Query material coverage in original-image coordinates from a preview."""

    def __init__(
        self,
        preview: Image.Image,
        original_size: tuple[int, int],
        intensity_threshold: int = 20,
    ) -> None:
        self.original_width, self.original_height = original_size
        if self.original_width <= 0 or self.original_height <= 0:
            raise ValueError("original_size must be positive")
        self.mask = build_material_mask(preview, intensity_threshold)
        numeric = self.mask.astype(np.int64)
        self.integral = np.pad(numeric, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        self.preview_height, self.preview_width = self.mask.shape

    def fraction(self, box: Sequence[int]) -> float:
        if len(box) != 4:
            raise ValueError("box must contain four coordinates")
        left, top, right, bottom = (int(value) for value in box)
        if (
            left < 0
            or top < 0
            or right > self.original_width
            or bottom > self.original_height
            or right <= left
            or bottom <= top
        ):
            raise ValueError("box is invalid in original-image coordinates")
        x0 = int(math.floor(left * self.preview_width / self.original_width))
        x1 = int(math.ceil(right * self.preview_width / self.original_width))
        y0 = int(math.floor(top * self.preview_height / self.original_height))
        y1 = int(math.ceil(bottom * self.preview_height / self.original_height))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.preview_width, x1), min(self.preview_height, y1)
        material = (
            self.integral[y1, x1]
            - self.integral[y0, x1]
            - self.integral[y1, x0]
            + self.integral[y0, x0]
        )
        return float(material / ((x1 - x0) * (y1 - y0)))
