from __future__ import annotations

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
