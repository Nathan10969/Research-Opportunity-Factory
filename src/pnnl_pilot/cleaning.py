from __future__ import annotations

from pathlib import PurePath

from PIL import Image


CLEANING_POLICY_VERSION = "fixed-bottom-band-v1"
FORBIDDEN_PATH_TOKENS = (
    "qa_overlays",
    "visualization/marker",
    "visualization/crop-7mm",
    "ss28",
    "ss32",
    "ss36",
)


class ContaminatedInput(ValueError):
    """Raised when a path or pixel region is forbidden for model input."""


def assert_training_path(path: str | PurePath) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if any(token in normalized for token in FORBIDDEN_PATH_TOKENS):
        raise ContaminatedInput(f"forbidden training input path: {path}")


def apply_fixed_metadata_crop(
    image: Image.Image, bottom_fraction: float = 0.06
) -> tuple[Image.Image, dict[str, object]]:
    if not 0 < bottom_fraction < 0.5:
        raise ValueError("bottom_fraction must be between 0 and 0.5")
    clean_height = int(image.height * (1.0 - bottom_fraction))
    box = (0, 0, image.width, clean_height)
    return image.crop(box), {
        "policy_version": CLEANING_POLICY_VERSION,
        "bottom_fraction": bottom_fraction,
        "crop_box": list(box),
    }

