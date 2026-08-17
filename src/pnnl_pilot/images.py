from __future__ import annotations

from PIL import Image, ImageDraw


def apply_center_mask(
    image: Image.Image,
    fraction: float,
    fill: tuple[int, int, int] = (127, 127, 127),
) -> Image.Image:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    output = image.convert("RGB").copy()
    if fraction == 0.0:
        return output
    side = max(1, round(min(output.size) * fraction))
    left = (output.width - side) // 2
    top = (output.height - side) // 2
    ImageDraw.Draw(output).rectangle(
        (left, top, left + side - 1, top + side - 1), fill=fill
    )
    return output
