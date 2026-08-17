import unittest

from PIL import Image

from pnnl_pilot.images import apply_center_mask


class ImageTransformTests(unittest.TestCase):
    def test_center_mask_changes_only_frozen_center_square(self):
        image = Image.new("RGB", (100, 80), (100, 100, 100))
        masked = apply_center_mask(image, fraction=0.20, fill=(0, 0, 0))
        self.assertEqual(masked.getpixel((0, 0)), (100, 100, 100))
        self.assertEqual(masked.getpixel((50, 40)), (0, 0, 0))
        self.assertEqual(masked.getpixel((39, 40)), (100, 100, 100))

    def test_zero_fraction_returns_copy(self):
        image = Image.new("RGB", (10, 10), "white")
        masked = apply_center_mask(image, fraction=0.0)
        self.assertIsNot(masked, image)
        self.assertEqual(
            list(masked.get_flattened_data()), list(image.get_flattened_data())
        )


if __name__ == "__main__":
    unittest.main()
